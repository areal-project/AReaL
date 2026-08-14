import sys
import os
from pathlib import Path
import logging
import tempfile
import shutil
import json
import argparse
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from memrl.configs.config import MempConfig
from memrl.providers.llm import OpenAILLM
from memrl.providers.embedding import OpenAIEmbedder
from memrl.service.memory_service import MemoryService
from memrl.service.strategies import BuildStrategy, RetrieveStrategy, UpdateStrategy, StrategyConfiguration
from memrl.agent.memp_agent import MempAgent
from memrl.run.alfworld_rl_runner import AlfworldRunner


def setup_logging(project_root: Path, name: str):
    log_dir = project_root / "logs" / name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = f"{name}_{time.strftime('%Y%m%d-%H%M%S')}.log"
    log_filepath = log_dir / log_filename
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    logging.info(f"Logging configured. Log file: {log_filepath}")
    return log_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run AlfWorld benchmark with memory-agent")
    p.add_argument(
        "--config",
        type=str,
        default=str(
            (project_root / "configs" / "rl_alf_config.local.yaml")
            if (project_root / "configs" / "rl_alf_config.local.yaml").exists()
            else (project_root / "configs" / "rl_alf_config.yaml")
        ),
    )
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--max_tokens", type=int, default=None)
    p.add_argument("--region", action="store_true", help="Enable region-aware memory service")
    p.add_argument("--mem0", action="store_true", help="Use Mem0 baseline (mem0 library for memory management)")
    p.add_argument("--selfrag", action="store_true", help="Enable Self-RAG: LLM critique filters irrelevant memories after retrieval")
    p.add_argument("--mem0_infer", type=str, default="true", choices=["true", "false"],
                   help="Mem0 infer mode: 'true' = LLM fact extraction (paper-faithful), 'false' = raw text storage (faster)")
    p.add_argument("--mem0_collection", type=str, default="memrl_mem0_baseline",
                   help="Qdrant collection name for Mem0 baseline")
    p.add_argument("--region_gating_mode", type=str, default="multiplicative",
                   choices=["multiplicative", "additive"])
    p.add_argument("--region_value_mode", type=str, default="shrinkage",
                   choices=["shrinkage", "category_q"])
    p.add_argument("--region_cluster_space", type=str, default="capability",
                   choices=["capability", "embedding"])
    p.add_argument("--region_utility_mode", type=str, default="beta",
                   choices=["beta", "ema"],
                   help="Region utility estimation: 'beta' (Bayesian) or 'ema' (exponential moving avg)")
    p.add_argument("--region_split_evidence_migration_mode", type=str, default=None,
                   choices=["soft_source_conserving", "hard_member_rebase"],
                   help=("Topology-only evidence migration. Defaults to soft_source_conserving; "
                         "online region updates remain soft in both modes."))
    p.add_argument("--region_freeze_topology", action="store_true",
                   help=("Disable region split/merge for this run while retaining online soft "
                         "region updates. Used to warm up source-attributed evidence after "
                         "resuming a legacy checkpoint."))
    p.add_argument("--region_cluster_init_step", type=int, default=None,
                   help="Global clean-trajectory step at which initial Region clustering is allowed.")
    p.add_argument("--region_merge_interval", type=int, default=None,
                   help="Global clean-trajectory cadence for mid-section split/merge checks.")
    p.add_argument("--region_disable_mid_epoch_topology", action="store_true",
                   help="Disable mid-section Region topology edits; end-of-section maintenance remains enabled.")
    p.add_argument("--region_topology_cooldown_sections", type=int, default=0,
                   help="Successful topology-edit cooldown in complete sections; 0 preserves legacy behavior.")
    p.add_argument("--region_reset_legacy_evidence_on_resume", action="store_true",
                   help=("After loading an aggregate-only legacy region checkpoint, clear its observed "
                         "region evidence but retain Q, geometry, membership, and warm-start priors. "
                         "Future evidence starts as source-conserving."))
    p.add_argument("--max_candidates_per_sim_key", type=int, default=0,
                   help=("Cap memory candidates emitted by each similar query key before reranking; "
                         "0 keeps legacy unlimited expansion."))
    p.add_argument("--region_evidence_sharpen_alpha", type=float, default=None,
                   help=("Exponent for online top-3 region evidence allocation (default 2.0). "
                         "Higher values make future region evidence less soft without changing geometry."))
    p.add_argument("--skip_initial_eval", action="store_true", help="Skip initial evaluation before training")
    p.add_argument(
        "--holdout_subtask",
        type=str,
        default=None,
        help=(
            "Single-bank zero-shot holdout. If set (e.g. 'pick_and_place_simple' or "
            "'alf/pick_and_place_simple'), games of this subtask are excluded from train "
            "and the memory pool. Held-out games from valid+test are combined into "
            "a dedicated eval bucket evaluated at each section's valid checkpoint."
        ),
    )
    p.add_argument(
        "--holdout_eval_pools",
        type=str,
        default="valid,test",
        help=(
            "Comma-separated list of splits to draw holdout eval games from. "
            "Choices: train, valid, test (any subset). Default 'valid,test' = "
            "strictest zero-shot on unseen environments. Use 'train,valid' to "
            "evaluate on holdout-subtask games from training environments."
        ),
    )
    p.add_argument(
        "--val_lambda_max",
        type=float,
        default=None,
        help=(
            "If set (e.g. 0.15), eval phase temporarily lowers "
            "region_manager.shrinkage_lambda_max so retrieval is region-utility-dominated. "
            "Mirrors BCB confgate experiment val phase. No effect on baseline (no region)."
        ),
    )
    p.add_argument(
        "--shrinkage_confidence_k",
        type=float,
        default=None,
        help=(
            "If set (e.g. 3.0), use confidence-gated lambda = lambda_max * n / (n + k) "
            "instead of James-Stein lambda = tau^2 / (tau^2 + sigma^2/n). Smoother ramp-up "
            "as per-memory evidence accumulates. No effect on baseline."
        ),
    )
    p.add_argument(
        "--propagation_eta",
        type=float,
        default=None,
        help="Override region_manager.propagation_eta (default in code = 0.03). BCB confgate uses 0.12.",
    )
    p.add_argument(
        "--eval_train",
        action="store_true",
        help=(
            "In mode=test, also evaluate on train_game_files (in addition to "
            "valid + test). Useful for no-mem baseline to get a train SR "
            "comparable to MemRL train SR. No effect when mode=train."
        ),
    )
    p.add_argument(
        "--id_eval_only",
        action="store_true",
        help="In mode=test, evaluate ID validation only and skip OOD/test.",
    )
    p.add_argument(
        "--explore_schedule",
        type=str,
        default="0,4,4,3,3,2,2,1,1,0",
        help=(
            "Comma-separated per-section exploration quota for region (n_explore "
            "out of k_retrieve memories). Index = section_num - 1. Default is "
            "'0,4,4,3,3,2,2,1,1,0' (tuned for 10-section). For 2-section short "
            "runs or aggressive S2 (explore=4/5=80% replacement hurts SR), try "
            "'0,2,2,1,1,1,1,0,0,0' (S2 quota=40%). No effect on baseline."
        ),
    )
    p.add_argument(
        "--no_z_norm",
        action="store_true",
        help=(
            "Disable z-score normalization on sim and q during retrieval scoring. "
            "Useful for region: z-norm absorbs region utility's absolute differences "
            "(0.05-0.11 raw gap) into relative ranks within candidate set. Without "
            "z-norm, raw sim (0.30-0.80) and raw shrinkage_q (0-1) are at similar "
            "scales and compete fairly via score = sim*w_sim + q*w_q."
        ),
    )
    # v5 region retrieve mode (see docs/ALFWORLD_REGION_IMPROVEMENT_PLAN.md §14)
    p.add_argument(
        "--region_retrieve_mode",
        type=str,
        default="global",
        choices=["global", "quota_fixed", "quota_adaptive", "utility_anchor"],
        help=(
            "v5 candidate generation mode for region retrieval. "
            "'global' (default) = current behavior, parent's top-k by sim+Q. "
            "'quota_fixed' = reserve quota_max slots in top-k for top-N-region members. "
            "'quota_adaptive' = quota_fixed + 4 safety gates (min-sim, utility-margin, "
            "subtask-confidence, OOD guard). "
            "'utility_anchor' (v5.5) = anchor top-N from best region's member_ids directly, "
            "fill remaining with sim refinement. Solves 'noop_no_member' problem of quota. "
            "See docs/ALFWORLD_REGION_IMPROVEMENT_PLAN.md §14/§22 for details."
        ),
    )
    p.add_argument("--quota_max", type=int, default=3,
        help="v5: max number of region-promoted slots in top-k (default 3 of 5).")
    p.add_argument("--quota_min_sim_floor", type=float, default=0.5,
        help="v5 quota_adaptive: drop region candidates with sim < this (default 0.5).")
    p.add_argument("--quota_utility_margin", type=float, default=0.15,
        help="v5 quota_adaptive: if u_top1 - u_top4 < this, cap quota at 1 (default 0.15).")
    p.add_argument("--quota_region_min_count", type=int, default=30,
        help="v5: only consider regions with counts_by_subtask[s] >= this (default 30).")
    # v5.5 utility_anchor mode (see docs/ALFWORLD_REGION_IMPROVEMENT_PLAN.md §22)
    p.add_argument("--utility_anchor_count", type=int, default=3,
        help="v5.5: number of anchor memories from best region(s) (default 3 of k=5).")
    p.add_argument("--utility_anchor_topk_regions", type=int, default=1,
        help="v5.5: pool anchors from top-N regions by utility[target] (default 1).")
    p.add_argument("--utility_anchor_min_count", type=int, default=30,
        help="v5.5: only consider regions with counts_by_subtask[s] >= this (default 30).")
    # v10 holdout retrieval (see docs/ALFWORLD_V10_HOLDOUT_IMPL.md)
    p.add_argument(
        "--holdout_retrieval_mode",
        type=str,
        default=None,
        choices=["pure_d1", "hybrid", "sim_d1"],
        help=(
            "v10: holdout retrieval mode. None (default) = current zero-shot transfer. "
            "'pure_d1' = inject fixed top-k by region D1 quality (all queries get same memories). "
            "'hybrid' = top-N anchors by D1 + remaining by sim*D1 (recommended). "
            "'sim_d1' = all top-k by sim*D1 with pool=holdout_pool_size. "
            "Only active when target_subtask == --holdout_subtask."
        ),
    )
    p.add_argument("--holdout_pool_size", type=int, default=None,
        help="v10: sim candidate pool size for hybrid/sim_d1 modes (default 500).")
    p.add_argument("--holdout_d1_anchors", type=int, default=None,
        help="v10 hybrid: number of fixed D1 anchor memories (default 3 of 5).")
    # Region failure summary injection (see docs/REGION_FAILURE_SUMMARY.md)
    p.add_argument("--failure_summary_n_slots", type=int, default=None,
        help=(
            "Number of top-k slots reserved for failure memories (default None=disabled). "
            "When set (e.g. 2), retrieval guarantees 2 failure-memory slots in top-5, "
            "optionally replacing their raw content with per-region failure summaries."
        ))
    p.add_argument("--failure_summary_path", type=str, default=None,
        help="Path to region_failure_summaries.json (from build_region_failure_summaries.py).")
    p.add_argument("--failure_summary_no_replace", action="store_true",
        help="If set, keep raw failure memory content (don't replace with region summary). "
             "Use for ablation: 'raw failure' arm vs 'region summary' arm.")
    p.add_argument("--failure_summary_force_recall", action="store_true",
        help="Reserve failure slots with failure-only recall. Default is conditional replacement "
             "of failures naturally present in baseline top-K.")
    p.add_argument("--failure_summary_mode", type=str, default="region",
        choices=["region", "inline", "global"],
        help="'region' (default): use per-region aggregated summary. "
             "'inline': aggregate top-k retrieved failures on-the-fly (ablation, no region needed). "
             "'global': aggregate ALL failure memories into one global summary (ablation, no region needed).")
    p.add_argument("--failure_summary_k", type=int, default=None,
        help="For mode=inline: how many failure memories to retrieve and aggregate. "
             "If None, uses all retrieved failures (= n_slots). Set higher (e.g. 20) for richer aggregation.")
    p.add_argument("--success_summary_n_slots", type=int, default=None,
        help=(
            "Number of slots for region success pattern summary (default None=disabled). "
            "Aggregates effective procedural steps from a region's success memories. "
            "Symmetric to failure summary."
        ))
    p.add_argument("--success_summary_mode", type=str, default="append",
        choices=["append", "replace"],
        help="'append' (default): add success summary as extra slot(s). "
             "'replace': replace raw success memories with summary (symmetric to failure).")
    return p.parse_args()


logger = logging.getLogger(__name__)


def main():
    args = parse_args()
    try:
        cfg = MempConfig.from_yaml(args.config)
        setup_logging(project_root, cfg.experiment.experiment_name)

        out_dir = Path(cfg.experiment.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        run_id = os.environ.get("MEMRL_RUN_ID") or time.strftime('%Y%m%d-%H%M%S')
        log_dir = out_dir / "alfworld" / f"exp_{cfg.experiment.experiment_name}_{run_id}" / "local_cache"
        log_dir.mkdir(parents=True, exist_ok=True)

        llm_provider = OpenAILLM(
            api_key=cfg.llm.api_key,
            base_url=cfg.llm.base_url,
            model=cfg.llm.model,
            default_temperature=(args.temperature if args.temperature is not None else cfg.llm.temperature),
            default_max_tokens=(args.max_tokens if args.max_tokens is not None else cfg.llm.max_tokens),
            token_log_dir=str(log_dir),
        )
        embedding_provider = OpenAIEmbedder(
            api_key=cfg.embedding.api_key,
            base_url=cfg.embedding.base_url,
            model=cfg.embedding.model,
            max_text_len=getattr(cfg.embedding, "max_text_len", 4096),
            token_log_dir=str(log_dir),
        )

        temp_dir = tempfile.mkdtemp(prefix="memp_alfworld_run_")
        logger.info(f"Using temporary directory for runtime artifacts: {temp_dir}")

        mos_config = {
            "chat_model": {
                "backend": "openai",
                "config": {
                    "model_name_or_path": cfg.llm.model,
                    "api_key": cfg.llm.api_key,
                    "api_base": cfg.llm.base_url,
                },
            },
            "mem_reader": {
                "backend": "simple_struct",
                "config": {
                    "llm": {
                        "backend": "openai",
                        "config": {
                            "model_name_or_path": cfg.llm.model,
                            "api_key": cfg.llm.api_key,
                            "api_base": cfg.llm.base_url,
                        },
                    },
                    "embedder": {
                        "backend": "universal_api",
                        "config": {
                            "provider": "openai",
                            "model_name_or_path": cfg.embedding.model,
                            "api_key": cfg.embedding.api_key,
                            "base_url": cfg.embedding.base_url,
                        },
                    },
                    "chunker": {"backend": "sentence", "config": {"chunk_size": 500}},
                },
            },
            "user_manager": {"backend": "sqlite", "config": {"db_path": os.path.join(temp_dir, "users.db")}},
            "top_k": 5,
        }

        mos_config_path = os.path.join(temp_dir, "mos_config.json")
        with open(mos_config_path, "w", encoding="utf-8") as f:
            json.dump(mos_config, f)

        build_strategy = BuildStrategy(cfg.memory.build_strategy)
        retrieve_strategy = RetrieveStrategy(cfg.memory.retrieve_strategy)
        update_strategy = UpdateStrategy(cfg.memory.update_strategy)

        enable_value_driven = cfg.experiment.enable_value_driven
        rl_config = cfg.rl_config

        # A restored Mem0 checkpoint needs its original entity scope.
        user_id = os.environ.get("MEMRL_MEM0_USER_ID", "").strip() or f"alf_{os.getpid()}"

        if args.mem0:
            from memrl.service.mem0_memory_service import Mem0MemoryService

            mem0_qdrant_path = os.path.join(temp_dir, "mem0_qdrant")
            mem0_llm_base_url = os.environ.get("MEMRL_MEM0_LLM_BASE_URL", "") or cfg.llm.base_url
            mem0_llm_model = os.environ.get("MEMRL_MEM0_LLM_MODEL", "") or cfg.llm.model
            memory_service = Mem0MemoryService(
                llm_base_url=mem0_llm_base_url,
                llm_model=mem0_llm_model,
                llm_api_key=cfg.llm.api_key,
                embed_base_url=cfg.embedding.base_url,
                embed_model=cfg.embedding.model,
                embed_api_key=cfg.embedding.api_key,
                embedding_dims=int(getattr(cfg.embedding, "dimension", 0)) or None,
                qdrant_path=mem0_qdrant_path,
                collection_name=args.mem0_collection,
                top_k=cfg.memory.k_retrieve,
                infer=(args.mem0_infer == "true"),
                user_id=user_id,
            )
            logger.info(
                "Using Mem0MemoryService for ALFWorld baseline (infer=%s, collection=%s, extraction_llm=%s)",
                args.mem0_infer, args.mem0_collection, mem0_llm_base_url,
            )
        elif args.region:
            from memrl.service.region_manager import RegionManager
            from memrl.service.region_memory_service import RegionMemoryService
            from memrl.configs.task_hierarchy import TASK_HIERARCHY

            # Read region params from config (if present), CLI overrides config
            region_cfg = getattr(cfg, 'region', None)
            def _rcfg(key, cli_val, default):
                if cli_val is not None:
                    return cli_val
                if region_cfg and hasattr(region_cfg, key):
                    return getattr(region_cfg, key)
                return default

            region_utility_mode = _rcfg('region_utility_mode',
                args.region_utility_mode if args.region_utility_mode != 'beta' else None, 'beta')
            propagation_eta = _rcfg('propagation_eta', args.propagation_eta, 0.03)
            shrinkage_k = _rcfg('shrinkage_confidence_k', args.shrinkage_confidence_k, None)
            split_evidence_mode = _rcfg(
                'region_split_evidence_migration_mode',
                args.region_split_evidence_migration_mode,
                'soft_source_conserving',
            )

            region_manager = RegionManager(
                task_hierarchy=TASK_HIERARCHY,
                min_cluster_size=15,
                temperature=0.1,
                shrinkage_top_n=1,
                region_utility_mode=region_utility_mode,
                bayesian_smoothing_C=0.5,
                propagation_enabled=True,
                propagation_eta=float(propagation_eta),
                propagation_k=30,
                propagation_sim_min=0.40,
                region_split_evidence_migration_mode=split_evidence_mode,
                region_topology_updates_enabled=not args.region_freeze_topology,
                region_evidence_sharpen_alpha=(args.region_evidence_sharpen_alpha
                    if args.region_evidence_sharpen_alpha is not None else 2.0),
                cluster_space=args.region_cluster_space,
            )
            if shrinkage_k is not None:
                region_manager.shrinkage_confidence_k = float(shrinkage_k)
                logger.info(
                    "Region: confidence-gated shrinkage enabled (k=%.2f)",
                    shrinkage_k,
                )
            memory_service = RegionMemoryService(
                mos_config_path=mos_config_path,
                llm_provider=llm_provider,
                embedding_provider=embedding_provider,
                strategy_config=StrategyConfiguration(build_strategy, retrieve_strategy, update_strategy),
                user_id=user_id,
                num_workers=cfg.experiment.batch_size,
                max_keywords=cfg.memory.max_keywords,
                add_similarity_threshold=getattr(cfg.memory, "add_similarity_threshold", 0.9),
                enable_value_driven=enable_value_driven,
                rl_config=rl_config,
                db_max_concurrency=4,
                vector_dimension=int(getattr(cfg.embedding, "dimension", 4096)),
                sim_norm_mean=getattr(cfg.memory, "sim_norm_mean", None),
                sim_norm_std=getattr(cfg.memory, "sim_norm_std", None),
                region_manager=region_manager,
                region_gating_mode=args.region_gating_mode,
                region_value_mode=args.region_value_mode,
                explore_schedule=args.explore_schedule,
                explore_success_ratio=0.7,
                # ALFWorld: disable per-query embedding cache (near-zero hit rate
                # due to mutable observations, would grow to ~12GB RAM + 20GB disk).
                query_embedding_cache_enabled=False,
                # Optional: disable z-norm on sim/q during scoring (releases more
                # region signal in additive gating by preserving raw magnitudes).
                use_z_score_normalization=(not args.no_z_norm),
                # v5: quota-based region recall (see docs §14)
                region_retrieve_mode=args.region_retrieve_mode,
                quota_max=args.quota_max,
                quota_min_sim_floor=args.quota_min_sim_floor,
                quota_utility_margin=args.quota_utility_margin,
                quota_region_min_count=args.quota_region_min_count,
                # v5.5 utility_anchor params (see docs §22)
                utility_anchor_count=args.utility_anchor_count,
                utility_anchor_topk_regions=args.utility_anchor_topk_regions,
                utility_anchor_min_count=args.utility_anchor_min_count,
                # v10: holdout-specific retrieval (see docs/ALFWORLD_V10_HOLDOUT_IMPL.md)
                # CLI overrides yaml. holdout_retrieval_mode=None means default zero-shot transfer.
                holdout_retrieval_mode=(
                    args.holdout_retrieval_mode
                    if args.holdout_retrieval_mode is not None
                    else getattr(cfg.experiment, "holdout_retrieval_mode", None)
                ),
                holdout_pool_size=(
                    args.holdout_pool_size
                    if args.holdout_pool_size is not None
                    else getattr(cfg.experiment, "holdout_pool_size", 500)
                ),
                holdout_d1_anchors=(
                    args.holdout_d1_anchors
                    if args.holdout_d1_anchors is not None
                    else getattr(cfg.experiment, "holdout_d1_anchors", 3)
                ),
                holdout_subtask=(args.holdout_subtask
                    or getattr(cfg.experiment, "holdout_subtask", None)),
                strip_thinking=getattr(cfg.experiment, "strip_thinking", False),
                max_trajectory_len=getattr(cfg.experiment, "max_trajectory_len", 0),
                max_candidates_per_sim_key=args.max_candidates_per_sim_key,
            )
            logger.info(
                "Using RegionMemoryService for ALFWorld (7 task type subtasks). "
                "z-score normalization: %s. region_retrieve_mode=%s (quota_max=%d).",
                "DISABLED (--no_z_norm)" if args.no_z_norm else "enabled (default)",
                args.region_retrieve_mode, args.quota_max,
            )
        else:
            memory_service = MemoryService(
                mos_config_path=mos_config_path,
                llm_provider=llm_provider,
                embedding_provider=embedding_provider,
                strategy_config=StrategyConfiguration(build_strategy, retrieve_strategy, update_strategy),
                user_id=user_id,
                num_workers=cfg.experiment.batch_size,
                max_keywords=cfg.memory.max_keywords,
                add_similarity_threshold=getattr(cfg.memory, "add_similarity_threshold", 0.9),
                enable_value_driven=enable_value_driven,
                rl_config=rl_config,
                db_max_concurrency=4,
                vector_dimension=int(getattr(cfg.embedding, "dimension", 4096)),
                sim_norm_mean=getattr(cfg.memory, "sim_norm_mean", None),
                sim_norm_std=getattr(cfg.memory, "sim_norm_std", None),
                # ALFWorld: disable per-query embedding cache (see above)
                query_embedding_cache_enabled=False,
                # Honor --no_z_norm flag for baseline as well (for fair comparison)
                use_z_score_normalization=(not args.no_z_norm),
                # Thinking model support: strip <think> and truncate trajectory in memory builder
                strip_thinking=getattr(cfg.experiment, "strip_thinking", False),
                max_trajectory_len=getattr(cfg.experiment, "max_trajectory_len", 0),
            )
            logger.info(
                "Using MemoryService for ALFWorld (baseline, no region). "
                "z-score normalization on sim/q during scoring: %s",
                "DISABLED (--no_z_norm)" if args.no_z_norm else "enabled (default)",
            )

        # Load checkpoint if specified
        load_from_checkpoint = getattr(cfg.memory, "load_from_checkpoint", False)
        checkpoint_path = getattr(cfg.memory, "checkpoint_path", None)
        if load_from_checkpoint and checkpoint_path:
            logger.info(f"Loading memory checkpoint from: {checkpoint_path}")
            try:
                num_loaded = memory_service.load_checkpoint_snapshot(checkpoint_path)
                logger.info(f"Loaded {num_loaded} memories from checkpoint")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}", exc_info=True)
                raise

        with open(project_root / cfg.experiment.few_shot_path, "r", encoding="utf-8") as f:
            few_shot_examples = json.load(f)
        agent = MempAgent(
            llm_provider=llm_provider,
            few_shot_examples=few_shot_examples,
            max_recent_turns=cfg.experiment.max_recent_turns,
            max_history_response_chars=getattr(cfg.experiment, "max_history_response_chars", 0),
            no_think=getattr(cfg.experiment, "no_think", False),
            force_think=getattr(cfg.experiment, "force_think", False),
        )

        alfworld_config_path = project_root / "configs" / "envs" / "alfworld.yaml"
        runner = AlfworldRunner(
            agent=agent,
            root=project_root,
            env_config=alfworld_config_path,
            memory_service=memory_service,
            exp_name=cfg.experiment.experiment_name,
            ck_dir=log_dir,
            random_seed=cfg.experiment.random_seed,
            num_section=cfg.experiment.num_sections,
            batch_size=cfg.experiment.batch_size,
            max_steps=cfg.experiment.max_steps,
            rl_config=rl_config,
            bon=cfg.experiment.bon,
            retrieve_k=cfg.memory.k_retrieve,
            mode=cfg.experiment.mode,
            valid_interval=cfg.experiment.valid_interval,
            test_interval=cfg.experiment.test_interval,
            dataset_ratio=cfg.experiment.dataset_ratio,
            ckpt_resume_enabled=getattr(cfg.experiment, "ckpt_resume_enabled", False),
            ckpt_resume_path=getattr(cfg.experiment, "ckpt_resume_path", None),
            ckpt_resume_epoch=getattr(cfg.experiment, "ckpt_resume_epoch", None),
            baseline_mode=getattr(cfg.experiment, "baseline_mode", None),
            baseline_k=getattr(cfg.experiment, "baseline_k", 10),
            holdout_subtask=args.holdout_subtask or getattr(cfg.experiment, "holdout_subtask", None),
            val_lambda_max=args.val_lambda_max if args.val_lambda_max is not None else getattr(cfg.experiment, "val_lambda_max", None),
            holdout_eval_pools=(
                [p.strip() for p in args.holdout_eval_pools.split(',') if p.strip()]
                if getattr(args, 'holdout_eval_pools', None) else None
            ),
            n_eval_runs=getattr(cfg.experiment, "n_eval_runs", 1),
            eval_temperature=getattr(cfg.experiment, "eval_temperature", None),
            reset_legacy_region_evidence_on_resume=args.region_reset_legacy_evidence_on_resume,
            region_evidence_sharpen_alpha_override=args.region_evidence_sharpen_alpha,
        )
        # Full memory snapshots define the true batch-level resume point.
        # Preserve the legacy default unless a specific experiment overrides it.
        batch_ckpt_interval = int(getattr(cfg.experiment, "batch_checkpoint_interval", 10) or 10)
        runner._batch_ckpt_interval = max(1, batch_ckpt_interval)
        runner._batch_ckpt_keep = max(
            1, int(getattr(cfg.experiment, "batch_checkpoint_keep", 3) or 3)
        )
        logger.info(
            "Batch checkpoint policy: every %d batches; keep latest %d per section.",
            runner._batch_ckpt_interval, runner._batch_ckpt_keep,
        )
        runner.skip_initial_eval = getattr(args, 'skip_initial_eval', False)
        runner.eval_train_in_test_mode = getattr(args, 'eval_train', False)
        runner.id_eval_only = getattr(args, 'id_eval_only', False)
        runner._selfrag_enabled = getattr(args, 'selfrag', False)
        if args.region_cluster_init_step is not None:
            runner._region_cluster_init_step = max(1, int(args.region_cluster_init_step))
        if args.region_merge_interval is not None:
            runner._region_merge_interval = max(1, int(args.region_merge_interval))
        runner._region_mid_epoch_topology_enabled = not args.region_disable_mid_epoch_topology
        runner._region_topology_cooldown_sections = max(0, int(args.region_topology_cooldown_sections))
        logger.info(
            "Region topology schedule: init_step=%d, mid_epoch=%s, merge_interval=%d, cooldown_sections=%d",
            getattr(runner, '_region_cluster_init_step', 1500),
            runner._region_mid_epoch_topology_enabled,
            getattr(runner, '_region_merge_interval', 1200),
            runner._region_topology_cooldown_sections,
        )

        # Region failure summary injection (see docs/REGION_FAILURE_SUMMARY.md)
        failure_summary_n_slots = getattr(args, 'failure_summary_n_slots', None)
        failure_summary_path = getattr(args, 'failure_summary_path', None)
        failure_summary_no_replace = getattr(args, 'failure_summary_no_replace', False)
        if failure_summary_n_slots and failure_summary_n_slots > 0:
            runner.configure_failure_summary(
                n_slots=failure_summary_n_slots,
                summaries_path=failure_summary_path,
                replace_with_summary=(not failure_summary_no_replace),
                mode=getattr(args, 'failure_summary_mode', 'region'),
                inline_k=getattr(args, 'failure_summary_k', None),
                force_recall=getattr(args, 'failure_summary_force_recall', False),
            )

        success_summary_n_slots = getattr(args, 'success_summary_n_slots', None)
        success_summary_mode = getattr(args, 'success_summary_mode', 'append')
        if success_summary_n_slots and success_summary_n_slots > 0:
            runner.configure_success_summary(
                n_slots=success_summary_n_slots,
                mode=success_summary_mode,
            )

        runner.run()

    except Exception as e:
        logger.error(f"An unhandled error occurred during the experiment: {e}", exc_info=True)
        # LOW #12: propagate failure as nonzero exit so orchestrators
        # (sbatch wrappers, watchdogs) don't mark broken runs as success.
        sys.exit(1)
    finally:
        # LOW #11: mos_config_path was already read at init; MOS keeps its config
        # in memory. Safe to remove temp_dir here.
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            # CPFS can transiently report ENOTEMPTY while directory updates are
            # propagating. Cleanup must not mask the experiment's real status.
            shutil.rmtree(temp_dir, ignore_errors=True)
            if os.path.exists(temp_dir):
                logger.warning("Temporary directory cleanup deferred: %s", temp_dir)
            else:
                logger.info(f"Cleaned up temporary directory: {temp_dir}")


if __name__ == "__main__":
    main()
