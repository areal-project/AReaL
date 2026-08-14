import argparse
import json as _json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from memrl.configs.config import MempConfig
from memrl.providers.llm import OpenAILLM
from memrl.providers.embedding import OpenAIEmbedder
from memrl.service.region_memory_service import RegionMemoryService
from memrl.service.region_manager import RegionManager
from memrl.service.strategies import (
    BuildStrategy,
    RetrieveStrategy,
    UpdateStrategy,
    StrategyConfiguration,
)
from memrl.run.bcb_region_runner import BCBRegionRunner, BCBSelection
from memrl.configs.task_hierarchy import TASK_HIERARCHY

DEFAULT_SPLIT_FILES = {
    "hard": project_root / "configs" / "bigcodebench" / "splits" / "hard_seed42.json",
    "full": project_root / "configs" / "bigcodebench" / "splits" / "full_seed42.json",
}


def setup_logging(project_root: Path, name: str) -> None:
    log_dir = project_root / "logs" / name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = f"{name}_{time.strftime('%Y%m%d-%H%M%S')}.log"
    log_filepath = log_dir / log_filename

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    logging.info("Logging configured. Log file: %s", log_filepath)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run BigCodeBench (BCB) with Region-based Memory Transfer"
    )
    p.add_argument(
        "--config",
        type=str,
        default=str(project_root / "configs" / "rl_bcb_config.deepseek_region.yaml"),
    )
    p.add_argument("--subset", type=str, default="full", choices=["hard", "full"])
    p.add_argument(
        "--split", type=str, default="instruct", choices=["instruct", "complete"]
    )
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--retrieve_threshold", type=float, default=None)
    p.add_argument("--memory_budget_tokens", type=int, default=None)
    p.add_argument("--split_file", type=str, default=None)
    p.add_argument("--data_path", type=str, default=None)
    p.add_argument(
        "--bcb_repo",
        type=str,
        default=str(project_root / "3rdparty" / "bigcodebench-main"),
    )
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--max_tokens", type=int, default=None)
    p.add_argument("--retrieve_k", type=int, default=None)
    p.add_argument("--eval_timeout", type=float, default=240.0)
    p.add_argument("--untrusted_hard_timeout", type=float, default=300.0)
    p.add_argument("--resume_from", type=str, default=None)
    p.add_argument("--resume_epoch", type=int, default=None)
    p.add_argument("--resume_step", type=int, default=None)
    p.add_argument("--checkpoint_interval", type=int, default=50)
    p.add_argument("--max_checkpoints", type=int, default=3)
    p.add_argument("--strip_think", action="store_true", default=False)
    p.add_argument(
        "--n_eval_runs", type=int, default=1,
        help="Number of independent eval runs for confidence intervals (default: 1).",
    )
    p.add_argument(
        "--eval_temperature", type=float, default=None,
        help="Temperature used only by independent multi-eval runs.",
    )
    p.add_argument(
        "--multi_eval_epochs", type=str, default=None,
        help="Which epochs receive multi-eval: last, all, or comma-separated epoch numbers.",
    )
    # Region-specific args
    p.add_argument("--k_global", type=int, default=30)
    p.add_argument("--k_local", type=int, default=10)
    p.add_argument("--region_retrieve_mode", type=str, default="global",
                   choices=["global", "quota_fixed", "quota_adaptive", "utility_anchor"],
                   help="'global' is the original (pre-v5) sim+rerank behavior (was 'post'). "
                        "quota_* and utility_anchor are ALFWorld v5/v5.5 modes that change "
                        "candidate generation; not applicable to BCB unless explicitly tested.")
    p.add_argument("--region_gating_mode", type=str, default="multiplicative",
                   choices=["multiplicative", "additive"])
    p.add_argument("--region_temperature", type=float, default=1.0,
                   help="Softmax temperature for region membership (lower=sharper)")
    p.add_argument("--shrinkage_top_n", type=int, default=3,
                   help="Top-N regions for shrinkage utility (1=hard assignment)")
    p.add_argument("--shrinkage_min_utility_margin", type=float, default=0.0,
                   help="Abstain to per-memory Q when Region top1-top2 utility margin is below this value (0=disabled; Leaf-v4 uses 0.05).")
    p.add_argument("--region_precluster_evidence_mode", choices=["off", "soft_source_backfill"], default="off",
                   help="Whether raw direct outcomes before the first cluster are routed once into initial Region source evidence.")
    p.add_argument("--region_precluster_evidence_scale", type=float, default=1.0,
                   help="Discount applied to raw pre-cluster evidence during one-time soft-source backfill (Leaf-v5 uses 0.75).")
    p.add_argument("--region_min_cluster_size", type=int, default=None,
                   help="Override min_cluster_size for HDBSCAN (default: k_local//2)")
    p.add_argument("--region_min_samples", type=int, default=0,
                   help="Override min_samples for HDBSCAN (0=auto: min_cluster_size//2)")
    p.add_argument("--region_cluster_selection_method", type=str, default="eom",
                   choices=["eom", "leaf"],
                   help="HDBSCAN cluster selection: 'eom' (fewer large clusters) or 'leaf' (more fine-grained)")
    p.add_argument("--region_max_region_share", type=float, default=0.0,
                   help="Force split regions larger than this fraction of total (0=disabled, e.g. 0.18)")
    p.add_argument("--region_topology_cooldown_epochs", type=int, default=0,
                   help="After initial clustering or any split/merge, require this many complete epochs before the next topology edit (0=disabled).")
    p.add_argument("--region_disable_mid_epoch_topology", action="store_true",
                   help="Disable periodic mid-epoch split/merge; topology edits remain at epoch boundary.")
    p.add_argument("--fixed_initial_topology_epoch", type=int, default=None,
                   help="For an explicit step-snapshot continuation only: mark an already-clustered initial topology as edited in this epoch so cooldown begins without reclustering.")
    p.add_argument("--region_utility_mode", type=str, default="ema",
                   choices=["ema", "beta"],
                   help="Region utility update: ema (weighted EMA) or beta (Beta posterior)")
    p.add_argument("--region_split_evidence_migration_mode",
                   choices=["soft_source_conserving", "hard_member_rebase", "legacy_qcount_reconstruction"],
                   default="soft_source_conserving",
                   help=("Topology-only posterior migration: conserve source-attributed soft "
                         "evidence (default), or rebase on current hard members for ablation. "
                         "Does not change online soft region updates."))
    p.add_argument("--region_split_range_fraction", type=float, default=0.15,
                   help="Variance split threshold as a fraction of utility range (historical default 0.15; Leaf-v2 uses 0.20).")
    p.add_argument("--region_max_variance_splits_per_epoch", type=int, default=0,
                   help="Maximum variance-triggered region splits at an epoch boundary (0=historical unlimited).")
    p.add_argument("--region_split_min_effective_evidence", type=float, default=0.0,
                   help="Minimum soft effective region evidence required before a variance split (0=disabled).")
    p.add_argument("--region_smoothing_C", type=float, default=0.5,
                   help="Bayesian smoothing prior count for region utility (lower = less smoothing, more signal)")
    p.add_argument("--shrinkage_lambda_max", type=float, default=None,
                   help="Max weight for per-memory Q in shrinkage (0.0=pure region utility, 0.8=default)")
    p.add_argument("--val_lambda_max", type=float, default=None,
                   help="Override lambda_max for val phase only (e.g., 0.15 for region-heavy val)")
    p.add_argument("--shrinkage_confidence_k", type=float, default=None,
                   help="Confidence-gated lambda: lambda=lmax*n/(n+k). Set k>0 to enable (e.g., 3.0)")
    p.add_argument("--shrinkage_tau_sq", type=float, default=None,
                   help="Prior variance of true Q within region (default 0.05)")
    p.add_argument("--shrinkage_sigma_sq", type=float, default=None,
                   help="Observation noise variance (default 0.25)")
    p.add_argument("--outcome_blend_beta", type=float, default=None,
                   help="Blend outcome signal into Q: final_q = beta*outcome + (1-beta)*q. "
                        "E.g., 0.3 strongly penalizes failure memories in ranking.")
    p.add_argument("--propagation_eta", type=float, default=0.03,
                   help="Learning rate for similarity-based Q propagation (fraction of alpha)")
    p.add_argument("--propagation_k", type=int, default=30,
                   help="Number of neighbors for Q propagation")
    p.add_argument("--propagation_sim_min", type=float, default=0.40,
                   help="Minimum cosine similarity for Q propagation")
    p.add_argument("--no_propagation", action="store_true",
                   help="Disable similarity-based Q propagation")
    p.add_argument("--region_variance_weighted_dist", action="store_true",
                   help="Weight distance dims by variance (high-variance dims dominate clustering)")
    p.add_argument("--task_cluster_k", type=int, default=12,
                   help="Number of auto task clusters (0=use domain subtasks)")
    p.add_argument("--explore_schedule", type=str, default="0,4,3,2,2,1,1,1,1,0",
                   help="Comma-separated exploration slots per epoch (e.g., '0,4,3,2,2,1,1,1,1,0')")
    p.add_argument("--explore_success_ratio", type=float, default=0.7,
                   help="Fraction of exploration slots to fill with success memories (0-1)")
    p.add_argument("--holdout_subtask", type=str, default=None,
                   help="Subtask to holdout from training (e.g., 'bcb/Computation'). "
                        "Tasks in this subtask are excluded from train and used as holdout val.")
    p.add_argument("--retrieval_mode", type=str, default="current",
                   choices=["current", "no_mem", "oracle"],
                   help="Retrieval mode for holdout eval. 'current'=normal region+transfer retrieval, "
                        "'no_mem'=skip retrieval (no memory in prompt), "
                        "'oracle'=hand-pick best memories by ground-truth overlap.")
    p.add_argument("--oracle_snapshot_dir", type=str, default=None,
                   help="Path to memory snapshot dir (.../snapshot/N) for oracle retrieval pool. "
                        "Required when --retrieval_mode=oracle.")
    # --- Region failure summary ---
    p.add_argument("--failure_summary_n_slots", type=int, default=0,
                   help="Number of top-K slots reserved for failure memories with region summary. "
                        "0=disabled (default). E.g., 2 = top-8 success + 2 failure summaries.")
    p.add_argument("--no_failure_summary_replace", action="store_true", default=False,
                   help="Keep raw failure content instead of replacing with region summary (ablation).")
    p.add_argument("--failure_summary_lib_filter", action="store_true", default=False,
                   help="Filter region failures by lib overlap with current task before building "
                        "summary. Only contract-relevant failures are included. Slot is dropped "
                        "entirely if no overlap is found.")
    p.add_argument("--failure_summary_contract_filter", action="store_true", default=False,
                   help="Conservatively gate FS evidence by task contract: exact return type, "
                        "or shared non-pure I/O family plus exception tag. If no compatible "
                        "failure exists, drop the FS slot and backfill success context.")
    p.add_argument("--failure_summary_min_compatible", type=int, default=1,
                   help="Minimum number of contract-compatible failure reflections required to inject an FS (default 1 preserves prior behavior).")
    p.add_argument("--failure_summary_force_recall", action="store_true", default=False,
                   help="Force-recall failure memories to fill FS slots when baseline top-K has none; total top-K remains fixed by replacing lowest-ranked slots.")
    p.add_argument("--failure_summary_fmmatch", action="store_true", default=False,
                   help="Use failure-mode-matched conditional summary (3-layer design): "
                        "for tasks with prior-epoch failures, embed the task's own failure mode, "
                        "find top-k similar failure reflections in its utility-region, then LLM "
                        "synthesizes a conditional if/elif/else contract guide. Tasks that always "
                        "pass get all-success memory (no failure slot). Cold-start tasks fall back "
                        "to the standard region failure summary. Benchmark-agnostic.")
    p.add_argument("--failure_summary_fmmatch_topk", type=int, default=5,
                   help="Number of most-similar failure reflections to feed the LLM in fmmatch mode.")
    p.add_argument("--failure_summary_fmmatch_backfill", action="store_true", default=False,
                   help="Backfill success memories when fmmatch discards failure mems from "
                        "selected set. Prevents context starvation (e.g., 1 success + 1 fmmatch "
                        "= 2 mems → backfilled to 4 success + 1 fmmatch = 5 mems).")
    # --- Region experience cards ---
    p.add_argument("--experience_cards_n_slots", type=int, default=0,
                   help="Number of top-K slots to replace with region experience cards. "
                        "0=disabled (default). E.g., 1 = replace weakest slot with atomic fact card.")
    # --- Retrieval gate ---
    p.add_argument("--retrieval_gate_signal", type=str, default=None,
                   choices=["success_ratio", "mean_q"],
                   help="Enable post-retrieval gate. If retrieved memory quality (measured by "
                        "this signal) is below --retrieval_gate_threshold, drop memory entirely.")
    p.add_argument("--retrieval_gate_threshold", type=float, default=0.8,
                   help="Threshold for retrieval gate. Default 0.8 for success_ratio, 0.4 for mean_q.")
    # --- Region meta header ---
    p.add_argument("--region_meta_header", action="store_true", default=False,
                   help="Prepend region reliability header + anti-anchoring instruction to memory context.")
    # --- Meta-info mode ---
    p.add_argument("--meta_info_mode", action="store_true", default=False,
                   help="Replace full memory content with structured meta-info hints "
                        "(libraries, return types, signatures, task summaries). "
                        "Reduces model anchoring to specific implementations.")
    p.add_argument("--eval_only", action="store_true",
                   help="Skip training; load checkpoint snapshot and run holdout eval once. "
                        "Useful for oracle/no_mem/current comparison on a fixed memory pool.")
    # --- val_ablations mode (formerly run_val_ablation.py) -----------------
    p.add_argument("--val_ablations", type=str, default=None,
                   help="Comma-separated ablation names (or 'all'). If set, switches to "
                        "val-only mode: loads --checkpoint, runs val for each ablation, "
                        "shares vLLM across ablations. See memrl/run/bcb_ablation_runner.py.")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Snapshot dir for --val_ablations mode (e.g., .../epoch10/snapshot/10).")
    p.add_argument("--ablation_output_dir", type=str, default=None,
                   help="Where to put per-ablation results (default: <output_dir>/val_ablation_<timestamp>).")
    return p.parse_args()


def _build_region_manager(args):
    """Construct RegionManager with all CLI region/shrinkage overrides applied.

    Used by both the training path and the val_ablation path so the same args
    map to the same RegionManager — eliminates the silent-drift bug class.
    """
    region_min_cs = args.region_min_cluster_size or max(3, args.k_local // 2)
    rm = RegionManager(
        task_hierarchy=TASK_HIERARCHY,
        min_cluster_size=region_min_cs,
        min_samples=args.region_min_samples,
        cluster_selection_method=args.region_cluster_selection_method,
        max_region_share=args.region_max_region_share,
        temperature=args.region_temperature,
        shrinkage_top_n=args.shrinkage_top_n,
        shrinkage_min_utility_margin=args.shrinkage_min_utility_margin,
        region_precluster_evidence_mode=args.region_precluster_evidence_mode,
        region_precluster_evidence_scale=args.region_precluster_evidence_scale,
        region_utility_mode=args.region_utility_mode,
        bayesian_smoothing_C=args.region_smoothing_C,
        propagation_enabled=not args.no_propagation,
        propagation_eta=args.propagation_eta,
        propagation_k=args.propagation_k,
        propagation_sim_min=args.propagation_sim_min,
        variance_weighted_dist=getattr(args, 'region_variance_weighted_dist', False),
        region_split_evidence_migration_mode=args.region_split_evidence_migration_mode,
        region_split_range_fraction=args.region_split_range_fraction,
        region_max_variance_splits_per_epoch=args.region_max_variance_splits_per_epoch,
        region_split_min_effective_evidence=args.region_split_min_effective_evidence,
    )
    if args.shrinkage_lambda_max is not None:
        rm.shrinkage_lambda_max = args.shrinkage_lambda_max
    if args.shrinkage_tau_sq is not None:
        rm.shrinkage_tau_sq = args.shrinkage_tau_sq
    if args.shrinkage_sigma_sq is not None:
        rm.shrinkage_sigma_sq = args.shrinkage_sigma_sq
    if args.shrinkage_confidence_k is not None:
        rm.shrinkage_confidence_k = args.shrinkage_confidence_k
    return rm


def _build_mos_config(cfg, temp_dir, top_k):
    """MemOS config JSON used by both training and val_ablation paths."""
    return {
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
                        "provider": cfg.embedding.provider,
                        "model_name_or_path": cfg.embedding.model,
                        "api_key": cfg.embedding.api_key,
                        "base_url": cfg.embedding.base_url,
                    },
                },
                "chunker": {"backend": "sentence", "config": {"chunk_size": 500}},
            },
        },
        "user_manager": {
            "backend": "sqlite",
            "config": {"db_path": os.path.join(temp_dir, "users.db")},
        },
        "top_k": int(top_k),
    }


def run_val_ablations(args, cfg, llm, embedder, ablation_names, out_root):
    """Val-only ablation grid (formerly run_val_ablation.py main).

    Loads --checkpoint once per ablation, shares LLM/embedder providers and
    val_query_cache across ablations. Each ablation is constructed via the
    SAME _build_region_manager / _build_mos_config helpers used by training,
    so config knobs cannot silently drift away from the training run.
    """
    import hashlib
    import shutil
    import tempfile

    from memrl.bigcodebench_eval.task_wrappers import load_bcb_data, split_dataset, get_prompt
    from memrl.run.bcb_ablation_runner import (
        ABLATIONS,
        AblationRunner,
        _parse_outcome,
        _read_outcome_from_mem,
    )

    logger = logging.getLogger(__name__)

    if not args.checkpoint:
        raise SystemExit("--val_ablations requires --checkpoint <snapshot_dir>")

    if ablation_names == "all":
        selected = list(ABLATIONS.keys())
    else:
        selected = [a.strip() for a in ablation_names.split(",")]
    unknown = [a for a in selected if a not in ABLATIONS]
    if unknown:
        raise SystemExit(
            f"Unknown ablation name(s): {unknown}. Valid: {sorted(ABLATIONS.keys())}"
        )

    out_dir = Path(args.ablation_output_dir or (
        out_root / f"val_ablation_{time.strftime('%Y%m%d_%H%M%S')}"
    )).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load problems + split (mirrors training: train_ratio=0.7, seed=42)
    problems = load_bcb_data(subset=args.subset, data_path=args.data_path)
    _, val_ids = split_dataset(
        problems, train_ratio=float(args.train_ratio), seed=int(args.seed),
        split_file=args.split_file,
    )
    prompts = [get_prompt(problems[tid], split=args.split) for tid in val_ids]

    # --- Embedding cache: prefer ckpt-cached embeddings to avoid embedder drift -----
    val_query_cache_path = out_dir / "val_query_embeddings.json"
    val_query_cache = {}
    ckpt_emb_path = os.path.join(args.checkpoint, "local_cache", "query_embeddings.json")
    if os.path.isfile(ckpt_emb_path):
        try:
            with open(ckpt_emb_path) as f:
                ckpt_emb = _json.load(f)
            for p in prompts:
                v = ckpt_emb.get(p)
                if isinstance(v, list):
                    val_query_cache[p] = v
            logger.info("Loaded %d/%d val query embeddings from ckpt cache",
                        len(val_query_cache), len(prompts))
            del ckpt_emb
        except Exception:
            logger.warning("Failed to read %s", ckpt_emb_path, exc_info=True)
    if len(val_query_cache) < len(prompts) and val_query_cache_path.exists():
        with open(val_query_cache_path) as f:
            prev = _json.load(f)
        added = 0
        for p in prompts:
            if p not in val_query_cache and isinstance(prev.get(p), list):
                val_query_cache[p] = prev[p]
                added += 1
        if added:
            logger.info("Filled %d more val query embeddings from prior out_dir", added)
    missing = [p for p in prompts if p not in val_query_cache]
    if missing:
        outcome_abls = [a for a in selected if ABLATIONS.get(a, {}).get("outcome_blend_beta") is not None]
        if outcome_abls:
            raise RuntimeError(
                f"{len(missing)}/{len(prompts)} val prompts missing from ckpt cache; "
                f"refusing to call live embedder because outcome ablations {outcome_abls} "
                f"require bit-identical embedding provenance."
            )
        logger.info("Computing %d missing val query embeddings via embedder...", len(missing))
        BATCH = 32
        for i in range(0, len(missing), BATCH):
            batch = missing[i:i + BATCH]
            try:
                vecs = embedder.embed(batch)
                for prompt, vec in zip(batch, vecs):
                    val_query_cache[prompt] = vec if isinstance(vec, list) else vec.tolist()
            except Exception as e:
                logger.warning("Batch embedding failed at %d: %s", i, e)
        with open(val_query_cache_path, "w") as f:
            _json.dump(val_query_cache, f)
    else:
        logger.info("All %d val query embeddings sourced from cache (no embedder calls)", len(val_query_cache))

    logger.info(
        "PROVIDERS llm: model=%s base_url=%s | embedder: model=%s base_url=%s",
        getattr(cfg.llm, "model", "?"), getattr(cfg.llm, "base_url", "?"),
        getattr(cfg.embedding, "model", "?"), getattr(cfg.embedding, "base_url", "?"),
    )

    results_summary = {}
    for abl_name in selected:
        abl = ABLATIONS[abl_name]
        logger.info("=" * 60)
        logger.info("Ablation: %s — %s", abl_name, abl["desc"])
        logger.info("=" * 60)

        abl_dir = out_dir / abl_name
        abl_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = tempfile.mkdtemp(prefix=f"ablation_{abl_name}_")
        try:
            user_id = f"ablation_{abl_name}_{os.getpid()}"
            mos_config = _build_mos_config(cfg, temp_dir, top_k=abl["k"])
            mos_config_path = os.path.join(temp_dir, "mos_config.json")
            with open(mos_config_path, "w") as f:
                _json.dump(mos_config, f)

            region_manager = _build_region_manager(args)

            memsvc = RegionMemoryService(
                mos_config_path=mos_config_path,
                llm_provider=llm,
                embedding_provider=embedder,
                strategy_config=StrategyConfiguration(
                    BuildStrategy(cfg.memory.build_strategy),
                    RetrieveStrategy(cfg.memory.retrieve_strategy),
                    UpdateStrategy(cfg.memory.update_strategy),
                ),
                user_id=user_id,
                num_workers=cfg.experiment.batch_size,
                max_keywords=cfg.memory.max_keywords,
                add_similarity_threshold=getattr(cfg.memory, "add_similarity_threshold", 0.9),
                enable_value_driven=cfg.experiment.enable_value_driven,
                rl_config=cfg.rl_config,
                db_max_concurrency=4,
                sim_norm_mean=getattr(cfg.memory, "sim_norm_mean", 0.1856827586889267),
                sim_norm_std=getattr(cfg.memory, "sim_norm_std", 0.09407906234264374),
                vector_dimension=getattr(cfg.embedding, "dimension", 4096),
                region_manager=region_manager,
                region_retrieve_mode=args.region_retrieve_mode,
                region_gating_mode=args.region_gating_mode,
                total_epochs=1,  # val-only mode; no exploration schedule needed
            )

            # Load checkpoint — copy qdrant to temp dir to avoid lock conflicts.
            logger.info("Loading checkpoint from: %s", args.checkpoint)
            qdrant_src = os.path.join(args.checkpoint, "qdrant")
            qdrant_dst = os.path.join(temp_dir, "qdrant")
            if os.path.isdir(qdrant_src):
                shutil.copytree(qdrant_src, qdrant_dst)
                meta_path = os.path.join(args.checkpoint, "snapshot_meta.json")
                if os.path.isfile(meta_path):
                    patched_meta = os.path.join(temp_dir, "snapshot_meta.json")
                    with open(meta_path) as f:
                        meta = _json.load(f)
                    meta["qdrant_dir"] = qdrant_dst
                    with open(patched_meta, "w") as f:
                        _json.dump(meta, f)
                for item in ["cube", "local_cache", "snapshot_meta.json"]:
                    src = os.path.join(args.checkpoint, item)
                    dst = os.path.join(temp_dir, item)
                    if os.path.isdir(src) and not os.path.exists(dst):
                        shutil.copytree(src, dst)
                    elif os.path.isfile(src) and not os.path.exists(dst):
                        shutil.copy2(src, dst)
                memsvc.load_checkpoint_snapshot(temp_dir)
            else:
                memsvc.load_checkpoint_snapshot(args.checkpoint)
            logger.info(
                "Checkpoint loaded: %d memories in dict_memory, %d in region_manager",
                sum(len(v) for v in getattr(memsvc, "dict_memory", {}).values()),
                len(region_manager.subtask_q),
            )

            # Apply outcome blend AFTER ckpt load (hard guarantee against snapshot reset).
            if abl.get("outcome_blend_beta") is not None:
                memsvc.configure_outcome_blend(abl["outcome_blend_beta"])

            # Invariance fingerprint: should be identical across ablations modulo Q.
            mem_cache = getattr(memsvc, "_mem_cache", None) or {}
            dict_memory = getattr(memsvc, "dict_memory", {}) or {}
            all_mem_ids = sorted({str(mid) for ids in dict_memory.values() for mid in ids})
            mem_ids_hash = hashlib.sha1(("\n".join(all_mem_ids)).encode()).hexdigest()[:12]
            subtask_q_hash = hashlib.sha1(
                _json.dumps(
                    sorted((mid, sorted((k, round(v, 6)) for k, v in (sub or {}).items()))
                           for mid, sub in (region_manager.subtask_q or {}).items()),
                    default=str,
                ).encode()
            ).hexdigest()[:12]

            outcome_present = outcome_success = outcome_failure = 0
            for mid, mo in mem_cache.items():
                o = _parse_outcome(_read_outcome_from_mem(mo))
                if o is None:
                    continue
                outcome_present += 1
                if o == "success":
                    outcome_success += 1
                elif o == "failure":
                    outcome_failure += 1
            mem_cache_size = len(mem_cache)
            coverage_pct = (100.0 * outcome_present / mem_cache_size) if mem_cache_size else 0.0
            logger.info(
                "[%s] FINGERPRINT mem_ids=%d hash=%s | subtask_q_hash=%s | outcome_coverage=%d/%d (%.1f%%, success=%d failure=%d)",
                abl_name, len(all_mem_ids), mem_ids_hash, subtask_q_hash,
                outcome_present, mem_cache_size, coverage_pct, outcome_success, outcome_failure,
            )

            sel = BCBSelection(
                subset=args.subset,
                split=args.split,
                train_ratio=float(args.train_ratio),
                seed=int(args.seed),
                split_file=args.split_file,
                data_path=args.data_path,
            )
            runner = AblationRunner(
                root=project_root,
                selection=sel,
                llm=llm,
                memory_service=memsvc,
                output_dir=str(abl_dir),
                model_name=cfg.llm.model,
                num_epochs=1,  # val-only mode, no training loop
                run_validation=True,  # always run val in ablation mode (overrides cfg.experiment.bcb_run_validation)
                temperature=(args.temperature if args.temperature is not None else cfg.llm.temperature),
                max_tokens=(args.max_tokens if args.max_tokens is not None else cfg.llm.max_tokens),
                retrieve_k=abl["k"],
                retrieve_threshold=args.retrieve_threshold,
                memory_budget_tokens=(
                    int(args.memory_budget_tokens)
                    if args.memory_budget_tokens is not None
                    else cfg.memory.memory_budget_tokens
                ),
                bcb_repo=args.bcb_repo,
                eval_timeout_s=float(args.eval_timeout),
                untrusted_hard_timeout_s=float(args.untrusted_hard_timeout),
                strip_think=args.strip_think,
                batch_size=int(cfg.experiment.batch_size),
                val_lambda_max=args.val_lambda_max,
                success_only=abl["success_only"],
                success_prerank=abl.get("success_prerank", False),
                w_q_override=abl["w_q"],
                lambda_max_override=abl["lambda_max"],
                tau_sq_override=abl["tau_sq"],
                sigma_sq_override=abl["sigma_sq"],
            )

            runner._problems = problems
            runner._train_ids = [tid for tid in problems if tid not in set(val_ids)]
            runner._val_ids = list(val_ids)

            # Fill-missing-only injection (don't overwrite ckpt-restored embeddings).
            if val_query_cache:
                if not hasattr(memsvc, "query_embeddings") or memsvc.query_embeddings is None:
                    memsvc.query_embeddings = {}
                qe = memsvc.query_embeddings
                added = 0
                for prompt, emb in val_query_cache.items():
                    if prompt not in qe:
                        qe[prompt] = emb
                        added += 1
                logger.info(
                    "Val query embeddings: %d already in memsvc cache, %d filled (total %d)",
                    len(val_query_cache) - added, added, len(qe),
                )

            rm_obj = getattr(memsvc, "region_manager", None)
            if rm_obj is not None and not rm_obj._subtask_embeddings:
                runner._register_subtask_embeddings()

            logger.info("Running val: %d tasks, k=%d, %s",
                        len(runner._val_ids), abl["k"], abl["desc"])
            val_res = runner._run_phase(
                epoch=1, phase="val", task_ids=runner._val_ids,
                epoch_dir=str(abl_dir), update_memory=False,
            )

            pass_pct = val_res.get("pass@1")
            if pass_pct is None and val_res["total"]:
                pass_pct = val_res["pass"] / val_res["total"]
            results_summary[abl_name] = {
                "desc": abl["desc"],
                "total": val_res["total"],
                "pass": val_res["pass"],
                "pass_at_1": pass_pct,
            }
            logger.info(
                "Ablation %s: pass=%d/%d (%.1f%%)",
                abl_name, val_res["pass"], val_res["total"],
                100.0 * (pass_pct or 0),
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    logger.info("=" * 60)
    logger.info("ABLATION SUMMARY")
    logger.info("=" * 60)
    for name, res in results_summary.items():
        logger.info("  %-22s: %d/%d = %.1f%%  (%s)",
                    name, res["pass"], res["total"],
                    100.0 * (res["pass_at_1"] or 0), res["desc"])
    summary_path = out_dir / "ablation_summary.json"
    with open(summary_path, "w") as f:
        _json.dump(results_summary, f, indent=2)
    logger.info("Summary saved to %s", summary_path)


def main() -> None:
    args = parse_args()
    setup_logging(project_root, "bcb_region")
    logger = logging.getLogger(__name__)

    # Global seed for reproducibility (best-effort).
    try:
        import random as _random
        _random.seed(int(args.seed))
        import numpy as _np
        _np.random.seed(int(args.seed))
    except Exception:
        pass
    try:
        import torch as _torch
        _torch.manual_seed(int(args.seed))
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(int(args.seed))
    except Exception:
        pass
    os.environ.setdefault("MEMRL_LLM_SEED", str(int(args.seed)))
    os.environ.setdefault("PYTHONHASHSEED", str(int(args.seed)))

    cfg = MempConfig.from_yaml(args.config)

    if args.split_file is None:
        default_split = DEFAULT_SPLIT_FILES.get(args.subset)
        if default_split is not None and default_split.exists():
            args.split_file = str(default_split)

    out_root = Path(args.output_dir or cfg.experiment.output_dir or "./results").resolve()
    out_dir = out_root / "bigcodebench_eval" / f"{args.split}_{args.subset}" / "region"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = (
        f"{time.strftime('%Y%m%d_%H%M%S')}_{cfg.llm.model.replace('/', '_')}"
        f"_region"
    )
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Providers
    token_log_dir = str((project_root / "logs" / "bcb_region").resolve())
    llm = OpenAILLM(
        api_key=cfg.llm.api_key,
        base_url=cfg.llm.base_url,
        model=cfg.llm.model,
        default_temperature=(
            args.temperature if args.temperature is not None else cfg.llm.temperature
        ),
        default_max_tokens=(args.max_tokens if args.max_tokens is not None else cfg.llm.max_tokens),
        token_log_dir=token_log_dir,
    )
    embedder = OpenAIEmbedder(
        api_key=cfg.embedding.api_key,
        base_url=cfg.embedding.base_url,
        model=cfg.embedding.model,
        max_text_len=getattr(cfg.embedding, "max_text_len", 8196),
    )

    # --- val_ablations mode: skip training, run ablation grid on a ckpt ---
    if args.val_ablations:
        run_val_ablations(args, cfg, llm, embedder, args.val_ablations, out_root)
        return

    # MemOS config JSON (shared with val_ablations path via _build_mos_config)
    temp_dir = tempfile.mkdtemp(prefix="memp_bcb_region_")
    user_id = f"bcb_region_{os.getpid()}"
    mos_config = _build_mos_config(
        cfg, temp_dir,
        top_k=(args.retrieve_k if args.retrieve_k is not None else cfg.memory.k_retrieve),
    )

    mos_config_path = os.path.join(temp_dir, "mos_config.json")
    with open(mos_config_path, "w", encoding="utf-8") as f:
        _json.dump(mos_config, f)

    # Create RegionManager (shared with val_ablations path via _build_region_manager)
    region_manager = _build_region_manager(args)

    # Initialize auto task cluster manager
    if args.task_cluster_k > 0:
        from memrl.configs.task_hierarchy import get_task_cluster_manager
        get_task_cluster_manager(K=args.task_cluster_k)
        logger.info("Task cluster K=%d (will be fitted at epoch 1)", args.task_cluster_k)

    # Create RegionMemoryService (extends MemoryService with region-aware retrieval)
    memsvc = RegionMemoryService(
        mos_config_path=mos_config_path,
        llm_provider=llm,
        embedding_provider=embedder,
        strategy_config=StrategyConfiguration(
            BuildStrategy(cfg.memory.build_strategy),
            RetrieveStrategy(cfg.memory.retrieve_strategy),
            UpdateStrategy(cfg.memory.update_strategy),
        ),
        user_id=user_id,
        num_workers=cfg.experiment.batch_size,
        max_keywords=cfg.memory.max_keywords,
        add_similarity_threshold=getattr(cfg.memory, "add_similarity_threshold", 0.9),
        enable_value_driven=cfg.experiment.enable_value_driven,
        rl_config=cfg.rl_config,
        db_max_concurrency=4,
        sim_norm_mean=getattr(cfg.memory, "sim_norm_mean", 0.1856827586889267),
        sim_norm_std=getattr(cfg.memory, "sim_norm_std", 0.09407906234264374),
        vector_dimension=getattr(cfg.embedding, "dimension", None),
        region_manager=region_manager,
        region_retrieve_mode=args.region_retrieve_mode,
        region_gating_mode=args.region_gating_mode,
        total_epochs=int(args.epochs),
        explore_schedule=args.explore_schedule,
        explore_success_ratio=args.explore_success_ratio,
    )

    if args.outcome_blend_beta is not None:
        memsvc.configure_outcome_blend(args.outcome_blend_beta)

    sel = BCBSelection(
        subset=args.subset,
        split=args.split,
        train_ratio=float(args.train_ratio),
        seed=int(args.seed),
        split_file=args.split_file,
        data_path=args.data_path,
    )

    runner = BCBRegionRunner(
        root=project_root,
        selection=sel,
        llm=llm,
        memory_service=memsvc,
        output_dir=str(run_dir),
        model_name=cfg.llm.model,
        num_epochs=int(args.epochs),
        run_validation=bool(getattr(cfg.experiment, "bcb_run_validation", False)),
        temperature=(
            args.temperature if args.temperature is not None else cfg.llm.temperature
        ),
        max_tokens=(
            args.max_tokens if args.max_tokens is not None else (cfg.llm.max_tokens or 1280)
        ),
        retrieve_k=(args.retrieve_k if args.retrieve_k is not None else cfg.memory.k_retrieve),
        retrieve_threshold=args.retrieve_threshold,
        memory_budget_tokens=(
            int(args.memory_budget_tokens)
            if args.memory_budget_tokens is not None
            else cfg.memory.memory_budget_tokens
        ),
        bcb_repo=args.bcb_repo,
        untrusted_hard_timeout_s=float(args.untrusted_hard_timeout),
        eval_timeout_s=float(args.eval_timeout),
        checkpoint_interval=int(args.checkpoint_interval),
        max_checkpoints=int(args.max_checkpoints),
        resume_checkpoint_path=args.resume_from,
        resume_epoch=args.resume_epoch,
        resume_step=args.resume_step,
        strip_think=args.strip_think,
        batch_size=int(cfg.experiment.batch_size),
        n_eval_runs=int(args.n_eval_runs),
        eval_temperature=args.eval_temperature,
        multi_eval_epochs=args.multi_eval_epochs,
        holdout_subtask=args.holdout_subtask,
        retrieval_mode=args.retrieval_mode,
        oracle_snapshot_dir=args.oracle_snapshot_dir,
        val_lambda_max=args.val_lambda_max,
        region_topology_cooldown_epochs=args.region_topology_cooldown_epochs,
        region_disable_mid_epoch_topology=args.region_disable_mid_epoch_topology,
        fixed_initial_topology_epoch=args.fixed_initial_topology_epoch,
    )

    logger.info("BCB Region run_dir: %s", run_dir)
    logger.info("Region config: min_cluster_size=%d, retrieve_mode=%s, gating_mode=%s, temperature=%.2f, shrinkage_top_n=%d, shrinkage_min_margin=%.3f, precluster_mode=%s, precluster_scale=%.3f, utility_mode=%s, topology_cooldown_epochs=%d, mid_epoch_topology=%s",
                region_manager.min_cluster_size, args.region_retrieve_mode, args.region_gating_mode,
                region_manager.temperature, region_manager.shrinkage_top_n,
                region_manager.shrinkage_min_utility_margin,
                region_manager.region_precluster_evidence_mode, region_manager.region_precluster_evidence_scale,
                region_manager.region_utility_mode,
                args.region_topology_cooldown_epochs,
                not args.region_disable_mid_epoch_topology)
    if args.holdout_subtask:
        logger.info("Holdout subtask: %s (excluded from train, evaluated per-epoch)", args.holdout_subtask)
    if args.shrinkage_lambda_max is not None:
        logger.info("Shrinkage override: lambda_max=%.2f, tau_sq=%s, sigma_sq=%s",
                    args.shrinkage_lambda_max,
                    args.shrinkage_tau_sq if args.shrinkage_tau_sq is not None else "default",
                    args.shrinkage_sigma_sq if args.shrinkage_sigma_sq is not None else "default")

    # Enable trajectory logging for Pilot B analysis
    trajectory_log_path = os.path.join(str(run_dir), "trajectory.jsonl")
    region_manager.enable_trajectory_logging(trajectory_log_path)
    logger.info("Trajectory logging enabled: %s", trajectory_log_path)

    # Region failure summary injection (if enabled via CLI)
    if args.failure_summary_n_slots > 0:
        runner.configure_failure_summary(
            n_slots=args.failure_summary_n_slots,
            replace_with_summary=not args.no_failure_summary_replace,
            lib_filter=args.failure_summary_lib_filter,
            contract_filter=args.failure_summary_contract_filter,
            min_compatible=args.failure_summary_min_compatible,
            fmmatch=args.failure_summary_fmmatch,
            fmmatch_topk=args.failure_summary_fmmatch_topk,
            fmmatch_backfill=args.failure_summary_fmmatch_backfill,
            force_recall=args.failure_summary_force_recall,
        )

    # Region experience cards injection (if enabled via CLI)
    if args.experience_cards_n_slots > 0:
        runner.configure_experience_cards(
            n_slots=args.experience_cards_n_slots,
        )

    # Retrieval gate (if enabled via CLI)
    if args.retrieval_gate_signal:
        runner.configure_retrieval_gate(
            signal=args.retrieval_gate_signal,
            threshold=args.retrieval_gate_threshold,
        )

    # Region meta header (if enabled via CLI)
    if args.region_meta_header:
        runner.configure_region_meta_header()

    # Meta-info mode (if enabled via CLI)
    if args.meta_info_mode:
        runner.configure_meta_info_mode()

    try:
        if args.eval_only:
            runner.run_eval_only()
        else:
            runner.run()
    finally:
        region_manager.close_trajectory_log()


if __name__ == "__main__":
    main()
