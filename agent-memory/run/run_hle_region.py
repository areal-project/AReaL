"""Entry script for HLE benchmark with Region-based Memory Transfer."""
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
from memrl.run.hle_region_runner import HLERegionRunner
from memrl.run.hle_runner import HLESelection
from memrl.configs.task_hierarchy import TASK_HIERARCHY


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
        description="Run HLE benchmark with Region-based Memory Transfer"
    )
    # --- Base HLE args ---
    p.add_argument(
        "--config",
        type=str,
        default=str(
            (project_root / "configs" / "rl_hle_config.local.yaml")
            if (project_root / "configs" / "rl_hle_config.local.yaml").exists()
            else (project_root / "configs" / "rl_hle_config.yaml")
        ),
    )
    p.add_argument("--train", type=str)
    p.add_argument("--num_valid", type=int, default=0)
    p.add_argument("--num_train", type=int, default=0)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--max_tokens", type=int, default=None)
    p.add_argument("--judge_model", type=str, default='gpt-4o-2024-11-20')
    p.add_argument("--judge_base_url", type=str, default=None,
                   help="Separate base_url for the judge (e.g. MatrixLLM); defaults to cfg.llm.base_url.")
    p.add_argument("--judge_api_key", type=str, default=None,
                   help="Separate api_key for the judge; defaults to cfg.llm.api_key.")
    p.add_argument("--text_only", action="store_true",
                   help="Drop image (multimodal) HLE questions, keeping text-only rows.")
    p.add_argument("--retrieve_k", type=int, default=None)
    p.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=['Computer Science/AI', 'Math', 'Biology/Medicine', 'Physics',
                 'Chemistry', 'Engineering', 'Humanities/Social Science', 'Other'],
    )
    p.add_argument("--category_ratio", type=float, default=1.0)
    p.add_argument(
        "--eval_categories",
        type=str,
        nargs="+",
        default=None,
    )
    p.add_argument(
        "--memory_filter_categories",
        type=str,
        nargs="+",
        default=None,
    )
    p.add_argument(
        "--holdout_categories",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Zero-shot holdout: exclude these categories from train and evaluate ONLY on them. "
            "Train = complement of the held-out set. Overrides --categories/--eval_categories. "
            "Valid names: 'Math', 'Physics', 'Chemistry', 'Biology/Medicine', "
            "'Computer Science/AI', 'Engineering', 'Humanities/Social Science', 'Other'."
        ),
    )

    # --- Region-specific args ---
    p.add_argument("--k_global", type=int, default=30)
    p.add_argument("--k_local", type=int, default=10)
    p.add_argument("--region_retrieve_mode", type=str, default="global",
                   choices=["global", "quota_fixed", "quota_adaptive", "utility_anchor"])
    p.add_argument("--region_gating_mode", type=str, default="multiplicative",
                   choices=["multiplicative", "additive"])
    p.add_argument("--region_temperature", type=float, default=1.0)
    p.add_argument("--shrinkage_top_n", type=int, default=3)
    p.add_argument("--region_min_cluster_size", type=int, default=None)
    p.add_argument("--region_utility_mode", type=str, default="ema",
                   choices=["ema", "beta"])
    p.add_argument("--region_smoothing_C", type=float, default=0.5)
    p.add_argument("--shrinkage_lambda_max", type=float, default=None)
    p.add_argument("--val_lambda_max", type=float, default=None)
    p.add_argument("--shrinkage_confidence_k", type=float, default=None)
    p.add_argument("--shrinkage_tau_sq", type=float, default=None)
    p.add_argument("--shrinkage_sigma_sq", type=float, default=None)
    p.add_argument("--outcome_blend_beta", type=float, default=None)
    p.add_argument("--propagation_eta", type=float, default=0.03)
    p.add_argument("--propagation_k", type=int, default=30)
    p.add_argument("--propagation_sim_min", type=float, default=0.40)
    p.add_argument("--no_propagation", action="store_true")
    p.add_argument("--explore_schedule", type=str, default="0,4,3,2,2,1,1,1,1,0")
    p.add_argument("--explore_success_ratio", type=float, default=0.7)
    p.add_argument("--region_cluster_init_step", type=int, default=500)
    p.add_argument("--region_merge_interval", type=int, default=400)
    p.add_argument(
        "--region_split_evidence_migration_mode",
        type=str,
        default="soft_source_conserving",
        choices=["soft_source_conserving", "hard_member_rebase"],
        help="How Beta evidence is migrated through Region split/merge topology changes.",
    )
    p.add_argument("--failure_summary_n_slots", type=int, default=0,
                   help="Reserve N top-k slots for region failure summaries (0 = disabled).")
    p.add_argument("--failure_summary_path", type=str, default=None,
                   help="Optional JSON of prebuilt region failure summaries (global fallback).")

    return p.parse_args()


def _build_region_manager(args):
    region_min_cs = args.region_min_cluster_size or max(3, args.k_local // 2)
    rm = RegionManager(
        task_hierarchy=TASK_HIERARCHY,
        min_cluster_size=region_min_cs,
        temperature=args.region_temperature,
        shrinkage_top_n=args.shrinkage_top_n,
        region_utility_mode=args.region_utility_mode,
        bayesian_smoothing_C=args.region_smoothing_C,
        propagation_enabled=not args.no_propagation,
        propagation_eta=args.propagation_eta,
        propagation_k=args.propagation_k,
        propagation_sim_min=args.propagation_sim_min,
        region_split_evidence_migration_mode=args.region_split_evidence_migration_mode,
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


def main() -> None:
    args = parse_args()
    cfg = MempConfig.from_yaml(args.config)
    setup_logging(project_root, f"hle_region_{cfg.experiment.experiment_name}")
    logger = logging.getLogger(__name__)

    # Seed
    try:
        import random as _random
        _random.seed(getattr(cfg.experiment, "random_seed", 42) or 42)
        import numpy as _np
        _np.random.seed(getattr(cfg.experiment, "random_seed", 42) or 42)
    except Exception:
        pass

    out_dir = Path(cfg.experiment.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = os.environ.get("MEMRL_RUN_ID", "") or time.strftime('%Y%m%d-%H%M%S')
    log_dir = out_dir / "hle" / f"exp_{cfg.experiment.experiment_name}_region_{run_id}" / "local_cache"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Providers
    llm = OpenAILLM(
        api_key=cfg.llm.api_key,
        base_url=cfg.llm.base_url,
        model=cfg.llm.model,
        default_temperature=cfg.llm.temperature,
        default_max_tokens=cfg.llm.max_tokens,
        token_log_dir=str(log_dir),
    )
    embedder = OpenAIEmbedder(
        api_key=cfg.embedding.api_key,
        base_url=cfg.embedding.base_url,
        model=cfg.embedding.model,
        max_text_len=getattr(cfg.embedding, "max_text_len", 4096),
        token_log_dir=str(log_dir),
    )
    llm_judge = None
    if args.judge_model:
        llm_judge = OpenAILLM(
            api_key=(args.judge_api_key or cfg.llm.api_key),
            base_url=(args.judge_base_url or cfg.llm.base_url),
            model=args.judge_model,
            default_temperature=0.0,
            default_max_tokens=4096,
            token_log_dir=str(log_dir),
        )

    # MemOS config
    temp_dir = tempfile.mkdtemp(prefix="memp_hle_region_")
    user_id = f"hle_region_{os.getpid()}"
    retrieve_k = args.retrieve_k if args.retrieve_k is not None else cfg.memory.k_retrieve
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
        "top_k": int(retrieve_k),
    }
    mos_config_path = os.path.join(temp_dir, "mos_config.json")
    with open(mos_config_path, "w", encoding="utf-8") as f:
        _json.dump(mos_config, f)

    # RegionManager
    region_manager = _build_region_manager(args)

    # RegionMemoryService
    num_sections = cfg.experiment.num_sections
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
        vector_dimension=int(getattr(cfg.embedding, 'dimension', 4096)),
        add_similarity_threshold=getattr(cfg.memory, "add_similarity_threshold", 0.9),
        enable_value_driven=cfg.experiment.enable_value_driven,
        rl_config=cfg.rl_config,
        db_max_concurrency=4,
        sim_norm_mean=getattr(cfg.memory, "sim_norm_mean", 0.1856827586889267),
        sim_norm_std=getattr(cfg.memory, "sim_norm_std", 0.09407906234264374),
        region_manager=region_manager,
        region_retrieve_mode=args.region_retrieve_mode,
        region_gating_mode=args.region_gating_mode,
        total_epochs=int(num_sections),
        explore_schedule=args.explore_schedule,
        explore_success_ratio=args.explore_success_ratio,
    )

    if args.outcome_blend_beta is not None:
        memsvc.configure_outcome_blend(args.outcome_blend_beta)

    # Load checkpoint if specified
    load_from_checkpoint = getattr(cfg.memory, "load_from_checkpoint", False)
    checkpoint_path = getattr(cfg.memory, "checkpoint_path", None)
    if load_from_checkpoint and checkpoint_path:
        logger.info("Loading memory checkpoint from: %s", checkpoint_path)
        try:
            num_loaded = memsvc.load_checkpoint_snapshot(checkpoint_path)
            logger.info("Loaded %s memories from checkpoint", num_loaded)
        except Exception as e:
            logger.error("Failed to load checkpoint: %s", e, exc_info=True)
            raise

    # Resolve train/eval categories, translating --holdout_categories into the
    # zero-shot holdout split (train = complement, eval = held-out only). This
    # script defaults --categories to the full 8-list, so forcing categories=None
    # is what prevents the held-out category from leaking into training.
    train_categories = args.categories or getattr(cfg.experiment, "hle_categories", None)
    eval_categories = args.eval_categories
    if args.holdout_categories:
        from memrl.configs.task_hierarchy import HLE_CATEGORIES
        unknown = [c for c in args.holdout_categories if c not in HLE_CATEGORIES]
        if unknown:
            raise ValueError(
                f"Unknown holdout category(s): {unknown}. Valid HLE categories: {HLE_CATEGORIES}"
            )
        train_categories = None  # forces _load complement branch (train excludes held-out)
        eval_categories = list(args.holdout_categories)
        logger.info(
            "[HOLDOUT] Zero-shot holdout mode: holding out %s, train = complement (%d categories)",
            args.holdout_categories, len(HLE_CATEGORIES) - len(args.holdout_categories),
        )

    sel = HLESelection(
        train_path=args.train or getattr(cfg.experiment, 'split_file', None),
        num_valid=(args.num_valid if args.num_valid and args.num_valid > 0 else None),
        num_train=(args.num_train if args.num_train and args.num_train > 0 else None),
        categories=train_categories,
        eval_categories=eval_categories,
        category_ratio=args.category_ratio if args.category_ratio is not None else getattr(cfg.experiment, "hle_category_ratio", None),
        text_only=args.text_only,
    )

    runner = HLERegionRunner(
        name=cfg.experiment.experiment_name,
        llm=llm,
        llm_judge=llm_judge,
        selection=sel,
        output_dir=out_dir,
        memory_service=memsvc,
        run_id=run_id,
        temperature=(args.temperature if args.temperature is not None else cfg.llm.temperature),
        max_tokens=(args.max_tokens if args.max_tokens is not None else (cfg.llm.max_tokens or 4096)),
        retrieve_k=retrieve_k,
        num_sections=num_sections,
        batch_size=cfg.experiment.batch_size,
        dataset_ratio=getattr(cfg.experiment, "dataset_ratio", 1.0),
        random_seed=getattr(cfg.experiment, "random_seed", 42) or 42,
        train_valid_split=getattr(cfg.experiment, "train_valid_split", 0.8),
        ckpt_eval_enabled=getattr(cfg.experiment, "ckpt_eval_enabled", False),
        ckpt_eval_path=getattr(cfg.experiment, "ckpt_eval_path", None),
        ckpt_resume_enabled=getattr(cfg.experiment, "ckpt_resume_enabled", False),
        ckpt_resume_path=getattr(cfg.experiment, "ckpt_resume_path", None),
        ckpt_resume_epoch=getattr(cfg.experiment, "ckpt_resume_epoch", None),
        ckpt_resume_prefer_current_run=getattr(cfg.experiment, "ckpt_resume_prefer_current_run", False),
        ckpt_save_every_n_batches=getattr(cfg.experiment, "ckpt_save_every_n_batches", 1),
        ckpt_max_keep=getattr(cfg.experiment, "ckpt_max_keep", 3),
        baseline_mode=getattr(cfg.experiment, "baseline_mode", False),
        baseline_k=getattr(cfg.experiment, "baseline_k", 0),
        mode=getattr(cfg.experiment, "mode", "train"),
        memory_filter_categories=args.memory_filter_categories,
        holdout_categories=args.holdout_categories,
        # Region-specific
        region_cluster_init_step=args.region_cluster_init_step,
        region_merge_interval=args.region_merge_interval,
        val_lambda_max=args.val_lambda_max,
    )

    logger.info("HLE Region run: %s", out_dir)
    logger.info(
        "Region config: min_cluster_size=%d, retrieve_mode=%s, gating_mode=%s, "
        "temperature=%.2f, shrinkage_top_n=%d, utility_mode=%s",
        region_manager.min_cluster_size, args.region_retrieve_mode, args.region_gating_mode,
        region_manager.temperature, region_manager.shrinkage_top_n, region_manager.region_utility_mode,
    )

    # Enable trajectory logging
    trajectory_log_path = str(log_dir / "trajectory.jsonl")
    region_manager.enable_trajectory_logging(trajectory_log_path)
    logger.info("Trajectory logging enabled: %s", trajectory_log_path)

    if args.failure_summary_n_slots > 0:
        runner.configure_failure_summary(
            n_slots=args.failure_summary_n_slots,
            summaries_path=args.failure_summary_path,
            replace_with_summary=True,
        )

    try:
        runner.run()
    finally:
        region_manager.close_trajectory_log()


if __name__ == "__main__":
    main()
