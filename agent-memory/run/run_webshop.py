#!/usr/bin/env python3
"""WebShop Runner Entry Point for MemRL (mirrors run_alfworld.py structure)."""
import argparse
import json
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
from memrl.service.memory_service import MemoryService
from memrl.service.strategies import BuildStrategy, RetrieveStrategy, UpdateStrategy, StrategyConfiguration
from memrl.service.value_driven import RLConfig
from memrl.agent.memp_agent import MempAgent
from memrl.run.webshop_rl_runner import WebShopRunner

logger = logging.getLogger(__name__)


def setup_logging(exp_name: str):
    log_dir = project_root / "logs" / "webshop"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{exp_name}_{time.strftime('%Y%m%d-%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ]
    )
    logger.info("Log file: %s", log_file)


def parse_args():
    parser = argparse.ArgumentParser(description="WebShop RL Runner")
    parser.add_argument("--config", type=str, required=True)

    # Data split paths
    parser.add_argument("--split_info_path", type=str, default=None,
                        help="Path to split_info.json (train/val/ood goals)")
    parser.add_argument("--indist_file_path", type=str, default=None,
                        help="Path to IN-DIST product file (items_shuffle_indist.json)")
    parser.add_argument("--ood_file_path", type=str, default=None,
                        help="Path to OOD product file (items_shuffle_ood_new.json)")

    # Region flags
    parser.add_argument("--region", action="store_true", help="Enable RegionMemoryService")
    parser.add_argument("--region_gating_mode", type=str, default="additive", choices=["additive", "multiplicative"])
    parser.add_argument("--task_cluster_k", type=int, default=8, help="Number of task clusters for region")

    # Shrinkage / region params
    parser.add_argument("--shrinkage_confidence_k", type=float, default=3.0)
    parser.add_argument("--propagation_eta", type=float, default=0.12)
    parser.add_argument("--val_lambda_max", type=float, default=None)
    parser.add_argument("--no_z_norm", action="store_true")

    # Failure summary
    parser.add_argument("--failure_summary_n_slots", type=int, default=0)
    parser.add_argument("--failure_summary_path", type=str, default=None)

    # Explore schedule
    parser.add_argument("--explore_schedule", type=str, default=None,
                        help="Comma-separated per-section explore quotas")

    # Eval control
    parser.add_argument("--skip_initial_eval", action="store_true")
    parser.add_argument("--no_memory", action="store_true", help="Run without memory (pure LLM baseline)")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Override experiment_name from config (for distinct output dirs per phase)")

    # Resume control
    parser.add_argument("--ckpt_resume_enabled", action="store_true",
                        help="Resume training from a checkpoint snapshot")
    parser.add_argument("--ckpt_resume_path", type=str, default=None,
                        help="Path to experiment dir or snapshot dir to resume from")
    parser.add_argument("--ckpt_resume_epoch", type=int, default=None,
                        help="Specific section/epoch to resume from (auto-detect latest if omitted)")

    # Parallelism
    parser.add_argument("--episode_workers", type=int, default=8,
                        help="Number of parallel worker threads for episode execution within a batch")

    # Smoke-test eval truncation: limit val/ood to first N sessions for fast feedback.
    parser.add_argument("--smoke_eval_n", type=int, default=None,
                        help="If set, truncate val/ood eval splits to the first N sessions. Used by sbatch_webshop_smoke.sh.")

    return parser.parse_args()


def main():
    args = parse_args()
    cfg = MempConfig.from_yaml(args.config)

    # Override experiment_name if provided (lets serial phases write to distinct dirs)
    if args.exp_name:
        cfg.experiment.experiment_name = args.exp_name

    setup_logging(cfg.experiment.experiment_name)
    logger.info("Config: %s", args.config)
    logger.info("Region: %s, failure_summary_n_slots: %d", args.region, args.failure_summary_n_slots)

    # Resolve data paths (CLI args override config)
    indist_file_path = args.indist_file_path or getattr(cfg.experiment, 'file_path', None)
    ood_file_path = args.ood_file_path or getattr(cfg.experiment, 'ood_file_path', None)
    split_info_path = args.split_info_path or getattr(cfg.experiment, 'split_info_path', None)

    if not split_info_path:
        raise ValueError("--split_info_path (or experiment.split_info_path in config) is required")
    if not indist_file_path:
        raise ValueError("--indist_file_path (or experiment.file_path in config) is required")
    if not ood_file_path:
        raise ValueError("--ood_file_path (or experiment.ood_file_path in config) is required")

    # Make paths absolute relative to project root
    split_info_path = str(project_root / split_info_path) if not os.path.isabs(split_info_path) else split_info_path
    indist_file_path = str(project_root / indist_file_path) if not os.path.isabs(indist_file_path) else indist_file_path
    ood_file_path = str(project_root / ood_file_path) if not os.path.isabs(ood_file_path) else ood_file_path

    logger.info("Split info: %s", split_info_path)
    logger.info("IN-DIST products: %s", indist_file_path)
    logger.info("OOD products: %s", ood_file_path)

    # Output dir
    out_dir = Path(cfg.experiment.output_dir)
    run_id = time.strftime('%Y%m%d-%H%M%S')
    ck_dir = out_dir / "webshop" / f"exp_{cfg.experiment.experiment_name}_{run_id}" / "local_cache"
    ck_dir.mkdir(parents=True, exist_ok=True)

    # Temp dir for mos_config
    temp_dir = tempfile.mkdtemp(prefix="memp_webshop_run_")

    # LLM
    llm = OpenAILLM(
        api_key=cfg.llm.api_key,
        base_url=cfg.llm.base_url,
        model=cfg.llm.model,
        default_max_tokens=cfg.llm.max_tokens,
        default_temperature=getattr(cfg.llm, 'temperature', 0.0),
    )

    # Embedding
    embedder = OpenAIEmbedder(
        api_key=cfg.embedding.api_key,
        base_url=cfg.embedding.base_url,
        model=cfg.embedding.model,
    )

    # MOS config (same pattern as run_alfworld.py)
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
    with open(mos_config_path, "w") as f:
        json.dump(mos_config, f)

    # Strategy config
    build_strategy = BuildStrategy(cfg.memory.build_strategy)
    retrieve_strategy = RetrieveStrategy(cfg.memory.retrieve_strategy)
    update_strategy = UpdateStrategy(cfg.memory.update_strategy)

    # RL config
    rl_config = cfg.rl_config
    user_id = "webshop_user"

    # Memory service
    if args.no_memory:
        memory_service = None
        logger.info("Running in NO-MEMORY mode (pure LLM baseline)")
    elif args.region:
        from memrl.service.region_memory_service import RegionMemoryService, RegionManager

        explore_schedule = None
        if args.explore_schedule:
            explore_schedule = [int(x) for x in args.explore_schedule.split(",")]

        region_manager = RegionManager(
            shrinkage_confidence_k=args.shrinkage_confidence_k,
            propagation_eta=args.propagation_eta,
        )

        memory_service = RegionMemoryService(
            mos_config_path=mos_config_path,
            llm_provider=llm,
            embedding_provider=embedder,
            strategy_config=StrategyConfiguration(build_strategy, retrieve_strategy, update_strategy),
            user_id=user_id,
            num_workers=cfg.experiment.batch_size,
            max_keywords=cfg.memory.max_keywords,
            add_similarity_threshold=getattr(cfg.memory, "add_similarity_threshold", 0.9),
            enable_value_driven=True,
            rl_config=rl_config,
            db_max_concurrency=4,
            sim_norm_mean=getattr(cfg.memory, "sim_norm_mean", None),
            sim_norm_std=getattr(cfg.memory, "sim_norm_std", None),
            query_embedding_cache_enabled=False,
            use_z_score_normalization=(not args.no_z_norm),
            vector_dimension=cfg.embedding.dimension,
            region_manager=region_manager,
            region_gating_mode=args.region_gating_mode,
            explore_schedule=explore_schedule,
        )
        logger.info("Using RegionMemoryService (gating=%s, shrinkage_k=%.1f, eta=%.2f)",
                    args.region_gating_mode, args.shrinkage_confidence_k, args.propagation_eta)
    else:
        memory_service = MemoryService(
            mos_config_path=mos_config_path,
            llm_provider=llm,
            embedding_provider=embedder,
            strategy_config=StrategyConfiguration(build_strategy, retrieve_strategy, update_strategy),
            user_id=user_id,
            num_workers=cfg.experiment.batch_size,
            max_keywords=cfg.memory.max_keywords,
            add_similarity_threshold=getattr(cfg.memory, "add_similarity_threshold", 0.9),
            enable_value_driven=True,
            rl_config=rl_config,
            db_max_concurrency=4,
            sim_norm_mean=getattr(cfg.memory, "sim_norm_mean", None),
            sim_norm_std=getattr(cfg.memory, "sim_norm_std", None),
            query_embedding_cache_enabled=False,
            use_z_score_normalization=(not args.no_z_norm),
            vector_dimension=cfg.embedding.dimension,
        )
        logger.info("Using base MemoryService (no region)")

    # Agent — load WebShop few-shot demos if configured.
    few_shot_examples = []
    fs_path = getattr(cfg.experiment, "few_shot_path", None)
    if fs_path:
        fs_full = fs_path if os.path.isabs(fs_path) else str(project_root / fs_path)
        try:
            with open(fs_full, "r", encoding="utf-8") as f:
                few_shot_examples = json.load(f)
            logger.info("Loaded %d webshop few-shot examples from %s",
                        len(few_shot_examples), fs_full)
        except Exception as e:
            logger.warning("Failed to load few_shot_path=%s: %s", fs_full, e)
            few_shot_examples = []
    agent = MempAgent(llm_provider=llm, few_shot_examples=few_shot_examples)

    # Runner
    runner = WebShopRunner(
        agent=agent,
        root=project_root,
        memory_service=memory_service,
        exp_name=cfg.experiment.experiment_name,
        num_sections=cfg.experiment.num_sections,
        batch_size=cfg.experiment.batch_size,
        max_steps=cfg.experiment.max_steps,
        rl_config=rl_config,
        ck_dir=str(ck_dir),
        retrieve_k=cfg.memory.k_retrieve,
        mode=cfg.experiment.mode,
        valid_interval=getattr(cfg.experiment, 'valid_interval', 1),
        test_interval=getattr(cfg.experiment, 'test_interval', 1),
        dataset_ratio=getattr(cfg.experiment, 'dataset_ratio', 1.0),
        random_seed=cfg.experiment.random_seed,
        file_path=indist_file_path,
        ood_file_path=ood_file_path,
        split_info_path=split_info_path,
        human_goals=True,
        skip_initial_eval=args.skip_initial_eval,
        val_lambda_max=args.val_lambda_max,
        task_cluster_k=args.task_cluster_k,
        ckpt_resume_enabled=args.ckpt_resume_enabled,
        ckpt_resume_path=args.ckpt_resume_path,
        ckpt_resume_epoch=args.ckpt_resume_epoch,
        episode_workers=args.episode_workers,
    )

    # Configure failure summary
    if args.failure_summary_n_slots > 0:
        runner.configure_failure_summary(
            n_slots=args.failure_summary_n_slots,
            summaries_path=args.failure_summary_path,
        )

    # Smoke-test mode: truncate eval splits to fast-iterate the prompt path.
    if args.smoke_eval_n is not None and args.smoke_eval_n > 0:
        n = args.smoke_eval_n
        runner.valid_sessions = runner.valid_sessions[:n]
        runner.valid_goals = runner.valid_goals[:n]
        runner.test_sessions = runner.test_sessions[:n]
        runner.test_goals = runner.test_goals[:n]
        logger.info("[smoke] Truncated eval splits to first %d sessions each", n)

    # Run
    runner.run()


if __name__ == "__main__":
    main()
