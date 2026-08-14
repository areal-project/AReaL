"""
Entry point for BCB Cross-Subtask Transfer (Holdout) Experiment.

Runs BCB with parallel holdout banks to validate cross-subtask transfer.
For each task, generates and evaluates with 8 memory banks:
  - full: normal MemRL (all subtask memories + Q)
  - holdout_X: excludes subtask X's memories, Q learned from other subtasks only

Usage:
    python run/run_bcb_holdout.py \
        --config configs/rl_bcb_config.deepseek_old_region_bs1.yaml \
        --split instruct --subset full --epochs 5 \
        --region_gating_mode additive \
        --output_dir ./results/holdout_transfer
"""

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
from memrl.service.holdout_memory_service import HoldoutRegionMemoryService
from memrl.service.holdout_region_manager import HoldoutRegionManager
from memrl.service.strategies import (
    BuildStrategy,
    RetrieveStrategy,
    UpdateStrategy,
    StrategyConfiguration,
)
from memrl.run.bcb_holdout_runner import BCBHoldoutRunner
from memrl.run.bcb_region_runner import BCBSelection
from memrl.configs.task_hierarchy import TASK_HIERARCHY

# BCB subtask categories
BCB_SUBTASKS = [
    "Computation", "General", "System", "Visualization",
    "Time", "Cryptography", "Network",
]

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
        description="Run BCB Cross-Subtask Transfer (Holdout) Experiment"
    )
    p.add_argument(
        "--config", type=str,
        default=str(project_root / "configs" / "rl_bcb_config.deepseek_old_region_bs1.yaml"),
    )
    p.add_argument("--subset", type=str, default="full", choices=["hard", "full"])
    p.add_argument("--split", type=str, default="instruct", choices=["instruct", "complete"])
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--retrieve_threshold", type=float, default=None)
    p.add_argument("--split_file", type=str, default=None)
    p.add_argument("--data_path", type=str, default=None)
    p.add_argument(
        "--bcb_repo", type=str,
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
    # Region-specific args
    p.add_argument("--region_gating_mode", type=str, default="additive",
                   choices=["multiplicative", "additive"])
    p.add_argument("--region_temperature", type=float, default=1.0)
    p.add_argument("--shrinkage_top_n", type=int, default=3)
    p.add_argument("--region_min_cluster_size", type=int, default=None)
    p.add_argument("--region_utility_mode", type=str, default="ema",
                   choices=["ema", "beta"])
    # Holdout-specific args
    p.add_argument("--holdout_subtasks", type=str, nargs="*", default=None,
                   help="Subtasks to hold out (default: all 7 BCB subtasks)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(project_root, "bcb_holdout")
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

    holdout_subtasks = args.holdout_subtasks or BCB_SUBTASKS

    out_root = Path(args.output_dir or cfg.experiment.output_dir or "./results").resolve()
    out_dir = out_root / "bigcodebench_eval" / f"{args.split}_{args.subset}" / "holdout"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = (
        f"{time.strftime('%Y%m%d_%H%M%S')}_{cfg.llm.model.replace('/', '_')}"
        f"_holdout"
    )
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Providers
    token_log_dir = str((project_root / "logs" / "bcb_holdout").resolve())
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

    # MemOS config JSON
    temp_dir = tempfile.mkdtemp(prefix="memp_bcb_holdout_")
    user_id = f"bcb_holdout_{os.getpid()}"
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
        "top_k": int(
            args.retrieve_k if args.retrieve_k is not None else cfg.memory.k_retrieve
        ),
    }

    mos_config_path = os.path.join(temp_dir, "mos_config.json")
    with open(mos_config_path, "w", encoding="utf-8") as f:
        _json.dump(mos_config, f)

    # Create HoldoutRegionManager
    region_min_cs = args.region_min_cluster_size or max(3, 10 // 2)
    holdout_manager = HoldoutRegionManager(
        holdout_subtasks=holdout_subtasks,
        task_hierarchy=TASK_HIERARCHY,
        min_cluster_size=region_min_cs,
        temperature=args.region_temperature,
        shrinkage_top_n=args.shrinkage_top_n,
        region_utility_mode=args.region_utility_mode,
    )

    # Create HoldoutRegionMemoryService
    memsvc = HoldoutRegionMemoryService(
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
        region_manager=holdout_manager,
        region_gating_mode=args.region_gating_mode,
    )

    sel = BCBSelection(
        subset=args.subset,
        split=args.split,
        train_ratio=float(args.train_ratio),
        seed=int(args.seed),
        split_file=args.split_file,
        data_path=args.data_path,
    )

    runner = BCBHoldoutRunner(
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
            int(args.memory_budget_tokens) if hasattr(args, 'memory_budget_tokens') and args.memory_budget_tokens is not None
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
    )

    logger.info("BCB Holdout Transfer Experiment")
    logger.info("  run_dir: %s", run_dir)
    logger.info("  holdout_subtasks: %s", holdout_subtasks)
    logger.info("  banks: %s", holdout_manager.bank_names)
    logger.info("  region_gating_mode: %s", args.region_gating_mode)
    logger.info("  region_utility_mode: %s", args.region_utility_mode)
    logger.info("  epochs: %d", args.epochs)
    logger.info("  batch_size: %d", cfg.experiment.batch_size)

    # Enable trajectory logging
    trajectory_log_path = os.path.join(str(run_dir), "trajectory.jsonl")
    holdout_manager.enable_trajectory_logging(trajectory_log_path)

    try:
        runner.run()
    finally:
        holdout_manager.close_trajectory_log()


if __name__ == "__main__":
    main()
