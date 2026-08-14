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
        description="Run BigCodeBench (BCB) multi-epoch memory benchmark"
    )
    p.add_argument(
        "--config",
        type=str,
        default=str(
            (project_root / "configs" / "rl_bcb_config.local.yaml")
            if (project_root / "configs" / "rl_bcb_config.local.yaml").exists()
            else (project_root / "configs" / "rl_bcb_config.yaml")
        ),
    )
    # Default to the full BigCodeBench set. Use `--subset hard` for the smaller subset.
    p.add_argument("--subset", type=str, default="full", choices=["hard", "full"])
    p.add_argument(
        "--split", type=str, default="instruct", choices=["instruct", "complete"]
    )
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--retrieve_threshold",
        type=float,
        default=None,
        help=(
            "BCB similarity threshold for MemoryService.retrieve(...). "
            "If omitted, falls back to rl_config.sim_threshold (or rl_config.tau)."
        ),
    )
    p.add_argument(
        "--memory_budget_tokens",
        type=int,
        default=None,
        help="Token budget for injected memory context (rough per-entry char budget).",
    )
    p.add_argument(
        "--split_file",
        type=str,
        default=None,
        help=(
            "Path to a JSON split file containing train_ids/val_ids. "
            "If omitted, uses legacy split files under configs/bigcodebench/splits/."
        ),
    )
    p.add_argument("--data_path", type=str, default=None)
    p.add_argument(
        "--bcb_repo",
        type=str,
        default=str(project_root / "3rdparty" / "bigcodebench-main"),
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override cfg.experiment.output_dir",
    )
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--max_tokens", type=int, default=None)
    p.add_argument("--retrieve_k", type=int, default=None)
    p.add_argument("--eval_timeout", type=float, default=240.0)
    p.add_argument("--untrusted_hard_timeout", type=float, default=300.0)
    p.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint snapshot dir to resume from (e.g., .../epoch1/snapshot/1 or .../epoch2/snapshot/step_350)",
    )
    p.add_argument(
        "--resume_epoch",
        type=int,
        default=None,
        help="Epoch to resume from (1-based). If --resume_step is also set, resumes mid-epoch.",
    )
    p.add_argument(
        "--resume_step",
        type=int,
        default=None,
        help="Step (sample index) to resume from within resume_epoch. Requires --resume_epoch.",
    )
    p.add_argument(
        "--checkpoint_interval",
        type=int,
        default=50,
        help="Save incremental checkpoint every N samples (default: 50)",
    )
    p.add_argument(
        "--max_checkpoints",
        type=int,
        default=3,
        help="Max mid-epoch checkpoints to keep per epoch (default: 3)",
    )
    p.add_argument(
        "--strip_think",
        action="store_true",
        default=False,
        help="Strip <think>...</think> blocks from LLM responses before extracting code (for reasoning models like DeepSeek-R1).",
    )
    p.add_argument(
        "--failure_summary_n_slots",
        type=int,
        default=0,
        help="Number of top-K slots reserved for failure memories with inline summary. "
             "0=disabled (default). Uses on-the-fly aggregation (no region required).",
    )
    p.add_argument(
        "--failure_summary_k",
        type=int,
        default=None,
        help="How many failure mems to aggregate for inline summary. None=all retrieved.",
    )
    # ---- Baseline flags (matching HLE/ALFWorld) ----
    p.add_argument(
        "--baseline_mode",
        type=str,
        default=None,
        choices=["passk", "reflection"],
        help="Baseline mode: 'passk' = k independent attempts, 'reflection' = k rounds with self-reflection.",
    )
    p.add_argument(
        "--baseline_k",
        type=int,
        default=10,
        help="Number of rounds for pass@k or reflection baseline (default: 10).",
    )
    p.add_argument(
        "--mem0",
        action="store_true",
        help="Use Mem0 baseline (mem0 library for memory management).",
    )
    p.add_argument(
        "--mem0_infer",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Mem0 infer mode: 'true' = LLM fact extraction, 'false' = raw text storage.",
    )
    p.add_argument(
        "--mem0_collection",
        type=str,
        default="memrl_bcb_mem0_baseline",
        help="Qdrant collection name for Mem0 baseline.",
    )
    p.add_argument(
        "--self_rag",
        action="store_true",
        help="Enable Self-RAG: LLM critique filters irrelevant memories before injection.",
    )
    p.add_argument(
        "--self_rag_inject_k",
        type=int,
        default=3,
        help="Number of memories to inject after Self-RAG critique filtering.",
    )
    p.add_argument(
        "--n_eval_runs",
        type=int,
        default=3,
        help="Number of eval runs per epoch for confidence interval (default: 3).",
    )
    p.add_argument(
        "--eval_temperature",
        type=float,
        default=0.2,
        help="Temperature for eval runs (non-zero for stochasticity). Default: 0.2.",
    )
    p.add_argument(
        "--multi_eval_epochs",
        type=str,
        default="last",
        help="Which epochs to run multi-eval: 'last' (default, only final epoch), "
             "'all' (every epoch), or comma-separated epoch numbers (e.g. '5,10').",
    )
    return p.parse_args()


def main() -> None:
    # Keep MemRL/MemOS imports out of module scope. Generated BigCodeBench
    # solutions may start multiprocessing children with the ``spawn`` method;
    # Python then re-executes this file as the child main module. Importing
    # MemoryService/MemOS during that bootstrap races MemOS logging/config
    # initialization and can fail with partially-initialized circular imports.
    # The main guard prevents this function from running in those children, so
    # lazy imports here make the entry point safe to re-execute.
    from memrl.configs.config import MempConfig
    from memrl.providers.llm import OpenAILLM
    from memrl.providers.embedding import OpenAIEmbedder
    from memrl.service.memory_service import MemoryService
    from memrl.service.strategies import (
        BuildStrategy,
        RetrieveStrategy,
        UpdateStrategy,
        StrategyConfiguration,
    )
    from memrl.run.bcb_runner import BCBRunner, BCBSelection

    args = parse_args()
    setup_logging(project_root, "bcb")
    logger = logging.getLogger(__name__)

    # Set global seeds for reproducibility (best-effort — vLLM still has some
    # non-determinism from BF16/FlashAttention; pair this with MEMRL_LLM_SEED).
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
    out_dir = out_root / "bigcodebench_eval" / f"{args.split}_{args.subset}" / "memory"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = (
        f"{time.strftime('%Y%m%d_%H%M%S')}_{cfg.llm.model.replace('/', '_')}"
        f"_rl-{'on' if cfg.experiment.enable_value_driven else 'off'}"
    )
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Providers
    token_log_dir = str((project_root / "logs" / "bcb").resolve())
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

    # MemOS config JSON for MemoryService (consistent with other runners).
    temp_dir = tempfile.mkdtemp(prefix="memp_bcb_run_")
    user_id = f"bcb_{os.getpid()}"
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

    if args.mem0:
        from memrl.service.mem0_memory_service import Mem0MemoryService
        mem0_qdrant_path = os.path.join(temp_dir, "mem0_qdrant")
        memsvc = Mem0MemoryService(
            llm_base_url=cfg.llm.base_url,
            llm_model=cfg.llm.model,
            llm_api_key=cfg.llm.api_key,
            embed_base_url=cfg.embedding.base_url,
            embed_model=cfg.embedding.model,
            embed_api_key=cfg.embedding.api_key,
            embedding_dims=int(getattr(cfg.embedding, "dimension", 0)) or None,
            qdrant_path=mem0_qdrant_path,
            collection_name=args.mem0_collection,
            top_k=int(args.retrieve_k if args.retrieve_k is not None else cfg.memory.k_retrieve),
            infer=(args.mem0_infer == "true"),
            user_id=user_id,
        )
        logger.info(
            "Using Mem0MemoryService (infer=%s, collection=%s)",
            args.mem0_infer, args.mem0_collection,
        )
    else:
        memsvc = MemoryService(
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
        )

    sel = BCBSelection(
        subset=args.subset,
        split=args.split,
        train_ratio=float(args.train_ratio),
        seed=int(args.seed),
        split_file=args.split_file,
        data_path=args.data_path,
    )

    runner = BCBRunner(
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
        batch_size=cfg.experiment.batch_size,
        baseline_mode=args.baseline_mode or getattr(cfg.experiment, "baseline_mode", None),
        baseline_k=args.baseline_k or getattr(cfg.experiment, "baseline_k", 10),
        self_rag=args.self_rag or getattr(cfg.experiment, "self_rag", False),
        self_rag_inject_k=args.self_rag_inject_k or getattr(cfg.experiment, "self_rag_inject_top_k", 3),
        n_eval_runs=int(args.n_eval_runs),
        eval_temperature=args.eval_temperature,
        multi_eval_epochs=args.multi_eval_epochs,
    )

    if args.failure_summary_n_slots > 0:
        runner.configure_failure_summary(
            n_slots=args.failure_summary_n_slots,
            inline_k=args.failure_summary_k,
        )

    if not args.resume_from:
        raise SystemExit("multi-eval-only requires --resume_from")
    logger.info("BCB multi-eval-only run_dir: %s", run_dir)
    from memrl.bigcodebench_eval.task_wrappers import load_bcb_data, split_dataset
    runner._problems = load_bcb_data(subset=runner.sel.subset, data_path=runner.sel.data_path)
    runner._train_ids, runner._val_ids = split_dataset(
        runner._problems, train_ratio=runner.sel.train_ratio, seed=runner.sel.seed,
        split_file=runner.sel.split_file,
    )
    runner._post_data_load_hook()
    logger.info("Loading completed checkpoint: %s", args.resume_from)
    runner.mem.load_checkpoint_snapshot(args.resume_from)
    runner._precompute_query_embeddings(runner._val_ids)
    eval_epoch = int(args.resume_epoch or args.epochs)
    epoch_dir = os.path.join(str(run_dir), f"epoch{eval_epoch}")
    os.makedirs(epoch_dir, exist_ok=True)
    val_res = runner._run_phase(
        epoch=eval_epoch, phase="val", task_ids=runner._val_ids,
        epoch_dir=epoch_dir, update_memory=False,
    )
    multi_res = runner._run_eval_multi(eval_epoch, epoch_dir)
    runner._save_json(os.path.join(str(run_dir), "summary.json"), {
        "eval_only": True, "source_checkpoint": args.resume_from,
        "epoch": eval_epoch, "val": val_res, "multi_eval": multi_res,
    })
    try:
        runner.writer.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
