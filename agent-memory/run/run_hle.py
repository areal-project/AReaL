import sys
import os
from pathlib import Path
import logging
import argparse
import json as _json
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from memrl.configs.config import MempConfig
from memrl.providers.llm import OpenAILLM
from memrl.providers.embedding import OpenAIEmbedder
from memrl.service.memory_service import MemoryService
from memrl.service.strategies import BuildStrategy, RetrieveStrategy, UpdateStrategy, StrategyConfiguration
from memrl.run.hle_runner import HLERunner, HLESelection


def setup_logging(project_root: Path, name: str):
    log_dir = project_root / "logs" / name
    log_dir.mkdir(parents=True, exist_ok=True)
    import time
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run HLE benchmark with memory-agent")
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
                   help="Base URL for judge LLM. Defaults to the solution LLM's base_url.")
    p.add_argument("--judge_api_key", type=str, default=None,
                   help="API key for judge LLM. Defaults to the solution LLM's api_key.")
    p.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=['Computer Science/AI', 'Math', 'Biology/Medicine', 'Physics', 'Chemistry', 'Engineering', 'Humanities/Social Science', 'Other'],
        help="Filter HLE rows to these categories (space-separated list).",
    )
    p.add_argument("--text_only", action="store_true",
                   help="Drop rows with images (for text-only LLMs like DeepSeek-V4-Pro).")
    p.add_argument(
        "--category_ratio",
        type=float,
        default=1.0,
        help="Per-category sampling ratio (0-1) after filtering categories.",
    )
    p.add_argument(
        "--eval_categories",
        type=str,
        nargs="+",
        default=None,
        help="Categories for evaluation (cross-category transfer). Train uses --categories, eval uses these.",
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
    p.add_argument(
        "--memory_filter_categories",
        type=str,
        nargs="+",
        default=None,
        help="Only retrieve memories tagged with these categories during eval.",
    )
    p.add_argument("--mem0", action="store_true", help="Use Mem0 baseline (mem0 library for memory management)")
    p.add_argument("--mem0_infer", type=str, default="true", choices=["true", "false"],
                   help="Mem0 infer mode: 'true' = LLM fact extraction, 'false' = raw text storage")
    p.add_argument("--mem0_collection", type=str, default="memrl_mem0_baseline",
                   help="Qdrant collection name for Mem0 baseline")
    p.add_argument("--self_rag", action="store_true",
                   help="Enable Self-RAG: LLM critique filters irrelevant memories before injection")
    p.add_argument("--self_rag_inject_k", type=int, default=3,
                   help="Number of memories to inject after Self-RAG critique filtering")
    return p.parse_args()


def main():
    logger = logging.getLogger(__name__)
    args = parse_args()
    try:
        cfg = MempConfig.from_yaml(args.config)
        setup_logging(project_root, cfg.experiment.experiment_name)

        out_dir = Path(cfg.experiment.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        run_id = os.environ.get("MEMRL_RUN_ID", "") or time.strftime('%Y%m%d-%H%M%S')
        log_dir = out_dir / "hle" / f"exp_{cfg.experiment.experiment_name}_{run_id}" / "local_cache"
        log_dir.mkdir(parents=True, exist_ok=True)

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
        # Optional separate judge LLM
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

        import tempfile
        temp_dir = tempfile.mkdtemp(prefix="memp_hle_run_")
        user_id = f"hle_{os.getpid()}"
        mos_config = {
            "chat_model": {
                "backend": "openai",
                "config": {
                    "model_name_or_path": cfg.llm.model,
                    "api_key": cfg.llm.api_key,
                    "api_base": cfg.llm.base_url
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
                top_k=cfg.memory.k_retrieve,
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
                add_similarity_threshold=getattr(cfg.memory, 'add_similarity_threshold', 0.9),
                enable_value_driven=cfg.experiment.enable_value_driven,
                rl_config=cfg.rl_config,
                db_max_concurrency=4,
                sim_norm_mean=getattr(cfg.memory, 'sim_norm_mean', 0.1856827586889267),
                sim_norm_std=getattr(cfg.memory, 'sim_norm_std', 0.09407906234264374),
                vector_dimension=int(getattr(cfg.embedding, 'dimension', 4096)),
            )

        # Load checkpoint if specified
        load_from_checkpoint = getattr(cfg.memory, "load_from_checkpoint", False)
        checkpoint_path = getattr(cfg.memory, "checkpoint_path", None)
        if load_from_checkpoint and checkpoint_path:
            logger.info(f"Loading memory checkpoint from: {checkpoint_path}")
            try:
                num_loaded = memsvc.load_checkpoint_snapshot(checkpoint_path)
                logger.info(f"Loaded {num_loaded} memories from checkpoint")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}", exc_info=True)
                raise

        # Resolve train/eval categories, translating --holdout_categories into the
        # zero-shot holdout split (train = complement, eval = held-out only).
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

        runner = HLERunner(
            name=cfg.experiment.experiment_name,
            llm=llm,
            llm_judge=llm_judge,
            selection=sel,
            output_dir=out_dir,
            memory_service=memsvc,
            run_id=run_id,
            temperature=(args.temperature if args.temperature is not None else cfg.llm.temperature),
            max_tokens=(args.max_tokens if args.max_tokens is not None else (cfg.llm.max_tokens or 4096)),
            retrieve_k=cfg.memory.k_retrieve,
            num_sections=cfg.experiment.num_sections,
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
            self_rag=args.self_rag or getattr(cfg.experiment, "self_rag", False),
            self_rag_inject_k=args.self_rag_inject_k or getattr(cfg.experiment, "self_rag_inject_top_k", 3),
            holdout_categories=args.holdout_categories,
        )
        runner.run()
    except Exception as e:
        logger.error(f"HLE run failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
