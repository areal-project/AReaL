# scripts/run_agent_experiment.py
import sys
import os
from pathlib import Path
import logging
import tempfile
import shutil
import json
import argparse

# --- Setup Project Path ---
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# --- Setup LLB Path ---
# 确保 LLB 在 sys.path 中，以便导入其组件
LLB_ROOT = project_root / "3rdparty" / "LifelongAgentBench"
if not LLB_ROOT.exists():
    raise RuntimeError(f"LLB directory not found: {LLB_ROOT}")

# Python 3.10 兼容：为 enum.StrEnum 提供兜底实现
try:
    import enum as _enum

    if not hasattr(_enum, "StrEnum"):

        class _StrEnum(str, _enum.Enum):
            pass

        _enum.StrEnum = _StrEnum  # type: ignore[attr-defined]
    import typing as _typing

    if not hasattr(_typing, "reveal_type"):

        def _noop_reveal_type(x):
            return x

        _typing.reveal_type = _noop_reveal_type  # type: ignore[attr-defined]
    if not hasattr(_typing, "Self"):
        _typing.Self = object  # type: ignore[attr-defined]
except Exception:
    pass

if str(LLB_ROOT) not in sys.path:
    sys.path.insert(0, str(LLB_ROOT))

# --- Import all our components ---
from memrl.configs.config import MempConfig
from memrl.service.memory_service import MemoryService
from memrl.service.strategies import (
    BuildStrategy,
    RetrieveStrategy,
    UpdateStrategy,
    StrategyConfiguration,
)
from memrl.providers.llm import OpenAILLM
from memrl.providers.embedding import OpenAIEmbedder
from memrl.run.llb_rl_runner import LLBRunner
from memrl.trace.llb_jsonl import apply_trace_env_from_experiment_config


# (The setup_logging function remains the same)
def setup_logging(project_root: Path, name: str):
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    import time

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
    logging.info(f"Logging configured. Log file: {log_filepath}")


logger = logging.getLogger(__name__)

def _default_llb_config_path(project_root: Path) -> Path:
    """Prefer a gitignored local config when present."""
    local_path = project_root / "configs" / "rl_llb_config.local.yaml"
    if local_path.exists():
        return local_path
    return project_root / "configs" / "rl_llb_config.yaml"


def parse_args(project_root: Path) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run LifelongAgentBench (LLB) with MemRL")
    p.add_argument(
        "--config",
        type=str,
        default=str(_default_llb_config_path(project_root)),
        help=(
            "Path to YAML config. If omitted, prefers configs/rl_llb_config.local.yaml "
            "when it exists, otherwise uses configs/rl_llb_config.yaml."
        ),
    )
    p.add_argument(
        "--failure_summary_n_slots",
        type=int,
        default=0,
        help="Number of top-k slots reserved for failure summary (0=disabled). "
             "Uses on-the-fly aggregation (no region required).",
    )
    p.add_argument(
        "--failure_summary_k",
        type=int,
        default=None,
        help="How many failure memories to aggregate into summary. None=use all retrieved.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output/checkpoint directory (ck_dir).",
    )
    p.add_argument(
        "--region", action="store_true",
        help="Enable region-aware memory service (auto_cluster on task embeddings).",
    )
    p.add_argument(
        "--region_k", type=int, default=8,
        help="Number of clusters for auto_cluster subtask definition (default 8).",
    )
    p.add_argument(
        "--region_gating_mode", type=str, default="multiplicative",
        choices=["multiplicative", "additive"],
        help="Region gating mode for retrieval scoring.",
    )
    p.add_argument(
        "--no_z_norm", action="store_true",
        help="Disable z-score normalization on sim/q during retrieval scoring.",
    )
    p.add_argument(
        "--explore_schedule", type=str, default="",
        help="Comma-separated per-epoch exploration injection count (e.g. '0,2,2,1,1,1,1,0,0,0').",
    )
    p.add_argument(
        "--shrinkage_confidence_k", type=float, default=None,
        help="Confidence-gated shrinkage k (replaces James-Stein tau/sigma). E.g. 3.0.",
    )
    p.add_argument(
        "--propagation_eta", type=float, default=0.03,
        help="Q propagation learning rate (default 0.03).",
    )
    p.add_argument(
        "--baseline_mode", type=str, default=None,
        choices=["passk", "reflection"],
        help="Run a baseline instead of MemRL training (passk or reflection).",
    )
    p.add_argument(
        "--baseline_k", type=int, default=None,
        help="Number of rounds for pass@k or reflection baseline (default from config or 10).",
    )
    p.add_argument(
        "--self_rag", action="store_true",
        help="Enable Self-RAG: LLM critique filters irrelevant memories after retrieval.",
    )
    p.add_argument(
        "--self_rag_inject_k", type=int, default=5,
        help="Max memories to inject after Self-RAG critique (default 5 for DB).",
    )
    p.add_argument(
        "--mem0", action="store_true",
        help="Use Mem0 baseline (mem0 library for memory management).",
    )
    p.add_argument(
        "--mem0_infer", type=str, default="true", choices=["true", "false"],
        help="Mem0 infer mode: 'true' = LLM fact extraction (paper), 'false' = raw text (debug).",
    )
    p.add_argument(
        "--mem0_collection", type=str, default="memrl_mem0_llb_baseline",
        help="Qdrant collection name for Mem0 baseline.",
    )
    p.add_argument(
        "--resume_eval_section", type=int, default=-1,
        help=(
            "After loading a completed section checkpoint, run missing validation "
            "before continuing training. Use a positive 1-based section explicitly, "
            "-1 to auto-detect the latest completed section, or 0 to disable. "
            "Completion markers in ck_dir make this idempotent across AIS retries."
        ),
    )
    return p.parse_args()


def _resolve_ck_dir(args, config, project_root: Path):
    """Resolve checkpoint directory from CLI --output_dir or config output_dir.

    Supports fixed run id via MEMRL_RUN_ID so platform retries can continue writing
    to the same checkpoint directory instead of starting a new timestamped run.
    """
    import os
    import time as _time

    out = args.output_dir or getattr(config.experiment, "output_dir", None)
    if out and out != "./results":
        base = Path(out)
        run_id = os.environ.get("MEMRL_RUN_ID", "") or _time.strftime("%Y%m%d-%H%M%S")
        return base / f"exp_{config.experiment.experiment_name}_{run_id}"
    return None  # Let LLBRunner use its default


def _infer_resume_section_from_ckpt_dir(ck_dir: Path) -> int:
    """Infer start_section (0-based index) from existing snapshot epochs in ck_dir.

    Returns:
        int: section index to start from. 0 means start from section 1.
    """
    import os

    snap_root = ck_dir / "snapshot"
    if not snap_root.exists() or not snap_root.is_dir():
        return 0

    epochs = []
    for name in os.listdir(snap_root):
        p = snap_root / name
        if p.is_dir() and name.isdigit():
            epochs.append(int(name))

    if not epochs:
        return 0

    completed = max(epochs)
    # start_section is 0-based index; completed sections are 1-based ids.
    return max(0, int(completed))


def _maybe_resume_from_ckpt_if_needed(memory_service, ck_dir: Path, logger) -> tuple[int, int]:
    """If ck_dir already has snapshots, resume memory state.

    Returns (start_section, resume_batch), both derived AUTHORITATIVELY from the
    loaded snapshot directory name (NOT from llb_batch_progress.json, which is only
    an advisory hint and can be stale/torn):
    - latest is a completed section N (1-based) -> (start_section=N, resume_batch=0)
    - latest is a mid-section batch "N_bM"      -> (start_section=N-1, resume_batch=M+1)
      i.e. re-enter section N (0-based N-1) and skip the M+1 mini-batches whose memory
      state is already captured in the "N_bM" snapshot.
    """
    import os, re

    try:
        snap_root = ck_dir / "snapshot"
        if not snap_root.exists() or not snap_root.is_dir():
            return 0, 0

        def _valid(p: Path) -> bool:
            return p.is_dir() and (
                (p / "snapshot_meta.json").is_file() or (p / "cube").is_dir()
            )

        section_ckpts = []       # list[int] section_num
        batch_ckpts = []         # list[tuple[int sec, int batch]]
        for name in os.listdir(snap_root):
            p = snap_root / name
            if not _valid(p):
                continue
            if name.isdigit():
                section_ckpts.append(int(name))
            else:
                m = re.match(r"^(\d+)_b(\d+)$", name)
                if m:
                    batch_ckpts.append((int(m.group(1)), int(m.group(2))))

        if not section_ckpts and not batch_ckpts:
            return 0, 0

        # Unified ranking: (section_num, rank) — completed section ranks above any of
        # its own batch ckpts, but below the next section's batches.
        _SEC_DONE = 10 ** 9
        candidates = []  # (sort_key, dir_name, start_section_0based, resume_batch)
        for n in section_ckpts:
            candidates.append(((n, _SEC_DONE), str(n), n, 0))           # next section, batch 0
        for sec, b in batch_ckpts:
            candidates.append(((sec, b), f"{sec}_b{b}", sec - 1, b + 1))  # re-enter, skip b+1

        candidates.sort(key=lambda x: x[0])

        # Try newest-first; if a snapshot is corrupt/half-written (e.g. preemption hit
        # mid-save), fall back to the next-newest instead of crashing the whole run.
        for _key, dir_name, start_section, resume_batch in reversed(candidates):
            target = snap_root / dir_name
            try:
                resumed = memory_service.load_checkpoint_snapshot(str(target))
            except Exception as e:
                logger.warning(
                    "Auto-resume: snapshot '%s' failed to load (%s); trying next-newest.",
                    dir_name, e,
                )
                continue
            logger.info(
                "Auto-resume loaded snapshot '%s' from %s (start_section=%d, resume_batch=%d, ckpt_id=%s)",
                dir_name, snap_root, start_section, resume_batch, resumed,
            )
            return max(0, int(start_section)), max(0, int(resume_batch))

        logger.warning("Auto-resume: all %d candidate snapshots failed to load; starting from 0.", len(candidates))
        return 0, 0
    except Exception as e:
        logger.warning(f"Auto-resume from ck_dir failed, fallback to start from 0: {e}")
        return 0, 0


def _resolve_resume_section(args, config, memory_service, ck_dir: Path, logger, explicit_resumed_section: int) -> tuple[int, int]:
    """Resolve final (start_section, resume_batch). Explicit checkpoint resume takes
    precedence and always starts at batch 0; otherwise auto-resume derives both from
    the latest snapshot dir name."""
    if explicit_resumed_section > 0:
        return explicit_resumed_section, 0

    # For clean runs (load_from_checkpoint=false), we still auto-resume if ck_dir already
    # contains snapshots (e.g., platform retry with same MEMRL_RUN_ID).
    if ck_dir is None:
        return 0, 0

    return _maybe_resume_from_ckpt_if_needed(memory_service, ck_dir, logger)


def main():
    """
    Main function to initialize all components and start the runner,
    using the correct MemoryService initialization method.
    """
    # --- Experiment Configuration ---

    try:
        # --- 1. INITIALIZE ALL COMPONENTS ---
        logger.info("Initializing all components...")

        # Load Config and Providers
        args = parse_args(project_root)
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = (project_root / config_path).resolve()
        config = MempConfig.from_yaml(str(config_path))
        # AIS runner injects a validated Matrix gateway token; override stale YAML
        # credentials for both direct providers and Mem0 internals before creation.
        _matrix_api_key = os.environ.get("MATRIX_API_KEY", "").strip()
        if _matrix_api_key:
            config.llm.api_key = _matrix_api_key
            config.embedding.api_key = _matrix_api_key
            logger.info("Using injected Matrix gateway credential for LLM and embedding providers.")
        setup_logging(project_root, config.experiment.experiment_name)

        # Optional JSONL tracing config (env vars take precedence).
        apply_trace_env_from_experiment_config(config.experiment)

        # LLB reflection-prompt variant: env var overrides yaml config (codex-recommended
        # precedence — env for hot-swap/rollback, config for reproducibility). We surface
        # the config value into the env var the updater reads, unless env is already set.
        _reflect_cfg = getattr(config.experiment, "llb_reflection_prompt", "legacy") or "legacy"
        _reflect_env = os.environ.get("MEMRL_LLB_REFLECTION_PROMPT")
        if _reflect_env:
            _reflect_effective, _reflect_source = _reflect_env, "env"
        else:
            os.environ["MEMRL_LLB_REFLECTION_PROMPT"] = str(_reflect_cfg)
            _reflect_effective, _reflect_source = str(_reflect_cfg), "config"
        logger.info(
            "LLB reflection prompt variant = %s (source=%s)",
            _reflect_effective, _reflect_source,
        )

        # Script detail level: env var overrides yaml (same precedence as reflection).
        # Surfaced into MEMRL_LLB_SCRIPT_DETAIL, which OpenAILLM.generate_script reads.
        _script_cfg = getattr(config.memory, "script_detail_level", "abstract") or "abstract"
        _script_env = os.environ.get("MEMRL_LLB_SCRIPT_DETAIL")
        if _script_env:
            _script_effective, _script_source = _script_env, "env"
        else:
            os.environ["MEMRL_LLB_SCRIPT_DETAIL"] = str(_script_cfg)
            _script_effective, _script_source = str(_script_cfg), "config"
        logger.info(
            "LLB script detail level = %s (source=%s)",
            _script_effective, _script_source,
        )

        # Use a temporary directory for all runtime artifacts (like mos_config.json and the DB)
        temp_dir = tempfile.mkdtemp(prefix="memp_live_agent_run_")
        logger.info(f"Using temporary directory for runtime artifacts: {temp_dir}")

        _llm_model = os.environ.get("MEMRL_LLM_MODEL") or config.llm.model
        llm_provider = OpenAILLM(
            api_key=config.llm.api_key,
            base_url=config.llm.base_url,
            model=_llm_model,
            default_temperature=config.llm.temperature,
            default_max_tokens=config.llm.max_tokens,
            provider=config.llm.provider,
            api_version=config.llm.api_version,
        )
        embedding_provider = OpenAIEmbedder(
            api_key=config.embedding.api_key,
            base_url=config.embedding.base_url,
            model=config.embedding.model,
            provider=config.embedding.provider,
            api_version=config.embedding.api_version,
        )

        # --- Use the detailed MemoryService setup from your demo script ---
        logger.info("Configuring and initializing MemoryService...")
        # user_id must be STABLE across retries so qdrant collection name matches.
        # Priority:
        #   1) Restore from latest snapshot's snapshot_meta.json (survives PID change)
        #   2) Derive from MEMRL_RUN_ID env var (stable across retries)
        #   3) Fall back to PID (unique per process, non-resumable)
        user_id = None
        try:
            _ck_dir_base = args.output_dir or getattr(config.experiment, "output_dir", None)
            if _ck_dir_base:
                _exp_name = getattr(config.experiment, "experiment_name", "exp")
                _run_id = os.environ.get("MEMRL_RUN_ID", "")
                _ck_dir_candidate = Path(_ck_dir_base) / f"exp_{_exp_name}_{_run_id}" if _run_id else Path(_ck_dir_base) / f"exp_{_exp_name}"
                _snap_root = _ck_dir_candidate / "snapshot"
                if _snap_root.exists():
                    import re
                    _cands = []
                    for _n in os.listdir(_snap_root):
                        _p = _snap_root / _n
                        if not (_p / "snapshot_meta.json").is_file():
                            continue
                        if _n.isdigit():
                            _cands.append(((int(_n), 10**9), _n))
                        else:
                            _m = re.match(r"^(\d+)_b(\d+)$", _n)
                            if _m:
                                _cands.append(((int(_m.group(1)), int(_m.group(2))), _n))
                    if _cands:
                        _cands.sort(key=lambda x: x[0])
                        _latest = _cands[-1][1]
                        _meta = json.loads((_snap_root / _latest / "snapshot_meta.json").read_text())
                        _restored_uid = _meta.get("user_id")
                        if _restored_uid:
                            user_id = _restored_uid
                            logger.info("Restored user_id from snapshot '%s': %s", _latest, user_id)
        except Exception as _e:
            logger.warning("Failed to restore user_id from snapshot: %s", _e)
        if not user_id:
            _run_id = os.environ.get("MEMRL_RUN_ID", "").strip()
            if _run_id:
                # Sanitize run_id for user_id (alnum + underscore only)
                import re
                _sanitized = re.sub(r"[^A-Za-z0-9_]", "_", _run_id)
                user_id = f"live_agent_{_sanitized}"
                logger.info("Derived stable user_id from MEMRL_RUN_ID: %s", user_id)
            else:
                user_id = f"live_agent_exp_{os.getpid()}"
                logger.warning("No MEMRL_RUN_ID; falling back to PID-based user_id: %s (NON-RESUMABLE)", user_id)
        build_strategy = BuildStrategy(config.memory.build_strategy)
        retrieve_strategy = RetrieveStrategy(config.memory.retrieve_strategy)
        update_strategy = UpdateStrategy(config.memory.update_strategy)

        # 1. Create the mos_config dictionary
        # Note: MemOS config validation does not allow 'api_version', so we rely on env vars.
        if config.llm.provider == "azure":
            os.environ["OPENAI_API_TYPE"] = "azure"
        if config.llm.api_version:
            os.environ["OPENAI_API_VERSION"] = config.llm.api_version

        # Keep the MemOS authorization DB outside /tmp. Long-running AIStudio jobs
        # can have stale /tmp directories reclaimed while the process is alive; when
        # users.db disappears, the in-memory cube remains usable but MOS permission
        # checks start rejecting every cache miss. Use a stable per-user path instead.
        _user_db_root = Path(
            os.environ.get(
                "MEMRL_USER_DB_DIR",
                str(project_root / ".runtime" / "memos_user_db"),
            )
        )
        _user_db_root.mkdir(parents=True, exist_ok=True)
        _safe_user_id = "".join(
            ch if (ch.isalnum() or ch in "-_.") else "_" for ch in user_id
        )
        _user_db_path = _user_db_root / f"{_safe_user_id}.db"
        logger.info("Using persistent MemOS user DB: %s", _user_db_path)

        mos_config = {
            "chat_model": {
                "backend": "openai",
                "config": {
                    "model_name_or_path": config.llm.model,
                    "api_key": config.llm.api_key,
                    "api_base": config.llm.base_url,
                },
            },
            "mem_reader": {
                "backend": "simple_struct",
                "config": {
                    "llm": {
                        "backend": "openai",
                        "config": {
                            "model_name_or_path": config.llm.model,
                            "api_key": config.llm.api_key,
                            "api_base": config.llm.base_url,
                        },
                    },
                    "embedder": {
                        "backend": "universal_api",
                        "config": {
                            "provider": config.embedding.provider,
                            "model_name_or_path": config.embedding.model,
                            "api_key": config.embedding.api_key,
                            "base_url": config.embedding.base_url,
                        },
                    },
                    "chunker": {"backend": "sentence", "config": {"chunk_size": 500}},
                },
            },
            "user_manager": {
                "backend": "sqlite",
                "config": {"db_path": str(_user_db_path)},
            },
            "top_k": 5,
        }

        # 2. Write the config to a temporary JSON file
        mos_config_path = os.path.join(temp_dir, "mos_config.json")
        with open(mos_config_path, "w") as f:
            json.dump(mos_config, f)

        # 3. rl_config:

        enable_value_driven = config.experiment.enable_value_driven
        rl_config = config.rl_config
        # LLB-only: optionally enforce a Q-value floor (memory_rl-style).
        # Keep the knob under experiment.* so other benchmarks are unaffected.
        if getattr(config.experiment, "llb_q_floor", None) is not None:
            try:
                # pydantic v2
                rl_config = rl_config.model_copy(
                    update={"q_floor": float(config.experiment.llb_q_floor)}
                )
            except Exception:
                # best-effort fallback for non-pydantic configs
                try:
                    setattr(rl_config, "q_floor", float(config.experiment.llb_q_floor))
                except Exception:
                    pass
        logger.info(
            "LLB effective q_floor=%s (experiment.llb_q_floor=%s)",
            getattr(rl_config, "q_floor", None),
            getattr(config.experiment, "llb_q_floor", None),
        )

        logger.info("Config:\n%s", config.model_dump_json(indent=2))

        # 4. Initialize MemoryService with the config path and providers
        if args.mem0:
            from memrl.service.mem0_memory_service import Mem0MemoryService

            mem0_qdrant_path = os.path.join(temp_dir, "mem0_qdrant")
            memory_service = Mem0MemoryService(
                llm_base_url=config.llm.base_url,
                llm_model=config.llm.model,
                llm_api_key=config.llm.api_key,
                embed_base_url=config.embedding.base_url,
                embed_model=config.embedding.model,
                embed_api_key=config.embedding.api_key,
                embedding_dims=config.embedding.dimension,
                qdrant_path=mem0_qdrant_path,
                collection_name=args.mem0_collection,
                top_k=config.memory.k_retrieve or 5,
                infer=(args.mem0_infer == "true"),
                user_id=config.memory.user_id,
            )
            logger.info(
                "Using Mem0MemoryService for LLB %s (infer=%s, collection=%s)",
                config.experiment.task, args.mem0_infer, args.mem0_collection,
            )
        elif args.region:
            from memrl.service.region_manager import RegionManager
            from memrl.service.region_memory_service import RegionMemoryService
            from memrl.configs.task_hierarchy import get_task_hierarchy

            task_type = config.experiment.task  # "os" or "db"
            benchmark_key = f"llb_{task_type}" if task_type in ("os", "db") else "llb_os"
            hierarchy = get_task_hierarchy(args.region_k)

            region_manager = RegionManager(
                task_hierarchy=hierarchy,
                min_cluster_size=15,
                temperature=0.1,
                shrinkage_top_n=1,
                region_utility_mode="beta",
                bayesian_smoothing_C=0.5,
                propagation_enabled=True,
                propagation_eta=args.propagation_eta,
                propagation_k=30,
                propagation_sim_min=0.40,
            )
            if args.shrinkage_confidence_k is not None:
                region_manager.shrinkage_confidence_k = float(args.shrinkage_confidence_k)

            memory_service = RegionMemoryService(
                mos_config_path=mos_config_path,
                llm_provider=llm_provider,
                embedding_provider=embedding_provider,
                strategy_config=StrategyConfiguration(
                    build_strategy, retrieve_strategy, update_strategy
                ),
                user_id=user_id,
                num_workers=config.experiment.batch_size,
                max_keywords=config.memory.max_keywords,
                add_similarity_threshold=config.memory.add_similarity_threshold,
                enable_value_driven=enable_value_driven,
                rl_config=rl_config,
                vector_dimension=config.embedding.dimension,
                sim_norm_mean=getattr(config.memory, "sim_norm_mean", None),
                sim_norm_std=getattr(config.memory, "sim_norm_std", None),
                region_manager=region_manager,
                region_gating_mode=args.region_gating_mode,
                use_z_score_normalization=(not args.no_z_norm),
                dedup_by_task_id=bool(getattr(config.experiment, "llb_dedup_by_task_id", False)),
                explore_schedule=args.explore_schedule or "",
                explore_success_ratio=0.7,
            )
            logger.info(
                "Using RegionMemoryService for LLB %s (K=%d, gating=%s, z_norm=%s)",
                task_type, args.region_k, args.region_gating_mode, not args.no_z_norm,
            )
        else:
            memory_service = MemoryService(
                mos_config_path=mos_config_path,
                llm_provider=llm_provider,
                embedding_provider=embedding_provider,
                strategy_config=StrategyConfiguration(
                    build_strategy, retrieve_strategy, update_strategy
                ),
                user_id=user_id,
                num_workers=config.experiment.batch_size,
                max_keywords=config.memory.max_keywords,
                add_similarity_threshold=config.memory.add_similarity_threshold,
                enable_value_driven=enable_value_driven,
                rl_config=rl_config,
                vector_dimension=config.embedding.dimension,
                use_z_score_normalization=bool(config.experiment.llb_use_z_score_normalization),
                dedup_by_task_id=bool(getattr(config.experiment, "llb_dedup_by_task_id", False)),
            )

        # Load from checkpoint if configured
        resumed_section = 0  # Default: start from beginning
        if config.memory.load_from_checkpoint and config.memory.checkpoint_path:
            checkpoint_path = Path(config.memory.checkpoint_path)
            if not checkpoint_path.is_absolute():
                checkpoint_path = project_root / checkpoint_path

            if checkpoint_path.exists():
                logger.info(f"Loading memory from checkpoint: {checkpoint_path}")
                try:
                    resumed_section = memory_service.load_checkpoint_snapshot(
                        str(checkpoint_path)
                    )
                    logger.info(
                        f"✓ Checkpoint loaded successfully from section {resumed_section}"
                    )
                except Exception as e:
                    logger.error(f"Failed to load checkpoint: {e}", exc_info=True)
                    raise
            else:
                logger.warning(f"Checkpoint path does not exist: {checkpoint_path}")
                logger.warning("Starting with fresh memory service")

        logger.info("All components initialized successfully.")

        ck_dir = _resolve_ck_dir(args, config, project_root)
        resumed_section, resumed_batch = _resolve_resume_section(
            args=args,
            config=config,
            memory_service=memory_service,
            ck_dir=ck_dir,
            logger=logger,
            explicit_resumed_section=resumed_section,
        )

        # Initialize the Runner with the fully constructed components
        # Note: LLBRunner will create LanguageModelAgent instances internally using the adapter
        runner = LLBRunner(
            root=project_root,
            memory_service=memory_service,
            llm_provider=llm_provider,
            embedding_provider=embedding_provider,
            exp_name=config.experiment.experiment_name,
            random_seed=config.experiment.random_seed,
            num_section=config.experiment.num_sections,
            batch_size=config.experiment.batch_size,
            max_steps=config.experiment.max_steps,
            rl_config=rl_config,
            bon=config.experiment.bon,
            retrieve_k=config.memory.k_retrieve,
            mode=config.experiment.mode,
            # enable_value_driven=enable_value_driven,
            task=config.experiment.task,
            split_file=config.experiment.split_file,
            valid_interval=config.experiment.valid_interval,
            test_interval=config.experiment.test_interval,
            # Backwards/forwards compatible naming across configs:
            # - older code used "train_set_ratio"
            # - current configs use "dataset_ratio"
            train_set_ratio=getattr(
                config.experiment,
                "train_set_ratio",
                getattr(config.experiment, "dataset_ratio", 1.0),
            ),
            start_section=resumed_section,  # Resume section index (0-based) if available
            start_batch=resumed_batch,  # Mid-section batch to resume at (authoritative, from snapshot name)
            algorithm=config.experiment.algorithm,
            val_before_train=config.experiment.val_before_train,
            valid_file=config.experiment.valid_file,  # Get from config
            ck_dir=_resolve_ck_dir(args, config, project_root),
            baseline_mode=args.baseline_mode or getattr(config.experiment, "baseline_mode", None),
            baseline_k=args.baseline_k or getattr(config.experiment, "baseline_k", 10),
            self_rag=args.self_rag or getattr(config.experiment, "self_rag", False),
            self_rag_inject_k=args.self_rag_inject_k,
            eval_runs=int(getattr(config.experiment, "eval_runs", 1)),
            eval_temperature=float(getattr(config.experiment, "eval_temperature", 0.0)),
            ckpt_save_every_n_batches=int(getattr(config.experiment, "ckpt_save_every_n_batches", 0)),
            ckpt_max_keep=int(getattr(config.experiment, "ckpt_max_keep", 3)),
            region_cluster_init_step=int(
                getattr(config.experiment, "region_cluster_init_step", 500)
            ),
            resume_eval_section=int(args.resume_eval_section),
        )

        # Configure failure summary if requested
        if args.failure_summary_n_slots > 0:
            runner.configure_failure_summary(
                n_slots=args.failure_summary_n_slots,
                inline_k=args.failure_summary_k,
                mode="region" if args.region else "inline",
            )

        # --- RUN THE EXPERIMENT ---
        runner.run()

    except Exception as e:
        logger.error(
            f"An unhandled error occurred during the experiment: {e}", exc_info=True
        )
    finally:
        # --- 3. CLEANUP ---
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")


if __name__ == "__main__":
    main()
