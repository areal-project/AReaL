# memp/run/llb_rl_runner.py
import logging
import os
import sys
import yaml
import time
import json
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Sequence, Set, Tuple
from datetime import datetime
from collections import defaultdict, OrderedDict
from tqdm import tqdm

import numpy as np
import pandas as pd
import psutil
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional dependency
    SummaryWriter = None  # type: ignore[assignment]

import contextlib

from .base_runner import BaseRunner
from memrl.service.memory_service import MemoryService
from memrl.service.value_driven import RLConfig
from memrl.providers.llm import OpenAILLM
from memrl.providers.embedding import OpenAIEmbedder
from memrl.utils.task_id import extract_task_id

from memrl.lifelongbench_eval.prompts import (
    DEFAULT_SYSTEM_PROMPT as LLB_DEFAULT_SYSTEM_PROMPT,
    build_llb_prompt_with_memory,
    build_llb_system_prompt,
)
from memrl.lifelongbench_eval.memory_context import format_llb_memory_context

# --- Setup LLB Path ---
# 动态查找项目根目录和 LLB 路径
_current_file = Path(__file__).resolve()
_project_root = _current_file.parent.parent.parent  # memp/run/llb_rl_runner.py -> memp/
LLB_ROOT = _project_root / "3rdparty" / "LifelongAgentBench"

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

# 导入 LLB 组件
from src.agents.instance.language_model_agent import LanguageModelAgent  # type: ignore
from src.typings import (  # type: ignore
    Session,
    SampleStatus,
    SessionMetricCalculationPartial,
    TaskName,
    SessionEvaluationOutcome,
)
from src.factories.chat_history_item import ChatHistoryItemFactory  # type: ignore

MAX_RETRIES = 4
RETRY_DELAY = 2

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = LLB_DEFAULT_SYSTEM_PROMPT


class LLBRunner(BaseRunner):
    """
    Runner for LifelongAgentBench tasks (DB, OS, KG).
    Handles memory-driven agent evaluation and training.
    """

    def __init__(
        self,
        root: Path,
        memory_service: MemoryService,
        llm_provider: OpenAILLM,
        embedding_provider: OpenAIEmbedder,
        exp_name: str,
        task: str,
        split_file: str,
        num_section: int,
        batch_size: int,
        max_steps: int,
        rl_config: Optional[RLConfig],
        retrieve_k: int = 1,
        mode: str = "train",
        bon: int = 0,
        random_seed: int = 42,
        valid_interval: int = 2,
        test_interval: int = 2,
        train_set_ratio: float = 1.0,
        start_section: int = 0,
        algorithm: str = "rl",
        val_before_train: bool = True,
        system_prompt: str = "",
        os_timeout: int = 20,
        sparql_url: Optional[str] = None,
        ontology_dir: Optional[str] = None,
        kg_offline_fallback: bool = False,
        limit: Optional[int] = None,
        valid_file: Optional[str] = None,
        ck_dir: Optional[Path] = None,
        baseline_mode: Optional[str] = None,
        baseline_k: int = 10,
        self_rag: bool = False,
        self_rag_inject_k: int = 5,
        eval_runs: int = 1,
        eval_temperature: float = 0.0,
        ckpt_save_every_n_batches: int = 0,
        ckpt_max_keep: int = 3,
        region_cluster_init_step: int = 500,
        start_batch: int = 0,
        resume_eval_section: int = 0,
    ):
        self.root = root
        self.memory_service = memory_service
        self.llm_provider = llm_provider
        self.embedding_provider = embedding_provider
        self.exp_name = exp_name
        self.task = task
        self.split_file = split_file
        self.random_seed = random_seed
        self.num_section = num_section
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.retrieve_k = retrieve_k
        self.mode = mode
        self.valid_interval = valid_interval
        self.test_interval = test_interval
        self.train_set_ratio = train_set_ratio
        self.start_section = start_section
        # Batch-level checkpointing (0 = disabled, keep legacy section-only behavior).
        # When > 0, save a memory snapshot every N mini-batches within a section so a
        # platform preemption mid-section can resume without redoing the whole section.
        self.ckpt_save_every_n_batches = max(0, int(ckpt_save_every_n_batches))
        self.ckpt_max_keep = max(1, int(ckpt_max_keep))
        configured_cluster_init_step = int(region_cluster_init_step)
        if configured_cluster_init_step < 0:
            raise ValueError(
                "region_cluster_init_step must be a non-negative integer, "
                f"got {configured_cluster_init_step}"
            )
        cluster_init_step_override = os.environ.get("MEMRL_REGION_CLUSTER_INIT_STEP")
        if cluster_init_step_override is not None:
            try:
                configured_cluster_init_step = int(cluster_init_step_override)
            except ValueError as exc:
                raise ValueError(
                    "MEMRL_REGION_CLUSTER_INIT_STEP must be a non-negative integer, "
                    f"got {cluster_init_step_override!r}"
                ) from exc
            if configured_cluster_init_step < 0:
                raise ValueError(
                    "MEMRL_REGION_CLUSTER_INIT_STEP must be a non-negative integer, "
                    f"got {cluster_init_step_override!r}"
                )
        # 0 disables only the mid-section trigger. The end-of-section fallback below
        # intentionally remains active so Region never proceeds to E2 unclustered.
        self.region_cluster_init_step = configured_cluster_init_step
        raw_topology_cooldown = os.environ.get("MEMRL_REGION_TOPOLOGY_COOLDOWN_SECTIONS", "0")
        try:
            self.region_topology_cooldown_sections = int(raw_topology_cooldown)
        except ValueError as exc:
            raise ValueError("MEMRL_REGION_TOPOLOGY_COOLDOWN_SECTIONS must be a non-negative integer") from exc
        if self.region_topology_cooldown_sections < 0:
            raise ValueError("MEMRL_REGION_TOPOLOGY_COOLDOWN_SECTIONS must be a non-negative integer")
        raw_mid_maintenance = os.environ.get("MEMRL_REGION_TOPOLOGY_MAINTENANCE_STEPS", "")
        try:
            self.region_topology_maintenance_steps = tuple(sorted({
                int(raw.strip()) for raw in raw_mid_maintenance.split(",") if raw.strip()
            }))
        except ValueError as exc:
            raise ValueError(
                "MEMRL_REGION_TOPOLOGY_MAINTENANCE_STEPS must be a comma-separated "
                "list of positive integer global training steps"
            ) from exc
        if any(step <= 0 for step in self.region_topology_maintenance_steps):
            raise ValueError(
                "MEMRL_REGION_TOPOLOGY_MAINTENANCE_STEPS must contain only positive integer steps"
            )
        # Mid-section resume batch, derived AUTHORITATIVELY from the loaded snapshot dir
        # name by run_llb (not from llb_batch_progress.json). 0 = start section at batch 0.
        self._resume_batch_start = max(0, int(start_batch))
        # Validation recovery for a section whose checkpoint was saved but whose
        # post-section validation was interrupted. Positive = explicit 1-based section;
        # -1 = auto-detect latest completed section; 0 = disabled.
        self.resume_eval_section = int(resume_eval_section)
        self.bon = bon
        self.algorithm = algorithm
        self.val_before_train = val_before_train
        self._base_system_prompt = (system_prompt or DEFAULT_SYSTEM_PROMPT).strip()
        self.system_prompt = build_llb_system_prompt(
            task=self.task,
            base_prompt=self._base_system_prompt,
        )
        self.os_timeout = os_timeout
        self.sparql_url = sparql_url
        self.ontology_dir = ontology_dir
        self.kg_offline_fallback = kg_offline_fallback
        self.limit = limit
        self.results_log = []
        self.valid_file = valid_file
        self.baseline_mode = baseline_mode
        self.baseline_k = baseline_k
        self.self_rag = bool(self_rag)
        self.self_rag_inject_k = max(1, int(self_rag_inject_k))
        self.eval_runs = max(1, int(eval_runs))
        self.eval_temperature = float(eval_temperature)

        self.rl_config: Optional[RLConfig] = rl_config

        # Optional per-task JSONL tracing (activated via TRACE_JSONL_PATH).
        from memrl.trace.llb_jsonl import LLBJsonlTracer
        from memrl.trace.tracing_llm import TracingLLMProvider

        self._trace = LLBJsonlTracer.from_env()

        # Create LLM adapter for LLB LanguageModelAgent (optionally wrapped for tracing)
        from memrl.lifelongbench_eval.lm_adapter import MempOpenAIAdapter

        provider_for_adapter = self.llm_provider
        if self._trace is not None:
            provider_for_adapter = TracingLLMProvider(self.llm_provider, tracer=self._trace)
        self.adapter = MempOpenAIAdapter(provider_for_adapter)

        # --- [TENSORBOARD] Initialize SummaryWriter ---
        tb_log_dir = (
            self.root
            / "logs"
            / "tensorboard"
            / f"exp_{self.exp_name}_{time.strftime('%Y%m%d-%H%M%S')}"
        )
        if SummaryWriter is None:
            # Keep runner functional even when tensorboard isn't installed.
            class _NoOpWriter:
                def add_scalar(self, *args: Any, **kwargs: Any) -> None:
                    return

                def close(self) -> None:
                    return

            self.writer = _NoOpWriter()
            logger.warning(
                "TensorBoard is not available (missing dependency). "
                "Proceeding without TensorBoard logging."
            )
        else:
            self.writer = SummaryWriter(log_dir=str(tb_log_dir))
            logger.info(f"TensorBoard logs will be saved to: {tb_log_dir}")
        self.ck_dir = (
            ck_dir if ck_dir is not None
            else self.root
            / "results"
            / "llb"
            / f"exp_{self.exp_name}_{time.strftime('%Y%m%d-%H%M%S')}"
        )

        # Inline failure summary (disabled by default; call configure_failure_summary).
        self._failure_summary_n_slots = 0
        self._failure_summary_inline_k: Optional[int] = None
        self._failure_summary_independent_pool = False
        self._failure_summary_min_success = 0
        self._failure_summary_min_similarity = 0.0
        self._failure_summary_min_evidence = 1
        self._failure_summary_db_structured = False
        self._failure_summary_preserve_selection = False
        self._failure_summary_fixed_budget = False

        # Build LLB task and load datasets
        self._build_llb_task()
        self._load_eval_datasets()

        # Durable per-task outcome ledger for true union cumulative SR.  This is
        # intentionally independent of any memory backend (especially Mem0,
        # whose atomic facts can be deduplicated/deleted and therefore cannot
        # serve as a task-outcome history).
        self._llb_cum_success_ids: Set[str] = set()
        self._llb_cum_total = len(self.dataset)
        self._llb_outcome_path = self.ck_dir / "task_outcomes.jsonl"
        self._llb_task_cum_state_path = self.ck_dir / "local_cache" / "cum_state.json"
        self._load_llb_cum_state()

    def _log_token_usage(self, section_num: int, mini_batch: Optional[int] = None):
        """Log current token usage for LLM and Embedding providers."""
        try:
            # Token usage logging is best-effort. Not all provider implementations
            # expose get_token_usage() (e.g., some OpenAI-compatible clients).
            if not hasattr(self.llm_provider, "get_token_usage") or not hasattr(
                self.embedding_provider, "get_token_usage"
            ):
                return

            llm_usage = self.llm_provider.get_token_usage()
            emb_usage = self.embedding_provider.get_token_usage()

            context = f"Section {section_num}"
            if mini_batch is not None:
                context += f" Mini-batch {mini_batch}"

            logger.info(f"\n=== Token Usage after {context} ===")
            logger.info(f"LLM Prompt Tokens:     {llm_usage.get('prompt_tokens', 0)}")
            logger.info(
                f"LLM Completion Tokens: {llm_usage.get('completion_tokens', 0)}"
            )
            logger.info(f"LLM Total Tokens:      {llm_usage.get('total_tokens', 0)}")
            logger.info(f"Embedding Total Tokens: {emb_usage.get('total_tokens', 0)}")
            logger.info(
                f"GRAND TOTAL:           {llm_usage.get('total_tokens', 0) + emb_usage.get('total_tokens', 0)}"
            )
            logger.info("==========================================\n")

            # Log to TensorBoard (only for section level to avoid clutter)
            if hasattr(self, "writer") and self.writer and mini_batch is None:
                self.writer.add_scalar(
                    "Token_Usage/LLM_Total",
                    llm_usage.get("total_tokens", 0),
                    section_num,
                )
                self.writer.add_scalar(
                    "Token_Usage/Embedding_Total",
                    emb_usage.get("total_tokens", 0),
                    section_num,
                )
                self.writer.add_scalar(
                    "Token_Usage/Grand_Total",
                    llm_usage.get("total_tokens", 0) + emb_usage.get("total_tokens", 0),
                    section_num,
                )

        except Exception as e:
            logger.warning(f"Failed to log token usage: {e}")

    def _check_memory_usage(self, context: str = ""):
        """Monitor memory usage to detect memory leaks."""
        try:
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            mem_mb = mem_info.rss / 1024 / 1024
            logger.info(f"[Memory Monitor] {context}: {mem_mb:.2f} MB")
        except Exception as e:
            logger.warning(f"Failed to check memory usage: {e}")

    def _build_llb_task(self):
        """Build LLB task object and load dataset."""
        from memrl.lifelongbench_eval.task_wrappers import (
            build_task,
            ensure_standard_prompts,
        )

        # Ensure standard prompts are generated
        ensure_standard_prompts()

        # Build task object
        self.task_obj, self.task_name = build_task(
            task=self.task,
            data_file_path=self.split_file,
            max_round=self.max_steps,
            os_timeout=self.os_timeout,
            kg_sparql_url=self.sparql_url,
            kg_ontology_dir=self.ontology_dir,
            kg_offline_fallback=self.kg_offline_fallback,
        )

        # Load dataset
        with open(self.split_file, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)

        logger.info(f"Loaded {len(self.dataset)} samples from {self.split_file}")

        # Split dataset into sections
        self._split_dataset()

    def _split_dataset(self):
        """Split dataset into sections based on num_section and train_set_ratio."""
        # Get all sample keys
        all_keys = sorted(list(self.dataset.keys()), key=lambda x: str(x))

        # Apply train_set_ratio
        if self.train_set_ratio < 1.0:
            num_total = len(all_keys)
            num_to_sample = int(num_total * self.train_set_ratio)
            logger.info(
                f"Sampling {num_to_sample} from {num_total} total samples ({self.train_set_ratio:.2%})"
            )
            random.seed(self.random_seed)
            all_keys = random.sample(all_keys, k=num_to_sample)

        # Apply limit if specified
        if self.limit is not None:
            all_keys = all_keys[: self.limit]
            logger.info(f"Limited to {len(all_keys)} samples")

        # Split into sections
        if self.num_section == 1:
            self.section_splits = [all_keys]
        else:
            # Copy all keys for each section instead of splitting
            self.section_splits = [list(all_keys) for _ in range(self.num_section)]

        logger.info(
            f"Split {len(all_keys)} samples into {len(self.section_splits)} sections"
        )
        for i, section_keys in enumerate(self.section_splits):
            logger.info(f"  Section {i}: {len(section_keys)} samples")

    def _load_eval_datasets(self):
        """Load validation dataset."""
        self.valid_dataset = {}

        if self.valid_file and os.path.exists(self.valid_file):
            with open(self.valid_file, "r", encoding="utf-8") as f:
                self.valid_dataset = json.load(f)
            logger.info(
                f"Loaded {len(self.valid_dataset)} validation samples from {self.valid_file}"
            )
        else:
            logger.info("No validation dataset specified or file not found")

    def _create_llb_agent(
        self, memory_context: Optional[str] = None
    ) -> LanguageModelAgent:
        """Create a LanguageModelAgent instance for LLB task execution.

        Args:
            memory_context: Optional memory context to prepend to system prompt

        Returns:
            LanguageModelAgent instance configured with system prompt
        """
        full_prompt = self._build_llb_full_prompt(memory_context=memory_context)

        # Create and return agent
        return LanguageModelAgent(
            language_model=self.adapter, system_prompt=full_prompt
        )

    def _build_llb_full_prompt(self, *, memory_context: Optional[str]) -> str:
        """Build the exact system prompt used by LanguageModelAgent."""
        # Align prompt assembly ordering with memory_rl/dev/feat-mdp-llb:
        # system prompt -> (optional) memory context -> strict output constraints at the very end.
        if memory_context:
            return build_llb_prompt_with_memory(
                task=self.task,
                base_prompt=self._base_system_prompt,
                memory_context=memory_context,
            )
        return self.system_prompt

    def _session_to_chat_messages(self, session: Session) -> List[Dict[str, str]]:
        """Best-effort extraction of full chat history as [{role, content}, ...]."""
        if session is None:
            return []

        ch = getattr(session, "chat_history", None)
        if ch is None and isinstance(session, dict):
            ch = session.get("chat_history")

        if not ch:
            return []

        # LLB ChatHistory type: has get_value_length/get_item_deep_copy.
        if hasattr(ch, "get_value_length") and hasattr(ch, "get_item_deep_copy"):
            msgs: List[Dict[str, str]] = []
            n = int(ch.get_value_length())
            for i in range(n):
                item = ch.get_item_deep_copy(i)
                role = getattr(item, "role", None)
                content = getattr(item, "content", "")
                role_s = str(role)
                # normalize common role strings for readability
                up = role_s.upper()
                if "USER" in up:
                    role_s = "user"
                elif "AGENT" in up or "ASSISTANT" in up:
                    role_s = "assistant"
                msgs.append({"role": role_s, "content": str(content or "")})
            return msgs

        # Fallback: list[dict] or list[object]
        msgs2: List[Dict[str, str]] = []
        if isinstance(ch, list):
            for m in ch:
                if isinstance(m, dict):
                    role = m.get("role") or m.get("speaker") or "unknown"
                    content = m.get("content") or m.get("text") or ""
                else:
                    role = getattr(m, "role", "unknown")
                    content = getattr(m, "content", str(m))
                msgs2.append({"role": str(role), "content": str(content or "")})
        return msgs2

    def process_retrieve_mems(
        self, retrieved_mems: List[dict]
    ) -> Dict[str, List[dict]]:
        """Process retrieved memories into success/failed categories.

        Args:
            retrieved_mems: List of retrieved memory dictionaries

        Returns:
            Dictionary with 'successed' and/or 'failed' keys containing categorized memories
        """
        success_mems = []
        failed_mems = []

        for mem in retrieved_mems:
            metadata = mem.get("metadata", {})

            raw_success = None
            if isinstance(metadata, dict):
                raw_success = metadata.get("success")
            else:
                model_extra = getattr(metadata, "model_extra", None)
                if isinstance(model_extra, dict) and "success" in model_extra:
                    raw_success = model_extra.get("success")
                else:
                    raw_success = getattr(metadata, "success", None)

            if isinstance(raw_success, str):
                is_success = raw_success.strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "y",
                    "success",
                }
            else:
                is_success = bool(raw_success)

            if is_success:
                success_mems.append(mem)
            else:
                failed_mems.append(mem)

        final_mems = {}
        if success_mems:
            final_mems["successed"] = success_mems
        if failed_mems:
            final_mems["failed"] = failed_mems

        return final_mems

    # ==================== Inline Failure Summary ====================

    def configure_failure_summary(self, n_slots: int = 1, inline_k: Optional[int] = None,
                                   mode: str = "inline", summaries_path: Optional[str] = None,
                                   replace_with_summary: bool = True,
                                   independent_pool: bool = False,
                                   min_success: int = 0,
                                   min_similarity: float = 0.0,
                                   min_evidence: int = 1,
                                   db_structured: bool = False,
                                   preserve_selection: bool = False,
                                   fixed_budget: bool = False):
        """Enable failure summary injection.

        Args:
            n_slots: number of top-k slots reserved for failure memories
            inline_k: for mode=inline, how many failures to aggregate. None = use all.
            mode: "region" uses per-region pre-computed summaries (needs summaries_path);
                  "inline" (default) aggregates retrieved failures on-the-fly.
            summaries_path: path to JSON with pre-computed per-region failure summaries.
            replace_with_summary: if True, replace failure content with summary text.
        """
        self._failure_summary_n_slots = n_slots
        self._failure_summary_inline_k = inline_k
        self._failure_summary_mode = mode
        self._failure_summary_replace = replace_with_summary
        self._failure_summary_independent_pool = bool(independent_pool)
        self._failure_summary_min_success = max(0, int(min_success))
        self._failure_summary_min_similarity = max(0.0, float(min_similarity))
        self._failure_summary_min_evidence = max(1, int(min_evidence))
        self._failure_summary_db_structured = bool(db_structured)
        self._failure_summary_preserve_selection = bool(preserve_selection)
        self._failure_summary_fixed_budget = bool(fixed_budget)
        self._region_failure_summaries = None

        if summaries_path and mode == "region" and replace_with_summary:
            import json as _json
            data = _json.loads(Path(summaries_path).read_text())
            self._region_failure_summaries = data.get("summaries", {})
            logger.info(
                "[Region Failure Summary] loaded %d region summaries from %s, n_slots=%d",
                len(self._region_failure_summaries), summaries_path, n_slots,
            )
        else:
            logger.info(
                "[Failure Summary] enabled: n_slots=%d, mode=%s, inline_k=%s, replace=%s, "
                "independent_pool=%s, min_success=%d, min_similarity=%.3f, min_evidence=%d",
                n_slots, mode, inline_k, replace_with_summary,
                self._failure_summary_independent_pool,
                self._failure_summary_min_success,
                self._failure_summary_min_similarity,
                self._failure_summary_min_evidence,
            )

    def _inject_failure_summary(
        self, processed_mems: Dict[str, List[dict]], task_description: str,
        candidate_mems: Optional[List[dict]] = None,
        target_skill_list: Optional[List[str]] = None,
    ) -> Dict[str, List[dict]]:
        """Post-process retrieved memories: inject failure summary.

        Supports two modes (set via configure_failure_summary):
        - "region": replace failure content with its region's dynamically-built summary
        - "inline": aggregate retrieved failures on-the-fly into frequency-based summary
        """
        n_slots = self._failure_summary_n_slots
        if n_slots <= 0:
            return processed_mems

        final_budget = int(getattr(self.rl_config, "topk", self.retrieve_k) or self.retrieve_k) if self.rl_config else int(self.retrieve_k)
        final_budget = max(1, min(int(self.retrieve_k), final_budget))
        effective_failure_slots = min(n_slots, final_budget)

        # Fair ALFWorld-style direct Region-FS: reserve failure slots, retrieve
        # missing failures from the full pool, replace with direct Region summary,
        # and always fill the final context to top-k when candidates are available.
        if getattr(self, "_failure_summary_fixed_budget", False):
            rm = getattr(self.memory_service, "region_manager", None)
            if rm is None or not getattr(rm, "regions", None):
                return processed_mems
            selected_success = list(processed_mems.get("successed", []))
            selected_failure = list(processed_mems.get("failed", []))
            candidate_buckets = self.process_retrieve_mems(candidate_mems or [])
            success_pool = selected_success + [m for m in candidate_buckets.get("successed", []) if m.get("memory_id") not in {x.get("memory_id") for x in selected_success}]
            failure_pool = selected_failure + [m for m in candidate_buckets.get("failed", []) if m.get("memory_id") not in {x.get("memory_id") for x in selected_failure}]
            failure_mems = failure_pool[:effective_failure_slots]
            target_success = final_budget - len(failure_mems)
            success_mems = success_pool[:target_success]
            # If failure supply is insufficient, fill every remaining slot with success.
            success_mems = success_pool[: final_budget - len(failure_mems)]
            if failure_mems:
                self._replace_failure_with_region_summary(failure_mems)
            result = {}
            if success_mems:
                result["successed"] = success_mems
            if failure_mems:
                result["failed"] = failure_mems
            return result

        # Exact corrected/direct Region-FS contract: preserve the base retrieval
        # and exploration selection. Only replace content of naturally selected
        # failure IDs; do not deduplicate, pull from a larger pool, or backfill.
        if getattr(self, "_failure_summary_preserve_selection", False):
            success_mems = list(processed_mems.get("successed", []))
            failure_mems = list(processed_mems.get("failed", []))
            if not failure_mems:
                return processed_mems
            mode = getattr(self, "_failure_summary_mode", "region")
            if mode == "region":
                self._replace_failure_with_region_summary(failure_mems)
            else:
                self._replace_failure_with_inline_summary(failure_mems)
            max_success = max(0, final_budget - effective_failure_slots)
            result = {}
            if success_mems[:max_success]:
                result["successed"] = success_mems[:max_success]
            if failure_mems[:effective_failure_slots]:
                result["failed"] = failure_mems[:effective_failure_slots]
            return result

        def _metadata_dict(mem: dict) -> dict:
            md = mem.get("metadata") or {}
            if isinstance(md, dict):
                return md
            extra = getattr(md, "model_extra", None)
            return extra if isinstance(extra, dict) else {}

        def _diversity_key(mem: dict):
            md = _metadata_dict(mem)
            task_id = md.get("task_id", md.get("sample_index"))
            task_key = (str(md.get("source_benchmark", "")), str(task_id)) if task_id is not None else None
            content = mem.get("content") or md.get("full_content") or ""
            content_key = " ".join(str(content).lower().split())
            return task_key, content_key

        def _dedupe(items: List[dict], limit: int) -> List[dict]:
            selected, seen_tasks, seen_content = [], set(), set()
            for mem in items:
                task_key, content_key = _diversity_key(mem)
                if task_key is not None and task_key in seen_tasks:
                    continue
                if content_key and content_key in seen_content:
                    continue
                selected.append(mem)
                if task_key is not None:
                    seen_tasks.add(task_key)
                if content_key:
                    seen_content.add(content_key)
                if len(selected) >= limit:
                    break
            return selected

        selected_success = list(processed_mems.get("successed", []))
        selected_failed = list(processed_mems.get("failed", []))
        if self._failure_summary_independent_pool and candidate_mems:
            candidate_buckets = self.process_retrieve_mems(candidate_mems)
            success_pool = list(candidate_buckets.get("successed", []))
            failure_pool = [
                mem for mem in candidate_buckets.get("failed", [])
                if float(mem.get("similarity", 0.0) or 0.0) >= self._failure_summary_min_similarity
            ]
        else:
            success_pool = selected_success
            failure_pool = selected_failed

        failure_mems = _dedupe(failure_pool, effective_failure_slots)
        min_success = min(final_budget, self._failure_summary_min_success)

        # Select successful exemplars first, then prevent an FS source task from
        # consuming a second slot. This is especially important for DB snapshots
        # that contain success/failure memories for the same task across epochs.
        provisional_success_target = max(0, final_budget - (effective_failure_slots if failure_mems else 0))
        success_mems = _dedupe(success_pool, provisional_success_target or final_budget)
        success_keys = {_diversity_key(mem)[0] for mem in success_mems}
        failure_mems = [
            mem for mem in failure_mems
            if _diversity_key(mem)[0] is None or _diversity_key(mem)[0] not in success_keys
        ]

        # Structured DB summaries require compatible evidence in the selected
        # Region. Try later failure candidates when the highest-sim failure has
        # no matching Region × SQL-signature evidence.
        if (getattr(self, "task", "") == "db"
                and getattr(self, "_failure_summary_db_structured", False)
                and failure_pool):
            structured_pick = None
            evidence_candidates = [
                candidate for candidate in failure_pool
                if _diversity_key(candidate)[0] not in success_keys
            ]
            # Source selection is deduplicated, but evidence aggregation keeps
            # distinct historical attempts even when their normalized reflection
            # text is identical; frequency is the evidence signal.
            compatible_candidates = _dedupe(evidence_candidates, len(evidence_candidates))
            # Mature path: Region x SQL-signature summary for failures that have
            # already received utility feedback and therefore have membership.
            for candidate in compatible_candidates:
                probe = [dict(candidate)]
                self._replace_failure_with_db_structured_summary(
                    probe, target_skill_list or []
                )
                if probe[0].get("_db_structured_failure_summary"):
                    structured_pick = probe[0]
                    break
            # Cold-start path (ALFWorld-like forced failure retrieval): aggregate
            # compatible retrieved failures without Region membership. The chosen
            # source ID remains in retrieved_ids_list, receives the current reward,
            # and enters utility Region topology only after that real feedback.
            if structured_pick is None and compatible_candidates:
                probe = [dict(compatible_candidates[0])]
                self._replace_failure_with_inline_db_structured_summary(
                    probe, evidence_candidates, target_skill_list or []
                )
                if probe[0].get("_db_structured_failure_summary"):
                    structured_pick = probe[0]
            failure_mems = [structured_pick] if structured_pick is not None else []

        want_failure = bool(failure_mems)
        max_success = final_budget - (effective_failure_slots if want_failure else 0)
        success_target = max_success if want_failure else final_budget
        success_mems = _dedupe(success_pool, success_target)

        if want_failure and len(success_mems) < min_success:
            failure_mems = []
            success_mems = _dedupe(success_pool, final_budget)
        elif want_failure:
            failure_mems = failure_mems[: max(0, final_budget - len(success_mems))]

        if failure_mems:
            mode = getattr(self, "_failure_summary_mode", "inline")
            use_db_structured = (
                getattr(self, "task", "") == "db"
                and getattr(self, "_failure_summary_db_structured", False)
            )
            if use_db_structured:
                if not failure_mems[0].get("_db_structured_failure_summary"):
                    self._replace_failure_with_db_structured_summary(
                        failure_mems, target_skill_list or []
                    )
                # Important: do not fall through to the legacy Region summary;
                # it would overwrite the structured content while retaining the marker.
            elif mode == "region":
                self._replace_failure_with_region_summary(failure_mems)
            else:
                self._replace_failure_with_inline_summary(failure_mems)
            failure_mems = _dedupe(failure_mems, effective_failure_slots)

        result = {}
        if success_mems:
            result["successed"] = success_mems[:final_budget]
        remaining = final_budget - len(result.get("successed", []))
        if failure_mems and remaining > 0:
            result["failed"] = failure_mems[:remaining]
        return result

    @staticmethod
    def _argmax_region_for_memory(region_manager, mem_id):
        """Return the memory's highest-weight region, tolerating old snapshots."""
        if region_manager is None or not getattr(region_manager, "regions", None) or not mem_id:
            return None
        weights = getattr(region_manager, "membership_weights", {}).get(mem_id)
        if weights is None:
            return None
        values = np.asarray(weights, dtype=float).reshape(-1)
        n = min(len(values), len(region_manager.regions))
        if n <= 0 or not np.isfinite(values[:n]).any():
            return None
        safe_values = np.where(np.isfinite(values[:n]), values[:n], -np.inf)
        return region_manager.regions[int(np.argmax(safe_values))]

    def _replace_failure_with_region_summary(self, failed_mems: List[dict]) -> None:
        """Replace failure content with its soft-argmax region's failure summary."""
        rm = getattr(self.memory_service, "region_manager", None)

        for fm in failed_mems:
            mem_id = fm.get("memory_id")
            region = self._argmax_region_for_memory(rm, mem_id)
            summary = region.failure_summary if region and getattr(region, "failure_summary", None) else ""
            if summary:
                fm["content"] = summary
                fm["_region_failure_summary"] = True
                fm["_region_summary_region_id"] = region.region_id

    def _replace_failure_with_inline_db_structured_summary(
        self, failed_mems: List[dict], failure_pool: List[dict],
        target_skill_list: List[str],
    ) -> None:
        """Cold-start Structured FS from retrieved failures before Region membership."""
        from memrl.lifelongbench_eval.db_failure_summary import (
            build_structured_db_summary, db_signature,
        )
        target_sig = db_signature(target_skill_list)
        exact_texts, shape_texts = [], []
        for mem in failure_pool:
            md = mem.get("metadata") or {}
            if not isinstance(md, dict):
                extra = getattr(md, "model_extra", None)
                md = extra if isinstance(extra, dict) else {}
            content = mem.get("content") or md.get("full_content") or ""
            if not content:
                continue
            sig = db_signature(md.get("skill_list", []))
            if sig == target_sig:
                exact_texts.append(content)
            elif sig[:2] == target_sig[:2]:
                shape_texts.append(content)
        evidence = exact_texts if len(exact_texts) >= 2 else exact_texts + shape_texts
        if len(evidence) < getattr(self, "_failure_summary_min_evidence", 1):
            return
        summary = build_structured_db_summary(evidence, target_skill_list, top_n=4)
        if not summary:
            return
        for fm in failed_mems:
            fm["content"] = summary
            fm["_region_failure_summary"] = True
            fm["_db_structured_failure_summary"] = True
            fm["_db_structured_cold_start"] = True
            fm["_db_summary_evidence_count"] = len(evidence)
            fm["_db_summary_signature"] = target_sig

    def _replace_failure_with_db_structured_summary(
        self, failed_mems: List[dict], target_skill_list: List[str]
    ) -> None:
        """Build Region × SQL-signature guardrails instead of schema-specific prose."""
        from memrl.lifelongbench_eval.db_failure_summary import (
            build_structured_db_summary, db_signature,
        )

        rm = getattr(self.memory_service, "region_manager", None)
        mem_cache = getattr(self.memory_service, "_mem_cache", {}) or {}
        target_sig = db_signature(target_skill_list)

        for fm in failed_mems:
            region = self._argmax_region_for_memory(rm, fm.get("memory_id"))
            if region is None:
                continue
            exact_texts, shape_texts = [], []
            for mem_id in getattr(region, "member_ids", []) or []:
                mem_obj = mem_cache.get(mem_id)
                if mem_obj is None:
                    continue
                if isinstance(mem_obj, dict):
                    md = mem_obj.get("metadata", {}) or {}
                else:
                    md = getattr(mem_obj, "metadata", {}) or {}
                    extra = getattr(md, "model_extra", None)
                    if isinstance(extra, dict):
                        md = extra
                if not isinstance(md, dict) or bool(md.get("success", False)):
                    continue
                content = md.get("full_content") or ""
                if not content:
                    continue
                sig = db_signature(md.get("skill_list", []))
                if sig == target_sig:
                    exact_texts.append(content)
                elif sig[:2] == target_sig[:2]:
                    shape_texts.append(content)
            # Exact operation+shape+modifier evidence first; back off only to
            # operation+shape within the same Region. Never aggregate the whole Region.
            evidence = exact_texts if len(exact_texts) >= 2 else exact_texts + shape_texts
            if len(evidence) < getattr(self, "_failure_summary_min_evidence", 1):
                continue
            summary = build_structured_db_summary(evidence, target_skill_list, top_n=4)
            if summary:
                fm["content"] = summary
                fm["_region_failure_summary"] = True
                fm["_db_structured_failure_summary"] = True
                fm["_region_summary_region_id"] = region.region_id
                fm["_db_summary_evidence_count"] = len(evidence)
                fm["_db_summary_signature"] = target_sig

    def _replace_failure_with_inline_summary(self, failed_mems: List[dict]) -> None:
        """Aggregate retrieved failure mems into on-the-fly summary."""
        from memrl.service.region_manager import RegionManager

        k = getattr(self, "_failure_summary_inline_k", None)
        mems_to_aggregate = failed_mems[:k] if k else failed_mems

        fields_list = []
        for fm in mems_to_aggregate:
            content = fm.get("content", "")
            if not content:
                continue
            fields = RegionManager._parse_failure_fields(content)
            if fields["failure_mode"] or fields["mistakes"]:
                fields_list.append(fields)

        if fields_list:
            summary = RegionManager._format_failure_summary(fields_list, top_n=3)
            if summary:
                failed_mems[0]["content"] = summary
                failed_mems[0]["_region_failure_summary"] = True

    # ==================== Self-RAG Critique ====================

    def _self_rag_critique(
        self, task_description: str, selected_mems: List[dict], inject_k: int
    ) -> List[dict]:
        """Use LLM to judge relevance of each retrieved memory, discard irrelevant ones."""
        if not selected_mems:
            return []
        numbered = []
        for i, m in enumerate(selected_mems):
            content = m.get("content") or ""
            numbered.append(f"[Memory {i+1}]\n{content[:2000]}")
        critique_prompt = (
            "You are a relevance judge. Given a task description and a list of retrieved memories "
            "from past problem-solving attempts, decide which memories are RELEVANT and could help "
            "solve the current task.\n\n"
            f"Task: {task_description[:2000]}\n\n"
            "Retrieved memories:\n" + "\n\n".join(numbered) + "\n\n"
            "Return ONLY a JSON list of the relevant memory numbers (1-indexed). "
            "If none are relevant, return an empty list: []\n"
            "Example: [1, 3]"
        )
        try:
            resp = self.llm_provider.generate(
                messages=[{"role": "user", "content": critique_prompt}],
                temperature=0.0,
                max_tokens=256,
            )
            import re
            match = re.search(r'\[[\d\s,]*\]', resp or "")
            if match:
                indices = json.loads(match.group())
                filtered = []
                for idx in indices:
                    if 1 <= idx <= len(selected_mems):
                        filtered.append(selected_mems[idx - 1])
                logger.info("[Self-RAG] Critique kept %d/%d memories", len(filtered), len(selected_mems))
                return filtered[:inject_k]
            logger.info("[Self-RAG] Critique returned no valid indices, using all %d memories", len(selected_mems))
        except Exception as e:
            logger.warning("[Self-RAG] Critique failed (%s), using all %d memories", e, len(selected_mems))
        return selected_mems[:inject_k]

    # ==================== Baseline Support ====================

    def _format_reflection_note(self, trajectory: Optional[str], success: bool) -> str:
        """Format a reflection note from a prior trajectory attempt."""
        status = "CORRECT" if success else "INCORRECT"
        traj_text = (trajectory or "")[:3000] or "(no trajectory recorded)"
        return (
            "You attempted this task before.\n"
            f"Result: {status}\n"
            "Previous trajectory:\n"
            f"{traj_text}\n\n"
            "Reflect on mistakes or improvements and solve the task again with a better plan."
        )

    def _sample_single_trajectory_baseline(
        self,
        sample_index: str,
        reflection_note: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Sample a single trajectory without memory retrieval.

        Used by Pass@k and Reflection baselines. Optionally injects a reflection
        note into the system prompt (for Reflection baseline).

        Returns:
            Dict with keys: sample_index, task_description, trajectory, success, steps
        """
        try:
            entry = self.dataset[sample_index]
            task_description = self._task_description_from_entry(entry)

            memory_context = reflection_note or ""
            full_prompt = self._build_llb_full_prompt(
                memory_context=memory_context if memory_context else None
            )

            agent = LanguageModelAgent(
                language_model=self.adapter, system_prompt=full_prompt
            )

            from memrl.lifelongbench_eval.task_wrappers import build_task

            task_obj, _ = build_task(
                task=self.task,
                data_file_path=self.split_file,
                max_round=self.max_steps,
                os_timeout=self.os_timeout,
                kg_sparql_url=self.sparql_url,
                kg_ontology_dir=self.ontology_dir,
                kg_offline_fallback=self.kg_offline_fallback,
            )

            session = Session(task_name=self.task_name, sample_index=sample_index)
            task_obj.reset(session)

            step_count = 0
            while session.sample_status == SampleStatus.RUNNING:
                agent.inference(session)
                task_obj.interact(session)
                step_count += 1
                if step_count > self.max_steps * 2:
                    break

            task_obj.complete(session)
            success = self._session_success(session)
            trajectory = self._session_to_trajectory(session)

            return {
                "sample_index": sample_index,
                "task_description": task_description,
                "trajectory": trajectory or "",
                "success": success,
                "steps": step_count,
            }
        except Exception as e:
            logger.error(f"Baseline trajectory failed for {sample_index}: {e}", exc_info=True)
            return None

    def _sample_from_indices(
        self,
        sample_indices: List[str],
        phase: str = "train",
        custom_dataset: Optional[Dict[str, Any]] = None,
        custom_data_file: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Sample trajectories for given sample indices.

        Args:
            sample_indices: List of sample keys to process
            phase: Phase name ('train', 'eval', etc.)
            custom_dataset: Optional custom dataset to use for evaluation
            custom_data_file: Optional custom data file path for building task_obj

        Returns:
            List of trajectory dictionaries with keys:
                - sample_index: str
                - task_description: str
                - retrieved_memories: List[dict]
                - retrieved_ids: List[str]
                - session: Session object
                - success: bool
                - steps: int (number of rounds)
        """
        completed_trajectories = []

        # Serial execution with optional inter-request delay.
        # Controlled by MEMRL_LLB_REQUEST_INTERVAL env var (seconds, default 0.5).
        # Set to 0 to disable delay; parallelism is removed to avoid API rate limits.
        try:
            request_interval = float(os.environ.get("MEMRL_LLB_REQUEST_INTERVAL", "0.5") or "0")
        except (TypeError, ValueError):
            request_interval = 0.5

        logger.info(
            f"Sampling {len(sample_indices)} trajectories sequentially "
            f"(interval={request_interval:.2f}s)..."
        )

        for idx in tqdm(sample_indices, desc=f"Sampling {phase}"):
            try:
                traj = self._sample_single_trajectory(
                    idx, phase, custom_dataset, custom_data_file
                )
                if traj is not None:
                    completed_trajectories.append(traj)
            except Exception as e:
                logger.error(
                    f"Error sampling trajectory for {idx}: {e}", exc_info=True
                )
            if request_interval > 0:
                time.sleep(request_interval)

        logger.info(
            f"Completed {len(completed_trajectories)}/{len(sample_indices)} trajectories"
        )
        return completed_trajectories

    def _sample_single_trajectory(
        self,
        sample_index: str,
        phase: str = "train",
        custom_dataset: Optional[Dict[str, Any]] = None,
        custom_data_file: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Sample a single trajectory.

        Args:
            sample_index: Sample key
            phase: Phase name
            custom_dataset: Optional custom dataset to use instead of self.dataset
            custom_data_file: Optional custom data file path for building task_obj

        Returns:
            Trajectory dictionary or None if failed
        """
        run_meta = {
            "exp_name": self.exp_name,
            "task": self.task,
            "mode": self.mode,
            "phase": phase,
            "split_file": str(self.split_file),
            "random_seed": int(self.random_seed),
            "max_steps": int(self.max_steps),
            "retrieve_k": int(self.retrieve_k),
            "algorithm": str(self.algorithm),
        }
        cm = (
            self._trace.task(
                sample_index=str(sample_index),
                run_meta=run_meta,
                task_description="",  # filled once we parse entry
            )
            if self._trace is not None
            else contextlib.nullcontext(None)
        )

        with cm as trace_ctx:
            try:
                # Use custom dataset if provided, otherwise use self.dataset
                dataset = custom_dataset if custom_dataset is not None else self.dataset
                data_file = (
                    custom_data_file if custom_data_file is not None else self.split_file
                )

                # Get task entry
                entry = dataset[sample_index]
                task_description = self._task_description_from_entry(entry)

                if trace_ctx is not None:
                    trace_ctx.task_description = task_description

                # Retrieve memories
                retrieved_mems = []
                topk_queries = []
                processed_mems = {}
                memory_context = ""

                if self.memory_service is not None:
                    try:
                        # Keep retrieval threshold aligned with memory_rl:
                        # prefer rl_config.sim_threshold; fall back to rl_config.tau; else 0.0.
                        thr = (
                            getattr(
                                self.rl_config,
                                "sim_threshold",
                                getattr(self.rl_config, "tau", 0.0),
                            )
                            if self.rl_config
                            else 0.0
                        )
                        # Compute target_subtask for region-aware retrieval
                        _retrieve_kwargs = {}
                        if hasattr(self.memory_service, "region_manager"):
                            from memrl.configs.task_hierarchy import (
                                get_primary_subtask, get_db_multi_axis_subtasks,
                            )
                            _target_subtask = get_primary_subtask(
                                f"llb_{self.task}",
                                {"skill_list": entry.get("skill_list", [])},
                            )
                            _retrieve_kwargs["target_subtask"] = _target_subtask
                            if (
                                self.task == "db"
                                and os.environ.get("MEMRL_DB_MULTI_AXIS", "0").lower()
                                in {"1", "true", "yes"}
                            ):
                                _retrieve_kwargs["target_subtask_weights"] = (
                                    get_db_multi_axis_subtasks(entry.get("skill_list", []))
                                )
                            _retrieve_kwargs["audit_context"] = {
                                "epoch": int(getattr(self.memory_service, "_current_epoch", 0) or 0),
                                "sample_index": str(sample_index),
                                "phase": "eval" if custom_dataset is not None else "train",
                            }

                        results = self.memory_service.retrieve_query(
                            task_description=task_description,
                            k=self.retrieve_k,
                            threshold=thr,
                            **_retrieve_kwargs,
                        )
                        # retrieve_query returns tuple: (dict with 'selected' key, topk_queries)
                        retrieval_payload = {}
                        if isinstance(results, tuple):
                            retrieval_payload = results[0] or {}
                            retrieved_mems = retrieval_payload.get("selected", [])
                            topk_queries = results[1]
                        else:
                            retrieved_mems = []
                            topk_queries = []

                        # Self-RAG: LLM critique to filter irrelevant memories
                        if self.self_rag and retrieved_mems:
                            retrieved_mems = self._self_rag_critique(
                                task_description, retrieved_mems, self.self_rag_inject_k
                            )

                        # Process and categorize memories
                        processed_mems = self.process_retrieve_mems(retrieved_mems)

                        # Inject failure summary if configured
                        if self._failure_summary_n_slots > 0 and processed_mems:
                            processed_mems = self._inject_failure_summary(
                                processed_mems,
                                task_description,
                                candidate_mems=list(retrieval_payload.get("candidates", []) or retrieved_mems),
                                target_skill_list=list(entry.get("skill_list", []) or []),
                            )

                        # Format memory context from categorized memories
                        if processed_mems:
                            memory_context = self._format_memory_context(processed_mems)

                        if hasattr(self.memory_service, "append_retrieval_audit"):
                            self.memory_service.append_retrieval_audit(
                                task_description, retrieval_payload, processed_mems
                            )

                        # DEBUG: confirm memory is actually injected (content non-empty)
                        n_succ = len(processed_mems.get("successed", []))
                        n_fail = len(processed_mems.get("failed", []))
                        content_lens = [
                            len(m.get("content") or "")
                            for mlist in processed_mems.values()
                            for m in mlist
                        ]
                        mc_len = len(memory_context) if memory_context else 0
                        logger.info(
                            f"[MEMORY INJECT] sample={sample_index} | "
                            f"success_mems={n_succ} fail_mems={n_fail} | "
                            f"content_lens={content_lens[:5]} | "
                            f"memory_context_len={mc_len} | "
                            f"preview={memory_context[:150] if memory_context else '(empty)'}"
                        )

                        if trace_ctx is not None:
                            from memrl.trace.llb_jsonl import summarize_text

                            def _mem_summary(m: Dict[str, Any]) -> Dict[str, Any]:
                                md = m.get("metadata")
                                md_summary = None
                                try:
                                    if hasattr(md, "model_dump"):
                                        md_summary = md.model_dump()
                                    elif isinstance(md, dict):
                                        md_summary = dict(md)
                                    elif md is not None:
                                        md_summary = {"repr": str(md)}
                                except Exception:
                                    md_summary = {"repr": str(md)}

                                # Align with retrieval task_id de-dup (task_id -> sample_index -> id).
                                # NOTE: task_id can legally be 0, so avoid truthiness-based fallbacks.
                                task_id = extract_task_id(md_summary if isinstance(md_summary, dict) else None)

                                return {
                                    "memory_id": m.get("memory_id"),
                                    "task_id": (str(task_id) if task_id is not None else None),
                                    "similarity": float(
                                        m.get("similarity", 0.0) or 0.0
                                    ),
                                    "similarity_z": float(
                                        m.get("similarity_z", 0.0) or 0.0
                                    ),
                                    "q_estimate": float(m.get("q_estimate", 0.0) or 0.0),
                                    "q_z": float(m.get("q_z", 0.0) or 0.0),
                                    "score": float(m.get("score", 0.0) or 0.0),
                                    "base_score": float(m.get("base_score", m.get("score", 0.0)) or 0.0),
                                    "region_value": (
                                        float(m.get("region_value"))
                                        if m.get("region_value") is not None else None
                                    ),
                                    "region_advantage": (
                                        float(m.get("region_advantage"))
                                        if m.get("region_advantage") is not None else None
                                    ),
                                    "region_failure_summary": bool(m.get("_region_failure_summary")),
                                    "region_summary_region_id": m.get("_region_summary_region_id"),
                                    "db_structured_failure_summary": bool(m.get("_db_structured_failure_summary")),
                                    "db_summary_evidence_count": m.get("_db_summary_evidence_count"),
                                    "db_structured_cold_start": bool(m.get("_db_structured_cold_start")),
                                    "db_summary_signature": m.get("_db_summary_signature"),
                                    "metadata": md_summary,
                                }

                            trace_ctx.retrieval = {
                                "params": {
                                    "k_retrieve": int(self.retrieve_k),
                                    "threshold": float(thr),
                                    "rl_topk": int(getattr(self.rl_config, "topk", 0) or 0)
                                    if self.rl_config
                                    else None,
                                    "dedup_by_task_id": bool(
                                        getattr(self.memory_service, "dedup_by_task_id", False)
                                    )
                                    if self.memory_service is not None
                                    else None,
                                    "weight_sim": float(
                                        getattr(self.rl_config, "weight_sim", 0.0) or 0.0
                                    )
                                    if self.rl_config
                                    else None,
                                    "weight_q": float(
                                        getattr(self.rl_config, "weight_q", 0.0) or 0.0
                                    )
                                    if self.rl_config
                                    else None,
                                },
                                "topk_queries": [
                                    {
                                        "query": summarize_text(str(q)),
                                        "similarity": float(sim),
                                    }
                                    for (q, sim) in (topk_queries or [])
                                ],
                                "selected_memories_by_bucket": {
                                    str(k): [_mem_summary(m) for m in v]
                                    for k, v in (processed_mems or {}).items()
                                },
                            }
                    except Exception as e:
                        logger.warning(f"Memory retrieval failed for {sample_index}: {e}")

                # Create agent with memory context
                full_prompt = self._build_llb_full_prompt(memory_context=memory_context)
                if trace_ctx is not None:
                    trace_ctx.set_full_system_prompt(full_prompt)
                agent = LanguageModelAgent(
                    language_model=self.adapter, system_prompt=full_prompt
                )

                # Create new task instance for this sample (avoid state pollution)
                from memrl.lifelongbench_eval.task_wrappers import build_task

                task_obj, _ = build_task(
                    task=self.task,
                    data_file_path=data_file,
                    max_round=self.max_steps,
                    os_timeout=self.os_timeout,
                    kg_sparql_url=self.sparql_url,
                    kg_ontology_dir=self.ontology_dir,
                    kg_offline_fallback=self.kg_offline_fallback,
                )

                # Create session
                session = Session(task_name=self.task_name, sample_index=sample_index)

                # Reset task
                task_obj.reset(session)

                # Run inference loop
                step_count = 0
                while session.sample_status == SampleStatus.RUNNING:
                    agent.inference(session)
                    task_obj.interact(session)
                    step_count += 1

                    # Safety check
                    if step_count > self.max_steps * 2:
                        logger.warning(
                            f"Sample {sample_index} exceeded max steps, terminating"
                        )
                        break

                # Complete the session
                task_obj.complete(session)

                # Check success
                success = self._session_success(session)
                # Extract failure evidence (empty for successes) to feed reflection.
                failure_reason = "" if success else self._session_failure_reason(session)

                # Convert session to trajectory string
                trajectory = self._session_to_trajectory(session)
                if not trajectory:
                    trajectory = ""  # Fallback to empty string

                if trace_ctx is not None:
                    trace_ctx.interaction = {
                        "chat_history_final": self._session_to_chat_messages(session),
                    }
                    trace_ctx.outcome = {
                        "success": bool(success),
                        "steps": int(step_count),
                    }

                return {
                    "sample_index": sample_index,
                    "task_description": task_description,
                    "trajectory": trajectory,  # String for add_memories
                    "retrieved_mems": processed_mems,  # Categorized selected memories
                    "retrieved_queries": (
                        topk_queries if topk_queries else [(task_description, 1.0)]
                    ),
                    "session": session,
                    "success": success,
                    "steps": step_count,
                    "skill_list": entry.get("skill_list", []),
                    "failure_reason": failure_reason,
                }

            except Exception as e:
                if trace_ctx is not None:
                    trace_ctx.error = {"type": type(e).__name__, "message": str(e)}
                logger.error(
                    f"Failed to sample trajectory for {sample_index}: {e}", exc_info=True
                )
                return None

    def _format_memory_context(
        self, processed_mems: Dict[str, List[dict]], budget_tokens: Optional[int] = None
    ) -> str:
        return format_llb_memory_context(
            processed_mems, task=self.task, budget_tokens=budget_tokens
        )

    def _session_to_trajectory(self, session: Any) -> Optional[str]:
        """
        将 LLB Session 的 chat_history 序列化为可用于记忆构建的 trajectory 文本。

        返回:
            多行字符串，每行形如 "<role>: <content>"；若无法提取则返回 None。
        """
        if session is None:
            return None

        # 兼容属性形式与 dict 形式
        ch = getattr(session, "chat_history", None)
        if ch is None and isinstance(session, dict):
            ch = session.get("chat_history")

        # 优先处理 LLB 自带的 ChatHistory 类型（带有 get_value_str / get_value_length）。
        # 该类型禁止直接访问 .value 属性，因此不能用 hasattr(x, "value") / getattr(x, "value")。
        try:
            from src.typings import ChatHistory as _LLBChatHistory, Role as _LLBRole  # type: ignore
        except Exception:
            _LLBChatHistory = None  # type: ignore
            _LLBRole = None  # type: ignore

        if _LLBChatHistory is not None and isinstance(ch, _LLBChatHistory):
            try:
                role_dict = {}
                if _LLBRole is not None:
                    try:
                        role_dict = {
                            _LLBRole.USER: "user",
                            _LLBRole.AGENT: "assistant",
                        }
                    except Exception:
                        role_dict = {}
                # 当 role_dict 为空时，LLB 的实现仍会正常工作，只是 role 文本会保持原样。
                traj = ch.get_value_str(
                    role_dict=role_dict, start_index=None, end_index=None
                )
            except Exception:
                traj = None
            return traj or None

        # 兜底：兼容老版本或非 LLB 的结构（list / dict）。
        msgs: Optional[list[Any]] = None
        if isinstance(ch, dict):
            v = ch.get("value") or ch.get("messages")
            if isinstance(v, list):
                msgs = v
        elif isinstance(ch, list):
            msgs = ch

        if not msgs:
            return None

        lines: list[str] = []
        for m in msgs:
            if isinstance(m, dict):
                role = m.get("role") or m.get("speaker") or "unknown"
                content = m.get("content") or m.get("text") or ""
            else:
                role, content = "unknown", str(m)
            lines.append(f"{role}: {content}")

        return "\n".join(lines)

    def _session_to_message_list(self, session: Session) -> List[str]:
        """Convert LLB Session's chat_history to list of message strings.

        Returns:
            List of formatted messages like ['role: content', ...]
        """
        if session is None:
            return []

        ch = getattr(session, "chat_history", None)
        if ch is None and isinstance(session, dict):
            ch = session.get("chat_history")

        if not ch:
            return []

        messages = []
        for msg in ch:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            messages.append(f"{role}: {content}")

        return messages

    def _task_description_from_entry(self, entry: Dict[str, Any]) -> str:
        if self.task in ("db_bench", "os_interaction", "db", "os"):
            return entry.get("instruction", "")
        return entry.get("question", "")

    def _session_success(self, session: Session) -> bool:
        """Check if session was successful."""
        if session is None:
            return False
        outcome = getattr(session, "evaluation_record", None)
        if outcome:
            outcome = getattr(outcome, "outcome", None)
            return outcome == SessionEvaluationOutcome.CORRECT
        return False

    def _session_failure_reason(self, session: Session) -> str:
        """Extract failure evidence from a failed LLB session.

        Reflection quality depends on telling the LLM *why* the task failed.
        Without this, the LLM sees only a (often syntactically valid) trajectory
        and wrongly concludes "no mistakes". We surface the concrete signals LLB
        already records: sample_status (e.g. protocol-validation failure, step
        limit), finish_reason (free-text detail), evaluation outcome, and any
        numeric detail. Returns "" when no useful signal is available.
        """
        if session is None:
            return ""
        parts: List[str] = []

        status = getattr(session, "sample_status", None)
        # SampleStatus is a StrEnum; str() gives the wire value like
        # "task_limit_reached" / "agent_validation_failed".
        #
        # The agent_validation_failed hint is protocol-specific. Under the v2
        # reflection switch we word it per-task (OS uses Act: bash / Act: finish;
        # DB uses Action: Operation / Action: Answer). Legacy stays byte-identical
        # (DB-worded) for reproducibility. This hint feeds eval_error -> reflection,
        # so a wrong protocol name here actively misleads OS reflections.
        import os as _os
        _reflect_variant = (
            _os.environ.get("MEMRL_LLB_REFLECTION_PROMPT", "legacy") or "legacy"
        ).strip().lower()
        _is_os_task = str(getattr(self, "task", "")).strip().lower() in (
            "os", "os_interaction"
        )
        if _reflect_variant in ("v2", "corrected", "new") and _is_os_task:
            _validation_hint = (
                "The agent response did NOT follow the required OS interaction "
                "protocol (must use 'Act: bash' with a ```bash``` code block to run "
                "commands, and 'Act: finish' to end)."
            )
        else:
            _validation_hint = (
                "The agent response did NOT follow the required interaction "
                "protocol/output format (e.g. missing the exact 'Action: "
                "Operation'/'Action: Answer' directive)."
            )
        status_map = {
            "completed": "The interaction completed but the final answer was judged INCORRECT by the evaluator.",
            "task_limit_reached": "The agent ran out of interaction steps (step limit reached) before producing a correct answer.",
            "agent_validation_failed": _validation_hint,
            "task_environment_error": "The task environment reported an error while executing the agent's action.",
            "agent_context_limit": "The conversation exceeded the model context limit.",
        }
        if status is not None:
            status_s = str(status)
            hint = status_map.get(status_s)
            parts.append(f"Sample status: {status_s}." + (f" {hint}" if hint else ""))

        outcome_rec = getattr(session, "evaluation_record", None)
        if outcome_rec is not None:
            outcome = getattr(outcome_rec, "outcome", None)
            if outcome is not None:
                parts.append(f"Evaluation outcome: {str(outcome)}.")
            detail = getattr(outcome_rec, "detail_dict", None)
            if detail:
                try:
                    parts.append(f"Evaluation detail: {json.dumps(detail, ensure_ascii=False)}.")
                except Exception:
                    parts.append(f"Evaluation detail: {detail}.")

        finish_reason = getattr(session, "finish_reason", None)
        if finish_reason:
            parts.append(f"Finish reason: {str(finish_reason)[:300]}")

        return " ".join(parts).strip()

    def _add_to_memid_pair_fifo(
        self,
        memid_pair: OrderedDict,
        key: str,
        values: List[str],
        max_capacity: int = 10000,
    ):
        """
        Add memory reference to memid_pair with FIFO eviction policy.

        Args:
            memid_pair: OrderedDict storing memory references
            key: New memory ID
            values: List of referenced memory IDs
            max_capacity: Maximum number of keys to keep (default 10000)
        """
        if key not in memid_pair:
            # Check capacity before adding
            if len(memid_pair) >= max_capacity:
                # Remove oldest entry (FIFO)
                oldest_key = next(iter(memid_pair))
                removed_value = memid_pair.pop(oldest_key)
                logger.debug(
                    f"[FIFO] Evicted oldest memid_pair entry: {oldest_key} (had {len(removed_value)} refs)"
                )

            memid_pair[key] = []

        # Extend with new values
        memid_pair[key].extend(values)

        # Move to end (mark as recently used)
        memid_pair.move_to_end(key)

        return memid_pair

    def _analyze_and_report_results(self):
        """
        Analyzes and reports the final results for both training and evaluation,
        including success rates and average steps for all phases.
        """
        if not self.results_log:
            logger.warning("No results were logged. Cannot perform analysis.")
            return

        logger.info(
            "\n" + "#" * 20 + " FULL EXPERIMENT FINISHED - FINAL RESULTS " + "#" * 20
        )
        results_df = pd.DataFrame(self.results_log)

        # Backwards-compatible schema handling:
        # Older / different logging paths may not have included a 'mode' field.
        # The analysis below expects it.
        if "mode" not in results_df.columns:
            results_df["mode"] = "train"

        train_modes = {"build", "update", "train", "test"}

        # --- Training Performance ---
        train_df = results_df[results_df["mode"].isin(train_modes)]
        if not train_df.empty:
            overall_success_rate = train_df["success"].mean()
            logger.info("\n--- Training Performance (on Train Set) ---")
            logger.info(f"Total Training Trajectories: {len(train_df)}")
            logger.info(f"Overall Success Rate: {overall_success_rate:.2%}")

            section_performance = (
                train_df.groupby("section")
                .agg(success_rate=("success", "mean"), avg_steps=("steps", "mean"))
                .reset_index()
            )
            logger.info("\n>>> Training Performance by Section <<<")
            print(
                section_performance.to_string(
                    index=False, formatters={"success_rate": "{:.2%}".format}
                )
            )

        # --- Evaluation Performance ---
        eval_df = results_df[~results_df["mode"].isin(train_modes)]
        if not eval_df.empty:
            logger.info("\n--- Evaluation Performance Summary ---")

            # Pivot table for Success Rate on Eval Sets
            logger.info("\n>>> Success Rate (%) by Evaluation Set <<<")
            # In eval logs, the 'success' column already holds the rate
            eval_success_summary = eval_df.pivot_table(
                index="after_section", columns="mode", values="success"
            )
            with pd.option_context("display.float_format", "{:.2%}".format):
                print(eval_success_summary)

            # Pivot table for Average Steps on Success on Eval Sets
            logger.info("\n>>> Average Steps on Success by Evaluation Set <<<")
            # In eval logs, the 'steps' column holds the average steps on success
            eval_steps_summary = eval_df.pivot_table(
                index="after_section", columns="mode", values="steps"
            )
            with pd.option_context("display.float_format", "{:.2f}".format):
                print(eval_steps_summary)

        # --- Save results to a CSV file ---
        log_dir = self.root / "logs"
        log_dir.mkdir(exist_ok=True)
        results_csv_path = (
            log_dir
            / f"experiment_results_{self.exp_name}_{time.strftime('%Y%m%d-%H%M%S')}.csv"
        )
        results_df.to_csv(results_csv_path, index=False)
        logger.info(f"\nDetailed results saved to: {results_csv_path}")

    def _evaluate(
        self, eval_dataset: Dict[str, Any], eval_type: str, after_section: int
    ) -> None:
        """Run evaluation on validation or test set.

        Args:
            eval_dataset: Dictionary of evaluation samples
            eval_type: String identifier ('Validation' or 'Test')
            after_section: Current section number for logging
        """
        if not eval_dataset:
            logger.warning(f"No {eval_type} dataset available for evaluation.")
            return

        logger.info(
            f"\n--- Starting {eval_type} Evaluation (after Section {after_section}) ---"
        )

        # Get all sample keys
        eval_keys = sorted(list(eval_dataset.keys()), key=lambda x: str(x))
        logger.info(f"Evaluating on {len(eval_keys)} {eval_type.lower()} samples...")

        # Split into mini-batches
        num_mini_batches = int(np.ceil(len(eval_keys) / self.batch_size))
        eval_mini_batches = [
            eval_keys[i * self.batch_size : (i + 1) * self.batch_size]
            for i in range(num_mini_batches)
        ]

        # Sample trajectories from all mini-batches using custom dataset
        # This creates separate task_obj instances for evaluation
        eval_trajectories = []
        for mini_batch_idx, mini_batch_keys in enumerate(
            tqdm(eval_mini_batches, desc=f"{eval_type} Evaluation")
        ):
            collected_trajs = self._sample_from_indices(
                sample_indices=mini_batch_keys,
                phase="eval",
                custom_dataset=eval_dataset,
                custom_data_file=self.valid_file,
            )
            eval_trajectories.extend(collected_trajs)

        if not eval_trajectories:
            logger.warning(f"No trajectories collected during {eval_type} evaluation.")
            self.writer.add_scalar(
                f"Evaluation/Success_Rate/{eval_type}", 0.0, after_section
            )
            self.writer.add_scalar(
                f"Evaluation/Avg_Steps/{eval_type}", 0.0, after_section
            )
            return

        # Calculate metrics
        successes = sum(1 for traj in eval_trajectories if traj["success"])
        success_rate = successes / len(eval_trajectories) if eval_trajectories else 0.0
        avg_steps = np.mean([traj["steps"] for traj in eval_trajectories])

        logger.info(
            f"--- {eval_type} Evaluation Complete (after Section {after_section}) ---"
        )
        logger.info(
            f"Success Rate: {success_rate:.2%} ({successes}/{len(eval_trajectories)})"
        )
        logger.info(f"Average Steps: {avg_steps:.2f}")

        # Log to TensorBoard
        self.writer.add_scalar(
            f"Evaluation/Success_Rate/{eval_type}", success_rate, after_section
        )
        self.writer.add_scalar(
            f"Evaluation/Avg_Steps/{eval_type}", avg_steps, after_section
        )

        # Log to results
        self.results_log.append(
            {
                "section": f"eval_s{after_section}",
                "after_section": after_section,
                "mode": eval_type,
                "success": success_rate,
                "steps": avg_steps,
            }
        )

        # Log token usage
        self._log_token_usage(after_section)

    def _evaluate_single(
        self, eval_dataset: Dict[str, Any], eval_type: str, after_section: int
    ) -> Dict[str, bool]:
        """Run one evaluation pass and return per-sample results."""
        eval_keys = sorted(list(eval_dataset.keys()), key=lambda x: str(x))
        num_mini_batches = int(np.ceil(len(eval_keys) / self.batch_size))
        eval_mini_batches = [
            eval_keys[i * self.batch_size : (i + 1) * self.batch_size]
            for i in range(num_mini_batches)
        ]
        eval_trajectories = []
        for mini_batch_keys in tqdm(eval_mini_batches, desc=f"{eval_type} eval"):
            collected_trajs = self._sample_from_indices(
                sample_indices=mini_batch_keys,
                phase="eval",
                custom_dataset=eval_dataset,
                custom_data_file=self.valid_file,
            )
            eval_trajectories.extend(collected_trajs)
        per_sample = {}
        for traj in eval_trajectories:
            sid = str(traj.get("sample_index", ""))
            per_sample[sid] = bool(traj["success"])
        return per_sample

    def _evaluate_multi(
        self, eval_dataset: Dict[str, Any], eval_type: str, after_section: int
    ) -> None:
        """Run multiple independent eval passes, compute mean+-CI and CSR."""
        if not eval_dataset:
            logger.warning(f"No {eval_type} dataset for multi-eval.")
            return
        n_runs = self.eval_runs
        logger.info(
            f"\n--- Multi-Eval: {n_runs} runs (1x temp=0.0 + {n_runs-1}x temp={self.eval_temperature}) "
            f"(after Section {after_section}) ---"
        )
        orig_temp = getattr(self.llm_provider, "default_temperature", 0.0)
        per_run_results = []
        try:
            for run_idx in range(1, n_runs + 1):
                # First run: deterministic (temp=0.0); rest: stochastic
                if run_idx == 1:
                    self.llm_provider.default_temperature = 0.0
                    run_label = "deterministic"
                else:
                    self.llm_provider.default_temperature = self.eval_temperature
                    run_label = f"temp={self.eval_temperature}"
                logger.info(f"  Eval run {run_idx}/{n_runs} ({run_label})...")
                per_sample = self._evaluate_single(eval_dataset, eval_type, after_section)
                sr = sum(per_sample.values()) / len(per_sample) if per_sample else 0.0
                logger.info(f"  Run {run_idx} SR: {sr:.2%} ({sum(per_sample.values())}/{len(per_sample)})")
                per_run_results.append({"run": run_idx, "sr": sr, "temperature": self.llm_provider.default_temperature, "results": per_sample})
        finally:
            self.llm_provider.default_temperature = orig_temp

        srs = [r["sr"] for r in per_run_results]
        mean_sr = float(np.mean(srs))
        std_sr = float(np.std(srs, ddof=1)) if len(srs) > 1 else 0.0
        if len(srs) > 1:
            from scipy import stats as _stats
            t_val = _stats.t.ppf(0.975, df=len(srs) - 1)
            margin = t_val * std_sr / np.sqrt(len(srs))
            ci_95 = [mean_sr - margin, mean_sr + margin]
        else:
            ci_95 = [mean_sr, mean_sr]

        all_samples = set()
        for r in per_run_results:
            all_samples.update(r["results"].keys())
        solved = {sid for sid in all_samples if any(r["results"].get(sid, False) for r in per_run_results)}
        csr = len(solved) / len(all_samples) if all_samples else 0.0

        summary = {
            "mean_sr": round(mean_sr, 4),
            "std_sr": round(std_sr, 4),
            "ci_95": [round(ci_95[0], 4), round(ci_95[1], 4)],
            "csr": round(csr, 4),
        }
        logger.info(
            f"--- Multi-Eval Summary (after Section {after_section}) ---\n"
            f"  Mean SR: {mean_sr:.2%} +- {std_sr:.2%}  CI95: [{ci_95[0]:.2%}, {ci_95[1]:.2%}]\n"
            f"  CSR: {csr:.2%} ({len(solved)}/{len(all_samples)})"
        )
        self.writer.add_scalar(f"Evaluation/Mean_SR/{eval_type}", mean_sr, after_section)
        self.writer.add_scalar(f"Evaluation/CSR/{eval_type}", csr, after_section)

        result_payload = {
            "after_section": after_section,
            "eval_runs": n_runs,
            "eval_temperature": self.eval_temperature,
            "per_run": per_run_results,
            "summary": summary,
        }
        self.ck_dir.mkdir(parents=True, exist_ok=True)
        result_path = self.ck_dir / f"eval_results_section_{after_section}.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result_payload, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"  Saved eval results to: {result_path}")

        self.results_log.append({
            "section": f"eval_s{after_section}",
            "after_section": after_section,
            "mode": f"{eval_type}_multi",
            "success": mean_sr,
            "steps": 0.0,
        })

    # ==================== Baseline Runners ====================

    def _run_passk_baseline(self) -> None:
        """Run Pass@k baseline: k independent attempts per task, any success counts."""
        all_keys = sorted(self.dataset.keys(), key=str)
        total_tasks = len(all_keys)
        solved: Set[str] = set()
        summary: List[Dict[str, Any]] = []

        self.ck_dir.mkdir(parents=True, exist_ok=True)
        result_path = self.ck_dir / "baseline_passk_results.jsonl"
        summary_path = self.ck_dir / "baseline_passk_summary.json"
        state_path = self.ck_dir / "baseline_passk_state.json"

        start_round = 1
        if state_path.exists():
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                solved = {str(x) for x in state.get("solved", [])}
                summary = state.get("summary", [])
                last_round = int(state.get("last_round", 0))
                if state.get("round_complete", False):
                    start_round = last_round + 1
                else:
                    start_round = last_round
                logger.info(
                    "Resuming pass@k from round %d (%d/%d solved)",
                    start_round, len(solved), total_tasks,
                )
            except Exception:
                logger.warning("Failed to load pass@k state, starting fresh", exc_info=True)

        if start_round > self.baseline_k:
            logger.info("pass@k already completed (last round %d).", start_round - 1)
            return

        try:
            request_interval = float(os.environ.get("MEMRL_LLB_REQUEST_INTERVAL", "0.5") or "0")
        except (TypeError, ValueError):
            request_interval = 0.5

        for round_idx in range(start_round, self.baseline_k + 1):
            logger.info("Starting pass@k round %d/%d (solved so far: %d/%d)",
                        round_idx, self.baseline_k, len(solved), total_tasks)

            for sample_index in tqdm(all_keys, desc=f"pass@k round {round_idx}"):
                if sample_index in solved:
                    continue
                traj = self._sample_single_trajectory_baseline(sample_index)
                if traj is not None:
                    if traj["success"]:
                        solved.add(str(sample_index))
                    payload = {
                        "round": round_idx,
                        "baseline": "passk",
                        "sample_index": sample_index,
                        "success": traj["success"],
                        "steps": traj["steps"],
                    }
                    with open(result_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
                if request_interval > 0:
                    time.sleep(request_interval)

            cum_acc = len(solved) / total_tasks if total_tasks > 0 else 0.0
            summary.append({
                "round": round_idx,
                "cum_acc": cum_acc,
                "solved": len(solved),
                "total": total_tasks,
            })
            logger.info("pass@k round %d cumulative acc: %.2f%% (%d/%d)",
                        round_idx, cum_acc * 100, len(solved), total_tasks)
            self.writer.add_scalar("Baseline/PassK_Cumulative_Acc", cum_acc, round_idx)

            with open(state_path, "w", encoding="utf-8") as f:
                json.dump({
                    "last_round": round_idx,
                    "round_complete": True,
                    "solved": sorted(solved),
                    "summary": summary,
                }, f, ensure_ascii=False)

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info("pass@k baseline complete. Final acc: %.2f%%",
                    summary[-1]["cum_acc"] * 100 if summary else 0.0)

    def _run_reflection_baseline(self) -> None:
        """Run Reflection/Self-RAG baseline: retry with reflection of prior attempt."""
        all_keys = sorted(self.dataset.keys(), key=str)
        total_tasks = len(all_keys)
        solved: Set[str] = set()
        reflection_notes: Dict[str, str] = {}
        summary: List[Dict[str, Any]] = []

        self.ck_dir.mkdir(parents=True, exist_ok=True)
        result_path = self.ck_dir / "baseline_reflection_results.jsonl"
        summary_path = self.ck_dir / "baseline_reflection_summary.json"
        state_path = self.ck_dir / "baseline_reflection_state.json"

        start_round = 1
        if state_path.exists():
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                solved = {str(x) for x in state.get("solved", [])}
                reflection_notes = {
                    str(k): v for k, v in state.get("reflection_notes", {}).items()
                }
                summary = state.get("summary", [])
                last_completed = int(state.get("last_completed_round", 0))
                start_round = max(1, last_completed + 1)
                logger.info(
                    "Resuming reflection from round %d (%d/%d solved)",
                    start_round, len(solved), total_tasks,
                )
            except Exception:
                logger.warning("Failed to load reflection state, starting fresh", exc_info=True)

        if start_round > self.baseline_k:
            logger.info("Reflection baseline already completed (last round %d).", start_round - 1)
            return

        try:
            request_interval = float(os.environ.get("MEMRL_LLB_REQUEST_INTERVAL", "0.5") or "0")
        except (TypeError, ValueError):
            request_interval = 0.5

        for round_idx in range(start_round, self.baseline_k + 1):
            logger.info("Starting reflection round %d/%d (solved so far: %d/%d)",
                        round_idx, self.baseline_k, len(solved), total_tasks)

            for sample_index in tqdm(all_keys, desc=f"reflection round {round_idx}"):
                if sample_index in solved:
                    continue
                note = reflection_notes.get(str(sample_index))
                traj = self._sample_single_trajectory_baseline(sample_index, reflection_note=note)
                if traj is not None:
                    reflection_notes[str(sample_index)] = self._format_reflection_note(
                        traj["trajectory"], traj["success"]
                    )
                    if traj["success"]:
                        solved.add(str(sample_index))
                    payload = {
                        "round": round_idx,
                        "baseline": "reflection",
                        "sample_index": sample_index,
                        "success": traj["success"],
                        "steps": traj["steps"],
                    }
                    with open(result_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
                if request_interval > 0:
                    time.sleep(request_interval)

            cum_acc = len(solved) / total_tasks if total_tasks > 0 else 0.0
            summary.append({
                "round": round_idx,
                "cum_acc": cum_acc,
                "solved": len(solved),
                "total": total_tasks,
            })
            logger.info("reflection round %d cumulative acc: %.2f%% (%d/%d)",
                        round_idx, cum_acc * 100, len(solved), total_tasks)
            self.writer.add_scalar("Baseline/Reflection_Cumulative_Acc", cum_acc, round_idx)

            with open(state_path, "w", encoding="utf-8") as f:
                json.dump({
                    "last_completed_round": round_idx,
                    "solved": sorted(solved),
                    "reflection_notes": reflection_notes,
                    "summary": summary,
                    "total": total_tasks,
                    "updated_at": datetime.now().isoformat(),
                }, f, ensure_ascii=False)

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info("Reflection baseline complete. Final acc: %.2f%%",
                    summary[-1]["cum_acc"] * 100 if summary else 0.0)

    def run(self):
        """Main entry point for running LLB evaluation with RL training.

        Supports multiple algorithms through a single unified run flow:
        - 'rl': Mini-batch level update_values() only (fast feedback)
        - 'mdp': Section level update_values_chain_mdp() only (slow propagation)
        - 'rl_mdp' or 'memp': Both mini-batch and section updates (combined)

        Flow:
        1. Section loop
        2. Mini-batch loop within each section
        3. Sample mini-batch trajectories
        4. [RL] Update Q-values for retrieved memories (if 'rl' in algorithm)
        5. Add memories using add_memories()
        6. [MDP] After section: call update_values_chain_mdp() (if 'mdp' in algorithm)
        """
        logger.info("Starting LLB RL evaluation...")
        logger.info(f"Task: {self.task}")
        logger.info(f"Dataset: {self.split_file}")
        logger.info(f"Num sections: {self.num_section}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Algorithm: {self.algorithm}")
        logger.info(f"Max steps per task: {self.max_steps}")

        if self.baseline_mode in ("passk", "reflection"):
            logger.info(f"Running baseline mode: {self.baseline_mode} (k={self.baseline_k})")
            if self.baseline_mode == "passk":
                self._run_passk_baseline()
            else:
                self._run_reflection_baseline()
            self.writer.close()
            return

        # Initial evaluation before training (if not resuming from checkpoint)
        if self.start_section == 0 and self.val_before_train:
            if self.valid_dataset:
                logger.info("\n" + "=" * 50)
                logger.info("Running initial validation evaluation before training...")
                logger.info("=" * 50)
                self._evaluate(self.valid_dataset, "Validation", 0)
            else:
                logger.info("Skipping initial validation (no validation dataset)")
        else:
            logger.info(
                f"Skipping initial validation (start_section={self.start_section}, val_before_train={self.val_before_train})"
            )

        # A section checkpoint is saved before its validation. If a job dies during
        # validation, normal checkpoint resume correctly starts at the next section but
        # would otherwise lose that validation result. Explicitly replay it once.
        if self.resume_eval_section != 0:
            _resume_eval_section = (
                self.start_section if self.resume_eval_section < 0 else self.resume_eval_section
            )
            _resume_eval_marker = self._validation_done_marker(_resume_eval_section)
            if _resume_eval_section <= 0:
                logger.info("No completed section requires resume validation.")
            elif _resume_eval_marker is not None and _resume_eval_marker.is_file():
                logger.info(
                    "Resume validation for section %d already completed (%s); skipping.",
                    _resume_eval_section, _resume_eval_marker,
                )
            elif self.start_section < _resume_eval_section:
                raise RuntimeError(
                    f"resume_eval_section={_resume_eval_section} requires a checkpoint "
                    f"at or after that section, but start_section={self.start_section}"
                )
            elif not self.valid_dataset:
                raise RuntimeError("resume validation requested but no validation dataset is configured")
            else:
                logger.info("\n" + "=" * 50)
                logger.info(
                    "Running missing validation evaluation after resumed Section %d...",
                    _resume_eval_section,
                )
                logger.info("=" * 50)
                self._evaluate(self.valid_dataset, "Validation", _resume_eval_section)
                self._mark_validation_done(_resume_eval_section)

        # Track memory references for chain MDP
        memid_pair = OrderedDict()  # new_id -> [referenced_ids]

        # Batch-level resume: self._resume_batch_start was set from the loaded snapshot
        # name by run_llb (authoritative). The inner loop skips that many mini-batches,
        # guarded by section_idx == self.start_section (only the resumed section).
        if self._resume_batch_start > 0:
            logger.info(
                f"[BatchCkpt] Will resume section {self.start_section + 1} at mini-batch "
                f"{self._resume_batch_start} (derived from loaded snapshot)."
            )

        # Main training loop: iterate through sections
        for section_idx in range(self.start_section, len(self.section_splits)):
            section_num = section_idx + 1
            section_keys = self.section_splits[section_idx]

            logger.info(
                "\n"
                + "#" * 20
                + f" STARTING SECTION {section_num}/{self.num_section}"
                + "#" * 20
            )
            logger.info(f"Total samples in section {section_num}: {len(section_keys)}")

            # Region: sync epoch number for exploration schedule / mixed ratio
            if hasattr(self.memory_service, 'set_current_epoch'):
                self.memory_service.set_current_epoch(section_num, num_epochs=self.num_section)

            # Split section into mini-batches
            num_mini_batches = int(np.ceil(len(section_keys) / self.batch_size))
            section_mini_batches = [
                section_keys[i * self.batch_size : (i + 1) * self.batch_size]
                for i in range(num_mini_batches)
            ]

            logger.info(
                f"Split into {len(section_mini_batches)} mini-batches of size <= {self.batch_size}"
            )

            section_trajectories = []
            des_id_list = []  # For chain MDP: [(task_desc, mem_id), ...]

            # Batch-level resume: only for the section we resumed into, skip the
            # mini-batches already completed in a prior (preempted) run. Memory/Q
            # state for those batches was restored from the batch snapshot.
            _skip_until_batch = 0
            if (
                self.ckpt_save_every_n_batches > 0
                and self._resume_batch_start > 0
                and section_idx == self.start_section
            ):
                _skip_until_batch = self._resume_batch_start
                logger.info(
                    f"[BatchCkpt] Section {section_num}: resuming, skipping first "
                    f"{_skip_until_batch} mini-batches (already done)."
                )
                # Consume the resume marker so later sections start fresh.
                self._resume_batch_start = 0

            # Inner loop: iterate through mini-batches
            for mini_batch_idx, mini_batch_keys in enumerate(
                tqdm(section_mini_batches, desc=f"Section {section_num}")
            ):
                if mini_batch_idx < _skip_until_batch:
                    continue
                logger.info(
                    f"Processing mini-batch {mini_batch_idx+1}/{len(section_mini_batches)} in section {section_num}..."
                )

                # 1. Sample trajectories for this mini-batch
                collected_trajs = self._sample_from_indices(
                    sample_indices=mini_batch_keys,
                    phase="train",
                )

                if not collected_trajs:
                    logger.warning(
                        f"No trajectories collected for mini-batch {mini_batch_idx+1}"
                    )
                    continue

                logger.info(
                    f"Mini-batch {mini_batch_idx+1} collected {len(collected_trajs)} trajectories."
                )
                section_trajectories.extend(collected_trajs)
                # Persist task-level outcomes before memory update/checkpoint work.
                # This preserves CSR auditability even if a backend later dedups
                # or deletes memories (notably Mem0 atomic facts).
                self._append_llb_task_outcomes(section_num, collected_trajs)

                # 2. Extract data for memory processing (matching alfworld structure)
                task_descriptions = [
                    traj["task_description"] for traj in collected_trajs
                ]
                trajectories = [
                    traj["trajectory"] for traj in collected_trajs
                ]  # Trajectory strings
                successes = [traj["success"] for traj in collected_trajs]

                # Extract retrieved memory IDs
                retrieved_ids_list = [
                    [
                        mem["memory_id"]
                        for mem_list in traj["retrieved_mems"].values()
                        for mem in mem_list
                        if "memory_id" in mem
                    ]
                    for traj in collected_trajs
                ]

                retrieved_queries = [
                    traj["retrieved_queries"] for traj in collected_trajs
                ]

                # 3. Update Q-values for retrieved memories (immediate feedback) - only if algorithm includes 'rl'
                if "rl" in self.algorithm.lower():
                    # Build target_subtasks for region per-subtask Q updates
                    _update_kwargs = {}
                    if hasattr(self.memory_service, "region_manager"):
                        from memrl.configs.task_hierarchy import (
                            get_primary_subtask, get_db_multi_axis_subtasks,
                        )
                        _target_subtasks = [
                            get_primary_subtask(
                                f"llb_{self.task}",
                                {"skill_list": traj.get("skill_list", [])},
                            )
                            for traj in collected_trajs
                        ]
                        _update_kwargs["target_subtasks"] = _target_subtasks
                        if (
                            self.task == "db"
                            and os.environ.get("MEMRL_DB_MULTI_AXIS", "0").lower()
                            in {"1", "true", "yes"}
                        ):
                            _update_kwargs["target_subtask_weights"] = [
                                get_db_multi_axis_subtasks(traj.get("skill_list", []))
                                for traj in collected_trajs
                            ]

                    updated_q_list = self.memory_service.update_values(
                        successes, retrieved_ids_list, **_update_kwargs
                    )
                    logger.info(
                        f"[RL] Updated Q-values for mini-batch {mini_batch_idx+1}: {len(updated_q_list)} memories"
                    )
                    # DEBUG: show Q update details (sample of updated values)
                    q_samples = [(mid, q) for mid, q in (updated_q_list or {}).items() if q is not None][:3]
                    if q_samples:
                        logger.info(
                            f"[RL Q-DEBUG] sample updates: {[(mid[:8], f'{q:.3f}') for mid, q in q_samples]} | "
                            f"successes={[int(s) for s in successes]} | "
                            f"ids_per_traj={[len(ids) for ids in retrieved_ids_list]}"
                        )
                else:
                    logger.debug(
                        f"Skipping mini-batch Q-value update (algorithm={self.algorithm})"
                    )

                # 4. Prepare metadata for new memories
                #
                # IMPORTANT (LLB alignment / de-dup):
                # - Persist task_id/sample_index so "dedup by task_id" works in retrieval,
                #   especially when multiple epochs create multiple memories for the same task.
                # - Use numeric task ids when possible to align with legacy memory_rl traces.
                metadatas_update: List[Dict[str, Any]] = []
                for traj in collected_trajs:
                    raw_sid = traj.get("sample_index")
                    task_id: Any = raw_sid
                    try:
                        if raw_sid is not None and str(raw_sid).strip().isdigit():
                            task_id = int(str(raw_sid).strip())
                    except Exception:
                        task_id = raw_sid

                    meta_entry = {
                        "source_benchmark": f"llb_{self.task}",
                        "phase": "train",
                        "lb_epoch": int(section_num),
                        "sample_index": task_id,
                        "task_id": task_id,
                        "skill_list": traj.get("skill_list", []),
                        "success": traj["success"],
                        "q_value": (
                            float(self.rl_config.q_init_pos)
                            if traj["success"]
                            else float(self.rl_config.q_init_neg)
                        ),
                        "q_visits": 0,
                        "q_updated_at": datetime.now().isoformat(),
                        "last_used_at": datetime.now().isoformat(),
                        "reward_ma": 0.0,
                    }

                    # Failure evidence for reflection (AdjustmentUpdater reads
                    # metadata["error"]/["eval_error"]). Only set on failures so
                    # the reflection LLM knows *why* the task failed instead of
                    # inspecting a superficially-valid trajectory and concluding
                    # "no mistakes".
                    if not traj["success"]:
                        fr = traj.get("failure_reason") or ""
                        if fr:
                            meta_entry["error"] = fr
                            meta_entry["eval_error"] = fr

                    # Region: tag source_subtask for per-subtask Q tracking
                    if hasattr(self.memory_service, "region_manager"):
                        from memrl.configs.task_hierarchy import get_primary_subtask
                        meta_entry["source_subtask"] = get_primary_subtask(
                            f"llb_{self.task}",
                            {"skill_list": traj.get("skill_list", [])},
                        )
                        if (
                            self.task == "db"
                            and os.environ.get("MEMRL_DB_MULTI_AXIS", "0").lower()
                            in {"1", "true", "yes"}
                        ):
                            from memrl.configs.task_hierarchy import get_db_multi_axis_subtasks
                            meta_entry["source_subtasks_weighted"] = [
                                {"subtask": st, "weight": weight}
                                for st, weight in get_db_multi_axis_subtasks(
                                    traj.get("skill_list", [])
                                )
                            ]

                    metadatas_update.append(meta_entry)

                # 5. Add memories using add_memories (batch update)
                result_vis = self.memory_service.add_memories(
                    task_descriptions=task_descriptions,
                    trajectories=trajectories,
                    successes=successes,
                    retrieved_memory_queries=retrieved_queries,
                    retrieved_memory_ids_list=retrieved_ids_list,
                    metadatas=metadatas_update,
                )

                # Track memory references for chain MDP. Updater results are
                # intentionally forward-compatible: legacy backends return
                # (task_desc, mem_id), while the batched write-embedding path returns
                # (task_desc, mem_id, precomputed_vector). Only the first two fields
                # belong to this runner contract. Map by task description rather than
                # result position because a partially failed batch can omit an item.
                _result_indices = defaultdict(list)
                for _idx, _task_desc in enumerate(task_descriptions):
                    _result_indices[str(_task_desc)].append(_idx)
                for result_pos, result_item in enumerate(result_vis or []):
                    if result_item is None:
                        logger.warning(
                            "[MemoryUpdate] skipping unavailable memory result at result position %d",
                            result_pos,
                        )
                        continue
                    if not isinstance(result_item, (tuple, list)) or len(result_item) < 2:
                        logger.warning(
                            "[MemoryUpdate] skipping malformed memory result at position %d: %r",
                            result_pos, result_item,
                        )
                        continue
                    task_desc, mem_id = result_item[:2]
                    _candidate_indices = _result_indices.get(str(task_desc), [])
                    if _candidate_indices:
                        source_idx = _candidate_indices.pop(0)
                    elif result_pos < len(retrieved_ids_list):
                        source_idx = result_pos
                        logger.warning(
                            "[MemoryUpdate] task %r not found in batch index map; "
                            "falling back to result position %d",
                            task_desc, result_pos,
                        )
                    else:
                        source_idx = None
                    if mem_id:
                        des_id_list.append((task_desc, mem_id))
                        if source_idx is not None and hasattr(
                            self.memory_service, "register_region_memory_from_metadata"
                        ):
                            try:
                                self.memory_service.register_region_memory_from_metadata(
                                    str(mem_id), metadatas_update[source_idx]
                                )
                            except Exception:
                                logger.warning(
                                    "[Region Register] failed for memory %s", mem_id,
                                    exc_info=True,
                                )
                        if source_idx is not None and retrieved_ids_list[source_idx]:
                            self._add_to_memid_pair_fifo(
                                memid_pair,
                                key=mem_id,
                                values=retrieved_ids_list[source_idx],
                                max_capacity=10000,
                            )

                logger.info(f"Mini-batch {mini_batch_idx+1} memory update complete.")
                self._log_token_usage(section_num, mini_batch=mini_batch_idx + 1)

                # Region: mid-epoch clustering maintenance
                rm = getattr(self.memory_service, "region_manager", None)
                if rm is not None:
                    global_step = (section_idx * len(section_keys)) + ((mini_batch_idx + 1) * self.batch_size)
                    if (
                        not rm._is_clustered
                        and self.region_cluster_init_step > 0
                        and global_step >= self.region_cluster_init_step
                    ):
                        rm.cluster_by_utility()
                        if rm._is_clustered:
                            rm.topology_last_edit_section = int(section_num)
                        logger.info(
                            "[Region] Initial clustering at global_step=%d "
                            "(configured init step=%d)",
                            global_step,
                            self.region_cluster_init_step,
                        )
                    elif rm._is_clustered:
                        for mem_id in rm.subtask_q:
                            if mem_id not in rm.membership_weights:
                                rm.assign_new_memory(mem_id)

                    # One-shot mid-section topology edits, separate from initial
                    # clustering. Completed steps are checkpointed by RegionManager.
                    if rm._is_clustered:
                        done_steps = getattr(rm, "topology_mid_maintenance_done_steps", set())
                        for maintenance_step in self.region_topology_maintenance_steps:
                            if global_step < maintenance_step or maintenance_step in done_steps:
                                continue
                            changed = rm.maybe_split_merge()
                            done_steps.add(maintenance_step)
                            rm.topology_mid_maintenance_done_steps = done_steps
                            if changed:
                                rm.topology_last_edit_section = int(section_num)
                            logger.info(
                                "[Region] Mid-section topology maintenance at global_step=%d "
                                "(configured step=%d): changed=%s regions=%d",
                                global_step, maintenance_step, changed, len(rm.regions),
                            )

                # Batch-level checkpoint: snapshot memory + progress every N batches so a
                # mid-section preemption can resume without redoing the whole section.
                if (
                    self.ckpt_save_every_n_batches > 0
                    and self.ck_dir is not None
                    and (mini_batch_idx + 1) % self.ckpt_save_every_n_batches == 0
                ):
                    try:
                        # NAMING CONVENTION: batch snapshot dir = f"{section_num}_b{mini_batch_idx}"
                        # where mini_batch_idx is 0-BASED (the last COMPLETED batch). On resume,
                        # run_llb derives resume_batch = mini_batch_idx + 1 (next batch to run).
                        batch_ckpt_id = f"{section_num}_b{mini_batch_idx}"
                        ckpt_meta = self.memory_service.save_checkpoint_snapshot(
                            str(self.ck_dir), ckpt_id=batch_ckpt_id
                        )
                        # next_batch = index of the next mini-batch to run on resume.
                        self._llb_save_batch_progress(
                            next_section=section_num, next_batch=mini_batch_idx + 1,
                            snapshot_id=batch_ckpt_id,
                        )
                        self._llb_cleanup_batch_ckpts(section_num, keep=self.ckpt_max_keep)
                        self._persist_llb_cum_state(
                            self.ck_dir / "snapshot" / batch_ckpt_id / "local_cache" / "cum_state.json"
                        )
                        logger.info(
                            f"[BatchCkpt] Saved batch ckpt {batch_ckpt_id} "
                            f"(section {section_num}, next_batch {mini_batch_idx + 1})"
                        )
                    except Exception as e:
                        logger.warning(f"[BatchCkpt] Failed to save batch ckpt: {e}")

            # Section complete - log section-level metrics
            logger.info(
                f"Section {section_num} complete. Total {len(section_trajectories)} trajectories collected."
            )

            # Calculate and log section metrics
            if section_trajectories:
                section_success = sum(
                    1 for traj in section_trajectories if traj["success"]
                )
                section_success_rate = section_success / len(section_trajectories)
                section_avg_steps = np.mean(
                    [traj["steps"] for traj in section_trajectories]
                )

                logger.info(
                    f"Section {section_num} Training Stats: Success Rate={section_success_rate:.2%}, Avg Steps={section_avg_steps:.2f}"
                )

                # TensorBoard logging
                self.writer.add_scalar(
                    "Train/Section_Success_Rate", section_success_rate, section_num
                )
                self.writer.add_scalar(
                    "Train/Section_Avg_Steps", section_avg_steps, section_num
                )

                # Log individual results
                for traj_data in section_trajectories:
                    self.results_log.append(
                        {
                            "section": section_num,
                            "mode": self.mode,
                            "success": traj_data["success"],
                            "steps": traj_data["steps"],
                        }
                    )

            # 6. After section: update values using chain MDP - only if algorithm includes 'mdp'
            if "mdp" in self.algorithm.lower() and self.rl_config and des_id_list:
                logger.info(
                    f"[MDP] Running update_values_chain_mdp for section {section_num}..."
                )
                successes_for_chain = [
                    (
                        1.0
                        if any(
                            t["task_description"] == desc and t["success"]
                            for t in section_trajectories
                        )
                        else 0.0
                    )
                    for desc, _ in des_id_list
                ]

                self.memory_service.update_values_chain_mdp(
                    des_id_list=des_id_list,
                    memid_pair=memid_pair,
                    successes=successes_for_chain,
                )
                logger.info(
                    f"[MDP] Chain MDP update complete for section {section_num}"
                )
            else:
                logger.debug(
                    f"Skipping chain MDP update (algorithm={self.algorithm}, rl_config={'present' if self.rl_config else 'missing'}, des_id_list={'present' if des_id_list else 'empty'})"
                )

            # Region: end-of-section clustering maintenance with optional
            # full-section cooldown. Online Q/evidence/membership updates continue;
            # only topology edits (initial cluster/split/merge) are rate-limited.
            rm = getattr(self.memory_service, "region_manager", None)
            if rm is not None:
                if not rm._is_clustered:
                    rm.cluster_by_utility()
                    if rm._is_clustered:
                        rm.topology_last_edit_section = int(section_num)
                    logger.info("[Region] End-of-section initial clustering")
                else:
                    for mem_id in rm.subtask_q:
                        if mem_id not in rm.membership_weights:
                            rm.assign_new_memory(mem_id)
                    last_edit = int(getattr(rm, "topology_last_edit_section", 0) or 0)
                    cooldown = int(self.region_topology_cooldown_sections)
                    if cooldown > 0 and last_edit > 0 and (section_num - last_edit) <= cooldown:
                        logger.info(
                            "[Region] End-of-section split/merge skipped by topology cooldown: "
                            "section=%d last_edit=%d cooldown=%d regions=%d",
                            section_num, last_edit, cooldown, len(rm.regions),
                        )
                    else:
                        changed = rm.maybe_split_merge()
                        if changed:
                            rm.topology_last_edit_section = int(section_num)
                        logger.info(
                            "[Region] End-of-section split/merge done. changed=%s regions=%d last_edit=%d",
                            changed, len(rm.regions), int(getattr(rm, "topology_last_edit_section", 0) or 0),
                        )

            # Save checkpoint
            ckpt_meta = self.memory_service.save_checkpoint_snapshot(
                self.ck_dir, ckpt_id=section_num
            )
            logger.info(f"Saved checkpoint: {ckpt_meta}")
            # Co-locate the exact union success set with the section snapshot.
            self._persist_llb_cum_state(
                self.ck_dir / "snapshot" / str(section_num) / "local_cache" / "cum_state.json"
            )
            self._llb_log_cumulative_success(section_num)

            # Section-level ckpt supersedes this section's batch ckpts: record that the
            # next run should start at the next section (batch 0) and drop batch snapshots.
            if self.ckpt_save_every_n_batches > 0 and self.ck_dir is not None:
                self._llb_save_batch_progress(next_section=section_num + 1, next_batch=0,
                                              snapshot_id=str(section_num))
                self._llb_cleanup_batch_ckpts(section_num, keep=0)

            # Log token usage for section
            self._log_token_usage(section_num)
            self._check_memory_usage(f"After section {section_num}")

            # Periodic evaluation
            if self.mode != "test":
                if self.valid_interval > 0 and section_num % self.valid_interval == 0:
                    if self.valid_dataset:
                        is_last_section = (section_idx == len(self.section_splits) - 1)
                        if is_last_section and self.eval_runs > 1:
                            self._evaluate_multi(self.valid_dataset, "Validation", section_num)
                        else:
                            self._evaluate(self.valid_dataset, "Validation", section_num)
                        self._mark_validation_done(section_num)
                    else:
                        logger.info(
                            f"Validation evaluation skipped (no validation dataset)"
                        )

        # Final analysis
        self._analyze_and_report_results()

        # Close TensorBoard writer
        self.writer.close()
        logger.info("\nTraining completed!")

    # Removed _update_memory_from_trajectories - now using add_memories() directly in run()

    def _validation_done_marker(self, section_num: int) -> Optional[Path]:
        if self.ck_dir is None or int(section_num) <= 0:
            return None
        return self.ck_dir / f"validation_section_{int(section_num)}.done"

    def _mark_validation_done(self, section_num: int) -> None:
        marker = self._validation_done_marker(section_num)
        if marker is None:
            return
        marker.parent.mkdir(parents=True, exist_ok=True)
        tmp_marker = marker.with_suffix(".done.tmp")
        with open(tmp_marker, "w", encoding="utf-8") as f:
            f.write(f"section={int(section_num)}\ncompleted_at={datetime.now().isoformat()}\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_marker, marker)
        logger.info("Recorded validation completion: %s", marker)

    def _save_checkpoint(self, section_idx: int):
        """Save memory checkpoint.

        Args:
            section_idx: Current section index
        """
        try:
            ckpt_path = self.ck_dir / f"section_{section_idx}.pkl"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            self.memory_service.save_checkpoint_snapshot(str(ckpt_path))
            logger.info(f"Saved checkpoint to {ckpt_path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}", exc_info=True)

    # ---------- Batch-level checkpoint helpers (mirror HLE runner) ----------

    # ---------- Durable task outcome / union cumulative helpers ----------

    def _load_llb_cum_state(self) -> None:
        """Restore the task-union success set without relying on memory contents."""
        candidates = [self._llb_task_cum_state_path]
        snap_root = self.ck_dir / "snapshot"
        if snap_root.is_dir():
            ranked = []
            import re
            for child in snap_root.iterdir():
                if not child.is_dir():
                    continue
                match = re.fullmatch(r"(\d+)(?:_b(\d+))?", child.name)
                if not match:
                    continue
                ranked.append(((int(match.group(1)), int(match.group(2) or 10**9)), child))
            for _key, child in sorted(ranked, reverse=True):
                candidates.append(child / "local_cache" / "cum_state.json")
        for path in candidates:
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                ids = payload.get("success_ids", [])
                self._llb_cum_success_ids = {str(x) for x in ids if x is not None}
                total = payload.get("total")
                if isinstance(total, int) and total > 0:
                    self._llb_cum_total = total
                logger.info(
                    "[LLB Cumulative] restored %d success IDs (denominator=%d) from %s",
                    len(self._llb_cum_success_ids), self._llb_cum_total, path,
                )
                return
            except Exception:
                logger.warning("Failed to restore LLB cumulative state from %s", path, exc_info=True)

        # Backward-compatible recovery for experiments started before the
        # dedicated cum_state ledger existed. Standard MemoryService snapshots
        # retain one train memory per (epoch, sample_index), including success.
        # Reconstruct only from a healthy latest snapshot; Mem0 snapshots do not
        # have textual_memory.json and therefore simply skip this fallback.
        for _key, child in sorted(ranked, reverse=True) if 'ranked' in locals() else []:
            memory_path = child / "cube" / "textual_memory.json"
            if not memory_path.is_file():
                continue
            try:
                rows = json.loads(memory_path.read_text(encoding="utf-8"))
                recovered = set()
                for row in rows:
                    metadata = ((row.get("payload") or {}).get("metadata") or {})
                    if metadata.get("phase") != "train":
                        continue
                    sid = metadata.get("sample_index")
                    if sid is not None and bool(metadata.get("success", False)):
                        recovered.add(str(sid))
                if recovered:
                    self._llb_cum_success_ids = recovered
                    self._llb_cum_total = len(self.dataset)
                    self._persist_llb_cum_state()
                    logger.info(
                        "[LLB Cumulative] reconstructed %d success IDs from legacy snapshot %s",
                        len(recovered), child,
                    )
                    return
            except Exception:
                logger.warning(
                    "Failed to reconstruct LLB cumulative state from %s", memory_path,
                    exc_info=True,
                )

    def _persist_llb_cum_state(self, path: Optional[Path] = None) -> None:
        """Atomically save true task-union state for resume and CSR audit."""
        path = path or self._llb_task_cum_state_path
        payload = {
            "success_ids": sorted(self._llb_cum_success_ids),
            "total": self._llb_cum_total,
            "updated_at": datetime.now().isoformat(),
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except Exception:
            logger.warning("Failed to persist LLB cumulative state to %s", path, exc_info=True)

    def _append_llb_task_outcomes(self, section_num: int, trajectories: List[Dict[str, Any]]) -> None:
        """Append one fsync'd task outcome record per completed train trajectory.

        The JSONL is intentionally append-only.  A resumed batch can yield a
        duplicate `(epoch, sample_index)` record; downstream CSR reconstruction
        uses OR over successes, while `cum_state.json` remains the authoritative
        live union set.
        """
        if not trajectories:
            return
        records = []
        for traj in trajectories:
            sample_index = traj.get("sample_index")
            if sample_index is None:
                continue
            success = bool(traj.get("success", False))
            sid = str(sample_index)
            if success:
                self._llb_cum_success_ids.add(sid)
            records.append({
                "epoch": int(section_num),
                "sample_index": sid,
                "success": success,
                "steps": traj.get("steps"),
                "recorded_at": datetime.now().isoformat(),
            })
        if not records:
            return
        try:
            self._llb_outcome_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._llb_outcome_path, "a", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                f.flush()
                os.fsync(f.fileno())
            self._persist_llb_cum_state()
        except Exception:
            logger.warning("Failed to append LLB task outcomes", exc_info=True)

    def _llb_log_cumulative_success(self, section_num: int) -> None:
        total = max(1, int(self._llb_cum_total))
        sr = len(self._llb_cum_success_ids) / total
        logger.info(
            "[LLB Cumulative] after Section %d: %.2f%% (%d/%d unique tasks succeeded at least once)",
            section_num, sr * 100, len(self._llb_cum_success_ids), total,
        )
        self.writer.add_scalar("Train/Cumulative_Success_Rate", sr, section_num)
        self.results_log.append({
            "section": f"cum_s{section_num}",
            "after_section": section_num,
            "mode": "train_cumulative",
            "success": sr,
            "steps": 0.0,
        })

    def _llb_cum_state_path(self) -> Optional[Path]:
        if self.ck_dir is None:
            return None
        return self.ck_dir / "llb_batch_progress.json"

    @staticmethod
    def _llb_is_valid_snapshot_dir(snapshot_dir: Path) -> bool:
        """True only for real memory snapshots (meta.json or cube/ present)."""
        try:
            if not snapshot_dir.is_dir():
                return False
            if (snapshot_dir / "snapshot_meta.json").is_file():
                return True
            if (snapshot_dir / "cube").is_dir():
                return True
        except Exception:
            return False
        return False

    def _llb_save_batch_progress(self, next_section: int, next_batch: int,
                                 snapshot_id: Optional[str] = None) -> None:
        """Persist which (section, batch) to resume from. next_section is 1-based
        section_num; next_batch is the index of the NEXT mini-batch to run.

        Advisory only — resume authority is the snapshot dir name (run_llb derives
        start_section/resume_batch from it). This file is for logging/debugging.
        Written atomically (tmp + os.replace) so a crash never leaves a torn file."""
        path = self._llb_cum_state_path()
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "next_section": int(next_section),
                "next_batch": int(next_batch),
                "snapshot_id": snapshot_id,
                "updated_at": datetime.now().isoformat(),
            }
            tmp = path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except Exception as e:
            logger.warning(f"[BatchCkpt] Failed to save batch progress: {e}")

    def _llb_cleanup_batch_ckpts(self, section_num: int, keep: int) -> None:
        """Remove old batch snapshots for a section, keeping only the latest `keep`.

        Uses a STRICT regex (^{section_num}_b\\d+$) so it can never match another
        section's dirs, the section snapshot itself, or unrelated junk directories."""
        if self.ck_dir is None:
            return
        try:
            import re, shutil
            snap_root = self.ck_dir / "snapshot"
            if not snap_root.is_dir():
                return
            pat = re.compile(rf"^{int(section_num)}_b(\d+)$")
            batch_dirs = []
            for p in snap_root.iterdir():
                if not (p.is_dir() and self._llb_is_valid_snapshot_dir(p)):
                    continue
                m = pat.match(p.name)
                if m:
                    batch_dirs.append((int(m.group(1)), p))
            if len(batch_dirs) <= keep:
                return
            batch_dirs.sort(key=lambda x: x[0])
            to_remove = batch_dirs[:-keep] if keep > 0 else batch_dirs
            for _b, d in to_remove:
                shutil.rmtree(d, ignore_errors=True)
                logger.info(f"[BatchCkpt] Removed old batch ckpt: {d.name}")
        except Exception as e:
            logger.warning(f"[BatchCkpt] Cleanup failed for section {section_num}: {e}")
