"""
BigCodeBench (BCB) multi-epoch runner for MemRL.

This runner implements the same high-level structure used by other benchmarks:
  - multi-epoch loop
  - per-epoch train then val
  - retrieval via MemoryService.retrieve_query (dict_memory + RL threshold)
  - train writes memories via MemoryService.add_memories (keeps dict_memory in sync)
  - value-driven Q updates via MemoryService.update_values (best-effort)
  - per-epoch snapshots via MemoryService.save_checkpoint_snapshot(target_ck_dir, ckpt_id)
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from memrl.bigcodebench_eval.bcb_adapter import extract_code_from_response
from memrl.bigcodebench_eval.eval_utils import (
    ensure_bigcodebench_on_path,
    run_untrusted_check_with_hard_timeout,
    sanitize_code,
)
from memrl.bigcodebench_eval.task_wrappers import get_prompt, load_bcb_data, split_dataset, write_samples

logger = logging.getLogger(__name__)

try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    SummaryWriter = None  # type: ignore[assignment]

# BCB official instruction prefix (aligned with bigcodebench/generate.py).
BCB_INSTRUCTION_PREFIX = "Please provide a self-contained Python script that solves the following problem in a markdown code block:"

# BCB official response prefix (prefill). Forces the model to start generating
# code immediately inside a fenced block, reducing preamble and improving
# code extraction reliability.
BCB_RESPONSE_PREFIX = "Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:\n```python"

# System prompt: only activated when memory context is available (epoch >= 2).
# For E1 (no memory), no system prompt is sent — matching BCB official eval.
DEFAULT_SYSTEM_PROMPT = """You are an expert Python programmer. You will receive retrieved memory context with past experiences from similar problems. Use them as references but always analyze the current task independently.

[MEMORY TYPE] SUCCESS_PROCEDURE: A successful approach—learn the pattern.
[MEMORY TYPE] FAILURE_REFLECTION: A failed attempt—avoid similar mistakes."""


@dataclass
class BCBSelection:
    subset: str = "hard"  # hard|full
    split: str = "instruct"  # instruct|complete
    train_ratio: float = 0.7
    seed: int = 42
    split_file: Optional[str] = None
    data_path: Optional[str] = None


class BCBRunner:
    def __init__(
        self,
        *,
        root: Path,
        selection: BCBSelection,
        llm: Any,
        memory_service: Any,
        output_dir: str,
        model_name: str,
        num_epochs: int = 3,
        run_validation: bool = False,
        temperature: float = 0.0,
        max_tokens: int = 1280,
        retrieve_k: int = 5,
        # BigCodeBench uses a dedicated similarity threshold knob (separate from RL tau).
        # If None, falls back to rl_config.sim_threshold (or rl_config.tau).
        retrieve_threshold: Optional[float] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        memory_budget_tokens: int = 0,
        bcb_repo: Optional[str] = None,
        untrusted_hard_timeout_s: float = 120.0,
        eval_timeout_s: float = 60.0,
        checkpoint_interval: int = 50,
        max_checkpoints: int = 3,
        resume_checkpoint_path: Optional[str] = None,
        resume_epoch: Optional[int] = None,
        resume_step: Optional[int] = None,
        strip_think: bool = False,
        batch_size: int = 1,
        baseline_mode: Optional[str] = None,
        baseline_k: int = 10,
        baseline_resume_results: Optional[str] = None,
        self_rag: bool = False,
        self_rag_inject_k: int = 3,
        n_eval_runs: int = 1,
        eval_temperature: Optional[float] = None,
        multi_eval_epochs: Optional[str] = None,
    ) -> None:
        self.root = Path(root)
        self.sel = selection
        self.llm = llm
        self.mem = memory_service
        self.output_dir = os.path.abspath(output_dir)
        self.model_name = str(model_name)
        self.num_epochs = int(num_epochs)
        self.run_validation = bool(run_validation)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.retrieve_k = int(retrieve_k)
        self.retrieve_threshold = (
            None if retrieve_threshold is None else float(retrieve_threshold)
        )
        self.system_prompt = str(system_prompt or "")
        # NOTE: In memory_rl this is called "budget_tokens" but is used as a rough character budget.
        self.memory_budget_tokens = int(memory_budget_tokens)
        self.bcb_repo = bcb_repo
        self.untrusted_hard_timeout_s = float(untrusted_hard_timeout_s)
        self.eval_timeout_s = float(eval_timeout_s)
        self.checkpoint_interval = max(1, int(checkpoint_interval))
        self.max_checkpoints = max(1, int(max_checkpoints))
        self.resume_checkpoint_path = resume_checkpoint_path
        self.resume_epoch = int(resume_epoch) if resume_epoch is not None else None
        self.resume_step = int(resume_step) if resume_step is not None else None
        self.strip_think = bool(strip_think)
        self.batch_size = max(1, int(batch_size))
        self.baseline_mode = (baseline_mode or "").strip().lower() or None
        self.baseline_k = max(1, int(baseline_k))
        self.baseline_resume_results = (
            os.path.abspath(str(baseline_resume_results)) if baseline_resume_results else None
        )
        self.self_rag = bool(self_rag)
        self.self_rag_inject_k = max(1, int(self_rag_inject_k))
        self.n_eval_runs = max(1, int(n_eval_runs))
        self.eval_temperature = float(eval_temperature) if eval_temperature is not None else None
        self.multi_eval_epochs: Optional[set] = None
        if multi_eval_epochs == "last":
            self.multi_eval_epochs = None  # resolved at runtime to {num_epochs}
        elif multi_eval_epochs == "all":
            self.multi_eval_epochs = None  # n_eval_runs applied to all epochs
        elif multi_eval_epochs:
            self.multi_eval_epochs = set(int(x.strip()) for x in multi_eval_epochs.split(",") if x.strip())
        self._multi_eval_mode = multi_eval_epochs or "last"

        # Inline failure summary (disabled by default; call configure_failure_summary).
        self._failure_summary_n_slots = 0
        self._failure_summary_inline_k: Optional[int] = None
        self._failure_inject_log_counter = 0

        ensure_bigcodebench_on_path(self.bcb_repo)

        self._problems: Dict[str, Dict[str, Any]] = {}
        self._train_ids: List[str] = []
        self._val_ids: List[str] = []
        self._lib2domain: Dict[str, str] = self._load_lib2domain()

        # --- [TENSORBOARD] Initialize SummaryWriter (optional) ---
        tb_log_dir = (
            self.root
            / "logs"
            / "tensorboard"
            / f"exp_bcb_{Path(self.output_dir).name}_{time.strftime('%Y%m%d-%H%M%S')}"
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
            logger.info("TensorBoard logs will be saved to: %s", tb_log_dir)

    def _tb_add_scalar(self, tag: str, value: Any, step: int) -> None:
        """Best-effort TensorBoard scalar logging."""
        try:
            self.writer.add_scalar(tag, value, global_step=int(step))
        except Exception:
            return

    def _load_lib2domain(self) -> Dict[str, str]:
        p = self.root / "3rdparty" / "bigcodebench-main" / "analysis" / "lib2domain.json"
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _get_task_domains(self, task: Dict[str, Any]) -> List[str]:
        libs = task.get("libs") or []
        if isinstance(libs, str):
            import ast
            try:
                libs = ast.literal_eval(libs)
            except Exception:
                libs = []
        domains = list({self._lib2domain[lib] for lib in libs if lib in self._lib2domain})
        domains.sort()
        return domains

    def _get_retrieve_threshold(self) -> float:
        """BCB threshold knob (aligned to memory_rl)."""
        if self.retrieve_threshold is not None:
            return float(self.retrieve_threshold)
        try:
            rl_cfg = getattr(self.mem, "rl_config", None)
            if rl_cfg is None:
                return 0.0
            return float(getattr(rl_cfg, "sim_threshold", getattr(rl_cfg, "tau", 0.0)))
        except Exception:
            return 0.0

    def _format_memory_context(
        self, selected_mems: List[Dict[str, Any]]
    ) -> str:
        # Align with memory_rl BCB adapter formatting.
        if not selected_mems:
            return ""

        # Deduplicate: if same task has both success and failure memory,
        # keep only the success one (failure reflection is redundant when
        # a working solution exists).
        selected_mems = self._dedup_success_over_failure(selected_mems)

        parts: List[str] = [
            "# Relevant Examples from Memory",
            "(Reference only — adapt to the current task, do not copy directly)\n",
        ]

        for i, c in enumerate(selected_mems, 1):
            meta_obj = c.get("metadata")
            meta: Dict[str, Any] = {}
            if meta_obj is not None:
                try:
                    if hasattr(meta_obj, "model_dump"):
                        meta = meta_obj.model_dump()  # type: ignore[assignment]
                    elif isinstance(meta_obj, dict):
                        meta = meta_obj
                except Exception:
                    meta = {}

            outcome = meta.get("outcome", "unknown")
            task_id = meta.get("task_id", "")

            mem_item = c.get("memory_item")
            task_desc = ""
            try:
                task_desc = str(getattr(mem_item, "memory", "") or "")
            except Exception:
                task_desc = ""
            # Mem0 returns atomic fact text as content rather than a memory_item.
            # Recover the original task from metadata so injected procedures and
            # failure reflections retain their source-task context.
            if not task_desc:
                task_desc = str(meta.get("task_description", "") or "")

            raw_content = c.get("content") or ""
            # Fallback: content may be None if memos metadata extraction failed.
            # Try metadata.full_content directly.
            if not raw_content:
                meta_full = meta.get("full_content", "")
                if meta_full:
                    raw_content = str(meta_full)
            # Last resort: use the memory text itself (task_description)
            if not raw_content and task_desc:
                raw_content = task_desc
            # Defense in depth: sanitize before injection (catches legacy memories
            # stored before strip_think was added).
            from memrl.utils.sanitize import sanitize_llm_output
            raw_content = sanitize_llm_output(raw_content)
            content = self._coerce_bcb_memory_content(
                raw_content=raw_content,
                outcome=outcome,
                task_description=task_desc,
            )
            if not content:
                continue

            # Truncate if needed (memory_rl uses a rough per-entry budget).
            # budget=0 means unlimited – skip truncation entirely.
            if self.memory_budget_tokens > 0 and len(content) > self.memory_budget_tokens // len(selected_mems):
                content = content[: self.memory_budget_tokens // len(selected_mems)] + "..."

            parts.append(f"## Example {i} [{outcome.upper()}]")
            if task_id:
                parts.append(f"Task: {task_id}")
            parts.append(content)
            parts.append("")

        return "\n".join(parts)

    @staticmethod
    def _dedup_success_over_failure(mems: list) -> list:
        """If same task has both success and failure memory, keep only success."""
        from collections import defaultdict

        # Group by task_id
        by_task: dict = defaultdict(list)
        no_task: list = []
        for c in mems:
            meta = c.get("metadata")
            if meta is None:
                no_task.append(c)
                continue
            if hasattr(meta, "model_extra"):
                tid = getattr(meta, "model_extra", {}).get("task_id", "")
            elif isinstance(meta, dict):
                tid = meta.get("task_id", "")
            else:
                tid = ""
            if tid:
                by_task[tid].append(c)
            else:
                no_task.append(c)

        result = list(no_task)
        for tid, group in by_task.items():
            # Check if there's a success in this group
            has_success = False
            for c in group:
                meta = c.get("metadata")
                outcome = ""
                if hasattr(meta, "model_extra"):
                    outcome = getattr(meta, "model_extra", {}).get("outcome", "")
                elif isinstance(meta, dict):
                    outcome = meta.get("outcome", "")
                if outcome in ("success", "True", "1"):
                    has_success = True
                    break

            if has_success:
                # Keep only successes for this task
                for c in group:
                    meta = c.get("metadata")
                    outcome = ""
                    if hasattr(meta, "model_extra"):
                        outcome = getattr(meta, "model_extra", {}).get("outcome", "")
                    elif isinstance(meta, dict):
                        outcome = meta.get("outcome", "")
                    if outcome in ("success", "True", "1"):
                        result.append(c)
            else:
                result.extend(group)

        return result

    @staticmethod
    def _coerce_bcb_memory_content(
        *,
        raw_content: str,
        outcome: str,
        task_description: str,
    ) -> str:
        """
        BCB-only prompt alignment:
        - memory_rl stores full_content using [MEMORY TYPE]/[TASK]/... blocks.
        - Other benchmarks must not be affected, so we coerce at *injection time*
          for BCB (even if the stored full_content is in legacy "Task: ..." style).
        """
        text = str(raw_content or "").strip()
        if not text:
            return ""

        # If it's already in the memory_rl format, keep as-is.
        if "[MEMORY TYPE]" in text.upper():
            return text

        out = str(outcome or "unknown").strip().lower()
        is_failure = out in {"failure", "fail", "failed", "0", "false", "no"}

        if is_failure:
            # Detect new compact format (case/whitespace tolerant)
            has_structured = (
                bool(re.search(r'(?im)^\s*failure[_ ]?mode\s*:', text)) and
                bool(re.search(r'(?im)^\s*mistakes\s*:', text))
            )
            if has_structured:
                return (
                    "[MEMORY TYPE] FAILURE_INSIGHT\n"
                    f"{text}"
                ).strip()
            # Legacy fallback: extract minimal reflection only
            m = re.search(r"(?is)(?:^|\n)reflection\s*:\s*(.*)$", text)
            reflection = (m.group(1) if m else text).strip()
            return (
                "[MEMORY TYPE] FAILURE_REFLECTION\n"
                "[KEY TAKEAWAYS]\n"
                f"{reflection[:1500]}"
            ).strip()

        # Success path: treat legacy body as execution trajectory.
        body = text
        m = re.match(r"(?is)^task\\s*:\\s*.*?\\n\\n(.*)$", text)
        if m:
            body = (m.group(1) or "").strip()
        td = task_description.strip() if task_description else ""
        return (
            "[MEMORY TYPE] SUCCESS_PROCEDURE\n"
            "[TASK]\n"
            f"{td}\n\n"
            "[EXECUTION TRAJECTORY]\n"
            f"{body}"
        ).strip()

    @staticmethod
    def _trajectory_from_raw_or_fallback(
        *,
        raw_response: str,
        prompt: str,
        code: str,
        eval_res: Dict[str, Any],
        retrieval: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Align with memory_rl BigCodeBench:
        - Prefer trajectory = model raw output when available
        - Otherwise fall back to a structured, step-style trajectory blob
        """
        if raw_response:
            from memrl.utils.sanitize import sanitize_llm_output
            return sanitize_llm_output(raw_response)

        steps: List[str] = []
        # Step 1: Original task prompt
        steps.append("[STEP 1] TASK PROMPT")
        steps.append(prompt or "")

        # Step 2: Memory retrieval info (if any)
        if retrieval:
            trace = retrieval.get("trace", {}) or {}
            steps.append("")
            steps.append("[STEP 2] MEMORY RETRIEVAL")
            steps.append(f"mode: {trace.get('mode', 'similarity')}")
            steps.append(
                f"retrieved_count: {trace.get('retrieved_count', retrieval.get('num_retrieved', 0))}"
            )
            steps.append(f"simmax: {trace.get('simmax', 0.0)}")
            steps.append(f"selected_memory_ids: {retrieval.get('selected_ids', [])}")

        # Step 3: Generated code
        steps.append("")
        steps.append("[STEP 3] GENERATED CODE")
        steps.append("```python")
        steps.append(code or "")
        steps.append("```")

        # Step 4: Evaluation result
        steps.append("")
        steps.append("[STEP 4] EVALUATION RESULT")
        status = eval_res.get("status", "UNKNOWN")
        steps.append(f"status: {status}")
        error_msg = eval_res.get("error", "")
        if error_msg:
            steps.append("error:")
            steps.append(str(error_msg))

        return "\n".join(steps)

    def _generate_raw(self, prompt: str, *, memory_context: str = "") -> str:
        messages: List[Dict[str, str]] = []

        system_parts: List[str] = []
        # Only inject system prompt when memory context is available (epoch >= 2).
        # For E1 (no memory), this matches BCB official eval: no system prompt.
        if memory_context:
            if self.system_prompt:
                system_parts.append(self.system_prompt)
            system_parts.append(memory_context)
        if system_parts:
            messages.append({"role": "system", "content": "\n\n".join(system_parts)})

        # Prepend BCB official instruction prefix to user message.
        # Append response_prefix to guide model toward code block format.
        user_content = f"{BCB_INSTRUCTION_PREFIX}\n{prompt}\n\n{BCB_RESPONSE_PREFIX}"
        messages.append({"role": "user", "content": user_content})

        try:
            resp = self.llm.generate(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            if "maximum context length" in str(e):
                logger.warning("Skipping task due to context length overflow: %s", str(e)[:200])
            else:
                logger.warning("LLM generation failed for BCB prompt", exc_info=True)
            return ""
        return resp or ""

    def _generate_code(self, prompt: str, *, memory_context: str = "") -> str:
        return extract_code_from_response(self._generate_raw(prompt, memory_context=memory_context), strip_think=self.strip_think)

    def _retrieve_for_task(self, prompt: str) -> Tuple[List[str], str, Dict[str, Any], Optional[List[Tuple[str, float]]]]:
        """Retrieve memory for a single task. Returns (selected_ids, mem_context, retrieval_trace, retrieved_topk_queries)."""
        selected_ids: List[str] = []
        retrieved_topk_queries: Optional[List[Tuple[str, float]]] = None
        mem_context = ""
        retrieval_trace: Dict[str, Any] = {}

        if self.mem is not None and self.retrieve_k > 0:
            try:
                thr = self._get_retrieve_threshold()
                ret = self.mem.retrieve_query(prompt, k=self.retrieve_k, threshold=thr)
                if isinstance(ret, tuple):
                    ret_result, retrieved_topk_queries = ret
                else:
                    ret_result, retrieved_topk_queries = ret, None

                selected_mems = (ret_result or {}).get("selected", []) if ret_result else []
                if not isinstance(selected_mems, list):
                    selected_mems = []

                selected_ids = [
                    str(m.get("memory_id") or m.get("id"))
                    for m in selected_mems
                    if isinstance(m, dict) and (m.get("memory_id") or m.get("id"))
                ]

                if self._failure_summary_n_slots > 0:
                    selected_mems = self._inject_failure_summary(selected_mems, prompt)

                if self.self_rag and selected_mems:
                    selected_mems = self._self_rag_critique(prompt, selected_mems, self.self_rag_inject_k)

                mem_context = self._format_memory_context(selected_mems)
                try:
                    retrieval_trace = {
                        "mode": "retrieve_query",
                        "retrieved_count": len(selected_ids),
                        "simmax": float((ret_result or {}).get("simmax", 0.0) or 0.0),
                    }
                except Exception:
                    retrieval_trace = {
                        "mode": "retrieve_query",
                        "retrieved_count": len(selected_ids),
                        "simmax": 0.0,
                    }
            except Exception:
                logger.debug("BCB retrieval failed", exc_info=True)

        return selected_ids, mem_context, retrieval_trace, retrieved_topk_queries

    # ==================== Inline Failure Summary ====================

    def configure_failure_summary(self, n_slots: int = 1, inline_k: Optional[int] = None):
        """Enable inline failure summary injection (no region required).

        After retrieval, reserves n_slots for failure memories, then aggregates
        their FAILURE_MODE/MISTAKES/FIXES fields into a frequency-based summary.
        """
        self._failure_summary_n_slots = n_slots
        self._failure_summary_inline_k = inline_k
        logger.info(
            "[Failure Summary] enabled (inline): n_slots=%d, inline_k=%s",
            n_slots, inline_k,
        )

    def _inject_failure_summary(self, selected_mems: list, prompt: str) -> list:
        """Post-process selected_mems: ensure failure slots and replace with inline summary."""
        n_slots = self._failure_summary_n_slots
        if n_slots <= 0 or not selected_mems:
            return selected_mems

        success_mems = []
        failure_mems = []
        for m in selected_mems:
            outcome = self._get_outcome(m)
            if outcome == "failure":
                failure_mems.append(m)
            else:
                success_mems.append(m)

        if len(failure_mems) < n_slots:
            extra_needed = n_slots - len(failure_mems)
            exclude_ids = {m.get("memory_id") or m.get("id") for m in selected_mems}
            extra_failure = self._retrieve_failure_only_bcb(prompt, k=extra_needed, exclude_ids=exclude_ids)
            failure_mems.extend(extra_failure)

        max_success = max(0, self.retrieve_k - n_slots)
        success_mems = success_mems[:max_success]
        failure_mems = failure_mems[:max(n_slots, self._failure_summary_inline_k or n_slots)]

        self._replace_failure_with_inline_summary(failure_mems)

        final_mems = success_mems + failure_mems

        self._failure_inject_log_counter += 1
        if self._failure_inject_log_counter <= 3 or self._failure_inject_log_counter % 50 == 0:
            logger.info(
                "[Failure Summary] task #%d: %d success + %d failure (inline aggregated)",
                self._failure_inject_log_counter, len(success_mems), len(failure_mems),
            )

        return final_mems

    def _replace_failure_with_inline_summary(self, failed_mems: List[Dict]) -> None:
        """Aggregate retrieved failure mems into a single inline summary.

        Parses FAILURE_MODE/MISTAKES/FIXES from the retrieved failures and
        frequency-aggregates them (same format as region summaries, without
        requiring region clustering).
        """
        from memrl.service.region_manager import RegionManager

        k = self._failure_summary_inline_k
        mems_to_aggregate = failed_mems[:k] if k else failed_mems

        if not mems_to_aggregate:
            return

        fields_list = []
        for fm in mems_to_aggregate:
            content = fm.get('content', '')
            if not content:
                continue
            fields = RegionManager._parse_failure_fields(content)
            if fields["failure_mode"] or fields["mistakes"]:
                fields_list.append(fields)

        if not fields_list:
            return

        summary = RegionManager._format_failure_summary(fields_list, top_n=3)
        if not summary:
            return

        failed_mems[0]['content'] = summary
        failed_mems[0]['_region_failure_summary'] = True
        del failed_mems[1:]

    def _retrieve_failure_only_bcb(self, prompt: str, k: int = 2,
                                   exclude_ids: Optional[set] = None) -> list:
        """Retrieve top-k failure memories by similarity only (no Q rerank)."""
        import math

        if not hasattr(self.mem, 'dict_memory') or not self.mem.dict_memory:
            return []
        if not hasattr(self.mem, '_mem_cache'):
            return []

        try:
            embed = getattr(self.mem.embedding_provider, 'embed', None)
            if not callable(embed):
                return []

            _qe = getattr(self.mem, 'query_embeddings', {})
            query_vec = _qe.get(prompt)
            if query_vec is None:
                vecs = embed([prompt])
                query_vec = vecs[0] if vecs else None
            if query_vec is None:
                return []

            query_norm = math.sqrt(sum(x * x for x in query_vec)) or 1e-8

            candidates = []
            mc = self.mem._mem_cache
            for query_key, mem_ids in self.mem.dict_memory.items():
                qv = _qe.get(query_key)
                if qv is None:
                    continue
                q_norm = math.sqrt(sum(x * x for x in qv)) or 1e-8
                sim = sum(a * b for a, b in zip(query_vec, qv)) / (query_norm * q_norm)
                for mid in mem_ids:
                    if exclude_ids and mid in exclude_ids:
                        continue
                    mem_obj = mc.get(mid)
                    if mem_obj is None:
                        continue
                    outcome = self._get_outcome_from_cache(mem_obj)
                    if outcome != "failure":
                        continue
                    content = self._get_content_from_cache(mem_obj)
                    candidates.append({
                        "memory_id": mid,
                        "content": content,
                        "similarity": sim,
                        "metadata": getattr(mem_obj, "metadata", None),
                        "memory_item": mem_obj,
                    })

            candidates.sort(key=lambda x: x["similarity"], reverse=True)
            return candidates[:k]
        except Exception:
            logger.warning("[Failure Summary] failure-only retrieval failed", exc_info=True)
            return []

    @staticmethod
    def _get_outcome(mem_dict: dict) -> str:
        """Extract outcome ('success'/'failure'/None) from a retrieved memory dict."""
        meta = mem_dict.get("metadata")
        if meta is None:
            return ""
        if hasattr(meta, "model_extra"):
            raw = (getattr(meta, "model_extra", {}) or {}).get("outcome", "")
        elif isinstance(meta, dict):
            raw = meta.get("outcome", "")
        else:
            return ""
        return str(raw).strip().lower() if raw else ""

    @staticmethod
    def _get_outcome_from_cache(mem_obj) -> str:
        """Extract outcome from a _mem_cache entry (pydantic or dict)."""
        meta = getattr(mem_obj, "metadata", None)
        if meta is None and isinstance(mem_obj, dict):
            meta = mem_obj.get("metadata")
        if meta is None:
            return ""
        if hasattr(meta, "model_extra"):
            raw = (getattr(meta, "model_extra", {}) or {}).get("outcome", "")
        elif isinstance(meta, dict):
            raw = meta.get("outcome", "")
        else:
            return ""
        return str(raw).strip().lower() if raw else ""

    @staticmethod
    def _get_content_from_cache(mem_obj) -> str:
        """Extract content from a _mem_cache entry."""
        if isinstance(mem_obj, dict):
            return mem_obj.get("content", "") or ""
        meta = getattr(mem_obj, "metadata", None)
        if meta is not None:
            if hasattr(meta, "model_extra"):
                c = (getattr(meta, "model_extra", {}) or {}).get("full_content", "")
                if c:
                    return c
            elif isinstance(meta, dict):
                c = meta.get("full_content", "")
                if c:
                    return c
        return getattr(mem_obj, "memory", "") or ""

    def _process_single_task(self, task_id: str, epoch: int, phase: str) -> Dict[str, Any]:
        """Process one task: retrieve → generate → eval. Returns result dict for sequential bookkeeping."""
        task = self._problems[task_id]
        prompt = get_prompt(task, split=self.sel.split)

        # Retrieve
        selected_ids, mem_context, retrieval_trace, retrieved_topk_queries = self._retrieve_for_task(prompt)

        # Generate
        raw_response = self._generate_raw(prompt, memory_context=mem_context)
        code = extract_code_from_response(raw_response, strip_think=self.strip_think)

        # Eval
        eval_res = self._evaluate_one(task=task, code=code)
        ok = eval_res.get("status") == "PASS"

        return {
            "task_id": task_id,
            "task": task,
            "prompt": prompt,
            "selected_ids": selected_ids,
            "mem_context": mem_context,
            "retrieval_trace": retrieval_trace,
            "retrieved_topk_queries": retrieved_topk_queries,
            "raw_response": raw_response,
            "code": code,
            "eval_res": eval_res,
            "ok": ok,
        }

    def _build_memory_metadata(
        self, res: Dict[str, Any], task_id: str, epoch: int, phase: str, ok: bool
    ) -> Dict[str, Any]:
        """Build metadata dict for a task result. Override in subclasses to add fields."""
        task = res["task"]
        return {
            "source_benchmark": "bigcodebench",
            "success": bool(ok),
            "task_id": task_id,
            "outcome": "success" if ok else "failure",
            "outcome_success": bool(ok),
            "entry_point": str(task.get("entry_point", "")) if isinstance(task, dict) else "",
            "libs": (task.get("libs") if isinstance(task, dict) else None),
            "domains": self._get_task_domains(task) if isinstance(task, dict) else [],
            "source": "conversation",
            "eval_status": res["eval_res"].get("status"),
            "eval_error": res["eval_res"].get("error"),
            "bcb_epoch": epoch,
            "phase": phase,
            "model": self.model_name,
        }

    def _post_batch_hook(self, **kwargs) -> None:
        """Hook called after each batch in _run_phase. Override in subclasses."""
        pass

    def _pre_epoch_hook(self, epoch: int) -> None:
        """Hook called at the start of each epoch. Override in subclasses."""
        pass

    def _post_train_hook(self, epoch: int, epoch_dir: str) -> None:
        """Hook called after train phase, before val phase. Override in subclasses."""
        pass

    def _post_data_load_hook(self) -> None:
        """Hook called after data loading and train/val split. Override in subclasses."""
        pass

    def _precompute_query_embeddings(self, task_ids: List[str]) -> None:
        """Pre-compute and cache query embeddings for given tasks.

        Ensures val retrieval is deterministic across vLLM instances by
        caching embeddings into self.mem.query_embeddings (persisted in checkpoint).
        """
        embed_fn = getattr(getattr(self.mem, 'embedding_provider', None), 'embed', None)
        if not callable(embed_fn):
            return
        if not hasattr(self.mem, 'query_embeddings') or self.mem.query_embeddings is None:
            self.mem.query_embeddings = {}
        qe = self.mem.query_embeddings

        prompts_to_embed = []
        for tid in task_ids:
            task = self._problems.get(tid)
            if task is None:
                continue
            prompt = get_prompt(task, split=self.sel.split)
            if prompt not in qe:
                prompts_to_embed.append(prompt)

        if not prompts_to_embed:
            return

        logger.info("Pre-computing %d query embeddings for deterministic retrieval", len(prompts_to_embed))
        BATCH = 32
        for i in range(0, len(prompts_to_embed), BATCH):
            batch = prompts_to_embed[i:i + BATCH]
            try:
                vecs = embed_fn(batch)
                for prompt, vec in zip(batch, vecs):
                    qe[prompt] = vec if isinstance(vec, list) else vec.tolist()
            except Exception as e:
                logger.warning("Query embedding batch failed at %d: %s", i, e)


    # -------------------------- I/O helpers --------------------------

    @staticmethod
    def _save_json(path: str, obj: Any) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, default=str)

    # -------------------------- evaluation --------------------------

    def _evaluate_one(self, *, task: Dict[str, Any], code: str) -> Dict[str, Any]:
        """Evaluate one solution using official BigCodeBench untrusted_check."""
        task_id = str(task.get("task_id", "unknown"))
        entry_point = str(task.get("entry_point", "task_func"))
        test_code = str(task.get("test", "") or "")

        if not test_code:
            return {"task_id": task_id, "status": "SYNTAX_OK", "error": "no_test_code"}

        # quick syntax check
        try:
            compile(code, "<string>", "exec")
        except SyntaxError as e:
            return {"task_id": task_id, "status": "SYNTAX_ERROR", "error": str(e)}

        # sanitize for evaluation robustness (best-effort)
        clean_code = sanitize_code(code, entry_point, bcb_repo=self.bcb_repo)

        # Calibrated evaluation: prepend code_prompt stub so test harness finds entry_point
        # (matches official BCB evaluate.py line 306, calibrated=True by default)
        code_prompt = str(task.get("code_prompt", ""))
        if code_prompt:
            clean_code = code_prompt + "\n    pass\n" + clean_code

        from bigcodebench.eval import PASS, FAIL, TIMEOUT  # type: ignore

        stat, details, err, hard_timed_out = run_untrusted_check_with_hard_timeout(
            code=clean_code,
            test_code=test_code,
            entry_point=entry_point,
            max_as_limit=30 * 1024,
            max_data_limit=30 * 1024,
            max_stack_limit=10,
            min_time_limit=1.0,
            gt_time_limit=float(self.eval_timeout_s),
            hard_timeout_s=float(self.untrusted_hard_timeout_s),
            bcb_repo=self.bcb_repo,
        )

        if hard_timed_out:
            return {"task_id": task_id, "status": "TIMEOUT", "error": err or "hard_timeout"}
        if err:
            return {"task_id": task_id, "status": "RUNTIME_ERROR", "error": err}
        if stat == PASS:
            return {"task_id": task_id, "status": "PASS"}
        if stat == TIMEOUT:
            return {"task_id": task_id, "status": "TIMEOUT", "error": "timeout"}
        if stat == FAIL:
            # Keep details small; they can be very long.
            return {"task_id": task_id, "status": "FAIL", "error": str(details)[:500] if details else "fail"}
        return {"task_id": task_id, "status": "UNKNOWN", "error": str(stat)}

    # -------------------------- checkpoint helpers --------------------------

    def _rotate_step_checkpoints(self, snapshot_parent: str) -> None:
        """Keep only the latest max_checkpoints step-level snapshots."""
        if not os.path.isdir(snapshot_parent):
            return
        step_dirs = []
        for name in os.listdir(snapshot_parent):
            if name.startswith("step_") and os.path.isdir(os.path.join(snapshot_parent, name)):
                try:
                    step_num = int(name.split("_", 1)[1])
                    step_dirs.append((step_num, name))
                except (ValueError, IndexError):
                    continue
        step_dirs.sort()
        while len(step_dirs) > self.max_checkpoints:
            _, old_name = step_dirs.pop(0)
            old_path = os.path.join(snapshot_parent, old_name)
            try:
                shutil.rmtree(old_path)
                logger.info("Rotated old checkpoint: %s", old_path)
            except Exception:
                logger.warning("Failed to remove old checkpoint: %s", old_path, exc_info=True)

    def _save_incremental_checkpoint(
        self, *, phase_dir: str, epoch_dir: str, idx: int, samples: List[Dict[str, Any]]
    ) -> None:
        """Save a mid-epoch checkpoint: memory snapshot + partial samples."""
        ckpt_id = f"step_{idx}"
        try:
            self.mem.save_checkpoint_snapshot(epoch_dir, ckpt_id=ckpt_id)
            logger.info("Saved incremental checkpoint: %s (step %d)", ckpt_id, idx)
        except Exception:
            logger.warning("Failed to save incremental checkpoint at step %d", idx, exc_info=True)
        partial_path = os.path.join(phase_dir, "samples_partial.jsonl")
        try:
            write_samples(samples, partial_path)
        except Exception:
            logger.warning("Failed to write partial samples at step %d", idx, exc_info=True)
        snapshot_parent = os.path.join(epoch_dir, "snapshot")
        self._rotate_step_checkpoints(snapshot_parent)

    # -------------------------- phases --------------------------

    def _run_phase(
        self,
        *,
        epoch: int,
        phase: str,
        task_ids: List[str],
        epoch_dir: str,
        update_memory: bool,
        start_idx: int = 0,
        phase_dir_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        assert phase in {"train", "val"}
        phase_dir = phase_dir_override or os.path.join(epoch_dir, phase)
        os.makedirs(phase_dir, exist_ok=True)

        samples: List[Dict[str, Any]] = []
        retrieval_logs: List[Dict[str, Any]] = []

        pass_count = 0
        total = len(task_ids)

        # If resuming mid-epoch, reload partial samples from previous run
        if start_idx > 0:
            partial_path = os.path.join(phase_dir, "samples_partial.jsonl")
            # If partial samples don't exist locally, try to copy from the
            # checkpoint's original run directory.
            if not os.path.isfile(partial_path) and self.resume_checkpoint_path:
                try:
                    # checkpoint path: .../epochN/snapshot/step_X
                    ckpt_p = Path(self.resume_checkpoint_path)
                    old_epoch_dir = ckpt_p.parent.parent  # .../epochN
                    old_partial = old_epoch_dir / phase / "samples_partial.jsonl"
                    if old_partial.is_file():
                        import shutil
                        os.makedirs(phase_dir, exist_ok=True)
                        shutil.copy2(str(old_partial), partial_path)
                        logger.info(
                            "Copied partial samples from old run: %s -> %s",
                            old_partial, partial_path,
                        )
                except Exception:
                    logger.warning("Failed to copy partial samples from checkpoint dir", exc_info=True)

            if os.path.isfile(partial_path):
                try:
                    with open(partial_path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                s = json.loads(line)
                                samples.append(s)
                                if s.get("status") == "PASS":
                                    pass_count += 1
                    if len(samples) < start_idx:
                        logger.warning(
                            "Partial samples (%d) < start_idx (%d), adjusting start_idx",
                            len(samples), start_idx,
                        )
                        start_idx = len(samples)
                    logger.info(
                        "Resumed %d partial samples (pass=%d) from %s",
                        len(samples), pass_count, partial_path,
                    )
                except Exception:
                    logger.warning("Failed to load partial samples, starting fresh", exc_info=True)
                    samples.clear()
                    pass_count = 0
                    start_idx = 0
            else:
                logger.warning("No partial samples file found at %s, starting from 0", partial_path)
                start_idx = 0

        # Buffered memory + Q updates (mini-batch-like), aligned with other runners.
        pending_task_descriptions: List[str] = []
        pending_trajectories: List[str] = []
        pending_successes: List[bool] = []
        pending_retrieved_ids: List[List[str]] = []
        pending_retrieved_queries: List[Optional[List[Tuple[str, float]]]] = []
        pending_metadatas: List[Dict[str, Any]] = []
        pending_target_subtasks: List[Optional[str]] = []

        # TensorBoard aggregation (throttled to the same cadence as memory flushes)
        tb_window_tasks = 0
        tb_retrieved_sum = 0
        tb_simmax_sum = 0.0

        def _flush_memory_updates(step_idx: Optional[int] = None) -> None:
            if not update_memory or not pending_task_descriptions or self.mem is None:
                return
            step_idx = int(step_idx or len(pending_task_descriptions))
            try:
                ts = pending_target_subtasks if any(x is not None for x in pending_target_subtasks) else None
                updated = self.mem.update_values(
                    [float(s) for s in pending_successes], pending_retrieved_ids,
                    target_subtasks=ts,
                )
                # Log Q update summary stats (best-effort).
                if isinstance(updated, dict) and updated:
                    vals = [v for v in updated.values() if isinstance(v, (int, float))]
                    if vals:
                        self._tb_add_scalar(
                            f"bcb/{phase}/q_updates/count", len(vals), step=step_idx
                        )
                        self._tb_add_scalar(
                            f"bcb/{phase}/q_updates/mean",
                            sum(vals) / float(len(vals)),
                            step=step_idx,
                        )
                        self._tb_add_scalar(
                            f"bcb/{phase}/q_updates/min", min(vals), step=step_idx
                        )
                        self._tb_add_scalar(
                            f"bcb/{phase}/q_updates/max", max(vals), step=step_idx
                        )
            except Exception:
                logger.debug("BCB Q update failed (batch)", exc_info=True)
            try:
                self.mem.add_memories(
                    task_descriptions=pending_task_descriptions,
                    trajectories=pending_trajectories,
                    successes=pending_successes,
                    retrieved_memory_queries=pending_retrieved_queries,
                    retrieved_memory_ids_list=pending_retrieved_ids,
                    metadatas=pending_metadatas,
                )
            except Exception:
                logger.warning("BCB add_memories failed (batch)", exc_info=True)
            finally:
                pending_task_descriptions.clear()
                pending_trajectories.clear()
                pending_successes.clear()
                pending_retrieved_ids.clear()
                pending_retrieved_queries.clear()
                pending_metadatas.clear()
                pending_target_subtasks.clear()

        for batch_start in range(0, total, self.batch_size):
            batch_task_ids = task_ids[batch_start : batch_start + self.batch_size]
            batch_end = batch_start + len(batch_task_ids)

            # Skip already-resumed tasks
            if batch_end <= start_idx:
                continue
            effective_batch = [
                tid for i, tid in enumerate(batch_task_ids, start=batch_start + 1)
                if i > start_idx
            ]
            if not effective_batch:
                continue

            # Parallel: retrieve + generate + eval for all tasks in batch
            results_map: Dict[str, Dict[str, Any]] = {}
            if self.batch_size > 1 and len(effective_batch) > 1:
                with ThreadPoolExecutor(max_workers=min(self.batch_size, len(effective_batch))) as executor:
                    futures = {
                        executor.submit(self._process_single_task, tid, epoch, phase): tid
                        for tid in effective_batch
                    }
                    for future in as_completed(futures):
                        tid = futures[future]
                        try:
                            results_map[tid] = future.result()
                        except Exception:
                            logger.warning("Task %s failed in parallel execution", tid, exc_info=True)
                            results_map[tid] = None  # type: ignore[assignment]
            else:
                for tid in effective_batch:
                    try:
                        results_map[tid] = self._process_single_task(tid, epoch, phase)
                    except Exception:
                        logger.warning("Task %s failed", tid, exc_info=True)
                        results_map[tid] = None  # type: ignore[assignment]

            # Sequential: collect results IN ORDER, update bookkeeping
            for tid in effective_batch:
                res = results_map.get(tid)
                if res is None:
                    continue

                idx = task_ids.index(tid) + 1
                ok = res["ok"]
                pass_count += 1 if ok else 0

                # TensorBoard aggregations
                if self.retrieve_k > 0:
                    tb_window_tasks += 1
                    tb_retrieved_sum += int(len(res["selected_ids"]))
                    try:
                        tb_simmax_sum += float(res["retrieval_trace"].get("simmax", 0.0) or 0.0)
                    except Exception:
                        pass

                retrieval_logs.append(
                    {
                        "task_id": tid,
                        "epoch": epoch,
                        "phase": phase,
                        "selected_ids": res["selected_ids"],
                        "retrieved_topk_queries": res["retrieved_topk_queries"],
                        "threshold": self._get_retrieve_threshold(),
                    }
                )

                sample = {
                    "task_id": tid,
                    "solution": res["code"],
                    "prompt": res["prompt"],
                    "raw_response": res["raw_response"],
                    "epoch": epoch,
                    "phase": phase,
                    "model": self.model_name,
                    "status": res["eval_res"].get("status"),
                    "error": res["eval_res"].get("error"),
                    "domains": self._get_task_domains(res["task"]) if isinstance(res["task"], dict) else [],
                    "selected_ids": res.get("selected_ids", []),
                    "mem_context": res.get("mem_context", ""),
                }
                samples.append(sample)

                if update_memory:
                    task = res["task"]
                    meta = self._build_memory_metadata(res, tid, epoch, phase, ok)
                    retrieval_for_traj = None
                    if res["selected_ids"] or res["retrieval_trace"]:
                        retrieval_for_traj = {
                            "selected_ids": list(res["selected_ids"]),
                            "num_retrieved": len(res["selected_ids"]),
                            "trace": res["retrieval_trace"],
                        }
                    pending_task_descriptions.append(res["prompt"])
                    pending_trajectories.append(
                        self._trajectory_from_raw_or_fallback(
                            raw_response=res["raw_response"],
                            prompt=res["prompt"],
                            code=res["code"],
                            eval_res=res["eval_res"],
                            retrieval=retrieval_for_traj,
                        )
                    )
                    pending_successes.append(bool(ok))
                    pending_retrieved_ids.append(list(res["selected_ids"]))
                    pending_retrieved_queries.append(res["retrieved_topk_queries"])
                    pending_metadatas.append(meta)
                    pending_target_subtasks.append(res.get("target_subtask"))

            # Flush memory if buffer is large enough
            if len(pending_task_descriptions) >= 25:
                _flush_memory_updates(step_idx=(epoch - 1) * max(total, 1) + batch_end)

            # Post-batch hook (for subclass extensions like mid-epoch clustering)
            self._post_batch_hook(epoch=epoch, phase=phase, batch_end=batch_end,
                                  update_memory=update_memory, total=total)

            # Checkpoint
            if batch_end % self.checkpoint_interval < self.batch_size or batch_end >= total:
                if batch_end % self.checkpoint_interval < self.batch_size:
                    self._save_incremental_checkpoint(
                        phase_dir=phase_dir, epoch_dir=epoch_dir, idx=batch_end, samples=samples,
                    )

            # Progress log
            processed = len(samples)
            logger.info("[bcb] epoch %d %s %d/%d pass=%d", epoch, phase, processed, total, pass_count)
            step = (epoch - 1) * max(total, 1) + processed
            self._tb_add_scalar(f"bcb/{phase}/processed", processed, step=step)
            self._tb_add_scalar(f"bcb/{phase}/pass", pass_count, step=step)
            self._tb_add_scalar(
                f"bcb/{phase}/pass_at_1",
                (pass_count / float(processed)) if processed else 0.0,
                step=step,
            )

            if self.retrieve_k > 0:
                denom = max(1, tb_window_tasks)
                self._tb_add_scalar(
                    f"bcb/{phase}/retrieved_count_avg",
                    tb_retrieved_sum / float(denom),
                    step=step,
                )
                self._tb_add_scalar(
                    f"bcb/{phase}/simmax_avg",
                    tb_simmax_sum / float(denom),
                    step=step,
                )
                tb_window_tasks = 0
                tb_retrieved_sum = 0
                tb_simmax_sum = 0.0

        _flush_memory_updates(step_idx=(epoch - 1) * max(total, 1) + total)

        # Clear any stale retrieval buffer entries (e.g., from val phase where
        # update_values is not called). Prevents cross-phase contamination.
        if hasattr(self.mem, '_retrieval_subtask_buffer'):
            self.mem._retrieval_subtask_buffer.clear()

        samples_path = os.path.join(phase_dir, "samples.jsonl")
        write_samples(samples, samples_path)
        self._save_json(
            os.path.join(phase_dir, "metrics.json"),
            {
                "epoch": epoch,
                "phase": phase,
                "subset": self.sel.subset,
                "split": self.sel.split,
                "model": self.model_name,
                "total": total,
                "pass": pass_count,
                "pass@1": (pass_count / total) if total else None,
                "timestamp": datetime.now().isoformat(),
            },
        )

        # store retrieval traces (useful for debugging)
        write_samples(retrieval_logs, os.path.join(phase_dir, "memory_retrieval.jsonl"))

        return {
            "total": total,
            "pass": pass_count,
            "pass@1": (pass_count / total) if total else None,
            "samples_path": samples_path,
        }

    # ==================== Self-RAG Critique ====================

    def _self_rag_critique(self, question: str, selected_mems: List[Dict[str, Any]], inject_k: int) -> List[Dict[str, Any]]:
        """Use LLM to judge relevance of each retrieved memory, discard irrelevant ones."""
        if not selected_mems:
            return []
        numbered = []
        for i, m in enumerate(selected_mems):
            content = m.get('content') or m.get('full_content') or ''
            numbered.append(f"[Memory {i+1}]\n{content[:2000]}")
        critique_prompt = (
            "You are a relevance judge. Given a coding task and a list of retrieved memories from past problem-solving attempts, "
            "decide which memories are RELEVANT and could help solve the current task.\n\n"
            f"Task: {question[:2000]}\n\n"
            "Retrieved memories:\n" + "\n\n".join(numbered) + "\n\n"
            "Return ONLY a JSON list of the relevant memory numbers (1-indexed). "
            "If none are relevant, return an empty list: []\n"
            "Example: [1, 3]"
        )
        try:
            resp = self.llm.generate(
                messages=[{"role": "user", "content": critique_prompt}],
                temperature=0.0,
                max_tokens=256,
            )
            match = re.search(r'\[[\d\s,]*\]', resp or "")
            if match:
                indices = json.loads(match.group())
                filtered = []
                for idx in indices:
                    if 1 <= idx <= len(selected_mems):
                        filtered.append(selected_mems[idx - 1])
                logger.info("[Self-RAG] Critique kept %d/%d memories", len(filtered), len(selected_mems))
                return filtered
            logger.info("[Self-RAG] Critique returned no valid indices, using all %d memories", len(selected_mems))
        except Exception as e:
            logger.warning("[Self-RAG] Critique failed (%s), using all %d memories", e, len(selected_mems))
        return selected_mems

    # ==================== Pass@K Baseline ====================

    def _run_passk_baseline(self) -> None:
        """Run pass@k baseline: k independent attempts per task, track cumulative pass rate."""
        total_tasks = len(self._train_ids)
        if total_tasks == 0:
            logger.warning("No train data for pass@k baseline; aborting.")
            return

        baseline_dir = os.path.join(self.output_dir, "baseline_passk")
        os.makedirs(baseline_dir, exist_ok=True)

        solved: set = set()
        summary = []
        result_path = os.path.join(baseline_dir, "results.jsonl")
        summary_path = os.path.join(baseline_dir, "summary.json")

        if self.baseline_resume_results and not os.path.exists(result_path):
            import shutil
            source = self.baseline_resume_results
            if not os.path.isfile(source):
                raise FileNotFoundError(f"pass@k resume results not found: {source}")
            shutil.copy2(source, result_path)
            logger.info("[pass@k resume] Seeded fresh run from external results: %s", source)

        start_round = 1
        completed_in_round: Dict[int, set] = {}
        if os.path.isfile(result_path):
            try:
                with open(result_path, "r", encoding="utf-8") as f:
                    for line in f:
                        rec = json.loads(line)
                        rd = int(rec.get("round", 0))
                        key = rec.get("task_id", "")
                        if rd not in completed_in_round:
                            completed_in_round[rd] = set()
                        if key:
                            completed_in_round[rd].add(key)
                        if rec.get("pass") and key:
                            solved.add(key)
                if completed_in_round:
                    max_round = max(completed_in_round.keys())
                    max_round_count = len(completed_in_round[max_round])
                    if max_round_count >= total_tasks:
                        start_round = max_round + 1
                    else:
                        start_round = max_round
                    logger.info(
                        "[pass@k resume] %d existing results across %d rounds, %d solved. Resuming from round %d.",
                        sum(len(v) for v in completed_in_round.values()), max_round, len(solved), start_round,
                    )
            except Exception as e:
                logger.warning("[pass@k resume] Failed to parse existing results, starting fresh: %s", e)
                solved = set()
                completed_in_round = {}
                start_round = 1

        import threading
        _write_lock = threading.Lock()

        for round_idx in range(start_round, self.baseline_k + 1):
            logger.info("[pass@k] Starting round %d/%d", round_idx, self.baseline_k)
            pending_ids = [tid for tid in self._train_ids if tid not in solved]
            if not pending_ids:
                logger.info("[pass@k] All tasks solved before round %d", round_idx)
                cum_sr = len(solved) / total_tasks if total_tasks else 0.0
                summary.append({"round": round_idx, "cum_sr": cum_sr, "solved": len(solved), "total": total_tasks})
                continue

            already_done = completed_in_round.get(round_idx, set())
            remaining_ids = [tid for tid in pending_ids if tid not in already_done]
            if not remaining_ids:
                logger.info("[pass@k round %d] All items already evaluated.", round_idx)
                cum_sr = len(solved) / total_tasks if total_tasks else 0.0
                summary.append({"round": round_idx, "cum_sr": cum_sr, "solved": len(solved), "total": total_tasks})
                continue

            logger.info("[pass@k round %d] %d pending, %d remaining to evaluate", round_idx, len(pending_ids), len(remaining_ids))

            for batch_start in range(0, len(remaining_ids), self.batch_size):
                batch_ids = remaining_ids[batch_start:batch_start + self.batch_size]
                results_map: Dict[str, Optional[Dict[str, Any]]] = {}

                if len(batch_ids) > 1:
                    from concurrent.futures import ThreadPoolExecutor, as_completed as _ac
                    with ThreadPoolExecutor(max_workers=min(self.batch_size, len(batch_ids))) as executor:
                        futures = {executor.submit(self._process_single_task, tid, round_idx, "passk"): tid for tid in batch_ids}
                        for future in _ac(futures):
                            tid = futures[future]
                            try:
                                results_map[tid] = future.result()
                            except Exception:
                                logger.warning("Task %s failed in pass@k round %d", tid, round_idx, exc_info=True)
                                results_map[tid] = None
                else:
                    for tid in batch_ids:
                        try:
                            results_map[tid] = self._process_single_task(tid, round_idx, "passk")
                        except Exception:
                            logger.warning("Task %s failed in pass@k round %d", tid, round_idx, exc_info=True)
                            results_map[tid] = None

                for tid in batch_ids:
                    res = results_map.get(tid)
                    if res is None:
                        continue
                    ok = res["ok"]
                    if ok:
                        solved.add(tid)
                    payload = {
                        "round": round_idx,
                        "task_id": tid,
                        "pass": ok,
                        "status": res["eval_res"].get("status"),
                        "error": res["eval_res"].get("error"),
                    }
                    with _write_lock:
                        with open(result_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

                logger.info("[pass@k round %d] batch %d-%d done, solved so far: %d/%d",
                            round_idx, batch_start, batch_start + len(batch_ids), len(solved), total_tasks)

            cum_sr = len(solved) / total_tasks if total_tasks else 0.0
            summary.append({"round": round_idx, "cum_sr": cum_sr, "solved": len(solved), "total": total_tasks})
            logger.info("[pass@k round %d] Cumulative SR: %.2f%% (%d/%d)", round_idx, cum_sr * 100, len(solved), total_tasks)
            self._tb_add_scalar("Baseline/PassK_Cumulative_SR", cum_sr, round_idx)

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info("[pass@k] Done. Final SR: %.2f%% (%d/%d)", (len(solved) / total_tasks * 100) if total_tasks else 0, len(solved), total_tasks)

    # ==================== Reflection Baseline ====================

    def _run_reflection_baseline(self) -> None:
        """Run reflection baseline: k rounds, each failed task gets a reflection note for next round."""
        total_tasks = len(self._train_ids)
        if total_tasks == 0:
            logger.warning("No train data for reflection baseline; aborting.")
            return

        baseline_dir = os.path.join(self.output_dir, "baseline_reflection")
        os.makedirs(baseline_dir, exist_ok=True)

        solved: set = set()
        summary = []
        reflection_notes: Dict[str, str] = {}
        result_path = os.path.join(baseline_dir, "results.jsonl")
        summary_path = os.path.join(baseline_dir, "summary.json")
        state_path = os.path.join(baseline_dir, "state.json")

        start_round = 1
        if os.path.isfile(state_path):
            try:
                state = json.load(open(state_path, "r", encoding="utf-8"))
                solved = set(state.get("solved", []))
                reflection_notes = {str(k): v for k, v in state.get("reflection_notes", {}).items()}
                start_round = max(1, int(state.get("last_completed_round", 0)) + 1)
                logger.info("[reflection] Resuming from round %d (%d solved)", start_round, len(solved))
            except Exception as e:
                logger.warning("[reflection] Failed to load state: %s", e)

        if start_round > self.baseline_k:
            logger.info("[reflection] Already completed.")
            return

        for round_idx in range(start_round, self.baseline_k + 1):
            logger.info("[reflection] Starting round %d/%d", round_idx, self.baseline_k)
            pending_ids = [tid for tid in self._train_ids if tid not in solved]
            if not pending_ids:
                logger.info("[reflection] All tasks solved before round %d", round_idx)
                cum_sr = len(solved) / total_tasks if total_tasks else 0.0
                summary.append({"round": round_idx, "cum_sr": cum_sr, "solved": len(solved), "total": total_tasks})
                continue

            for batch_start in range(0, len(pending_ids), self.batch_size):
                batch_ids = pending_ids[batch_start:batch_start + self.batch_size]

                for tid in batch_ids:
                    try:
                        note = reflection_notes.get(tid, "")
                        mem_ctx = f"# Reflection from previous attempt\n{note}" if note else ""
                        task = self._problems[tid]
                        prompt = get_prompt(task, split=self.sel.split)
                        raw_response = self._generate_raw(prompt, memory_context=mem_ctx)
                        code = extract_code_from_response(raw_response, strip_think=self.strip_think)
                        eval_res = self._evaluate_one(task=task, code=code)
                        ok = eval_res.get("status") == "PASS"

                        if ok:
                            solved.add(tid)
                        else:
                            reflection_notes[tid] = self._format_reflection_note(prompt, raw_response, ok)

                        payload = {
                            "round": round_idx,
                            "task_id": tid,
                            "pass": ok,
                            "status": eval_res.get("status"),
                            "error": eval_res.get("error"),
                        }
                        with open(result_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
                    except Exception:
                        logger.warning("Task %s failed in reflection round %d", tid, round_idx, exc_info=True)

                logger.info("[reflection round %d] batch %d-%d done, solved: %d/%d",
                            round_idx, batch_start, batch_start + len(batch_ids), len(solved), total_tasks)

            cum_sr = len(solved) / total_tasks if total_tasks else 0.0
            summary.append({"round": round_idx, "cum_sr": cum_sr, "solved": len(solved), "total": total_tasks})
            logger.info("[reflection round %d] Cumulative SR: %.2f%% (%d/%d)", round_idx, cum_sr * 100, len(solved), total_tasks)
            self._tb_add_scalar("Baseline/Reflection_Cumulative_SR", cum_sr, round_idx)

            try:
                with open(state_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "last_completed_round": round_idx,
                        "solved": sorted(solved),
                        "reflection_notes": reflection_notes,
                        "total": total_tasks,
                        "updated_at": time.strftime('%Y-%m-%dT%H:%M:%S'),
                    }, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning("[reflection] Failed to save state: %s", e)

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info("[reflection] Done. Final SR: %.2f%% (%d/%d)", (len(solved) / total_tasks * 100) if total_tasks else 0, len(solved), total_tasks)

    def _format_reflection_note(self, prompt: str, response: str, success: bool) -> str:
        """Generate a reflection note for a failed attempt."""
        if success:
            return ""
        try:
            reflect_prompt = (
                "You attempted to solve the following coding task but failed. "
                "Analyze what went wrong and write a brief reflection note (2-3 sentences) "
                "that could help you solve this task on a future attempt.\n\n"
                f"Task:\n{prompt[:3000]}\n\n"
                f"Your response:\n{response[:3000]}\n\n"
                "Reflection:"
            )
            note = self.llm.generate(
                messages=[{"role": "user", "content": reflect_prompt}],
                temperature=0.0,
                max_tokens=512,
            )
            return (note or "").strip()
        except Exception as e:
            logger.warning("[reflection] Failed to generate note: %s", e)
            return ""

    # ==================== Multi-Eval with CI ====================

    def _run_eval_multi(self, epoch: int, epoch_dir: str) -> Dict[str, Any]:
        """Run val phase n_eval_runs times, compute mean ± 95% CI, save per-task results."""
        rates = []
        orig_temp = self.temperature
        for run_idx in range(self.n_eval_runs):
            logger.info("[bcb] epoch %d eval run %d/%d", epoch, run_idx + 1, self.n_eval_runs)
            if self.eval_temperature is not None:
                self.temperature = self.eval_temperature

            val_dir = os.path.join(epoch_dir, f"val_run{run_idx}")
            val_res = self._run_phase(
                epoch=epoch,
                phase="val",
                task_ids=self._val_ids,
                epoch_dir=epoch_dir,
                update_memory=False,
                phase_dir_override=val_dir,
            )
            self.temperature = orig_temp
            rates.append(val_res["pass@1"] or 0.0)

        import numpy as _np
        mean_sr = float(_np.mean(rates))
        ci_half = 0.0
        if len(rates) >= 2:
            import scipy.stats as st
            ci = st.t.interval(0.95, len(rates) - 1, loc=mean_sr, scale=st.sem(rates))
            ci_half = float((ci[1] - ci[0]) / 2)

        summary = {
            "epoch": epoch,
            "n_runs": len(rates),
            "mean_pass@1": mean_sr,
            "ci_95": ci_half,
            "individual_runs": rates,
        }
        self._save_json(os.path.join(epoch_dir, "val_multi_summary.json"), summary)
        self._tb_add_scalar("bcb/val/pass_at_1_mean", mean_sr, step=epoch)
        self._tb_add_scalar("bcb/val/pass_at_1_ci95", ci_half, step=epoch)
        logger.info(
            "[bcb] epoch %d val: mean=%.2f%% ± %.2f%% (95%% CI), runs=%s",
            epoch, mean_sr * 100, ci_half * 100, [f"{r:.2%}" for r in rates],
        )
        return summary

    # -------------------------- public API --------------------------

    def run(self) -> Dict[str, Any]:
        os.makedirs(self.output_dir, exist_ok=True)

        # load problems + split once
        self._problems = load_bcb_data(subset=self.sel.subset, data_path=self.sel.data_path)
        self._train_ids, self._val_ids = split_dataset(
            self._problems,
            train_ratio=self.sel.train_ratio,
            seed=self.sel.seed,
            split_file=self.sel.split_file,
        )

        self._post_data_load_hook()

        # Dispatch baseline modes before normal epoch loop
        if self.baseline_mode in {"passk", "reflection"}:
            if self.baseline_mode == "passk":
                self._run_passk_baseline()
            else:
                self._run_reflection_baseline()
            try:
                self.writer.close()
            except Exception:
                pass
            return {}

        # Resume from checkpoint if specified
        start_epoch = 1
        resume_step = 0
        if self.resume_checkpoint_path:
            logger.info("Loading checkpoint from: %s", self.resume_checkpoint_path)
            try:
                loaded_ckpt = self.mem.load_checkpoint_snapshot(self.resume_checkpoint_path)
                logger.info("Checkpoint loaded, ckpt_id=%s", loaded_ckpt)
            except Exception:
                logger.error("Failed to load checkpoint from %s", self.resume_checkpoint_path, exc_info=True)
                raise

            if self.resume_epoch is not None:
                if self.resume_step is not None:
                    start_epoch = self.resume_epoch
                    resume_step = self.resume_step
                    logger.info("Resuming epoch %d from step %d", start_epoch, resume_step)
                else:
                    start_epoch = self.resume_epoch + 1
                    logger.info("Resuming from epoch %d (completed epoch %d)", start_epoch, self.resume_epoch)
            else:
                start_epoch = (int(loaded_ckpt) if loaded_ckpt else 0) + 1
                logger.info("Auto-detected resume from epoch %d", start_epoch)

        run_cfg = {
            "subset": self.sel.subset,
            "split": self.sel.split,
            "train_ratio": self.sel.train_ratio,
            "seed": self.sel.seed,
            "num_epochs": self.num_epochs,
            "run_validation": self.run_validation,
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "retrieve_k": self.retrieve_k,
            "retrieve_threshold": self._get_retrieve_threshold(),
            "bcb_repo": self.bcb_repo,
            "resume_from": self.resume_checkpoint_path,
            "resume_epoch": self.resume_epoch,
            "resume_step": self.resume_step,
            "start_epoch": start_epoch,
            "checkpoint_interval": self.checkpoint_interval,
            "max_checkpoints": self.max_checkpoints,
            "created_at": datetime.now().isoformat(),
        }
        self._save_json(os.path.join(self.output_dir, "run_config.json"), run_cfg)

        # Pre-compute query embeddings for all tasks (train + val) once.
        # Cached in self.mem.query_embeddings → persisted in checkpoint.
        # Saves repeated embedding API calls and ensures deterministic retrieval.
        self._precompute_query_embeddings(self._train_ids)
        if self.run_validation:
            self._precompute_query_embeddings(self._val_ids)

        epoch_summaries: List[Dict[str, Any]] = []
        for epoch in range(start_epoch, self.num_epochs + 1):
            self._pre_epoch_hook(epoch=epoch)

            epoch_dir = os.path.join(self.output_dir, f"epoch{epoch}")
            os.makedirs(epoch_dir, exist_ok=True)

            phase_start_idx = resume_step if (epoch == start_epoch and resume_step > 0) else 0

            train_res = self._run_phase(
                epoch=epoch,
                phase="train",
                task_ids=self._train_ids,
                epoch_dir=epoch_dir,
                update_memory=True,
                start_idx=phase_start_idx,
            )

            self._post_train_hook(epoch=epoch, epoch_dir=epoch_dir)

            val_res = None
            if self.run_validation:
                do_multi = False
                if self.n_eval_runs > 1:
                    if self._multi_eval_mode == "all":
                        do_multi = True
                    elif self._multi_eval_mode == "last":
                        do_multi = (epoch == self.num_epochs)
                    elif self.multi_eval_epochs and epoch in self.multi_eval_epochs:
                        do_multi = True

                # Always run standard val (temp=0.0) first
                val_res = self._run_phase(
                    epoch=epoch,
                    phase="val",
                    task_ids=self._val_ids,
                    epoch_dir=epoch_dir,
                    update_memory=False,
                )

                # Then run multi-eval on top (separate runs with eval_temperature)
                if do_multi:
                    self._run_eval_multi(epoch, epoch_dir)

            # per-epoch snapshot
            try:
                self.mem.save_checkpoint_snapshot(epoch_dir, ckpt_id=str(epoch))
            except Exception:
                logger.warning("Failed to save checkpoint snapshot for epoch %d", epoch, exc_info=True)

            epoch_summary = {"epoch": epoch, "train": train_res, "val": val_res}
            self._save_json(os.path.join(epoch_dir, "epoch_summary.json"), epoch_summary)
            epoch_summaries.append(epoch_summary)

        # final snapshot (best-effort)
        try:
            self.mem.save_checkpoint_snapshot(self.output_dir, ckpt_id="final")
        except Exception:
            logger.warning("Failed to save final snapshot", exc_info=True)

        final = {
            "output_dir": self.output_dir,
            "epochs": epoch_summaries,
        }
        self._save_json(os.path.join(self.output_dir, "summary.json"), final)
        try:
            self.writer.close()
        except Exception:
            pass
        return final
