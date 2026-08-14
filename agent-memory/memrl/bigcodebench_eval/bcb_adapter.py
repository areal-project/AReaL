"""
BigCodeBench adapter: wraps memp LLM + MemoryService for code generation.

This keeps the BigCodeBench-specific prompting logic isolated, while allowing
the runner to inject a MemoryService instance (for retrieval + RL updates).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)


def extract_code_from_response(text: str, *, strip_think: bool = False) -> str:
    """Best-effort extraction of Python code from an LLM response.

    Handles multi-block responses (e.g. a solution block followed by an
    "Example usage:" block) by selecting the first block that defines a
    function, rather than stripping the outer fence pair — which merges
    multiple blocks and lets trailing prose leak in as the "solution".
    """
    if not text:
        return ""

    # Strip <think>...</think> reasoning before fence parsing.
    s = re.sub(r"<think>.*?</think>", "", str(text), flags=re.DOTALL).strip()
    if not s:
        return ""

    # Collect all fenced code blocks.
    blocks = re.findall(r"```(?:python|py)?\s*(.*?)```", s, flags=re.DOTALL | re.IGNORECASE)
    blocks = [b.strip() for b in blocks if b.strip()]
    if blocks:
        # Prefer the first block that defines a function (the real solution)
        # over trailing example/demo blocks.
        for b in blocks:
            if re.search(r"^\s*(?:async\s+)?def\s+\w+", b, flags=re.MULTILINE):
                return b
        # No def found — take the longest block (most substantive code).
        return max(blocks, key=len)

    # Handle truncated response: opening fence but no closing fence.
    m = re.match(r"```(?:python|py)?\s*(.*)", s, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return (m.group(1) or "").strip()

    # Fallback: no fences — return as-is.
    return s


class MempBCBDecoder:
    """Generate code for BCB prompts with optional memory augmentation."""

    def __init__(
        self,
        name: str,
        llm_provider: Any,
        mem_service: Any | None,
        *,
        temperature: float = 0.0,
        max_new_tokens: int = 1280,
        system_prompt: str = "",
        retrieve_k: int = 5,
        memory_budget_tokens: int = 0,
    ) -> None:
        self.name = name
        self._llm = llm_provider
        self._mem = mem_service
        self.temperature = float(temperature)
        self.max_new_tokens = int(max_new_tokens)
        self._sys_prompt = str(system_prompt or "")
        self._k = int(retrieve_k)
        self._budget = int(memory_budget_tokens)

        # per-prompt traces (used by runner for logging + RL updates)
        self._last_retrieval: Dict[str, Any] = {}
        self._last_retrievals: List[Dict[str, Any]] = []

    @property
    def last_retrieval(self) -> Dict[str, Any]:
        return self._last_retrieval

    @property
    def last_retrievals(self) -> List[Dict[str, Any]]:
        return self._last_retrievals

    def _format_memory_context(self, candidates: List[Dict[str, Any]]) -> str:
        # Align formatting with memory_rl BigCodeBench adapter.
        if not candidates:
            return ""

        parts: List[str] = ["# Relevant Code Examples from Memory\n"]

        for i, c in enumerate(candidates, 1):
            raw_content = c.get("content", c.get("full_content", "")) or ""

            # Truncate if needed (rough per-entry budget by characters).
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

            content = self._coerce_bcb_memory_content(
                raw_content=str(raw_content or ""),
                outcome=str(outcome or ""),
                task_description=task_desc,
            )
            if not content:
                continue

            # budget=0 means unlimited – skip truncation entirely.
            if self._budget > 0 and len(content) > self._budget // len(candidates):
                content = content[: self._budget // len(candidates)] + "..."

            parts.append(f"## Example {i} [{str(outcome).upper()}]")
            if task_id:
                parts.append(f"Task: {task_id}")
            parts.append(str(content))
            parts.append("")

        return "\n".join(parts)

    @staticmethod
    def _coerce_bcb_memory_content(*, raw_content: str, outcome: str, task_description: str) -> str:
        text = str(raw_content or "").strip()
        if not text:
            return ""
        if "[MEMORY TYPE]" in text.upper():
            return text

        out = str(outcome or "unknown").strip().lower()
        is_failure = out in {"failure", "fail", "failed", "0", "false", "no"}
        if is_failure:
            # Prefer a line that begins with "Reflection:"; avoid matching "TASK REFLECTION:" header.
            m = re.search(r"(?is)(?:^|\n)reflection\s*:\s*(.*)$", text)
            reflection = (m.group(1) if m else text).strip()
            td = task_description.strip() if task_description else ""
            return (
                "[MEMORY TYPE] FAILURE_REFLECTION\n"
                "[TASK]\n"
                f"{td}\n\n"
                "[REFLECTION]\n"
                f"{reflection}"
            ).strip()

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

    def _get_retrieve_threshold(self) -> float:
        """Align with other benchmarks: use mem_service.rl_config.sim_threshold (fallback tau)."""
        try:
            rl_cfg = getattr(self._mem, "rl_config", None)
            if rl_cfg is None:
                return 0.0
            return float(getattr(rl_cfg, "sim_threshold", getattr(rl_cfg, "tau", 0.0)))
        except Exception:
            return 0.0

    def _retrieve_memory(self, query: str) -> Tuple[str, List[str], Dict[str, Any], Any]:
        if not self._mem:
            return "", [], {}, None

        # Align with current service ecosystem: retrieve_query drives selection.
        thr = self._get_retrieve_threshold()
        ret = self._mem.retrieve_query(query, k=self._k, threshold=thr)
        if isinstance(ret, tuple):
            ret_result, topk_queries = ret
        else:
            ret_result, topk_queries = ret, None

        selected = (ret_result or {}).get("selected", []) if ret_result else []
        if not isinstance(selected, list):
            selected = []
        selected_ids = [
            str(c.get("memory_id") or c.get("id"))
            for c in selected
            if isinstance(c, dict) and (c.get("memory_id") or c.get("id"))
        ]
        simmax = 0.0
        try:
            simmax = max(
                (float(c.get("similarity", 0.0) or 0.0) for c in selected), default=0.0
            )
        except Exception:
            simmax = 0.0

        return (
            self._format_memory_context(selected),
            selected_ids,
            {
                "mode": "retrieve_query",
                "k": self._k,
                "threshold": thr,
                "simmax": simmax,
                "selected_count": len(selected_ids),
            },
            topk_queries,
        )

    def _generate_single(self, prompt: str, mem_context: str) -> str:
        sys_prompt = self._sys_prompt
        if mem_context:
            sys_prompt = (sys_prompt + "\n\n" + mem_context).strip()

        messages = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            resp = self._llm.generate(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
            )
        except Exception as e:
            logger.error("LLM generation failed: %s", e, exc_info=True)
            return ""

        return extract_code_from_response(resp)

    def codegen(self, prompts: List[str], do_sample: bool = False, num_samples: int = 1) -> List[List[str]]:
        # BigCodeBench expects List[List[str]] (k samples per prompt).
        out: List[List[str]] = []
        self._last_retrievals = []

        for prompt in prompts:
            mem_context, selected_ids, trace, topk_queries = self._retrieve_memory(prompt)
            code = self._generate_single(prompt, mem_context)
            info = {
                "prompt": prompt,
                "selected_ids": selected_ids,
                "trace": trace,
                "retrieved_topk_queries": topk_queries,
                "num_retrieved": len(selected_ids),
                "memory_context": mem_context,
            }
            self._last_retrieval = info
            self._last_retrievals.append(info)
            out.append([code])

        return out
