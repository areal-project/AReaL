"""
BCB Holdout Runner for Cross-Subtask Transfer Experiments.

Extends BCBRegionRunner with parallel holdout bank processing:
- Each task is processed once for full bank (normal) and once per holdout bank
- Holdout banks use different memory sets (excluding held-out subtask memories)
- Generation and evaluation run in parallel across banks
- Q updates are routed to appropriate banks via HoldoutRegionManager

The full bank drives memory addition (only full bank adds new memories).
Holdout banks' region utility updates happen automatically inside
HoldoutRegionManager.update_subtask_q (called by parent's _flush_memory_updates).

Per-bank pass counts are logged by overriding _run_phase and collecting
bank_results from each task's result dict.
"""

import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from memrl.run.bcb_region_runner import BCBRegionRunner
from memrl.service.holdout_memory_service import HoldoutRegionMemoryService
from memrl.service.holdout_region_manager import HoldoutRegionManager
from memrl.configs.task_hierarchy import get_primary_subtask

logger = logging.getLogger(__name__)


class BCBHoldoutRunner(BCBRegionRunner):
    """
    BCB runner with parallel holdout bank processing for cross-subtask transfer.

    For each task, _process_single_task:
    1. Full bank: retrieve → generate → eval (becomes the "primary" result for parent)
    2. Per holdout bank: retrieve (excluding held-out subtask mems) → generate → eval
    3. All banks run in parallel via ThreadPoolExecutor

    Parent's _run_phase handles:
    - Bookkeeping (samples, logs) using full bank's result
    - Memory addition using full bank's result
    - Q updates via _flush_memory_updates → update_values → HoldoutRegionManager.update_subtask_q
      which automatically updates all holdout banks

    This class adds:
    - Per-bank pass count tracking (via bank_results in each task's result)
    - Holdout summary logging per phase
    - Holdout metrics in epoch summaries
    """

    def __init__(self, *, memory_service: HoldoutRegionMemoryService, **kwargs):
        if not isinstance(memory_service, HoldoutRegionMemoryService):
            raise TypeError(
                f"BCBHoldoutRunner requires HoldoutRegionMemoryService, got {type(memory_service)}"
            )
        super().__init__(memory_service=memory_service, **kwargs)
        self._holdout_mem: HoldoutRegionMemoryService = memory_service

        # Thread-safe per-bank pass counts
        self._bank_lock = threading.Lock()
        self._bank_pass_counts: Dict[str, int] = {}
        self._bank_total_counts: Dict[str, int] = {}

        # Holdout bank results log (written to file per epoch)
        self._holdout_results_log: List[Dict[str, Any]] = []

    def _process_single_task(self, task_id: str, epoch: int, phase: str) -> Dict[str, Any]:
        """
        Override: process task with all banks in parallel.

        The full bank's result is returned as the primary result (same schema as parent).
        bank_results dict is attached for holdout tracking.
        """
        task = self._problems[task_id]
        prompt = self._get_prompt_for_task(task)
        domains = self._get_task_domains(task)
        target_subtask = get_primary_subtask("bigcodebench", {"domains": domains})

        # Multi-bank retrieval
        retrieval_sets = self._holdout_mem.retrieve_query_multi_bank(
            task_description=prompt,
            target_subtask=target_subtask,
            k=self.retrieve_k,
            threshold=self._get_retrieve_threshold(),
        )

        # Parallel generation + eval for all banks
        bank_results = {}
        bank_names = list(retrieval_sets.keys())

        if len(bank_names) > 1:
            with ThreadPoolExecutor(max_workers=len(bank_names)) as executor:
                futures = {}
                for bank_name in bank_names:
                    retrieval = retrieval_sets[bank_name]
                    future = executor.submit(
                        self._generate_and_eval_one_bank,
                        prompt, retrieval, task, bank_name,
                    )
                    futures[future] = bank_name

                for future in as_completed(futures):
                    bank_name = futures[future]
                    try:
                        bank_results[bank_name] = future.result()
                    except Exception:
                        logger.warning("Bank %s failed for task %s", bank_name, task_id, exc_info=True)
                        bank_results[bank_name] = {
                            "ok": False, "selected_ids": [], "code": "",
                            "raw_response": "", "eval_res": {},
                        }
        else:
            for bank_name in bank_names:
                retrieval = retrieval_sets[bank_name]
                try:
                    bank_results[bank_name] = self._generate_and_eval_one_bank(
                        prompt, retrieval, task, bank_name,
                    )
                except Exception:
                    logger.warning("Bank %s failed for task %s", bank_name, task_id, exc_info=True)
                    bank_results[bank_name] = {
                        "ok": False, "selected_ids": [], "code": "",
                        "raw_response": "", "eval_res": {},
                    }

        # Track per-bank pass counts (thread-safe)
        with self._bank_lock:
            for bank_name, bank_res in bank_results.items():
                ok = bank_res.get("ok", False)
                self._bank_pass_counts[bank_name] = self._bank_pass_counts.get(bank_name, 0) + (1 if ok else 0)
                self._bank_total_counts[bank_name] = self._bank_total_counts.get(bank_name, 0) + 1

            # Log holdout entry with per-bank selected_ids for post-hoc analysis
            holdout_entry = {"task_id": task_id, "target_subtask": target_subtask}
            for bn, br in bank_results.items():
                holdout_entry[f"{bn}_ok"] = br.get("ok", False)
                holdout_entry[f"{bn}_n_retrieved"] = len(br.get("selected_ids", []))
                holdout_entry[f"{bn}_selected_ids"] = br.get("selected_ids", [])
            self._holdout_results_log.append(holdout_entry)

        # Update holdout banks' region utilities with each bank's OWN reward + memory IDs.
        # Full bank Q is updated by parent's _flush_memory_updates via update_values().
        # Holdout banks need separate updates here because parent doesn't know about them.
        if phase == "train":
            self._holdout_mem.update_values_multi_bank(bank_results, target_subtask)

        # Use full bank's result as the primary result
        full_result = bank_results.get("full", {})

        return {
            "task_id": task_id,
            "task": task,
            "prompt": prompt,
            "selected_ids": full_result.get("selected_ids", []),
            "mem_context": full_result.get("mem_context", ""),
            "retrieval_trace": full_result.get("retrieval_trace", {}),
            "retrieved_topk_queries": None,
            "raw_response": full_result.get("raw_response", ""),
            "code": full_result.get("code", ""),
            "eval_res": full_result.get("eval_res", {}),
            "ok": full_result.get("ok", False),
            "target_subtask": target_subtask,
            "bank_results": bank_results,
        }

    def _generate_and_eval_one_bank(
        self,
        prompt: str,
        retrieval: Dict[str, Any],
        task: Dict[str, Any],
        bank_name: str,
    ) -> Dict[str, Any]:
        """Generate and evaluate for a single bank."""
        selected_mems = retrieval.get("selected", [])
        selected_ids = [
            str(m.get("memory_id") or m.get("id"))
            for m in selected_mems
            if isinstance(m, dict) and (m.get("memory_id") or m.get("id"))
        ]
        mem_context = self._format_memory_context(selected_mems)

        raw_response = self._generate_raw(prompt, memory_context=mem_context)
        from memrl.bigcodebench_eval.bcb_adapter import extract_code_from_response
        code = extract_code_from_response(raw_response, strip_think=self.strip_think)

        eval_res = self._evaluate_one(task=task, code=code)
        ok = eval_res.get("status") == "PASS"

        retrieval_trace = {
            "mode": "retrieve_query",
            "retrieved_count": len(selected_ids),
            "simmax": float(retrieval.get("simmax", 0.0) or 0.0),
            "bank": bank_name,
        }

        return {
            "ok": ok,
            "selected_ids": selected_ids,
            "mem_context": mem_context,
            "retrieval_trace": retrieval_trace,
            "raw_response": raw_response,
            "code": code,
            "eval_res": eval_res,
        }

    def _run_phase(self, epoch, phase, task_ids, epoch_dir=None, update_memory=True, start_idx=0):
        """Override to reset counters before phase and log holdout summary after."""
        bank_names = self._holdout_mem.get_holdout_bank_names()

        with self._bank_lock:
            self._bank_pass_counts = {b: 0 for b in bank_names}
            self._bank_total_counts = {b: 0 for b in bank_names}
            self._holdout_results_log = []

        result = super()._run_phase(
            epoch=epoch, phase=phase, task_ids=task_ids,
            epoch_dir=epoch_dir, update_memory=update_memory, start_idx=start_idx,
        )

        # Log per-bank summary
        with self._bank_lock:
            summary_parts = []
            for b in bank_names:
                p = self._bank_pass_counts.get(b, 0)
                t = self._bank_total_counts.get(b, 0)
                sr = f"{100*p/t:.1f}%" if t > 0 else "N/A"
                summary_parts.append(f"{b}={p}/{t}({sr})")

            logger.info(
                "[bcb] epoch %d %s holdout summary: %s",
                epoch, phase, " | ".join(summary_parts),
            )

            # Save holdout results log
            if epoch_dir:
                holdout_log_path = os.path.join(epoch_dir, f"holdout_{phase}_results.jsonl")
                try:
                    with open(holdout_log_path, "w") as f:
                        for entry in self._holdout_results_log:
                            f.write(json.dumps(entry) + "\n")
                    logger.info("Saved holdout results log: %s", holdout_log_path)
                except Exception:
                    logger.warning("Failed to save holdout results log", exc_info=True)

            # Add holdout metrics to result
            if result:
                result["holdout_pass_counts"] = dict(self._bank_pass_counts)
                result["holdout_total_counts"] = dict(self._bank_total_counts)

        return result
