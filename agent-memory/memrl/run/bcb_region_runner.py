"""
Region-aware BCB runner that extends BCBRunner with region-based memory.

This runner overrides:
1. _retrieve_for_task: passes target_subtask to retrieve_query for region gating
2. _build_metadata_for_task: adds source_subtask to metadata
3. run: triggers region clustering after each train phase
4. Hooks into parent's update_values to pass target_subtasks
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from memrl.run.bcb_runner import BCBRunner, BCBSelection
from memrl.service.region_memory_service import RegionMemoryService
from memrl.configs.task_hierarchy import get_primary_subtask

logger = logging.getLogger(__name__)


class BCBRegionRunner(BCBRunner):
    """
    BCB runner with region-aware memory retrieval and utility tracking.

    Overrides _retrieve_for_task to inject target_subtask for region gating.
    Uses parent's batch parallelism for performance.
    """

    def __init__(
        self,
        *,
        root: Path,
        selection: BCBSelection,
        llm: Any,
        memory_service: RegionMemoryService,  # Must be RegionMemoryService
        output_dir: str,
        model_name: str,
        holdout_subtask: Optional[str] = None,
        retrieval_mode: str = "current",
        oracle_snapshot_dir: Optional[str] = None,
        val_lambda_max: Optional[float] = None,
        region_topology_cooldown_epochs: int = 0,
        region_disable_mid_epoch_topology: bool = False,
        fixed_initial_topology_epoch: Optional[int] = None,
        **kwargs
    ):
        if not isinstance(memory_service, RegionMemoryService):
            raise TypeError(
                f"BCBRegionRunner requires RegionMemoryService, got {type(memory_service)}"
            )
        if retrieval_mode not in ("current", "no_mem", "oracle"):
            raise ValueError(f"retrieval_mode must be current/no_mem/oracle, got {retrieval_mode}")

        super().__init__(
            root=root,
            selection=selection,
            llm=llm,
            memory_service=memory_service,
            output_dir=output_dir,
            model_name=model_name,
            **kwargs
        )

        self._batch_target_subtasks: List[str] = []
        self.holdout_subtask: Optional[str] = holdout_subtask
        self._holdout_ids: List[str] = []
        self.retrieval_mode: str = retrieval_mode
        self.val_lambda_max: Optional[float] = val_lambda_max
        self.region_topology_cooldown_epochs = max(0, int(region_topology_cooldown_epochs))
        self.region_disable_mid_epoch_topology = bool(region_disable_mid_epoch_topology)
        self.fixed_initial_topology_epoch = (
            max(1, int(fixed_initial_topology_epoch))
            if fixed_initial_topology_epoch is not None else None
        )
        self._fixed_initial_topology_marked = False
        self.oracle_snapshot_dir: Optional[str] = oracle_snapshot_dir
        self._oracle_memory_pool: Optional[List[Dict[str, Any]]] = None
        if retrieval_mode == "oracle":
            if not oracle_snapshot_dir:
                raise ValueError("oracle_snapshot_dir is required when retrieval_mode='oracle'")
            from memrl.run.oracle_retrieval import load_memory_pool
            self._oracle_memory_pool = load_memory_pool(oracle_snapshot_dir)
            logger.info("Oracle mode: loaded %d memories from %s",
                        len(self._oracle_memory_pool), oracle_snapshot_dir)

        # Region failure summary injection (disabled by default).
        # Call configure_failure_summary() after construction to enable.
        self._failure_summary_n_slots = 0
        self._failure_summary_replace = True
        self._failure_summary_lib_filter = False
        # Conservative BCB-specific gate: only inject a region FS when failures
        # share a minimal observable task contract with the query. Disabled by
        # default so historical runs retain their exact behavior.
        self._failure_summary_contract_filter = False
        self._failure_summary_force_recall = False
        # Prior behavior injects with one compatible reflection. Higher values
        # provide a precision/abstention gate for BCB's sparse exact contracts.
        self._failure_summary_min_compatible = 1
        # Failure-mode-matched conditional summary (new design). When True, for a task
        # with prior-epoch failures, retrieve region failures whose FAILURE_MODE is
        # semantically similar to THIS task's own prior failure mode, then synthesize a
        # conditional (if/elif/else) contract summary via LLM. See docs/experiments/bcb.
        self._failure_summary_fmmatch = False
        self._fmmatch_topk = 5
        self._fmmatch_backfill = False
        self._failure_inject_log_counter = 0

        # Region experience cards injection (disabled by default).
        # Call configure_experience_cards() after construction to enable.
        self._experience_cards_n_slots = 0
        self._experience_cards_log_counter = 0

        # Retrieval gate (disabled by default).
        # Call configure_retrieval_gate() after construction to enable.
        self._retrieval_gate_enabled = False
        self._retrieval_gate_signal = "success_ratio"
        self._retrieval_gate_threshold = 0.8
        self._retrieval_gate_log_counter = 0
        self._retrieval_gate_gated_count = 0
        self._retrieval_gate_total_count = 0

        # Region meta header (disabled by default).
        # Call configure_region_meta_header() after construction to enable.
        self._region_meta_header_enabled = False
        self._region_meta_header_log_counter = 0

        # Meta-info mode (disabled by default).
        # Call configure_meta_info_mode() after construction to enable.
        self._meta_info_mode = False
        self._meta_info_log_counter = 0

    def configure_failure_summary(self, n_slots: int = 2, replace_with_summary: bool = True,
                                  lib_filter: bool = False, contract_filter: bool = False,
                                  min_compatible: int = 1,
                                  fmmatch: bool = False, fmmatch_topk: int = 5,
                                  fmmatch_backfill: bool = False,
                                  force_recall: bool = False):
        """Enable region failure summary injection for BCB.

        When enabled, retrieval post-processes selected_mems to:
        1. Reserve `n_slots` positions for failure memories
        2. If not enough failures in top-K, do failure-only sim retrieval
        3. Replace failure content with aggregated region failure summary

        Args:
            n_slots: failure memory slots to reserve (of retrieve_k total)
            replace_with_summary: if True, replace failure raw content with region summary
            lib_filter: if True, only include failures whose libs overlap with the
                current task's libs (deprecated: library overlap is not a contract).
            contract_filter: if True, build a summary only from failures matching
                the query return type, or matching non-pure I/O family plus explicit
                exception contract. With no compatible evidence, drop the FS slot
                and backfill success context rather than inject generic advice.
            min_compatible: minimum compatible failure reflections required to
                inject a contract-filtered summary. Below this, abstain and backfill.
            fmmatch: if True, use failure-mode-matched conditional summary (3-layer design):
                utility-gated region selection → failure-mode similarity recall → LLM
                conditional if/elif/else contract synthesis. Benchmark-agnostic. Overrides
                lib_filter when set.
            fmmatch_topk: number of most-similar region failure reflections to feed the LLM.
        """
        self._failure_summary_n_slots = n_slots
        self._failure_summary_replace = replace_with_summary
        self._failure_summary_lib_filter = lib_filter
        self._failure_summary_contract_filter = contract_filter
        self._failure_summary_min_compatible = max(1, int(min_compatible))
        self._failure_summary_fmmatch = fmmatch
        self._fmmatch_topk = fmmatch_topk
        self._fmmatch_backfill = fmmatch_backfill
        self._failure_summary_force_recall = bool(force_recall)
        logger.info(
            "[Failure Summary] enabled: n_slots=%d, replace=%s, lib_filter=%s, "
            "contract_filter=%s, min_compatible=%d, fmmatch=%s, fmmatch_backfill=%s, force_recall=%s",
            n_slots, replace_with_summary, lib_filter, contract_filter,
            self._failure_summary_min_compatible, fmmatch, fmmatch_backfill,
            self._failure_summary_force_recall,
        )

    def configure_experience_cards(self, n_slots: int = 1):
        """Enable region experience card injection for BCB.

        Reserves `n_slots` positions in top-K for a synthetic "experience card"
        memory. The card content is the region's aggregated atomic facts
        (API gotchas, edge cases, constraints) — compact and directly actionable.

        Unlike failure_summary (which replaces existing failure content),
        experience cards REPLACE the lowest-scored memory in top-K with a
        synthetic card block from the task's nearest region.
        """
        self._experience_cards_n_slots = n_slots
        logger.info("[Experience Cards] enabled: n_slots=%d", n_slots)

    def configure_retrieval_gate(self, signal: str = "success_ratio",
                                  threshold: float = 0.8):
        """Enable post-retrieval gate: drop memory if quality signal is below threshold.

        Args:
            signal: 'success_ratio' (fraction of success memories in top-K)
                    or 'mean_q' (average Q value of retrieved memories).
            threshold: if signal < threshold, memory context is dropped entirely.
        """
        assert signal in ("success_ratio", "mean_q"), f"Unknown gate signal: {signal}"
        self._retrieval_gate_enabled = True
        self._retrieval_gate_signal = signal
        self._retrieval_gate_threshold = threshold
        logger.info("[Retrieval Gate] enabled: signal=%s, threshold=%.2f", signal, threshold)

    def _apply_retrieval_gate(self, selected_mems: list) -> bool:
        """Return True if memory should be DROPPED (gated out)."""
        if not self._retrieval_gate_enabled or not selected_mems:
            return False

        mem_cache = getattr(self.mem, '_mem_cache', {}) or {}
        q_cache = getattr(self.mem, '_q_cache', None) or {}
        if not q_cache:
            q_cache = getattr(self.mem, '_global_q_cache', None) or {}

        n_success = 0
        n_total = 0
        q_values = []

        for m in selected_mems:
            mid = str(m.get("memory_id") or m.get("id", ""))
            if not mid or mid.startswith("__"):
                continue
            n_total += 1

            cached = mem_cache.get(mid)
            outcome = ""
            if cached is not None:
                if isinstance(cached, dict):
                    meta = cached.get("metadata", {})
                    if isinstance(meta, dict):
                        outcome = str(meta.get("outcome", "")).lower()
                else:
                    meta = getattr(cached, "metadata", None)
                    if meta is not None:
                        if isinstance(meta, dict):
                            outcome = str(meta.get("outcome", "")).lower()
                        else:
                            outcome = str(getattr(meta, "outcome", "")).lower()
            if "success" in outcome:
                n_success += 1

            q = q_cache.get(mid)
            if isinstance(q, dict):
                q_values.extend(q.values())
            elif isinstance(q, (int, float)):
                q_values.append(q)

        if self._retrieval_gate_signal == "success_ratio":
            value = n_success / n_total if n_total > 0 else 0.0
        else:
            value = sum(q_values) / len(q_values) if q_values else 0.0

        gate_out = value < self._retrieval_gate_threshold
        self._retrieval_gate_total_count += 1
        if gate_out:
            self._retrieval_gate_gated_count += 1

        if self._retrieval_gate_total_count <= 5 or self._retrieval_gate_total_count % 50 == 0:
            logger.info(
                "[Retrieval Gate] task #%d: signal=%s value=%.3f threshold=%.2f → %s "
                "(cumulative gated=%d/%d = %.1f%%)",
                self._retrieval_gate_total_count,
                self._retrieval_gate_signal, value, self._retrieval_gate_threshold,
                "DROP" if gate_out else "KEEP",
                self._retrieval_gate_gated_count, self._retrieval_gate_total_count,
                100.0 * self._retrieval_gate_gated_count / self._retrieval_gate_total_count,
            )

        return gate_out

    def configure_region_meta_header(self):
        """Enable region reliability header + anti-anchoring instruction in memory context."""
        self._region_meta_header_enabled = True
        logger.info("[Region Meta Header] enabled")

    def configure_meta_info_mode(self):
        """Enable meta-info mode: replace full memory content with structured hints.

        When enabled, _format_memory_context extracts libraries, return types,
        function signatures, and task summaries from retrieved memories instead
        of injecting full solution code. This reduces model anchoring to specific
        implementations while preserving useful structural hints.
        """
        self._meta_info_mode = True
        logger.info("[Meta-Info Mode] enabled — memory content replaced with structured hints")

    def _format_memory_context(self, selected_mems, **kwargs):
        """Override to support meta-info mode and region meta header."""
        if self._meta_info_mode and selected_mems:
            return self._format_meta_info_context(selected_mems)

        base_context = super()._format_memory_context(selected_mems)
        if not self._region_meta_header_enabled or not base_context or not selected_mems:
            return base_context

        # Compute reliability signals from retrieved memories
        mem_cache = getattr(self.mem, '_mem_cache', {}) or {}

        n_success = 0
        n_failure = 0
        n_total = 0
        sims = []
        for m in selected_mems:
            mid = str(m.get("memory_id") or m.get("id", ""))
            if not mid or mid.startswith("__"):
                continue
            n_total += 1
            sim = m.get("similarity", 0.0)
            if isinstance(sim, (int, float)):
                sims.append(float(sim))

            cached = mem_cache.get(mid)
            outcome = ""
            if cached is not None:
                if isinstance(cached, dict):
                    meta = cached.get("metadata", {})
                    if isinstance(meta, dict):
                        outcome = str(meta.get("outcome", "")).lower()
                else:
                    meta = getattr(cached, "metadata", None)
                    if meta is not None:
                        if isinstance(meta, dict):
                            outcome = str(meta.get("outcome", "")).lower()
                        else:
                            outcome = str(getattr(meta, "outcome", "")).lower()
            if "success" in outcome:
                n_success += 1
            elif "fail" in outcome:
                n_failure += 1

        success_ratio = n_success / n_total if n_total > 0 else 0.0
        top1_sim = sims[0] if sims else 0.0
        avg_sim = sum(sims) / len(sims) if sims else 0.0

        # Build concise, direct instruction — no confidence labels, just rules
        # Key insight: the model needs to know WHEN to ignore, not a trust score
        header = (
            "## Memory Usage Instructions\n"
            "The following examples were retrieved from past tasks. "
            f"({n_success} succeeded, {n_failure} failed, "
            f"best similarity: {top1_sim:.2f}, avg: {avg_sim:.2f})\n\n"
            "Rules:\n"
            "1. First reason about what the CURRENT task needs (required APIs, logic, edge cases).\n"
            "2. Only reuse code patterns from examples that use the SAME key APIs as the current task.\n"
            "3. If an example uses different libraries or a different approach, IGNORE its structure — "
            "at most note its error handling or edge cases.\n"
            "4. [FAILURE] examples show what NOT to do — extract the mistake, not the code.\n\n"
        )

        self._region_meta_header_log_counter += 1
        if self._region_meta_header_log_counter <= 3 or self._region_meta_header_log_counter % 50 == 0:
            logger.info(
                "[Region Meta Header] task #%d: success=%d/%d, top1_sim=%.2f, avg_sim=%.2f",
                self._region_meta_header_log_counter, n_success, n_total, top1_sim, avg_sim,
            )

        return header + base_context

    def _format_meta_info_context(self, selected_mems: list) -> str:
        """Format retrieved memories as structured meta-info hints instead of full content.

        Extracts libraries, return types, function signatures, and task summaries
        from each retrieved memory, then aggregates into a compact prompt.
        """
        import ast as _ast
        from collections import Counter

        mem_cache = getattr(self.mem, '_mem_cache', {}) or {}

        all_libs = Counter()
        return_types = []
        signatures = []
        task_summaries = []

        for m in selected_mems:
            mid = str(m.get("memory_id") or m.get("id", ""))
            if not mid or mid.startswith("__"):
                continue

            # Get memory entry from cache for metadata access
            cached = mem_cache.get(mid)
            meta = {}
            mem_text = ""
            if cached is not None:
                if isinstance(cached, dict):
                    meta = cached.get("metadata", {}) or {}
                    mem_text = cached.get("memory", "") or ""
                else:
                    meta_obj = getattr(cached, "metadata", None)
                    if meta_obj is not None:
                        if isinstance(meta_obj, dict):
                            meta = meta_obj
                        elif hasattr(meta_obj, "model_dump"):
                            meta = meta_obj.model_dump()
                        elif hasattr(meta_obj, "model_extra"):
                            meta = getattr(meta_obj, "model_extra", {}) or {}
                    mem_text = getattr(cached, "memory", "") or ""

            if not isinstance(meta, dict):
                meta = {}

            # Fallback: try metadata/content from the selected_mems entry itself
            if not meta and isinstance(m.get("metadata"), dict):
                meta = m["metadata"]
            if not mem_text:
                mem_text = m.get("memory", "") or m.get("content", "") or ""

            # 1. Extract libs
            libs_field = meta.get("libs", [])
            if isinstance(libs_field, str):
                try:
                    libs_field = _ast.literal_eval(libs_field)
                except Exception:
                    libs_field = []
            if isinstance(libs_field, list):
                for lib in libs_field:
                    if isinstance(lib, str) and lib:
                        all_libs[lib] += 1

            # 2. Extract return type from memory text
            rt = self._extract_return_type(mem_text)
            if rt and rt not in return_types:
                return_types.append(rt)

            # 3. Extract function signature
            sig = self._extract_signature(mem_text)
            if sig and sig not in signatures:
                signatures.append(sig)

            # 4. Extract task summary (first sentence)
            summary = self._extract_task_summary(mem_text)
            if summary and summary not in task_summaries:
                task_summaries.append(summary)

        # If nothing was extracted, return empty to avoid injecting useless boilerplate
        if not all_libs and not return_types and not signatures and not task_summaries:
            return ""

        # Build formatted output
        parts = [
            "# Hints from Similar Tasks",
            "(Reference only — implement based on the current task's specific requirements)\n",
        ]

        if all_libs:
            top_libs = [lib for lib, _ in all_libs.most_common(10)]
            parts.append("## Libraries Used in Similar Tasks")
            parts.append(", ".join(top_libs))
            parts.append("")

        if return_types:
            parts.append("## Return Type Patterns")
            for rt in return_types[:4]:
                parts.append(f"- {rt}")
            parts.append("")

        if signatures:
            parts.append("## Function Signatures Seen")
            for sig in signatures[:3]:
                parts.append(f"- {sig}")
            parts.append("")

        if task_summaries:
            parts.append("## Similar Task Descriptions")
            for ts in task_summaries[:4]:
                parts.append(f"- {ts}")
            parts.append("")

        result = "\n".join(parts)

        # Logging
        self._meta_info_log_counter += 1
        if self._meta_info_log_counter <= 3 or self._meta_info_log_counter % 50 == 0:
            logger.info(
                "[Meta-Info] task #%d: libs=%d unique, return_types=%d, sigs=%d, "
                "summaries=%d, output_len=%d",
                self._meta_info_log_counter, len(all_libs), len(return_types),
                len(signatures), len(task_summaries), len(result),
            )

        return result

    @staticmethod
    def _extract_return_type(mem_text: str) -> str:
        """Extract return type description from memory text."""
        if "output with:" not in mem_text:
            return ""
        try:
            section = mem_text.split("output with:")[1]
            # End at "You should" or "Note that" or code block
            for marker in ("You should", "Note that", "```"):
                if marker in section:
                    section = section.split(marker)[0]
            lines = [l.strip() for l in section.strip().split("\n") if l.strip()]
            # Take first 2 meaningful lines
            ret_lines = lines[:2]
            return " ".join(ret_lines)[:150] if ret_lines else ""
        except Exception:
            return ""

    @staticmethod
    def _extract_signature(mem_text: str) -> str:
        """Extract function signature from code block in memory text."""
        import re
        match = re.search(r"(def\s+\w+\([^)]*\)(?:\s*->\s*[^:]+)?)\s*:", mem_text)
        if match:
            return match.group(1).strip()
        return ""

    @staticmethod
    def _extract_task_summary(mem_text: str) -> str:
        """Extract first sentence as task summary."""
        if not mem_text:
            return ""
        # Skip if starts with common non-summary patterns
        text = mem_text.strip()
        # Take first sentence (up to first period followed by space or newline)
        import re
        match = re.match(r"([^.]+\.)", text)
        if match:
            summary = match.group(1).strip()
            if 10 < len(summary) <= 120:
                return summary
        # Fallback: first line up to 100 chars
        first_line = text.split("\n")[0].strip()
        return first_line[:100] if len(first_line) > 10 else ""

    def _inject_experience_cards(self, selected_mems: list, target_subtask: str) -> list:
        """Replace lowest-scored memory slot(s) with region experience cards.

        Only activates after clustering (when regions have experience_cards).
        """
        n_slots = self._experience_cards_n_slots
        if n_slots <= 0 or not selected_mems:
            return selected_mems

        rm = getattr(self.mem, 'region_manager', None)
        if not rm or not rm.regions:
            return selected_mems

        # Find the best region for this target_subtask
        best_region = None
        best_utility = -1.0
        for region in rm.regions:
            u = region.utility_by_subtask.get(target_subtask, 0.0)
            if u > best_utility:
                best_utility = u
                best_region = region

        if not best_region or not best_region.experience_cards:
            return selected_mems

        # Format cards as a compact block
        cards = best_region.experience_cards[:6]
        card_content = (
            "[REGION EXPERIENCE — constraints/gotchas for this task type]\n"
            + "\n".join(f"  {c}" for c in cards)
        )

        # Replace last n_slots memories (lowest scored) with card block
        n_replace = min(n_slots, len(selected_mems))
        for i in range(n_replace):
            idx = len(selected_mems) - 1 - i
            selected_mems[idx] = {
                "memory_id": f"__experience_card_{i}__",
                "content": card_content,
                "metadata": {"outcome": "experience_card"},
                "similarity": 0.0,
                "_experience_card": True,
            }

        # Logging
        self._experience_cards_log_counter += 1
        if self._experience_cards_log_counter <= 3 or self._experience_cards_log_counter % 50 == 0:
            logger.info(
                "[Experience Cards] task #%d: injected %d card slot(s), %d cards, len=%d",
                self._experience_cards_log_counter, n_replace, len(cards), len(card_content),
            )

        return selected_mems

    def _inject_failure_summary(
        self,
        selected_mems: list,
        prompt: str,
        task: Optional[Dict[str, Any]] = None,
        candidate_mems: Optional[list] = None,
    ) -> list:
        """Post-process selected_mems to inject failure summary.

        Splits into success+failure, ensures at least n_slots failures,
        replaces failure content with region summary, recombines.
        Returns modified selected_mems list.

        Guard: only activates AFTER region clustering has completed
        (rm.regions is non-empty). Before that, returns unmodified.
        """
        n_slots = self._failure_summary_n_slots
        if n_slots <= 0 or not selected_mems:
            return selected_mems

        # Guard: don't inject before regions are built (E1 cold start).
        rm = getattr(self.mem, 'region_manager', None)
        if not rm or not rm.regions:
            return selected_mems

        # ===== fmmatch 3-layer design: three-state branching =====
        if self._failure_summary_fmmatch and task is not None:
            task_id = str(task.get("task_id", ""))
            state, failure_mode = self._get_task_failure_history(task_id)

            if state == "only_success":
                # Task has only ever passed — no failure injection, all success.
                self._failure_inject_log_counter += 1
                if self._failure_inject_log_counter <= 3 or self._failure_inject_log_counter % 50 == 0:
                    logger.info(
                        "[fmmatch] task #%d (%s): only_success → all success mems, no failure slot",
                        self._failure_inject_log_counter, task_id,
                    )
                return selected_mems  # unchanged, all success

            if state == "has_failure" and failure_mode:
                # Use failure-mode-matched conditional summary from utility-gated region.
                # L1: utility gating — pick the region(s) this task's failure mem belongs to.
                # Use the region of the first failure memory in selected, or the largest.
                target_region = None
                # Try to find the region where this task's own failure memory sits
                mem_to_region = {}
                for region in rm.regions:
                    for mid in region.member_ids:
                        mem_to_region[mid] = region
                # Look for a failure mem from selected_mems that belongs to a region
                for m in selected_mems:
                    mid = m.get("memory_id")
                    if mid and mid in mem_to_region:
                        target_region = mem_to_region[mid]
                        break
                if target_region is None:
                    # Fallback: largest region
                    target_region = max(rm.regions, key=lambda r: len(r.member_ids))

                task_prompt = prompt[:1000]
                fmmatch_summary = self._build_fmmatch_summary(
                    target_region, failure_mode, task_prompt
                )

                if fmmatch_summary:
                    # Inject as the failure slot: keep success + 1 fmmatch slot.
                    # Backfill success mems so total = n_selected (avoid context starvation).
                    # n_selected = len(selected_mems) which is rl_config.topk (Phase-B output).
                    success_mems = [m for m in selected_mems if self._get_outcome(m) != "failure"]
                    n_selected = len(selected_mems) if selected_mems else 5
                    target_n_success = max(0, n_selected - 1)  # reserve 1 slot for fmmatch
                    if self._fmmatch_backfill and len(success_mems) < target_n_success:
                        # Backfill: retrieve extra success-only memories
                        exclude_ids = {
                            x for x in (m.get("memory_id") or m.get("id") for m in selected_mems)
                            if x is not None
                        }
                        extra_needed = target_n_success - len(success_mems)
                        extra_success = self._retrieve_success_only_bcb(
                            prompt, k=extra_needed, exclude_ids=exclude_ids
                        )
                        success_mems.extend(extra_success)
                    success_mems = success_mems[:target_n_success]
                    fmmatch_mem = {
                        "content": "[CONDITIONAL FAILURE GUIDE (failure-mode matched)]\n" + fmmatch_summary,
                        "memory_id": "__fmmatch_synthetic__",
                        "_region_failure_summary": True,
                    }
                    final_mems = success_mems + [fmmatch_mem]
                    self._failure_inject_log_counter += 1
                    if self._failure_inject_log_counter <= 3 or self._failure_inject_log_counter % 50 == 0:
                        logger.info(
                            "[fmmatch] task #%d (%s): has_failure → conditional summary injected "
                            "(len=%d, n_success=%d/%d)",
                            self._failure_inject_log_counter, task_id, len(fmmatch_summary),
                            len(success_mems), target_n_success,
                        )
                    return final_mems

            # state == "no_history" OR fmmatch failed to produce summary → fall through
            # to the original region.failure_summary aggregation below.

        # ===== Canonical conditional Region failure replacement =====
        # Operate only on failures already selected by the baseline top-K.
        # Do not force-recall a failure and do not reorder/displace any slot.
        final_mems = list(selected_mems)
        failure_mems = [m for m in final_mems if self._get_outcome(m) == "failure"]
        eligible_failures = failure_mems[: min(n_slots, len(failure_mems))]

        if not eligible_failures and self._failure_summary_force_recall:
            exclude_ids = {
                m.get("memory_id") or m.get("id") for m in final_mems
            }
            recalled = self._retrieve_failure_only_bcb(
                prompt, k=min(n_slots, len(final_mems)), exclude_ids=exclude_ids
            )
            if recalled:
                eligible_failures = recalled[:min(n_slots, len(final_mems))]
                # Reserve fixed slots by replacing the lowest-ranked baseline
                # memories; total prompt budget and selected count stay constant.
                final_mems = final_mems[: max(0, len(final_mems) - len(eligible_failures))]
                final_mems.extend(eligible_failures)

        if not eligible_failures:
            self._failure_inject_log_counter += 1
            if self._failure_inject_log_counter <= 3 or self._failure_inject_log_counter % 50 == 0:
                logger.info(
                    "[Failure Summary] task #%d: abstain_no_failure_available; "
                    "original top-%d preserved",
                    self._failure_inject_log_counter, len(final_mems),
                )
            return final_mems

        if self._failure_summary_replace:
            task_libs = None
            if self._failure_summary_lib_filter and task is not None:
                from memrl.run.oracle_retrieval import _parse_libs
                task_libs = _parse_libs(task.get("libs"))
            n_replaced, n_dropped = self._replace_bcb_failure_with_summary(
                eligible_failures, task_libs=task_libs, task=task, prompt=prompt,
            )
        else:
            n_replaced = 0
            n_dropped = 0

        # A failed summary lookup retains the raw selected failure. Contract-gate
        # variants may mark a slot for removal; canonical fallback restores the
        # untouched original top-K rather than changing IDs or order.
        if n_dropped > 0 or any(fm.get("_drop_slot") for fm in eligible_failures):
            return list(selected_mems)

        self._failure_inject_log_counter += 1
        if self._failure_inject_log_counter <= 3 or self._failure_inject_log_counter % 50 == 0:
            logger.info(
                "[Failure Summary] task #%d: selected_failures=%d, replaced=%d, "
                "top-%d IDs/order preserved",
                self._failure_inject_log_counter, len(eligible_failures), n_replaced,
                len(final_mems),
            )
        return final_mems

    def _retrieve_failure_only_bcb(self, prompt: str, k: int = 2,
                                   exclude_ids: set = None) -> list:
        """Retrieve top-k failure memories by sim only (no Q rerank).

        Adapted from ALFWorld's _retrieve_failure_only for BCB's metadata schema
        (uses 'outcome' field instead of 'success' bool).
        """
        import math

        if not hasattr(self.mem, 'dict_memory') or not self.mem.dict_memory:
            return []
        if not hasattr(self.mem, '_mem_cache'):
            return []

        try:
            from memrl.service.memory_service import get_embedding_with_retry
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

    def _retrieve_success_only_bcb(self, prompt: str, k: int = 4,
                                   exclude_ids: set = None) -> list:
        """Retrieve top-k success memories by sim only (no Q rerank).

        Used to backfill success memories when fmmatch discards failure mems
        from the selected set, preventing context starvation.
        """
        import math

        if k <= 0:
            return []
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
            query_dim = len(query_vec)

            seen_mids = set()
            candidates = []
            mc = self.mem._mem_cache
            for query_key, mem_ids in self.mem.dict_memory.items():
                qv = _qe.get(query_key)
                if qv is None or len(qv) != query_dim:
                    continue
                q_norm = math.sqrt(sum(x * x for x in qv)) or 1e-8
                sim = sum(a * b for a, b in zip(query_vec, qv)) / (query_norm * q_norm)
                for mid in mem_ids:
                    if mid in seen_mids:
                        continue
                    if exclude_ids and mid in exclude_ids:
                        continue
                    mem_obj = mc.get(mid)
                    if mem_obj is None:
                        continue
                    outcome = self._get_outcome_from_cache(mem_obj)
                    if outcome != "success":
                        continue
                    seen_mids.add(mid)
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
            logger.warning("[fmmatch] success-only backfill retrieval failed", exc_info=True)
            return []

    def _replace_bcb_failure_with_summary(self, failure_mems: list,
                                          task_libs: Optional[set] = None,
                                          task: Optional[Dict[str, Any]] = None,
                                          prompt: str = "") -> tuple:
        """Replace failure memory content with its region's aggregated failure summary.

        Lib filtering is retained as a legacy ablation. Contract filtering is the
        preferred safe path: it requires either an exact return-type match, or a
        shared non-pure I/O family and explicit exception tag. If no compatible
        evidence exists, the failure slot is marked for dropping.

        Returns (n_replaced, n_dropped).
        """
        rm = getattr(self.mem, 'region_manager', None)
        if not rm or not rm.regions:
            return 0, 0

        # Build mem_id → region mapping
        mem_to_region = {}
        for region in rm.regions:
            for mid in region.member_ids:
                mem_to_region[mid] = region

        # BCB's task dict often contains only task_id/domains at this layer;
        # `prompt` is the authoritative query specification received by
        # _inject_failure_summary(). Merge it explicitly so contract gating is
        # based on the actual evaluated task, never an empty placeholder.
        contract_task = dict(task or {})
        if prompt:
            contract_task["prompt"] = prompt
        task_contract = self._extract_bcb_task_contract(contract_task) \
            if self._failure_summary_contract_filter else None
        n_replaced = 0
        n_dropped = 0
        for fm in failure_mems:
            mid = fm.get("memory_id")
            region = mem_to_region.get(mid)
            if not region:
                continue

            if task_contract is not None:
                summary, matched, total = self._build_contract_filtered_summary(
                    region, task_contract
                )
                if not summary:
                    fm["_drop_slot"] = True
                    n_dropped += 1
                    logger.info(
                        "[Failure Summary contract_gate] skipped: task=%s "
                        "return=%s io=%s exceptions=%s compatible=%d/%d",
                        task_contract["task_id"], task_contract["return_type"],
                        sorted(task_contract["io_families"]),
                        sorted(task_contract["exception_tags"]), matched, total,
                    )
                    continue
                fm["content"] = "[REGION FAILURE PATTERNS (contract-matched)]\n" + summary
                fm["_region_failure_summary"] = True
                n_replaced += 1
                logger.info(
                    "[Failure Summary contract_gate] injected: task=%s "
                    "return=%s io=%s exceptions=%s compatible=%d/%d",
                    task_contract["task_id"], task_contract["return_type"],
                    sorted(task_contract["io_families"]),
                    sorted(task_contract["exception_tags"]), matched, total,
                )
            elif task_libs and self._failure_summary_lib_filter:
                summary = self._build_lib_filtered_summary(region, task_libs)
                if not summary:
                    fm["_drop_slot"] = True
                    n_dropped += 1
                    continue
                fm["content"] = "[REGION FAILURE PATTERNS (lib-filtered)]\n" + summary
                fm["_region_failure_summary"] = True
                n_replaced += 1
            elif region.failure_summary:
                fm["content"] = "[REGION FAILURE PATTERNS]\n" + region.failure_summary
                fm["_region_failure_summary"] = True
                n_replaced += 1

        return n_replaced, n_dropped

    @staticmethod
    def _extract_bcb_task_contract(task: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract conservative, lexical BCB task-contract features.

        These are intentionally simple and deterministic: the gate must avoid
        generic cross-contract failure advice, not predict semantic similarity.
        """
        import re

        task = task or {}
        prompt = str(task.get("prompt") or task.get("task_description") or "")
        normalized = " ".join(prompt.lower().split())
        return_match = re.search(
            r"(?:function should )?output with:\s*([^:\n]+)\s*:", prompt,
            flags=re.IGNORECASE,
        )
        return_type = " ".join(return_match.group(1).lower().split()) \
            if return_match else "unknown"
        io_families = set()
        if re.search(r"\b(?:file|csv|json|excel|path|directory|folder|read|write|save|open|archive|move|copy|delete)\b", normalized):
            io_families.add("filesystem")
        if re.search(r"\b(?:plot|chart|graph|histogram|matplotlib|figure|axes)\b", normalized):
            io_families.add("plot")
        if re.search(r"\b(?:url|http|https|request|api|web)\b", normalized):
            io_families.add("network")
        exception_tags = set(re.findall(
            r"\b([a-z_]*?(?:error|exception))\b", normalized,
        ))
        if "raise" in normalized and not exception_tags:
            exception_tags.add("raises")
        return {
            "task_id": str(task.get("task_id") or "<unknown>"),
            "return_type": return_type,
            "io_families": io_families,
            "exception_tags": exception_tags,
        }

    @staticmethod
    def _contract_compatible(query: Dict[str, Any], candidate: Dict[str, Any]) -> bool:
        """Whether a failure task can safely contribute to query's FS evidence."""
        q_return = query["return_type"]
        c_return = candidate["return_type"]
        if q_return != "unknown" and q_return == c_return:
            return True
        shared_io = query["io_families"] & candidate["io_families"]
        shared_exceptions = query["exception_tags"] & candidate["exception_tags"]
        return bool(shared_io and shared_exceptions)

    def _build_contract_filtered_summary(self, region, task_contract: Dict[str, Any]) -> tuple:
        """Aggregate only failure reflections with an observable compatible contract."""
        from memrl.service.region_manager import RegionManager

        mem_cache = getattr(self.mem, "_mem_cache", None) or {}
        fields_list = []
        total_failures = 0
        for mem_id in region.member_ids:
            mem_obj = mem_cache.get(mem_id)
            if mem_obj is None:
                continue
            metadata = (mem_obj.get("metadata") if isinstance(mem_obj, dict)
                        else getattr(mem_obj, "metadata", None))
            extras = (getattr(metadata, "model_extra", None) or
                      (metadata if isinstance(metadata, dict) else {}))
            outcome = str(extras.get("outcome", "")).lower()
            success = extras.get("success")
            if outcome != "failure" and success is not False:
                continue
            total_failures += 1
            source_task = {
                "task_id": extras.get("task_id"),
                "task_description": extras.get("task_description") or extras.get("memory") or "",
            }
            candidate_contract = self._extract_bcb_task_contract(source_task)
            if not self._contract_compatible(task_contract, candidate_contract):
                continue
            content = extras.get("full_content") or ""
            fields = RegionManager._parse_failure_fields(content)
            if fields["failure_mode"] or fields["mistakes"] or fields["fixes"]:
                fields_list.append(fields)
        matched = len(fields_list)
        min_compatible = max(1, int(getattr(self, "_failure_summary_min_compatible", 1) or 1))
        if matched < min_compatible:
            return "", matched, total_failures
        return RegionManager._format_failure_summary(fields_list, top_n=3), matched, total_failures

    def _build_lib_filtered_summary(self, region, task_libs: set) -> str:
        """Build failure summary from region members filtered by lib overlap with current task."""
        from memrl.run.oracle_retrieval import _parse_libs
        from memrl.service.region_manager import RegionManager

        mem_cache = getattr(self.mem, '_mem_cache', None)
        if mem_cache is None:
            return ""

        filtered_fields = []
        n_total_failures = 0
        for mem_id in region.member_ids:
            mem_obj = mem_cache.get(mem_id)
            if mem_obj is None:
                continue
            md = getattr(mem_obj, 'metadata', None)
            if md is None:
                continue
            extras = getattr(md, 'model_extra', None) or (md if isinstance(md, dict) else {})
            success = extras.get('success')
            if success is not False:
                continue
            n_total_failures += 1

            mem_libs = _parse_libs(extras.get('libs'))
            if not (mem_libs & task_libs):
                continue

            content = extras.get('full_content') or ""
            if not content:
                continue
            fields = RegionManager._parse_failure_fields(content)
            if fields["failure_mode"] or fields["mistakes"]:
                filtered_fields.append(fields)

        if not filtered_fields:
            return ""

        summary = RegionManager._format_failure_summary(filtered_fields, top_n=3)
        logger.debug(
            "[Failure Summary lib_filter] %d/%d region failures matched libs, summary_len=%d",
            len(filtered_fields), n_total_failures, len(summary),
        )
        return summary

    # ===================================================================
    # Failure-mode-matched conditional summary (fmmatch, 3-layer design)
    # ===================================================================

    def _get_task_failure_history(self, task_id: str):
        """Look up a task's own prior failure/success state from the memory pool.

        Returns:
            ("has_failure", failure_mode_str) — task failed before, with its FAILURE_MODE
            ("only_success", None) — task has only successes in pool
            ("no_history", None) — task has no records (cold start, epoch 1)
        """
        from memrl.service.region_manager import RegionManager

        mem_cache = getattr(self.mem, '_mem_cache', None)
        if not mem_cache:
            return ("no_history", None)

        has_success = False
        latest_failure_content = None
        latest_failure_epoch = -1

        for mem_id, mem_obj in mem_cache.items():
            meta = getattr(mem_obj, "metadata", None)
            if meta is None and isinstance(mem_obj, dict):
                meta = mem_obj.get("metadata")
            if meta is None:
                continue
            extras = (getattr(meta, "model_extra", {}) or {}) if hasattr(meta, "model_extra") else (meta if isinstance(meta, dict) else {})
            if extras.get("task_id") != task_id:
                continue

            outcome = str(extras.get("outcome", "")).lower()
            if outcome == "success":
                has_success = True
            elif outcome == "failure":
                try:
                    epoch = int(extras.get("bcb_epoch", 0) or 0)
                except (ValueError, TypeError):
                    epoch = 0
                if epoch > latest_failure_epoch:
                    latest_failure_epoch = epoch
                    latest_failure_content = extras.get("full_content") or ""

        if latest_failure_content:
            fields = RegionManager._parse_failure_fields(latest_failure_content)
            fm = fields.get("failure_mode", "")
            return ("has_failure", fm if fm else latest_failure_content[:200])
        elif has_success:
            return ("only_success", None)
        else:
            return ("no_history", None)

    def _build_fmmatch_summary(self, region, task_failure_mode: str, task_prompt: str) -> str:
        """Build failure summary via failure-mode-matched recall + LLM conditional synthesis.

        Three-layer design:
          L1 (utility gating): caller already selected high-utility region(s)
          L2 (failure-mode recall): embed task's own failure_mode, find top-k most similar
              failure reflections in this region (failure-mode ↔ reflection, same modality)
          L3 (LLM synthesis): from top-k reflections + task spec, LLM generates a
              conditional if/elif/else contract summary ("when X → do Y / avoid Z")

        Returns empty string if insufficient matching reflections.
        """
        import math
        from memrl.service.region_manager import RegionManager

        if not task_failure_mode or not task_failure_mode.strip():
            logger.info("[fmmatch] empty task_failure_mode, skipping")
            return ""

        mem_cache = getattr(self.mem, '_mem_cache', None)
        if not mem_cache:
            logger.info("[fmmatch] no mem_cache, skipping")
            return ""

        if not region.member_ids:
            logger.info("[fmmatch] empty region, skipping")
            return ""

        embed_fn = getattr(self.mem.embedding_provider, 'embed', None)
        if not callable(embed_fn):
            logger.info("[fmmatch] no embed_fn, skipping")
            return ""

        # Collect all failure reflections in this region (batch embedding later)
        candidates = []
        for mem_id in region.member_ids:
            mem_obj = mem_cache.get(mem_id)
            if mem_obj is None:
                continue
            if self._get_outcome_from_cache(mem_obj) != "failure":
                continue
            content = self._get_content_from_cache(mem_obj)
            if not content:
                continue
            fields = RegionManager._parse_failure_fields(content)
            reflection_text = fields.get("failure_mode", "")
            if not reflection_text:
                reflection_text = fields.get("mistakes", [""])[0] if fields.get("mistakes") else ""
            if not reflection_text or len(reflection_text) < 5:
                continue
            candidates.append({
                "mem_id": mem_id,
                "reflection_text": reflection_text,
                "failure_mode": fields.get("failure_mode", ""),
                "mistakes": fields.get("mistakes", []),
                "fixes": fields.get("fixes", []),
                "avoids": fields.get("avoids", []),
            })

        logger.info(
            "[fmmatch] region has %d members, %d failure candidates with parseable reflection",
            len(region.member_ids), len(candidates),
        )

        if not candidates:
            return ""

        # Batch embed: task failure mode + all reflection texts in one call
        texts_to_embed = [task_failure_mode] + [c["reflection_text"] for c in candidates]
        try:
            all_vecs = embed_fn(texts_to_embed)
        except Exception as e:
            logger.warning("[fmmatch] batch embed failed: %s", str(e)[:100])
            return ""

        if not all_vecs or len(all_vecs) < 2:
            return ""

        fm_vec = all_vecs[0]
        if fm_vec is None:
            return ""
        fm_norm = math.sqrt(sum(x * x for x in fm_vec)) or 1e-8

        # Score each candidate by cosine similarity
        scored = []
        for i, cand in enumerate(candidates):
            r_vec = all_vecs[i + 1]
            if r_vec is None:
                continue
            if len(r_vec) != len(fm_vec):
                continue
            r_norm = math.sqrt(sum(x * x for x in r_vec)) or 1e-8
            sim = sum(a * b for a, b in zip(fm_vec, r_vec)) / (fm_norm * r_norm)
            cand["sim"] = sim
            scored.append(cand)

        if not scored:
            return ""

        scored.sort(key=lambda x: x["sim"], reverse=True)
        topk = scored[:self._fmmatch_topk]

        sims = [s["sim"] for s in topk]
        logger.info(
            "[fmmatch] similarity stats: best=%.3f mean=%.3f worst=%.3f (threshold=0.3, topk=%d/%d)",
            sims[0], sum(sims) / len(sims), sims[-1], len(topk), len(scored),
        )

        if topk[0]["sim"] < 0.3:
            logger.info("[fmmatch] best sim %.3f < 0.3 threshold, skipping LLM synthesis", topk[0]["sim"])
            return ""

        # L3: LLM synthesis — generate conditional contract summary
        reflections_text = ""
        for i, s in enumerate(topk, 1):
            reflections_text += f"\nReflection {i} (similarity={s['sim']:.2f}):\n"
            reflections_text += f"  FAILURE_MODE: {s['failure_mode']}\n"
            if s["mistakes"]:
                reflections_text += f"  MISTAKES: {'; '.join(s['mistakes'][:3])}\n"
            if s["fixes"]:
                reflections_text += f"  FIXES: {'; '.join(s['fixes'][:3])}\n"

        synthesis_prompt = (
            f"You are helping a code generation model avoid past mistakes.\n\n"
            f"Current task (to solve next):\n{task_prompt[:800]}\n\n"
            f"This task previously failed with:\n  {task_failure_mode}\n\n"
            f"Below are failure reflections from similar tasks that made similar mistakes:\n"
            f"{reflections_text}\n"
            f"Based on the current task spec AND these similar failure patterns, generate a "
            f"COMPACT conditional guide (≤150 words). Use if/elif/else structure:\n"
            f"- if <situation A> → must do X / avoid Y\n"
            f"- elif <situation B> → must do X / avoid Y\n"
            f"- else → default approach\n\n"
            f"Be specific to THIS task's requirements. Only include conditions that are "
            f"directly relevant (skip irrelevant reflections). Plain text, no code blocks."
        )

        try:
            summary = self.llm.generate(
                messages=[{"role": "user", "content": synthesis_prompt}],
                temperature=0.2,
                max_tokens=512,
            )
            summary = (summary or "").strip()
        except Exception as e:
            logger.warning("[fmmatch] LLM synthesis failed: %s", str(e)[:100])
            return ""

        if len(summary) < 30:
            logger.info("[fmmatch] LLM output too short (%d chars), discarding", len(summary))
            return ""

        logger.info(
            "[fmmatch] SUCCESS: built conditional summary from %d reflections (best_sim=%.2f), summary_len=%d",
            len(topk), topk[0]["sim"], len(summary),
        )
        return summary

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

    def run_eval_only(self) -> Dict[str, Any]:
        """Skip training; just run eval once on holdout or val set.

        Requires resume_checkpoint_path to be set (loads memory pool from snapshot).
        If --holdout_subtask is set, evaluates on holdout tasks.
        Otherwise, evaluates on the standard val set.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        self._problems = self._load_problems()
        from memrl.run.bcb_runner import split_dataset
        self._train_ids, self._val_ids = split_dataset(
            self._problems,
            train_ratio=self.sel.train_ratio,
            seed=self.sel.seed,
            split_file=self.sel.split_file,
        )
        self._post_data_load_hook()

        if self.resume_checkpoint_path:
            logger.info("eval_only: loading checkpoint from %s", self.resume_checkpoint_path)
            self.mem.load_checkpoint_snapshot(self.resume_checkpoint_path)

        if hasattr(self.mem, 'region_manager') and self.mem.region_manager is not None:
            rm = self.mem.region_manager
            if not getattr(rm, '_subtask_embeddings', None):
                logger.info("eval_only: registering subtask embeddings (none in checkpoint)")
                try:
                    self._register_subtask_embeddings()
                except Exception:
                    logger.warning("eval_only: subtask embedding registration failed", exc_info=True)

        if self._holdout_ids:
            epoch_dir = os.path.join(self.output_dir, f"eval_only_{self.retrieval_mode}")
            os.makedirs(epoch_dir, exist_ok=True)
            logger.info("eval_only: retrieval_mode=%s, holdout=%s, n=%d",
                        self.retrieval_mode, self.holdout_subtask, len(self._holdout_ids))
            self._run_holdout_eval(epoch=0, epoch_dir=epoch_dir)
            logger.info("eval_only: done. results at %s", epoch_dir)
            return {"output_dir": epoch_dir, "retrieval_mode": self.retrieval_mode}
        else:
            epoch_dir = os.path.join(self.output_dir, "eval_only_val")
            os.makedirs(epoch_dir, exist_ok=True)
            logger.info("eval_only: running on val set, n=%d", len(self._val_ids))
            self._precompute_query_embeddings(self._val_ids)
            val_res = self._run_phase(
                epoch=0, phase="val", task_ids=self._val_ids,
                epoch_dir=epoch_dir, update_memory=False,
            )
            logger.info("eval_only val: pass=%d/%d (%.1f%%)",
                        val_res["pass"], val_res["total"],
                        100.0 * val_res["pass"] / val_res["total"] if val_res["total"] else 0)
            logger.info("eval_only: done. results at %s", epoch_dir)
            return {"output_dir": epoch_dir, "val_results": val_res}

    def _load_problems(self):
        from memrl.bigcodebench_eval.task_wrappers import load_bcb_data
        return load_bcb_data(subset=self.sel.subset, data_path=self.sel.data_path)

    def _run_phase(self, *, epoch, phase, task_ids, epoch_dir, update_memory, start_idx=0):
        """Override to switch lambda_max for val phase if val_lambda_max is set."""
        rm = getattr(self.mem, 'region_manager', None)
        saved_lmax = None
        if phase == "val" and self.val_lambda_max is not None and rm is not None:
            saved_lmax = getattr(rm, 'shrinkage_lambda_max', None)
            rm.shrinkage_lambda_max = self.val_lambda_max
            logger.info("Val phase: switching lambda_max to %.2f", self.val_lambda_max)
        try:
            result = super()._run_phase(
                epoch=epoch, phase=phase, task_ids=task_ids,
                epoch_dir=epoch_dir, update_memory=update_memory, start_idx=start_idx,
            )
        finally:
            if saved_lmax is not None and rm is not None:
                rm.shrinkage_lambda_max = saved_lmax
            elif phase == "val" and self.val_lambda_max is not None and rm is not None:
                if hasattr(rm, 'shrinkage_lambda_max'):
                    del rm.shrinkage_lambda_max
        return result

    # -- Hooks (called by parent BCBRunner.run) --

    def _post_data_load_hook(self) -> None:
        """In holdout mode: move holdout subtask tasks to _holdout_ids, skip val entirely."""
        if not self.holdout_subtask:
            return

        original_train = len(self._train_ids)
        original_val = len(self._val_ids)
        holdout_ids = []
        keep_train = []

        for tid in self._train_ids:
            task = self._problems[tid]
            domains = self._get_task_domains(task)
            subtask = get_primary_subtask("bigcodebench", {"domains": domains})
            if subtask == self.holdout_subtask:
                holdout_ids.append(tid)
            else:
                keep_train.append(tid)

        # Val set: also collect holdout subtask tasks into holdout_ids
        for tid in self._val_ids:
            task = self._problems[tid]
            domains = self._get_task_domains(task)
            subtask = get_primary_subtask("bigcodebench", {"domains": domains})
            if subtask == self.holdout_subtask:
                holdout_ids.append(tid)

        self._train_ids = keep_train
        # Dedupe (train/val should not overlap, but defensive)
        seen = set()
        deduped = []
        for tid in holdout_ids:
            if tid not in seen:
                seen.add(tid)
                deduped.append(tid)
        self._holdout_ids = deduped
        # Disable val — only train + holdout eval matters
        self.run_validation = False

        logger.info(
            "Holdout subtask=%s: holdout_ids=%d (train=%d, val=%d), "
            "remaining train=%d, val disabled",
            self.holdout_subtask, len(holdout_ids),
            original_train - len(keep_train),
            len(holdout_ids) - (original_train - len(keep_train)),
            len(self._train_ids),
        )

        if not holdout_ids:
            logger.warning(
                "No tasks matched holdout_subtask=%s — check subtask naming",
                self.holdout_subtask,
            )

    def _pre_epoch_hook(self, epoch: int) -> None:
        if hasattr(self.mem, 'set_current_epoch'):
            self.mem.set_current_epoch(epoch, num_epochs=self.num_epochs)

        if epoch == 1:
            self._init_task_clusters()

        # Register subtask embeddings if missing (epoch 1 or after checkpoint resume)
        rm = getattr(self.mem, 'region_manager', None)
        if rm is not None and not rm._subtask_embeddings:
            self._register_subtask_embeddings()

    def _post_train_hook(self, epoch: int, epoch_dir: str) -> None:
        if self.mem.region_manager:
            try:
                self._recluster_regions(epoch, epoch_dir)
            except Exception as e:
                logger.error("Region clustering failed at epoch %d: %s", epoch, e, exc_info=True)

        if self._holdout_ids:
            try:
                self._run_holdout_eval(epoch, epoch_dir)
            except Exception as e:
                logger.error("Holdout eval failed at epoch %d: %s", epoch, e, exc_info=True)

    def _init_task_clusters(self) -> None:
        """Initialize auto task clusters from task embeddings or checkpoint."""
        from memrl.configs.task_hierarchy import get_task_cluster_manager, _task_cluster_mgr
        import numpy as np
        import glob
        import os

        # If entry point did not initialize the cluster manager (task_cluster_k=0),
        # skip auto clustering and use domain subtasks instead.
        if _task_cluster_mgr is None:
            logger.info("Task cluster manager not initialized (task_cluster_k=0), using domain subtasks")
            return

        tcm = get_task_cluster_manager()
        if tcm._fitted:
            return

        # Try loading from existing checkpoint first
        tc_files = sorted(glob.glob(os.path.join(self.output_dir, "**/task_clusters.json"), recursive=True))
        if tc_files:
            if tcm.load(tc_files[-1]):
                return

        embed_fn = getattr(getattr(self.mem, 'embedding_provider', None), 'embed', None)
        if not callable(embed_fn):
            logger.warning("No embedding provider, skipping task cluster init")
            return

        from memrl.bigcodebench_eval.task_wrappers import get_prompt
        qe = getattr(self.mem, 'query_embeddings', {})

        # Collect cached and uncached prompts
        task_embeddings = {}
        uncached_tids = []
        uncached_prompts = []
        for tid in self._train_ids:
            task = self._problems[tid]
            prompt = get_prompt(task, split=self.sel.split)
            emb = qe.get(prompt)
            if emb is not None:
                task_embeddings[tid] = np.array(emb)
            else:
                uncached_tids.append(tid)
                uncached_prompts.append(prompt)

        # Batch compute uncached embeddings
        if uncached_prompts:
            BATCH = 32
            for i in range(0, len(uncached_prompts), BATCH):
                batch_prompts = uncached_prompts[i:i + BATCH]
                batch_tids = uncached_tids[i:i + BATCH]
                try:
                    vecs = embed_fn(batch_prompts)
                    for tid, vec in zip(batch_tids, vecs):
                        task_embeddings[tid] = np.array(vec)
                except Exception as e:
                    logger.warning("Batch embedding failed: %s", e)

        if len(task_embeddings) >= 50:
            tcm.fit(task_embeddings)
            logger.info("Task clusters initialized: K=%d from %d tasks", tcm.K, len(task_embeddings))
        else:
            logger.warning("Too few task embeddings (%d) for clustering", len(task_embeddings))

    def _register_subtask_embeddings(self) -> None:
        """Compute and register subtask embeddings for zero-shot transfer.

        Each subtask's embedding = mean of its task embeddings.
        Uses ALL tasks (not just train split) so holdout subtasks get embeddings too.
        """
        import numpy as np
        from memrl.configs.task_hierarchy import get_primary_subtask
        from memrl.bigcodebench_eval.task_wrappers import get_prompt

        rm = getattr(self.mem, 'region_manager', None)
        if rm is None:
            return

        embed_fn = getattr(getattr(self.mem, 'embedding_provider', None), 'embed', None)
        if not callable(embed_fn):
            return

        if not hasattr(self.mem, 'query_embeddings') or self.mem.query_embeddings is None:
            self.mem.query_embeddings = {}
        qe = self.mem.query_embeddings

        # Use ALL tasks (not just _train_ids) so holdout subtasks get embeddings
        all_task_ids = list(self._problems.keys())
        subtask_embs = {}
        uncached_prompts = []
        uncached_subtasks = []

        for tid in all_task_ids:
            task = self._problems[tid]
            domains = self._get_task_domains(task)
            subtask = get_primary_subtask("bigcodebench", {"domains": domains})
            prompt = get_prompt(task, split=self.sel.split)

            emb = qe.get(prompt)
            if emb is not None:
                if subtask not in subtask_embs:
                    subtask_embs[subtask] = []
                subtask_embs[subtask].append(np.array(emb))
            else:
                uncached_prompts.append(prompt)
                uncached_subtasks.append(subtask)

        # Batch embed uncached prompts
        if uncached_prompts:
            try:
                new_embs = embed_fn(uncached_prompts)
                for prompt, subtask, emb in zip(uncached_prompts, uncached_subtasks, new_embs):
                    qe[prompt] = emb
                    if subtask not in subtask_embs:
                        subtask_embs[subtask] = []
                    subtask_embs[subtask].append(np.array(emb))
            except Exception as e:
                logger.warning("Failed to embed uncached prompts: %s", e)

        # Register mean embedding per subtask
        for subtask, embs in subtask_embs.items():
            if embs:
                mean_emb = np.mean(embs, axis=0)
                rm.set_subtask_embedding(subtask, mean_emb)

        logger.info("Registered %d subtask embeddings for zero-shot transfer", len(subtask_embs))

    def _run_holdout_eval(self, epoch: int, epoch_dir: str) -> None:
        """Evaluate on holdout subtask tasks using zero-shot transfer."""
        holdout_dir = os.path.join(epoch_dir, "holdout")
        os.makedirs(holdout_dir, exist_ok=True)

        rm = self.mem.region_manager
        if rm and self.holdout_subtask:
            known = list(rm._known_subtasks) if hasattr(rm, '_known_subtasks') else []
            is_unseen = self.holdout_subtask not in known
            if not is_unseen:
                logger.warning(
                    "HOLDOUT LEAKAGE: %s found in _known_subtasks — transfer is not truly zero-shot",
                    self.holdout_subtask,
                )
            logger.info(
                "Holdout eval epoch %d: subtask=%s, unseen=%s, holdout_tasks=%d, known_subtasks=%s",
                epoch, self.holdout_subtask, is_unseen, len(self._holdout_ids),
                known[:10],
            )

        # Snapshot known_subtasks before holdout eval to detect leakage
        known_before = set(rm._known_subtasks) if rm and hasattr(rm, '_known_subtasks') else set()

        # Clear retrieval buffer before holdout to prevent cross-phase contamination
        if hasattr(self.mem, '_retrieval_subtask_buffer'):
            self.mem._retrieval_subtask_buffer.clear()

        holdout_res = self._run_phase(
            epoch=epoch,
            phase="val",
            task_ids=self._holdout_ids,
            epoch_dir=holdout_dir,
            update_memory=False,
        )

        # Clear buffer after holdout — holdout subtask entries must not leak to next train update
        if hasattr(self.mem, '_retrieval_subtask_buffer'):
            self.mem._retrieval_subtask_buffer.clear()

        # Verify holdout eval didn't mutate _known_subtasks
        if rm and hasattr(rm, '_known_subtasks'):
            known_after = set(rm._known_subtasks)
            leaked = known_after - known_before
            if leaked:
                logger.error("HOLDOUT LEAKAGE: eval added subtasks to _known_subtasks: %s", leaked)

        holdout_summary = {
            "epoch": epoch,
            "holdout_subtask": self.holdout_subtask,
            "total": holdout_res["total"],
            "pass": holdout_res["pass"],
            "pass_at_1": holdout_res.get("pass@1"),
        }

        # Log transfer diagnostics
        if rm and self.holdout_subtask:
            try:
                import numpy as np
                transfer_info = {}
                utils = []
                for rid, region in enumerate(rm.regions):
                    u, pc, strategy = rm._estimate_region_utility_zero_shot(region, self.holdout_subtask)
                    transfer_info[rid] = {"utility": round(float(u), 4), "pseudo_count": pc, "strategy": strategy}
                    utils.append(float(u))
                holdout_summary["zero_shot_region_utilities"] = transfer_info

                # Spread diagnostic: detect residual washout / signal cancellation.
                # Healthy: regions disagree on holdout utility (signal preserved).
                # Pathological: all regions predict ~mu_bar (residual cancelled).
                if utils:
                    spread = {
                        "min": round(min(utils), 4),
                        "max": round(max(utils), 4),
                        "range": round(max(utils) - min(utils), 4),
                        "mean": round(float(np.mean(utils)), 4),
                        "std": round(float(np.std(utils)), 4),
                        "n_regions": len(utils),
                    }
                    holdout_summary["zero_shot_spread"] = spread
                    if spread["range"] < 0.05:
                        logger.warning(
                            "Holdout epoch %d: zero-shot SIGNAL WASHOUT — range=%.3f, std=%.3f. "
                            "Regions agree on holdout utility ~%.3f. "
                            "residual term likely cancelled out (check tau, embedding distinctness).",
                            epoch, spread["range"], spread["std"], spread["mean"],
                        )
                    else:
                        logger.info(
                            "Holdout epoch %d: zero-shot spread mean=%.3f std=%.3f range=[%.3f, %.3f]",
                            epoch, spread["mean"], spread["std"], spread["min"], spread["max"],
                        )

                logger.info(
                    "Holdout epoch %d: zero-shot utilities per region: %s",
                    epoch, transfer_info,
                )
            except Exception as e:
                logger.debug("Failed to log zero-shot utilities: %s", e)

        summary_path = os.path.join(holdout_dir, "holdout_summary.json")
        with open(summary_path, "w") as f:
            json.dump(holdout_summary, f, indent=2)

        logger.info(
            "Holdout epoch %d: %s pass=%d/%d (%.1f%%)",
            epoch, self.holdout_subtask,
            holdout_res["pass"], holdout_res["total"],
            100.0 * holdout_res.get("pass@1", 0.0),
        )

    def _pre_epoch_hook(self, epoch: int) -> None:
        """Mark an explicit fixed initial topology on a step-snapshot branch.

        This is intentionally narrow: it is used only when a caller resumes
        from an already-clustered initial snapshot and explicitly supplies the
        epoch. Fresh runs and ordinary completed-epoch resumes are unchanged.
        """
        target = self.fixed_initial_topology_epoch
        if self._fixed_initial_topology_marked or target is None or int(epoch) != target:
            return
        rm = self.mem.region_manager
        if not getattr(rm, "_is_clustered", False):
            raise RuntimeError(
                "fixed_initial_topology_epoch requires an already-clustered resume snapshot"
            )
        rm.topology_last_edit_section = int(target)
        self._fixed_initial_topology_marked = True
        logger.info(
            "[bcb] fixed initial topology branch: marked restored %d-region state as epoch %d edit; cooldown=%d",
            len(rm.regions), target, self.region_topology_cooldown_epochs,
        )

    def _topology_edit_allowed(self, epoch: int) -> bool:
        """Whether an epoch-boundary/mid-epoch topology edit is permitted."""
        cooldown = self.region_topology_cooldown_epochs
        if cooldown <= 0:
            return True
        rm = self.mem.region_manager
        last = int(getattr(rm, "topology_last_edit_section", 0) or 0)
        # cooldown=1 protects the edit epoch *and one complete following
        # epoch*. If topology changed at E2, E3 gathers evidence under fixed
        # identities and E3-end remains protected; the next eligible edit is
        # E4-end. This mirrors the ALFWorld one-full-section cooldown.
        return last <= 0 or (int(epoch) - last) > cooldown

    def _mark_topology_edit(self, epoch: int, reason: str) -> None:
        rm = self.mem.region_manager
        rm.topology_last_edit_section = int(epoch)
        logger.info(
            "[bcb] topology edit recorded at epoch %d (%s); cooldown=%d epoch(s)",
            epoch, reason, self.region_topology_cooldown_epochs,
        )

    def _recluster_regions(self, epoch: int, epoch_dir: str):
        """Epoch-end region maintenance: initial cluster OR assign + split/merge."""
        import json
        import os

        rm = self.mem.region_manager

        if not rm._is_clustered:
            logger.info("Epoch %d: initial clustering by utility patterns...", epoch)
            rm.cluster_by_utility()
            self._mark_topology_edit(epoch, "initial_cluster")
        else:
            # Assign any unassigned memories
            for mem_id in list(rm.subtask_q):
                if mem_id not in rm.membership_weights:
                    rm.assign_new_memory(mem_id)
            if not self._topology_edit_allowed(epoch):
                logger.info(
                    "Epoch %d: topology cooldown active (last_edit=%d, cooldown=%d); split/merge skipped, regions=%d",
                    epoch, int(getattr(rm, "topology_last_edit_section", 0) or 0),
                    self.region_topology_cooldown_epochs, len(rm.regions),
                )
            else:
                changed = rm.maybe_split_merge()
                if changed:
                    self._mark_topology_edit(epoch, "epoch_boundary_split_merge")
                logger.info(
                    "Epoch %d: incremental region update (split/merge=%s, regions=%d)",
                    epoch, changed, len(rm.regions),
                )

        rm.classify_transfer_patterns()

        # Save region summary per epoch
        summary = rm.get_region_summary()
        summary["epoch"] = epoch
        summary_path = os.path.join(epoch_dir, "region_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Save full region_manager state for calibrator training
        region_mgr_path = os.path.join(epoch_dir, "region_manager.json")
        try:
            rm.save(region_mgr_path)
            logger.info("Epoch %d: saved region_manager to %s", epoch, region_mgr_path)
        except Exception as e:
            logger.warning("Failed to save region_manager: %s", e)

        # Save task cluster centroids for checkpoint resume
        from memrl.configs.task_hierarchy import _task_cluster_mgr
        if _task_cluster_mgr is not None and _task_cluster_mgr._fitted:
            tc_path = os.path.join(epoch_dir, "task_clusters.json")
            try:
                _task_cluster_mgr.save(tc_path)
            except Exception as e:
                logger.warning("Failed to save task clusters: %s", e)

        logger.info(
            "Epoch %d: utility clustering done. %d regions, %d memories assigned.",
            epoch, len(rm.regions), len(rm.membership_weights),
        )

    def _retrieve_for_task(self, prompt: str, task: Optional[Dict[str, Any]] = None, target_subtask: Optional[str] = None) -> Tuple[List[str], str, Dict[str, Any], Optional[List[Tuple[str, float]]]]:
        """Override to inject target_subtask for region gating, or to swap in oracle/no_mem retrieval."""
        if self.mem is None or self.retrieve_k <= 0:
            return [], "", {}, None

        # No-memory mode: skip retrieval entirely
        if self.retrieval_mode == "no_mem":
            return [], "", {"mode": "no_mem", "retrieved_count": 0}, None

        # Oracle mode: pick best memories by ground-truth overlap with task
        if self.retrieval_mode == "oracle" and task is not None:
            from memrl.run.oracle_retrieval import (
                select_oracle_memories,
                memory_to_selected_format,
            )
            top = select_oracle_memories(
                target_task=task,
                memory_pool=self._oracle_memory_pool or [],
                top_k=self.retrieve_k,
                holdout_subtask=self.holdout_subtask,
            )
            if not top:
                # Hard fail rather than silently degrade to no_mem (would mask oracle ceiling).
                raise RuntimeError(
                    f"Oracle retrieval returned 0 memories for task {task.get('task_id')}; "
                    f"pool size={len(self._oracle_memory_pool or [])}, "
                    f"holdout_subtask={self.holdout_subtask}"
                )
            selected_mems = [memory_to_selected_format(m) for m, _ in top]
            selected_ids = [m["memory_id"] for m in selected_mems]
            mem_context = self._format_memory_context(selected_mems)
            retrieved_topk_queries = [
                [str(m.get("content", ""))[:500], float(score)]
                for (m, score) in top
            ]
            return selected_ids, mem_context, {
                "mode": "oracle",
                "retrieved_count": len(selected_ids),
                "top_score": float(top[0][1]) if top else 0.0,
            }, retrieved_topk_queries

        # Default: current retrieval (region transfer)
        try:
            thr = self._get_retrieve_threshold()

            if target_subtask is None:
                domains = self._get_task_domains(task) if task else []
                task_emb = self._get_task_embedding(prompt) if task else None
                target_subtask = get_primary_subtask("bigcodebench", {
                    "domains": domains, "embedding": task_emb,
                })

            # Call with target_subtask and use_region_gating
            ret = self.mem.retrieve_query(
                prompt,
                k=self.retrieve_k,
                threshold=thr,
                target_subtask=target_subtask,
                eval_mode=None,
                use_region_gating=True,
            )

            if isinstance(ret, tuple):
                ret_result, retrieved_topk_queries = ret
            else:
                ret_result, retrieved_topk_queries = ret, None

            selected_mems = (ret_result or {}).get("selected", []) if ret_result else []
            if not isinstance(selected_mems, list):
                selected_mems = []

            if selected_mems:
                logger.info(
                    "Region retrieval: %d selected, first content_len=%d",
                    len(selected_mems),
                    len(str(selected_mems[0].get("content", "") or "")),
                )

            # --- Region failure summary injection (if enabled) ---
            if self._failure_summary_n_slots > 0:
                selected_mems = self._inject_failure_summary(
                    selected_mems,
                    prompt,
                    task=task,
                    candidate_mems=(ret_result or {}).get("candidates", []),
                )

            # Reward credit must follow the memories that actually remain in the
            # final prompt after RFS replacement. A newly recalled failure slot
            # receives credit; a displaced success memory does not.
            selected_ids = [
                str(m.get("memory_id") or m.get("id"))
                for m in selected_mems
                if isinstance(m, dict)
                and (m.get("memory_id") or m.get("id"))
                and not str(m.get("memory_id") or m.get("id")).startswith("__")
            ]

            # --- Region experience cards injection (if enabled) ---
            if self._experience_cards_n_slots > 0:
                selected_mems = self._inject_experience_cards(selected_mems, target_subtask)

            # --- Retrieval gate: drop memory if quality signal below threshold ---
            if self._retrieval_gate_enabled and self._apply_retrieval_gate(selected_mems):
                selected_mems = []
                selected_ids = []

            mem_context = self._format_memory_context(selected_mems)

            retrieval_trace = {
                "mode": "retrieve_query",
                "retrieved_count": len(selected_ids),
                "simmax": float((ret_result or {}).get("simmax", 0.0) or 0.0),
            }
            # Diagnostic: keep top-(K+1) candidate scores so callers can compute
            # margin / Δscore for paper-grade reviewer-defense ablations. Cheap
            # (~200 bytes per task at K=5). Caller may ignore.
            cand_list = (ret_result or {}).get("candidates", []) or []
            if cand_list:
                diag = []
                for c in cand_list[: self.retrieve_k + 1]:
                    if not isinstance(c, dict):
                        continue
                    diag.append({
                        "mid": str(c.get("memory_id") or c.get("id") or ""),
                        "score": float(c.get("score", 0.0) or 0.0),
                        "sim_z": float(c.get("similarity_z", 0.0) or 0.0),
                        "q_z": float(c.get("q_z", 0.0) or 0.0),
                        "q_est": float(c.get("q_estimate", 0.0) or 0.0),
                    })
                retrieval_trace["candidates_diag"] = diag

            return selected_ids, mem_context, retrieval_trace, retrieved_topk_queries

        except Exception:
            logger.warning("BCB region retrieval failed", exc_info=True)
            return [], "", {}, None

    def _process_single_task(self, task_id: str, epoch: int, phase: str) -> Dict[str, Any]:
        """
        Override to pass task to _retrieve_for_task and track target_subtask.
        """
        task = self._problems[task_id]
        prompt = self._get_prompt_for_task(task)

        # Compute target_subtask once, pass to retrieval
        domains = self._get_task_domains(task)
        task_emb = self._get_task_embedding(prompt)
        target_subtask = get_primary_subtask("bigcodebench", {
            "domains": domains, "embedding": task_emb,
        })

        # Retrieve with task context for region gating
        selected_ids, mem_context, retrieval_trace, retrieved_topk_queries = self._retrieve_for_task(prompt, task, target_subtask=target_subtask)

        # Generate
        raw_response = self._generate_raw(prompt, memory_context=mem_context)
        from memrl.bigcodebench_eval.bcb_adapter import extract_code_from_response
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
            "target_subtask": target_subtask,  # REGION: track for update_values
        }

    def _get_prompt_for_task(self, task: Dict[str, Any]) -> str:
        """Helper to get prompt for task."""
        from memrl.bigcodebench_eval.task_wrappers import get_prompt
        return get_prompt(task, split=self.sel.split)

    def _get_task_embedding(self, prompt: str):
        """Get embedding for a task prompt. Uses cached embeddings from memory service."""
        import numpy as np
        qe = getattr(self.mem, 'query_embeddings', {})
        emb = qe.get(prompt)
        if emb is not None:
            return np.array(emb) if not isinstance(emb, np.ndarray) else emb
        # Compute on the fly
        embed_fn = getattr(getattr(self.mem, 'embedding_provider', None), 'embed', None)
        if callable(embed_fn):
            try:
                vec = embed_fn([prompt])[0]
                return np.array(vec)
            except Exception:
                pass
        return None

    def _build_memory_metadata(
        self, res: Dict[str, Any], task_id: str, epoch: int, phase: str, ok: bool
    ) -> Dict[str, Any]:
        """Override to add source_subtask to metadata."""
        meta = super()._build_memory_metadata(res, task_id, epoch, phase, ok)
        target_subtask = res.get("target_subtask")
        if target_subtask:
            meta["source_subtask"] = target_subtask
        return meta

    def _post_batch_hook(self, **kwargs) -> None:
        """Incremental region management: first cluster at 500, then assign + split/merge."""
        batch_end = kwargs.get("batch_end", 0)
        update_memory = kwargs.get("update_memory", False)
        epoch = kwargs.get("epoch", 0)

        INITIAL_CLUSTER_STEP = 500

        # Canonical topology changes only at the coarse update boundary.  When
        # mid-epoch topology is disabled, this includes the initial clustering;
        # _post_train_hook performs it once at the epoch boundary.
        if self.region_disable_mid_epoch_topology:
            return

        if not (update_memory and batch_end >= INITIAL_CLUSTER_STEP and self.mem.region_manager):
            return

        rm = self.mem.region_manager
        task_count = batch_end
        prev_count = batch_end - self.batch_size

        if not rm._is_clustered:
            if prev_count < INITIAL_CLUSTER_STEP <= task_count:
                try:
                    rm.cluster_by_utility()
                    self._mark_topology_edit(epoch, "initial_cluster")
                    logger.info(
                        "[bcb] epoch %d INITIAL cluster at step %d: %d regions",
                        epoch, batch_end, len(rm.regions),
                    )
                except Exception as e:
                    logger.warning("Initial clustering failed: %s", e)
        else:
            # Already clustered — assign new memories + periodic split/merge
            # Assign any memories not yet assigned to a region
            for mem_id in list(rm.subtask_q):
                if mem_id not in rm.membership_weights:
                    rm.assign_new_memory(mem_id)

            # The topology-stable schedule disables historical mid-epoch
            # maintenance so a region identity survives a complete epoch.
            if self.region_disable_mid_epoch_topology:
                return
            if task_count >= INITIAL_CLUSTER_STEP and (task_count // 400) > (prev_count // 400):
                if not self._topology_edit_allowed(epoch):
                    logger.info(
                        "[bcb] epoch %d topology cooldown active at step %d; mid-epoch split/merge skipped",
                        epoch, batch_end,
                    )
                    return
                try:
                    changed = rm.maybe_split_merge()
                    if changed:
                        self._mark_topology_edit(epoch, "mid_epoch_split_merge")
                        logger.info(
                            "[bcb] epoch %d split/merge at step %d: now %d regions",
                            epoch, batch_end, len(rm.regions),
                        )
                except Exception as e:
                    logger.warning("Split/merge failed: %s", e)

