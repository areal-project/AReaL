"""
Holdout Region Memory Service for Cross-Subtask Transfer Experiments.

Extends RegionMemoryService with:
1. Multi-bank retrieval: retrieve_query_multi_bank() returns results for full + holdout banks
2. Exclude-subtask filtering: holdout banks exclude memories from the held-out subtask
3. Multi-bank Q updates: update_values_multi_bank() routes rewards to all banks
"""

from typing import Any, Dict, List, Optional
import logging

from memrl.service.region_memory_service import RegionMemoryService
from memrl.service.holdout_region_manager import HoldoutRegionManager

logger = logging.getLogger(__name__)


class HoldoutRegionMemoryService(RegionMemoryService):
    """
    Memory service with parallel holdout bank support.

    For each query, can retrieve from multiple banks:
    - full: normal retrieval (all memories, all Q)
    - holdout_X: retrieval excluding memories from subtask X, using holdout Q

    The underlying MemoryService/qdrant store is shared. The difference is:
    - Which memories are included (exclude_origin_subtask filter)
    - Which Q values are used for scoring (holdout bank's shrinkage Q)
    """

    @property
    def holdout_manager(self) -> Optional[HoldoutRegionManager]:
        if isinstance(self.region_manager, HoldoutRegionManager):
            return self.region_manager
        return None

    def retrieve_query_multi_bank(
        self,
        task_description: str,
        target_subtask: str,
        k: int = 5,
        threshold: float = 0.0,
        eval_mode: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve memory candidates from all banks (full + holdout banks).

        Returns:
            Dict mapping bank_name -> retrieval result dict
        """
        hm = self.holdout_manager
        if hm is None:
            return {"full": self.retrieve_query(
                task_description, k=k, threshold=threshold,
                target_subtask=target_subtask, eval_mode=eval_mode,
            )}

        results = {}

        # 1. Full bank: normal retrieval
        results["full"] = self.retrieve_query(
            task_description, k=k, threshold=threshold,
            target_subtask=target_subtask, eval_mode=eval_mode,
        )

        # Pop the subtask buffer entry added by full retrieval
        # (we'll manage the buffer manually for multi-bank)
        if self._retrieval_subtask_buffer:
            self._retrieval_subtask_buffer.pop()

        # 2. Holdout banks
        for holdout_st in hm.holdout_subtasks:
            bank_name = f"holdout_{holdout_st}"
            results[bank_name] = self._retrieve_for_holdout_bank(
                task_description, target_subtask, bank_name, holdout_st,
                k=k, threshold=threshold, eval_mode=eval_mode,
            )

        # Add single buffer entry for the full bank's Q update
        self._retrieval_subtask_buffer.append(target_subtask)

        return results

    def _retrieve_for_holdout_bank(
        self,
        task_description: str,
        target_subtask: str,
        bank_name: str,
        holdout_subtask: str,
        k: int = 5,
        threshold: float = 0.0,
        eval_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve for a holdout bank:
        1. Use holdout bank's Q cache for scoring
        2. Exclude memories originating from the held-out subtask
        """
        hm = self.holdout_manager

        # Build holdout Q cache
        original_q_cache = self._q_cache
        self._q_cache = hm.build_holdout_q_cache(target_subtask, bank_name)

        try:
            # Fetch more candidates since we'll filter some out
            fetch_k = k * 3
            raw_result = super(RegionMemoryService, self).retrieve_query(
                task_description, fetch_k, threshold
            )
        finally:
            self._q_cache = original_q_cache

        if isinstance(raw_result, tuple):
            result, _ = raw_result
        else:
            result = raw_result

        if not result.get("selected"):
            return {"actions": [], "selected": [], "candidates": [], "simmax": 0.0}

        candidates = result["selected"]

        # Filter out memories from the held-out subtask
        candidates = self._filter_exclude_subtask(candidates, holdout_subtask)

        # Apply region gating for multiplicative mode (using holdout bank's utilities)
        if (self.region_manager and target_subtask and
                self.region_gating_mode == "multiplicative"):
            candidates = self._apply_holdout_region_gating(
                candidates, target_subtask, bank_name, eval_mode or "train"
            )

        candidates = candidates[:k]

        return {
            "actions": [c["memory_id"] for c in candidates],
            "selected": candidates,
            "candidates": result.get("candidates", []),
            "simmax": result.get("simmax", 0.0),
        }

    def _filter_exclude_subtask(
        self, candidates: List[Dict[str, Any]], exclude_subtask: str
    ) -> List[Dict[str, Any]]:
        """Filter out memories originating from a specific subtask."""
        filtered = []
        for c in candidates:
            meta = c.get("metadata", {})
            if hasattr(meta, "model_extra"):
                source_subtask = meta.model_extra.get("source_subtask")
            elif isinstance(meta, dict):
                source_subtask = meta.get("source_subtask")
            else:
                source_subtask = None

            if source_subtask != exclude_subtask:
                filtered.append(c)

        return filtered

    def _apply_holdout_region_gating(
        self,
        candidates: List[Dict[str, Any]],
        target_subtask: str,
        bank_name: str,
        mode: str,
    ) -> List[Dict[str, Any]]:
        """Apply region gating using holdout bank's utilities."""
        hm = self.holdout_manager
        if hm is None:
            return candidates

        for c in candidates:
            mem_id = c.get("memory_id")
            if not mem_id:
                continue
            try:
                region_score = hm.get_holdout_region_utility(mem_id, target_subtask, bank_name)
            except Exception:
                region_score = 1.0

            old_score = c.get("score", 1.0)
            c["score"] = old_score * region_score
            c["region_gating_score"] = region_score

        candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return candidates

    def update_values_multi_bank(
        self,
        bank_results: Dict[str, Dict[str, Any]],
        target_subtask: str,
    ) -> None:
        """
        Update holdout bank region utilities from parallel generation results.

        Each holdout bank uses its OWN reward and memory IDs.
        Full bank is NOT updated here — it's handled by parent's _flush_memory_updates
        which calls update_values() → update_subtask_q() normally.

        Args:
            bank_results: {bank_name: {"ok": bool, "selected_ids": [str, ...]}}
            target_subtask: the subtask of the current query
        """
        hm = self.holdout_manager
        if hm is None:
            return

        for holdout_st in hm.holdout_subtasks:
            bank_name = f"holdout_{holdout_st}"
            bank_result = bank_results.get(bank_name)
            if not bank_result:
                continue
            reward = float(bank_result["ok"])
            mem_ids = bank_result.get("selected_ids", [])
            if not mem_ids:
                continue
            hm.update_single_holdout_bank(
                bank_name=bank_name,
                memory_ids=mem_ids,
                target_subtask=target_subtask,
                reward=reward,
            )

    def get_holdout_bank_names(self) -> List[str]:
        """Return list of all bank names (full + holdout)."""
        hm = self.holdout_manager
        if hm is None:
            return ["full"]
        return hm.bank_names
