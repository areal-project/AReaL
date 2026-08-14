"""
Holdout Region Manager for Cross-Subtask Transfer Experiments.

Extends RegionManager with parallel holdout banks. Each holdout bank
maintains independent region utility stats that exclude rewards from
the held-out subtask's queries.

Usage:
    mgr = HoldoutRegionManager(
        holdout_subtasks=["Computation", "Network", "Cryptography", ...],
        task_hierarchy=..., alpha=0.1, ...
    )
    # During training:
    mgr.update_subtask_q(mem_ids, target_subtask="Computation", reward=1.0)
    # → updates full bank normally
    # → updates all holdout banks EXCEPT holdout_Computation (since target == held-out)

    # At eval, get holdout-aware Q:
    q = mgr.get_holdout_region_utility(mem_id, target_subtask="Computation", holdout_bank="holdout_Computation")
    # → returns region utility estimated from OTHER subtasks only
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from memrl.service.region_manager import RegionManager, Region

logger = logging.getLogger(__name__)


class HoldoutRegionManager(RegionManager):
    """
    RegionManager with parallel holdout banks for cross-subtask transfer validation.

    Shared state (from parent):
      - subtask_q: per-memory per-subtask Q values
      - subtask_q_counts: observation counts
      - membership_weights: soft region membership
      - regions: cluster structure (centroids, members)

    Holdout-specific state:
      - holdout_utilities[bank][region_id][subtask] -> utility value
      - holdout_counts[bank][region_id][subtask] -> count
      - holdout_success_sums[bank][region_id][subtask] -> success sum (for beta mode)
      - holdout_total_counts[bank][region_id][subtask] -> total count (for beta mode)
    """

    def __init__(self, holdout_subtasks: List[str], **kwargs):
        super().__init__(**kwargs)
        self.holdout_subtasks = holdout_subtasks
        self.bank_names = ["full"] + [f"holdout_{st}" for st in holdout_subtasks]

        # Per-bank region utilities (independent from parent's region.utility_by_subtask)
        # Structure: {bank_name: {region_id: {subtask: float}}}
        self.holdout_utilities: Dict[str, Dict[int, Dict[str, float]]] = {
            bank: defaultdict(dict) for bank in self.bank_names if bank != "full"
        }
        self.holdout_counts: Dict[str, Dict[int, Dict[str, int]]] = {
            bank: defaultdict(dict) for bank in self.bank_names if bank != "full"
        }
        # For beta mode
        self.holdout_success_sums: Dict[str, Dict[int, Dict[str, float]]] = {
            bank: defaultdict(dict) for bank in self.bank_names if bank != "full"
        }
        self.holdout_total_counts: Dict[str, Dict[int, Dict[str, float]]] = {
            bank: defaultdict(dict) for bank in self.bank_names if bank != "full"
        }

    def update_subtask_q(
        self,
        memory_ids: List[str],
        target_subtask: str,
        reward: float,
    ):
        """
        Update Q values for FULL bank only.

        Holdout banks are updated separately via update_single_holdout_bank(),
        called by HoldoutRegionMemoryService.update_values_multi_bank() with
        each holdout bank's own reward and memory IDs.
        """
        super().update_subtask_q(memory_ids, target_subtask, reward)

    def update_single_holdout_bank(
        self,
        bank_name: str,
        memory_ids: List[str],
        target_subtask: str,
        reward: float,
    ):
        """
        Update region utilities for a single holdout bank using that bank's
        own reward and memory IDs (from holdout-specific inference).

        Skips update if target_subtask matches the held-out subtask.
        """
        if not self._is_clustered:
            return

        holdout_st = bank_name.replace("holdout_", "")
        if target_subtask == holdout_st:
            return

        for mem_id in memory_ids:
            weights = self.membership_weights.get(mem_id)
            if weights is None:
                weights = self._assign_membership_lazy(mem_id)
                if weights is None:
                    continue
            for rid, w in enumerate(weights):
                if w < 0.01 or rid >= len(self.regions):
                    continue

                if self.region_utility_mode == "beta":
                    self.holdout_success_sums[bank_name][rid][target_subtask] = (
                        self.holdout_success_sums[bank_name][rid].get(target_subtask, 0.0) + w * reward
                    )
                    self.holdout_total_counts[bank_name][rid][target_subtask] = (
                        self.holdout_total_counts[bank_name][rid].get(target_subtask, 0.0) + w
                    )
                    a0, b0 = 2.0, 2.0
                    s = self.holdout_success_sums[bank_name][rid][target_subtask]
                    n = self.holdout_total_counts[bank_name][rid][target_subtask]
                    self.holdout_utilities[bank_name][rid][target_subtask] = (s + a0) / (n + a0 + b0)
                else:
                    old_u = self.holdout_utilities[bank_name][rid].get(target_subtask, 0.5)
                    self.holdout_utilities[bank_name][rid][target_subtask] = (
                        old_u + self.alpha * w * (reward - old_u)
                    )

                self.holdout_counts[bank_name][rid][target_subtask] = (
                    self.holdout_counts[bank_name][rid].get(target_subtask, 0) + 1
                )

    def get_holdout_region_utility(
        self,
        mem_id: str,
        target_subtask: str,
        bank_name: str,
    ) -> float:
        """
        Get the predicted utility for a memory on a held-out subtask using region transfer.

        For non-held-out subtask: return the holdout bank's region utility directly.
        For held-out subtask: use Gaussian conditional prediction (Eq 7 from theory):
            Û(r, s*) = μ_{s*} + Σ_{s*O} Σ_{OO}^{-1} (U(r, O) - μ_O)
        Falls back to simple mean if covariance estimation fails.
        """
        if not self._is_clustered:
            return 0.5

        weights = self.membership_weights.get(mem_id)
        if weights is None:
            return 0.5

        holdout_st = bank_name.replace("holdout_", "")

        effective_top_n = self.shrinkage_top_n
        pairs = []
        for rid, w in enumerate(weights):
            if w < 0.001 or rid >= len(self.regions):
                continue

            if target_subtask == holdout_st:
                utility = self._predict_holdout_utility_for_region(rid, holdout_st, bank_name)
            else:
                utility = self.holdout_utilities[bank_name].get(rid, {}).get(target_subtask, 0.5)

            pairs.append((w, utility))

        if not pairs:
            return 0.5

        pairs.sort(key=lambda x: x[0], reverse=True)
        pairs = pairs[:effective_top_n]

        total_w = sum(w for w, _ in pairs)
        if total_w < 1e-9:
            return 0.5

        return float(sum((w / total_w) * u for w, u in pairs))

    def _predict_holdout_utility_for_region(
        self, region_id: int, holdout_st: str, bank_name: str
    ) -> float:
        """
        Predict held-out subtask utility for a region using Gaussian conditional.

        Theory (Eq 7): Û(r, s*) = μ_{s*} + Σ_{s*O} Σ_{OO}^{-1} (U(r, O) - μ_O)

        Estimates covariance from the full bank's region-subtask utility matrix
        (all regions × all subtasks), then conditions on observed subtasks
        to predict the held-out one.

        Falls back to simple mean if not enough data for covariance estimation.
        """
        n_regions = len(self.regions)
        observed_subtasks = [st for st in self._known_subtasks if st != holdout_st]

        if not observed_subtasks or n_regions < 3:
            # Not enough data, fall back to simple mean
            region_utils = self.holdout_utilities[bank_name].get(region_id, {})
            observed = [u for st, u in region_utils.items() if st != holdout_st]
            return float(np.mean(observed)) if observed else 0.5

        # Build region-subtask utility matrix from FULL bank (parent's regions)
        # Use full bank utilities for covariance estimation (more data)
        all_subtasks = self._known_subtasks
        st_to_idx = {st: i for i, st in enumerate(all_subtasks)}

        if holdout_st not in st_to_idx:
            region_utils = self.holdout_utilities[bank_name].get(region_id, {})
            observed = [u for st, u in region_utils.items() if st != holdout_st]
            return float(np.mean(observed)) if observed else 0.5

        U_matrix = np.full((n_regions, len(all_subtasks)), 0.5)
        for rid in range(n_regions):
            for j, st in enumerate(all_subtasks):
                u = self.regions[rid].utility_by_subtask.get(st)
                if u is not None:
                    U_matrix[rid, j] = u

        # Column means
        mu = U_matrix.mean(axis=0)

        # Covariance matrix (with diagonal loading for stability)
        centered = U_matrix - mu
        cov = centered.T @ centered / max(n_regions - 1, 1)
        reg = 1e-4 * np.eye(len(all_subtasks))
        cov += reg

        # Partition into observed (O) and held-out (s*)
        s_star_idx = st_to_idx[holdout_st]
        o_indices = [st_to_idx[st] for st in observed_subtasks if st in st_to_idx]

        if not o_indices:
            return 0.5

        Sigma_star_O = cov[s_star_idx, o_indices]  # (|O|,)
        Sigma_OO = cov[np.ix_(o_indices, o_indices)]  # (|O|, |O|)

        # Get this region's observed utilities from holdout bank
        region_utils = self.holdout_utilities[bank_name].get(region_id, {})
        U_r_O = np.array([region_utils.get(st, 0.5) for st in observed_subtasks if st in st_to_idx])
        mu_O = mu[o_indices]
        mu_star = mu[s_star_idx]

        try:
            # Eq 7: Û(r, s*) = μ_{s*} + Σ_{s*O} Σ_{OO}^{-1} (U(r,O) - μ_O)
            Sigma_OO_inv = np.linalg.solve(Sigma_OO, np.eye(len(o_indices)))
            prediction = mu_star + Sigma_star_O @ Sigma_OO_inv @ (U_r_O - mu_O)
            return float(np.clip(prediction, 0.0, 1.0))
        except np.linalg.LinAlgError:
            # Singular matrix, fall back to simple mean
            observed = [u for st, u in region_utils.items() if st != holdout_st]
            return float(np.mean(observed)) if observed else 0.5

    def compute_holdout_shrinkage_q(
        self,
        mem_id: str,
        target_subtask: str,
        bank_name: str,
        tau_sq: float = 0.05,
        sigma_sq: float = 0.25,
        lambda_max: float = 0.8,
    ) -> float:
        """
        Compute shrinkage Q for a holdout bank.

        For the held-out subtask, per-memory Q has no data (n=0),
        so shrinkage fully trusts region utility (lambda=0).
        This is the intended behavior — tests whether region utility transfers.
        """
        holdout_st = bank_name.replace("holdout_", "")

        if target_subtask == holdout_st:
            # Held-out: no per-memory Q for this subtask, fully use region prediction
            return self.get_holdout_region_utility(mem_id, target_subtask, bank_name)

        # Not held-out: use normal shrinkage with holdout bank's region utility
        n_ms = self.get_observation_count(mem_id, target_subtask)
        per_memory_q = self.get_subtask_q(mem_id, target_subtask)
        region_utility = self.get_holdout_region_utility(mem_id, target_subtask, bank_name)

        if n_ms == 0:
            return region_utility

        lambda_ms = tau_sq / (tau_sq + sigma_sq / n_ms)
        lambda_ms = min(lambda_ms, lambda_max)

        shrinkage_q = lambda_ms * per_memory_q + (1 - lambda_ms) * region_utility
        return float(max(0.0, min(1.0, shrinkage_q)))

    def build_holdout_q_cache(
        self,
        target_subtask: str,
        bank_name: str,
    ) -> Dict[str, float]:
        """
        Build a flat Q cache for retrieval scoring in a holdout bank.

        Returns {mem_id: Q_estimate} suitable for overriding _q_cache.
        """
        flat = {}
        for mem_id in self.subtask_q:
            flat[mem_id] = self.compute_holdout_shrinkage_q(mem_id, target_subtask, bank_name)
        return flat

    # ---------- Persistence ----------

    def save(self, path: str):
        """Save full state including holdout banks."""
        # Save parent state first
        super().save(path)

        # Append holdout state to the same file
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)

        state["holdout_subtasks"] = self.holdout_subtasks
        state["holdout_utilities"] = {
            bank: {str(rid): utils for rid, utils in regions.items()}
            for bank, regions in self.holdout_utilities.items()
        }
        state["holdout_counts"] = {
            bank: {str(rid): counts for rid, counts in regions.items()}
            for bank, regions in self.holdout_counts.items()
        }
        state["holdout_success_sums"] = {
            bank: {str(rid): sums for rid, sums in regions.items()}
            for bank, regions in self.holdout_success_sums.items()
        }
        state["holdout_total_counts"] = {
            bank: {str(rid): counts for rid, counts in regions.items()}
            for bank, regions in self.holdout_total_counts.items()
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        logger.info("HoldoutRegionManager saved (holdout banks: %d)", len(self.holdout_subtasks))

    @classmethod
    def load(cls, path: str) -> "HoldoutRegionManager":
        """Load holdout region manager state."""
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)

        holdout_subtasks = state.get("holdout_subtasks", [])

        mgr = cls(
            holdout_subtasks=holdout_subtasks,
            task_hierarchy=state["task_hierarchy"],
            alpha=state.get("alpha", 0.1),
            min_cluster_size=state.get("min_cluster_size", 5),
            temperature=state.get("temperature", 1.0),
            shrinkage_top_n=state.get("shrinkage_top_n", 3),
            region_utility_mode=state.get("region_utility_mode", "ema"),
        )

        # Restore parent state
        mgr.subtask_q = state.get("subtask_q", {})
        mgr.subtask_q_counts = state.get("subtask_q_counts", {})
        mgr._known_subtasks = state.get("known_subtasks", [])
        mgr._is_clustered = state.get("is_clustered", False)
        mgr._global_reward_sum = state.get("global_reward_sum", 0.0)
        mgr._global_reward_count = state.get("global_reward_count", 0)
        mgr.membership_weights = {
            k: np.array(v) for k, v in state.get("membership_weights", {}).items()
        }
        mgr.regions = [
            Region(
                region_id=r["region_id"],
                centroid=np.array(r["centroid"]) if r.get("centroid") else None,
                member_ids=r.get("member_ids", []),
                utility_by_subtask=r.get("utility_by_subtask", {}),
                counts_by_subtask=r.get("counts_by_subtask", {}),
                success_sum_by_subtask=r.get("success_sum_by_subtask", {}),
                total_count_by_subtask=r.get("total_count_by_subtask", {}),
            )
            for r in state.get("regions", [])
        ]

        # Restore holdout state
        for bank, regions in state.get("holdout_utilities", {}).items():
            mgr.holdout_utilities[bank] = {int(rid): utils for rid, utils in regions.items()}
        for bank, regions in state.get("holdout_counts", {}).items():
            mgr.holdout_counts[bank] = {int(rid): counts for rid, counts in regions.items()}
        for bank, regions in state.get("holdout_success_sums", {}).items():
            mgr.holdout_success_sums[bank] = {int(rid): sums for rid, sums in regions.items()}
        for bank, regions in state.get("holdout_total_counts", {}).items():
            mgr.holdout_total_counts[bank] = {int(rid): counts for rid, counts in regions.items()}

        return mgr
