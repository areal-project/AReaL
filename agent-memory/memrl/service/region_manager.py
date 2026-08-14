"""
Region-based memory management with utility-driven soft clustering.

Core design:
- Each memory has a per-subtask Q value (utility vector)
- Regions are formed by HDBSCAN clustering on utility vectors
- Soft membership: each memory has weights to all regions (distance-based softmax)
- Region score = weighted sum of region utilities on target subtask
- Handles the case where a memory is useful for different subtasks in different ways

This replaces hard assignment with soft membership so a memory can
"belong to" multiple regions with different weights.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Trajectory logging for Pilot B analysis
_trajectory_log_file = None
_trajectory_log_enabled = False


@dataclass
class Region:
    """A cluster of memories with similar utility patterns."""
    region_id: int
    centroid: Optional[np.ndarray] = None
    member_ids: List[str] = field(default_factory=list)
    utility_by_subtask: Dict[str, float] = field(default_factory=dict)
    counts_by_subtask: Dict[str, int] = field(default_factory=dict)
    # Beta posterior accumulators for region utility (P0 fix)
    success_sum_by_subtask: Dict[str, float] = field(default_factory=dict)
    total_count_by_subtask: Dict[str, float] = field(default_factory=dict)
    # Warm-start prior derived from member memories' per-memory Q at cluster time.
    prior_alpha_by_subtask: Dict[str, float] = field(default_factory=dict)
    prior_beta_by_subtask: Dict[str, float] = field(default_factory=dict)
    # Region failure summary (generated dynamically after clustering)
    # See docs/REGION_FAILURE_SUMMARY.md
    failure_summary: str = ""
    # Region success pattern summary (aggregated effective steps from success members)
    # Symmetric to failure_summary. Built after clustering.
    success_summary: str = ""
    # Region experience cards: atomic fact cards distilled from pass+fail members.
    # Each card is a compact constraint/gotcha (<80 tokens). Built after clustering.
    experience_cards: List[str] = field(default_factory=list)


class RegionManager:
    """
    Utility-driven region manager with soft membership.

    - Maintains per-subtask Q for each memory
    - Clusters memories by utility pattern (HDBSCAN, auto K)
    - Soft membership: each memory has weights to all regions
    - Region score = weighted average of region utilities
    """

    def __init__(
        self,
        task_hierarchy: Dict[str, Dict[str, Any]],
        K_global: int = 30,
        K_local: int = 10,
        alpha: float = 0.1,
        min_cluster_size: int = 30,
        min_samples: int = 0,
        cluster_selection_method: str = "eom",
        max_region_share: float = 0.0,
        temperature: float = 1.0,
        shrinkage_top_n: int = 3,
        shrinkage_min_utility_margin: float = 0.0,
        region_precluster_evidence_mode: str = "off",
        region_precluster_evidence_scale: float = 1.0,
        region_utility_mode: str = "ema",
        bayesian_smoothing_C: float = 0.5,
        propagation_enabled: bool = True,
        propagation_eta: float = 0.03,
        propagation_k: int = 8,
        propagation_sim_min: float = 0.60,
        cluster_space: str = "capability",
        variance_weighted_dist: bool = False,
        region_split_evidence_migration_mode: str = "soft_source_conserving",
        region_topology_updates_enabled: bool = True,
        region_evidence_sharpen_alpha: float = 2.0,
        region_split_range_fraction: float = 0.15,
        region_max_variance_splits_per_epoch: int = 0,
        region_split_min_effective_evidence: float = 0.0,
        region_progressive_best_split: bool = False,
        region_max_merges_per_epoch: int = 0,
        region_split_min_child_size: int = 1,
        region_protect_new_split_children: bool = False,
    ):
        self.task_hierarchy = task_hierarchy
        self.K_global = K_global
        self.K_local = K_local
        self.alpha = alpha
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_method = cluster_selection_method
        self.max_region_share = max_region_share
        self.temperature = temperature
        self.shrinkage_top_n = shrinkage_top_n
        if cluster_space not in {"capability", "embedding"}:
            raise ValueError(f"unknown cluster_space={cluster_space!r}")
        self.cluster_space = cluster_space
        if not np.isfinite(shrinkage_min_utility_margin) or float(shrinkage_min_utility_margin) < 0.0:
            raise ValueError("shrinkage_min_utility_margin must be a finite non-negative float")
        # Leaf-v4 safety gate: when Region utilities for the target subtask are
        # nearly tied, Region has no reliable winner. In that case retrieval
        # abstains from shrinkage and retains the per-memory Q ordering.
        self.shrinkage_min_utility_margin = float(shrinkage_min_utility_margin)
        valid_precluster_modes = {"off", "soft_source_backfill"}
        if region_precluster_evidence_mode not in valid_precluster_modes:
            raise ValueError(
                "region_precluster_evidence_mode must be one of "
                f"{sorted(valid_precluster_modes)}, got {region_precluster_evidence_mode!r}"
            )
        if not np.isfinite(region_precluster_evidence_scale) or float(region_precluster_evidence_scale) < 0.0:
            raise ValueError("region_precluster_evidence_scale must be a finite non-negative float")
        # Leaf-v5: retain raw direct outcomes observed before the first Region
        # cluster and route them once through the initial soft memberships. This
        # preserves pre-cluster signal without reconstructing pseudo-evidence
        # from EMA Q or q_count.
        self.region_precluster_evidence_mode = region_precluster_evidence_mode
        self.region_precluster_evidence_scale = float(region_precluster_evidence_scale)
        self._precluster_evidence_backfilled = False
        self.region_utility_mode = region_utility_mode  # "ema" or "beta"
        self.bayesian_smoothing_C = bayesian_smoothing_C
        self.variance_weighted_dist = variance_weighted_dist
        valid_migration_modes = {"soft_source_conserving", "hard_member_rebase", "legacy_qcount_reconstruction"}
        if region_split_evidence_migration_mode not in valid_migration_modes:
            raise ValueError(
                "region_split_evidence_migration_mode must be one of "
                f"{sorted(valid_migration_modes)}, got {region_split_evidence_migration_mode!r}"
            )
        # This controls topology changes only. Online region updates remain the
        # existing top-K sharpened soft update in both modes.
        self.region_split_evidence_migration_mode = region_split_evidence_migration_mode
        # Warm-up branches can collect source-attributed soft evidence without
        # applying a topology edit to a legacy checkpoint that lacks it.
        self.region_topology_updates_enabled = bool(region_topology_updates_enabled)
        # Last section (1-based) that actually changed Region topology. Runner
        # persists this through snapshots so cooldown survives preemption/resume.
        self.topology_last_edit_section = 0
        # One-shot global steps already used for mid-section topology maintenance.
        # Persisted so eviction/resume cannot replay the same structural edit.
        self.topology_mid_maintenance_done_steps: set[int] = set()
        self.region_evidence_sharpen_alpha = max(0.1, float(region_evidence_sharpen_alpha))
        if not np.isfinite(region_split_range_fraction) or not (0.0 < float(region_split_range_fraction) <= 1.0):
            raise ValueError("region_split_range_fraction must be a finite float in (0, 1]")
        self.region_split_range_fraction = float(region_split_range_fraction)
        # 0 keeps the historical unlimited behavior. Positive values limit only
        # variance-triggered splits; the dominant-share path is intrinsically
        # one split/cycle and remains unchanged.
        self.region_max_variance_splits_per_epoch = max(0, int(region_max_variance_splits_per_epoch))
        self.region_split_min_effective_evidence = max(0.0, float(region_split_min_effective_evidence))
        self.region_progressive_best_split = bool(region_progressive_best_split)
        self.region_max_merges_per_epoch = max(0, int(region_max_merges_per_epoch))
        self.region_split_min_child_size = max(1, int(region_split_min_child_size))
        self.region_protect_new_split_children = bool(region_protect_new_split_children)

        # Similarity propagation params (validated)
        self.propagation_enabled = propagation_enabled
        self.propagation_eta = max(0.0, min(1.0, propagation_eta))
        self.propagation_k = max(0, propagation_k)
        self.propagation_sim_min = max(0.0, min(0.99, propagation_sim_min))

        # Embedding reference (set by RegionMemoryService)
        self._embedding_lookup = None  # Callable: mem_id -> np.ndarray or None
        self._invalidate_embedding_cache = None  # Callable: () -> None

        # Subtask embeddings for zero-shot transfer: {subtask_name: np.ndarray}
        self._subtask_embeddings: Dict[str, np.ndarray] = {}

        # Per-subtask Q for each memory: {mem_id: {subtask: q_value}}
        self.subtask_q: Dict[str, Dict[str, float]] = {}

        # Per-(memory, subtask) interaction counts for shrinkage estimation
        self.subtask_q_counts: Dict[str, Dict[str, int]] = {}

        # Direct, unpropagated reward evidence owned by each memory. Unlike Q,
        # these are Beta sufficient statistics and may safely follow a member
        # when its region is split.
        self.memory_success_sum_by_subtask: Dict[str, Dict[str, float]] = {}
        self.memory_total_count_by_subtask: Dict[str, Dict[str, float]] = {}
        # Leaf-v5 keeps an isolated pre-cluster raw-outcome window. It is not
        # used by normal online updates and is cleared immediately after its
        # one-time source-conserving initial-cluster backfill.
        self.precluster_success_sum_by_subtask: Dict[str, Dict[str, float]] = {}
        self.precluster_total_count_by_subtask: Dict[str, Dict[str, float]] = {}
        # Old checkpoints lack these ledgers; their aggregate region evidence
        # cannot be attributed exactly, so they use a conservative fallback.
        self._has_complete_memory_evidence_ledger = True

        # Source-attributed soft region evidence.  A value at
        # [region_id][memory_id][subtask] is not a second observation: it is
        # the exact weighted slice of that memory's direct feedback that the
        # online top-K soft update wrote into this region.  Keeping this ledger
        # lets split reroute existing region evidence without ever reconstructing
        # it from EMA Q or q_count.
        self.region_source_success_by_region: Dict[int, Dict[str, Dict[str, float]]] = {}
        self.region_source_total_by_region: Dict[int, Dict[str, Dict[str, float]]] = {}
        self._has_complete_region_source_evidence_ledger = True

        # Regions
        self.regions: List[Region] = []

        # Soft membership: {mem_id: [weight_region_0, weight_region_1, ...]}
        self.membership_weights: Dict[str, np.ndarray] = {}

        # All known subtasks
        self._known_subtasks: List[str] = []

        # Clustering state
        self._is_clustered = False

        # Global reward tracking for Bayesian prior
        self._global_reward_sum = 0.0
        self._global_reward_count = 0

        # Trajectory logging state
        self._trajectory_log_file = None
        self._trajectory_step = 0

    def enable_trajectory_logging(self, log_path: str) -> None:
        """Enable per-update trajectory logging for Pilot B analysis.

        Writes one JSON line per (memory, subtask) update to log_path.
        Call once after init, before training starts.
        """
        global _trajectory_log_enabled
        p = Path(log_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._trajectory_log_file = open(p, "a", encoding="utf-8")
        _trajectory_log_enabled = True
        logger.info("Trajectory logging enabled: %s", log_path)

    def close_trajectory_log(self) -> None:
        if self._trajectory_log_file is not None:
            self._trajectory_log_file.close()
            self._trajectory_log_file = None

    def _get_global_mean(self) -> float:
        """Global mean reward. Returns 0.5 (neutral prior) when no data."""
        if self._global_reward_count == 0:
            return 0.5
        return self._global_reward_sum / self._global_reward_count

    def set_subtask_embedding(self, subtask: str, embedding: np.ndarray) -> None:
        """Register a subtask's embedding for zero-shot transfer."""
        self._subtask_embeddings[subtask] = np.array(embedding)

    # ---------- Per-subtask Q Updates ----------

    def update_subtask_q(
        self,
        memory_ids: List[str],
        target_subtask: str,
        reward: float,
        evidence_weight: float = 1.0,
    ):
        """
        Update per-subtask Q for retrieved memories.
        Also updates region-level stats weighted by membership.

        Additionally, if advanced_transfer_mgr is available and we have enough
        subtask data, update calibration model online.
        """
        evidence_weight = float(max(0.0, evidence_weight))
        if evidence_weight <= 0.0:
            return
        if memory_ids:
            if len(self.subtask_q) % 50 == 0 and len(self.subtask_q) > 0:
                logger.info(
                    "update_subtask_q: tracking %d mems, %d subtasks, global_count=%d",
                    len(self.subtask_q), len(self._known_subtasks), self._global_reward_count,
                )
        self._global_reward_sum += evidence_weight * reward
        self._global_reward_count += evidence_weight

        # Track if this is a new subtask
        is_new_subtask = target_subtask not in self._known_subtasks

        if is_new_subtask:
            self._known_subtasks.append(target_subtask)

        # Increment step counter for trajectory logging
        self._trajectory_step += 1

        for mem_id in memory_ids:
            if mem_id not in self.subtask_q:
                self.subtask_q[mem_id] = {}

            q_dict = self.subtask_q[mem_id]
            old_q = q_dict.get(target_subtask, 0.5)

            # Get observation count BEFORE update
            if mem_id not in self.subtask_q_counts:
                self.subtask_q_counts[mem_id] = {}
            cnt_dict = self.subtask_q_counts[mem_id]
            n_before = cnt_dict.get(target_subtask, 0)

            # Compute shrinkage Q BEFORE update (for trajectory logging)
            shrinkage_q_before = None
            region_utility_before = None
            lambda_before = None
            if self._trajectory_log_file is not None and self._is_clustered:
                try:
                    region_utility_before = self._get_weighted_region_utility(
                        mem_id, target_subtask, top_n=self.shrinkage_top_n
                    )
                    if n_before == 0:
                        shrinkage_q_before = region_utility_before
                        lambda_before = 0.0
                    else:
                        # Use same tau_sq, sigma_sq as compute_shrinkage_q
                        tau_sq = 0.05
                        sigma_sq = 0.25
                        lambda_ms = tau_sq / (tau_sq + sigma_sq / n_before)
                        lambda_ms = min(lambda_ms, 0.8)  # lambda_max
                        shrinkage_q_before = lambda_ms * old_q + (1 - lambda_ms) * region_utility_before
                        lambda_before = lambda_ms
                except Exception:
                    pass  # Best-effort logging

            # Update per-memory Q
            new_q = old_q + self.alpha * evidence_weight * (reward - old_q)
            q_dict[target_subtask] = new_q

            # Update observation count
            cnt_dict[target_subtask] = n_before + evidence_weight

            # Record direct, unpropagated evidence separately from EMA Q.
            # Once clustered, it belongs to the normal member ledger. Before
            # the first cluster, Leaf-v5 optionally stores the same raw outcome
            # in a temporary ledger which is routed exactly once after initial
            # soft memberships are available.
            if self._is_clustered:
                mem_success = self.memory_success_sum_by_subtask.setdefault(mem_id, {})
                mem_total = self.memory_total_count_by_subtask.setdefault(mem_id, {})
            elif self.region_precluster_evidence_mode == "soft_source_backfill":
                mem_success = self.precluster_success_sum_by_subtask.setdefault(mem_id, {})
                mem_total = self.precluster_total_count_by_subtask.setdefault(mem_id, {})
            else:
                mem_success = mem_total = None
            if mem_success is not None and mem_total is not None:
                mem_success[target_subtask] = (
                    mem_success.get(target_subtask, 0.0) + evidence_weight * float(reward)
                )
                mem_total[target_subtask] = (
                    mem_total.get(target_subtask, 0.0) + evidence_weight
                )

            # Trajectory logging (if enabled)
            if self._trajectory_log_file is not None:
                try:
                    log_entry = {
                        "step": self._trajectory_step,
                        "memory_id": mem_id,
                        "target_subtask": target_subtask,
                        "reward": float(reward),
                        "n_before": n_before,
                        "n_after": n_before + evidence_weight,
                        "evidence_weight": evidence_weight,
                        "q_before": float(old_q),
                        "q_after": float(new_q),
                        "region_utility": float(region_utility_before) if region_utility_before is not None else None,
                        "shrinkage_q_before": float(shrinkage_q_before) if shrinkage_q_before is not None else None,
                        "lambda": float(lambda_before) if lambda_before is not None else None,
                        "is_clustered": self._is_clustered,
                    }
                    self._trajectory_log_file.write(json.dumps(log_entry) + "\n")
                    # Flush periodically to avoid losing data on crash
                    if self._trajectory_step % 100 == 0:
                        self._trajectory_log_file.flush()
                except Exception as e:
                    logger.warning("Trajectory logging failed: %s", e)

        # Update region-level stats with top-3 sharpened weights.
        # Sharpen (w^α, α=2) + renormalize over top-3 to concentrate learning
        # signal while still letting neighboring regions learn.
        _UPDATE_TOP_K = 3
        # Configurable only for the online top-K evidence allocation.  It does
        # not alter geometry/memberships or the top-1 shrinkage lookup.
        _SHARPEN_ALPHA = self.region_evidence_sharpen_alpha

        if self._is_clustered:
            for mem_id in memory_ids:
                weights = self.membership_weights.get(mem_id)
                if weights is None:
                    weights = self._assign_membership_lazy(mem_id)
                    if weights is None:
                        continue

                # Top-K regions by membership weight
                n_regions = min(len(weights), len(self.regions))
                top_k = min(_UPDATE_TOP_K, n_regions)
                top_rids = sorted(range(n_regions), key=lambda i: weights[i], reverse=True)[:top_k]

                # Sharpen + renormalize
                sharp = [weights[rid] ** _SHARPEN_ALPHA for rid in top_rids]
                total_sharp = sum(sharp) or 1e-8
                norm_weights = [s / total_sharp for s in sharp]

                for idx, rid in enumerate(top_rids):
                    w = norm_weights[idx]
                    if w < 0.01:
                        continue
                    region = self.regions[rid]

                    if self.region_utility_mode == "beta":
                        region.success_sum_by_subtask[target_subtask] = (
                            region.success_sum_by_subtask.get(target_subtask, 0.0)
                            + w * evidence_weight * reward
                        )
                        region.total_count_by_subtask[target_subtask] = (
                            region.total_count_by_subtask.get(target_subtask, 0.0)
                            + w * evidence_weight
                        )
                        # Keep a source-attributed copy of exactly the same
                        # soft-weighted evidence.  Do this at the same update
                        # point as region evidence; Q, q_count and propagated
                        # values never enter this ledger.
                        src_success = self.region_source_success_by_region.setdefault(region.region_id, {})
                        src_total = self.region_source_total_by_region.setdefault(region.region_id, {})
                        src_success_mem = src_success.setdefault(mem_id, {})
                        src_total_mem = src_total.setdefault(mem_id, {})
                        src_success_mem[target_subtask] = (
                            src_success_mem.get(target_subtask, 0.0)
                            + w * evidence_weight * float(reward)
                        )
                        src_total_mem[target_subtask] = (
                            src_total_mem.get(target_subtask, 0.0)
                            + w * evidence_weight
                        )
                        # Beta posterior = warm-start prior + post-cluster weighted evidence.
                        # Fall back to (2, 2) uninformative prior for subtasks never seen at cluster time.
                        a0 = region.prior_alpha_by_subtask.get(target_subtask, 2.0)
                        b0 = region.prior_beta_by_subtask.get(target_subtask, 2.0)
                        s = region.success_sum_by_subtask[target_subtask]
                        n = region.total_count_by_subtask[target_subtask]
                        region.utility_by_subtask[target_subtask] = (s + a0) / (n + a0 + b0)
                    else:
                        old_u = region.utility_by_subtask.get(target_subtask, 0.5)
                        region.utility_by_subtask[target_subtask] = (
                            old_u + self.alpha * w * evidence_weight * (reward - old_u)
                        )

                    region.counts_by_subtask[target_subtask] = (
                        region.counts_by_subtask.get(target_subtask, 0) + 1
                    )

        # Similarity propagation: spread reward signal to embedding-similar memories
        if self.propagation_enabled and self._embedding_lookup and memory_ids:
            self._propagate_q_to_neighbors(
                memory_ids, target_subtask, reward, evidence_weight=evidence_weight
            )

    def _propagate_q_to_neighbors(
        self,
        source_mem_ids: List[str],
        target_subtask: str,
        reward: float,
        evidence_weight: float = 1.0,
    ) -> None:
        """One-hop Q propagation from retrieved memories to embedding-similar neighbors."""
        # Invalidate embedding cache so new memories are discoverable
        if hasattr(self, '_invalidate_embedding_cache') and self._invalidate_embedding_cache:
            self._invalidate_embedding_cache()

        eta = self.propagation_eta
        k = self.propagation_k
        sim_min = self.propagation_sim_min
        gamma = 2.0
        prop_count_scale = 0.05
        source_set = set(source_mem_ids)

        for src_id in source_mem_ids:
            src_emb = self._embedding_lookup(src_id)
            if src_emb is None:
                continue

            neighbors = self._find_similar_memories(src_emb, source_set, k * 3)
            if not neighbors:
                continue

            count = 0
            for nbr_id, sim in neighbors:
                if count >= k:
                    break
                if not np.isfinite(sim) or sim < sim_min:
                    continue
                count += 1

                sim_norm = (sim - sim_min) / max(1e-8, 1.0 - sim_min)
                sim_norm = max(0.0, min(1.0, sim_norm))
                w = eta * evidence_weight * (sim_norm ** gamma)
                if w <= 1e-6:
                    continue

                if nbr_id not in self.subtask_q:
                    self.subtask_q[nbr_id] = {}
                q_dict = self.subtask_q[nbr_id]
                old_q = q_dict.get(target_subtask, 0.5)
                q_dict[target_subtask] = old_q + w * (reward - old_q)

                # NOTE: do NOT increment subtask_q_counts for propagated updates.
                # Propagated Q comes from embedding similarity (orthogonal to task
                # utility). Incrementing counts would inflate shrinkage λ, making
                # the system over-trust propagated values as if directly observed.

    def _find_similar_memories(
        self,
        query_emb: np.ndarray,
        exclude_ids: set,
        top_k: int,
    ) -> List[Tuple[str, float]]:
        """Find top-k similar memories by cosine similarity, excluding given IDs."""
        if self._embedding_lookup is None:
            return []

        query_norm = np.linalg.norm(query_emb)
        if query_norm < 1e-8 or not np.all(np.isfinite(query_emb)):
            return []
        q = query_emb / query_norm
        q_dim = len(q)

        results = []
        for mem_id in self.subtask_q:
            if mem_id in exclude_ids:
                continue
            emb = self._embedding_lookup(mem_id)
            if emb is None:
                continue
            if len(emb) != q_dim:
                continue
            if not np.all(np.isfinite(emb)):
                continue
            emb_norm = np.linalg.norm(emb)
            if emb_norm < 1e-8:
                continue
            sim = float(np.dot(q, emb / emb_norm))
            results.append((mem_id, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_subtask_q(self, mem_id: str, target_subtask: str) -> float:
        """Get a memory's Q value for a specific subtask."""
        q_dict = self.subtask_q.get(mem_id)
        if q_dict is None:
            return 0.5
        return q_dict.get(target_subtask, 0.5)

    def get_observation_count(self, mem_id: str, target_subtask: str) -> float:
        """Get observation count for a (memory, subtask) pair. May be fractional from propagation."""
        cnt_dict = self.subtask_q_counts.get(mem_id)
        if cnt_dict is None:
            return 0
        return cnt_dict.get(target_subtask, 0)

    def _get_weighted_region_utility(self, mem_id: str, target_subtask: str, top_n: int = 0) -> float:
        """
        Weighted region utility using only the top-N strongest membership regions.

        Full soft membership over many regions (e.g. 79) produces near-uniform
        weights (~1/79), washing out any signal. Using top-N and renormalizing
        concentrates weight on the most relevant regions.

        When top_n=1, this is hard assignment (sharpest signal).
        """
        if not self._is_clustered:
            return 0.5

        weights = self.membership_weights.get(mem_id)
        if weights is None:
            return 0.5

        effective_top_n = top_n if top_n > 0 else self.shrinkage_top_n

        # Collect (weight, region_utility) pairs for valid regions
        is_zero_shot = target_subtask not in self._known_subtasks
        pairs = []
        for rid, w in enumerate(weights):
            if w < 0.001 or rid >= len(self.regions):
                continue
            region = self.regions[rid]
            if target_subtask in region.utility_by_subtask:
                u = region.utility_by_subtask[target_subtask]
            elif is_zero_shot:
                u, _, _ = self._estimate_region_utility_zero_shot(region, target_subtask)
            else:
                u = 0.5
            pairs.append((w, u))

        if not pairs:
            return 0.5

        # Keep only top-N by weight
        pairs.sort(key=lambda x: x[0], reverse=True)
        pairs = pairs[:effective_top_n]

        # Renormalize weights
        total_w = sum(w for w, _ in pairs)
        if total_w < 1e-9:
            return 0.5

        utility = sum((w / total_w) * u for w, u in pairs)
        return utility

    def get_subtask_utility_margin(self, target_subtask: str) -> Optional[float]:
        """Return the top1-top2 Region utility margin for a subtask.

        ``None`` means fewer than two Region utilities are available, so the
        abstention gate does not fire. This measures confidence globally per
        query subtask while shrinkage remains memory-specific.
        """
        if not self._is_clustered or len(self.regions) < 2:
            return None
        utilities = []
        is_zero_shot = target_subtask not in self._known_subtasks
        for region in list(self.regions):
            if target_subtask in region.utility_by_subtask:
                utility = region.utility_by_subtask[target_subtask]
            elif is_zero_shot:
                utility, _, _ = self._estimate_region_utility_zero_shot(region, target_subtask)
            else:
                continue
            if isinstance(utility, (int, float)) and np.isfinite(utility):
                utilities.append(float(utility))
        if len(utilities) < 2:
            return None
        utilities.sort(reverse=True)
        return max(0.0, utilities[0] - utilities[1])

    def compute_shrinkage_q(
        self,
        mem_id: str,
        target_subtask: str,
        tau_sq: float = 0.05,
        sigma_sq: float = 0.25,
        lambda_max: float = 0.8,
    ) -> float:
        """
        Compute shrinkage-based Q value blending per-memory Q with region utility.

        Uses James-Stein shrinkage: for cold-start memories (n=0), fully trust
        region utility; as observations accumulate, gradually trust per-memory Q.

        Instance-level overrides (shrinkage_tau_sq, shrinkage_sigma_sq,
        shrinkage_lambda_max) take precedence over argument defaults.

        Args:
            mem_id: Memory ID
            target_subtask: Target subtask
            tau_sq: Prior variance of true Q within region (default 0.05)
            sigma_sq: Observation noise variance (default 0.25 ≈ p(1-p) for p=0.5)
            lambda_max: Maximum weight for per-memory Q (default 0.8).
                        Prevents full algorithm degradation to per-memory-only.

        Returns:
            Shrinkage-adjusted Q value in [0, 1]
        """
        _tau = getattr(self, 'shrinkage_tau_sq', None)
        _sigma = getattr(self, 'shrinkage_sigma_sq', None)
        _lmax = getattr(self, 'shrinkage_lambda_max', None)
        _conf_k = getattr(self, 'shrinkage_confidence_k', None)
        tau_sq = _tau if _tau is not None else tau_sq
        sigma_sq = _sigma if _sigma is not None else sigma_sq
        lambda_max = _lmax if _lmax is not None else lambda_max

        n_ms = self.get_observation_count(mem_id, target_subtask)
        if not isinstance(n_ms, (int, float)) or n_ms < 0 or not np.isfinite(n_ms):
            n_ms = 0
        per_memory_q = self.get_subtask_q(mem_id, target_subtask)

        min_margin = float(getattr(self, "shrinkage_min_utility_margin", 0.0) or 0.0)
        if min_margin > 0.0:
            utility_margin = self.get_subtask_utility_margin(target_subtask)
            if utility_margin is not None and utility_margin < min_margin:
                return float(max(0.0, min(1.0, per_memory_q)))

        region_utility = self._get_weighted_region_utility(mem_id, target_subtask)

        if n_ms == 0:
            if abs(per_memory_q - 0.5) > 0.01:
                return 0.1 * per_memory_q + 0.9 * region_utility
            return region_utility

        # Confidence-gated lambda: lambda = lambda_max * n / (n + k)
        if _conf_k is not None and _conf_k > 0:
            lambda_ms = lambda_max * n_ms / (n_ms + _conf_k)
        else:
            # Standard James-Stein shrinkage
            lambda_ms = tau_sq / (tau_sq + sigma_sq / n_ms)
            lambda_ms = min(lambda_ms, lambda_max)

        shrinkage_q = lambda_ms * per_memory_q + (1 - lambda_ms) * region_utility

        return float(max(0.0, min(1.0, shrinkage_q)))

    def _assign_membership_lazy(self, mem_id: str) -> Optional[np.ndarray]:
        """Assign membership weights for a memory added after clustering.

        Delegates to assign_new_memory for consistent masked distance + auto temperature.
        """
        if not self._is_clustered or not self.regions:
            return None
        q_dict = self.subtask_q.get(mem_id, {})
        if not q_dict:
            return None
        self.assign_new_memory(mem_id)
        return self.membership_weights.get(mem_id)

    # ---------- Utility-Based Clustering with Soft Membership ----------

    def _compute_subtask_correlation(self) -> Optional[np.ndarray]:
        """Compute empirical correlation matrix between subtasks from co-observed Q values.

        Returns [n_subtasks x n_subtasks] correlation matrix, or None if insufficient data.
        Diagonal is 1.0. Off-diagonal is Pearson correlation from memories that have
        both subtasks observed.
        """
        subtasks = self._known_subtasks
        n_st = len(subtasks)
        if n_st < 2:
            return None

        corr = np.eye(n_st)
        min_pairs = 10

        for i in range(n_st):
            for j in range(i + 1, n_st):
                vals_i, vals_j = [], []
                for mem_id, q_dict in self.subtask_q.items():
                    if subtasks[i] in q_dict and subtasks[j] in q_dict:
                        vals_i.append(q_dict[subtasks[i]])
                        vals_j.append(q_dict[subtasks[j]])
                if len(vals_i) >= min_pairs:
                    vi = np.array(vals_i)
                    vj = np.array(vals_j)
                    std_i = np.std(vi)
                    std_j = np.std(vj)
                    if std_i > 1e-6 and std_j > 1e-6:
                        r = float(np.corrcoef(vi, vj)[0, 1])
                        if np.isfinite(r):
                            corr[i, j] = corr[j, i] = r

        return corr

    def _impute_cross_subtask(self, X: np.ndarray) -> np.ndarray:
        """Fill missing Q values using cross-subtask correlation.

        For each memory with observed Q on subtask A but missing on subtask B:
          Q_imputed[B] = global_mean[B] + corr(A,B) * (Q[A] - global_mean[A])

        When multiple observed subtasks exist, use weighted average of predictions
        (weighted by abs(correlation)).

        Returns X with NaN cells filled where possible. Original observed values
        are NOT modified.
        """
        subtasks = self._known_subtasks
        n_st = len(subtasks)

        corr = self._compute_subtask_correlation()
        if corr is None:
            return X

        global_mean = np.zeros(n_st)
        for j in range(n_st):
            col = X[:, j]
            obs = col[~np.isnan(col)]
            global_mean[j] = float(np.mean(obs)) if len(obs) > 0 else 0.5

        X_imputed = X.copy()
        n_imputed = 0

        for i in range(len(X)):
            observed_dims = np.where(~np.isnan(X[i]))[0]
            missing_dims = np.where(np.isnan(X[i]))[0]

            if len(observed_dims) == 0 or len(missing_dims) == 0:
                continue

            for m in missing_dims:
                predictions = []
                weights = []
                for o in observed_dims:
                    r = corr[o, m]
                    if abs(r) < 0.05:
                        continue
                    pred = global_mean[m] + r * (X[i, o] - global_mean[o])
                    pred = max(0.0, min(1.0, pred))
                    predictions.append(pred)
                    weights.append(abs(r))

                if predictions:
                    total_w = sum(weights)
                    imputed_val = sum(p * w for p, w in zip(predictions, weights)) / total_w
                    X_imputed[i, m] = imputed_val
                    n_imputed += 1

        logger.info(
            "Cross-subtask imputation: filled %d/%d missing cells (%.1f%% of NaN)",
            n_imputed, int(np.isnan(X).sum()),
            100 * n_imputed / max(1, int(np.isnan(X).sum())),
        )
        return X_imputed

    def cluster_by_utility(self):
        """
        Cluster memories by utility vectors. Compute soft membership weights
        based on distance to each region centroid (softmax).
        """
        logger.info(
            "cluster_by_utility called: subtask_q=%d mems, known_subtasks=%d, global_count=%d",
            len(self.subtask_q), len(self._known_subtasks), self._global_reward_count,
        )
        if not self.subtask_q or not self._known_subtasks:
            logger.warning("No utility data available for clustering")
            return

        mem_ids = list(self.subtask_q.keys())
        n_mems = len(mem_ids)

        if n_mems < 3:
            logger.warning("Too few memories (%d) for clustering", n_mems)
            return

        subtasks = self._known_subtasks
        n_subtasks = len(subtasks)

        # Build utility matrix + observation mask [n_mems x n_subtasks]
        # NaN for unobserved subtasks (masked distance ignores these dims)
        X = np.full((n_mems, n_subtasks), np.nan)
        for i, mem_id in enumerate(mem_ids):
            q_dict = self.subtask_q[mem_id]
            for j, st in enumerate(subtasks):
                if st in q_dict:
                    X[i, j] = q_dict[st]

        # Cross-subtask imputation: fill missing Q values using correlation
        # between subtasks. This densifies the utility matrix so HDBSCAN can
        # find meaningful clusters even when per-memory observations are sparse.
        X_dense = self._impute_cross_subtask(X)

        embedding_matrix = None
        if self.cluster_space == "embedding":
            rows, kept_ids, kept_X = [], [], []
            for i, mid in enumerate(mem_ids):
                emb = self._embedding_lookup(mid) if self._embedding_lookup else None
                if emb is None:
                    continue
                arr = np.asarray(emb, dtype=float).reshape(-1)
                norm = np.linalg.norm(arr)
                if norm <= 1e-8 or not np.all(np.isfinite(arr)):
                    continue
                rows.append(arr / norm)
                kept_ids.append(mid)
                kept_X.append(X[i])
            if len(rows) < 3:
                raise RuntimeError(f"embedding clustering has only {len(rows)} usable memories")
            mem_ids = kept_ids
            X = np.asarray(kept_X)
            n_mems = len(mem_ids)
            X_dense = self._impute_cross_subtask(X)
            embedding_matrix = np.vstack(rows)
            sim = np.clip(embedding_matrix @ embedding_matrix.T, -1.0, 1.0)
            dist_matrix = 1.0 - sim
            np.fill_diagonal(dist_matrix, 0.0)
            labels = self._hdbscan_cluster_precomputed(dist_matrix, n_mems)
            logger.info("embedding-space HDBSCAN: usable=%d dim=%d", n_mems, embedding_matrix.shape[1])
        else:
            dist_matrix = self._masked_distance_matrix(X_dense, variance_weighted=self.variance_weighted_dist)
            labels = self._hdbscan_cluster_precomputed(dist_matrix, n_mems)
        n_clusters = int(labels.max()) + 1

        # Compute centroids using observed-only means per dimension
        centroids = np.zeros((n_clusters, n_subtasks))
        for c in range(n_clusters):
            cluster_X = X[labels == c]
            for j in range(n_subtasks):
                col = cluster_X[:, j]
                observed = col[~np.isnan(col)]
                centroids[c, j] = float(np.mean(observed)) if len(observed) > 0 else 0.0

        # Build regions
        self.regions = []
        for rid in range(n_clusters):
            mask = labels == rid
            members = [mem_ids[i] for i in range(n_mems) if mask[i]]

            utility_by_subtask: Dict[str, float] = {}
            counts_by_subtask: Dict[str, int] = {}
            success_sum_by_subtask: Dict[str, float] = {}
            total_count_by_subtask: Dict[str, float] = {}
            prior_alpha_by_subtask: Dict[str, float] = {}
            prior_beta_by_subtask: Dict[str, float] = {}
            # Cap warm-start prior strength so per-memory historical Q informs
            # the region prior without dominating subsequent membership-weighted
            # reward updates. ESS=5 means the prior is worth 5 pseudo-trials.
            PRIOR_ESS = 5.0
            # Eps-clamp keeps prior away from exact 0/1, avoiding degenerate Beta.
            EPS = 0.01
            for st in subtasks:
                vals = [
                    self.subtask_q[m].get(st)
                    for m in members
                    if st in self.subtask_q.get(m, {})
                ]
                if vals:
                    mean_q = float(np.mean(vals))
                    mean_q_clamped = min(max(mean_q, EPS), 1.0 - EPS)
                    utility_by_subtask[st] = mean_q
                    counts_by_subtask[st] = len(vals)

                    # Warm-start prior: clipped Beta(α₀, β₀) with α₀+β₀=PRIOR_ESS
                    # centered on the cluster's average per-memory Q. This keeps
                    # the prior informative but bounded so post-cluster reward
                    # updates (w*reward, w) can actually move the posterior.
                    prior_alpha_by_subtask[st] = PRIOR_ESS * mean_q_clamped
                    prior_beta_by_subtask[st] = PRIOR_ESS * (1.0 - mean_q_clamped)

                    # Post-cluster evidence accumulators start empty; they only
                    # receive weighted reward updates from update_subtask_q.
                    success_sum_by_subtask[st] = 0.0
                    total_count_by_subtask[st] = 0.0

            self.regions.append(Region(
                region_id=rid,
                centroid=centroids[rid],
                member_ids=members,
                utility_by_subtask=utility_by_subtask,
                counts_by_subtask=counts_by_subtask,
                success_sum_by_subtask=success_sum_by_subtask,
                total_count_by_subtask=total_count_by_subtask,
                prior_alpha_by_subtask=prior_alpha_by_subtask,
                prior_beta_by_subtask=prior_beta_by_subtask,
            ))
            if embedding_matrix is not None:
                erows = embedding_matrix[labels == rid]
                ec = np.mean(erows, axis=0)
                en = np.linalg.norm(ec)
                self.regions[-1]._embedding_centroid = ec / en if en > 1e-8 else ec

        # Compute soft membership weights using imputed matrix for denser distance
        self.membership_weights = {}
        all_dists = []
        for i, mem_id in enumerate(mem_ids):
            dists = []
            for r in range(n_clusters):
                shared = 0
                sq_sum = 0.0
                for j in range(n_subtasks):
                    if not np.isnan(X_dense[i, j]):
                        shared += 1
                        sq_sum += (X_dense[i, j] - centroids[r, j]) ** 2
                if shared > 0:
                    dists.append(float(np.sqrt(sq_sum / shared)))
                else:
                    dists.append(1.0)
            all_dists.append(dists)

        # Temperature: use configured value, fall back to auto (0.5 * median_dist)
        flat_dists = [d for row in all_dists for d in row]
        median_d = float(np.median(flat_dists)) if flat_dists else 1.0
        auto_tau = max(median_d * 0.5, 0.01)
        tau = min(self.temperature, auto_tau) if self.temperature > 0 else auto_tau

        for i, mem_id in enumerate(mem_ids):
            dists = np.array(all_dists[i])
            neg_dists = -dists / tau
            neg_dists -= neg_dists.max()
            exp_vals = np.exp(neg_dists)
            weights = exp_vals / exp_vals.sum()
            self.membership_weights[mem_id] = weights

        # HDBSCAN labels seed the centroids, but soft membership weights are the
        # canonical Region assignment used by gating and summary routing. Make
        # hard member_ids agree with the soft argmax before building summaries.
        self.rebuild_hard_memberships_from_weights()

        self._is_clustered = True
        self._backfill_precluster_evidence_once()
        self._cluster_version = getattr(self, '_cluster_version', 0) + 1
        logger.info(
            "Utility clustering (soft): %d regions from %d memories (%d dims)",
            n_clusters, n_mems, n_subtasks,
        )

        # Build per-region failure summaries from member memories
        # (see docs/REGION_FAILURE_SUMMARY.md)
        self._build_region_failure_summaries()
        # Build per-region success pattern summaries (symmetric to failure)
        self._build_region_success_summaries()
        # Build per-region experience cards (atomic fact cards from pass+fail)
        self._build_region_experience_cards()

    def _backfill_precluster_evidence_once(self) -> None:
        """Route raw pre-cluster outcomes into initial Region source ledgers.

        The routing exactly mirrors online beta evidence allocation: retain the
        top three memberships, square/sharpen, renormalize, and ignore tiny
        shares. The configured scale deliberately discounts cluster-fitting
        history; it is never derived from Q values or q-counts.
        """
        if self._precluster_evidence_backfilled:
            return
        self._precluster_evidence_backfilled = True
        if (self.region_precluster_evidence_mode != "soft_source_backfill"
                or self.region_precluster_evidence_scale <= 0.0
                or self.region_utility_mode != "beta"
                or not self.regions):
            self.precluster_success_sum_by_subtask = {}
            self.precluster_total_count_by_subtask = {}
            return

        scale = self.region_precluster_evidence_scale
        routed_total = 0.0
        routed_success = 0.0
        source_ids = set(self.precluster_success_sum_by_subtask) | set(self.precluster_total_count_by_subtask)
        for mem_id in source_ids:
            weights = self.membership_weights.get(mem_id)
            if weights is None:
                continue
            n_regions = min(len(weights), len(self.regions))
            if n_regions <= 0:
                continue
            top_rids = sorted(range(n_regions), key=lambda rid: weights[rid], reverse=True)[:min(3, n_regions)]
            sharp = [float(weights[rid]) ** self.region_evidence_sharpen_alpha for rid in top_rids]
            denom = sum(sharp)
            if denom <= 0.0 or not np.isfinite(denom):
                continue
            for rid, sharp_weight in zip(top_rids, sharp):
                weight = sharp_weight / denom
                if weight < 0.01:
                    continue
                src_success = self.region_source_success_by_region.setdefault(int(self.regions[rid].region_id), {}).setdefault(mem_id, {})
                src_total = self.region_source_total_by_region.setdefault(int(self.regions[rid].region_id), {}).setdefault(mem_id, {})
                subtasks = set(self.precluster_success_sum_by_subtask.get(mem_id, {})) | set(self.precluster_total_count_by_subtask.get(mem_id, {}))
                for subtask in subtasks:
                    success = scale * weight * float(self.precluster_success_sum_by_subtask.get(mem_id, {}).get(subtask, 0.0))
                    total = scale * weight * float(self.precluster_total_count_by_subtask.get(mem_id, {}).get(subtask, 0.0))
                    if success:
                        src_success[subtask] = src_success.get(subtask, 0.0) + success
                        routed_success += success
                    if total:
                        src_total[subtask] = src_total.get(subtask, 0.0) + total
                        routed_total += total

        self._refresh_source_evidence_from_ledgers(list(self._known_subtasks))
        logger.info(
            "Initial-cluster pre-evidence backfill: mode=%s scale=%.3f sources=%d "
            "routed_success=%.6f routed_total=%.6f",
            self.region_precluster_evidence_mode, scale, len(source_ids),
            routed_success, routed_total,
        )
        # Do not replay this history after checkpoint resume. The routed source
        # slices above are persisted and will thereafter migrate conservatively.
        self.precluster_success_sum_by_subtask = {}
        self.precluster_total_count_by_subtask = {}

    @staticmethod
    def _masked_distance_matrix(X: np.ndarray, variance_weighted: bool = False) -> np.ndarray:
        """Compute pairwise distance matrix ignoring NaN (unobserved) dimensions.

        For each pair (i, j), distance = sqrt(weighted mean of squared diffs on
        co-observed dims). When variance_weighted=True, dimensions are weighted
        by their variance so high-variance (informative) dims dominate clustering.
        Pairs with no co-observed dims get max distance.
        """
        n, d = X.shape
        D = np.zeros((n, n))
        observed = ~np.isnan(X)  # [n, d] bool

        if variance_weighted:
            dim_std = np.ones(d)
            for j in range(d):
                col = X[:, j]
                obs = col[~np.isnan(col)]
                if len(obs) > 1:
                    dim_std[j] = max(float(np.std(obs)), 1e-6)
            dim_weight = dim_std ** 2
            dim_weight = dim_weight / dim_weight.sum() * d
        else:
            dim_weight = np.ones(d)

        for i in range(n):
            for j in range(i + 1, n):
                mask = observed[i] & observed[j]  # co-observed dims
                n_shared = mask.sum()
                if n_shared == 0:
                    D[i, j] = D[j, i] = 1.0  # max distance for no overlap
                else:
                    diff = X[i, mask] - X[j, mask]
                    w = dim_weight[mask]
                    D[i, j] = D[j, i] = float(np.sqrt(np.sum(w * diff ** 2) / np.sum(w)))
        return D

    def _hdbscan_cluster_precomputed(self, dist_matrix: np.ndarray, n_samples: int) -> np.ndarray:
        """HDBSCAN on precomputed distance matrix. Noise assigned to nearest."""
        import hdbscan

        min_cs = max(2, min(self.min_cluster_size, n_samples // 5))
        min_s = self.min_samples if self.min_samples > 0 else max(1, min_cs // 2)
        min_s = min(min_s, max(1, min_cs))

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cs,
            min_samples=min_s,
            metric="precomputed",
            cluster_selection_method=self.cluster_selection_method,
        )
        labels = clusterer.fit_predict(dist_matrix)

        logger.debug("HDBSCAN precomputed: min_cs=%d, min_s=%d, method=%s, n_clusters=%d",
                      min_cs, min_s, self.cluster_selection_method,
                      labels.max() + 1 if labels.max() >= 0 else 0)

        n_clusters = labels.max() + 1 if labels.max() >= 0 else 0
        if n_clusters == 0:
            return np.zeros(n_samples, dtype=int)

        # Assign noise to nearest cluster centroid
        noise_mask = labels == -1
        if noise_mask.any():
            for i in np.where(noise_mask)[0]:
                # Find nearest non-noise point's cluster
                non_noise = np.where(labels >= 0)[0]
                if len(non_noise) > 0:
                    nearest = non_noise[dist_matrix[i, non_noise].argmin()]
                    labels[i] = labels[nearest]

        return labels

    def register_memory(
        self, mem_id: str, source_subtask: str, initial_q: float = 0.0,
        *, assign_if_clustered: bool = True,
    ) -> bool:
        """Register a newly stored memory in Region geometry without reward evidence.

        This creates only a per-memory Q coordinate used by clustering/membership.
        It deliberately does NOT increment visit counts, direct evidence ledgers,
        source evidence ledgers, or Region Beta posteriors. Those are updated only
        when the memory is actually retrieved and receives an outcome.
        """
        if not mem_id or not source_subtask:
            return False
        try:
            q = float(initial_q)
        except (TypeError, ValueError):
            q = 0.0
        if not np.isfinite(q):
            q = 0.0
        q = float(max(0.0, min(1.0, q)))
        if source_subtask not in self._known_subtasks:
            self._known_subtasks.append(source_subtask)
        q_dict = self.subtask_q.setdefault(str(mem_id), {})
        changed = source_subtask not in q_dict
        q_dict.setdefault(source_subtask, q)
        self.subtask_q_counts.setdefault(str(mem_id), {})
        if assign_if_clustered and self._is_clustered and str(mem_id) not in self.membership_weights:
            self.assign_new_memory(str(mem_id))
        return changed

    def assign_new_memory(self, mem_id: str) -> None:
        """Assign a new memory to existing regions using masked distance."""
        if not self._is_clustered or not self.regions:
            return

        subtasks = self._known_subtasks
        if not subtasks:
            return

        q_dict = self.subtask_q.get(mem_id, {})

        if self.cluster_space == "embedding":
            emb = self._embedding_lookup(mem_id) if self._embedding_lookup else None
            if emb is None:
                return
            vec = np.asarray(emb, dtype=float).reshape(-1)
            norm = np.linalg.norm(vec)
            if norm <= 1e-8 or not np.all(np.isfinite(vec)):
                return
            vec = vec / norm
            dists = np.asarray([
                1.0 - float(np.dot(vec, getattr(r, "_embedding_centroid", vec * 0)))
                for r in self.regions
            ])
            median_d = float(np.median(dists)) if len(dists) > 1 else 1.0
            tau = min(self.temperature, max(median_d * 0.5, 0.01)) if self.temperature > 0 else max(median_d * 0.5, 0.01)
            logits = -dists / tau
            logits -= logits.max()
            weights = np.exp(logits)
            weights /= weights.sum()
            self.membership_weights[mem_id] = weights
            self.regions[int(np.argmin(dists))].member_ids.append(mem_id)
            return

        # Masked distance to each region centroid (only co-observed dims)
        centroids = [r.centroid for r in self.regions if r.centroid is not None]
        if not centroids:
            return

        dists = []
        for centroid in centroids:
            shared = 0
            sq_sum = 0.0
            for j, st in enumerate(subtasks):
                if st in q_dict and j < len(centroid):
                    shared += 1
                    sq_sum += (q_dict[st] - centroid[j]) ** 2
            if shared > 0:
                dists.append(float(np.sqrt(sq_sum / shared)))
            else:
                dists.append(1.0)
        dists = np.array(dists)

        # Temperature: use configured value, fall back to auto
        median_d = float(np.median(dists)) if len(dists) > 1 else 1.0
        auto_tau = max(median_d * 0.5, 0.01)
        tau = min(self.temperature, auto_tau) if self.temperature > 0 else auto_tau
        neg_dists = -dists / tau
        neg_dists -= neg_dists.max()
        exp_vals = np.exp(neg_dists)
        weights = exp_vals / exp_vals.sum()
        self.membership_weights[mem_id] = weights

        best_rid = int(np.argmin(dists))
        if mem_id not in self.regions[best_rid].member_ids:
            self.regions[best_rid].member_ids.append(mem_id)

    def _refresh_direct_evidence_from_members(self, subtasks: List[str]) -> None:
        """Rebuild every region's direct posterior evidence from member ledgers.

        Must run after hard-membership rebuilds because soft-to-hard argmax can
        move a memory across children after split/merge. Only direct outcome
        ledgers are aggregated; EMA Q, q_count, scores, and propagation stay out.
        """
        if not self._has_complete_memory_evidence_ledger:
            return
        for region in self.regions:
            state_subtasks = set(subtasks)
            for mid in region.member_ids:
                state_subtasks.update(self.memory_success_sum_by_subtask.get(mid, {}))
                state_subtasks.update(self.memory_total_count_by_subtask.get(mid, {}))
            region.success_sum_by_subtask = {
                st: sum(float(self.memory_success_sum_by_subtask.get(mid, {}).get(st, 0.0))
                        for mid in region.member_ids)
                for st in state_subtasks
            }
            region.total_count_by_subtask = {
                st: sum(float(self.memory_total_count_by_subtask.get(mid, {}).get(st, 0.0))
                        for mid in region.member_ids)
                for st in state_subtasks
            }
            for st in state_subtasks:
                a0 = region.prior_alpha_by_subtask.get(st, 2.0)
                b0 = region.prior_beta_by_subtask.get(st, 2.0)
                n = region.total_count_by_subtask.get(st, 0.0)
                succ = region.success_sum_by_subtask.get(st, 0.0)
                region.utility_by_subtask[st] = (succ + a0) / (n + a0 + b0)

    @staticmethod
    def _copy_source_ledger(ledger: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        return {mid: {st: float(v) for st, v in by_st.items()} for mid, by_st in ledger.items()}

    @staticmethod
    def _merge_source_ledgers(
        left: Dict[str, Dict[str, float]], right: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        merged = RegionManager._copy_source_ledger(left)
        for mid, by_st in right.items():
            dst = merged.setdefault(mid, {})
            for st, value in by_st.items():
                dst[st] = dst.get(st, 0.0) + float(value)
        return merged

    def _attach_source_ledgers_to_regions(self) -> None:
        """Attach current ID-keyed source ledgers to region objects during a topology edit."""
        for region in self.regions:
            region._source_success_ledger = self._copy_source_ledger(
                self.region_source_success_by_region.get(region.region_id, {})
            )
            region._source_total_ledger = self._copy_source_ledger(
                self.region_source_total_by_region.get(region.region_id, {})
            )

    def _set_region_source_ledger(
        self, region: Region, success: Dict[str, Dict[str, float]], total: Dict[str, Dict[str, float]]
    ) -> None:
        region._source_success_ledger = self._copy_source_ledger(success)
        region._source_total_ledger = self._copy_source_ledger(total)

    def _sync_region_source_ledgers_from_regions(self) -> None:
        """Re-key temporary object-attached ledgers after region IDs are rebuilt."""
        if not self._has_complete_region_source_evidence_ledger:
            return
        self.region_source_success_by_region = {
            int(region.region_id): self._copy_source_ledger(getattr(region, "_source_success_ledger", {}))
            for region in self.regions
        }
        self.region_source_total_by_region = {
            int(region.region_id): self._copy_source_ledger(getattr(region, "_source_total_ledger", {}))
            for region in self.regions
        }

    def _refresh_source_evidence_from_ledgers(self, subtasks: List[str]) -> None:
        """Recompute Beta evidence from the saved source slices, preserving soft ownership."""
        if not self._has_complete_region_source_evidence_ledger:
            return
        for region in self.regions:
            success = self.region_source_success_by_region.get(region.region_id, {})
            total = self.region_source_total_by_region.get(region.region_id, {})
            state_subtasks = set(subtasks)
            state_subtasks.update(region.prior_alpha_by_subtask)
            state_subtasks.update(region.prior_beta_by_subtask)
            for by_st in success.values():
                state_subtasks.update(by_st)
            for by_st in total.values():
                state_subtasks.update(by_st)
            region.success_sum_by_subtask = {
                st: sum(float(by_st.get(st, 0.0)) for by_st in success.values())
                for st in state_subtasks
            }
            region.total_count_by_subtask = {
                st: sum(float(by_st.get(st, 0.0)) for by_st in total.values())
                for st in state_subtasks
            }
            for st in state_subtasks:
                a0 = region.prior_alpha_by_subtask.get(st, 2.0)
                b0 = region.prior_beta_by_subtask.get(st, 2.0)
                s = region.success_sum_by_subtask.get(st, 0.0)
                n = region.total_count_by_subtask.get(st, 0.0)
                denom = n + a0 + b0
                if denom > 1e-12:
                    region.utility_by_subtask[st] = (s + a0) / denom
                else:
                    # Legacy checkpoints can contain explicit zero/zero priors for
                    # unobserved region-subtask pairs. With no routed evidence there
                    # is no posterior update to make; preserve the prior utility if
                    # present instead of dividing by zero.
                    region.utility_by_subtask[st] = region.utility_by_subtask.get(st, 0.5)

    def _split_child_routing_weights(self, mem_id: str, child_centroids: List[np.ndarray]) -> List[float]:
        """Sibling-local soft routing weights for one source's parent-owned evidence."""
        if not child_centroids:
            return []
        q_dict = self.subtask_q.get(mem_id, {})
        dists: List[float] = []
        for centroid in child_centroids:
            sq_sum = 0.0
            shared = 0
            for j, st in enumerate(self._known_subtasks):
                if st in q_dict and j < len(centroid):
                    value = float(q_dict[st])
                    if np.isfinite(value) and np.isfinite(centroid[j]):
                        sq_sum += (value - float(centroid[j])) ** 2
                        shared += 1
            dists.append(float(np.sqrt(sq_sum / shared)) if shared else 1.0)
        arr = np.asarray(dists, dtype=float)
        if not np.all(np.isfinite(arr)):
            best = int(np.nanargmin(np.where(np.isfinite(arr), arr, np.inf))) if np.isfinite(arr).any() else 0
            return [1.0 if i == best else 0.0 for i in range(len(child_centroids))]
        median_d = float(np.median(arr)) if len(arr) > 1 else 1.0
        auto_tau = max(median_d * 0.5, 0.01)
        tau = min(self.temperature, auto_tau) if self.temperature > 0 else auto_tau
        logits = -arr / tau
        logits -= logits.max()
        weights = np.exp(logits)
        denom = float(weights.sum())
        if not np.isfinite(denom) or denom <= 0:
            best = int(np.argmin(arr))
            return [1.0 if i == best else 0.0 for i in range(len(child_centroids))]
        return (weights / denom).tolist()

    def _split_posterior_states(
        self, parent: Region, child_members: List[List[str]], child_centroids: List[np.ndarray], subtasks: List[str]
    ) -> List[Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]]:
        """Build all child states together so soft source evidence is exactly conserved."""
        state_subtasks = set(subtasks)
        state_subtasks.update(parent.prior_alpha_by_subtask)
        state_subtasks.update(parent.prior_beta_by_subtask)
        prior_alpha = {st: float(parent.prior_alpha_by_subtask[st]) for st in state_subtasks if st in parent.prior_alpha_by_subtask}
        prior_beta = {st: float(parent.prior_beta_by_subtask[st]) for st in state_subtasks if st in parent.prior_beta_by_subtask}
        n_children = len(child_members)
        child_success = [{} for _ in range(n_children)]
        child_total = [{} for _ in range(n_children)]

        source_success = getattr(parent, "_source_success_ledger", {})
        source_total = getattr(parent, "_source_total_ledger", {})
        if self._has_complete_region_source_evidence_ledger:
            source_ids = set(source_success) | set(source_total)
            for mem_id in source_ids:
                routing = self._split_child_routing_weights(mem_id, child_centroids)
                for st in set(source_success.get(mem_id, {})) | set(source_total.get(mem_id, {})):
                    s = float(source_success.get(mem_id, {}).get(st, 0.0))
                    n = float(source_total.get(mem_id, {}).get(st, 0.0))
                    state_subtasks.add(st)
                    for idx, rho in enumerate(routing):
                        if s:
                            child_success[idx].setdefault(mem_id, {})[st] = rho * s
                        if n:
                            child_total[idx].setdefault(mem_id, {})[st] = rho * n
        elif self.region_split_evidence_migration_mode == "soft_source_conserving":
            logger.warning(
                "Split region %d without source-attributed soft evidence ledger; children inherit prior only.",
                parent.region_id,
            )

        result = []
        for idx, members in enumerate(child_members):
            if self.region_split_evidence_migration_mode == "legacy_qcount_reconstruction":
                # Historical Leaf control: reconstruct child Beta evidence from
                # per-memory EMA Q and q_count. This intentionally reproduces
                # the old pseudo-evidence semantics and must never be used as
                # the default/correct implementation.
                legacy_subtasks = set(state_subtasks)
                for mid in members:
                    legacy_subtasks.update(self.subtask_q.get(mid, {}))
                    legacy_subtasks.update(self.subtask_q_counts.get(mid, {}))
                success_sum, total_count = {}, {}
                for st in legacy_subtasks:
                    success = 0.0
                    total = 0.0
                    for mid in members:
                        count = float(self.subtask_q_counts.get(mid, {}).get(st, 0.0))
                        if count > 0:
                            success += float(self.subtask_q.get(mid, {}).get(st, 0.5)) * count
                            total += count
                    if total > 0:
                        success_sum[st] = success
                        total_count[st] = total
                # Legacy child construction did not inherit explicit warm priors.
                legacy_source_s, legacy_source_n = {}, {}
                result.append((success_sum, total_count, {}, {}, legacy_source_s, legacy_source_n))
                continue
            if self.region_split_evidence_migration_mode == "hard_member_rebase":
                if self._has_complete_memory_evidence_ledger:
                    hard_subtasks = set(state_subtasks)
                    for mid in members:
                        hard_subtasks.update(self.memory_success_sum_by_subtask.get(mid, {}))
                        hard_subtasks.update(self.memory_total_count_by_subtask.get(mid, {}))
                    success_sum = {st: sum(float(self.memory_success_sum_by_subtask.get(mid, {}).get(st, 0.0)) for mid in members) for st in hard_subtasks}
                    total_count = {st: sum(float(self.memory_total_count_by_subtask.get(mid, {}).get(st, 0.0)) for mid in members) for st in hard_subtasks}
                else:
                    logger.warning("Split region %d without direct member evidence ledger; hard-rebase child inherits prior only.", parent.region_id)
                    success_sum, total_count = {}, {}
            else:
                success_sum = {st: sum(float(by_st.get(st, 0.0)) for by_st in child_success[idx].values()) for st in state_subtasks}
                total_count = {st: sum(float(by_st.get(st, 0.0)) for by_st in child_total[idx].values()) for st in state_subtasks}
            result.append((success_sum, total_count, dict(prior_alpha), dict(prior_beta), child_success[idx], child_total[idx]))
        return result

    def reset_legacy_observed_evidence(self, reason: str = "legacy checkpoint migration") -> None:
        """Discard un-attributable aggregate evidence while retaining geometry/prior.

        Old checkpoints contain region-level observed evidence but lack the
        source-attributed soft ledger required to route it exactly through later
        topology changes.  This method deliberately does *not* alter Q values,
        centroids, memberships, member lists, or warm-start priors.  It starts a
        clean observed-evidence window whose future updates are source-conserving.
        """
        if not self._is_clustered:
            logger.warning("Legacy evidence reset requested before clustering; no region state changed")
            return
        for region in self.regions:
            state_subtasks = set(self._known_subtasks)
            state_subtasks.update(region.prior_alpha_by_subtask)
            state_subtasks.update(region.prior_beta_by_subtask)
            region.success_sum_by_subtask = {st: 0.0 for st in state_subtasks}
            region.total_count_by_subtask = {st: 0.0 for st in state_subtasks}
            for st in state_subtasks:
                a0 = region.prior_alpha_by_subtask.get(st, 2.0)
                b0 = region.prior_beta_by_subtask.get(st, 2.0)
                region.utility_by_subtask[st] = a0 / (a0 + b0) if (a0 + b0) > 0 else 0.5
        self.region_source_success_by_region = {}
        self.region_source_total_by_region = {}
        # New updates now build a complete source ledger from a known zero base.
        self._has_complete_region_source_evidence_ledger = True
        logger.warning(
            "Reset legacy aggregate region observed evidence (%s): retained %d regions, "
            "their geometry/membership/Q/prior, and started a clean source-conserving window.",
            reason, len(self.regions),
        )

    def maybe_split_merge(self) -> bool:
        """Utility-driven split/merge.

        Split: region's internal utility variance is high → members have
               divergent utility patterns, should be separated.
        Merge: two regions' utility_by_subtask are nearly identical →
               they represent the same utility pattern, should combine.
        Dominant-region split: if max_region_share > 0 and a region holds
               more than that fraction of total memories, force split it.

        Utility stats (Beta success_sum, total_count) are preserved/migrated.
        Returns True if any change was made.
        """
        if not self._is_clustered or len(self.regions) < 1:
            return False
        if not self.region_topology_updates_enabled:
            logger.info("Region split/merge skipped: topology updates are frozen for this branch")
            return False

        changed = False

        subtasks = self._known_subtasks if self._known_subtasks else []
        if not subtasks:
            return False

        # Region IDs are re-numbered during split/merge. Keep source ledgers on
        # objects for the edit, then re-key them after the final topology exists.
        self._attach_source_ledgers_to_regions()

        # Split callers construct all sibling states together below.  This is
        # essential for soft_source_conserving: a source slice is routed over
        # sibling children as one normalized distribution, never reconstructed
        # independently from each child.

        # --- Dominant-region forced split (before variance-based split) ---
        if self.max_region_share > 0:
            total_members = sum(len(r.member_ids) for r in self.regions)
            if total_members > 0:
                for region in self.regions:
                    share = len(region.member_ids) / total_members
                    if share > self.max_region_share and len(region.member_ids) >= self.min_cluster_size * 4:
                        members = region.member_ids
                        X = np.full((len(members), len(subtasks)), np.nan)
                        for i, mid in enumerate(members):
                            q_dict = self.subtask_q.get(mid, {})
                            for j, st in enumerate(subtasks):
                                if st in q_dict:
                                    X[i, j] = q_dict[st]
                        X_imputed = X.copy()
                        for j in range(len(subtasks)):
                            col = X_imputed[:, j]
                            observed = col[~np.isnan(col)]
                            fill_val = float(np.mean(observed)) if len(observed) > 0 else 0.5
                            col[np.isnan(col)] = fill_val
                        from sklearn.cluster import KMeans
                        km = KMeans(n_clusters=2, random_state=42, n_init=5)
                        labels = km.fit_predict(X_imputed)

                        min_child = max(3, self.min_cluster_size // 2)
                        child_sizes = [int((labels == k).sum()) for k in range(2)]
                        if min(child_sizes) < min_child:
                            continue

                        child_specs = []
                        for k in range(2):
                            mask = labels == k
                            sub_members = [members[i] for i in range(len(members)) if mask[i]]
                            sub_X = X[mask]
                            centroid = np.zeros(len(subtasks))
                            for j_c in range(len(subtasks)):
                                col_c = sub_X[:, j_c]
                                obs_c = col_c[~np.isnan(col_c)]
                                centroid[j_c] = float(np.mean(obs_c)) if len(obs_c) > 0 else 0.5
                            child_specs.append((sub_members, centroid))
                        split_states = self._split_posterior_states(
                            region, [spec[0] for spec in child_specs], [spec[1] for spec in child_specs], subtasks
                        )
                        new_regions = [r for r in self.regions if r.region_id != region.region_id]
                        for (sub_members, centroid), state in zip(child_specs, split_states):
                            util_st, cnt_st = {}, {}
                            for st in subtasks:
                                vals = [self.subtask_q[m].get(st) for m in sub_members if st in self.subtask_q.get(m, {})]
                                if vals:
                                    util_st[st] = float(np.mean(vals))
                                    cnt_st[st] = len(vals)
                            ssum_st, tcount_st, prior_a_st, prior_b_st, source_s, source_n = state
                            child = Region(
                                region_id=len(new_regions), centroid=centroid, member_ids=sub_members,
                                utility_by_subtask=util_st, counts_by_subtask=cnt_st,
                                success_sum_by_subtask=ssum_st, total_count_by_subtask=tcount_st,
                                prior_alpha_by_subtask=prior_a_st, prior_beta_by_subtask=prior_b_st,
                            )
                            self._set_region_source_ledger(child, source_s, source_n)
                            new_regions.append(child)
                        logger.info(
                            "Dominant-region split: region %d (%d members, %.1f%% share) → 2 sub-regions",
                            region.region_id, len(members), share * 100,
                        )
                        self.regions = new_regions
                        for idx, r in enumerate(self.regions):
                            r.region_id = idx
                        self.membership_weights = {}
                        self._recompute_all_memberships()
                        self._sync_region_source_ledgers_from_regions()
                        if self.region_split_evidence_migration_mode == "soft_source_conserving":
                            self._refresh_source_evidence_from_ledgers(subtasks)
                        elif self.region_split_evidence_migration_mode == "hard_member_rebase":
                            self._refresh_direct_evidence_from_members(subtasks)
                        else:
                            logger.warning("LEGACY CONTROL: preserving Q×q_count reconstructed child evidence")
                        self._build_region_failure_summaries()
                        self._build_region_success_summaries()
                        self._cluster_version = getattr(self, '_cluster_version', 0) + 1
                        return True  # skip variance split/merge this cycle

        # Compute global utility range for relative thresholds
        all_utils = []
        for region in self.regions:
            all_utils.extend(region.utility_by_subtask.values())
        if len(all_utils) < 2:
            return False
        util_range = max(all_utils) - min(all_utils)
        if util_range < 0.01:
            return False  # utilities haven't differentiated yet

        # Relative thresholds: fraction of current utility range. The default
        # preserves the historical 0.15 behavior; Leaf-v2 can raise it through
        # an explicit CLI/config parameter rather than an implicit environment
        # variable.
        split_range_fraction = self.region_split_range_fraction
        split_var_threshold = (util_range * split_range_fraction) ** 2
        merge_util_threshold = util_range * 0.10  # merge if utility diff < 10% of range
        logger.info(
            "Region topology thresholds: utility_range=%.6f, split_fraction=%.4f, "
            "split_var_threshold=%.8f, merge_util_threshold=%.6f",
            util_range, split_range_fraction, split_var_threshold, merge_util_threshold,
        )

        # Optional DB-style progressive split: score every eligible parent
        # first, then allow only the strongest normalized variance candidate.
        # This removes region-list ordering from a split-cap=1 policy.
        progressive_best_region_id = None
        progressive_candidates = []
        if self.region_progressive_best_split:
            for candidate_region in self.regions:
                if len(candidate_region.member_ids) < self.min_cluster_size * 2:
                    continue
                candidate_X = np.full((len(candidate_region.member_ids), len(subtasks)), np.nan)
                for i_c, mid_c in enumerate(candidate_region.member_ids):
                    q_dict_c = self.subtask_q.get(mid_c, {})
                    for j_c, st_c in enumerate(subtasks):
                        if st_c in q_dict_c:
                            candidate_X[i_c, j_c] = q_dict_c[st_c]
                candidate_var = float(np.nanvar(candidate_X, axis=0).mean())
                candidate_evidence = float(sum(candidate_region.total_count_by_subtask.values()))
                if candidate_var <= split_var_threshold:
                    continue
                if candidate_evidence < self.region_split_min_effective_evidence:
                    continue
                normalized_score = candidate_var / max(split_var_threshold, 1e-12)
                progressive_candidates.append(
                    (normalized_score, candidate_var, candidate_evidence, int(candidate_region.region_id))
                )
            if progressive_candidates:
                progressive_candidates.sort(reverse=True)
                progressive_best_region_id = progressive_candidates[0][3]
                logger.info(
                    "Progressive split selected region %d: score=%.4f intra_var=%.6f evidence=%.3f candidates=%d",
                    progressive_best_region_id, progressive_candidates[0][0],
                    progressive_candidates[0][1], progressive_candidates[0][2],
                    len(progressive_candidates),
                )
            else:
                logger.info("Progressive split found no eligible parent region")

        # --- Split regions with high internal utility variance ---
        new_regions = []
        protected_new_children = set()
        variance_splits = 0
        for region in self.regions:
            if len(region.member_ids) < self.min_cluster_size * 2:
                # Too small to split meaningfully
                region.region_id = len(new_regions)
                new_regions.append(region)
                continue

            # Compute intra-region utility variance (NaN-aware)
            members = region.member_ids
            X = np.full((len(members), len(subtasks)), np.nan)
            for i, mid in enumerate(members):
                q_dict = self.subtask_q.get(mid, {})
                for j, st in enumerate(subtasks):
                    if st in q_dict:
                        X[i, j] = q_dict[st]

            intra_var = float(np.nanvar(X, axis=0).mean())

            if intra_var > split_var_threshold:
                if (
                    self.region_progressive_best_split
                    and int(region.region_id) != progressive_best_region_id
                ):
                    region.region_id = len(new_regions)
                    new_regions.append(region)
                    continue
                effective_evidence = float(sum(region.total_count_by_subtask.values()))
                if (
                    self.region_max_variance_splits_per_epoch > 0
                    and variance_splits >= self.region_max_variance_splits_per_epoch
                ):
                    logger.info(
                        "Variance split skipped for region %d: split cap %d already reached",
                        region.region_id, self.region_max_variance_splits_per_epoch,
                    )
                    region.region_id = len(new_regions)
                    new_regions.append(region)
                    continue
                if effective_evidence < self.region_split_min_effective_evidence:
                    logger.info(
                        "Variance split skipped for region %d: effective evidence %.3f < gate %.3f",
                        region.region_id, effective_evidence, self.region_split_min_effective_evidence,
                    )
                    region.region_id = len(new_regions)
                    new_regions.append(region)
                    continue
                # Split via K-means(k=2). Impute NaN with per-subtask mean
                # within this region (not 0.0) to avoid treating unobserved as bad.
                X_imputed = X.copy()
                for j in range(len(subtasks)):
                    col = X_imputed[:, j]
                    observed = col[~np.isnan(col)]
                    fill_val = float(np.mean(observed)) if len(observed) > 0 else 0.5
                    col[np.isnan(col)] = fill_val
                from sklearn.cluster import KMeans
                km = KMeans(n_clusters=2, random_state=42, n_init=5)
                labels = km.fit_predict(X_imputed)

                child_specs = []
                for k in range(2):
                    mask = labels == k
                    sub_members = [members[i] for i in range(len(members)) if mask[i]]
                    if not sub_members:
                        continue
                    sub_X = X[mask]
                    child_centroid = np.zeros(len(subtasks))
                    for j_c in range(len(subtasks)):
                        col_c = sub_X[:, j_c]
                        obs_c = col_c[~np.isnan(col_c)]
                        child_centroid[j_c] = float(np.mean(obs_c)) if len(obs_c) > 0 else 0.0
                    child_specs.append((mask, sub_members, child_centroid))
                child_sizes = [len(spec[1]) for spec in child_specs]
                if len(child_specs) != 2 or min(child_sizes) < self.region_split_min_child_size:
                    logger.info(
                        "Variance split rejected for region %d: child sizes=%s min_child=%d",
                        region.region_id, child_sizes, self.region_split_min_child_size,
                    )
                    region.region_id = len(new_regions)
                    new_regions.append(region)
                    continue
                split_states = self._split_posterior_states(
                    region, [spec[1] for spec in child_specs], [spec[2] for spec in child_specs], subtasks
                )

                for (mask, sub_members, precomputed_centroid), split_state in zip(child_specs, split_states):
                    new_rid = len(new_regions)
                    # Observed-only centroid (consistent with cluster_by_utility)
                    centroid = precomputed_centroid

                    # Geometry utilities are recomputed from child members, but
                    # posterior evidence is inherited from the parent state rather
                    # than rebuilt from historical member q/count products.
                    util_st, cnt_st = {}, {}
                    for st in subtasks:
                        vals = [self.subtask_q[m].get(st) for m in sub_members
                                if st in self.subtask_q.get(m, {})]
                        if vals:
                            util_st[st] = float(np.mean(vals))
                            cnt_st[st] = len(vals)
                    ssum_st, tcount_st, prior_a_st, prior_b_st, source_s, source_n = split_state

                    child = Region(
                        region_id=new_rid, centroid=centroid, member_ids=sub_members,
                        utility_by_subtask=util_st, counts_by_subtask=cnt_st,
                        success_sum_by_subtask=ssum_st, total_count_by_subtask=tcount_st,
                        prior_alpha_by_subtask=prior_a_st, prior_beta_by_subtask=prior_b_st,
                    )
                    self._set_region_source_ledger(child, source_s, source_n)
                    new_regions.append(child)
                    if self.region_protect_new_split_children:
                        protected_new_children.add(id(child))

                logger.info(
                    "Split region %d (%d members, intra_var=%.4f) into 2 sub-regions",
                    region.region_id, len(members), intra_var,
                )
                changed = True
                variance_splits += 1
            else:
                region.region_id = len(new_regions)
                new_regions.append(region)

        self.regions = new_regions

        # --- Merge regions with similar utility patterns ---
        if len(self.regions) >= 2:
            merged = set()
            final_regions = []
            merges_done = 0
            for i in range(len(self.regions)):
                if i in merged:
                    continue
                ri = self.regions[i]
                best_merge = -1
                best_diff = float('inf')
                merge_cap_reached = (
                    self.region_max_merges_per_epoch > 0
                    and merges_done >= self.region_max_merges_per_epoch
                )

                for j in range(i + 1, len(self.regions)):
                    if merge_cap_reached:
                        break
                    if j in merged:
                        continue
                    rj = self.regions[j]
                    if (
                        self.region_protect_new_split_children
                        and (id(ri) in protected_new_children or id(rj) in protected_new_children)
                    ):
                        continue

                    # Mean absolute difference in utility_by_subtask
                    diffs = []
                    for st in subtasks:
                        ui = ri.utility_by_subtask.get(st, 0.5)
                        uj = rj.utility_by_subtask.get(st, 0.5)
                        diffs.append(abs(ui - uj))
                    mean_diff = float(np.mean(diffs)) if diffs else 1.0

                    if mean_diff < merge_util_threshold and mean_diff < best_diff:
                        best_diff = mean_diff
                        best_merge = j

                if best_merge >= 0:
                    rj = self.regions[best_merge]
                    ri.member_ids.extend(rj.member_ids)
                    # Source soft evidence is additive under merge: both slices
                    # retain their originating memory IDs, so a later split can
                    # reroute them again without posterior drift.
                    merged_source_success = self._merge_source_ledgers(
                        getattr(ri, "_source_success_ledger", {}), getattr(rj, "_source_success_ledger", {})
                    )
                    merged_source_total = self._merge_source_ledgers(
                        getattr(ri, "_source_total_ledger", {}), getattr(rj, "_source_total_ledger", {})
                    )
                    self._set_region_source_ledger(ri, merged_source_success, merged_source_total)
                    # Recompute centroid from merged members (observed-only means)
                    X_merge = np.full((len(ri.member_ids), len(subtasks)), np.nan)
                    for idx_m, mid in enumerate(ri.member_ids):
                        q_dict = self.subtask_q.get(mid, {})
                        for k, st in enumerate(subtasks):
                            if st in q_dict:
                                X_merge[idx_m, k] = q_dict[st]
                    new_centroid = np.zeros(len(subtasks))
                    for k in range(len(subtasks)):
                        col = X_merge[:, k]
                        observed = col[~np.isnan(col)]
                        new_centroid[k] = float(np.mean(observed)) if len(observed) > 0 else 0.0
                    ri.centroid = new_centroid

                    # Rebuild observed evidence from the direct per-memory ledger
                    # whenever it is available. This keeps split/merge cycles
                    # path-independent: evidence follows members rather than
                    # drifting through region-level aggregate operations.
                    # Legacy checkpoints without a ledger retain additive region
                    # evidence as the compatibility fallback.
                    evidence_subtasks = set(subtasks)
                    evidence_subtasks.update(ri.success_sum_by_subtask)
                    evidence_subtasks.update(ri.total_count_by_subtask)
                    evidence_subtasks.update(rj.success_sum_by_subtask)
                    evidence_subtasks.update(rj.total_count_by_subtask)
                    if self.region_split_evidence_migration_mode == "soft_source_conserving" and self._has_complete_region_source_evidence_ledger:
                        for by_st in getattr(ri, "_source_success_ledger", {}).values():
                            evidence_subtasks.update(by_st)
                        for by_st in getattr(ri, "_source_total_ledger", {}).values():
                            evidence_subtasks.update(by_st)
                        ri.success_sum_by_subtask = {
                            st: sum(float(by_st.get(st, 0.0)) for by_st in getattr(ri, "_source_success_ledger", {}).values())
                            for st in evidence_subtasks
                        }
                        ri.total_count_by_subtask = {
                            st: sum(float(by_st.get(st, 0.0)) for by_st in getattr(ri, "_source_total_ledger", {}).values())
                            for st in evidence_subtasks
                        }
                    elif self._has_complete_memory_evidence_ledger:
                        for mid in ri.member_ids:
                            evidence_subtasks.update(self.memory_success_sum_by_subtask.get(mid, {}))
                            evidence_subtasks.update(self.memory_total_count_by_subtask.get(mid, {}))
                        ri.success_sum_by_subtask = {
                            st: sum(float(self.memory_success_sum_by_subtask.get(mid, {}).get(st, 0.0))
                                    for mid in ri.member_ids)
                            for st in evidence_subtasks
                        }
                        ri.total_count_by_subtask = {
                            st: sum(float(self.memory_total_count_by_subtask.get(mid, {}).get(st, 0.0))
                                    for mid in ri.member_ids)
                            for st in evidence_subtasks
                        }
                    else:
                        ri.success_sum_by_subtask = {
                            st: (float(ri.success_sum_by_subtask.get(st, 0.0)) +
                                 float(rj.success_sum_by_subtask.get(st, 0.0)))
                            for st in evidence_subtasks
                        }
                        ri.total_count_by_subtask = {
                            st: (float(ri.total_count_by_subtask.get(st, 0.0)) +
                                 float(rj.total_count_by_subtask.get(st, 0.0)))
                            for st in evidence_subtasks
                        }

                    # Cap merged prior ESS so repeated merge/split cycles can't inflate
                    # the prior without bound. PRIOR_ESS_CAP=20 ≈ 4× single-region prior.
                    PRIOR_ESS_CAP = 20.0
                    for st in subtasks:
                        # Merge warm-start priors additively, then cap their ESS so
                        # repeated merges don't anchor utility to stale history.
                        merged_a = (ri.prior_alpha_by_subtask.get(st, 0.0) +
                                    rj.prior_alpha_by_subtask.get(st, 0.0))
                        merged_b = (ri.prior_beta_by_subtask.get(st, 0.0) +
                                    rj.prior_beta_by_subtask.get(st, 0.0))
                        merged_ess = merged_a + merged_b
                        if merged_ess > PRIOR_ESS_CAP:
                            scale = PRIOR_ESS_CAP / merged_ess
                            merged_a *= scale
                            merged_b *= scale
                        ri.prior_alpha_by_subtask[st] = merged_a
                        ri.prior_beta_by_subtask[st] = merged_b
                        a0 = merged_a if merged_a > 0 else 2.0
                        b0 = merged_b if merged_b > 0 else 2.0
                        n = ri.total_count_by_subtask[st]
                        s = ri.success_sum_by_subtask[st]
                        if (n + a0 + b0) > 0:
                            ri.utility_by_subtask[st] = (s + a0) / (n + a0 + b0)
                        ri.counts_by_subtask[st] = (
                            ri.counts_by_subtask.get(st, 0) +
                            rj.counts_by_subtask.get(st, 0)
                        )
                    merged.add(best_merge)
                    merges_done += 1
                    logger.info(
                        "Merged regions %d+%d (utility_diff=%.4f)",
                        i, best_merge, best_diff,
                    )
                    changed = True

                ri.region_id = len(final_regions)
                final_regions.append(ri)

            self.regions = final_regions

        if changed:
            self._recompute_all_memberships()
            self._sync_region_source_ledgers_from_regions()
            if self.region_split_evidence_migration_mode == "soft_source_conserving":
                self._refresh_source_evidence_from_ledgers(subtasks)
            elif self.region_split_evidence_migration_mode == "hard_member_rebase":
                # Hard memberships may change after soft weights are recomputed.
                # In hard-rebase mode evidence follows final argmax members.
                self._refresh_direct_evidence_from_members(subtasks)
            else:
                logger.warning("LEGACY CONTROL: preserving historical aggregate evidence after topology edit")
            # Rebuild summaries after split/merge changes region membership
            self._build_region_failure_summaries()
            self._build_region_success_summaries()

        if changed:
            self._cluster_version = getattr(self, '_cluster_version', 0) + 1
        return changed

    def retemper_memberships_and_reroute_source_evidence(self, temperature: float) -> None:
        """Sharpen existing geometry and exactly reroute source-attributed evidence.

        Region centroids/topology and per-memory Q are unchanged. Soft memberships
        are recomputed at ``temperature``. The complete source ledger is first
        collapsed across old regions, then redistributed over the new top-3
        memberships using the configured evidence-sharpen exponent. This preserves
        every source/subtask's global success and total evidence exactly.
        """
        if not self._is_clustered or not self.regions:
            raise RuntimeError("cannot retemper an unclustered RegionManager")
        if not self._has_complete_region_source_evidence_ledger:
            raise RuntimeError("complete source evidence ledger required for exact reroute")
        new_temperature = float(temperature)
        if not np.isfinite(new_temperature) or new_temperature <= 0:
            raise ValueError(f"invalid region temperature: {temperature!r}")

        # Collapse old region ownership into source-attributed global evidence.
        global_success: Dict[str, Dict[str, float]] = {}
        global_total: Dict[str, Dict[str, float]] = {}
        for ledger, target in (
            (self.region_source_success_by_region, global_success),
            (self.region_source_total_by_region, global_total),
        ):
            for by_source in ledger.values():
                for mem_id, by_subtask in by_source.items():
                    dst = target.setdefault(mem_id, {})
                    for subtask, value in by_subtask.items():
                        dst[subtask] = dst.get(subtask, 0.0) + float(value)

        before_success = sum(sum(v.values()) for v in global_success.values())
        before_total = sum(sum(v.values()) for v in global_total.values())
        self.temperature = new_temperature
        self._recompute_all_memberships()

        self.region_source_success_by_region = {int(r.region_id): {} for r in self.regions}
        self.region_source_total_by_region = {int(r.region_id): {} for r in self.regions}
        all_sources = set(global_success) | set(global_total)
        for mem_id in all_sources:
            weights = self.membership_weights.get(mem_id)
            if weights is None:
                continue
            values = np.asarray(weights, dtype=float).reshape(-1)
            n_regions = min(len(values), len(self.regions))
            if n_regions <= 0 or not np.isfinite(values[:n_regions]).any():
                continue
            top_k = min(3, n_regions)
            safe = np.where(np.isfinite(values[:n_regions]), values[:n_regions], 0.0)
            top_rids = np.argsort(-safe)[:top_k]
            sharp = np.power(np.maximum(safe[top_rids], 0.0), self.region_evidence_sharpen_alpha)
            denom = float(sharp.sum())
            routed = (sharp / denom) if denom > 0 else np.full(top_k, 1.0 / top_k)
            for rid, weight in zip(top_rids.tolist(), routed.tolist()):
                region_id = int(self.regions[rid].region_id)
                for source, target in (
                    (global_success.get(mem_id, {}), self.region_source_success_by_region),
                    (global_total.get(mem_id, {}), self.region_source_total_by_region),
                ):
                    if not source:
                        continue
                    dst = target.setdefault(region_id, {}).setdefault(mem_id, {})
                    for subtask, value in source.items():
                        dst[subtask] = dst.get(subtask, 0.0) + float(weight) * float(value)

        self._refresh_source_evidence_from_ledgers(list(self._known_subtasks))
        self._cluster_version = getattr(self, '_cluster_version', 0) + 1
        self._build_region_failure_summaries()
        self._build_region_success_summaries()
        after_success = sum(
            float(v) for by_source in self.region_source_success_by_region.values()
            for by_subtask in by_source.values() for v in by_subtask.values()
        )
        after_total = sum(
            float(v) for by_source in self.region_source_total_by_region.values()
            for by_subtask in by_source.values() for v in by_subtask.values()
        )
        if not np.isclose(before_success, after_success, rtol=1e-10, atol=1e-8):
            raise RuntimeError(f"success evidence not conserved: {before_success} -> {after_success}")
        if not np.isclose(before_total, after_total, rtol=1e-10, atol=1e-8):
            raise RuntimeError(f"total evidence not conserved: {before_total} -> {after_total}")
        logger.info(
            "Retempered Region memberships: temperature=%.4f, memories=%d, regions=%d, "
            "source evidence conserved (success=%.6f, total=%.6f)",
            self.temperature, len(self.membership_weights), len(self.regions),
            after_success, after_total,
        )

    def rebuild_hard_memberships_from_weights(self) -> None:
        """Rebuild canonical hard assignments from saved soft memberships.

        ``member_ids`` is intentionally a single-region (argmax) assignment used
        by split/merge and region summaries. Soft overlap belongs exclusively in
        ``membership_weights``. This also repairs older checkpoints whose member
        lists accumulated stale assignments across split/merge events.
        """
        for region in self.regions:
            region.member_ids = []

        assigned = 0
        skipped = 0
        for mem_id, weights in self.membership_weights.items():
            values = np.asarray(weights, dtype=float).reshape(-1)
            n = min(len(values), len(self.regions))
            if n == 0 or not np.isfinite(values[:n]).any():
                skipped += 1
                continue
            safe_values = np.where(np.isfinite(values[:n]), values[:n], -np.inf)
            region_id = int(np.argmax(safe_values))
            self.regions[region_id].member_ids.append(mem_id)
            assigned += 1

        logger.info(
            "Rebuilt hard region memberships from soft weights: %d assigned, "
            "%d skipped, %d member slots across %d regions",
            assigned, skipped,
            sum(len(region.member_ids) for region in self.regions),
            len(self.regions),
        )

    def _recompute_all_memberships(self) -> None:
        """Recompute soft weights and canonical hard assignments after split/merge."""
        if not self.regions:
            return

        # Membership weights are canonical when present, but a just-created
        # split can temporarily have no weights for its child members. Rebuild
        # from the union so no member disappears during a split/merge cycle.
        mem_ids = {
            mem_id for mem_id in self.membership_weights
            if mem_id in self.subtask_q
        }
        mem_ids.update(
            mem_id for region in self.regions for mem_id in region.member_ids
            if mem_id in self.subtask_q
        )
        # Clear stale hard assignments before assign_new_memory appends each
        # memory to exactly one nearest/argmax region.
        for region in self.regions:
            region.member_ids = []

        for mem_id in sorted(mem_ids):
            self.assign_new_memory(mem_id)

        total_slots = sum(len(region.member_ids) for region in self.regions)
        unique_members = {
            mem_id for region in self.regions for mem_id in region.member_ids
        }
        if total_slots != len(unique_members) or total_slots != len(mem_ids):
            logger.warning(
                "Hard region membership rebuild invariant failed: memories=%d, "
                "unique=%d, slots=%d",
                len(mem_ids), len(unique_members), total_slots,
            )
        else:
            logger.info(
                "Recomputed region memberships: %d memories, %d member slots",
                len(mem_ids), total_slots,
            )

    def _hdbscan_cluster(self, X: np.ndarray) -> np.ndarray:
        """HDBSCAN clustering. Noise points assigned to nearest cluster."""
        import hdbscan

        n_samples = len(X)
        min_cs = max(2, min(self.min_cluster_size, n_samples // 5))
        min_s = self.min_samples if self.min_samples > 0 else max(1, min_cs // 2)
        min_s = min(min_s, max(1, min_cs))

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cs,
            min_samples=min_s,
            metric="euclidean",
            cluster_selection_method=self.cluster_selection_method,
        )
        labels = clusterer.fit_predict(X)

        logger.debug("HDBSCAN euclidean: min_cs=%d, min_s=%d, method=%s, n_clusters=%d",
                      min_cs, min_s, self.cluster_selection_method,
                      labels.max() + 1 if labels.max() >= 0 else 0)

        n_clusters = labels.max() + 1 if labels.max() >= 0 else 0

        if n_clusters == 0:
            return np.zeros(n_samples, dtype=int)

        # Assign noise to nearest centroid
        centroids = np.array([
            X[labels == c].mean(axis=0) for c in range(n_clusters)
        ])
        noise_mask = labels == -1
        if noise_mask.any():
            from scipy.spatial.distance import cdist
            dists = cdist(X[noise_mask], centroids)
            labels[noise_mask] = dists.argmin(axis=1)

        return labels

    # ---------- Zero-Shot Transfer Support ----------

    # AdvancedTransferManager for zero-shot transfer (optional)
    advanced_transfer_mgr: Any = None

    def _estimate_region_utility_zero_shot(
        self,
        region: "Region",
        target_subtask: str,
    ) -> Tuple[float, int, str]:
        """
        Estimate region utility for an unseen target_subtask.

        Strategy (prioritized):
        1. Embedding-weighted residual transfer from observed subtasks
        2. Advanced transfer manager (if available)
        3. Fallback: generalization score (mean utility penalized by variance)

        Returns:
            (estimated_utility, pseudo_count, strategy)
        """
        known_utilities = region.utility_by_subtask
        if not known_utilities:
            return 0.5, 0, "no_data"

        # Strategy 1: Embedding-weighted residual transfer
        if self._subtask_embeddings and target_subtask in self._subtask_embeddings:
            target_emb = self._subtask_embeddings[target_subtask]
            target_norm = np.linalg.norm(target_emb)
            if target_norm > 1e-8:
                target_emb = target_emb / target_norm

                # Compute softmax weights from cosine similarity
                sims = {}
                for src_st in known_utilities:
                    src_emb = self._subtask_embeddings.get(src_st)
                    if src_emb is None:
                        continue
                    src_norm = np.linalg.norm(src_emb)
                    if src_norm < 1e-8:
                        continue
                    sims[src_st] = float(np.dot(target_emb, src_emb / src_norm))

                if sims:
                    # Softmax with temperature
                    tau = 5.0
                    max_sim = max(sims.values())
                    weights = {s: np.exp(tau * (sim - max_sim)) for s, sim in sims.items()}
                    total_w = sum(weights.values())
                    if np.isfinite(total_w) and total_w > 0:
                        weights = {s: w / total_w for s, w in weights.items()}

                        # Per-source-subtask global mean, count-weighted across regions.
                        all_region_utils = {}
                        for s in sims:
                            num, den = 0.0, 0.0
                            for r in self.regions:
                                if s in r.utility_by_subtask:
                                    n_rs = r.counts_by_subtask.get(s, 0)
                                    if n_rs > 0:
                                        num += n_rs * r.utility_by_subtask[s]
                                        den += n_rs
                            if den > 0:
                                all_region_utils[s] = num / den
                            else:
                                col = [r.utility_by_subtask[s] for r in self.regions if s in r.utility_by_subtask]
                                all_region_utils[s] = float(np.mean(col)) if col else 0.5

                        # Prior for holdout: count-weighted mean over all (region, subtask)
                        # cells. Must be independent of mu_s, otherwise the residual term
                        # sum_s alpha_s * (u(r,s) - mu_s) cancels with mu_target.
                        g_num, g_den = 0.0, 0.0
                        for r in self.regions:
                            for s_any, u_any in r.utility_by_subtask.items():
                                n_any = r.counts_by_subtask.get(s_any, 0)
                                if n_any > 0:
                                    g_num += n_any * u_any
                                    g_den += n_any
                        mu_target = g_num / g_den if g_den > 0 else 0.5

                        # hat_u(r, s*) = mu_target + sum_s alpha_s * (u(r,s) - mu_s)
                        estimated = mu_target + sum(
                            weights[s] * (known_utilities[s] - all_region_utils[s])
                            for s in weights
                        )
                        if not np.isfinite(estimated):
                            estimated = 0.5
                        estimated = float(max(0.05, min(0.95, estimated)))

                        max_weight = max(weights.values())
                        pseudo_count = max(1, int(max_weight * 5))

                        return estimated, pseudo_count, "embedding_residual"

        # Strategy 2: Advanced transfer with learned calibration
        if self.advanced_transfer_mgr is not None:
            known_subtasks = list(known_utilities.keys())

            transfer_weights = {}
            for src in known_subtasks:
                weight = self.advanced_transfer_mgr.get_calibrated_transfer_weight(
                    src, target_subtask
                )
                if weight > 0:
                    transfer_weights[src] = weight

            if transfer_weights:
                transfer_weights = self.advanced_transfer_mgr.apply_negative_transfer_guard(
                    transfer_weights, known_utilities
                )

                total = sum(transfer_weights.values())
                if total > 0:
                    transfer_weights = {k: v/total for k, v in transfer_weights.items()}

                estimated = sum(
                    w * known_utilities[src]
                    for src, w in transfer_weights.items()
                )

                max_weight = max(transfer_weights.values()) if transfer_weights else 0
                pseudo_count = int(max_weight * 5)

                return estimated, pseudo_count, "advanced_transfer"

        # Strategy 3: Generalization score (fallback)
        utils = list(known_utilities.values())
        mean_u = float(np.mean(utils))
        var_u = float(np.var(utils))

        generalization_score = mean_u * np.exp(-var_u * 5)
        generalization_score = max(0.1, min(0.9, generalization_score))

        return generalization_score, 1, "generalization_fallback"

    # ---------- Region Gating Score (Soft Membership) ----------

    def compute_region_gating_score(
        self,
        memory_id: str,
        target_subtask: str,
        eval_mode: str = "train",
        allowed_sources: Any = None,
    ) -> float:
        """
        Compute region gating score using soft membership.

        Beta mode: utility_by_subtask already contains Beta posterior mean
        (with a0=b0=2 prior), so we use it directly — no extra Bayesian
        smoothing layer to avoid redundant conservatism.

        EMA mode: utility_by_subtask is raw EMA, apply Bayesian smoothing
        with C to handle low-count subtasks.
        """
        if not self._is_clustered:
            return 1.0

        weights = self.membership_weights.get(memory_id)
        if weights is None:
            return self._get_global_mean()

        is_zero_shot = target_subtask not in self._known_subtasks
        use_bayesian_smoothing = (self.region_utility_mode != "beta")

        global_mean = self._get_global_mean() if use_bayesian_smoothing else 0.0
        C = self.bayesian_smoothing_C if use_bayesian_smoothing else 0.0

        score = 0.0
        for rid, w in enumerate(weights):
            if w < 0.001 or rid >= len(self.regions):
                continue
            region = self.regions[rid]

            if is_zero_shot and region.counts_by_subtask.get(target_subtask, 0) == 0:
                # zero-shot estimate is already an informed prior; skip EMA smoothing
                # (n=0 would otherwise collapse (n*u + C*g)/(n+C) to global_mean).
                utility, _, _ = self._estimate_region_utility_zero_shot(
                    region, target_subtask
                )
                region_score = utility
            else:
                utility = region.utility_by_subtask.get(target_subtask, 0.5)
                if use_bayesian_smoothing:
                    n = region.counts_by_subtask.get(target_subtask, 0)
                    region_score = (n * utility + C * global_mean) / (n + C)
                else:
                    region_score = utility

            score += w * region_score

        return float(max(0.01, min(1.0, score)))

    # ---------- Convenience ----------

    def cluster_from_memory_service(self, memory_service: Any):
        """Compatibility method. Just calls cluster_by_utility."""
        self.cluster_by_utility()

    def classify_transfer_patterns(self):
        """Log region utility patterns (informational)."""
        for region in self.regions:
            if not region.utility_by_subtask:
                continue
            high = [st for st, u in region.utility_by_subtask.items() if u > 0.6]
            low = [st for st, u in region.utility_by_subtask.items() if u < 0.3]
            if high:
                logger.debug(
                    "Region %d: high on %s, low on %s",
                    region.region_id, high[:3], low[:3],
                )

    def update_benchmark_utilities(self):
        """No-op for compatibility."""
        pass

    def top_regions_for_subtask(
        self,
        target_subtask: str,
        top_n: int = 3,
        min_count: int = 30,
    ) -> List["Region"]:
        """Return top-N regions ranked by utility for target_subtask.

        Used by v5 quota-based retrieval (see docs/ALFWORLD_REGION_IMPROVEMENT_PLAN.md §14).
        Filters out regions with too-sparse evidence (counts_by_subtask < min_count)
        to avoid noisy high-utility estimates from tiny samples.

        Args:
            target_subtask: subtask key (e.g. "alf/pick_and_place_simple")
            top_n: how many top regions to return
            min_count: require region.counts_by_subtask[target_subtask] >= min_count

        Returns:
            List of Region objects sorted by utility_by_subtask[target_subtask] desc.
            Empty if region_manager not clustered or no qualifying regions.
        """
        if not self._is_clustered or not self.regions:
            return []
        # Snapshot regions list to avoid concurrent modification during retrieval
        # (background clustering may rebuild self.regions). Copying refs is cheap.
        regions_snapshot = list(self.regions)
        scored: List[Tuple[float, "Region"]] = []
        for r in regions_snapshot:
            counts = getattr(r, "counts_by_subtask", {}) or {}
            n = counts.get(target_subtask, 0)
            if n < min_count:
                continue
            util_dict = getattr(r, "utility_by_subtask", {}) or {}
            util = util_dict.get(target_subtask, None)
            if util is None:
                continue
            scored.append((float(util), r))
        scored.sort(key=lambda x: -x[0])
        return [r for _, r in scored[:top_n]]

    def get_region_summary(self) -> Dict[str, Any]:
        """Generate summary for logging."""
        return {
            "n_regions": len(self.regions),
            "n_memories_tracked": len(self.subtask_q),
            "n_memories_clustered": len(self.membership_weights),
            "n_subtask_dimensions": len(self._known_subtasks),
            "known_subtasks": self._known_subtasks,
            "global_mean_reward": self._get_global_mean(),
            "global_reward_count": self._global_reward_count,
            "region_sizes": [len(r.member_ids) for r in self.regions],
        }

    # ---------- Persistence ----------

    def save(self, path: str):
        """Save region manager state."""
        state = {
            "task_hierarchy": self.task_hierarchy,
            "alpha": self.alpha,
            "min_cluster_size": self.min_cluster_size,
            "temperature": self.temperature,
            "shrinkage_top_n": self.shrinkage_top_n,
            "cluster_space": self.cluster_space,
            "shrinkage_min_utility_margin": self.shrinkage_min_utility_margin,
            "region_precluster_evidence_mode": self.region_precluster_evidence_mode,
            "region_precluster_evidence_scale": self.region_precluster_evidence_scale,
            "precluster_evidence_backfilled": self._precluster_evidence_backfilled,
            "region_utility_mode": self.region_utility_mode,
            "region_split_evidence_migration_mode": self.region_split_evidence_migration_mode,
            "region_topology_updates_enabled": self.region_topology_updates_enabled,
            "topology_last_edit_section": int(getattr(self, "topology_last_edit_section", 0)),
            "topology_mid_maintenance_done_steps": sorted(
                int(step) for step in getattr(self, "topology_mid_maintenance_done_steps", set())
            ),
            "region_evidence_sharpen_alpha": self.region_evidence_sharpen_alpha,
            "region_split_range_fraction": self.region_split_range_fraction,
            "region_max_variance_splits_per_epoch": self.region_max_variance_splits_per_epoch,
            "region_split_min_effective_evidence": self.region_split_min_effective_evidence,
            "region_progressive_best_split": self.region_progressive_best_split,
            "region_max_merges_per_epoch": self.region_max_merges_per_epoch,
            "region_split_min_child_size": self.region_split_min_child_size,
            "region_protect_new_split_children": self.region_protect_new_split_children,
            "bayesian_smoothing_C": self.bayesian_smoothing_C,
            "subtask_q": self.subtask_q,
            "subtask_q_counts": self.subtask_q_counts,
            "memory_success_sum_by_subtask": self.memory_success_sum_by_subtask,
            "memory_total_count_by_subtask": self.memory_total_count_by_subtask,
            "precluster_success_sum_by_subtask": self.precluster_success_sum_by_subtask,
            "precluster_total_count_by_subtask": self.precluster_total_count_by_subtask,
            "has_complete_memory_evidence_ledger": self._has_complete_memory_evidence_ledger,
            "region_source_success_by_region": self.region_source_success_by_region,
            "region_source_total_by_region": self.region_source_total_by_region,
            "has_complete_region_source_evidence_ledger": self._has_complete_region_source_evidence_ledger,
            "known_subtasks": self._known_subtasks,
            "is_clustered": self._is_clustered,
            "global_reward_sum": self._global_reward_sum,
            "global_reward_count": self._global_reward_count,
            "membership_weights": {
                k: v.tolist() for k, v in self.membership_weights.items()
            },
            "regions": [
                {
                    "region_id": r.region_id,
                    "centroid": r.centroid.tolist() if r.centroid is not None else None,
                    "member_ids": r.member_ids,
                    "utility_by_subtask": r.utility_by_subtask,
                    "counts_by_subtask": r.counts_by_subtask,
                    "success_sum_by_subtask": r.success_sum_by_subtask,
                    "total_count_by_subtask": r.total_count_by_subtask,
                    "prior_alpha_by_subtask": r.prior_alpha_by_subtask,
                    "prior_beta_by_subtask": r.prior_beta_by_subtask,
                }
                for r in self.regions
            ],
            "subtask_embeddings": {
                k: v.tolist() for k, v in self._subtask_embeddings.items()
            },
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        logger.info("RegionManager saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "RegionManager":
        """Load region manager state."""
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)

        mgr = cls(
            task_hierarchy=state["task_hierarchy"],
            alpha=state.get("alpha", 0.1),
            min_cluster_size=state.get("min_cluster_size", 5),
            temperature=state.get("temperature", 1.0),
            shrinkage_top_n=state.get("shrinkage_top_n", 3),
            cluster_space=state.get("cluster_space", "capability"),
            shrinkage_min_utility_margin=state.get("shrinkage_min_utility_margin", 0.0),
            region_precluster_evidence_mode=state.get("region_precluster_evidence_mode", "off"),
            region_precluster_evidence_scale=state.get("region_precluster_evidence_scale", 1.0),
            region_utility_mode=state.get("region_utility_mode", "ema"),
            region_split_evidence_migration_mode=state.get("region_split_evidence_migration_mode", "soft_source_conserving"),
            region_topology_updates_enabled=state.get("region_topology_updates_enabled", True),
            region_evidence_sharpen_alpha=state.get("region_evidence_sharpen_alpha", 2.0),
            region_split_range_fraction=state.get("region_split_range_fraction", 0.15),
            region_max_variance_splits_per_epoch=state.get("region_max_variance_splits_per_epoch", 0),
            region_split_min_effective_evidence=state.get("region_split_min_effective_evidence", 0.0),
            region_progressive_best_split=state.get("region_progressive_best_split", False),
            region_max_merges_per_epoch=state.get("region_max_merges_per_epoch", 0),
            region_split_min_child_size=state.get("region_split_min_child_size", 1),
            region_protect_new_split_children=state.get("region_protect_new_split_children", False),
            bayesian_smoothing_C=state.get("bayesian_smoothing_C", 0.5),
        )
        mgr.topology_last_edit_section = int(state.get("topology_last_edit_section", 0) or 0)
        mgr.topology_mid_maintenance_done_steps = {
            int(step) for step in state.get("topology_mid_maintenance_done_steps", [])
        }
        mgr.subtask_q = state.get("subtask_q", {})
        mgr.subtask_q_counts = state.get("subtask_q_counts", {})
        ledger_present = (
            "memory_success_sum_by_subtask" in state
            and "memory_total_count_by_subtask" in state
        )
        mgr.memory_success_sum_by_subtask = state.get("memory_success_sum_by_subtask", {})
        mgr.memory_total_count_by_subtask = state.get("memory_total_count_by_subtask", {})
        mgr.precluster_success_sum_by_subtask = state.get("precluster_success_sum_by_subtask", {})
        mgr.precluster_total_count_by_subtask = state.get("precluster_total_count_by_subtask", {})
        mgr._precluster_evidence_backfilled = bool(state.get("precluster_evidence_backfilled", False))
        mgr._has_complete_memory_evidence_ledger = bool(
            state.get("has_complete_memory_evidence_ledger", ledger_present)
        ) and ledger_present
        if not mgr._has_complete_memory_evidence_ledger:
            logger.warning(
                "RegionManager checkpoint at %s lacks per-memory direct reward evidence; "
                "hard-member-rebase split will use prior-only children, never q*count reconstruction.", path
            )
        source_ledger_present = (
            "region_source_success_by_region" in state and "region_source_total_by_region" in state
        )
        mgr.region_source_success_by_region = {
            int(rid): mgr._copy_source_ledger(ledger)
            for rid, ledger in state.get("region_source_success_by_region", {}).items()
        }
        mgr.region_source_total_by_region = {
            int(rid): mgr._copy_source_ledger(ledger)
            for rid, ledger in state.get("region_source_total_by_region", {}).items()
        }
        mgr._has_complete_region_source_evidence_ledger = bool(
            state.get("has_complete_region_source_evidence_ledger", source_ledger_present)
        ) and source_ledger_present
        if not mgr._has_complete_region_source_evidence_ledger:
            logger.warning(
                "RegionManager checkpoint at %s lacks source-attributed soft region evidence; "
                "soft_source_conserving split will use prior-only children rather than q*count reconstruction.", path
            )
        mgr._known_subtasks = state.get("known_subtasks", [])
        mgr._is_clustered = state.get("is_clustered", False)
        # A legacy *pre-cluster* checkpoint has no region-level evidence yet:
        # there are no regions/source slices to reconstruct or lose. Initializing
        # empty source ledgers as complete is exact, and lets every post-resume
        # soft update be tracked from the first clustering event. A legacy
        # clustered checkpoint remains incomplete and keeps the prior-only
        # fallback at a future split rather than inventing historical slices.
        if not source_ledger_present and not mgr._is_clustered:
            mgr.region_source_success_by_region = {}
            mgr.region_source_total_by_region = {}
            mgr._has_complete_region_source_evidence_ledger = True
            logger.info(
                "Legacy pre-cluster RegionManager checkpoint at %s: initialized "
                "an empty complete source-evidence ledger for exact future soft splits.",
                path,
            )
        mgr._global_reward_sum = state.get("global_reward_sum", 0.0)
        mgr._global_reward_count = state.get("global_reward_count", 0)

        # Restore membership weights
        for k, v in state.get("membership_weights", {}).items():
            mgr.membership_weights[k] = np.array(v)

        # Restore regions
        for r_data in state.get("regions", []):
            centroid = np.array(r_data["centroid"]) if r_data["centroid"] else None
            # Backward compat: old checkpoints lack prior_*_by_subtask. Derive an
            # approximate warm-start prior from the stored utility AND aggressively
            # downscale the polluted success_sum/total_count counters by 10× since
            # they were contaminated with raw-count init (see succ>count bug fix).
            # This is a one-time migration penalty and preserves the relative
            # utility ranking while letting new evidence dominate going forward.
            stored_util = r_data.get("utility_by_subtask", {}) or {}
            PRIOR_ESS = 5.0
            EPS = 0.01
            prior_a = r_data.get("prior_alpha_by_subtask")
            prior_b = r_data.get("prior_beta_by_subtask")
            success_sum = r_data.get("success_sum_by_subtask", {})
            total_count = r_data.get("total_count_by_subtask", {})
            if prior_a is None or prior_b is None:
                logger.warning(
                    "RegionManager checkpoint at %s is missing prior_(alpha|beta)_by_subtask "
                    "(pre-succ-bug-fix format). Deriving warm-start prior from stored utility "
                    "and downscaling polluted success/total counters by 10× for safe migration.",
                    path,
                )
                prior_a = {st: PRIOR_ESS * min(max(float(u), EPS), 1.0 - EPS)
                           for st, u in stored_util.items()}
                prior_b = {st: PRIOR_ESS * (1.0 - min(max(float(u), EPS), 1.0 - EPS))
                           for st, u in stored_util.items()}
                # Downscale polluted counters so new evidence can actually move them.
                success_sum = {st: float(v) * 0.1 for st, v in success_sum.items()}
                total_count = {st: float(v) * 0.1 for st, v in total_count.items()}
            mgr.regions.append(Region(
                region_id=r_data["region_id"],
                centroid=centroid,
                member_ids=r_data["member_ids"],
                utility_by_subtask=r_data["utility_by_subtask"],
                counts_by_subtask=r_data["counts_by_subtask"],
                success_sum_by_subtask=success_sum,
                total_count_by_subtask=total_count,
                prior_alpha_by_subtask=prior_a,
                prior_beta_by_subtask=prior_b,
            ))

        # Restore subtask embeddings
        for k, v in state.get("subtask_embeddings", {}).items():
            mgr._subtask_embeddings[k] = np.array(v)

        logger.info("RegionManager loaded from %s", path)
        return mgr

    # ====================================================================
    # v10 holdout retrieval helpers (see docs/ALFWORLD_V10_HOLDOUT_IMPL.md)
    # ====================================================================
    # Pure additions — do NOT modify existing methods. Used only when
    # `holdout_retrieval_mode` is set on RegionMemoryService.

    def compute_region_d1_quality(
        self, region_id: int, hide_subtask: Optional[str] = None
    ) -> float:
        """D1 = mean utility over known subtasks (excluding `hide_subtask`).

        Used as a deployable region-quality prior for ALFWorld holdout, where
        zero-shot transfer (`_estimate_region_utility_zero_shot`) is information-
        empty (per `analysis/pseudo_holdout_meta_eval.py`). D1 has 14pp
        quality-bin gap on holdout (`analysis/region_quality_uplift.py`).

        Args:
            region_id: target region's int id
            hide_subtask: subtask to exclude from mean (None = use all known)

        Returns:
            float in [0, 1]; falls back to 0.5 if region missing or no data
        """
        region = next((r for r in self.regions if r.region_id == region_id), None)
        if region is None or not region.utility_by_subtask:
            return 0.5
        vals = [
            u for s, u in region.utility_by_subtask.items()
            if hide_subtask is None or s != hide_subtask
        ]
        if not vals:
            return 0.5
        return float(np.mean(vals))

    def build_pure_d1_anchors(
        self, hide_subtask: Optional[str], k: int = 5,
    ) -> List[str]:
        """Pre-compute top-k memory IDs by region D1 quality (offline).

        Picks memories greedily from highest-D1 regions until k are collected.
        For ALFWorld holdout, these become the "Pure-D1 anchor" memories that
        are injected for every query (no query adaptation).

        Args:
            hide_subtask: subtask to hide from D1 (the holdout subtask itself)
            k: number of anchor memory IDs to return

        Returns:
            list of memory_id strings, length up to k
        """
        if not self.regions:
            return []
        # Score regions by D1
        region_d1 = [
            (r, self.compute_region_d1_quality(r.region_id, hide_subtask=hide_subtask))
            for r in self.regions
        ]
        region_d1.sort(key=lambda x: -x[1])
        anchors: List[str] = []
        seen: set = set()
        for region, d1 in region_d1:
            for mid in region.member_ids:
                if mid in seen:
                    continue
                anchors.append(mid)
                seen.add(mid)
                if len(anchors) >= k:
                    logger.info(
                        "[v10 anchors] built %d anchors from highest-D1 regions, "
                        "top region R%d D1=%.3f",
                        len(anchors), region_d1[0][0].region_id, region_d1[0][1],
                    )
                    return anchors
        logger.info(
            "[v10 anchors] only built %d/%d anchors (insufficient region members)",
            len(anchors), k,
        )
        return anchors

    # ====================================================================
    # Region failure summary (see docs/REGION_FAILURE_SUMMARY.md)
    # ====================================================================
    # Built dynamically after clustering / split-merge. Each region gets a
    # textual summary aggregated from its member failure memories' structured
    # fields (FAILURE_MODE / MISTAKES / FIXES / AVOID).
    # The summary is consumed at retrieval time to replace raw failure memory
    # content in prompt injection.

    def _build_region_failure_summaries(self, top_n: int = 3) -> None:
        """Build per-region failure summary from member failure memories.

        Requires `_mem_cache_lookup` callback to be set by the memory service
        (provides access to memory metadata content). If not set, summaries
        remain empty strings.

        Args:
            top_n: top-N items per field (mistakes/fixes) by frequency
        """
        mem_lookup = getattr(self, '_mem_cache_lookup', None)
        if mem_lookup is None:
            logger.debug(
                "[region failure summary] _mem_cache_lookup not set; skipping summary build"
            )
            return

        n_built = 0
        n_empty = 0
        total_members = 0
        resolved_members = 0
        for region in self.regions:
            failure_fields_list = []
            for mem_id in region.member_ids:
                total_members += 1
                try:
                    content, success = mem_lookup(mem_id)
                except Exception:
                    continue
                resolved_members += 1
                if success is not False:  # only failure memories
                    continue
                if not content:
                    continue
                fields = self._parse_failure_fields(content)
                if fields["failure_mode"] or fields["mistakes"]:
                    failure_fields_list.append(fields)

            if not failure_fields_list:
                region.failure_summary = ""
                n_empty += 1
                continue

            region.failure_summary = self._format_failure_summary(
                failure_fields_list, top_n=top_n,
            )
            n_built += 1

        coverage = (resolved_members / total_members) if total_members else 0.0
        logger.info(
            "[region failure summary] built %d summaries (%d empty regions, no failure mems); "
            "_mem_cache coverage: %d/%d (%.1f%%)",
            n_built, n_empty, resolved_members, total_members, coverage * 100,
        )

    @staticmethod
    def _parse_failure_fields(content: str) -> dict:
        """Parse FAILURE_MODE / MISTAKES / FIXES / AVOID from memory content."""
        import re as _re
        result = {"failure_mode": "", "mistakes": [], "fixes": [], "avoids": []}
        if "FAILURE_MODE:" in content:
            mode = content.split("FAILURE_MODE:")[1].split("\n")[0].strip()
            result["failure_mode"] = mode
        current_section = None
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("MISTAKES:"):
                current_section = "mistakes"
            elif line.startswith("FIXES:"):
                current_section = "fixes"
            elif line.startswith("AVOID:"):
                current_section = "avoids"
            elif line.startswith("- ") and current_section:
                item = line[2:].strip()
                if item and len(item) > 10:
                    result[current_section].append(item)
        return result

    @staticmethod
    def _normalize_failure_item(text: str) -> str:
        """Light normalization: lowercase object/receptacle names for grouping."""
        import re as _re
        text = text.strip().rstrip(".")
        text = _re.sub(
            r'\b(cabinet|drawer|shelf|sidetable|countertop|fridge|microwave|sinkbasin|toilet|bed|sofa|armchair|desk|dresser|garbagecan|coffeetable|diningtable|stoveburner)\s*\d*\b',
            '[receptacle]', text, flags=_re.IGNORECASE,
        )
        text = _re.sub(
            r'\b(kettle|apple|cup|mug|plate|knife|fork|spoon|pen|pencil|book|laptop|cellphone|keychain|statue|pillow|towel|cloth|spraybottle|toiletpaper|soapbar|soapbottle|creditcard|cd|bowl|potato|tomato|egg|bread|lettuce|butterknife|spatula|dishsponge|alarmclock|vase|newspaper|candle|lightswitch|desklamp|floorlamp|remote|watch|baseballbat|basketball|tennisracket|tissuebox|box|safe)\b',
            '[object]', text, flags=_re.IGNORECASE,
        )
        return text

    @classmethod
    def _format_failure_summary(cls, failure_fields_list: list, top_n: int = 3) -> str:
        """Build structured summary text from aggregated failure fields."""
        from collections import Counter
        mode_counter = Counter()
        mistake_counter = Counter()
        fix_counter = Counter()
        for fields in failure_fields_list:
            if fields["failure_mode"]:
                mode_counter[fields["failure_mode"]] += 1
            for m in fields["mistakes"]:
                mistake_counter[cls._normalize_failure_item(m)] += 1
            for f in fields["fixes"]:
                fix_counter[cls._normalize_failure_item(f)] += 1

        lines = [f"Common failure patterns (from {len(failure_fields_list)} failed attempts in this task region):"]
        top_modes = mode_counter.most_common(top_n)
        if top_modes:
            modes_str = "; ".join(f"{mode} ({n}x)" for mode, n in top_modes)
            lines.append(f"FAILURE TYPES: {modes_str}")
        top_mistakes = mistake_counter.most_common(top_n)
        if top_mistakes:
            lines.append("KEY MISTAKES TO AVOID:")
            for mistake, _ in top_mistakes:
                lines.append(f"- {mistake}")
        top_fixes = fix_counter.most_common(top_n)
        if top_fixes:
            lines.append("RECOVERY STRATEGIES:")
            for fix, _ in top_fixes:
                lines.append(f"- {fix}")
        return "\n".join(lines)

    def _build_region_success_summaries(self, top_n: int = 4) -> None:
        """Build per-region success pattern summary from member success memories.

        Symmetric to _build_region_failure_summaries, but aggregates the recurring
        procedural steps + key actions from SUCCESS memories (proceduralization
        SCRIPT format) into a compact "effective strategies" block. Helps tasks
        that need correct multi-step procedures (e.g. pick_two: search→stash→
        search→place) where generic per-memory scripts are too homogeneous.

        Requires `_mem_cache_lookup` callback (same as failure summaries).
        """
        mem_lookup = getattr(self, '_mem_cache_lookup', None)
        if mem_lookup is None:
            logger.debug(
                "[region success summary] _mem_cache_lookup not set; skipping summary build"
            )
            return

        n_built = 0
        n_empty = 0
        for region in self.regions:
            steps_list = []
            for mem_id in region.member_ids:
                try:
                    content, success = mem_lookup(mem_id)
                except Exception:
                    continue
                if success is not True:  # only success memories
                    continue
                if not content:
                    continue
                parsed = self._parse_success_steps(content)
                if parsed["steps"] or parsed["actions"]:
                    steps_list.append(parsed)

            if not steps_list:
                region.success_summary = ""
                n_empty += 1
                continue

            region.success_summary = self._format_success_summary(steps_list, top_n=top_n)
            n_built += 1

        logger.info(
            "[region success summary] built %d summaries (%d empty regions, no success mems)",
            n_built, n_empty,
        )

    @staticmethod
    def _parse_success_steps(content: str) -> dict:
        """Parse proceduralization SCRIPT into step labels + action keywords.

        Success memory content looks like:
          Task: ... Your task is to: put a cool cup in cabinet.
          1. **Locate the Object**: Identify the object by checking countertops...
          2. **Prepare the Object**: cool the object using the fridge...
          3. **Place the Object**: find the cabinet, open it first, put inside...

        Returns {"steps": [step_label,...], "actions": [action_keyword,...]}.
        """
        import re as _re
        result = {"steps": [], "actions": []}
        if not content:
            return result

        # Step labels: "**Step Name**" inside numbered list
        for m in _re.finditer(r'\*\*([^*]+)\*\*', content):
            label = m.group(1).strip().rstrip(':').strip()
            if label and len(label) <= 60:
                result["steps"].append(label)

        # Action keywords: map ALFWorld verbs/idioms to canonical actions.
        # Scan lowercased content for recurring effective actions.
        low = content.lower()
        action_patterns = {
            "open receptacle before placing": r'open\b.*(before|then).*(put|place)|if.*closed.*open',
            "check drawers/cabinets/shelves first": r'(drawer|cabinet|shelf|countertop|sinkbasin).*(check|search|look|likely)',
            "cool object in fridge": r'cool\b.*fridge|fridge.*cool',
            "heat object in microwave": r'heat\b.*microwave|microwave.*heat',
            "clean object in sinkbasin": r'clean\b.*(sink|basin)|(sink|basin).*clean',
            "examine object under lamp": r'(examine|look at|inspect).*(lamp|light)|(lamp|light).*(examine|look)',
            "take object then navigate to target": r'take\b.*(then|and).*(go|navigate|move)',
            "systematically check locations": r'systematic|one by one|each location',
            "verify receptacle is valid": r'verif|valid receptacle|appropriate.*receptacle|suitable.*placement',
        }
        for canonical, pat in action_patterns.items():
            if _re.search(pat, low):
                result["actions"].append(canonical)
        return result

    @classmethod
    def _format_success_summary(cls, steps_list: list, top_n: int = 4) -> str:
        """Build structured success-pattern text from aggregated success steps."""
        from collections import Counter
        step_counter = Counter()
        action_counter = Counter()
        for parsed in steps_list:
            # Dedup within a single memory so one memory contributes each item once
            for s in set(parsed["steps"]):
                step_counter[cls._normalize_failure_item(s)] += 1
            for a in set(parsed["actions"]):
                action_counter[a] += 1

        lines = [f"Effective strategies (from {len(steps_list)} successful attempts in this task region):"]
        top_steps = step_counter.most_common(top_n)
        if top_steps:
            steps_str = "; ".join(f"{step} ({n}x)" for step, n in top_steps)
            lines.append(f"COMMON STEPS: {steps_str}")
        top_actions = action_counter.most_common(top_n)
        if top_actions:
            lines.append("KEY ACTIONS THAT WORKED:")
            for action, _ in top_actions:
                lines.append(f"- {action}")
        return "\n".join(lines)

    def _build_region_experience_cards(self, max_cards_per_region: int = 5) -> None:
        """Build per-region experience cards: atomic fact/constraint/gotcha items.

        Unlike failure_summary (which only uses failure memories), this uses BOTH
        success and failure members to distill transferable knowledge:
        - From failures: MISTAKES + FIXES (already parsed by _parse_failure_fields)
        - From successes: key API patterns / edge-case handling extracted from content

        Each card is a compact string (<120 chars) that's useful across tasks in the
        same region. Cards are frequency-ranked (most common patterns first).

        Requires `_mem_cache_lookup` callback (same as failure summaries).
        """
        from collections import Counter

        mem_lookup = getattr(self, '_mem_cache_lookup', None)
        if mem_lookup is None:
            return

        n_built = 0
        for region in self.regions:
            card_candidates = Counter()

            for mem_id in region.member_ids:
                try:
                    content, success = mem_lookup(mem_id)
                except Exception:
                    continue
                if not content:
                    continue

                if success is False:
                    # From failure: extract MISTAKES and FIXES as cards
                    fields = self._parse_failure_fields(content)
                    for m in fields["mistakes"]:
                        normalized = self._normalize_failure_item(m)
                        if len(normalized) > 15:
                            card_candidates[f"[avoid] {normalized}"] += 1
                    for f in fields["fixes"]:
                        normalized = self._normalize_failure_item(f)
                        if len(normalized) > 15:
                            card_candidates[f"[do] {normalized}"] += 1
                else:
                    # From success: extract key patterns (simpler heuristic)
                    # Look for common success patterns in the content
                    cards = self._extract_success_patterns(content)
                    for c in cards:
                        card_candidates[c] += 1

            # Take top-N most frequent cards for this region
            top_cards = [card for card, _count in card_candidates.most_common(max_cards_per_region)]
            region.experience_cards = top_cards
            if top_cards:
                n_built += 1

        logger.info(
            "[region experience cards] built cards for %d/%d regions (max %d cards each)",
            n_built, len(self.regions), max_cards_per_region,
        )

    @staticmethod
    def _extract_success_patterns(content: str) -> list:
        """Extract atomic fact cards from a success memory's content.

        Heuristic: look for import statements, key API calls, and edge-case
        handling patterns. Returns list of short card strings.
        """
        import re
        cards = []
        # Extract "import X" patterns (what libs were needed)
        imports = re.findall(r'(?:from\s+(\S+)\s+import|import\s+(\S+))', content)
        # Extract edge-case patterns (if X is None, if len(X) == 0, etc.)
        edge_cases = re.findall(r'(?:if\s+(?:not\s+)?(?:len\([^)]+\)\s*==\s*0|[\w.]+\s+is\s+None|[\w.]+\.empty))', content)
        for ec in edge_cases[:2]:
            cards.append(f"[edge_case] handle: {ec.strip()[:80]}")
        # Extract explicit type conversions / assertions
        type_checks = re.findall(r'(?:isinstance\([^)]+\)|assert\s+\w+)', content)
        for tc in type_checks[:1]:
            cards.append(f"[contract] {tc.strip()[:80]}")
        return cards
