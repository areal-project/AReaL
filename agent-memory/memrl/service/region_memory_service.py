"""
Region-aware memory service with per-subtask Q values.

Key changes from baseline MemRL:
1. Per-subtask Q: each memory has Q[subtask] instead of a single scalar Q
2. Retrieval uses Q[target_subtask] for the hybrid score
3. Region gating multiplier based on utility-clustered regions
4. Global Q (per-benchmark) for inter-transfer evaluation
"""

from typing import Any, Dict, List, Optional, Tuple
import logging
import os
import threading
import json
import hashlib
from datetime import datetime

from memrl.service.memory_service import MemoryService
from memrl.service.region_manager import RegionManager
from memrl.service.weighted_region_routing import rank_regions_weighted, apply_region_quota, dedupe_ranked_candidates

logger = logging.getLogger(__name__)


class _ThreadLocalQCache(dict):
    """Dict subclass that checks thread-local override before base cache.

    Allows parallel threads to each use their own per-subtask Q cache
    without a global lock, while writes always go to the base cache.
    Inherits from dict so isinstance checks pass (parent code uses
    isinstance(self._q_cache, dict)).
    """

    def __init__(self, base_cache: dict, thread_local: threading.local):
        # Don't call super().__init__() with data — we delegate all storage to _base
        super().__init__()
        object.__setattr__(self, '_base', base_cache)
        object.__setattr__(self, '_tl', thread_local)

    def _active(self):
        return getattr(self._tl, 'q_cache_override', None)

    def __contains__(self, key):
        override = self._active()
        if override is not None:
            return key in override
        return key in self._base

    def __getitem__(self, key):
        override = self._active()
        if override is not None:
            return override[key]
        return self._base[key]

    def get(self, key, default=None):
        override = self._active()
        if override is not None:
            return override.get(key, default)
        return self._base.get(key, default)

    def __setitem__(self, key, value):
        # When a thread-local override is active (parallel eval), absorb writes
        # silently to avoid corrupting the shared _base from concurrent threads.
        if self._active() is not None:
            return
        self._base[key] = value

    def __delitem__(self, key):
        del self._base[key]

    def __len__(self):
        return len(self._base)

    def __iter__(self):
        return iter(self._base)

    def pop(self, key, *args):
        return self._base.pop(key, *args)

    def keys(self):
        return self._base.keys()

    def values(self):
        return self._base.values()

    def items(self):
        return self._base.items()

    def update(self, *args, **kwargs):
        self._base.update(*args, **kwargs)

    def clear(self):
        self._base.clear()


class RegionMemoryService(MemoryService):
    """
    Memory service with per-subtask Q and utility-based region gating.

    Extends MemoryService:
    - Overrides Q lookup to use per-subtask Q[target_subtask] during retrieval
    - Adds region gating as a score multiplier after hybrid scoring
    - Maintains global Q (per-benchmark) for inter-transfer evaluation
    """

    def __init__(self, *args, region_manager: Optional[RegionManager] = None, **kwargs):
        # Extract region config before passing to super
        self.region_gating_mode = kwargs.pop("region_gating_mode", "multiplicative")
        self.region_value_mode = kwargs.pop("region_value_mode", "shrinkage")
        if self.region_value_mode not in {"shrinkage", "category_q"}:
            raise ValueError(f"unknown region_value_mode={self.region_value_mode!r}")
        explore_schedule_str = kwargs.pop("explore_schedule", "0,4,3,2,2,1,1,1,1,0")
        self.explore_success_ratio = kwargs.pop("explore_success_ratio", 0.7)
        # v5 retrieve mode (see docs/ALFWORLD_REGION_IMPROVEMENT_PLAN.md §14)
        # "global"          = current behavior, global sim top-k + rerank
        # "quota_fixed"     = dual recall + strict quota_max region picks, no gates
        # "quota_adaptive"  = quota_fixed + 5 safety gates
        self.retrieve_mode = os.environ.get(
            "MEMRL_REGION_RETRIEVE_MODE", kwargs.pop("region_retrieve_mode", "global")
        )
        self.quota_max = int(kwargs.pop("quota_max", 3))
        self.weighted_quota_count = int(os.environ.get("MEMRL_WEIGHTED_REGION_QUOTA", "2"))
        self.weighted_quota_min_sim = float(os.environ.get("MEMRL_WEIGHTED_REGION_MIN_SIM", "0.45"))
        self.weighted_quota_margin = float(os.environ.get("MEMRL_WEIGHTED_REGION_UTILITY_MARGIN", "0.03"))
        self.weighted_quota_min_count = float(os.environ.get("MEMRL_WEIGHTED_REGION_MIN_COUNT", "30"))
        # Optional explicit centered Region advantage. Disabled by default so
        # historical experiments retain their exact shrinkage-only semantics.
        self.explicit_region_lambda = float(os.environ.get("MEMRL_EXPLICIT_REGION_LAMBDA", "0"))
        self.explicit_region_min_range = float(os.environ.get("MEMRL_EXPLICIT_REGION_MIN_RANGE", "0.03"))
        self.quota_min_sim_floor = float(kwargs.pop("quota_min_sim_floor", 0.5))
        self.quota_utility_margin = float(kwargs.pop("quota_utility_margin", 0.15))
        self.quota_ood_dist_threshold = float(kwargs.pop("quota_ood_dist_threshold", 0.7))
        self.quota_subtask_conf_thresholds = kwargs.pop(
            "quota_subtask_conf_thresholds", [0.5, 0.7, 0.9]
        )
        self.quota_region_min_count = int(kwargs.pop("quota_region_min_count", 30))
        # v5.5 utility_anchor mode (see docs/ALFWORLD_REGION_IMPROVEMENT_PLAN.md §22)
        # Inspired by v10 H: anchor from best region by utility[target] directly,
        # NOT filtered through sim-recalled narrow pool. Solves "noop_no_member 26%"
        # root cause observed in 929226 v5 quota cells.
        self.utility_anchor_count = int(kwargs.pop("utility_anchor_count", 3))
        self.utility_anchor_topk_regions = int(kwargs.pop("utility_anchor_topk_regions", 1))
        self.utility_anchor_min_count = int(kwargs.pop("utility_anchor_min_count", 30))
        # v10 holdout retrieval (see docs/ALFWORLD_V10_HOLDOUT_IMPL.md)
        # None     = default (zero-shot transfer via shrinkage_q, info-empty per analysis)
        # pure_d1  = always inject fixed top-k by region D1 (no query, no sim)
        # hybrid   = top-`holdout_d1_anchors` by D1 + remaining by sim*D1
        # sim_d1   = all top-k by sim*D1 with pool=holdout_pool_size
        self.holdout_retrieval_mode = kwargs.pop("holdout_retrieval_mode", None)
        self.holdout_pool_size = int(kwargs.pop("holdout_pool_size", 500))
        self.holdout_d1_anchors = int(kwargs.pop("holdout_d1_anchors", 3))
        self._holdout_subtask = kwargs.pop("holdout_subtask", None)
        self.retrieval_audit_path = os.environ.get("MEMRL_RETRIEVAL_AUDIT_PATH", "").strip()
        # Feedback-first Region membership (ALFWorld semantics): newly created
        # memories enter utility topology only after being retrieved and receiving
        # an actual task reward via update_subtask_q.
        self.region_register_on_create = os.environ.get(
            "MEMRL_REGION_REGISTER_ON_CREATE", "1"
        ).lower() in {"1", "true", "yes"}
        self.region_backfill_on_restore = os.environ.get(
            "MEMRL_REGION_BACKFILL_ON_RESTORE", "1"
        ).lower() in {"1", "true", "yes"}
        self._retrieval_audit_lock = threading.Lock()
        # Cache: anchor mem_ids (list[str]) for current clustering state
        # Invalidated whenever region clustering changes (handled in _post_cluster_hook).
        self._v10_pure_d1_anchors: Optional[List[str]] = None
        super().__init__(*args, **kwargs)
        self.region_manager = region_manager
        if self.retrieve_mode not in {"global", "quota_fixed", "quota_adaptive", "utility_anchor", "weighted_quota"}:
            raise ValueError(
                f"region_retrieve_mode must be one of "
                f"global/quota_fixed/quota_adaptive/utility_anchor/weighted_quota, got {self.retrieve_mode!r}"
            )
        # v5 codex review 2 fix: fail-fast parameter validation
        if self.quota_max < 0:
            raise ValueError(f"quota_max must be >= 0, got {self.quota_max}")
        if self.weighted_quota_count < 0:
            raise ValueError(f"weighted_quota_count must be >= 0, got {self.weighted_quota_count}")
        if not (0.0 <= self.weighted_quota_min_sim <= 1.0):
            raise ValueError("MEMRL_WEIGHTED_REGION_MIN_SIM must be in [0, 1]")
        if self.weighted_quota_margin < 0.0 or self.weighted_quota_min_count < 0.0:
            raise ValueError("weighted Region margin/min-count must be non-negative")
        if not (0.0 <= self.quota_min_sim_floor <= 1.0):
            raise ValueError(
                f"quota_min_sim_floor must be in [0, 1], got {self.quota_min_sim_floor}"
            )
        if self.quota_region_min_count < 1:
            raise ValueError(
                f"quota_region_min_count must be >= 1, got {self.quota_region_min_count}"
            )
        thresholds_list = list(self.quota_subtask_conf_thresholds)
        if len(thresholds_list) != 3:
            raise ValueError(
                f"quota_subtask_conf_thresholds must have length 3 "
                f"(lo, mid, hi), got {thresholds_list}"
            )
        if thresholds_list != sorted(thresholds_list):
            raise ValueError(
                f"quota_subtask_conf_thresholds must be monotonic non-decreasing, "
                f"got {thresholds_list}"
            )
        self.quota_subtask_conf_thresholds = thresholds_list
        # v5.5 utility_anchor validation
        if self.utility_anchor_count < 0:
            raise ValueError(f"utility_anchor_count must be >= 0, got {self.utility_anchor_count}")
        if self.utility_anchor_topk_regions < 1:
            raise ValueError(f"utility_anchor_topk_regions must be >= 1, got {self.utility_anchor_topk_regions}")
        if self.utility_anchor_min_count < 1:
            raise ValueError(f"utility_anchor_min_count must be >= 1, got {self.utility_anchor_min_count}")
        # Lock for _quota_stats updates (parallel batch safety)
        self._quota_stats_lock = threading.Lock()
        logger.info(
            "[RegionMemoryService] retrieve_mode=%s quota_max=%d (gates: min_sim>=%.2f, "
            "utility_margin>%.2f, ood_dist<%.2f, subtask_conf thresholds=%s, region_min_count=%d)",
            self.retrieve_mode, self.quota_max, self.quota_min_sim_floor,
            self.quota_utility_margin, self.quota_ood_dist_threshold,
            self.quota_subtask_conf_thresholds, self.quota_region_min_count,
        )
        # v10 holdout retrieval validation (see docs/ALFWORLD_V10_HOLDOUT_IMPL.md)
        if self.holdout_retrieval_mode is not None:
            if self.holdout_retrieval_mode not in {"pure_d1", "hybrid", "sim_d1"}:
                raise ValueError(
                    f"holdout_retrieval_mode must be one of "
                    f"pure_d1/hybrid/sim_d1/None, got {self.holdout_retrieval_mode!r}"
                )
            if self.holdout_pool_size < 1:
                raise ValueError(f"holdout_pool_size must be >= 1, got {self.holdout_pool_size}")
            if self.holdout_d1_anchors < 0:
                raise ValueError(f"holdout_d1_anchors must be >= 0, got {self.holdout_d1_anchors}")
            # v10 codex review #3 fix: hard-fail if holdout_subtask not set.
            # Otherwise v10 silently never triggers and we get default zero-shot
            # transfer for entire 30h job with no obvious failure signal.
            if not self._holdout_subtask:
                raise ValueError(
                    f"holdout_retrieval_mode={self.holdout_retrieval_mode!r} requires "
                    f"holdout_subtask to be set (got None or empty). Pass "
                    f"--holdout_subtask alf/<name> or set experiment.holdout_subtask in yaml."
                )
            logger.info(
                "[RegionMemoryService v10] holdout_retrieval_mode=%s, pool=%d, anchors=%d, holdout_subtask=%r",
                self.holdout_retrieval_mode, self.holdout_pool_size,
                self.holdout_d1_anchors, self._holdout_subtask,
            )
            # First-hit counter so we can verify the intercept actually fires.
            self._v10_intercept_hits = 0
            self._v10_intercept_misses = 0

        # Parse exploration schedule (tolerate empty/blank string -> no schedule)
        self._explore_schedule = [
            int(x) for x in explore_schedule_str.split(",") if x.strip()
        ]

        # Wire up embedding lookup for similarity propagation
        if self.region_manager:
            self.region_manager._embedding_lookup = self._get_mem_embedding
            self.region_manager._invalidate_embedding_cache = self.invalidate_embedding_cache
            # Wire up mem cache lookup for failure summary generation
            self.region_manager._mem_cache_lookup = self._get_mem_content_and_success

        # Thread-local storage for per-thread Q cache override (parallel batch safety)
        self._thread_local = threading.local()
        # Wrap _q_cache with thread-local-aware proxy so parallel threads
        # each see their own per-subtask Q during retrieval without a global lock.
        self._q_cache_base = self._q_cache  # save reference to actual dict
        self._q_cache = _ThreadLocalQCache(self._q_cache_base, self._thread_local)

        # Global Q for inter-transfer: {mem_id: {benchmark: q_value}}
        self._global_q_cache: Dict[str, Dict[str, float]] = {}
        self._global_q_cache_max_size = 1000000

        # Track current target_subtask for Q lookup override
        self._current_target_subtask: Optional[str] = None

        # Buffer: track (target_subtask, selected_ids) from retrieve_query calls
        # Consumed by update_values when target_subtasks not explicitly passed
        self._retrieval_subtask_buffer: List[str] = []

        # Epoch tracking for annealing exploration
        self._current_epoch: int = 1
        self._num_epochs: int = 10

    def set_current_epoch(self, epoch: int, num_epochs: int = 10) -> None:
        """Set current epoch for exploration annealing."""
        self._current_epoch = epoch
        self._num_epochs = num_epochs

    def retrieve_query(
        self,
        task_description: str,
        k: int = 5,
        threshold: float = 0.0,
        # Region-aware parameters
        filter_source_subtasks: Optional[List[str]] = None,
        filter_source_benchmark: Optional[str] = None,
        target_subtask: Optional[str] = None,
        target_subtask_weights: Optional[List[Tuple[str, float]]] = None,
        eval_mode: Optional[str] = None,
        use_region_gating: bool = True,
        audit_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Region-aware retrieval with per-subtask Q.

        Scoring: hybrid_score = sim * w_sim + Q[target_subtask] * w_q
        Then: final_score = hybrid_score * region_gating_score
        """
        # Optional weighted multi-axis target.  The legacy single-label path
        # remains unchanged when this is None.  Normalize defensively so one
        # task always contributes total evidence/score mass 1.0.
        weighted_targets: Optional[List[Tuple[str, float]]] = None
        if target_subtask_weights:
            cleaned = [(str(st), max(0.0, float(w))) for st, w in target_subtask_weights if st]
            total_weight = sum(w for _, w in cleaned)
            if total_weight > 0:
                weighted_targets = [(st, w / total_weight) for st, w in cleaned]
                target_subtask = max(weighted_targets, key=lambda item: item[1])[0]

        routing_audit = {"applied": False, "reason": "not_evaluated", "region_scores": []}
        # `k` is the recall budget (LLB config k_retrieve=10). The parent RL
        # selector has its own final context budget (rl_config.topk=5). All
        # Region quota, exploration and diversity refill must respect final_k.
        configured_topk = int(getattr(self.rl_config, "topk", k) or k) if self.rl_config else int(k)
        final_k = max(1, min(int(k), configured_topk))

        # ----------------------------------------------------------------
        # v10 holdout retrieval intercept (see docs/ALFWORLD_V10_HOLDOUT_IMPL.md)
        # When target_subtask is the configured holdout and a v10 mode is set,
        # bypass zero-shot transfer (which is info-empty per offline meta-eval)
        # and route to D1-based retrieval. Default path is untouched when
        # holdout_retrieval_mode is None.
        # ----------------------------------------------------------------
        if self.holdout_retrieval_mode is not None:
            if (self.region_manager is not None
                    and target_subtask is not None
                    and target_subtask == self._holdout_subtask):
                self._v10_intercept_hits += 1
                if self._v10_intercept_hits == 1:
                    logger.info(
                        "[v10 INTERCEPT FIRST HIT] mode=%s target=%r holdout=%r — D1 path active",
                        self.holdout_retrieval_mode, target_subtask, self._holdout_subtask,
                    )
                return self._retrieve_v10_holdout(
                    task_description=task_description,
                    k=k,
                    threshold=threshold,
                    target_subtask=target_subtask,
                    filter_source_subtasks=filter_source_subtasks,
                    filter_source_benchmark=filter_source_benchmark,
                    eval_mode=eval_mode,
                )
            else:
                self._v10_intercept_misses += 1
                if self._v10_intercept_misses == 1:
                    logger.warning(
                        "[v10 INTERCEPT MISS] mode=%s set but target_subtask=%r != holdout=%r "
                        "(or region_manager is None). Falling back to default retrieval.",
                        self.holdout_retrieval_mode, target_subtask, self._holdout_subtask,
                    )
                # Periodically warn if we're consistently missing intercept
                if self._v10_intercept_misses % 100 == 0:
                    logger.warning(
                        "[v10 INTERCEPT MISS] %d misses, %d hits so far",
                        self._v10_intercept_misses, self._v10_intercept_hits,
                    )

        # Set target subtask so _get_q_value can use it
        # NOTE: this field is currently unused but kept for potential subclass use.
        # Setting it inside the lock to avoid thread race.

        # Serialize _q_cache swap to prevent race conditions in parallel batches.
        # Build the per-subtask Q cache outside the lock (read-only, thread-safe).
        subtask_q_cache = None
        if eval_mode == "inter" and target_subtask:
            target_benchmark = target_subtask.split("/", 1)[0] if "/" in target_subtask else target_subtask
            subtask_q_cache = self._build_global_q_for_benchmark(target_benchmark)
        elif weighted_targets and self.region_manager:
            subtask_q_cache = self._build_weighted_subtask_q_cache(
                weighted_targets, use_shrinkage=(
                    self.region_gating_mode == "additive"
                    and getattr(self, "region_value_mode", "shrinkage") != "category_q"
                )
            )
        elif target_subtask and self.region_manager:
            if self.region_value_mode == "category_q":
                subtask_q_cache = self._build_subtask_q_cache(target_subtask)
            elif self.region_gating_mode == "additive":
                subtask_q_cache = self._build_shrinkage_q_cache(target_subtask)
            else:
                subtask_q_cache = self._build_subtask_q_cache(target_subtask)

        fetch_k = k * 3 if (filter_source_subtasks or filter_source_benchmark) else k

        # Thread-local Q override: each thread sees its own subtask_q_cache
        # via the _ThreadLocalQCache proxy, no lock needed.
        self._thread_local.q_cache_override = subtask_q_cache
        self._thread_local.target_subtask = target_subtask
        try:
            raw_result = super().retrieve_query(task_description, fetch_k, threshold)
        finally:
            self._thread_local.q_cache_override = None
            self._thread_local.target_subtask = None

        # Parent returns (dict, sim_list) tuple
        if isinstance(raw_result, tuple):
            result, sim_list = raw_result
        else:
            result, sim_list = raw_result, None
        if result is None:
            result = {"selected": [], "candidates": [], "simmax": 0.0, "actions": []}

        if not result.get("selected"):
            if target_subtask:
                self._retrieval_subtask_buffer.append(target_subtask)
            return result, sim_list

        candidates = result["selected"]

        # Source filtering
        if filter_source_subtasks:
            candidates = self._filter_by_source_subtasks(candidates, filter_source_subtasks)

        if filter_source_benchmark:
            candidates = self._filter_by_source_benchmark(candidates, filter_source_benchmark)

        if not candidates:
            if target_subtask:
                self._retrieval_subtask_buffer.append(target_subtask)
            return {"actions": [], "selected": [], "candidates": [], "simmax": result.get("simmax", 0.0)}, sim_list

        # Optional explicit Region intervention over the full recall pool.
        # Unlike additive shrinkage (which only changes Q), this term is visible
        # and auditable in the final score and can promote a candidate that was
        # recalled in k1 but not selected by the parent top-k.
        if (use_region_gating and self.region_manager and target_subtask
                and self.explicit_region_lambda != 0.0):
            rerank_pool = list(result.get("candidates", []) or candidates)
            if filter_source_subtasks:
                rerank_pool = self._filter_by_source_subtasks(rerank_pool, filter_source_subtasks)
            if filter_source_benchmark:
                rerank_pool = self._filter_by_source_benchmark(rerank_pool, filter_source_benchmark)
            region_values = []
            for cand in rerank_pool:
                mem_id = cand.get("memory_id")
                if weighted_targets:
                    region_value = sum(
                        weight * self.region_manager.compute_region_gating_score(
                            mem_id, subtask, eval_mode or "train", None
                        )
                        for subtask, weight in weighted_targets
                    )
                else:
                    region_value = self.region_manager.compute_region_gating_score(
                        mem_id, target_subtask, eval_mode or "train", None
                    )
                region_values.append(float(region_value))
            value_range = (max(region_values) - min(region_values)) if region_values else 0.0
            if region_values and value_range >= self.explicit_region_min_range:
                center = sum(region_values) / len(region_values)
                for cand, region_value in zip(rerank_pool, region_values):
                    cand["base_score"] = float(cand.get("score", 0.0) or 0.0)
                    cand["region_value"] = region_value
                    cand["region_advantage"] = region_value - center
                    cand["score"] = cand["base_score"] + self.explicit_region_lambda * cand["region_advantage"]
                rerank_pool.sort(key=lambda c: float(c.get("score", 0.0)), reverse=True)
                candidates = rerank_pool[:final_k]
                result["candidates"] = rerank_pool
                routing_audit.update({
                    "explicit_region_applied": True,
                    "explicit_region_lambda": float(self.explicit_region_lambda),
                    "explicit_region_range": float(value_range),
                })
            else:
                routing_audit.update({
                    "explicit_region_applied": False,
                    "explicit_region_reason": "utility_range",
                    "explicit_region_range": float(value_range),
                })

        # Region gating (only for multiplicative mode; additive already baked into Q)
        if (use_region_gating and self.region_manager and target_subtask and
                self.region_value_mode != "category_q" and
                not weighted_targets and self.region_gating_mode == "multiplicative"):
            candidates = self._apply_region_gating(
                candidates, target_subtask, eval_mode or "train", None
            )

        # Weighted DB multi-label routing: reserve a small number of slots from
        # the clearly best Region, while retaining global hybrid fill. Abstain
        # when Region utilities are too close or evidence is insufficient.
        if (self.retrieve_mode == "weighted_quota" and self.region_manager
                and weighted_targets and candidates):
            routing_pool = list(result.get("candidates", []) or candidates)
            if filter_source_subtasks:
                routing_pool = self._filter_by_source_subtasks(routing_pool, filter_source_subtasks)
            if filter_source_benchmark:
                routing_pool = self._filter_by_source_benchmark(routing_pool, filter_source_benchmark)
            candidates, routing_audit = self._apply_weighted_region_quota(
                global_ranked=candidates, candidate_pool=routing_pool,
                weighted_targets=weighted_targets, k=final_k,
            )

        # v5 quota-based recall (see docs/ALFWORLD_REGION_IMPROVEMENT_PLAN.md §14)
        # Branch only when mode != "global" and we have region structure available.
        # codex review #3 fix: use `candidates` (post source filtering + gating) as
        # BOTH global_ranked and all_candidates so quota never pulls from filtered-out
        # memories. Trade-off: candidates may be limited (k*3 fetch_k); accept narrower
        # pool over breaking gating semantics.
        if (self.retrieve_mode in {"quota_fixed", "quota_adaptive"} and self.region_manager
                and target_subtask and candidates):
            candidates = self._apply_quota_recall(
                global_ranked=candidates,
                all_candidates=candidates,
                target_subtask=target_subtask,
                eval_mode=eval_mode,
                k=final_k,
            )
        # v5.5 utility_anchor mode (see docs/ALFWORLD_REGION_IMPROVEMENT_PLAN.md §22)
        # Inspired by v10 H, but for known target: anchor from best region's
        # member_ids DIRECTLY (NOT filtered through sim pool). Solves v5 quota's
        # noop_no_member=26% by decoupling anchor from sim recall.
        elif (self.retrieve_mode == "utility_anchor" and self.region_manager
                and target_subtask):
            candidates = self._apply_utility_anchor(
                sim_ranked=candidates,
                target_subtask=target_subtask,
                eval_mode=eval_mode,
                k=final_k,
                candidate_pool=list(result.get("candidates", []) or candidates),
            )

        candidates = candidates[:final_k]

        # Exploration from config schedule
        if (eval_mode is None and len(candidates) >= 2
                and len(result.get("candidates", [])) > len(candidates)):
            epoch_idx = self._current_epoch - 1  # 0-based
            if epoch_idx < len(self._explore_schedule):
                n_explore = self._explore_schedule[epoch_idx]
            else:
                n_explore = 0
            n_explore = min(n_explore, len(candidates) - 1, final_k - 1)
            if n_explore > 0:
                self._inject_exploration(
                    candidates, result.get("candidates", []),
                    n_explore=n_explore,
                    success_ratio=self.explore_success_ratio,
                )

        # Final common diversity pass applies to both Region-routed and global
        # fallback paths, after exploration so exploration cannot reintroduce
        # same-task memories from different epochs. Backfill from the larger pool.
        diversity_pool = list(result.get("candidates", []) or [])
        if filter_source_subtasks:
            diversity_pool = self._filter_by_source_subtasks(diversity_pool, filter_source_subtasks)
        if filter_source_benchmark:
            diversity_pool = self._filter_by_source_benchmark(diversity_pool, filter_source_benchmark)
        if os.environ.get("MEMRL_FINAL_MEMORY_DEDUP", "0").lower() in {"1", "true", "yes"}:
            candidates = dedupe_ranked_candidates(candidates, diversity_pool, k=final_k)
        else:
            candidates = candidates[:final_k]

        # Track target_subtask for auto-consumption by update_values
        if target_subtask:
            self._retrieval_subtask_buffer.append(target_subtask)

        return {
            "actions": [c["memory_id"] for c in candidates],
            "selected": candidates,
            "candidates": result.get("candidates", []),
            "simmax": result.get("simmax", 0.0),
            "region_routing": routing_audit,
            "target_subtask_weights": weighted_targets or [],
            "audit_context": audit_context or {},
            "recall_k": int(k),
            "final_k": int(final_k),
        }, sim_list

    def _fill_untracked_memories(self, flat: Dict[str, float]) -> Dict[str, float]:
        """Fill Q values for memories not in region_manager.subtask_q.

        If enough subtask Q entries exist (>=10), rescale parent scalar Q to
        match subtask Q distribution. Otherwise fall back to parent Q directly
        to avoid unstable rescaling from tiny samples.
        """
        MIN_ENTRIES_FOR_RESCALE = 10

        use_rescale = len(flat) >= MIN_ENTRIES_FOR_RESCALE
        if use_rescale:
            sq_mean = sum(flat.values()) / len(flat)
            sq_var = sum((v - sq_mean)**2 for v in flat.values()) / len(flat)
            sq_std = sq_var ** 0.5
            if sq_std < 0.005:
                use_rescale = False

        if use_rescale:
            pq_vals = list(self._q_cache.values())
            if pq_vals:
                pq_mean = sum(pq_vals) / len(pq_vals)
                pq_std = max((sum((v - pq_mean)**2 for v in pq_vals) / len(pq_vals)) ** 0.5, 1e-9)
            else:
                pq_mean, pq_std = 0.0, 1.0

            for mem_id, q_val in self._q_cache.items():
                if mem_id not in flat:
                    flat[mem_id] = (q_val - pq_mean) / pq_std * sq_std + sq_mean
        else:
            for mem_id, q_val in self._q_cache.items():
                if mem_id not in flat:
                    flat[mem_id] = q_val

        default_q = (sum(flat.values()) / len(flat)) if flat else 0.5
        if hasattr(self, 'dict_memory') and self.dict_memory:
            for query_key, mem_ids in self.dict_memory.items():
                for mid in mem_ids:
                    if mid not in flat:
                        flat[mid] = default_q

        return flat

    # ====================================================================
    # v10 holdout retrieval (pure_d1 / hybrid / sim_d1)
    # See docs/ALFWORLD_V10_HOLDOUT_IMPL.md
    # ====================================================================

    def _v10_get_anchor_ids(self, hide_subtask: Optional[str], k: int) -> List[str]:
        """Top-k Pure-D1 anchor memory IDs.

        v10 codex review #1 fix: NO CACHE. Region clustering mutates mid-run
        (split/merge every batch via cluster_by_utility), so cached anchors
        become stale and bias P/H toward outdated memories silently. Recompute
        every call — cost is small (~50 regions × O(subtasks)) vs LLM/retrieval.
        """
        return self.region_manager.build_pure_d1_anchors(
            hide_subtask=hide_subtask, k=k,
        )

    def _v10_invalidate_anchor_cache(self) -> None:
        """Deprecated — anchors are recomputed every call (see _v10_get_anchor_ids)."""
        # Kept for API compatibility if external code calls it.
        self._v10_pure_d1_anchors = None

    def _v10_build_candidate_from_mem_id(
        self, mem_id: str, similarity: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        """Build a candidate dict (memory_id, content, similarity, metadata) by mem_id.

        Used by Pure-D1 path which has no sim score; sets similarity=0 by default.
        Falls back to None if mem_obj can't be loaded.
        """
        try:
            mem_obj = self._mem_cache.get(mem_id)
            if mem_obj is None:
                with self._db_gate:
                    mem_obj = self.mos.get(
                        mem_cube_id=self.default_cube_id,
                        memory_id=mem_id,
                        user_id=self.user_id,
                    )
                if mem_obj is not None:
                    self._add_to_mem_cache(mem_id, mem_obj)
            if mem_obj is None:
                return None
            md = getattr(mem_obj, "metadata", {})
            content = None
            try:
                if hasattr(md, "model_extra"):
                    content = md.model_extra.get("full_content")
                elif isinstance(md, dict):
                    content = md.get("full_content")
            except Exception:
                content = None
            return {
                "memory_id": mem_id,
                "content": content,
                "similarity": float(similarity),
                "metadata": md,
                "memory_item": mem_obj,
            }
        except Exception:
            logger.info(f"[v10] failed to load memory {mem_id}", exc_info=True)
            return None

    def _retrieve_v10_holdout(
        self,
        task_description: str,
        k: int,
        threshold: float,
        target_subtask: str,
        filter_source_subtasks: Optional[List[str]] = None,
        filter_source_benchmark: Optional[str] = None,
        eval_mode: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Optional[List]]:
        """v10 holdout retrieval — dispatch to pure_d1 / hybrid / sim_d1.

        Bypasses zero-shot transfer entirely. Uses D1 region quality as prior.
        """
        mode = self.holdout_retrieval_mode
        hide = self._holdout_subtask  # Hide holdout subtask when computing D1
        sim_list: Optional[List] = None  # Returned in tuple, may stay None

        if mode == "pure_d1":
            # All queries get the same top-k anchor memories (no query, no sim)
            anchor_ids = self._v10_get_anchor_ids(hide_subtask=hide, k=k)
            candidates = []
            for mid in anchor_ids:
                c = self._v10_build_candidate_from_mem_id(mid, similarity=0.0)
                if c is not None:
                    candidates.append(c)

        elif mode in ("hybrid", "sim_d1"):
            # Both need sim recall over a large pool, then rerank with D1.
            # Use parent retrieve_query with fetch_k=pool_size to get full
            # candidate pool (parent does sim top-k). Then D1-rerank.
            self._thread_local.target_subtask = target_subtask
            try:
                raw = super().retrieve_query(
                    task_description, self.holdout_pool_size, threshold,
                )
            finally:
                self._thread_local.target_subtask = None
            if isinstance(raw, tuple):
                pool_result, sim_list = raw
            else:
                pool_result, sim_list = raw, None
            if pool_result is None:
                pool_result = {"selected": [], "candidates": [], "simmax": 0.0, "actions": []}
            pool_candidates = pool_result.get("selected", []) or []

            # Source filtering applied on the large pool
            if filter_source_subtasks:
                pool_candidates = self._filter_by_source_subtasks(
                    pool_candidates, filter_source_subtasks,
                )
            if filter_source_benchmark:
                pool_candidates = self._filter_by_source_benchmark(
                    pool_candidates, filter_source_benchmark,
                )

            # Build mem_id -> region_id map (computed lazily once)
            mem_to_region = {}
            for r in self.region_manager.regions:
                for mid in r.member_ids:
                    mem_to_region[mid] = r.region_id

            # Rerank by sim * D1
            def _d1_for(mid: str) -> float:
                rid = mem_to_region.get(mid)
                if rid is None:
                    return 0.5
                return self.region_manager.compute_region_d1_quality(rid, hide_subtask=hide)

            for c in pool_candidates:
                d1 = _d1_for(c["memory_id"])
                c["_v10_d1"] = d1
                c["_v10_score"] = float(c.get("similarity", 0.0)) * d1
            pool_candidates.sort(key=lambda c: c.get("_v10_score", 0.0), reverse=True)

            if mode == "hybrid":
                # Top-N anchors (no sim) + remaining from sim*D1 pool
                n_anchors = min(self.holdout_d1_anchors, k)
                anchor_ids = self._v10_get_anchor_ids(hide_subtask=hide, k=n_anchors)
                candidates = []
                seen_ids = set()
                for mid in anchor_ids:
                    c = self._v10_build_candidate_from_mem_id(mid, similarity=0.0)
                    if c is not None:
                        candidates.append(c)
                        seen_ids.add(mid)
                # Fill remaining slots from sim*D1 pool (dedupe by mem_id)
                for c in pool_candidates:
                    if len(candidates) >= k:
                        break
                    if c["memory_id"] in seen_ids:
                        continue
                    candidates.append(c)
                    seen_ids.add(c["memory_id"])
            else:  # mode == "sim_d1"
                candidates = pool_candidates[:k]
        else:
            # Defensive: caller-side validation should have caught this
            raise ValueError(f"unknown holdout_retrieval_mode {mode!r}")

        # Track target_subtask for auto-consumption by update_values
        if target_subtask:
            self._retrieval_subtask_buffer.append(target_subtask)

        simmax = max((float(c.get("similarity", 0.0)) for c in candidates), default=0.0)
        return {
            "actions": [c["memory_id"] for c in candidates],
            "selected": candidates,
            "candidates": candidates,  # v10 doesn't track pre-rerank pool separately
            "simmax": simmax,
        }, sim_list

    def _build_weighted_subtask_q_cache(
        self, target_weights: List[Tuple[str, float]], *, use_shrinkage: bool
    ) -> Dict[str, float]:
        """Weighted multi-axis Q cache for DB task signatures."""
        if not self.region_manager:
            return self._q_cache_base
        cleaned = [(str(st), max(0.0, float(w))) for st, w in target_weights if st]
        total = sum(w for _, w in cleaned)
        if total <= 0:
            return self._q_cache_base
        cleaned = [(st, w / total) for st, w in cleaned]
        flat: Dict[str, float] = {}
        for mem_id in list(self.region_manager.subtask_q):
            value = 0.0
            for subtask, weight in cleaned:
                q = (
                    self.region_manager.compute_shrinkage_q(mem_id, subtask)
                    if use_shrinkage
                    else self.region_manager.get_subtask_q(mem_id, subtask)
                )
                value += weight * q
            flat[mem_id] = value
        return self._apply_outcome_blend(self._fill_untracked_memories(flat))

    def _build_subtask_q_cache(self, target_subtask: str) -> Dict[str, float]:
        """Build flat Q cache using per-subtask Q[target_subtask] for each memory."""
        if not self.region_manager:
            return self._q_cache_base

        flat = {}
        for mem_id in list(self.region_manager.subtask_q):
            flat[mem_id] = self.region_manager.get_subtask_q(mem_id, target_subtask)

        return self._apply_outcome_blend(self._fill_untracked_memories(flat))

    def _build_shrinkage_q_cache(self, target_subtask: str) -> Dict[str, float]:
        """Build Q cache using James-Stein shrinkage: blend per-memory Q with region utility.

        Cold-start memories (no observations for this subtask) get region utility.
        Warm memories gradually shift to their own per-subtask Q.
        Result is cached per target_subtask (invalidated when region clustering changes).
        Thread-safe via double-checked locking.
        """
        if not self.region_manager:
            return self._q_cache_base

        # Fast path: check cache without lock
        cache_key = (target_subtask, getattr(self.region_manager, '_cluster_version', 0))
        cached = getattr(self, '_shrinkage_q_cache_store', None)
        if cached is not None and cached[0] == cache_key:
            return cached[1]

        # Slow path: lock and double-check
        if not hasattr(self, '_shrinkage_cache_lock'):
            import threading
            self._shrinkage_cache_lock = threading.Lock()
        with self._shrinkage_cache_lock:
            cached = getattr(self, '_shrinkage_q_cache_store', None)
            if cached is not None and cached[0] == cache_key:
                return cached[1]

            flat = {}
            for mem_id in list(self.region_manager.subtask_q):
                flat[mem_id] = self.region_manager.compute_shrinkage_q(mem_id, target_subtask)

            flat = self._fill_untracked_memories(flat)
            result = self._apply_outcome_blend(flat)
            self._shrinkage_q_cache_store = (cache_key, result)
            return result

    def configure_outcome_blend(self, beta) -> None:
        """Single entry point for setting outcome_blend_beta.

        Validates, clamps to [0, 1], and logs. Use this instead of assigning
        memsvc.outcome_blend_beta directly so semantics stay consistent across
        callers (run_bcb_region, run_val_ablation, etc.).
        """
        if beta is None:
            self.outcome_blend_beta = None
            return
        try:
            beta = float(beta)
        except (TypeError, ValueError):
            logger.warning("Invalid outcome_blend_beta=%r, ignoring", beta)
            self.outcome_blend_beta = None
            return
        clamped = max(0.0, min(1.0, beta))
        if clamped != beta:
            logger.warning("outcome_blend_beta=%g clamped to %g", beta, clamped)
        self.outcome_blend_beta = clamped
        logger.info("outcome_blend_beta set to %.3f", clamped)

    def _apply_outcome_blend(self, flat: Dict[str, float]) -> Dict[str, float]:
        """Blend outcome signal into Q values.

        final_q = beta * outcome_q + (1-beta) * q
        where outcome_q = 1.0 for success, 0.0 for failure.

        Memories whose outcome cannot be parsed are SKIPPED (Q left untouched).
        This is intentional: unknown should not be treated as "neutral 0.5",
        because that would pull warm, well-estimated Q values toward 0.5.
        Mutates `flat` in place and returns it.
        """
        beta = getattr(self, 'outcome_blend_beta', None)
        if beta is None or beta <= 0:
            return flat
        # Defensive clamp in case attr was set directly (bypassing configure_outcome_blend).
        try:
            beta = max(0.0, min(1.0, float(beta)))
        except (TypeError, ValueError):
            return flat

        mem_cache = getattr(self, '_mem_cache', None) or {}
        for mem_id in flat:
            mem_obj = mem_cache.get(mem_id)
            if mem_obj is None:
                continue
            meta = getattr(mem_obj, 'metadata', None)
            if meta is None and isinstance(mem_obj, dict):
                meta = mem_obj.get('metadata')
            outcome = None
            if meta is not None:
                if hasattr(meta, 'model_extra'):
                    outcome = (getattr(meta, 'model_extra', {}) or {}).get('outcome')
                elif isinstance(meta, dict):
                    outcome = meta.get('outcome')
            if outcome is None:
                continue
            if isinstance(outcome, bool):
                outcome_q = 1.0 if outcome else 0.0
            else:
                s = str(outcome).strip().lower()
                if s in ('success', 'true', '1'):
                    outcome_q = 1.0
                elif s in ('failure', 'fail', 'false', '0'):
                    outcome_q = 0.0
                else:
                    continue
            flat[mem_id] = beta * outcome_q + (1 - beta) * flat[mem_id]

        return flat

    def _build_global_q_for_benchmark(self, target_benchmark: str) -> Dict[str, float]:
        """Build flat Q cache from global Q for a specific target benchmark."""
        flat = {}
        for mem_id, q_dict in self._global_q_cache.items():
            flat[mem_id] = q_dict.get(target_benchmark, 0.5)

        # Ensure all memories have an entry (same 0.5 neutral prior fix)
        for mem_id, q_val in self._q_cache.items():
            if mem_id not in flat:
                flat[mem_id] = q_val
        if hasattr(self, 'dict_memory') and self.dict_memory:
            for query_key, mem_ids in self.dict_memory.items():
                for mid in mem_ids:
                    if mid not in flat:
                        flat[mid] = 0.5

        return flat

    def _get_mem_embedding(self, mem_id: str):
        """Lookup embedding vector for a memory by its ID.

        Maintains a reverse index mem_id→query_key. Rebuilds in bulk
        on first miss to avoid repeated full scans of dict_memory.
        """
        import numpy as np

        if not hasattr(self, '_mem_id_to_query_key'):
            self._mem_id_to_query_key = {}
            self._rev_map_built = False
        rev = self._mem_id_to_query_key

        if mem_id in rev:
            qk = rev[mem_id]
            emb = getattr(self, 'query_embeddings', {}).get(qk)
            if emb is not None:
                return np.array(emb) if not isinstance(emb, np.ndarray) else emb
            return None

        # Bulk rebuild reverse map once on first miss
        if not self._rev_map_built:
            dm = getattr(self, 'dict_memory', {})
            for qk, mids in dm.items():
                for mid in mids:
                    if mid not in rev:
                        rev[mid] = qk
            self._rev_map_built = True

        qk = rev.get(mem_id)
        if qk is None:
            return None
        emb = getattr(self, 'query_embeddings', {}).get(qk)
        if emb is not None:
            return np.array(emb) if not isinstance(emb, np.ndarray) else emb
        return None

    def _get_mem_content_and_success(self, mem_id: str):
        """Callback for RegionManager._build_region_failure_summaries.

        Returns (content: str, success: bool|None) for a given memory ID.
        Reads from _mem_cache (Pydantic MemoryItem objects).
        """
        mc = getattr(self, '_mem_cache', None)
        if mc is None:
            raise LookupError(f"_mem_cache not available for {mem_id}")
        mem_obj = mc.get(mem_id)
        if mem_obj is None:
            raise LookupError(f"{mem_id} not in _mem_cache")
        md = getattr(mem_obj, 'metadata', {})
        content = None
        success = None
        if hasattr(md, 'model_extra'):
            content = md.model_extra.get('full_content')
            success = md.model_extra.get('success')
        elif isinstance(md, dict):
            content = md.get('full_content')
            success = md.get('success')
        return content, success

    @staticmethod
    def _audit_metadata(md):
        if isinstance(md, dict):
            return md
        extra = getattr(md, "model_extra", None)
        if isinstance(extra, dict):
            return extra
        try:
            dumped = md.model_dump()
            return dumped if isinstance(dumped, dict) else {}
        except Exception:
            return {}

    def _audit_region_id(self, mem_id):
        weights = getattr(self.region_manager, "membership_weights", {}).get(mem_id) if self.region_manager else None
        if weights is None:
            return None
        try:
            values = list(weights)
            return int(max(range(len(values)), key=lambda i: values[i])) if values else None
        except Exception:
            return None

    def append_retrieval_audit(self, task_description, result_payload, processed_mems):
        if not self.retrieval_audit_path:
            return
        try:
            def summarize(mem, bucket=None):
                md = self._audit_metadata(mem.get("metadata"))
                content = mem.get("content") or ""
                item = {
                    "memory_id": mem.get("memory_id"),
                    "task_id": md.get("task_id", md.get("sample_index")),
                    "region_id": self._audit_region_id(mem.get("memory_id")),
                    "content_len": len(content),
                    "content_hash": hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:16],
                }
                if bucket is None:
                    item.update({
                        "similarity": float(mem.get("similarity", 0.0) or 0.0),
                        "q_estimate": float(mem.get("q_estimate", 0.0) or 0.0),
                        "score": float(mem.get("score", 0.0) or 0.0),
                        "success": md.get("success"),
                    })
                else:
                    item.update({"bucket": str(bucket), "region_failure_summary": bool(mem.get("_region_failure_summary"))})
                return item
            raw = [summarize(x) for x in list((result_payload or {}).get("selected", []) or [])]
            injected = [summarize(x, bucket) for bucket, values in (processed_mems or {}).items() for x in values]
            record = {
                "recorded_at": datetime.utcnow().isoformat() + "Z",
                **dict((result_payload or {}).get("audit_context", {}) or {}),
                "query": task_description,
                "query_hash": hashlib.sha256(task_description.encode("utf-8", errors="replace")).hexdigest()[:16],
                "target_subtask_weights": (result_payload or {}).get("target_subtask_weights", []),
                "routing": (result_payload or {}).get("region_routing", {}),
                "selected_raw": raw,
                "injected_final": injected,
                "raw_unique_tasks": len({str(x["task_id"]) for x in raw if x["task_id"] is not None}),
                "raw_unique_content": len({x["content_hash"] for x in raw}),
                "injected_unique_tasks": len({str(x["task_id"]) for x in injected if x["task_id"] is not None}),
                "injected_unique_content": len({x["content_hash"] for x in injected}),
            }
            path = os.path.abspath(self.retrieval_audit_path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            with self._retrieval_audit_lock:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(line); f.flush(); os.fsync(f.fileno())
        except Exception:
            logger.warning("Failed to append retrieval audit", exc_info=True)

    def invalidate_embedding_cache(self):
        """Call when dict_memory is modified to force reverse map rebuild."""
        self._rev_map_built = False

    def _apply_utility_anchor(
        self,
        sim_ranked: List[Dict[str, Any]],
        target_subtask: str,
        eval_mode: Optional[str],
        k: int,
        candidate_pool: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """v5.5 utility_anchor: anchor from best region's members + sim refinement.

        Strategy (inspired by v10 H, but for known target):
          1. Anchor (utility-first): pick top `utility_anchor_count` mem_ids from
             best `utility_anchor_topk_regions` regions by region.utility_by_subtask[target].
             Anchors are ranked by per-memory Q (region_manager.get_subtask_q).
          2. Refinement (sim-based): fill remaining slots from sim_ranked, dedup against anchors.

        Key difference from v5 quota_*: anchors come DIRECTLY from region.member_ids,
        NOT filtered through sim-recalled narrow pool. Solves 929226 quota's
        noop_no_member=26% by decoupling anchor recall from sim recall.

        Args:
            sim_ranked: candidates already ranked by parent's sim+Q score
            target_subtask: e.g. "alf/pick_and_place_simple" (known in main exp)
            eval_mode: "train"/"valid"/"test" or None
            k: final number of memories to return

        Returns:
            anchors (top utility_anchor_count) + sim refinement (k - n_anchors).
            Falls back to sim_ranked if no top regions available.
        """
        # Stats
        if not hasattr(self, "_anchor_stats"):
            self._anchor_stats = {
                "calls": 0, "noop_no_top_regions": 0,
                "applied": 0, "sum_anchors": 0, "sum_sim_fill": 0,
            }
        with self._quota_stats_lock:
            self._anchor_stats["calls"] += 1

        if not self.region_manager or not target_subtask:
            return sim_ranked

        anchor_cnt = min(self.utility_anchor_count, k)
        if anchor_cnt <= 0:
            return sim_ranked

        calibrated = os.environ.get("MEMRL_UTILITY_ANCHOR_CALIBRATED", "0").strip().lower() not in {"0", "false", "no"}
        if calibrated:
            pool = list(candidate_pool or sim_ranked)
            observed_min = max(0.0, float(os.environ.get("MEMRL_UTILITY_ANCHOR_OBS_MIN", "10") or "10"))
            margin_min = max(0.0, float(os.environ.get("MEMRL_UTILITY_ANCHOR_MARGIN", "0.01") or "0.01"))
            sim_floor = max(0.0, float(os.environ.get("MEMRL_UTILITY_ANCHOR_SIM_FLOOR", "0.50") or "0.50"))
            confidence_scale = max(0.0, float(os.environ.get("MEMRL_UTILITY_ANCHOR_CONF_SCALE", "0.50") or "0.50"))
            ranked_regions = []
            for region in list(self.region_manager.regions):
                observed_n = float((region.total_count_by_subtask or {}).get(target_subtask, 0.0) or 0.0)
                if observed_n < observed_min:
                    continue
                utility = float((region.utility_by_subtask or {}).get(target_subtask, 0.5))
                conservative = utility - confidence_scale / max((observed_n + 1.0) ** 0.5, 1.0)
                ranked_regions.append((conservative, utility, observed_n, region))
            ranked_regions.sort(key=lambda row: (-row[0], -row[2], row[3].region_id))
            if not ranked_regions:
                return sim_ranked
            if len(ranked_regions) > 1 and ranked_regions[0][0] - ranked_regions[1][0] < margin_min:
                logger.info(
                    "[CALIBRATED ANCHOR] abstain low margin: subtask=%s top=%.4f second=%.4f",
                    target_subtask, ranked_regions[0][0], ranked_regions[1][0],
                )
                return sim_ranked
            best_score, best_utility, best_n, best_region = ranked_regions[0]
            member_set = set(best_region.member_ids)
            top5_sims = sorted((float(c.get("similarity", 0.0) or 0.0) for c in sim_ranked), reverse=True)
            tail_sim = top5_sims[min(len(top5_sims), k) - 1] if top5_sims else sim_floor
            dynamic_floor = max(sim_floor, 0.90 * tail_sim)
            eligible = []
            for cand in pool:
                mid = cand.get("memory_id")
                if not mid or mid not in member_set:
                    continue
                sim = float(cand.get("similarity", 0.0) or 0.0)
                obs = float(self.region_manager.get_observation_count(mid, target_subtask))
                if sim < dynamic_floor or obs <= 0:
                    continue
                eligible.append((float(cand.get("score", 0.0) or 0.0), sim, obs, cand))
            eligible.sort(key=lambda row: (-row[0], -row[1], -row[2]))
            anchors = [row[3] for row in eligible[:anchor_cnt]]
            if not anchors:
                logger.info(
                    "[CALIBRATED ANCHOR] abstain no relevant observed members: subtask=%s region=%s obs=%.2f floor=%.3f",
                    target_subtask, best_region.region_id, best_n, dynamic_floor,
                )
                return sim_ranked
            anchor_ids = {c.get("memory_id") for c in anchors}
            fill = [c for c in sim_ranked if c.get("memory_id") not in anchor_ids]
            final = (anchors + fill)[:k]
            logger.info(
                "[CALIBRATED ANCHOR] applied subtask=%s region=%s conservative=%.4f utility=%.4f obs=%.2f anchors=%d floor=%.3f",
                target_subtask, best_region.region_id, best_score, best_utility, best_n, len(anchors), dynamic_floor,
            )
            return final

        # 1. Get top utility regions for target_subtask
        top_regions = self.region_manager.top_regions_for_subtask(
            target_subtask,
            top_n=max(self.utility_anchor_topk_regions, 1),
            min_count=self.utility_anchor_min_count,
        )
        if not top_regions:
            with self._quota_stats_lock:
                self._anchor_stats["noop_no_top_regions"] += 1
            return sim_ranked

        # 2. Collect candidate mem_ids from these regions
        # Ranked by per-memory Q (region_manager.get_subtask_q)
        region_mem_pool: List[Tuple[float, str]] = []  # (Q, mem_id) for ranking
        seen_mems_in_anchor: set = set()
        for region in top_regions:
            for mid in region.member_ids:
                if mid in seen_mems_in_anchor:
                    continue
                # Use per-memory Q for this subtask (high Q = good fit)
                q = self.region_manager.get_subtask_q(mid, target_subtask)
                region_mem_pool.append((float(q), mid))
                seen_mems_in_anchor.add(mid)

        if not region_mem_pool:
            with self._quota_stats_lock:
                self._anchor_stats["noop_no_top_regions"] += 1
            return sim_ranked

        # Sort by per-memory Q desc, take top anchor_cnt
        region_mem_pool.sort(key=lambda x: -x[0])

        # 3. Build anchor candidates (need to load actual memory objects)
        # Use sim from sim_ranked if mem_id appears there (warm-cache), else 0.0
        sim_map: Dict[str, float] = {
            c.get("memory_id"): c.get("similarity", 0.0) for c in sim_ranked
            if c.get("memory_id")
        }
        anchors: List[Dict[str, Any]] = []
        anchor_ids_used: set = set()
        for q, mid in region_mem_pool:
            if len(anchors) >= anchor_cnt:
                break
            # Reuse sim from sim_ranked if available (avoid re-fetching), else 0
            sim_val = sim_map.get(mid, 0.0)
            cand = self._v10_build_candidate_from_mem_id(mid, similarity=sim_val)
            if cand is not None:
                anchors.append(cand)
                anchor_ids_used.add(mid)

        if not anchors:
            with self._quota_stats_lock:
                self._anchor_stats["noop_no_top_regions"] += 1
            return sim_ranked

        # 4. Fill remaining slots with sim_ranked, dedup
        sim_fill: List[Dict[str, Any]] = []
        for cand in sim_ranked:
            mid = cand.get("memory_id")
            if mid in anchor_ids_used:
                continue
            sim_fill.append(cand)
            anchor_ids_used.add(mid)
            if len(anchors) + len(sim_fill) >= k:
                break

        final = anchors + sim_fill

        with self._quota_stats_lock:
            self._anchor_stats["applied"] += 1
            self._anchor_stats["sum_anchors"] += len(anchors)
            self._anchor_stats["sum_sim_fill"] += len(sim_fill)
            should_log = self._anchor_stats["calls"] % 200 == 0
            snapshot = dict(self._anchor_stats) if should_log else None
        if should_log and snapshot:
            s = snapshot
            n = max(s["calls"], 1)
            logger.info(
                "[ANCHOR STATS] calls=%d apply_rate=%.2f noop_no_region=%.2f "
                "avg_anchors=%.2f avg_sim_fill=%.2f",
                s["calls"], s["applied"] / n, s["noop_no_top_regions"] / n,
                s["sum_anchors"] / max(s["applied"], 1),
                s["sum_sim_fill"] / max(s["applied"], 1),
            )
        return final

    def _apply_weighted_region_quota(
        self, global_ranked, candidate_pool, weighted_targets, k,
    ):
        if not hasattr(self, "_weighted_quota_stats"):
            self._weighted_quota_stats = {"calls": 0, "applied": 0, "abstain_margin": 0, "no_pick": 0, "picks": 0}
        ranked = rank_regions_weighted(
            self.region_manager.regions, weighted_targets, self.weighted_quota_min_count
        )
        with self._quota_stats_lock:
            self._weighted_quota_stats["calls"] += 1
        if not ranked:
            return global_ranked, {"applied": False, "reason": "no_eligible_region", "region_scores": []}
        top_score, _top_count, top_region = ranked[0]
        second_score = ranked[1][0] if len(ranked) > 1 else 0.0
        margin = top_score - second_score
        if len(ranked) > 1 and margin < self.weighted_quota_margin:
            with self._quota_stats_lock:
                self._weighted_quota_stats["abstain_margin"] += 1
            return global_ranked, {"applied": False, "reason": "utility_margin", "top_region_id": int(top_region.region_id), "margin": float(margin), "region_scores": [{"region_id": int(r.region_id), "utility": float(sc), "weighted_count": float(ct)} for sc, ct, r in ranked]}
        final, picks = apply_region_quota(
            global_ranked, candidate_pool, top_region.member_ids,
            quota=self.weighted_quota_count, sim_floor=self.weighted_quota_min_sim, k=k,
        )
        with self._quota_stats_lock:
            if picks:
                self._weighted_quota_stats["applied"] += 1
                self._weighted_quota_stats["picks"] += len(picks)
            else:
                self._weighted_quota_stats["no_pick"] += 1
            calls = self._weighted_quota_stats["calls"]
            stats = dict(self._weighted_quota_stats)
        if calls == 1 or calls % 100 == 0:
            logger.info(
                "[WEIGHTED REGION QUOTA] calls=%d applied=%d abstain_margin=%d no_pick=%d "
                "avg_picks=%.2f top_region=%d utility=%.4f margin=%.4f",
                calls, stats["applied"], stats["abstain_margin"], stats["no_pick"],
                stats["picks"] / max(1, stats["applied"]), top_region.region_id, top_score, margin,
            )
        return final, {"applied": bool(picks), "reason": "applied" if picks else "no_similar_region_candidate", "top_region_id": int(top_region.region_id), "margin": float(margin), "quota_picks": len(picks), "region_scores": [{"region_id": int(r.region_id), "utility": float(sc), "weighted_count": float(ct)} for sc, ct, r in ranked]}

    def _apply_quota_recall(
        self,
        global_ranked: List[Dict[str, Any]],
        all_candidates: List[Dict[str, Any]],
        target_subtask: str,
        eval_mode: Optional[str],
        k: int,
    ) -> List[Dict[str, Any]]:
        """v5 quota-based candidate reshuffling.

        Strategy: take top-N regions for target_subtask, find their members in
        the existing candidate pool (all_candidates), and reserve quota slots
        for them in the final top-k. Members not in pool are NOT injected here
        (to avoid breaking the Q-cache assumption of parent retrieval); future
        work can add a separate region sim recall pass.

        Args:
            global_ranked: candidates already ranked by parent's sim+Q score
                          (post source filtering, post region gating)
            all_candidates: full enriched candidate set from parent.retrieve_query
                          (typically larger than global_ranked when k limits selection)
            target_subtask: e.g. "alf/pick_and_place_simple"
            eval_mode: "train"/"valid"/"test" or None
            k: final number of memories to return

        Returns:
            Reordered candidates list with region quota satisfied; first
            quota_actual entries are region-promoted top-1-per-region; rest
            are global-ranked fill (deduplicated).
        """
        # Counters (incremented across calls for monitoring)
        # Thread-safe: protected by self._quota_stats_lock (codex review #1 fix)
        with self._quota_stats_lock:
            if not hasattr(self, "_quota_stats"):
                self._quota_stats = {
                    "calls": 0, "noop_no_top_regions": 0,
                    "noop_no_member_in_pool": 0, "applied": 0,
                    "sum_region_picks": 0, "sum_top_regions": 0,
                }
            self._quota_stats["calls"] += 1

        if not self.region_manager or not target_subtask:
            return global_ranked

        # Cap quota by k (defensive)
        quota_cap = min(self.quota_max, k)

        # 1. Get top-N regions for this subtask. In adaptive mode we need ≥4 for
        # Gate B (utility margin); in fixed mode quota_cap is enough.
        top_n_to_fetch = max(quota_cap, 4) if self.retrieve_mode == "quota_adaptive" else quota_cap
        top_regions = self.region_manager.top_regions_for_subtask(
            target_subtask,
            top_n=top_n_to_fetch,
            min_count=self.quota_region_min_count,
        )
        if not top_regions:
            with self._quota_stats_lock:
                self._quota_stats["noop_no_top_regions"] += 1
            return global_ranked

        # 2. Build mem_id -> region_idx for the top quota_cap regions
        # (only first quota_cap regions can claim quota slots; >quota_cap used only for Gate B)
        mem_to_region_idx: Dict[str, int] = {}
        for i in range(min(quota_cap, len(top_regions))):
            region = top_regions[i]
            for mid in region.member_ids:
                if mid not in mem_to_region_idx:
                    mem_to_region_idx[mid] = i

        # 3. Determine effective quota (adaptive gates if enabled)
        if self.retrieve_mode == "quota_adaptive":
            quota_max = self._compute_adaptive_quota(
                top_regions=top_regions,
                target_subtask=target_subtask,
                global_ranked=global_ranked,
                eval_mode=eval_mode,
            )
            quota_max = min(quota_max, quota_cap)
        else:
            quota_max = quota_cap

        if quota_max <= 0:
            return global_ranked

        # 4. Pick top-1 candidate per region from all_candidates (sorted by score)
        # Apply min-sim floor in BOTH fixed and adaptive (codex review #4 fix:
        # fixed mode also needs floor to prevent injecting low-quality memories).
        sim_floor = self.quota_min_sim_floor
        region_picks: List[Dict[str, Any]] = []
        seen_regions: set = set()
        seen_mems: set = set()

        for cand in all_candidates:
            mid = cand.get("memory_id")
            if not mid or mid in seen_mems:
                continue
            region_idx = mem_to_region_idx.get(mid)
            if region_idx is None or region_idx in seen_regions:
                continue
            if cand.get("similarity", 0.0) < sim_floor:
                continue
            region_picks.append(cand)
            seen_regions.add(region_idx)
            seen_mems.add(mid)
            if len(region_picks) >= quota_max:
                break

        if not region_picks:
            with self._quota_stats_lock:
                self._quota_stats["noop_no_member_in_pool"] += 1
            return global_ranked

        # 5. Fill rest with global_ranked, deduplicated
        global_fill: List[Dict[str, Any]] = []
        for cand in global_ranked:
            mid = cand.get("memory_id")
            if mid in seen_mems:
                continue
            global_fill.append(cand)
            seen_mems.add(mid)
            if len(region_picks) + len(global_fill) >= k:
                break

        final = region_picks + global_fill

        # Stats + occasional log (every 200 calls to avoid spam)
        # Thread-safe: snapshot under lock, then log outside (codex review #1 fix)
        with self._quota_stats_lock:
            self._quota_stats["applied"] += 1
            self._quota_stats["sum_region_picks"] += len(region_picks)
            self._quota_stats["sum_top_regions"] += len(top_regions)
            should_log = self._quota_stats["calls"] % 200 == 0
            stats_snapshot = dict(self._quota_stats) if should_log else None
        if should_log and stats_snapshot:
            s = stats_snapshot
            n = max(s["calls"], 1)
            logger.info(
                "[QUOTA STATS] calls=%d apply_rate=%.2f noop_no_region=%.2f "
                "noop_no_member=%.2f avg_picks=%.2f avg_top_regions=%.2f",
                s["calls"], s["applied"] / n, s["noop_no_top_regions"] / n,
                s["noop_no_member_in_pool"] / n,
                s["sum_region_picks"] / max(s["applied"], 1),
                s["sum_top_regions"] / max(s["applied"], 1),
            )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[QUOTA RECALL] mode=%s target=%s top_regions=%d quota_eff=%d "
                "region_picks=%d global_fill=%d final=%d/%d",
                self.retrieve_mode, target_subtask.split('/')[-1],
                len(top_regions), quota_max,
                len(region_picks), len(global_fill), len(final), k,
            )
        return final

    def _compute_adaptive_quota(
        self,
        top_regions: List[Any],
        target_subtask: str,
        global_ranked: List[Dict[str, Any]],
        eval_mode: Optional[str],
    ) -> int:
        """Adaptive quota with 4 safety gates (subtask_conf, utility_margin, OOD).

        Returns effective quota_max ∈ [0, self.quota_max].
        """
        quota = self.quota_max

        # Gate A: subtask confidence (ALFWorld uses oracle, default to 1.0
        # unless we add classifier; OOD eval may want to inject doubt via flag)
        subtask_conf = getattr(self, "_subtask_confidence_override", 1.0)
        thresholds = sorted(self.quota_subtask_conf_thresholds)
        if subtask_conf < thresholds[0]:
            return 0  # fully abstain from region quota
        elif subtask_conf < thresholds[1]:
            quota = min(quota, 1)
        elif subtask_conf < thresholds[2]:
            quota = min(quota, 2)
        # else quota stays at self.quota_max

        # Gate B: utility margin (need top-1 to be meaningfully better than top-4)
        if len(top_regions) >= 4:
            u_top1 = top_regions[0].utility_by_subtask.get(target_subtask, 0.0)
            u_top4 = top_regions[3].utility_by_subtask.get(target_subtask, 0.0)
            if (u_top1 - u_top4) < self.quota_utility_margin:
                quota = min(quota, 1)

        # Gate C: OOD guard (query embedding far from best region centroid)
        # Skipped for now: would require passing query_vec into this function;
        # MVP keeps it out, can add in v5.1 if needed.

        return max(0, quota)

    def _inject_exploration(
        self,
        selected: List[Dict[str, Any]],
        full_pool: List[Dict[str, Any]],
        n_explore: int = 2,
        success_ratio: float = 0.7,
    ) -> None:
        """Replace bottom n_explore slots with UCB-scored exploration candidates.

        Prioritizes under-observed memories via UCB bonus.
        success_ratio controls fraction of explore slots filled with
        success memories (to avoid flooding prompt with failure reflections).
        """
        import random
        import math

        selected_ids = {c.get("memory_id") for c in selected}
        explore_candidates = [
            c for c in full_pool
            if c.get("memory_id") and c["memory_id"] not in selected_ids
        ]
        if not explore_candidates or n_explore <= 0:
            return

        # Split into success and failure pools
        success_pool = []
        failure_pool = []
        for c in explore_candidates:
            meta = c.get("metadata")
            outcome = None
            success_value = None
            if meta:
                if hasattr(meta, "model_extra"):
                    extra = getattr(meta, "model_extra", {}) or {}
                    outcome = extra.get("outcome")
                    success_value = extra.get("success")
                elif isinstance(meta, dict):
                    outcome = meta.get("outcome")
                    success_value = meta.get("success")
            if outcome not in (None, ""):
                is_success = str(outcome).strip().lower() in {"success", "true", "1"}
            else:
                # HLE/BCB memories store a boolean `success`; ALFWorld legacy
                # memories may store string `outcome`. Support both schemas.
                is_success = (
                    bool(success_value) if isinstance(success_value, bool)
                    else str(success_value).strip().lower() in {"success", "true", "1"}
                )
            if is_success:
                success_pool.append(c)
            else:
                failure_pool.append(c)

        # Score by UCB (prioritize under-observed)
        rm = self.region_manager
        total_updates = rm._global_reward_count if rm else 1

        def ucb_score(c):
            mid = c.get("memory_id", "")
            if rm and mid in rm.subtask_q_counts:
                n_obs = sum(rm.subtask_q_counts[mid].values())
            else:
                n_obs = 0
            return math.sqrt(math.log(max(total_updates, 2)) / max(n_obs, 1))

        # Determine how many success vs failure explore slots
        n_success = max(1, round(n_explore * success_ratio))
        n_failure = n_explore - n_success

        picks = []
        # Pick from success pool (UCB sorted, sample from top-3x)
        if success_pool and n_success > 0:
            success_pool.sort(key=ucb_score, reverse=True)
            top = success_pool[:n_success * 3]
            picks.extend(random.sample(top, min(n_success, len(top))))
        # Pick from failure pool
        if failure_pool and n_failure > 0:
            failure_pool.sort(key=ucb_score, reverse=True)
            top = failure_pool[:n_failure * 3]
            picks.extend(random.sample(top, min(n_failure, len(top))))
        # If not enough from one pool, fill from the other
        while len(picks) < n_explore:
            remaining = [c for c in explore_candidates if c not in picks]
            if not remaining:
                break
            remaining.sort(key=ucb_score, reverse=True)
            picks.append(remaining[0])

        for i, pick in enumerate(picks):
            idx = len(selected) - 1 - i
            if idx >= 0:
                selected[idx] = pick

    def _filter_by_source_subtasks(
        self, candidates: List[Dict[str, Any]], allowed_subtasks: List[str]
    ) -> List[Dict[str, Any]]:
        allowed_set = set(allowed_subtasks)
        filtered = []
        for c in candidates:
            meta = c.get("metadata", {})
            if hasattr(meta, "model_extra"):
                source_subtask = meta.model_extra.get("source_subtask")
            elif isinstance(meta, dict):
                source_subtask = meta.get("source_subtask")
            else:
                source_subtask = None
            if source_subtask in allowed_set:
                filtered.append(c)
        return filtered

    def _filter_by_source_benchmark(
        self, candidates: List[Dict[str, Any]], allowed_benchmark: str
    ) -> List[Dict[str, Any]]:
        filtered = []
        for c in candidates:
            meta = c.get("metadata", {})
            if hasattr(meta, "model_extra"):
                source_benchmark = meta.model_extra.get("source_benchmark")
            elif isinstance(meta, dict):
                source_benchmark = meta.get("source_benchmark")
            else:
                source_benchmark = None
            if source_benchmark == allowed_benchmark:
                filtered.append(c)
        return filtered

    def _apply_region_gating(
        self,
        candidates: List[Dict[str, Any]],
        target_subtask: str,
        mode: str,
        allowed_sources: Any,
    ) -> List[Dict[str, Any]]:
        """Apply region gating as Q-branch adjustment (not full-score multiply).

        Gates only the Q component to avoid suppressing semantically relevant
        candidates. Region score is centered around batch mean so it can
        both boost and penalize.

        new_score = old_score + q_z * w_q * (region_score - mean_region_score)
        """
        w_q = getattr(self, 'weight_q', 0.5)

        # First pass: compute region scores and batch mean
        region_scores = {}
        for c in candidates:
            mem_id = c.get("memory_id")
            if not mem_id:
                continue
            try:
                rs = self.region_manager.compute_region_gating_score(
                    mem_id, target_subtask, mode, allowed_sources
                )
            except Exception:
                rs = 0.5
            region_scores[mem_id] = rs

        if not region_scores:
            return candidates

        mean_rs = sum(region_scores.values()) / len(region_scores)

        # Second pass: adjust scores
        for c in candidates:
            mem_id = c.get("memory_id")
            rs = region_scores.get(mem_id, mean_rs)
            old_score = c.get("score", 0.0)
            q_z = c.get("q_z", 0.0)
            c["score"] = old_score + q_z * w_q * (rs - mean_rs)
            c["region_gating_score"] = rs

        candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return candidates

    def update_values(
        self,
        rewards: List[float],
        retrieved_memory_ids_list: List[List[str]],
        target_subtasks: Optional[List[str]] = None,
        target_subtask_weights: Optional[List[List[Tuple[str, float]]]] = None,
    ) -> Dict[str, Optional[float]]:
        """
        Update all Q values:
        1. Parent's scalar Q (MemRL baseline compatibility)
        2. Per-subtask Q via region_manager
        3. Global Q (per-benchmark, for inter-transfer)
        """
        # 1. Parent scalar Q update (±1 reward internally)
        result = super().update_values(rewards, retrieved_memory_ids_list)

        # Prefer explicitly passed target_subtasks; fall back to buffer
        if target_subtasks is None and self._retrieval_subtask_buffer:
            target_subtasks = list(self._retrieval_subtask_buffer)
        self._retrieval_subtask_buffer.clear()

        # 2. Per-subtask Q + region utility update (0/1 reward)
        if self.region_manager and target_subtasks:
            n = len(rewards)
            if len(target_subtasks) != n or len(retrieved_memory_ids_list) != n:
                logger.error(
                    "update_values: length mismatch (rewards=%d, ids=%d, subtasks=%d), "
                    "skipping region Q update to avoid misaligned supervision",
                    n, len(retrieved_memory_ids_list), len(target_subtasks),
                )
            else:
                non_empty = sum(1 for ids in retrieved_memory_ids_list if ids)
                logger.info(
                    "update_values: %d rewards, %d non-empty retrieved_ids, %d target_subtasks",
                    n, non_empty, len(target_subtasks),
                )
                for idx, (reward, mem_ids, target_subtask) in enumerate(
                    zip(rewards, retrieved_memory_ids_list, target_subtasks)
                ):
                    if not mem_ids:
                        continue
                    weighted = (
                        target_subtask_weights[idx]
                        if target_subtask_weights is not None and idx < len(target_subtask_weights)
                        else None
                    )
                    if weighted:
                        cleaned = [(str(st), max(0.0, float(w))) for st, w in weighted if st]
                        total_weight = sum(w for _, w in cleaned)
                        if total_weight > 0:
                            for subtask, weight in cleaned:
                                self.region_manager.update_subtask_q(
                                    mem_ids, subtask, reward,
                                    evidence_weight=weight / total_weight,
                                )
                    elif target_subtask:
                        self.region_manager.update_subtask_q(mem_ids, target_subtask, reward)

        # 3. Global Q update (per-benchmark, 0/1 reward)
        if target_subtasks:
            alpha = getattr(self.rl_config, 'alpha', 0.1) if hasattr(self, 'rl_config') and self.rl_config else 0.1
            n = len(rewards)
            if len(target_subtasks) != n:
                pass  # Already logged above; skip global Q too
            else:
                for reward, mem_ids, target_subtask in zip(rewards, retrieved_memory_ids_list, target_subtasks):
                    if not target_subtask or not mem_ids:
                        continue
                    benchmark = target_subtask.split("/", 1)[0] if "/" in target_subtask else target_subtask
                    for mem_id in mem_ids:
                        if mem_id not in self._global_q_cache:
                            if len(self._global_q_cache) >= self._global_q_cache_max_size:
                                oldest = next(iter(self._global_q_cache))
                                self._global_q_cache.pop(oldest, None)
                            self._global_q_cache[mem_id] = {}
                        q_dict = self._global_q_cache[mem_id]
                        old_q = q_dict.get(benchmark, 0.5)
                        new_q = old_q + alpha * (reward - old_q)
                        q_dict[benchmark] = new_q

        return result

    def get_global_q(self, mem_id: str, target_benchmark: str) -> float:
        """Get global Q value for a memory on a specific target benchmark."""
        q_dict = self._global_q_cache.get(mem_id)
        if q_dict is None:
            return 0.5
        return q_dict.get(target_benchmark, 0.5)

    # ---------- Checkpoint Persistence ----------

    def _persist_local_caches(self, snapshot_root: str) -> None:
        """Extend parent to also save region state and global Q."""
        import json
        import os

        super()._persist_local_caches(snapshot_root)

        cache_dir = os.path.join(snapshot_root, "local_cache")
        os.makedirs(cache_dir, exist_ok=True)

        # Save region_manager state (subtask_q, regions, membership, UCB)
        if self.region_manager:
            try:
                rm_path = os.path.join(cache_dir, "region_manager.json")
                self.region_manager.save(rm_path)
                logger.info("Persisted region_manager to %s", rm_path)
            except Exception:
                logger.warning("Failed to persist region_manager", exc_info=True)

        # Save global Q cache (for inter-transfer)
        if self._global_q_cache:
            try:
                gq_path = os.path.join(cache_dir, "global_q_cache.json")
                with open(gq_path, "w", encoding="utf-8") as f:
                    json.dump(self._global_q_cache, f, ensure_ascii=False)
                logger.info("Persisted global_q_cache (%d entries) to %s",
                            len(self._global_q_cache), gq_path)
            except Exception:
                logger.warning("Failed to persist global_q_cache", exc_info=True)

    @staticmethod
    def _region_metadata_dict(mem_obj) -> Dict[str, Any]:
        if isinstance(mem_obj, dict):
            md = mem_obj.get("metadata", {}) or {}
        else:
            md = getattr(mem_obj, "metadata", {}) or {}
            extra = getattr(md, "model_extra", None)
            if isinstance(extra, dict):
                md = extra
        return md if isinstance(md, dict) else {}

    def register_region_memory_from_metadata(self, mem_id: str, metadata=None) -> bool:
        """Register a stored memory in Region geometry, without outcome evidence."""
        if not self.region_manager or not mem_id or not self.region_register_on_create:
            return False
        if metadata is None:
            mem_obj = (getattr(self, "_mem_cache", {}) or {}).get(str(mem_id))
            metadata = self._region_metadata_dict(mem_obj)
        elif not isinstance(metadata, dict):
            metadata = self._region_metadata_dict(type("M", (), {"metadata": metadata})())
        source_subtask = str((metadata or {}).get("source_subtask") or "")
        if not source_subtask:
            return False
        initial_q = (metadata or {}).get("q_value", 0.0)
        return self.region_manager.register_memory(
            str(mem_id), source_subtask, initial_q, assign_if_clustered=True
        )

    def backfill_missing_region_memories(self) -> Dict[str, int]:
        """Backfill legacy/newly-written cache entries missing Region coordinates."""
        stats = {"scanned": 0, "registered": 0, "assigned": 0, "skipped": 0}
        cache = getattr(self, "_mem_cache", {}) or {}
        for mem_id, mem_obj in cache.items():
            stats["scanned"] += 1
            before_member = str(mem_id) in self.region_manager.membership_weights
            md = self._region_metadata_dict(mem_obj)
            if self.register_region_memory_from_metadata(str(mem_id), md):
                stats["registered"] += 1
            elif str(mem_id) not in self.region_manager.subtask_q:
                stats["skipped"] += 1
            if not before_member and str(mem_id) in self.region_manager.membership_weights:
                stats["assigned"] += 1
        logger.info("[Region Backfill] %s", stats)
        return stats

    def _restore_local_caches(self, cache_dir: str) -> bool:
        """Extend parent to also restore region state and global Q."""
        import json
        import os

        restored = super()._restore_local_caches(cache_dir)

        # Re-wrap _q_cache with thread-local proxy (parent's restore replaces it with a plain dict)
        if not isinstance(self._q_cache, _ThreadLocalQCache):
            self._q_cache_base = self._q_cache
            self._q_cache = _ThreadLocalQCache(self._q_cache_base, self._thread_local)

        # Restore region_manager state
        rm_path = os.path.join(cache_dir, "region_manager.json")
        if self.region_manager and os.path.isfile(rm_path):
            try:
                from memrl.service.region_manager import RegionManager
                loaded = RegionManager.load(rm_path)
                self.region_manager.subtask_q = loaded.subtask_q
                self.region_manager._known_subtasks = loaded._known_subtasks
                self.region_manager._is_clustered = loaded._is_clustered
                self.region_manager._global_reward_sum = loaded._global_reward_sum
                self.region_manager._global_reward_count = loaded._global_reward_count
                self.region_manager.subtask_q_counts = getattr(loaded, 'subtask_q_counts', {})
                self.region_manager.memory_success_sum_by_subtask = getattr(
                    loaded, 'memory_success_sum_by_subtask', {}
                )
                self.region_manager.memory_total_count_by_subtask = getattr(
                    loaded, 'memory_total_count_by_subtask', {}
                )
                self.region_manager._has_complete_memory_evidence_ledger = bool(
                    getattr(loaded, '_has_complete_memory_evidence_ledger', False)
                )
                self.region_manager.region_source_success_by_region = getattr(
                    loaded, 'region_source_success_by_region', {}
                )
                self.region_manager.region_source_total_by_region = getattr(
                    loaded, 'region_source_total_by_region', {}
                )
                self.region_manager._has_complete_region_source_evidence_ledger = bool(
                    getattr(loaded, '_has_complete_region_source_evidence_ledger', False)
                )
                self.region_manager.region_split_evidence_migration_mode = getattr(
                    loaded, 'region_split_evidence_migration_mode', 'soft_source_conserving'
                )
                self.region_manager.regions = loaded.regions
                self.region_manager.membership_weights = loaded.membership_weights
                self.region_manager._subtask_embeddings = getattr(loaded, '_subtask_embeddings', {})
                # Ensure every stored success/failure memory participates in Region
                # geometry. This adds Q coordinates only; no reward evidence is created.
                if self.region_backfill_on_restore:
                    self.backfill_missing_region_memories()
                else:
                    logger.info(
                        "[Region Backfill] disabled (feedback-first membership); "
                        "stored memories enter Region only after utility feedback"
                    )
                # Canonicalize hard membership before rebuilding summaries. Older
                # checkpoints may contain member_ids accumulated across prior
                # split/merge events; membership_weights remains the source of truth.
                self.region_manager.rebuild_hard_memberships_from_weights()
                restored = True
                # Rebuild failure/success summaries after restore. They are not
                # persisted (rebuilt from member memories), and eval-only resume
                # no longer triggers recluster, so rebuild here to ensure they're
                # populated for retrieval injection. See docs/RESUME_EVAL_DRIFT.md
                try:
                    self.region_manager._build_region_failure_summaries()
                    self.region_manager._build_region_success_summaries()
                except Exception:
                    logger.warning("Failed to rebuild region summaries after restore", exc_info=True)
                logger.info(
                    "Restored region_manager: %d subtask_q, %d regions, clustered=%s, %d subtask_embeddings",
                    len(self.region_manager.subtask_q),
                    len(self.region_manager.regions),
                    self.region_manager._is_clustered,
                    len(self.region_manager._subtask_embeddings),
                )
            except Exception:
                logger.warning("Failed to restore region_manager from %s", rm_path, exc_info=True)

        # Restore global Q cache
        gq_path = os.path.join(cache_dir, "global_q_cache.json")
        if os.path.isfile(gq_path):
            try:
                with open(gq_path, "r", encoding="utf-8") as f:
                    self._global_q_cache = json.load(f)
                restored = True
                logger.info("Restored global_q_cache (%d entries)", len(self._global_q_cache))
            except Exception:
                logger.warning("Failed to restore global_q_cache from %s", gq_path, exc_info=True)

        return restored
