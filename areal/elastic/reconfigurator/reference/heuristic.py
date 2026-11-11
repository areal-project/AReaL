# SPDX-License-Identifier: Apache-2.0

"""Reference v2: diverse heuristic queries, feedback recombination and recovery."""

from __future__ import annotations

import math
from collections import Counter
from time import perf_counter

from areal.elastic.reconfigurator.reference.candidates import (
    HeuristicPreparation,
    distance,
)
from areal.elastic.reconfigurator.reference.optimizer import (
    OnlineOptimizer,
    credit_deficit,
    meets_target,
    select_prediction,
)

POLICY = dict(
    version="reference-v2-heuristic",
    current_queries=1,
    history_queries=7,
    neighbor_queries=8,
    initial_diverse_queries=32,
    feedback_queries=16,
    minimum_search_spacing_seconds=60,
    search_rate_window_seconds=240,
    searches_per_rate_window=3,
    unresolved_retry_seconds=120,
    minimum_new_completed_episodes=8,
    shape_change_ratio=0.5,
    recovery_slo_tolerance=0.02,
)


def recovery_key(prediction):
    """Prefer admission replenishment and draining, with cost as the final tie."""
    return (
        -(prediction.completed_updates or 0),
        prediction.pending_requests
        if prediction.pending_requests is not None
        else math.inf,
        -(prediction.completed_requests or 0),
        credit_deficit(prediction),
        -prediction.worst_request_fraction,
        prediction.mean_gpu_seconds,
        prediction.layout.order_key,
    )


def select_recovery(predictions, current, target, *, trigger, recovering):
    baseline = select_prediction(predictions, current, target, trigger=trigger)
    if baseline is None or not recovering or trigger != "pressure":
        return baseline, "reference"
    runnable = [
        p
        for p in predictions
        if p.supported
        and not p.pruned
        and (p.actor_memory is None or p.actor_memory.admissible)
        and p.work_ids == baseline.work_ids
        and p.worst_request_fraction is not None
        and p.mean_gpu_seconds is not None
    ]
    if any(meets_target(p, target) for p in runnable):
        return baseline, "reference"
    # Do not sacrifice substantial attainable SLO for throughput. Within a two
    # percentage-point band, old deadline misses should not reward downsizing.
    best_fraction = max(p.worst_request_fraction for p in runnable)
    band = [
        p
        for p in runnable
        if p.worst_request_fraction >= best_fraction - POLICY["recovery_slo_tolerance"]
    ]
    return min(band, key=recovery_key), "queue_recovery"


class HeuristicOptimizer(OnlineOptimizer):
    """Use the v2 policy for factorized grids; retain explicit-catalog contracts."""

    def __init__(self, *, max_queries=64, max_seconds=30.0, chunk_size=1):
        super().__init__(max_queries=max_queries, max_seconds=max_seconds)
        if type(chunk_size) is not int or chunk_size < 1:
            raise ValueError("candidate chunk size must be a positive integer")
        self.chunk_size = chunk_size
        self.history = []
        self.search_index = 0
        self.recovering = False
        self.selection_mode = "reference"
        self.preparation = HeuristicPreparation()

    def select(self, predictions, current, target, *, trigger="pressure"):
        best, self.selection_mode = select_recovery(
            predictions, current, target, trigger=trigger, recovering=self.recovering
        )
        return best

    def search(
        self, evaluator, view, forecasts, layouts, *, preferred=(), trigger="pressure"
    ):
        if not isinstance(layouts, dict):
            self.recovering = False
            return super().search(
                evaluator,
                view,
                forecasts,
                layouts,
                preferred=preferred,
                trigger=trigger,
            )
        return self.search_grid(
            evaluator, view, forecasts, layouts, preferred=preferred, trigger=trigger
        )

    def search_grid(
        self, evaluator, view, forecasts, grid, *, preferred=(), trigger="pressure"
    ):
        started = perf_counter()
        self.search_index += 1
        actors, rollouts, count = self.preparation.grid_roles(grid)
        a_scores, r_scores = self.preparation.prepare(
            evaluator, view, forecasts, actors, rollouts
        )
        a_keys, r_keys = (
            {a.actor_key for a in actors},
            {r.rollout_key for r in rollouts},
        )
        seen, pool, labels = set(), [], []

        def add(layout, label):
            # Execution normalizes DP=1 to disabled DP Attention. Always retain
            # the live layout even if its spelling is outside the raw grid, and
            # avoid querying an equivalent unresolved candidate a second time.
            effective = evaluator.cost_models.resolve(layout)
            if (
                effective in seen
                or (
                    layout != view.layout
                    and (
                        layout.actor_key not in a_keys
                        or layout.rollout_key not in r_keys
                    )
                )
                or layout.actor_gpus + layout.rollout_gpus > grid["gpu_budget"]
            ):
                return False
            seen.add(effective)
            pool.append(layout)
            labels.append(label)
            return True

        add(view.layout, "current")
        for c in reversed(self.history[-POLICY["history_queries"] :]):
            add(c, "same_run_history")
        neighbors = self.preparation.ordered(
            grid, view.layout, actors, rollouts, a_scores, r_scores
        )
        # Split the neighbor allowance across roles; large Actor neighborhoods
        # cannot consume all Rollout or joint probes.
        counts = Counter()
        for c in neighbors:
            if distance(c, view.layout) != 1:
                if c != view.layout:
                    break
                continue
            role = "actor" if c.actor_key != view.layout.actor_key else "rollout"
            if counts[role] < POLICY["neighbor_queries"] // 2 and add(c, "neighbor"):
                counts[role] += 1
            if sum(counts.values()) == POLICY["neighbor_queries"]:
                break
        diverse = iter(
            self.preparation.diverse(
                grid, view.layout, actors, rollouts, a_scores, r_scores
            )
        )
        limit = min(
            self.max_queries, getattr(evaluator, "max_queries", self.max_queries)
        )
        initial_limit = max(1, limit - POLICY["feedback_queries"])
        while len(pool) < initial_limit:
            c = next(diverse, None)
            if c is None:
                break
            add(c, "diverse_joint")
        fallback = self.preparation.ordered(
            grid, view.layout, actors, rollouts, a_scores, r_scores
        )
        while len(pool) < initial_limit:
            c = next(fallback, None)
            if c is None:
                break
            add(c, "grid_fallback")
        preparation = perf_counter() - started
        self.recovering = (
            sum(not e.admitted for e in view.episodes) >= evaluator.config.batch_size
        )
        work, pids = [], set()
        seconds_limit = min(
            self.max_seconds, getattr(evaluator, "max_seconds", self.max_seconds)
        )

        def evaluate_until(stop):
            while len(work) < min(stop, len(pool), limit):
                if work and perf_counter() - started >= seconds_limit:
                    break
                size = 1 if not work else min(8, stop - len(work), limit - len(work))
                wave = tuple(
                    enumerate(pool[len(work) : len(work) + size], start=len(work))
                )
                result = evaluator.evaluate_wave(
                    view, forecasts, wave, chunk_size=self.chunk_size
                )
                work.extend(result.predictions)
                pids.update(result.worker_pids)

        evaluate_until(initial_limit)
        # Recombine roles of actually supported predictions, then spend any
        # remaining allowance on the diverse stream/full-grid fallback.
        usable = [
            p
            for p in work
            if p.supported
            and not p.pruned
            and (p.actor_memory is None or p.actor_memory.admissible)
            and p.worst_request_fraction is not None
            and p.mean_gpu_seconds is not None
        ]
        ranked = sorted(
            usable,
            key=lambda p: (
                not meets_target(p, evaluator.slo.minimum_request_fraction),
                -p.worst_request_fraction,
                *recovery_key(p),
            ),
        )
        best_a, best_r = {}, {}
        for p in ranked:
            if len(best_a) < 4:
                best_a.setdefault(p.layout.actor_key, p.layout)
            if len(best_r) < 4:
                best_r.setdefault(p.layout.rollout_key, p.layout)
        for actor in best_a.values():
            for rollout in best_r.values():
                if len(pool) < limit:
                    add(actor.with_rollout(rollout), "feedback_joint")
        for source, label in ((diverse, "diverse_joint"), (fallback, "grid_fallback")):
            while len(pool) < limit:
                c = next(source, None)
                if c is None:
                    break
                add(c, label)
        evaluate_until(limit)
        best = self.select(
            work, view.layout, evaluator.slo.minimum_request_fraction, trigger=trigger
        )
        if best is not None:
            self.history = [c for c in self.history if c != best.layout] + [best.layout]
        evaluator.execution_stats = dict(
            mode=POLICY["version"],
            considered=count,
            actor_options=len(actors),
            rollout_options=len(rollouts),
            preparation_seconds=preparation,
            model_queries=len(work),
            completed=sum(p.supported for p in work),
            unsupported=sum(not p.supported for p in work),
            elapsed_seconds=perf_counter() - started,
            max_queries=limit,
            max_seconds=seconds_limit,
            evaluation_order=[p.layout.key for p in work],
            workers_used=len(pids),
            current_is_explicit_fallback=True,
            query_groups=dict(Counter(labels[: len(work)])),
            query_labels=labels[: len(work)],
            distance_counts=dict(
                Counter(distance(p.layout, view.layout) for p in work)
            ),
            recovering=self.recovering,
            selection_mode=self.selection_mode,
            heuristic_preparation=self.preparation.statistics(),
        )
        return tuple(work)
