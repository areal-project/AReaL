# SPDX-License-Identifier: Apache-2.0

"""Joint selection and conservative bounds; no future-workload access."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import zip_longest
from math import inf, isfinite
from time import perf_counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from areal.elastic.contracts import (
        CandidatePrediction,
        ExecutionConfig,
        OnlineLayout,
    )


def meets_target(prediction: CandidatePrediction, target: float) -> bool:
    """Historical candidates must meet SLO and replenish admission credits."""
    return (
        (prediction.actor_memory is None or prediction.actor_memory.admissible)
        and prediction.worst_request_fraction is not None
        and (prediction.worst_request_fraction + 1e-12 >= target)
        and all(s.sustainable is True for s in prediction.actor_supply)
    )


def credit_deficit(prediction: CandidatePrediction) -> float:
    """Only deficits break SLO ties; surplus capacity does not earn free GPUs."""
    return max(
        (
            max(0.0, -s.credit_balance_rate)
            if s.credit_balance_rate is not None
            else 0.0
            if s.sustainable
            else inf
            for s in prediction.actor_supply
        ),
        default=0.0,
    )


def select_prediction(predictions, current, target, *, trigger="pressure"):
    """Apply the trigger's SLO filter, then minimize switching plus window cost."""
    if not predictions:
        return None
    work = predictions[0].work_ids
    runnable = [
        p
        for p in predictions
        if p.supported
        and (p.actor_memory is None or p.actor_memory.admissible)
        and (not p.pruned)
        and (p.work_ids == work)
        and (p.worst_request_fraction is not None)
        and (p.mean_gpu_seconds is not None)
    ]
    eligible = [p for p in runnable if meets_target(p, target)]
    if trigger == "pressure":
        eligible = [p for p in eligible if p.layout != current]
    if eligible:
        return min(
            eligible,
            key=lambda p: (p.mean_gpu_seconds, p.layout != current, p.layout.order_key),
        )
    if trigger == "slack":
        return next((p for p in runnable if p.layout == current), None)
    return min(
        runnable,
        key=lambda p: (
            -p.worst_request_fraction,
            credit_deficit(p),
            p.mean_gpu_seconds,
            p.layout != current,
            p.layout.order_key,
        ),
        default=None,
    )


@dataclass(frozen=True)
class SearchBound:
    context_id: tuple
    target: float
    best: CandidatePrediction
    fraction_pruning: bool = True

    def rejection(self, fraction_upper: float, mean_cost_lower: float) -> str:
        best = self.best
        if meets_target(best, self.target):
            if self.fraction_pruning and fraction_upper + 1e-12 < self.target:
                return "Request-fraction upper bound is below an eligible incumbent"
            if mean_cost_lower > best.mean_gpu_seconds + 1e-08:
                return "Score lower bound exceeds eligible incumbent"
        elif (
            not best.actor_supply
            and self.fraction_pruning
            and (fraction_upper + 1e-12 < best.worst_request_fraction)
        ):
            return "Request-fraction upper bound is below the best runnable incumbent"
        return ""


def candidate_order(layouts, current, preferred, score):
    """Prioritize useful seeds/neighbors; every original candidate remains."""

    def distance(layout: ExecutionConfig | OnlineLayout):
        return sum(
            (
                a != b
                for a, b in zip_longest(
                    layout.actor_key + layout.rollout_key,
                    current.actor_key + current.rollout_key,
                )
            )
        )

    return tuple(
        sorted(
            enumerate(layouts),
            key=lambda item: (
                item[1] != current,
                item[1] not in preferred,
                distance(item[1]) != 1,
                score(item[1]),
                item[1].order_key,
                item[0],
            ),
        )
    )


class OnlineOptimizer:
    """Candidate ordering and bounded search over an evaluate-only model port."""

    select = staticmethod(select_prediction)

    def __init__(self, *, max_queries=64, max_seconds=30.0):
        if type(max_queries) is not int or max_queries < 1:
            raise ValueError("Search requires a positive query budget")
        if not isfinite(max_seconds) or max_seconds <= 0:
            raise ValueError("Search requires a positive wall-time budget")
        self.max_queries, self.max_seconds = max_queries, max_seconds

    def search(
        self, evaluator, view, forecasts, layouts, *, preferred=(), trigger="pressure"
    ):
        started = perf_counter()
        query_limit = min(
            self.max_queries, getattr(evaluator, "max_queries", self.max_queries)
        )
        seconds_limit = min(
            self.max_seconds, getattr(evaluator, "max_seconds", self.max_seconds)
        )
        evaluator.prepare_context(view, forecasts)
        order = (
            evaluator.order_candidates(view, forecasts, layouts, preferred)
            if evaluator.historical
            else tuple(enumerate(layouts))
        )
        ordered = [None] * len(layouts)
        best = None
        work, cpus, preparations, pids = ([], 0.0, 0, set())
        waves = [order[:1]] + [order[i : i + 8] for i in range(1, len(order), 8)]
        for wave in waves:
            remaining = query_limit - len(work)
            if remaining <= 0 or (work and perf_counter() - started >= seconds_limit):
                break
            wave = wave[:remaining]
            bound = (
                SearchBound(
                    evaluator.context_id,
                    evaluator.slo.minimum_request_fraction,
                    best,
                    fraction_pruning=False,
                )
                if evaluator.historical
                and best
                and (not (trigger == "pressure" and best.layout == view.layout))
                else None
            )
            result = evaluator.evaluate_wave(view, forecasts, wave, bound=bound)
            pids.update(result.worker_pids)
            cpus += result.task_cpu_seconds
            preparations += result.input_preparations
            for index, prediction in zip(
                result.indices, result.predictions, strict=True
            ):
                ordered[index] = prediction
                work.append(prediction)
            if evaluator.historical:
                best = self.select(
                    tuple(p for p in ordered if p is not None),
                    view.layout,
                    evaluator.slo.minimum_request_fraction,
                    trigger=trigger,
                )
        evaluator.execution_stats = dict(
            mode="bounded_process_pool"
            if evaluator.parallel_available
            else "bounded_serial",
            considered=len(ordered),
            completed=sum(p.supported for p in work),
            pruned=sum(p.pruned for p in work),
            unsupported=sum(not p.supported and (not p.pruned) for p in work),
            model_queries=len(work),
            max_queries=query_limit,
            max_seconds=seconds_limit,
            elapsed_seconds=perf_counter() - started,
            budget_exhausted=len(work) < len(layouts),
            evaluation_order=tuple(p.layout.key for p in work),
            workers_used=len(pids),
            worker_task_cpu_seconds=cpus,
            input_preparations=preparations,
        )
        return tuple(p for p in ordered if p is not None)

    def evaluate_many(self, evaluator, view, forecasts, layouts):
        result = evaluator.evaluate_wave(
            view, forecasts, tuple(enumerate(layouts)), chunk_size=8
        )
        if evaluator.parallel_available:
            evaluator.execution_stats = dict(
                mode="process_pool",
                workers_used=len(result.worker_pids),
                chunks=result.chunks,
                worker_task_cpu_seconds=result.task_cpu_seconds,
                input_preparations=result.input_preparations,
            )
        ordered = [None] * len(layouts)
        for index, prediction in zip(result.indices, result.predictions, strict=True):
            ordered[index] = prediction
        return tuple(ordered)
