# SPDX-License-Identifier: Apache-2.0

"""Joint selection and conservative bounds; no future-workload access."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .cost_model._rollout.types import ServiceSlice

if TYPE_CHECKING:
    from ...contracts import CandidatePrediction, OnlineLayout


def select_prediction(predictions, current, target, *, trigger="pressure"):
    """Apply the trigger's SLO filter, then minimize switching plus window cost."""
    if not predictions:
        return None
    work = predictions[0].work_ids
    runnable = [
        p
        for p in predictions
        if p.supported
        and not p.pruned
        and p.work_ids == work
        and p.worst_request_fraction is not None
        and p.mean_gpu_seconds is not None
    ]
    eligible = [p for p in runnable if p.worst_request_fraction + 1e-12 >= target]
    if trigger == "pressure":
        eligible = [p for p in eligible if p.layout != current]
    if eligible:
        return min(
            eligible, key=lambda p: (p.mean_gpu_seconds, p.layout != current, p.layout)
        )
    if trigger == "slack":
        return next((p for p in runnable if p.layout == current), None)
    return min(
        runnable,
        key=lambda p: (
            -p.worst_request_fraction,
            p.mean_gpu_seconds,
            p.layout != current,
            p.layout,
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
        if best.worst_request_fraction + 1e-12 >= self.target:
            if self.fraction_pruning and fraction_upper + 1e-12 < self.target:
                return "Request-fraction upper bound is below an eligible incumbent"
            if mean_cost_lower > best.mean_gpu_seconds + 1e-8:
                return "Score lower bound exceeds eligible incumbent"
        elif (
            self.fraction_pruning
            and fraction_upper + 1e-12 < best.worst_request_fraction
        ):
            return "Request-fraction upper bound is below the best runnable incumbent"
        # Preserve all ties and candidates that could still reach the target.
        return ""


def candidate_order(layouts, current, preferred, score):
    """Prioritize useful seeds/neighbors; every original candidate remains."""

    def distance(layout: OnlineLayout):
        return sum(
            a != b
            for a, b in zip(
                (
                    layout.actor_dp,
                    layout.actor_tp,
                    layout.rollout_replicas,
                    layout.rollout_tp,
                ),
                (
                    current.actor_dp,
                    current.actor_tp,
                    current.rollout_replicas,
                    current.rollout_tp,
                ),
                strict=True,
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
                item[1],
                item[0],
            ),
        )
    )


class OnlineOptimizer:
    select = staticmethod(select_prediction)

    def search(
        self, evaluator, view, forecasts, layouts, *, preferred=(), trigger="pressure"
    ):
        evaluator.prepare_context(view, forecasts)
        if not evaluator._historical:
            return evaluator.evaluate_many(view, forecasts, layouts)
        from areal.elastic.reconfigurator.parallel import evaluate_chunk

        actor_scores, rollout_scores = ({}, {})
        turns = tuple(t for e in evaluator._known_batch for t in e)[-16:]

        def score(layout):
            actor_key = (layout.actor_dp, layout.actor_tp)
            rollout_key = (layout.rollout_replicas, layout.rollout_tp)
            try:
                if actor_key not in actor_scores:
                    actor_scores[actor_key] = (
                        sum(
                            (
                                s
                                for _, s in evaluator.models.actor(
                                    layout, evaluator._known_batch
                                )
                            )
                        )
                        if evaluator._known_batch
                        else 0.0
                    )
                if rollout_key not in rollout_scores:
                    cfg = evaluator.models.rollout_config(layout)
                    seconds = 0.0
                    for t in turns:
                        q = min(t.input_tokens, cfg.max_batch_tokens)
                        prefill = evaluator.models.service(
                            layout,
                            0,
                            (
                                ServiceSlice(
                                    "rank", 0, q, "prefill", q == t.input_tokens
                                ),
                            ),
                        )
                        decode = evaluator.models.service(
                            layout,
                            t.input_tokens,
                            (ServiceSlice("rank", t.input_tokens, 1, "decode", True),),
                        )
                        seconds += (
                            math.ceil(t.input_tokens / q) * prefill
                            + t.output_tokens * decode
                        )
                    rollout_scores[rollout_key] = (
                        seconds
                        / max(1, len(turns))
                        * evaluator.config.batch_size
                        / layout.rollout_replicas
                    )
                return max(actor_scores[actor_key], rollout_scores[rollout_key])
            except ValueError:
                return math.inf

        order = candidate_order(layouts, view.layout, preferred, score)
        ordered = [None] * len(layouts)
        best = None
        work, cpus, preparations, pids = ([], 0.0, 0, set())
        parallel = (
            evaluator.executor is not None
            and getattr(evaluator.models, "worker_spec", None) is not None
        )
        payload = (
            "basic",
            getattr(evaluator.models, "worker_spec", None),
            evaluator.source_ids,
            evaluator.config,
            evaluator.handoff,
            evaluator.slo,
            evaluator.delay,
            evaluator.prune_irreversible_slo,
            evaluator.coalesce_decode,
            view,
            forecasts,
            evaluator._context_id,
        )
        waves = [order[:1]] + [order[i : i + 8] for i in range(1, len(order), 8)]
        for wave in waves:
            bound = (
                SearchBound(
                    evaluator._context_id,
                    evaluator.slo.minimum_request_fraction,
                    best,
                    fraction_pruning=False,
                )
                if best and (not (trigger == "pressure" and best.layout == view.layout))
                else None
            )
            if parallel and len(wave) > 1:
                chunks = tuple(
                    evaluator.executor.map(
                        evaluate_chunk,
                        (
                            (payload + (0.0, "all_completed", bound), wave[i : i + 2])
                            for i in range(0, len(wave), 2)
                        ),
                    )
                )
                for chunk in chunks:
                    pids.add(chunk.pid)
                    cpus += chunk.task_cpu_seconds
                    preparations += chunk.input_preparations
                    for index, prediction in zip(
                        chunk.indices, chunk.predictions, strict=True
                    ):
                        ordered[index] = prediction
                        work.append(prediction)
            else:
                for index, layout in wave:
                    prediction = evaluator.evaluate(
                        view, forecasts, layout, bound=bound
                    )
                    ordered[index] = prediction
                    work.append(prediction)
            best = select_prediction(
                tuple(p for p in ordered if p is not None),
                view.layout,
                evaluator.slo.minimum_request_fraction,
                trigger=trigger,
            )
        evaluator.execution_stats = dict(
            mode="bounded_process_pool" if parallel else "bounded_serial",
            considered=len(ordered),
            completed=sum(p.supported for p in ordered),
            pruned=sum(p.pruned for p in ordered),
            unsupported=sum(not p.supported and (not p.pruned) for p in ordered),
            evaluation_order=tuple(p.layout.key for p in work),
            workers_used=len(pids),
            worker_task_cpu_seconds=cpus,
            input_preparations=preparations,
        )
        return tuple(ordered)

    def evaluate_many(self, evaluator, view, forecasts, layouts):
        spec = getattr(evaluator.models, "worker_spec", None)
        if evaluator.executor is None or spec is None:
            return tuple(
                evaluator.evaluate(view, forecasts, layout) for layout in layouts
            )
        from areal.elastic.reconfigurator.parallel import evaluate_chunk

        evaluator.prepare_context(view, forecasts)
        payload = (
            "basic",
            spec,
            evaluator.models.source_ids,
            evaluator.config,
            evaluator.handoff,
            evaluator.slo,
            evaluator.delay,
            evaluator.prune_irreversible_slo,
            evaluator.coalesce_decode,
            view,
            forecasts,
            evaluator._context_id,
        )
        indexed = tuple(enumerate(layouts))
        jobs = ((payload, indexed[i : i + 8]) for i in range(0, len(indexed), 8))
        results = tuple(evaluator.executor.map(evaluate_chunk, jobs))
        evaluator.execution_stats = dict(
            mode="process_pool",
            workers_used=len({chunk.pid for chunk in results}),
            chunks=len(results),
            worker_task_cpu_seconds=sum(chunk.task_cpu_seconds for chunk in results),
            input_preparations=sum(chunk.input_preparations for chunk in results),
        )
        ordered = [None] * len(layouts)
        for chunk in results:
            for index, prediction in zip(chunk.indices, chunk.predictions, strict=True):
                ordered[index] = prediction
        return tuple(ordered)
