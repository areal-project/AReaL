# SPDX-License-Identifier: Apache-2.0

"""Basic v1 candidate scoring through the aggregate JointSimulator."""

from __future__ import annotations

from concurrent.futures import Executor
from dataclasses import replace

from areal.elastic.contracts import (
    CandidatePrediction,
    Forecast,
    OnlineLayout,
    OnlineObservation,
)
from areal.elastic.contracts import SimulationConfig as OnlineRunConfig
from areal.elastic.contracts import SimulationModels as OnlineModels
from areal.elastic.reconfigurator.basic.core import fingerprint
from areal.elastic.reconfigurator.basic.optimizer import (
    SearchBound,
)
from areal.elastic.serving import ServingSLO, request_slo_status, rolling_slo
from areal.elastic.simulation.online_inputs import ForecastInputs
from areal.elastic.simulation.online_transitions import (
    HandoffConfig,
)

from .optimizer import OnlineOptimizer
from .simulator import JointSimulator


class OnlineForecastModel:
    """A candidate evaluator whose only workload input is the sampled forecast."""

    def __init__(
        self,
        models: OnlineModels,
        config: OnlineRunConfig,
        handoff: HandoffConfig,
        slo: ServingSLO,
        decision_delay_seconds: float = 0.0,
        *,
        prune_irreversible_slo: bool = True,
        coalesce_decode: bool = True,
        executor: Executor | None = None,
        optimizer=None,
    ):
        self.models, self.config, self.handoff, self.slo = models, config, handoff, slo
        self.delay = decision_delay_seconds
        self.prune_irreversible_slo = prune_irreversible_slo
        self.coalesce_decode = coalesce_decode
        self.executor = executor
        self.optimizer = optimizer or OnlineOptimizer()
        self._view = None
        self._forecasts = None
        self._work_ids = ()
        self._prepared: dict[int, ForecastInputs] = {}
        self.input_preparations = 0
        self.execution_stats = {"mode": "serial"}

    @property
    def source_ids(self):
        return self.models.source_ids

    def prepare_context(self, view, forecasts):
        if self._view is view and self._forecasts is forecasts:
            return
        self._view, self._forecasts = view, forecasts
        self._work_ids = tuple(fingerprint(f) for f in forecasts)
        self._context_id = (
            fingerprint(view),
            self._work_ids,
            fingerprint(
                self.config, self.handoff, self.slo, self.delay, self.source_ids
            ),
        )
        self._prepared = {}
        self._historical = any(f.replay is not None for f in forecasts)
        self._past = frozenset(
            r.request_id for r in view.requests if r.completed_at is not None
        )
        self._remaining = tuple(
            sum(len(e.turns) for e in f.episodes) - len(self._past) for f in forecasts
        )
        self._known_batch = ()
        if self._historical:
            completed = {
                e.episode_id: e.completed_turns
                for e in view.episodes
                if e.completed_at is not None
            }
            identities = (
                view.actor.waiting.batch
                if view.actor.waiting is not None
                else view.actor.batch
            )
            if identities:
                self._known_batch = tuple(completed[e] for e in identities)
            elif len(completed) >= self.config.batch_size:
                self._known_batch = tuple(
                    list(completed.values())[-self.config.batch_size :]
                )

    def evaluate(
        self,
        view: OnlineObservation,
        forecasts: tuple[Forecast, ...],
        layout: OnlineLayout,
        *,
        bound: SearchBound | None = None,
    ) -> CandidatePrediction:
        self.prepare_context(view, forecasts)
        work_ids = self._work_ids
        if bound is not None and bound.context_id != self._context_id:
            bound = None
        bound = None
        costs, fractions, window_costs = [], [], []
        switching_costs, horizons, window_stats, window_results = [], [], [], []
        historical = self._historical
        coverage = "legacy_sampled_execution"
        known_batch = self._known_batch
        if historical:
            if known_batch:
                try:
                    if not self.models.actor(layout, known_batch):
                        raise ValueError("Actor model supplied no update stages")
                except ValueError as exc:
                    return CandidatePrediction(
                        layout,
                        False,
                        None,
                        None,
                        work_ids,
                        str(exc),
                        self.models.source_ids,
                        actor_coverage="known_batch_rejected",
                    )
                coverage = "known_complete_batch_capacity_checked"
        past = self._past
        if self.prune_irreversible_slo and not historical:
            failed = sum(
                r.completed_at is None
                and request_slo_status(r, self.slo, now=view.now) is False
                for r in view.requests
            )
            for remaining in self._remaining:
                upper = 1 - failed / remaining if remaining else 1.0
                if upper + 1e-12 < self.slo.minimum_request_fraction:
                    return CandidatePrediction(
                        layout,
                        False,
                        None,
                        None,
                        work_ids,
                        f"Observed irreversible SLO failures: {failed}; sampled future scope: {remaining}; joint_fraction_upper_bound={upper:.12g}; no cost simulation performed",
                        self.models.source_ids,
                    )
        for index, forecast in enumerate(forecasts):
            horizon = forecast.input_closes_at - view.now
            running_cost = horizon * (layout.actor_gpus + layout.rollout_gpus)
            try:
                prepared = self._prepared.get(index)
                if prepared is None:
                    prepared = ForecastInputs.prepare(
                        view,
                        forecast,
                        self.config.max_new_tokens,
                        self.config.max_context_tokens,
                    )
                    self._prepared[index] = prepared
                    self.input_preparations += 1
                simulation = JointSimulator.from_forecast(
                    view,
                    forecast,
                    self.models,
                    self.config,
                    self.handoff,
                    self.slo,
                    coalesce_decode=self.coalesce_decode,
                    _prepared=prepared,
                )
                simulation._forecast_target = layout
                if historical:
                    # Keep residency through H even when the observed input has
                    # already closed. The synthetic close event still ends at H.
                    simulation.input_closed = False
                if layout != view.layout:
                    simulation.decision_pending = True
                    simulation.schedule_at(
                        view.now + self.delay, "decision_apply", layout, priority=9
                    )
                if historical:
                    result = simulation.run(until=forecast.input_closes_at)
                    window_costs.append(result["total_gpu_seconds"])
                    if self.handoff.mode == "cold":
                        running_cost = result["total_gpu_seconds"]

                    window_results.append(result)
                    window_stats.append(
                        rolling_slo(
                            replace(
                                view,
                                now=simulation.now,
                                requests=tuple(
                                    r.view.snapshot()
                                    for key, r in simulation.requests.items()
                                    if key not in past
                                ),
                            ),
                            self.slo,
                            horizon,
                        )
                    )
                    # Only estimate an unfinished transition beyond H. Do not
                    # drain target work or let tail outcomes alter window SLO.
                    if result["status"] == "running" and (
                        simulation.pending_handoff or simulation.decision_pending
                    ):
                        result = simulation.run(stop_after_handoff=True)
                else:
                    result = simulation.run()
            except ValueError as exc:
                return CandidatePrediction(
                    layout,
                    False,
                    None,
                    None,
                    work_ids,
                    str(exc),
                    self.models.source_ids,
                )
            if (
                result["status"]
                not in (("complete", "running") if historical else ("complete",))
                or simulation.counters["handoff_deferred"]
                or (
                    historical
                    and (simulation.pending_handoff or simulation.decision_pending)
                )
            ):
                return CandidatePrediction(
                    layout,
                    False,
                    None,
                    None,
                    work_ids,
                    result["reason"]
                    if result["status"] not in ("complete", "running")
                    else "No spare GPUs for TP-local overlap",
                    self.models.source_ids,
                )
            if historical and not known_batch:
                changes_actor = (layout.actor_dp, layout.actor_tp) != (
                    view.layout.actor_dp,
                    view.layout.actor_tp,
                )
                owners = {
                    e["target"]
                    for e in simulation.events
                    if e["kind"] == "actor_cutover"
                }
                exercised = any(
                    e["kind"] == "actor_start"
                    and e["deployment"] in owners
                    and any(
                        row["episodes"] == e["episodes"]
                        and row["committed_at"] is not None
                        and row["committed_at"] <= forecast.input_closes_at
                        for row in simulation.training_records
                    )
                    for e in simulation.events
                )
                if changes_actor and not exercised:
                    return CandidatePrediction(
                        layout,
                        False,
                        None,
                        None,
                        work_ids,
                        "Actor capacity unknown: no observed full batch or post-cutover complete update",
                        self.models.source_ids,
                        actor_coverage="unknown_no_complete_batch",
                    )
                coverage = (
                    "post_cutover_complete_update"
                    if exercised
                    else "current_actor_unchanged"
                )
            if historical:
                cost = simulation.switching_gpu_seconds()
                switching_costs.append(cost)
                horizons.append(horizon)
                costs.append(cost + running_cost)
                fractions.append(window_stats[-1].fraction)
                continue
            future = [
                r.view for key, r in simulation.requests.items() if key not in past
            ]
            passed = sum(
                r.completed_at is not None
                and request_slo_status(r, self.slo, now=r.completed_at) is True
                for r in future
            )
            fractions.append(passed / len(future) if future else 1.0)
            costs.append(result["total_gpu_seconds"])
        return CandidatePrediction(
            layout,
            True,
            min(fractions) if all(f is not None for f in fractions) else None,
            sum(costs) / len(costs),
            work_ids,
            "Basic v1 Joint aggregate forecast; cold score = actual GPU residency integral; target tail not scored"
            if historical and self.handoff.mode == "cold"
            else "Basic v1 Joint aggregate forecast; score = switching GPU seconds + window seconds * target GPU count; target tail not scored"
            if historical
            else "Sampled arrivals close at horizon; all sampled serving and full training batches drain afterwards",
            self.models.source_ids,
            sum(window_costs) / len(window_costs) if historical else None,
            0.0 if historical else None,
            coverage,
            switching_gpu_seconds=sum(switching_costs) / len(costs)
            if historical
            else None,
            running_gpu_seconds_per_second=sum(window_costs) / sum(horizons)
            if historical and self.handoff.mode == "cold"
            else layout.actor_gpus + layout.rollout_gpus
            if historical
            else None,
            evaluation_seconds=sum(horizons) / len(costs) if historical else None,
            running_gpu_seconds=sum(window_costs) / len(costs)
            if historical and self.handoff.mode == "cold"
            else (layout.actor_gpus + layout.rollout_gpus) * sum(horizons) / len(costs)
            if historical
            else None,
            score_gpu_seconds=sum(costs) / len(costs) if historical else None,
            completed_requests=sum(s.completed for s in window_stats)
            if historical
            else None,
            pending_requests=sum(s.pending for s in window_stats)
            if historical
            else None,
            failed_pending_requests=sum(s.decided - s.completed for s in window_stats)
            if historical
            else None,
            completed_updates=sum(
                r["completed_updates"] - view.training_version for r in window_results
            )
            if historical
            else None,
            retained_episodes=sum(r["retained_episodes"] for r in window_results)
            if historical
            else None,
        )

    def search(self, view, forecasts, layouts, *, preferred=(), trigger="pressure"):
        return self.optimizer.search(
            self, view, forecasts, layouts, preferred=preferred, trigger=trigger
        )

    def evaluate_many(self, view, forecasts, layouts):
        return self.optimizer.evaluate_many(self, view, forecasts, layouts)
