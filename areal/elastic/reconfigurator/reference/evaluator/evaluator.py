# SPDX-License-Identifier: Apache-2.0

"""Complete Reference candidate evaluation with private hot/cold execution."""

from __future__ import annotations

import heapq
import math
from collections import Counter
from collections.abc import Callable
from concurrent.futures import Executor
from dataclasses import asdict, dataclass, field, replace
from functools import cached_property

from areal.elastic.contracts import (
    ActorMemoryRejected,
    ActorMemorySafety,
    ActorSupply,
    ActorView,
    ActorWaitView,
    CandidatePrediction,
    CostModels,
    EpisodeSample,
    EpisodeView,
    ExecutionConfig,
    Forecast,
    HandoffConfig,
    OnlineLayout,
    OnlineObservation,
    RequestView,
    ServiceView,
    SimulationConfig,
    ToolView,
    Turn,
)
from areal.elastic.layout import ExecutionLayout, ReconfigurationPlanner
from areal.elastic.reconfigurator.reference.evaluator.contracts import Request
from areal.elastic.reconfigurator.reference.evaluator.cost_model.contracts import (
    ServiceSlice,
)
from areal.elastic.reconfigurator.reference.evaluator.cost_model.rollout.scheduling import (
    scheduler_for,
)
from areal.elastic.reconfigurator.reference.identity import fingerprint
from areal.elastic.reconfigurator.reference.optimizer import (
    OnlineOptimizer,
    SearchBound,
    candidate_order,
)
from areal.elastic.serving import ServingSLO, request_slo_status, rolling_slo
from areal.elastic.simulation_primitives import (
    GPUAllocator,
    ReplayInputs,
    ReplicaState,
    RequestState,
    SerializedLink,
    WorkInputs,
)


@dataclass(init=False)
class _Request:
    view: RequestState
    reservation: int
    order: int

    def __init__(
        self,
        view: RequestView,
        reservation: int,
        order: int,
        *,
        values: tuple | None = None,
    ):
        self.view = (
            RequestState(*values) if values is not None else RequestState.copy(view)
        )
        self.reservation, self.order = (reservation, order)

    @property
    def group_id(self):
        return self.view.episode_id

    @cached_property
    def request(self):
        return Request(
            f"request-{self.view.turn}", self.view.input_tokens, self.reservation
        )

    @property
    def prefilled(self):
        return self.view.prefilled

    @property
    def generated(self):
        return self.view.generated

    @property
    def kv_tokens(self):
        return self.view.prefilled + max(0, self.view.generated - 1)


@dataclass
class _Episode:
    episode_id: str
    admitted: bool = False
    completed_at: float | None = None
    past: list[Turn] = field(default_factory=list)
    current: str | None = None
    tool: ToolView | None = None


@dataclass
class _Deployment:
    identity: int
    role: str
    layout: ExecutionConfig | OnlineLayout
    status: str
    version: int
    busy: dict[int, ServiceView] = field(default_factory=dict)
    transfers: int = 0
    replicas: dict[int, ReplicaState] = field(default_factory=dict)

    def __post_init__(self):
        if self.role == "rollout":
            self.replicas = {
                i: ReplicaState() for i in range(self.layout.rollout_replicas)
            }


@dataclass
class _ForecastSLOGuard:
    excluded: set[str]
    total: int
    failed: set[str] = field(default_factory=set)
    check_after: float = 0.0


@dataclass
class _CandidateGuard:
    bound: SearchBound
    excluded: frozenset[str]
    total: int
    completed_cost: float
    fraction_cap: float
    samples: int
    failed: set[str] = field(default_factory=set)
    check_after: float = 0.0
    upper: float = 1.0
    lower: float = 0.0
    fixed_running_cost: float | None = None


class MemoryCheckedCostModels:
    def __init__(self, cost_models, target, reserve, observed_at):
        self.cost_models, self.target, self.reserve = (cost_models, target, reserve)
        self.checked = set()
        self.report = ActorMemorySafety(
            True,
            reserve,
            0,
            None,
            None,
            None,
            (),
            observed_at,
            "Only batches used by candidate evaluation are checked",
            scope="evaluation",
        )

    def __getattr__(self, name):
        return getattr(self.cost_models, name)

    def actor(self, layout, batch):
        if layout.actor_key == self.target.actor_key and batch not in self.checked:
            try:
                estimate = getattr(self.cost_models, "actor_memory", None)
                if estimate is None:
                    raise ValueError("Actor memory estimate unavailable")
                used, limit, measured_oom = estimate(layout, batch)
                if (
                    not math.isfinite(used)
                    or used < 0
                    or (not math.isfinite(limit))
                    or (limit <= 0)
                ):
                    raise ValueError("Invalid Actor memory estimate")
                self.checked.add(batch)
                report = replace(self.report, checked_batches=len(self.checked))
                if report.peak_gib is None or used > report.peak_gib:
                    report = replace(
                        report,
                        peak_gib=used,
                        capacity_gib=limit,
                        allowed_gib=limit * (1 - self.reserve),
                    )
                self.report = report
                if measured_oom or used > limit * (1 - self.reserve):
                    raise ValueError(
                        "Evaluated Actor batch exceeds the memory safety budget or has a measured OOM"
                    )
            except ValueError as exc:
                self.report = replace(self.report, admissible=False, reason=str(exc))
        if not self.report.admissible:
            raise ValueError(self.report.reason)
        return self.cost_models.actor(layout, batch)


class Evaluator:
    """Check candidates, predict their execution and aggregate scores. Each forecast owns its own state."""

    _host_steps = {"close_admission", "set_layout", "rebind_runtime", "pause_rollout"}

    def __init__(
        self,
        cost_models: CostModels,
        config: SimulationConfig,
        handoff: HandoffConfig,
        slo: ServingSLO,
        decision_delay_seconds: float = 0.0,
        *,
        prune_irreversible_slo: bool = True,
        executor: Executor | None = None,
        optimizer=None,
        actor_memory_reserve_fraction: float = 0.0,
        actor_memory_scope: str = "all_completed",
    ):
        self.cost_models, self.config, self.handoff, self.slo = (
            cost_models,
            config,
            handoff,
            slo,
        )
        self.delay = decision_delay_seconds
        self.prune_irreversible_slo = prune_irreversible_slo
        self.executor = executor
        self.optimizer = optimizer or OnlineOptimizer()
        if not 0 <= actor_memory_reserve_fraction < 1:
            raise ValueError("Actor memory reserve must be in [0, 1)")
        self.actor_memory_reserve_fraction = actor_memory_reserve_fraction
        if actor_memory_scope not in ("all_completed", "evaluation"):
            raise ValueError("Actor memory scope must be all_completed or evaluation")
        self.actor_memory_scope = actor_memory_scope
        self._view = None
        self._forecasts = None
        self._work_ids = ()
        self._prepared: dict[int, ReplayInputs] = {}
        self.input_preparations = 0
        self.execution_stats = {"mode": "serial"}

    @property
    def source_ids(self):
        return self.cost_models.source_ids

    def prepare_context(self, view, forecasts):
        if self._view is view and self._forecasts is forecasts:
            return
        self._view, self._forecasts = (view, forecasts)
        self._work_ids = tuple(fingerprint(f) for f in forecasts)
        self._context_id = (
            fingerprint(view),
            self._work_ids,
            fingerprint(
                self.config,
                self.handoff,
                self.slo,
                self.delay,
                self.source_ids,
                self.actor_memory_reserve_fraction,
                self.actor_memory_scope,
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
        self._memory_checks = {}
        self._memory_batches = {}
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
        if (
            self.actor_memory_reserve_fraction
            and self.actor_memory_scope == "all_completed"
        ):
            ordered = sorted(
                view.episodes, key=lambda e: (e.available_at, e.episode_id)
            )
            size = self.config.batch_size
            for start in range(0, len(ordered) - size + 1, size):
                rows = ordered[start : start + size]
                if all(
                    e.completed_at is not None and e.completed_at <= view.now
                    for e in rows
                ):
                    self._memory_batches[tuple(e.episode_id for e in rows)] = tuple(
                        e.completed_turns for e in rows
                    )
            if self._known_batch:
                identities = identities or tuple(list(completed)[-size:])
                self._memory_batches[tuple(identities)] = self._known_batch

    def actor_memory_safety(self, view, layout):
        reserve = self.actor_memory_reserve_fraction
        if not reserve:
            return None
        key = layout.actor_key
        if key in self._memory_checks:
            return self._memory_checks[key]
        peak = capacity = allowed = None
        worst, count, admissible = ((), 0, True)
        reason = "All completed historical batches retain the configured memory reserve"
        estimate = getattr(self.cost_models, "actor_memory", None)
        if estimate is None or not self._memory_batches:
            admissible = False
            reason = "Actor memory safety unknown: need memory estimates and a completed observed batch"
        else:
            try:
                for ids, batch in self._memory_batches.items():
                    used, limit, measured_oom = estimate(layout, batch)
                    if (
                        not math.isfinite(used)
                        or used < 0
                        or (not math.isfinite(limit))
                        or (limit <= 0)
                    ):
                        raise ValueError("Invalid Actor memory estimate")
                    count += 1
                    if peak is None or used > peak:
                        peak, capacity, allowed, worst = (
                            used,
                            limit,
                            limit * (1 - reserve),
                            ids,
                        )
                    if measured_oom or used > limit * (1 - reserve):
                        admissible = False
                        reason = "Historical Actor shape exceeds the memory safety budget or has a measured OOM"
            except ValueError as exc:
                admissible = False
                reason = f"Actor memory safety unavailable: {exc}"
        result = ActorMemorySafety(
            admissible, reserve, count, peak, capacity, allowed, worst, view.now, reason
        )
        self._memory_checks[key] = result
        return result

    def actor_supply(self, view, forecast, layout, *, cost_models=None):
        """Estimate ongoing demand from offered arrivals and causal turn templates.

        Incomplete final batches wrap to the start of the same historical sequence
        and carry fractional batch weight. This estimates a repeated workload's
        capacity, not extra training in the serving window or an exact queue bound.
        """
        cost_models = cost_models or self.cost_models
        replay = forecast.replay
        recent = sorted(
            (
                e
                for e in view.episodes
                if replay.template_started_at <= e.available_at < view.now
            ),
            key=lambda e: (e.available_at, e.episode_id),
        )
        rate = (
            len(recent) / replay.template_seconds
            if replay.template_seconds > 0
            else 0.0
        )
        if view.input_closed:
            rate = 0.0
        templates = {e.episode_id: e.turns for e in forecast.episodes}
        rows = tuple(templates[e.episode_id] for e in recent) or self._known_batch
        batch_size = self.config.batch_size
        actor_seconds = publication_seconds = capacity = balance = None
        sustainable = True if rate == 0 else None
        batches = padded = 0
        reason = (
            "No ongoing offered arrivals"
            if rate == 0
            else "No historical Actor workload"
        )
        if rows:
            weighted_seconds = weight = 0.0
            try:
                for offset in range(0, len(rows), batch_size):
                    count = min(batch_size, len(rows) - offset)
                    batch = tuple(
                        rows[(offset + i) % len(rows)] for i in range(batch_size)
                    )
                    stages = cost_models.actor(layout, batch)
                    if not stages:
                        raise ValueError("Actor model supplied no capacity stages")
                    seconds = sum((seconds for _, seconds in stages))
                    if not math.isfinite(seconds) or seconds <= 0:
                        raise ValueError(
                            "Actor capacity requires a positive finite cycle"
                        )
                    weighted_seconds += seconds * count / batch_size
                    weight += count / batch_size
                    batches += 1
                    padded += batch_size - count
                actor_seconds = weighted_seconds / weight
                publication_seconds = cost_models.publication(layout, (layout,))
                if not math.isfinite(publication_seconds) or publication_seconds < 0:
                    raise ValueError("Publication capacity requires a finite duration")
                capacity = batch_size / (actor_seconds + publication_seconds)
                balance = capacity - rate
                sustainable = balance >= -1e-12
                reason = "Target-only historical Actor stages plus steady-layout publication; excludes input waiting, cutover and old Actor work"
            except ValueError as exc:
                reason = f"Actor supply estimate unavailable: {exc}"
        return ActorSupply(
            len(recent),
            replay.template_seconds,
            rate,
            actor_seconds,
            publication_seconds,
            capacity,
            balance,
            sustainable,
            batches,
            padded,
            sum(e.completed_at is None for e in recent),
            view.input_closed,
            reason,
        )

    def evaluate(
        self,
        view: OnlineObservation,
        forecasts: tuple[Forecast, ...],
        layout: ExecutionConfig | OnlineLayout,
        *,
        bound: SearchBound | None = None,
    ) -> CandidatePrediction:
        effective = getattr(self.cost_models, "resolve", lambda c: c)(layout)
        if effective != layout:
            # Cache and execute the bound configuration; preserve the caller's
            # candidate identity in the evaluate-only response.
            return replace(
                self.evaluate(view, forecasts, effective, bound=bound), layout=layout
            )
        self.prepare_context(view, forecasts)
        if (
            self.actor_memory_reserve_fraction
            and self.actor_memory_scope == "evaluation"
        ):
            cost_models = MemoryCheckedCostModels(
                self.cost_models, layout, self.actor_memory_reserve_fraction, view.now
            )
            prediction = self._evaluate(
                view, forecasts, layout, bound=bound, cost_models=cost_models
            )
            memory = cost_models.report
            if not memory.admissible:
                prediction = replace(
                    prediction,
                    supported=False,
                    worst_request_fraction=None,
                    mean_gpu_seconds=None,
                    reason=memory.reason,
                    actor_coverage="evaluation_memory_rejected",
                    pruned=False,
                )
            return replace(prediction, actor_memory=memory)
        memory = self.actor_memory_safety(view, layout)
        if memory is not None and (not memory.admissible):
            return CandidatePrediction(
                layout,
                False,
                None,
                None,
                self._work_ids,
                memory.reason,
                self.source_ids,
                actor_coverage="historical_memory_rejected",
                actor_memory=memory,
            )
        prediction = self._evaluate(view, forecasts, layout, bound=bound)
        return (
            replace(prediction, actor_memory=memory)
            if memory is not None
            else prediction
        )

    def _evaluate(self, view, forecasts, layout, *, bound=None, cost_models=None):
        cost_models = cost_models or self.cost_models
        self.prepare_context(view, forecasts)
        work_ids = self._work_ids
        if bound is not None and bound.context_id != self._context_id:
            bound = None
        if self.handoff.mode == "cold":
            bound = None
        costs, fractions, window_costs = ([], [], [])
        running_costs = []
        switching_costs, horizons, window_stats, window_results = ([], [], [], [])
        historical = self._historical
        coverage = "legacy_sampled_execution"
        known_batch = self._known_batch
        supply = (
            tuple(
                self.actor_supply(view, f, layout, cost_models=cost_models)
                for f in forecasts
            )
            if historical
            else ()
        )
        if historical:
            if known_batch:
                try:
                    if not cost_models.actor(layout, known_batch):
                        raise ValueError("Actor model supplied no update stages")
                except ValueError as exc:
                    return CandidatePrediction(
                        layout,
                        False,
                        None,
                        None,
                        work_ids,
                        str(exc),
                        self.cost_models.source_ids,
                        actor_coverage="known_batch_rejected",
                        actor_supply=supply,
                    )
                coverage = "known_complete_batch_capacity_checked"
        past = self._past
        if self.prune_irreversible_slo and (not historical):
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
                        self.cost_models.source_ids,
                    )
        for index, forecast in enumerate(forecasts):
            horizon = forecast.input_closes_at - view.now
            running_cost = horizon * (layout.actor_gpus + layout.rollout_gpus)
            try:
                prepared = self._prepared.get(index)
                if prepared is None:
                    prepared = ReplayInputs.prepare(
                        view,
                        forecast,
                        self.config.max_new_tokens,
                        self.config.max_context_tokens,
                    )
                    self._prepared[index] = prepared
                    self.input_preparations += 1
                simulation = self._from_forecast(
                    view,
                    forecast,
                    cost_models,
                    self.config,
                    self.handoff,
                    self.slo,
                    _prepared=prepared,
                )
                simulation._forecast_target = layout
                if historical:
                    simulation.input_closed = False
                if bound is not None:
                    simulation._candidate_guard = _CandidateGuard(
                        bound,
                        past,
                        self._remaining[index],
                        sum(costs),
                        min((f for f in fractions if f is not None), default=1.0),
                        len(forecasts),
                        fixed_running_cost=running_cost if historical else None,
                    )
                if self.prune_irreversible_slo and (not historical):
                    simulation._forecast_slo_guard = _ForecastSLOGuard(
                        past, self._remaining[index]
                    )
                if layout != view.layout:
                    simulation.decision_pending = True
                    simulation.schedule_at(
                        view.now + self.delay, "decision_apply", layout, priority=9
                    )
                if historical:
                    result = simulation._run(until=forecast.input_closes_at)
                    window_costs.append(result["total_gpu_seconds"])
                    if self.handoff.mode == "cold":
                        # A rebuild must not win merely by leaving target GPUs
                        # unallocated briefly. Keep the measured integral in
                        # window_gpu_seconds; selection reserves the target for
                        # the full horizon, plus any larger actual residency.
                        running_cost = max(running_cost, result["total_gpu_seconds"])
                    window_results.append(result)
                    window_stats.append(
                        rolling_slo(
                            replace(
                                view,
                                now=simulation.now,
                                requests=tuple(
                                    (
                                        r.view.snapshot()
                                        for key, r in simulation.requests.items()
                                        if key not in past
                                    )
                                ),
                            ),
                            self.slo,
                            horizon,
                        )
                    )
                    if result["status"] == "running" and (
                        simulation.pending_handoff or simulation.decision_pending
                    ):
                        result = simulation._run(stop_after_handoff=True)
                else:
                    result = simulation._run()
            except ValueError as exc:
                return CandidatePrediction(
                    layout,
                    False,
                    None,
                    None,
                    work_ids,
                    str(exc),
                    self.cost_models.source_ids,
                    actor_supply=supply,
                )
            if result["status"] == "candidate_pruned":
                guard = simulation._candidate_guard
                return CandidatePrediction(
                    layout,
                    False,
                    None,
                    None,
                    work_ids,
                    result["reason"],
                    self.source_ids,
                    actor_coverage=coverage,
                    pruned=True,
                    fraction_upper_bound=guard.upper,
                    mean_cost_lower_bound=guard.lower,
                    pruning_incumbent=(
                        bound.best.layout.key,
                        bound.best.worst_request_fraction,
                        bound.best.mean_gpu_seconds,
                    ),
                    actor_supply=supply,
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
                    self.cost_models.source_ids,
                    actor_supply=supply,
                )
            if historical and (not known_batch):
                changes_actor = layout.actor_key != view.layout.actor_key
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
                        and (row["committed_at"] <= forecast.input_closes_at)
                        for row in simulation.training_records
                    )
                    for e in simulation.events
                )
                if changes_actor and (not exercised):
                    return CandidatePrediction(
                        layout,
                        False,
                        None,
                        None,
                        work_ids,
                        "Actor capacity unknown: no observed full batch or post-cutover complete update",
                        self.cost_models.source_ids,
                        actor_coverage="unknown_no_complete_batch",
                        actor_supply=supply,
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
                running_costs.append(running_cost)
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
            "Fixed serving window; cold score = max(actual residency, target reservation); target tail not scored"
            if historical and self.handoff.mode == "cold"
            else "Fixed serving window; score = switching GPU seconds + window seconds * target GPU count; target tail not scored"
            if historical
            else "Sampled arrivals close at horizon; all sampled serving and full training batches drain afterwards",
            self.cost_models.source_ids,
            sum(window_costs) / len(window_costs) if historical else None,
            0.0 if historical else None,
            coverage,
            switching_gpu_seconds=sum(switching_costs) / len(costs)
            if historical
            else None,
            running_gpu_seconds_per_second=sum(running_costs) / sum(horizons)
            if historical
            else None,
            evaluation_seconds=sum(horizons) / len(costs) if historical else None,
            running_gpu_seconds=sum(running_costs) / len(costs) if historical else None,
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
            actor_supply=supply,
            admission_credits_start=(self.config.staleness + view.provider_version + 1)
            * self.config.batch_size
            - sum(e.admitted for e in view.episodes)
            if historical
            else None,
            admission_credits_end=sum(r["admission_credits"] for r in window_results)
            / len(window_results)
            if historical
            else None,
            admission_credit_blocked_seconds=sum(
                r["admission_credit_blocked_seconds"] for r in window_results
            )
            / len(window_results)
            if historical
            else None,
        )

    def search(self, view, forecasts, layouts, *, preferred=(), trigger="pressure"):
        return self.optimizer.search(
            self, view, forecasts, layouts, preferred=preferred, trigger=trigger
        )

    def evaluate_many(self, view, forecasts, layouts):
        return self.optimizer.evaluate_many(self, view, forecasts, layouts)

    @property
    def historical(self):
        return self._historical

    @property
    def context_id(self):
        return self._context_id

    @property
    def parallel_available(self):
        return (
            self.executor is not None
            and getattr(self.cost_models, "worker_spec", None) is not None
        )

    def order_candidates(self, view, forecasts, layouts, preferred=()):
        self.prepare_context(view, forecasts)
        actor_scores, rollout_scores = ({}, {})
        turns = tuple(t for e in self._known_batch for t in e)[-16:]

        def score(layout):
            actor_key = layout.actor_key
            rollout_key = layout.rollout_key
            try:
                if actor_key not in actor_scores:
                    actor_scores[actor_key] = (
                        sum(
                            (
                                s
                                for _, s in self.cost_models.actor(
                                    layout, self._known_batch
                                )
                            )
                        )
                        if self._known_batch
                        else 0.0
                    )
                if rollout_key not in rollout_scores:
                    cfg = self.cost_models.rollout_config(layout)
                    seconds = 0.0
                    for t in turns:
                        q = min(t.input_tokens, cfg.max_batch_tokens)
                        prefill = self.cost_models.service(
                            layout,
                            0,
                            (
                                ServiceSlice(
                                    "rank", 0, q, "prefill", q == t.input_tokens
                                ),
                            ),
                        )
                        decode = self.cost_models.service(
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
                        * self.config.batch_size
                        / layout.rollout_replicas
                    )
                return max(actor_scores[actor_key], rollout_scores[rollout_key])
            except ValueError:
                return math.inf

        return candidate_order(layouts, view.layout, preferred, score)

    def evaluate_wave(self, view, forecasts, indexed, *, bound=None, chunk_size=2):
        """Execute a frozen candidate wave; the optimizer owns which wave to request."""
        from areal.elastic.contracts import CandidateWave

        self.prepare_context(view, forecasts)
        indexed = tuple(indexed)
        if not self.parallel_available or len(indexed) <= 1:
            return CandidateWave(
                tuple((i for i, _ in indexed)),
                tuple(
                    (self.evaluate(view, forecasts, c, bound=bound) for _, c in indexed)
                ),
            )
        from areal.elastic.reconfigurator.parallel import evaluate_chunk

        payload = (
            "reference",
            self.cost_models.worker_spec,
            self.source_ids,
            self.config,
            self.handoff,
            self.slo,
            self.delay,
            self.prune_irreversible_slo,
            view,
            forecasts,
            self.context_id,
            self.actor_memory_reserve_fraction,
            self.actor_memory_scope,
        )
        if bound is not None:
            payload += (bound,)
        chunks = tuple(
            self.executor.map(
                evaluate_chunk,
                (
                    (payload, indexed[i : i + chunk_size])
                    for i in range(0, len(indexed), chunk_size)
                ),
            )
        )
        return CandidateWave(
            tuple(i for chunk in chunks for i in chunk.indices),
            tuple(p for chunk in chunks for p in chunk.predictions),
            tuple(sorted({chunk.pid for chunk in chunks})),
            sum(chunk.task_cpu_seconds for chunk in chunks),
            sum(chunk.input_preparations for chunk in chunks),
            len(chunks),
        )

    def _initialize(
        self,
        episodes: tuple[EpisodeSample, ...],
        initial: ExecutionConfig | OnlineLayout,
        cost_models: CostModels,
        config: SimulationConfig,
        handoff: HandoffConfig,
        slo=None,
        *,
        input_closes_at: float,
        reconfigurator: object | None = None,
        event_sink: Callable[[dict], None] | None = None,
        start_time: float = 0.0,
        candidate_executor: Executor | None = None,
        _work: WorkInputs | None = None,
    ):
        initial = getattr(cost_models, "resolve", lambda c: c)(initial)
        self._logical = initial
        self._native_step = None
        self._model_gate = "open"
        self._drain_requests = set()
        self._admission_open = True
        self._checkpoint_version = handoff.native_checkpoint_version
        if config.batch_size % initial.actor_dp:
            raise ValueError("Actor DP must divide the episode batch")
        work = _work or WorkInputs.prepare(episodes, input_closes_at, start_time)
        self._truth = work.truth
        self.cost_models, self.config, self.handoff, self.slo = (
            cost_models,
            config,
            handoff,
            slo,
        )
        self.now = start_time
        if initial.actor_gpus + initial.rollout_gpus > self.handoff.steady_gpu_budget:
            raise ValueError("Initial layout exceeds the steady GPU budget")
        self.pool = GPUAllocator(handoff.node_gpus, now=start_time)
        self.link = SerializedLink(handoff.effective_bandwidth_GBps, start_time)
        self.deployments = {
            0: _Deployment(0, "actor", initial, "active", 0),
            1: _Deployment(1, "rollout", initial, "active", 0),
        }
        if (
            self.pool.acquire(
                "0",
                "actor",
                initial.actor_gpus,
                tp=initial.actor_tp,
                spread=isinstance(initial, ExecutionConfig),
            )
            is None
            or self.pool.acquire(
                "1",
                "rollout",
                initial.rollout_gpus,
                tp=initial.rollout_tp,
                spread=isinstance(initial, ExecutionConfig),
            )
            is None
        ):
            raise ValueError(
                "Initial deployment does not fit the explicit resource pool"
            )
        self.actor_id, self.rollout_id, self.next_deployment = (0, 1, 2)
        self._last_layout = initial
        self.episodes: dict[str, _Episode] = {}
        self.arrival_order: list[str] = []
        self.requests: dict[str, _Request] = {}
        self._live_requests: dict[str, _Request] = {}
        self._unassigned: dict[str, _Request] = {}
        self._tool_episodes: dict[str, _Episode] = {}
        self._episode_order: dict[str, int] = {}
        self._active_episodes = self._completed_episodes = self._admission_cursor = 0
        self.admission_credit_blocked_seconds = 0.0
        self.actor_batch: tuple[str, ...] = ()
        self.actor_wait: ActorWaitView | None = None
        self.actor_wait_seconds = 0.0
        self._forecast_target: ExecutionConfig | OnlineLayout | None = None
        self.actor_stages: tuple[tuple[str, float], ...] = ()
        self.actor_phase = 0
        self.actor_phase_started = start_time
        self.consumed = self.training_version = self.provider_version = 0
        self.input_closed = False
        self.publication = "idle"
        self.pending_handoff: dict | None = None
        self._handoff_old_costs: dict[str, float] = {}
        self.decision_pending = False
        self.reconfigurator = reconfigurator
        self.sink = event_sink
        self.events: list[dict] = []
        self.decisions: list[dict] = []
        self.training_records: list[dict] = []
        self.counters = Counter()
        self.heap: list[tuple[float, int, int, str, object]] = list(work.heap)
        self.sequence = work.sequence
        self.done = False
        self.processed_events = 0
        self.status, self.reason = ("running", "")
        self.training_limit: int | None = None
        self.candidate_executor = candidate_executor
        self.decode_steps_skipped = self.decode_segments = 0
        self.decode_plan_builds = self.decode_plan_hits = 0
        self.decode_batch_reuses = 0
        self.local_service_ticks = 0
        self.decode_run_reuses = 0
        self.control = None
        self._restoring_actor = None
        self.emit("allocation", leases=dict(self.pool.leases))
        self._forecast_slo_guard = None
        self._candidate_guard = None
        if handoff.mode == "cold":
            self.execution_layout = handoff.native_source or self._physical_source()
            if handoff.native_source is not None:
                source = handoff.native_source
                if (
                    initial.actor_gpus != source.world_size
                    or initial.rollout_gpus != len(source.rollout_devices)
                ):
                    raise ValueError(
                        "Native initial state disagrees with the physical source layout"
                    )
                self.pool.leases = {}
                self.pool.peak_gpus = 0
                self.pool.acquire_exact(
                    "0",
                    "actor",
                    self._positions(source.actor_devices[: source.world_size]),
                )
                self.pool.acquire_exact(
                    "1", "rollout", self._positions(source.rollout_devices)
                )
                self.events[0]["leases"] = dict(self.pool.leases)

    @property
    def layout(self):
        if self.handoff.mode == "cold":
            return self._logical
        if self.rollout_id not in self.deployments:
            return self._last_layout
        actor = (
            self.deployments[self.actor_id].layout
            if self.actor_id in self.deployments
            else self._last_layout
        )
        rollout = self.deployments[self.rollout_id].layout
        self._last_layout = actor.with_rollout(rollout)
        return self._last_layout

    @classmethod
    def _from_forecast(
        cls,
        view: OnlineObservation,
        forecast: Forecast,
        cost_models: CostModels,
        config: SimulationConfig,
        handoff: HandoffConfig,
        slo=None,
        *,
        _prepared: ReplayInputs | None = None,
        **simulation_options,
    ):
        if view.busy:
            raise ValueError(
                "Forecast branching waits for a stable deployment/publication boundary"
            )
        prepared = _prepared or ReplayInputs.prepare(
            view, forecast, config.max_new_tokens, config.max_context_tokens
        )
        if prepared.view is not view or prepared.forecast is not forecast:
            raise ValueError(
                "Prepared inputs belong to a different observation/forecast"
            )
        simulation = cls.__new__(cls)
        simulation._initialize(
            forecast.episodes,
            view.layout,
            cost_models,
            config,
            handoff,
            slo,
            input_closes_at=forecast.input_closes_at,
            start_time=view.now,
            _work=prepared.work,
            **simulation_options,
        )
        if not view.actor.running:
            del simulation.deployments[0]
        owners = {
            view.actor_deployment: (
                "0",
                "actor",
                view.layout.actor_gpus,
                view.layout.actor_tp,
            ),
            view.rollout_deployment: (
                "1",
                "rollout",
                view.layout.rollout_gpus,
                view.layout.rollout_tp,
            ),
        }
        leases = {}
        occupied = set()
        for identity, role, devices in view.leases:
            if identity not in owners:
                raise ValueError(
                    "Stable forecast observation contains an unknown GPU lease"
                )
            owner, expected_role, count, tp = owners[identity]
            if (
                role != expected_role
                or len(devices) != count
                or len({node for node, _ in devices})
                > (
                    len(handoff.node_gpus)
                    if isinstance(view.layout, ExecutionConfig) or count > 8
                    else 1
                )
                or any(
                    len({node for node, _ in devices[i : i + tp]}) != 1
                    for i in range(0, count, tp)
                )
            ):
                raise ValueError(
                    "Forecast placement disagrees with the current deployment"
                )
            for node, rank in devices:
                if (
                    (node, rank) in occupied
                    or node < 0
                    or node >= len(handoff.node_gpus)
                    or (not 0 <= rank < handoff.node_gpus[node])
                ):
                    raise ValueError(
                        "Invalid or overlapping visible TP-local GPU leases"
                    )
                occupied.add((node, rank))
            leases[owner] = (role, devices)
        if set(leases) != ({"0", "1"} if view.actor.running else {"1"}):
            raise ValueError("Forecast requires complete current GPU placement")
        simulation.pool.leases = leases
        simulation.pool.peak_gpus = len(occupied)
        simulation.events[0]["leases"] = dict(leases)
        simulation.input_closed = view.input_closed
        simulation.consumed = view.consumed_episodes
        simulation.training_version = view.training_version
        simulation.provider_version = view.provider_version
        if view.actor.waiting is not None:
            simulation.actor_wait = replace(view.actor.waiting, deployment=0)
        for deployment in simulation.deployments.values():
            deployment.version = view.provider_version
        for episode in view.episodes:
            simulation.episodes[episode.episode_id] = _Episode(
                episode.episode_id,
                episode.admitted,
                episode.completed_at,
                list(episode.completed_turns),
                episode.current.request_id if episode.current else None,
                episode.tool,
            )
            simulation.arrival_order.append(episode.episode_id)
        for current, reservation, values in prepared.requests:
            simulation.requests[current.request_id] = _Request(
                current, reservation, len(simulation.requests), values=values
            )
        simulation._live_requests = {
            key: request
            for key, request in simulation.requests.items()
            if request.view.completed_at is None
        }
        for key, request in simulation._live_requests.items():
            current = request.view
            if current.deployment is None:
                simulation._unassigned[key] = request
            else:
                simulation.deployments[current.deployment].replicas[
                    current.replica
                ].add(request)
        simulation._tool_episodes = {
            key: episode
            for key, episode in simulation.episodes.items()
            if episode.tool is not None
        }
        simulation._episode_order = {
            key: index for index, key in enumerate(simulation.arrival_order)
        }
        simulation._active_episodes = prepared.active
        simulation._completed_episodes = prepared.completed
        simulation._admission_cursor = prepared.admission_cursor
        for service in view.services:
            group = simulation.deployments[1].replicas[service.replica]
            slices = tuple(ServiceSlice(*s) for s in service.slices)
            seconds = cost_models.service(view.layout, group.kv_tokens, slices)
            simulation.deployments[1].busy[service.replica] = service
            simulation.schedule_at(
                max(view.now, service.started_at + seconds),
                "service_complete",
                (1, service.replica),
            )
        for episode in view.episodes:
            if episode.tool is not None and episode.tool.started_at is not None:
                turn = simulation._truth[episode.episode_id].turns[
                    len(episode.completed_turns)
                ]
                simulation.schedule_at(
                    max(view.now, episode.tool.started_at + float(turn.tool_seconds)),
                    "tool_complete",
                    episode.episode_id,
                )
        if view.actor.batch:
            simulation.actor_batch = view.actor.batch
            simulation.actor_stages = cost_models.actor(
                view.layout,
                tuple(tuple(simulation.episodes[e].past) for e in view.actor.batch),
            )
            simulation.actor_phase = view.actor.phase
            simulation.actor_phase_started = view.actor.phase_started_at
            simulation.training_records.append(
                dict(
                    restored=True,
                    step=view.training_version,
                    episodes=view.actor.batch,
                    committed_at=None,
                )
            )
            seconds = simulation.actor_stages[view.actor.phase][1]
            simulation.schedule_at(
                max(view.now, view.actor.phase_started_at + seconds),
                "actor_phase_complete",
            )
        if handoff.mode == "cold":
            simulation.execution_layout = (
                handoff.native_source or simulation._physical_source()
            )
        return simulation

    def _base_result(self):
        self.flush_decode()
        rows = tuple(r.view for r in self.requests.values())
        complete = [r for r in rows if r.completed_at is not None]
        result = dict(
            status=self.status,
            final_layout=asdict(self.layout),
            reason=self.reason,
            seconds=self.now,
            arrived_episodes=len(self.arrival_order),
            completed_episodes=sum(
                e.completed_at is not None for e in self.episodes.values()
            ),
            completed_updates=self.training_version,
            published_version=self.provider_version,
            trained_episodes=self.consumed,
            actor_wait=asdict(self.actor_wait) if self.actor_wait else None,
            actor_wait_seconds=self.actor_wait_seconds
            + (self.now - self.actor_wait.since if self.actor_wait else 0.0),
            retained_episodes=len(self.arrival_order) - self.consumed,
            admission_credits=self.admission_credits(),
            admission_credit_blocked_seconds=self.admission_credit_blocked_seconds,
            ready_requests=len(rows),
            completed_requests=len(complete),
            actor_gpu_seconds=self.pool.gpu_seconds["actor"],
            rollout_gpu_seconds=self.pool.gpu_seconds["rollout"],
            total_gpu_seconds=sum(self.pool.gpu_seconds.values()),
            peak_gpus=self.pool.peak_gpus,
            remaining_leases=dict(self.pool.leases),
            counters=dict(self.counters),
            source_ids=self.cost_models.source_ids,
            assumptions=(
                "GPU-token timing; HTTP/network response tails not measured",
                "Decode-first batches; declared generation-budget KV reservation; no prefix reuse",
                "Publication preserves request progress, unlike native abort/recompute",
                "Immutable published snapshots assumed; complete Actor state handoff at a single-writer boundary",
                "Startup/state/publication transfers serialize on one assumed effective shared link",
                self.handoff.provenance,
                *getattr(self.cost_models, "scenario_notes", ()),
            ),
        )
        if self.control is not None:
            result["control"] = dict(clock="measured", **self.control.stats)
        return result

    def reconfiguration_allowed(self):
        return self._restoring_actor is None

    def emit(self, kind: str, **data):
        self.counters[kind] += 1
        if kind == "service_batch":
            return
        event = dict(time=self.now, kind=kind, **data)
        if self.sink:
            self.sink(event)
        if kind != "service_batch":
            self.events.append(event)

    def schedule_at(self, when: float, kind: str, data=None, *, priority: int = 0):
        if when < self.now - 1e-09:
            raise ValueError("An event cannot be scheduled in the past")
        self.sequence += 1
        heapq.heappush(
            self.heap, (max(self.now, when), priority, self.sequence, kind, data)
        )

    def _run(
        self,
        until: float | None = None,
        *,
        stop_after_handoff: bool = False,
        until_update: int | None = None,
    ) -> dict:
        paused = False
        try:
            self.tick()
            while not self.done:
                if (
                    until_update is not None
                    and self.provider_version >= until_update
                    and (self.publication == "idle")
                ):
                    paused = True
                    break
                if (
                    stop_after_handoff
                    and (not self.pending_handoff)
                    and (not self.decision_pending)
                ):
                    paused = True
                    break
                if not self.heap:
                    self.status, self.reason = (
                        "blocked_actor_memory" if self.actor_wait else "incomplete",
                        self.actor_wait.reason
                        if self.actor_wait
                        else "No event can make progress",
                    )
                    break
                from areal.elastic.simulation_primitives import advance_decode

                advance_decode(self, until)
                when = self.heap[0][0]
                if until is not None and when > until:
                    self.advance(until)
                    paused = True
                    break
                if when > self.config.max_simulated_seconds:
                    self.status, self.reason = (
                        "incomplete",
                        "Explicit simulated-time budget reached"
                        + (
                            "; Actor waiting: " + self.actor_wait.reason
                            if self.actor_wait
                            else ""
                        ),
                    )
                    break
                self.advance(when)
                local_services = set()
                service_only = (self.pending_handoff is None) and (
                    self.publication == "idle"
                )
                completed = self.counters["request_complete"]
                while self.heap and self.heap[0][0] == self.now:
                    _, _, _, kind, data = heapq.heappop(self.heap)
                    if kind == "service_complete":
                        local_services.add(data)
                    else:
                        service_only = False
                    self.handle(kind, data)
                    self.processed_events += 1
                    if self.processed_events > self.config.max_events:
                        self.status, self.reason = (
                            "incomplete",
                            "Explicit event budget reached",
                        )
                        self.done = True
                        break
                if not self.done:
                    self.tick(
                        local_services
                        if service_only
                        and self.counters["request_complete"] == completed
                        else None
                    )
                    self.check_prediction_bounds()
            if self.status == "running" and (not paused):
                self.status, self.reason = (
                    "complete",
                    "Input closed, serving drained and all full batches trained",
                )
        except (ValueError, ArithmeticError) as exc:
            self.status, self.reason = ("unsupported", str(exc))
        return self._result()

    def handle(self, kind: str, data):
        if kind == "arrival":
            self.arrive(data)
        elif kind == "input_closed":
            self.input_closed = True
            self.emit("input_closed")
        elif kind == "service_complete":
            self.finish_service(*data)
        elif kind == "tool_complete":
            self.finish_tool(data)
        elif kind == "actor_phase_complete":
            self.finish_actor_phase()
        elif kind == "publication_complete":
            self.finish_publication(data)
        else:
            self.handle_transition_event(kind, data)

    def arrive(self, identity: str):
        truth = self._truth[identity]
        self.episodes[identity] = _Episode(identity)
        self._episode_order[identity] = len(self.arrival_order)
        self.arrival_order.append(identity)
        self.new_request(self.episodes[identity], truth.available_at)
        self.emit("episode_arrival", episode=identity, family=truth.family)

    def new_request(self, episode: _Episode, ready_at: float):
        index = len(episode.past)
        turn = self._truth[episode.episode_id].turns[index]
        if (
            turn.output_tokens > self.config.max_new_tokens
            or turn.input_tokens + turn.output_tokens > self.config.max_context_tokens
        ):
            raise ValueError(
                "Workload exceeds the explicitly configured generation/context budget"
            )
        identity = f"{episode.episode_id}/request-{index}"
        view = RequestView(
            identity, episode.episode_id, index, turn.input_tokens, ready_at
        )
        reservation = min(
            self.config.max_new_tokens,
            self.config.max_context_tokens - turn.input_tokens,
        )
        self.requests[identity] = _Request(view, reservation, len(self.requests))
        self._live_requests[identity] = self.requests[identity]
        self._unassigned[identity] = self.requests[identity]
        episode.current = identity
        self.emit(
            "request_ready",
            request=identity,
            input_tokens=turn.input_tokens,
            ready_at=ready_at,
        )

    def tick(self, local_services: set[tuple[int, int]] | None = None):
        if local_services is not None:
            self.local_service_ticks += 1
            self.service(local_services)
            return
        if self.publication == "pending" and (
            not any(d.busy for d in self.deployments.values())
        ):
            self.start_publication()
        self.before_work()
        if self.publication == "idle":
            self.admit()
            self.start_tools()
            self.service()
            self.start_actor()
        else:
            self.start_tools()
        self.after_work()
        if (
            self.input_closed
            and len(self.episodes) == len(self._truth)
            and (self._completed_episodes == len(self.episodes))
            and (not self.actor_batch)
            and (self.publication == "idle")
            and (self.pending_handoff is None)
            and (not self.decision_pending)
            and (self._restoring_actor is None)
            and (len(self.arrival_order) - self.consumed < self.config.batch_size)
        ):
            if self.control is not None:
                self.control.cancel()
            for identity in tuple(self.deployments):
                self.release_deployment(identity)
            self.done = True

    def admission_credits(self):
        return (
            (self.config.staleness + self.provider_version + 1) * self.config.batch_size
            - self._completed_episodes
            - self._active_episodes
        )

    def advance(self, when):
        if (
            self._admission_cursor < len(self.arrival_order)
            and self.admission_credits() <= 0
        ):
            self.admission_credit_blocked_seconds += when - self.now
        self.pool.advance(when)
        self.now = when

    def _base_admit(self):
        active = self._active_episodes
        credits = self.admission_credits()
        while self._admission_cursor < len(self.arrival_order):
            identity = self.arrival_order[self._admission_cursor]
            episode = self.episodes[identity]
            limit = (
                getattr(self.layout, "rollout_max_concurrent_rollouts", None)
                or self.config.client_concurrency
            )
            if active >= limit or credits <= 0:
                break
            self._admission_cursor += 1
            episode.admitted = True
            active += 1
            credits -= 1
            self.emit("episode_admitted", episode=identity)
        self._active_episodes = active

    def _cold_admit(self):
        if self._admission_open:
            self._base_admit()

    def admit(self, *args, **kwargs):
        if self.handoff.mode == "cold":
            return self._cold_admit(*args, **kwargs)
        return self._base_admit(*args, **kwargs)

    def _base_service(self, local_services: set[tuple[int, int]] | None = None):
        target = self.deployments[self.rollout_id]
        cfg = self.cost_models.rollout_config(target.layout)
        unassigned = self._unassigned.values() if local_services is None else ()
        for request in sorted(unassigned, key=lambda r: r.order):
            view = request.view
            if not self.episodes[view.episode_id].admitted:
                continue
            replica = scheduler_for(cfg.backend).choose_replica(
                cfg, request, target.replicas
            )
            if replica is not None:
                view.deployment, view.replica = (target.identity, replica)
                target.replicas[replica].add(request)
                del self._unassigned[view.request_id]
                self.emit(
                    "request_assigned",
                    request=view.request_id,
                    deployment=target.identity,
                    replica=replica,
                )
        for deployment in tuple(self.deployments.values()):
            if deployment.role != "rollout" or deployment.status not in {
                "active",
                "draining",
            }:
                continue
            cfg = self.cost_models.rollout_config(deployment.layout)
            for replica in range(cfg.replicas):
                if (
                    local_services is not None
                    and (deployment.identity, replica) not in local_services
                ):
                    continue
                if replica in deployment.busy:
                    continue
                group = deployment.replicas[replica]
                if not group.rows:
                    continue
                plan = group.decode_plan
                continuation = (
                    (plan is not None)
                    and plan.continuation
                    and (plan.revision == group.revision)
                    and (plan.version == deployment.version)
                    and (plan.limit > 0)
                    and (plan.first_seconds > 0)
                    and (
                        group.kv_tokens + len(plan.batch.slices)
                        <= cfg.kv_capacity_tokens
                    )
                )
                if continuation:
                    duration = plan.first_seconds
                    rows = tuple(
                        (
                            (key, context + 1, query, phase, emits)
                            for key, context, query, phase, emits in plan.batch.slices
                        )
                    )
                    self.decode_batch_reuses += 1
                else:
                    group.flush_decode()
                    slices, _ = scheduler_for(cfg.backend).batch(
                        cfg, list(group.rows.values())
                    )
                    duration = self.cost_models.service(
                        deployment.layout, group.kv_tokens, slices
                    )
                    rows = tuple(
                        (
                            s.request_id,
                            s.context_tokens,
                            s.query_tokens,
                            s.kind,
                            s.emit_token,
                        )
                        for s in slices
                    )
                batch = ServiceView(replica, self.now, rows, deployment.version)
                if continuation:
                    plan.batch = batch
                    plan.continuation = False
                deployment.busy[replica] = batch
                self.schedule_at(
                    self.now + duration,
                    "service_complete",
                    (deployment.identity, replica),
                )

    def _cold_service(self, local_services=None):
        if (
            self._model_gate in {"closed", "paused"}
            or self.rollout_id not in self.deployments
        ):
            return
        if self._model_gate != "draining":
            return self._base_service(local_services)
        hidden = {
            k: r for k, r in self._unassigned.items() if k not in self._drain_requests
        }
        self._unassigned = {
            k: r for k, r in self._unassigned.items() if k in self._drain_requests
        }
        try:
            self._base_service(local_services)
        finally:
            self._unassigned.update(hidden)

    def service(self, *args, **kwargs):
        if self.handoff.mode == "cold":
            return self._cold_service(*args, **kwargs)
        return self._base_service(*args, **kwargs)

    def finish_service(self, identity: int, replica: int):
        deployment = self.deployments[identity]
        batch = deployment.busy.pop(replica)
        group = deployment.replicas[replica]
        plan = group.decode_plan
        defer = (
            (plan is not None)
            and (plan.batch is batch)
            and (plan.revision == group.revision)
            and (plan.version == deployment.version)
            and (plan.limit > 1)
        )
        if defer:
            group.defer_decode(plan, 1, self.now, self.now, 0.0)
            group.kv_tokens += len(batch.slices)
            self.counters["token_generated"] += len(batch.slices)
        else:
            group.flush_decode()
            for key, _, query, phase, emit_token in batch.slices:
                request = self.requests[key]
                view = request.view
                kv_delta = query if phase == "prefill" else 0
                if phase == "prefill":
                    view.prefilled += query
                    group.revision += 1
                if emit_token:
                    if view.generated:
                        kv_delta += 1
                    view.add_tokens(
                        1,
                        batch.version,
                        self.now,
                        self.now - view.last_token_at
                        if view.last_token_at is not None
                        else 0.0,
                    )
                    self.counters["token_generated"] += 1
                group.kv_tokens += kv_delta
                truth = self._truth[view.episode_id].turns[view.turn]
                if view.generated == truth.output_tokens:
                    self.finish_request(request)
        if (
            (plan is not None)
            and (plan.batch is batch)
            and (plan.revision == group.revision)
            and (plan.version == deployment.version)
            and (plan.limit > 0)
        ):
            plan.first_seconds += plan.increment
            plan.limit -= 1
            plan.continuation = plan.limit > 0
        self.emit(
            "service_batch",
            deployment=identity,
            replica=replica,
            version=batch.version,
            start=batch.started_at,
            end=self.now,
            slices=batch.slices,
        )

    def finish_request(self, request: _Request):
        view = request.view
        view.completed_at = self.now
        self.deployments[view.deployment].replicas[view.replica].remove(request)
        del self._live_requests[view.request_id]
        self.request_finished(view)
        episode = self.episodes[view.episode_id]
        truth = self._truth[view.episode_id].turns[view.turn]
        self.emit(
            "request_complete",
            request=view.request_id,
            deployment=view.deployment,
            generated=view.generated,
            ttft_seconds=view.first_token_at - view.ready_at,
            maximum_tbt_seconds=view.maximum_tbt_seconds,
        )
        if truth.tool_seconds is None:
            episode.past.append(truth)
            episode.current = None
            episode.completed_at = self.now
            self._active_episodes -= 1
            self._completed_episodes += 1
            self.emit("episode_complete", episode=episode.episode_id)
        else:
            episode.tool = ToolView(truth.tool_pool, self.now, None)
            self._tool_episodes[episode.episode_id] = episode

    def start_tools(self):
        waiting = sorted(
            self._tool_episodes.values(),
            key=lambda e: self._episode_order[e.episode_id],
        )
        for pool, capacity in self.config.tool_capacities:
            active = sum(
                e.tool is not None
                and e.tool.pool == pool
                and (e.tool.started_at is not None)
                for e in waiting
            )
            for episode in waiting:
                tool = episode.tool
                if active >= capacity:
                    break
                if tool is None or tool.pool != pool or tool.started_at is not None:
                    continue
                episode.tool = replace(tool, started_at=self.now)
                turn = self._truth[episode.episode_id].turns[len(episode.past)]
                self.schedule_at(
                    self.now + float(turn.tool_seconds),
                    "tool_complete",
                    episode.episode_id,
                )
                self.emit("tool_start", episode=episode.episode_id, pool=pool)
                active += 1

    def finish_tool(self, identity: str):
        episode = self.episodes[identity]
        truth = self._truth[identity].turns[len(episode.past)]
        self.emit(
            "tool_complete",
            episode=identity,
            pool=episode.tool.pool,
            seconds=self.now - episode.tool.started_at,
            output_tokens=truth.tool_output_tokens,
        )
        episode.past.append(truth)
        episode.tool = None
        del self._tool_episodes[identity]
        self.new_request(episode, self.now)

    def _base_start_actor(self):
        if self._restoring_actor is not None:
            return
        if (
            self.training_limit is not None
            and self.training_version >= self.training_limit
        ):
            return
        if self.actor_batch or self.publication != "idle":
            return
        pending = self.pending_handoff
        if pending and (pending.get("actor_ready") or pending.get("rollout_fence")):
            return
        batch = tuple(
            self.arrival_order[self.consumed : self.consumed + self.config.batch_size]
        )
        if len(batch) != self.config.batch_size or any(
            self.episodes[e].completed_at is None for e in batch
        ):
            return
        if self.actor_id not in self.deployments:
            identity = self.next_deployment
            if (
                self.pool.acquire(
                    str(identity),
                    "actor",
                    self.layout.actor_gpus,
                    tp=self.layout.actor_tp,
                    spread=isinstance(self.layout, ExecutionConfig),
                )
                is None
            ):
                raise ValueError("Initial Actor does not fit the declared GPU pool")
            self.next_deployment += 1
            self._restoring_actor = identity
            self.add_restoring_actor(identity)
            self.emit("actor_restore_start", deployment=identity)
            self.schedule_at(
                self.now + self.handoff.actor_startup_seconds,
                "process_started",
                identity,
            )
            return
        wait = self.actor_wait
        if wait and wait.deployment == self.actor_id and (wait.batch == batch):
            self.reject_unrepairable_forecast_wait()
            return
        try:
            stages = self.cost_models.actor(
                self.layout, tuple(tuple(self.episodes[e].past) for e in batch)
            )
        except ActorMemoryRejected as exc:
            self.actor_wait = ActorWaitView(
                batch, self.actor_id, wait.since if wait else self.now, str(exc)
            )
            self.emit(
                "actor_wait",
                deployment=self.actor_id,
                episodes=batch,
                cursor=self.consumed,
                version=self.training_version,
                since=self.actor_wait.since,
                reason=str(exc),
            )
            self.reject_unrepairable_forecast_wait()
            return
        if not stages:
            raise ValueError("Actor model did not supply update stages")
        if wait:
            seconds = self.now - wait.since
            self.actor_wait_seconds += seconds
            self.emit(
                "actor_wait_complete",
                deployment=self.actor_id,
                previous_deployment=wait.deployment,
                episodes=batch,
                cursor=self.consumed,
                version=self.training_version,
                since=wait.since,
                seconds=seconds,
            )
            self.actor_wait = None
        self.actor_batch, self.actor_stages = (batch, stages)
        versions = Counter()
        for request in self.requests.values():
            if request.view.episode_id in batch:
                versions.update(dict(request.view.version_counts))
        self.training_records.append(
            dict(
                step=self.training_version,
                started_at=self.now,
                preupdate_version=self.training_version,
                episodes=batch,
                behavior_versions=dict(versions),
                maximum_version_lag=self.training_version - min(versions),
                committed_at=None,
            )
        )
        self.emit(
            "actor_start",
            deployment=self.actor_id,
            episodes=batch,
            version=self.training_version,
            cursor=self.consumed,
        )
        self.actor_phase = 0
        self.start_actor_phase()

    def _cold_start_actor(self):
        if self.pending_handoff is None and (not self.decision_pending):
            self._base_start_actor()

    def start_actor(self, *args, **kwargs):
        if self.handoff.mode == "cold":
            return self._cold_start_actor(*args, **kwargs)
        return self._base_start_actor(*args, **kwargs)

    def start_actor_phase(self):
        name, seconds = self.actor_stages[self.actor_phase]
        self.actor_phase_started = self.now
        self.emit("actor_phase_start", deployment=self.actor_id, phase=name)
        self.schedule_at(self.now + seconds, "actor_phase_complete")

    def finish_actor_phase(self):
        name, _ = self.actor_stages[self.actor_phase]
        self.emit("actor_phase_complete", deployment=self.actor_id, phase=name)
        self.actor_phase += 1
        if self.actor_phase < len(self.actor_stages):
            self.start_actor_phase()
            return
        self.training_version += 1
        self.deployments[self.actor_id].version = self.training_version
        self.consumed += len(self.actor_batch)
        self.training_records[-1]["committed_at"] = self.now
        self.actor_batch = ()
        self.publication = "pending"
        self.emit("actor_complete", version=self.training_version, cursor=self.consumed)

    def start_publication(self):
        recipients = tuple(
            d
            for d in self.deployments.values()
            if d.role == "rollout" and d.status in {"active", "draining"}
        )
        seconds = self.cost_models.publication(
            self.layout, tuple(d.layout for d in recipients)
        )
        start, end = self.link.reserve(self.now, fitted_seconds=seconds)
        for deployment in recipients:
            deployment.transfers += 1
        self.publication = "running"
        self.emit(
            "publication_start",
            version=self.training_version,
            transfer_start=start,
            recipients=tuple(d.identity for d in recipients),
            ends_at=end,
        )
        self.schedule_at(
            end,
            "publication_complete",
            (self.training_version, tuple(d.identity for d in recipients)),
        )

    def _base_finish_publication(self, data):
        version, identities = data
        for identity in identities:
            deployment = self.deployments[identity]
            for group in deployment.replicas.values():
                group.flush_decode()
            deployment.version = version
            deployment.transfers -= 1
        self.provider_version = version
        self.publication = "idle"
        self.emit("weights_published", version=version, recipients=identities)
        if self.control is not None:
            self.control.publication_complete()

    def _cold_finish_publication(self, data):
        self._base_finish_publication(data)
        if self._native_step == "publish_weights":
            self._complete_step("publish_weights")

    def finish_publication(self, *args, **kwargs):
        if self.handoff.mode == "cold":
            return self._cold_finish_publication(*args, **kwargs)
        return self._base_finish_publication(*args, **kwargs)

    def release_deployment(self, identity: int):
        deployment = self.deployments[identity]
        if deployment.busy or deployment.transfers:
            raise ValueError(
                "Cannot release a deployment with live GPU work or transfers"
            )
        for group in deployment.replicas.values():
            group.flush_decode()
        self.pool.release(str(identity))
        del self.deployments[identity]
        self.emit(
            "deployment_release",
            deployment=identity,
            role=deployment.role,
            leases=dict(self.pool.leases),
        )
        self.deployment_released(identity, deployment)

    def observe(self) -> OnlineObservation:
        self.flush_decode()
        snapshots = {key: r.view.snapshot() for key, r in self.requests.items()}
        episodes = []
        for identity in self.arrival_order:
            episode = self.episodes[identity]
            truth = self._truth[identity]
            current = (
                snapshots[episode.current] if episode.current is not None else None
            )
            episodes.append(
                EpisodeView(
                    identity,
                    truth.family,
                    truth.available_at,
                    episode.admitted,
                    episode.completed_at,
                    tuple(episode.past),
                    current,
                    episode.tool,
                )
            )
        return OnlineObservation(
            self.now,
            self.layout,
            tuple(episodes),
            tuple(snapshots.values()),
            tuple(self.deployments[self.rollout_id].busy.values()),
            ActorView(
                self.actor_batch,
                self.actor_phase,
                self.actor_phase_started,
                self.actor_wait,
                self.actor_id in self.deployments,
            ),
            self.consumed,
            self.training_version,
            self.provider_version,
            self.pending_handoff is not None
            or self.publication != "idle"
            or self.decision_pending
            or (not self.reconfiguration_allowed()),
            self.input_closed,
            self.actor_id,
            self.rollout_id,
            tuple(
                (
                    (int(owner), role, devices)
                    for owner, (role, devices) in sorted(self.pool.leases.items())
                )
            ),
        )

    def flush_decode(self):
        for deployment in self.deployments.values():
            for group in deployment.replicas.values():
                group.flush_decode()

    def _hot_before_work(self):
        self.maybe_actor_handoff()
        self.maybe_rollout_cutover()

    def _cold_before_work(self):
        if self.pending_handoff is None or self._native_step is not None:
            return
        if self.actor_batch or self.publication != "idle":
            return
        steps = self.pending_handoff["steps"]
        if not self.pending_handoff["started"]:
            self.pending_handoff["started"] = True
            if self._checkpoint_version != self.training_version:
                self._seconds("checkpoint")
                steps.insert(0, "checkpoint")
        if not steps:
            self._admission_open = True
            self._model_gate = "open"
            self.pending_handoff = None
            self.emit("handoff_complete", layout=asdict(self.layout), mode="cold")
            if self.reconfigurator is not None:
                self.reconfigurator.switched(self.now)
            return
        self._native_step = steps.pop(0)
        step = self._native_step
        self.emit("cold_phase_start", phase=step)
        if step == "drain_model_calls":
            self._model_gate = "draining"
            self._drain_requests = {
                k
                for k, r in self._live_requests.items()
                if self.episodes[r.view.episode_id].admitted
            }
        elif step == "publish_weights":
            self._model_gate = "paused"
        else:
            if step == "close_admission":
                self._admission_open = False
            if step == "pause_rollout":
                self._model_gate = "paused"
            if step in {"restore_actor", "rebuild_rollout"}:
                self._allocate(step)
            self.schedule_at(
                self.now + self._seconds(step), "cold_phase_complete", step
            )

    def before_work(self, *args, **kwargs):
        if self.handoff.mode == "cold":
            return self._cold_before_work(*args, **kwargs)
        return self._hot_before_work(*args, **kwargs)

    def _hot_after_work(self):
        self.retire_rollouts()
        self.maybe_finish_handoff()

    def _cold_after_work(self):
        if self._native_step == "drain_model_calls" and (
            not self._drain_requests & self._live_requests.keys()
        ):
            self._model_gate = "closed"
            self._complete_step("drain_model_calls")
        elif self._native_step == "publish_weights" and self.publication == "idle":
            if not any(d.busy for d in self.deployments.values()):
                self.start_publication()

    def after_work(self, *args, **kwargs):
        if self.handoff.mode == "cold":
            return self._cold_after_work(*args, **kwargs)
        return self._hot_after_work(*args, **kwargs)

    def check_prediction_bounds(self):
        self.check_forecast_slo()
        self.check_candidate_bound()

    def _hot_reject_unrepairable_forecast_wait(self):
        target = self._forecast_target
        if target is None:
            return
        changes_actor = target.actor_key != self.layout.actor_key
        pending = self.pending_handoff
        repair_planned = self.decision_pending or (
            pending is not None
            and pending["new_actor"] is not None
            and (pending["new_actor"] != self.actor_id)
        )
        if not changes_actor or not repair_planned:
            raise ActorMemoryRejected(
                self.actor_wait.reason
                + "; no planned Actor migration can repair this batch"
            )

    def _cold_reject_unrepairable_forecast_wait(self):
        if (
            self.pending_handoff is not None
            and "restore_actor" in self.pending_handoff["plan"].steps
        ):
            return
        self._hot_reject_unrepairable_forecast_wait()

    def reject_unrepairable_forecast_wait(self, *args, **kwargs):
        if self.handoff.mode == "cold":
            return self._cold_reject_unrepairable_forecast_wait(*args, **kwargs)
        return self._hot_reject_unrepairable_forecast_wait(*args, **kwargs)

    def _hot_deployment_released(self, identity, deployment):
        if (
            self.pending_handoff is not None
            and str(identity) in self._handoff_old_costs
        ):
            self.emit(
                "role_handoff_complete",
                role=deployment.role,
                source=identity,
                target=self.actor_id if deployment.role == "actor" else self.rollout_id,
            )
            if self.reconfigurator is not None:
                self.reconfigurator.role_switched(deployment.role, self.now)

    def _cold_deployment_released(self, identity, deployment):
        if self.pending_handoff is not None and self.reconfigurator is not None:
            self.reconfigurator.role_switched(deployment.role, self.now)

    def deployment_released(self, *args, **kwargs):
        if self.handoff.mode == "cold":
            return self._cold_deployment_released(*args, **kwargs)
        return self._hot_deployment_released(*args, **kwargs)

    def _hot_handle_transition_event(self, kind, data):
        if kind == "decision_apply":
            self.decision_pending = False
            self.request_handoff(data)
        elif kind == "process_started":
            self.load_deployment(data)
        elif kind == "deployment_loaded":
            deployment, version = data
            warmup = (
                self.handoff.actor_warmup_seconds
                if self.deployments[deployment].role == "actor"
                else self.handoff.rollout_warmup_seconds
            )
            self.schedule_at(
                self.now + warmup, "deployment_prepared", (deployment, version)
            )
        elif kind == "deployment_prepared":
            self.prepared(*data)
        elif kind == "rollout_synchronized":
            self.rollout_synchronized(*data)
        elif kind == "actor_synchronized":
            self.actor_synchronized(data)
        else:
            raise ValueError(f"Unknown online event {kind}")

    def _cold_handle_transition_event(self, kind, data):
        if kind == "cold_phase_complete":
            self._complete_step(data)
        elif kind == "cold_continue":
            pass
        else:
            self._hot_handle_transition_event(kind, data)

    def handle_transition_event(self, *args, **kwargs):
        if self.handoff.mode == "cold":
            return self._cold_handle_transition_event(*args, **kwargs)
        return self._hot_handle_transition_event(*args, **kwargs)

    def request_finished(self, view):
        guard = self._forecast_slo_guard
        if (
            guard is not None
            and view.request_id not in guard.excluded
            and self.request_has_failed(view)
        ):
            guard.failed.add(view.request_id)
        bound = self._candidate_guard
        if (
            bound is not None
            and view.request_id not in bound.excluded
            and self.request_has_failed(view)
        ):
            bound.failed.add(view.request_id)

    def _result(self):
        result = self._base_result()
        rows = tuple(r.view for r in self.requests.values())
        passed = [
            r
            for r in rows
            if r.completed_at is not None
            and request_slo_status(r, self.slo, now=r.completed_at) is True
        ]
        result.update(
            jointly_passed_requests=len(passed),
            joint_request_fraction=len(passed) / len(rows) if rows else None,
        )
        return result

    def check_forecast_slo(self):
        guard = self._forecast_slo_guard
        if guard is None or self.done or self.now < guard.check_after:
            return
        guard.check_after = self.now + 1.0
        self.flush_decode()
        failed = len(guard.failed) + sum(
            self.request_has_failed(r.view) for r in self._live_requests.values()
        )
        upper = 1 - failed / guard.total if guard.total else 1.0
        if upper + 1e-12 < self.slo.minimum_request_fraction:
            self.status = "forecast_slo_infeasible"
            self.reason = f"Sampled branch has irreversible SLO failures: {failed}/{guard.total}; joint_fraction_upper_bound={upper:.12g}; stopped before full cost prediction"
            self.done = True

    def request_has_failed(self, view: RequestView):
        return request_slo_status(view, self.slo, now=self.now) is False

    def check_candidate_bound(self):
        guard = self._candidate_guard
        if guard is None or self.done or self.now < guard.check_after:
            return
        guard.check_after = self.now + 1.0
        guard.lower = (
            guard.completed_cost
            + (
                guard.fixed_running_cost + self.switching_gpu_seconds()
                if guard.fixed_running_cost is not None
                else sum(self.pool.gpu_seconds.values())
            )
        ) / guard.samples
        reason = guard.bound.rejection(guard.fraction_cap, guard.lower)
        guard.upper = guard.fraction_cap
        if not reason and guard.bound.fraction_pruning:
            self.flush_decode()
            failed = len(guard.failed) + sum(
                r.view.request_id not in guard.excluded
                and self.request_has_failed(r.view)
                for r in self._live_requests.values()
            )
            guard.upper = min(
                guard.fraction_cap, 1 - failed / guard.total if guard.total else 1.0
            )
            reason = guard.bound.rejection(guard.upper, guard.lower)
        if reason:
            self.status, self.reason, self.done = ("candidate_pruned", reason, True)

    def _hot_request_handoff(self, target: ExecutionConfig | OnlineLayout):
        if not self.reconfiguration_allowed():
            return False
        if target.actor_gpus + target.rollout_gpus > self.handoff.steady_gpu_budget:
            raise ValueError("Target layout exceeds the steady GPU budget")
        if self.pending_handoff is not None or target == self.layout:
            return False
        if self.config.batch_size % target.actor_dp:
            raise ValueError("Target Actor DP must divide the configured episode batch")
        current = self.layout
        if (
            self.reconfigurator is not None
            and self.reconfigurator.config.independent_role_history
        ):
            allowed = self.reconfigurator.allowed_roles(self.now)
            if (
                not allowed["actor"]
                and target.actor_key != current.actor_key
                or (
                    not allowed["rollout"] and target.rollout_key != current.rollout_key
                )
            ):
                return False
        if self.actor_id not in self.deployments:
            return False
        roles = []
        if current.actor_key != target.actor_key:
            roles.append(("actor", target.actor_gpus))
        if current.rollout_key != target.rollout_key:
            roles.append(("rollout", target.rollout_gpus))
        acquired = {}
        previous_peak = self.pool.peak_gpus
        for role, count in roles:
            identity = self.next_deployment
            self.next_deployment += 1
            tp = target.actor_tp if role == "actor" else target.rollout_tp
            if (
                self.pool.acquire(
                    str(identity),
                    role,
                    count,
                    tp=tp,
                    spread=isinstance(target, ExecutionConfig),
                )
                is None
            ):
                for previous in acquired.values():
                    self.pool.release(str(previous))
                self.pool.peak_gpus = previous_peak
                self.emit(
                    "handoff_deferred",
                    target=asdict(target),
                    reason="Insufficient spare GPUs for node-local TP groups",
                )
                return False
            acquired[role] = identity
        self.pending_handoff = dict(
            target=target,
            new_actor=acquired.get("actor"),
            new_rollout=acquired.get("rollout"),
            actor_ready=False,
            actor_syncing=False,
            rollout_fence=False,
            rollout_switched="rollout" not in acquired,
        )
        self._handoff_old_costs = {
            str(
                self.actor_id if role == "actor" else self.rollout_id
            ): self.pool.lease_gpu_seconds.get(
                str(self.actor_id if role == "actor" else self.rollout_id), 0.0
            )
            for role in acquired
        }
        for role, identity in acquired.items():
            self.deployments[identity] = _Deployment(
                identity, role, target, "preparing", self.provider_version
            )
            seconds = (
                self.handoff.actor_startup_seconds
                if role == "actor"
                else self.handoff.rollout_startup_seconds
            )
            self.schedule_at(self.now + seconds, "process_started", identity)
        self.emit(
            "handoff_start",
            source=asdict(current),
            target=asdict(target),
            leases=dict(self.pool.leases),
        )
        return True

    def _cold_request_handoff(self, target):
        if self.pending_handoff is not None or target == self.layout:
            return False
        if (
            target.actor_gpus + target.rollout_gpus > self.handoff.steady_gpu_budget
            or self.config.batch_size % target.actor_dp
        ):
            raise ValueError(
                "Cold target must fit the steady GPU budget and logical batch"
            )
        physical = self._target(target)
        plan = ReconfigurationPlanner().plan(
            self.execution_layout,
            physical,
            actor_running=self.actor_id in self.deployments,
            force_actor=target.actor_key != self.layout.actor_key,
            force_rollout=target.rollout_key != self.layout.rollout_key,
        )
        steps = plan.steps
        for step in steps:
            if step not in {"drain_model_calls", "publish_weights"}:
                self._seconds(step)
        self.pending_handoff = dict(
            target=target,
            plan=plan,
            steps=list(steps),
            new_actor=None,
            new_rollout=None,
            actor_ready=False,
            rollout_fence=True,
            started=False,
        )
        self.emit(
            "cold_handoff_start",
            source=asdict(self.layout),
            target=asdict(target),
            plan=asdict(plan),
        )
        return True

    def request_handoff(self, target):
        target = getattr(self.cost_models, "resolve", lambda c: c)(target)
        if self.handoff.mode == "cold":
            return self._cold_request_handoff(target)
        return self._hot_request_handoff(target)

    def _hot_switching_gpu_seconds(self):
        """Old changed-role leases beyond target residency; no double counting."""
        return sum(
            (
                self.pool.lease_gpu_seconds.get(owner, 0.0) - before
                for owner, before in self._handoff_old_costs.items()
            )
        )

    def _cold_switching_gpu_seconds(self):
        return 0.0

    def switching_gpu_seconds(self, *args, **kwargs):
        if self.handoff.mode == "cold":
            return self._cold_switching_gpu_seconds(*args, **kwargs)
        return self._hot_switching_gpu_seconds(*args, **kwargs)

    def load_deployment(self, identity: int):
        deployment = self.deployments[identity]
        size = (
            self.handoff.actor_state_bytes
            if deployment.role == "actor"
            else self.handoff.weight_bytes
        )
        if deployment.role == "rollout":
            size *= len({node for node, _ in self.pool.leases[str(identity)][1]})
        start, end = self.link.reserve(self.now, byte_count=size)
        self.emit(
            "deployment_load",
            deployment=identity,
            role=deployment.role,
            snapshot_version=self.provider_version,
            transfer_start=start,
            ends_at=end,
            bytes=size,
        )
        self.schedule_at(end, "deployment_loaded", (identity, self.provider_version))

    def add_restoring_actor(self, identity):
        self.deployments[identity] = _Deployment(
            identity, "actor", self.layout, "preparing", self.provider_version
        )

    def prepared(self, identity: int, version: int):
        deployment = self.deployments[identity]
        deployment.version, deployment.status = (version, "prepared")
        if deployment.role == "actor":
            if self._restoring_actor == identity:
                self.actor_id = identity
                deployment.status = "active"
                self._restoring_actor = None
            else:
                self.pending_handoff["actor_ready"] = True
        self.emit(
            "deployment_prepared",
            deployment=identity,
            role=deployment.role,
            version=version,
        )

    def maybe_rollout_cutover(self):
        pending = self.pending_handoff
        if not pending or pending["new_rollout"] is None or pending["rollout_switched"]:
            return
        new = self.deployments[pending["new_rollout"]]
        if new.status != "prepared" or self.publication != "idle":
            return
        if new.version != self.provider_version:
            pending["rollout_fence"] = True
            new.status = "syncing"
            new.transfers += 1
            nodes = len({node for node, _ in self.pool.leases[str(new.identity)][1]})
            byte_count = self.handoff.weight_bytes * nodes
            start, end = self.link.reserve(
                self.now,
                byte_count=byte_count,
                setup_seconds=self.handoff.rollout_sync_setup_seconds,
            )
            self.emit(
                "rollout_catchup_start",
                deployment=new.identity,
                version=self.provider_version,
                transfer_start=start,
                ends_at=end,
                bytes=byte_count,
                nodes=nodes,
            )
            self.schedule_at(
                end, "rollout_synchronized", (new.identity, self.provider_version)
            )
            return
        old = self.deployments[self.rollout_id]
        old.status, new.status = ("draining", "active")
        self.rollout_id = new.identity
        pending["rollout_switched"] = True
        pending["rollout_fence"] = False
        self.emit(
            "rollout_cutover",
            source=old.identity,
            target=new.identity,
            version=new.version,
        )

    def rollout_synchronized(self, identity: int, version: int):
        deployment = self.deployments[identity]
        deployment.version, deployment.status = (version, "prepared")
        deployment.transfers -= 1
        self.emit("rollout_catchup_complete", deployment=identity, version=version)

    def maybe_actor_handoff(self):
        pending = self.pending_handoff
        if (
            not pending
            or not pending["actor_ready"]
            or pending["actor_syncing"]
            or (pending["new_actor"] == self.actor_id)
            or self.actor_batch
            or (self.publication != "idle")
        ):
            return
        pending["actor_syncing"] = True
        start, end = self.link.reserve(
            self.now,
            byte_count=self.handoff.actor_state_bytes,
            setup_seconds=self.handoff.actor_sync_setup_seconds,
        )
        snapshot = (self.provider_version, self.consumed)
        self.emit(
            "actor_handoff_start",
            source=self.actor_id,
            target=pending["new_actor"],
            version=snapshot[0],
            cursor=snapshot[1],
            transfer_start=start,
            ends_at=end,
        )
        self.schedule_at(end, "actor_synchronized", snapshot)

    def actor_synchronized(self, snapshot):
        if snapshot != (self.provider_version, self.consumed) or self.actor_batch:
            raise ValueError(
                "Training authority changed while its boundary snapshot was transferring"
            )
        pending = self.pending_handoff
        old = self.actor_id
        self.actor_id = pending["new_actor"]
        self.deployments[self.actor_id].status = "active"
        self.deployments[self.actor_id].version = self.provider_version
        pending["actor_ready"] = False
        self.release_deployment(old)
        self.emit(
            "actor_cutover",
            source=old,
            target=self.actor_id,
            version=self.provider_version,
            cursor=self.consumed,
        )

    def retire_rollouts(self):
        for identity, deployment in tuple(self.deployments.items()):
            if (
                deployment.role != "rollout"
                or deployment.status != "draining"
                or deployment.busy
                or deployment.transfers
            ):
                continue
            if not any(group.rows for group in deployment.replicas.values()):
                self.release_deployment(identity)

    def maybe_finish_handoff(self):
        pending = self.pending_handoff
        if not pending:
            return
        actor_done = (
            pending["new_actor"] is None or pending["new_actor"] == self.actor_id
        )
        active = {self.actor_id, self.rollout_id}
        if self._restoring_actor is not None:
            active.add(self._restoring_actor)
        if (
            actor_done
            and pending["rollout_switched"]
            and (set(self.deployments) <= active)
        ):
            self.emit("handoff_complete", layout=asdict(self.layout))
            self.pending_handoff = None
            if self.reconfigurator:
                self.reconfigurator.switched(self.now)

    def _physical_source(self):
        source = self.pool.leases
        actor = source.get(str(self.actor_id))
        if actor is None:
            probe = GPUAllocator(self.handoff.node_gpus, leases=dict(source))
            devices = probe.acquire(
                "source-actor",
                "actor",
                self.layout.actor_gpus,
                tp=self.layout.actor_tp,
                spread=isinstance(self.layout, ExecutionConfig),
            )
            if devices is None:
                raise ValueError(
                    "Stopped Actor recovery does not fit the declared pool"
                )
        else:
            devices = actor[1]
        return ExecutionLayout(
            self.layout.actor_gpus,
            self._ids(devices),
            self._ids(source[str(self.rollout_id)][1]),
        )

    def _ids(self, positions):
        all_positions = tuple(
            (
                (node, rank)
                for node, n in enumerate(self.handoff.node_gpus)
                for rank in range(n)
            )
        )
        identifiers = self.handoff.native_devices or tuple(range(len(all_positions)))
        mapping = dict(zip(all_positions, identifiers, strict=True))
        return tuple(mapping[p] for p in positions)

    def _positions(self, identifiers):
        all_positions = tuple(
            (
                (node, rank)
                for node, n in enumerate(self.handoff.node_gpus)
                for rank in range(n)
            )
        )
        devices = self.handoff.native_devices or tuple(range(len(all_positions)))
        mapping = dict(zip(devices, all_positions, strict=True))
        try:
            return tuple(mapping[d] for d in identifiers)
        except KeyError as error:
            raise ValueError(
                "Target GPU is outside the immutable reservation"
            ) from error

    def _target(self, layout):
        if self.handoff.native_source is not None:
            target = next(
                (
                    p.execution
                    for p in self.handoff.native_placements
                    if p.layout == layout
                ),
                None,
            )
            if target is None:
                raise ValueError(
                    "Native prediction has no physical mapping for this candidate"
                )
            return target
        actor_changed = layout.actor_key != self.layout.actor_key
        rollout_changed = layout.rollout_key != self.layout.rollout_key
        probe = GPUAllocator(self.handoff.node_gpus, leases=dict(self.pool.leases))
        for identity, changed in (
            (self.actor_id, actor_changed),
            (self.rollout_id, rollout_changed),
        ):
            if changed and str(identity) in probe.leases:
                probe.release(str(identity))
        actor = (
            probe.acquire(
                "target-actor",
                "actor",
                layout.actor_gpus,
                tp=layout.actor_tp,
                spread=isinstance(layout, ExecutionConfig),
            )
            if actor_changed or str(self.actor_id) not in probe.leases
            else probe.leases[str(self.actor_id)][1]
        )
        rollout = (
            probe.acquire(
                "target-rollout",
                "rollout",
                layout.rollout_gpus,
                tp=layout.rollout_tp,
                spread=isinstance(layout, ExecutionConfig),
            )
            if rollout_changed
            else probe.leases[str(self.rollout_id)][1]
        )
        if actor is None or rollout is None:
            raise ValueError("Cold target does not fit the declared TP-local pool")
        return ExecutionLayout(layout.actor_gpus, self._ids(actor), self._ids(rollout))

    def _seconds(self, step):
        values = dict(self.handoff.phase_seconds)
        if step in self._host_steps:
            return values.get(step, 0.0)
        if step not in values:
            raise ValueError(f"Cold transition timing unavailable for {step}")
        return values[step]

    def _allocate(self, step):
        role = "actor" if step == "restore_actor" else "rollout"
        plan = self.pending_handoff["plan"]
        devices = (
            plan.target.actor_devices[: plan.target.world_size]
            if role == "actor"
            else plan.target.rollout_devices
        )
        identity = self.next_deployment
        self.next_deployment += 1
        self.pool.acquire_exact(str(identity), role, self._positions(devices))
        self.deployments[identity] = _Deployment(
            identity,
            role,
            self.pending_handoff["target"],
            "preparing",
            self.provider_version,
        )
        self.pending_handoff["new_" + role] = identity

    def _complete_step(self, step):
        pending = self.pending_handoff
        if step == "checkpoint":
            self._checkpoint_version = self.training_version
        elif step == "release_actor":
            self.release_deployment(self.actor_id)
        elif step == "stop_rollout":
            self.release_deployment(self.rollout_id)
        elif step == "set_layout":
            self.execution_layout = pending["plan"].target
            self._logical = pending["target"]
            for d in self.deployments.values():
                d.layout = self.layout
        elif step == "restore_actor":
            old = self.actor_id
            self.actor_id = pending["new_actor"]
            self.deployments[self.actor_id].status = "active"
            self.emit(
                "actor_cutover",
                source=old,
                target=self.actor_id,
                version=self.provider_version,
                cursor=self.consumed,
            )
        elif step == "rebuild_rollout":
            old = self.rollout_id
            self.rollout_id = pending["new_rollout"]
            self.deployments[self.rollout_id].status = "active"
            self.emit(
                "rollout_cutover",
                source=old,
                target=self.rollout_id,
                version=self.provider_version,
            )
        self.emit("cold_phase_complete", phase=step)
        self._native_step = None
        self.schedule_at(self.now, "cold_continue")
