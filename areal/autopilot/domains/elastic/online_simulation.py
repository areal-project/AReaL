# SPDX-License-Identifier: Apache-2.0

"""Persistent online serving/training environment with a private workload oracle.

Only observe() crosses into Basic. Counterfactuals are reconstructed from that
observation and sampled continuations; the real future event heap is never copied.
"""

from __future__ import annotations

import heapq
import math
from collections import Counter
from collections.abc import Callable
from concurrent.futures import Executor
from dataclasses import asdict, dataclass, field, replace
from functools import cached_property
from typing import Protocol

from areal.elastic.policy.basic.forecast import (
    ActorView,
    ActorWaitView,
    EpisodeSample,
    EpisodeView,
    Forecast,
    OnlineLayout,
    OnlineObservation,
    RequestView,
    ServiceView,
    ToolView,
    Turn,
)
from areal.elastic.policy.basic.online import (
    ActorMemorySafety,
    ActorSupply,
    CandidatePrediction,
    OnlinePolicy,
    rolling_slo,
)
from areal.elastic.policy.basic.search import (
    SearchBound,
    candidate_order,
    select_prediction,
)
from areal.elastic.policy.basic.serving import ServingSLO
from areal.elastic.policy.reference.core import fingerprint
from areal.elastic.policy.reference.cost_model._rollout.scheduling import policy_for
from areal.elastic.policy.reference.cost_model._rollout.types import (
    ServiceSlice,
)
from areal.elastic.policy.reference.cost_model.rollout import RolloutConfig
from areal.elastic.policy.reference.scenario import Request

from .online_inputs import ForecastInputs, WorkInputs
from .online_memory import EvaluationMemoryModels
from .online_state import ReplicaState, RequestState
from .online_transitions import (
    ActorMemoryRejected,
    GPUAllocator,
    HandoffConfig,
    SerializedLink,
)


class OnlineModels(Protocol):
    source_ids: tuple[str, ...]

    def rollout_config(self, layout: OnlineLayout) -> RolloutConfig: ...
    def service(
        self, layout: OnlineLayout, kv: int, slices: tuple[ServiceSlice, ...]
    ) -> float: ...
    def actor(
        self, layout: OnlineLayout, batch: tuple[tuple[Turn, ...], ...]
    ) -> tuple[tuple[str, float], ...]: ...
    def publication(
        self, layout: OnlineLayout, recipients: tuple[OnlineLayout, ...]
    ) -> float: ...


@dataclass(frozen=True)
class OnlineRunConfig:
    batch_size: int
    staleness: int
    client_concurrency: int
    tool_capacities: tuple[tuple[str, int], ...]
    max_new_tokens: int
    max_context_tokens: int
    max_events: int
    max_simulated_seconds: float
    actor_lifecycle_enabled: bool = False
    actor_lifecycle_history_seconds: float = 120.0
    actor_lifecycle_margin_seconds: float = 5.0
    actor_stopped_rollout_reconfig: bool = False

    def __post_init__(self):
        for name in (
            "batch_size",
            "client_concurrency",
            "max_new_tokens",
            "max_context_tokens",
            "max_events",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"Invalid {name}")
        if self.staleness < 0 or self.max_simulated_seconds <= 0:
            raise ValueError("Explicit staleness and simulation budget are required")
        if not self.tool_capacities or any(n < 1 for _, n in self.tool_capacities):
            raise ValueError("Explicit positive tool pool capacities are required")
        if (
            not math.isfinite(self.actor_lifecycle_history_seconds)
            or self.actor_lifecycle_history_seconds <= 0
            or not math.isfinite(self.actor_lifecycle_margin_seconds)
            or self.actor_lifecycle_margin_seconds < 0
        ):
            raise ValueError(
                "Lifecycle needs a positive history and nonnegative margin"
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
        self.reservation, self.order = reservation, order

    @property
    def group_id(self):
        return self.view.episode_id

    @cached_property
    def request(self):
        # The scheduler sees a public generation budget, never the actual stop length.
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
    layout: OnlineLayout
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


class OnlineSimulation:
    def __init__(
        self,
        episodes: tuple[EpisodeSample, ...],
        initial: OnlineLayout,
        models: OnlineModels,
        config: OnlineRunConfig,
        handoff: HandoffConfig,
        slo: ServingSLO,
        *,
        input_closes_at: float,
        policy: OnlinePolicy | None = None,
        event_sink: Callable[[dict], None] | None = None,
        start_time: float = 0.0,
        coalesce_decode: bool = False,
        candidate_executor: Executor | None = None,
        _work: WorkInputs | None = None,
        measured_search_time: bool = False,
        search_clock: Callable[[], float] | None = None,
        _lifecycle_view=None,
    ):
        if config.batch_size % initial.actor_dp:
            raise ValueError("Actor DP must divide the episode batch")
        work = _work or WorkInputs.prepare(episodes, input_closes_at, start_time)
        self._truth = work.truth
        self.models, self.config, self.handoff, self.slo = models, config, handoff, slo
        self.now = start_time
        if initial.actor_gpus + initial.rollout_gpus > 16:
            raise ValueError(
                "Initial Actor and Rollout share a sixteen-GPU steady budget"
            )
        self.pool = GPUAllocator(handoff.node_gpus, now=start_time)
        self.link = SerializedLink(handoff.effective_bandwidth_GBps, start_time)
        self.deployments = {
            0: _Deployment(0, "actor", initial, "active", 0),
            1: _Deployment(1, "rollout", initial, "active", 0),
        }
        if (
            self.pool.acquire("0", "actor", initial.actor_gpus, tp=initial.actor_tp)
            is None
            or self.pool.acquire(
                "1", "rollout", initial.rollout_gpus, tp=initial.rollout_tp
            )
            is None
        ):
            raise ValueError(
                "Initial deployment does not fit the explicit resource pool"
            )
        self.actor_id, self.rollout_id, self.next_deployment = 0, 1, 2
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
        self._forecast_target: OnlineLayout | None = None
        self.actor_stages: tuple[tuple[str, float], ...] = ()
        self.actor_phase = 0
        self.actor_phase_started = start_time
        self.consumed = self.training_version = self.provider_version = 0
        self.input_closed = False
        self.publication = "idle"
        self.pending_handoff: dict | None = None
        self._handoff_old_costs: dict[str, float] = {}
        self.decision_pending = False
        self.policy = policy
        self.sink = event_sink
        self.events: list[dict] = []
        self.decisions: list[dict] = []
        self.training_records: list[dict] = []
        self.counters = Counter()
        self.heap: list[tuple[float, int, int, str, object]] = list(work.heap)
        self.sequence = work.sequence
        self.done = False
        self.processed_events = 0
        self.status, self.reason = "running", ""
        self._forecast_slo_guard: _ForecastSLOGuard | None = None
        self._candidate_guard: _CandidateGuard | None = None
        self.coalesce_decode = coalesce_decode
        self.candidate_executor = candidate_executor
        self.decode_steps_skipped = self.decode_segments = 0
        self.decode_plan_builds = self.decode_plan_hits = 0
        self.decode_batch_reuses = 0
        self.local_service_ticks = 0
        self.decode_run_reuses = 0
        self.control = None
        if measured_search_time and policy is not None:
            from .online_control import OnlineControl

            self.control = OnlineControl(self, search_clock)
        self.lifecycle = None
        if config.actor_lifecycle_enabled:
            from .online_lifecycle import ActorLifecycle

            self.lifecycle = ActorLifecycle(self, _lifecycle_view)
        if policy:
            self.schedule_at(
                start_time + policy.config.check_interval_seconds, "policy", priority=10
            )
        self.emit("allocation", leases=dict(self.pool.leases))

    @property
    def layout(self):
        if self.rollout_id not in self.deployments:
            return self._last_layout
        actor = (
            self.deployments[self.actor_id].layout
            if self.actor_id in self.deployments
            else self._last_layout
        )
        rollout = self.deployments[self.rollout_id].layout
        self._last_layout = OnlineLayout(
            actor.actor_dp, actor.actor_tp, rollout.rollout_replicas, rollout.rollout_tp
        )
        return self._last_layout

    def reconfiguration_allowed(self):
        return (
            self.lifecycle is None
            or self.lifecycle.state == "running"
            or (
                self.lifecycle.state == "stopped"
                and self.config.actor_stopped_rollout_reconfig
            )
        )

    def emit(self, kind: str, **data):
        self.counters[kind] += 1
        if kind == "service_batch" and self.sink is None:
            return
        event = dict(time=self.now, kind=kind, **data)
        if self.sink:
            self.sink(event)
        if kind != "service_batch":
            self.events.append(event)

    def schedule_at(self, when: float, kind: str, data=None, *, priority: int = 0):
        if when < self.now - 1e-9:
            raise ValueError("An event cannot be scheduled in the past")
        self.sequence += 1
        heapq.heappush(
            self.heap, (max(self.now, when), priority, self.sequence, kind, data)
        )

    def run(
        self, until: float | None = None, *, stop_after_handoff: bool = False
    ) -> dict:
        paused = False
        try:
            self.tick()
            while not self.done:
                if (
                    stop_after_handoff
                    and not self.pending_handoff
                    and not self.decision_pending
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
                if self.coalesce_decode:
                    from .online_decode import advance_decode

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
                service_only = (
                    self.coalesce_decode
                    and self.sink is None
                    and self.pending_handoff is None
                    and self.publication == "idle"
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
                    self.check_forecast_slo()
                    self.check_candidate_bound()
            if self.status == "running" and not paused:
                self.status, self.reason = (
                    "complete",
                    "Input closed, serving drained and all full batches trained",
                )
        except (ValueError, ArithmeticError) as exc:
            self.status, self.reason = "unsupported", str(exc)
        return self.result()

    def check_forecast_slo(self):
        guard = self._forecast_slo_guard
        if guard is None or self.done or self.now < guard.check_after:
            return
        guard.check_after = self.now + 1.0  # CPU pruning cadence; no new time events.
        failed = len(guard.failed) + sum(
            self.request_has_failed(r.view) for r in self._live_requests.values()
        )
        upper = 1 - failed / guard.total if guard.total else 1.0
        if upper + 1e-12 < self.slo.minimum_request_fraction:
            self.status = "forecast_slo_infeasible"
            self.reason = (
                f"Sampled branch has irreversible SLO failures: {failed}/{guard.total}; "
                f"joint_fraction_upper_bound={upper:.12g}; stopped before full cost prediction"
            )
            self.done = True

    def request_has_failed(self, view: RequestView):
        ttft = (
            view.first_token_at if view.first_token_at is not None else self.now
        ) - view.ready_at
        gap = max(
            view.maximum_tbt_seconds,
            self.now - view.last_token_at
            if view.completed_at is None and view.last_token_at is not None
            else 0.0,
        )
        return (
            ttft > self.slo.ttft_seconds + 1e-12 or gap > self.slo.tbt_seconds + 1e-12
        )

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
            self.status, self.reason, self.done = "candidate_pruned", reason, True

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
        elif kind == "actor_lifecycle_check":
            self.lifecycle.request_check()
        elif kind == "actor_lifecycle_saved":
            self.lifecycle.saved()
        elif kind == "actor_lifecycle_wake":
            if self.lifecycle.wake_at == data:
                self.lifecycle.wake("estimated_batch_ready")
        elif kind == "policy":
            self.poll_policy()
        elif kind == "search_complete":
            self.control.complete(data)
        elif kind == "verification_complete":
            self.control.verified(data)
        elif kind == "decision_ready":
            self.control.ready(data)
        elif kind == "decision_apply":
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
            # No admission credit, tool readiness, complete training batch or
            # deployment boundary changed. Preserve the usual timestamp-wide
            # event handling before restarting only the completed replicas.
            self.local_service_ticks += 1
            self.service(local_services)
            return
        if self.publication == "pending" and not any(
            d.busy for d in self.deployments.values()
        ):
            self.start_publication()
        self.maybe_actor_handoff()
        self.maybe_rollout_cutover()
        if self.publication == "idle":
            self.admit()
            self.start_tools()
            self.service()
            if self.lifecycle is not None:
                self.lifecycle.tick()
            self.start_actor()
        else:
            self.start_tools()  # Tool execution remains independent of GPU publication.
        self.retire_rollouts()
        self.maybe_finish_handoff()
        if (
            self.input_closed
            and len(self.episodes) == len(self._truth)
            and self._completed_episodes == len(self.episodes)
            and not self.actor_batch
            and self.publication == "idle"
            and self.pending_handoff is None
            and not self.decision_pending
            and (
                self.lifecycle is None or self.lifecycle.state in {"running", "stopped"}
            )
            and len(self.arrival_order) - self.consumed < self.config.batch_size
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
        # Count only intervals with tasks waiting for staleness admission credit.
        if (
            self._admission_cursor < len(self.arrival_order)
            and self.admission_credits() <= 0
        ):
            self.admission_credit_blocked_seconds += when - self.now
        self.pool.advance(when)
        self.now = when

    def admit(self):
        active = self._active_episodes
        credits = self.admission_credits()
        while self._admission_cursor < len(self.arrival_order):
            identity = self.arrival_order[self._admission_cursor]
            episode = self.episodes[identity]
            if active >= self.config.client_concurrency or credits <= 0:
                break
            self._admission_cursor += 1
            episode.admitted = True
            active += 1
            credits -= 1
            self.emit("episode_admitted", episode=identity)
        self._active_episodes = active

    def service(self, local_services: set[tuple[int, int]] | None = None):
        target = self.deployments[self.rollout_id]
        cfg = self.models.rollout_config(target.layout)
        unassigned = self._unassigned.values() if local_services is None else ()
        for request in sorted(unassigned, key=lambda r: r.order):
            view = request.view
            if not self.episodes[view.episode_id].admitted:
                continue
            need = view.input_tokens + request.reservation - 1
            candidates = [
                i
                for i, group in target.replicas.items()
                if len(group.rows) < cfg.max_batch_sequences
                and group.reserved_tokens + need <= cfg.kv_capacity_tokens
            ]
            if candidates:
                replica = min(
                    candidates,
                    key=lambda i: (
                        target.replicas[i].kv_tokens,
                        len(target.replicas[i].rows),
                        i,
                    ),
                )
                view.deployment, view.replica = target.identity, replica
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
            cfg = self.models.rollout_config(deployment.layout)
            for replica in range(cfg.replicas):
                if (
                    local_services is not None
                    and (
                        deployment.identity,
                        replica,
                    )
                    not in local_services
                ):
                    continue
                if replica in deployment.busy:
                    continue
                group = deployment.replicas[replica]
                if not group.rows:
                    continue
                plan = group.decode_plan
                continuation = (
                    self.coalesce_decode
                    and self.sink is None
                    and self._forecast_slo_guard is None
                    and plan is not None
                    and plan.continuation
                    and plan.revision == group.revision
                    and plan.version == deployment.version
                    and plan.limit > 0
                    and plan.first_seconds > 0
                    and group.kv_tokens + len(plan.batch.slices)
                    <= cfg.kv_capacity_tokens
                )
                if continuation:
                    duration = plan.first_seconds
                    rows = tuple(
                        (key, context + 1, query, phase, emits)
                        for key, context, query, phase, emits in plan.batch.slices
                    )
                    self.decode_batch_reuses += 1
                else:
                    group.flush_decode()
                    slices, _ = policy_for(cfg.backend).batch(
                        cfg, list(group.rows.values())
                    )
                    duration = self.models.service(
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
                batch = ServiceView(
                    replica,
                    self.now,
                    rows,
                    deployment.version,
                )
                if continuation:
                    plan.batch = batch
                    plan.continuation = False
                deployment.busy[replica] = batch
                self.schedule_at(
                    self.now + duration,
                    "service_complete",
                    (deployment.identity, replica),
                )

    def finish_service(self, identity: int, replica: int):
        deployment = self.deployments[identity]
        batch = deployment.busy.pop(replica)
        group = deployment.replicas[replica]
        plan = group.decode_plan
        defer = (
            self.coalesce_decode
            and self.sink is None
            and self._forecast_slo_guard is None
            and plan is not None
            and plan.batch is batch
            and plan.revision == group.revision
            and plan.version == deployment.version
            and plan.limit > 1
        )
        if defer:
            # The retained ordinary iteration cannot finish any selected
            # request. Keep it in the same summary across unrelated events.
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
            self.coalesce_decode
            and self.sink is None
            and self._forecast_slo_guard is None
            and plan is not None
            and plan.batch is batch
            and plan.revision == group.revision
            and plan.version == deployment.version
            and plan.limit > 0
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
        # Match original episode-arrival priority, rather than tool-ready FIFO.
        waiting = sorted(
            self._tool_episodes.values(),
            key=lambda e: self._episode_order[e.episode_id],
        )
        for pool, capacity in self.config.tool_capacities:
            active = sum(
                e.tool is not None
                and e.tool.pool == pool
                and e.tool.started_at is not None
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

    def start_actor(self):
        if self.lifecycle is not None and self.lifecycle.state != "running":
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
        wait = self.actor_wait
        if wait and wait.deployment == self.actor_id and wait.batch == batch:
            self.reject_unrepairable_forecast_wait()
            return
        try:
            stages = self.models.actor(
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
        self.actor_batch, self.actor_stages = batch, stages
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

    def reject_unrepairable_forecast_wait(self):
        target = self._forecast_target
        if target is None:
            return  # A real policy or externally scheduled action may repair it.
        changes_actor = (target.actor_dp, target.actor_tp) != (
            self.layout.actor_dp,
            self.layout.actor_tp,
        )
        pending = self.pending_handoff
        repair_planned = self.decision_pending or (
            pending is not None
            and pending["new_actor"] is not None
            and pending["new_actor"] != self.actor_id
        )
        if not changes_actor or not repair_planned:
            raise ActorMemoryRejected(
                self.actor_wait.reason
                + "; no planned Actor migration can repair this batch"
            )

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
        seconds = self.models.publication(
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

    def finish_publication(self, data):
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
        if self.lifecycle is not None:
            self.lifecycle.request_check()
        if self.control is not None:
            self.control.publication_complete()

    def request_handoff(self, target: OnlineLayout):
        if not self.reconfiguration_allowed():
            return False
        if target.actor_gpus + target.rollout_gpus > 16:
            raise ValueError(
                "Target Actor and Rollout share a sixteen-GPU steady budget"
            )
        if self.pending_handoff is not None or target == self.layout:
            return False
        if self.config.batch_size % target.actor_dp:
            raise ValueError("Target Actor DP must divide the configured episode batch")
        current = self.layout
        if self.policy is not None and self.policy.config.independent_role_history:
            allowed = self.policy.allowed_roles(self.now)
            if (
                not allowed["actor"]
                and (target.actor_dp, target.actor_tp)
                != (current.actor_dp, current.actor_tp)
            ) or (
                not allowed["rollout"]
                and (target.rollout_replicas, target.rollout_tp)
                != (current.rollout_replicas, current.rollout_tp)
            ):
                return False
        stopped = self.lifecycle is not None and self.lifecycle.state == "stopped"
        if stopped and (target.actor_dp, target.actor_tp) != (
            current.actor_dp,
            current.actor_tp,
        ):
            return False
        roles = []
        if (current.actor_dp, current.actor_tp) != (target.actor_dp, target.actor_tp):
            roles.append(("actor", target.actor_gpus))
        if (current.rollout_replicas, current.rollout_tp) != (
            target.rollout_replicas,
            target.rollout_tp,
        ):
            roles.append(("rollout", target.rollout_gpus))
        acquired = {}
        previous_peak = self.pool.peak_gpus
        for role, count in roles:
            identity = self.next_deployment
            self.next_deployment += 1
            tp = target.actor_tp if role == "actor" else target.rollout_tp
            if self.pool.acquire(str(identity), role, count, tp=tp) is None:
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
        recovery_placement = None
        if stopped:
            # Probe recovery while both Rollout generations still occupy GPUs.
            # This is a feasibility check, not a charged or persistent lease.
            actual_peak = self.pool.peak_gpus
            recovery_placement = self.pool.acquire(
                "actor-recovery-probe", "actor", current.actor_gpus, tp=current.actor_tp
            )
            if recovery_placement is None:
                for previous in acquired.values():
                    self.pool.release(str(previous))
                self.pool.peak_gpus = previous_peak
                self.emit(
                    "handoff_deferred",
                    target=asdict(target),
                    reason="Rollout overlap would prevent Actor recovery",
                )
                return False
            self.pool.release("actor-recovery-probe")
            self.pool.peak_gpus = actual_peak
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
            actor_stopped=stopped,
            actor_recovery_placement=recovery_placement,
        )
        return True

    def switching_gpu_seconds(self):
        """Old changed-role leases beyond target residency; no double counting."""
        return sum(
            self.pool.lease_gpu_seconds.get(owner, 0.0) - before
            for owner, before in self._handoff_old_costs.items()
        )

    def load_deployment(self, identity: int):
        deployment = self.deployments[identity]
        size = (
            self.handoff.actor_state_bytes
            if deployment.role == "actor"
            else self.handoff.weight_bytes
        )
        if deployment.role == "rollout":
            # One host-side model image per destination node; replica startup
            # within a node retains the existing shared-image assumption.
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
        deployment.version, deployment.status = version, "prepared"
        if deployment.role == "actor":
            if self.lifecycle is not None and self.lifecycle.restore_id == identity:
                self.lifecycle.prepared(identity)
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
            # Finish the current update, but fence further updates while final
            # synchronization catches up with its last published version.
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
        old.status, new.status = "draining", "active"
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
        deployment.version, deployment.status = version, "prepared"
        deployment.transfers -= 1
        self.emit("rollout_catchup_complete", deployment=identity, version=version)

    def maybe_actor_handoff(self):
        pending = self.pending_handoff
        if (
            not pending
            or not pending["actor_ready"]
            or pending["actor_syncing"]
            or pending["new_actor"] == self.actor_id
            or self.actor_batch
            or self.publication != "idle"
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
            if self.policy is not None:
                self.policy.role_switched(deployment.role, self.now)

    def maybe_finish_handoff(self):
        pending = self.pending_handoff
        if not pending:
            return
        actor_done = (
            pending["new_actor"] is None or pending["new_actor"] == self.actor_id
        )
        active = {self.actor_id, self.rollout_id}
        if self.lifecycle is not None and self.lifecycle.restore_id is not None:
            active.add(self.lifecycle.restore_id)
        if (
            actor_done
            and pending["rollout_switched"]
            and set(self.deployments) <= active
        ):
            self.emit("handoff_complete", layout=asdict(self.layout))
            self.pending_handoff = None
            if self.policy:
                self.policy.switched(self.now)

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
                self.lifecycle.observe() if self.lifecycle is not None else None,
            ),
            self.consumed,
            self.training_version,
            self.provider_version,
            self.pending_handoff is not None
            or self.publication != "idle"
            or self.decision_pending
            or not self.reconfiguration_allowed(),
            self.input_closed,
            self.actor_id,
            self.rollout_id,
            tuple(
                (int(owner), role, devices)
                for owner, (role, devices) in sorted(self.pool.leases.items())
            ),
        )

    def poll_policy(self):
        if not self.policy or self.done:
            return
        if self.lifecycle is not None:
            self.lifecycle.tick()
        if self.control is not None:
            return self.control.poll()
        view = self.observe()
        model = OnlineForecastModel(
            self.models,
            self.config,
            self.handoff,
            self.slo,
            self.policy.config.decision_delay_seconds,
            executor=self.candidate_executor,
            actor_memory_reserve_fraction=self.policy.config.actor_memory_reserve_fraction,
            actor_memory_scope=self.policy.config.actor_memory_scope,
        )
        decision = self.policy.decide(view, model)
        record = asdict(decision)
        record["actor_lifecycle_state"] = (
            view.actor.lifecycle.state if view.actor.lifecycle is not None else None
        )
        record["role_history"] = self.policy.role_history_status(view.now)
        record["observation_id"] = fingerprint(view)
        record["evaluation"] = model.execution_stats
        self.decisions.append(record)
        self.emit("decision", **record)
        if decision.target != view.layout:
            self.decision_pending = True
            self.schedule_at(
                self.now + self.policy.config.decision_delay_seconds,
                "decision_apply",
                decision.target,
                priority=9,
            )
        self.schedule_at(
            self.now + self.policy.config.check_interval_seconds, "policy", priority=10
        )

    def flush_decode(self):
        for deployment in self.deployments.values():
            for group in deployment.replicas.values():
                group.flush_decode()

    def result(self):
        self.flush_decode()
        rows = tuple(r.view for r in self.requests.values())
        complete = [r for r in rows if r.completed_at is not None]
        passed = [
            r
            for r in complete
            if r.first_token_at - r.ready_at <= self.slo.ttft_seconds + 1e-12
            and r.maximum_tbt_seconds <= self.slo.tbt_seconds + 1e-12
        ]
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
            jointly_passed_requests=len(passed),
            joint_request_fraction=len(passed) / len(rows) if rows else None,
            actor_gpu_seconds=self.pool.gpu_seconds["actor"],
            rollout_gpu_seconds=self.pool.gpu_seconds["rollout"],
            total_gpu_seconds=sum(self.pool.gpu_seconds.values()),
            peak_gpus=self.pool.peak_gpus,
            remaining_leases=dict(self.pool.leases),
            counters=dict(self.counters),
            source_ids=self.models.source_ids,
            assumptions=(
                "GPU-token timing; HTTP/network response tails not measured",
                "Decode-first batches; declared generation-budget KV reservation; no prefix reuse",
                "Publication preserves request progress, unlike native abort/recompute",
                "Immutable published snapshots assumed; complete Actor state handoff at a single-writer boundary",
                "Startup/state/publication transfers serialize on one assumed effective shared link",
                self.handoff.provenance,
                *getattr(self.models, "scenario_notes", ()),
            ),
        )
        if self.control is not None:
            result["control"] = dict(clock="measured", **self.control.stats)
        if self.lifecycle is not None:
            result["actor_lifecycle"] = dict(
                state=self.lifecycle.state,
                suspends=self.counters["actor_suspended"],
                resumes=self.counters["actor_resumed"],
                snapshot=self.lifecycle.snapshot,
            )
        return result

    @classmethod
    def from_forecast(
        cls,
        view: OnlineObservation,
        forecast: Forecast,
        models: OnlineModels,
        config: OnlineRunConfig,
        handoff: HandoffConfig,
        slo: ServingSLO,
        *,
        coalesce_decode: bool = False,
        _prepared: ForecastInputs | None = None,
    ):
        if view.busy:
            raise ValueError(
                "Forecast branching waits for a stable deployment/publication boundary"
            )
        prepared = _prepared or ForecastInputs.prepare(
            view, forecast, config.max_new_tokens, config.max_context_tokens
        )
        if prepared.view is not view or prepared.forecast is not forecast:
            raise ValueError(
                "Prepared inputs belong to a different observation/forecast"
            )
        simulation = cls(
            forecast.episodes,
            view.layout,
            models,
            config,
            handoff,
            slo,
            input_closes_at=forecast.input_closes_at,
            start_time=view.now,
            coalesce_decode=coalesce_decode,
            _work=prepared.work,
            _lifecycle_view=view.actor.lifecycle,
        )
        stopped = (
            view.actor.lifecycle is not None and view.actor.lifecycle.state == "stopped"
        )
        if stopped:
            if simulation.lifecycle is None:
                raise ValueError("Stopped Actor observation requires lifecycle support")
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
                or len({node for node, _ in devices}) > (2 if count > 8 else 1)
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
                    or not 0 <= rank < handoff.node_gpus[node]
                ):
                    raise ValueError(
                        "Invalid or overlapping visible TP-local GPU leases"
                    )
                occupied.add((node, rank))
            leases[owner] = role, devices
        if set(leases) != ({"1"} if stopped else {"0", "1"}):
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
        # Rebuild indexes solely from the visible restored state. Completed
        # history remains available to observe()/training, outside token hot paths.
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
        # Recompute the remaining known GPU iteration, not its hidden stop lengths.
        for service in view.services:
            group = simulation.deployments[1].replicas[service.replica]
            slices = tuple(ServiceSlice(*s) for s in service.slices)
            seconds = models.service(view.layout, group.kv_tokens, slices)
            simulation.deployments[1].busy[service.replica] = service
            simulation.schedule_at(
                max(view.now, service.started_at + seconds),
                "service_complete",
                (1, service.replica),
            )
        # Tools' true end times are not in the view. Their sampled total duration
        # is conditioned on already elapsed time by HistoricalForecaster.
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
            simulation.actor_stages = models.actor(
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
        return simulation


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
        actor_memory_reserve_fraction: float = 0.0,
        actor_memory_scope: str = "all_completed",
    ):
        self.models, self.config, self.handoff, self.slo = models, config, handoff, slo
        self.delay = decision_delay_seconds
        self.prune_irreversible_slo = prune_irreversible_slo
        self.coalesce_decode = coalesce_decode
        self.executor = executor
        if not 0 <= actor_memory_reserve_fraction < 1:
            raise ValueError("Actor memory reserve must be in [0, 1)")
        self.actor_memory_reserve_fraction = actor_memory_reserve_fraction
        if actor_memory_scope not in ("all_completed", "evaluation"):
            raise ValueError("Actor memory scope must be all_completed or evaluation")
        self.actor_memory_scope = actor_memory_scope
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
            # Preserve the real arrival-order batch boundaries, including holes
            # caused by unfinished episodes. Never fill them with future turns.
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
        key = layout.actor_dp, layout.actor_tp
        if key in self._memory_checks:
            return self._memory_checks[key]
        peak = capacity = allowed = None
        worst, count, admissible = (), 0, True
        reason = "All completed historical batches retain the configured memory reserve"
        estimate = getattr(self.models, "actor_memory", None)
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
                        or not math.isfinite(limit)
                        or limit <= 0
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

    def actor_supply(self, view, forecast, layout, *, models=None):
        """Estimate ongoing demand from offered arrivals and causal turn templates.

        Incomplete final batches wrap to the start of the same historical sequence
        and carry fractional batch weight. This estimates a repeated workload's
        capacity, not extra training in the serving window or an exact queue bound.
        """
        models = models or self.models
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
                    stages = models.actor(layout, batch)
                    if not stages:
                        raise ValueError("Actor model supplied no capacity stages")
                    seconds = sum(seconds for _, seconds in stages)
                    if not math.isfinite(seconds) or seconds <= 0:
                        raise ValueError(
                            "Actor capacity requires a positive finite cycle"
                        )
                    weighted_seconds += seconds * count / batch_size
                    weight += count / batch_size
                    batches += 1
                    padded += batch_size - count
                actor_seconds = weighted_seconds / weight
                publication_seconds = models.publication(layout, (layout,))
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

    def search(self, view, forecasts, layouts, *, preferred=(), trigger="pressure"):
        self.prepare_context(view, forecasts)
        if not self._historical:
            return self.evaluate_many(view, forecasts, layouts)
        from .online_parallel import evaluate_chunk

        actor_scores, rollout_scores = {}, {}
        turns = tuple(t for e in self._known_batch for t in e)[-16:]

        def score(layout):
            actor_key = layout.actor_dp, layout.actor_tp
            rollout_key = layout.rollout_replicas, layout.rollout_tp
            try:
                if actor_key not in actor_scores:
                    actor_scores[actor_key] = (
                        sum(s for _, s in self.models.actor(layout, self._known_batch))
                        if self._known_batch
                        else 0.0
                    )
                if rollout_key not in rollout_scores:
                    cfg = self.models.rollout_config(layout)
                    seconds = 0.0
                    for t in turns:
                        q = min(t.input_tokens, cfg.max_batch_tokens)
                        prefill = self.models.service(
                            layout,
                            0,
                            (
                                ServiceSlice(
                                    "rank", 0, q, "prefill", q == t.input_tokens
                                ),
                            ),
                        )
                        decode = self.models.service(
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
                return math.inf  # Priority only; the real evaluation records support.

        order = candidate_order(layouts, view.layout, preferred, score)
        ordered = [None] * len(layouts)
        best = None
        work, cpus, preparations, pids = [], 0.0, 0, set()
        parallel = (
            self.executor is not None
            and getattr(self.models, "worker_spec", None) is not None
        )
        payload = (
            getattr(self.models, "worker_spec", None),
            self.source_ids,
            self.config,
            self.handoff,
            self.slo,
            self.delay,
            self.prune_irreversible_slo,
            self.coalesce_decode,
            view,
            forecasts,
            self._context_id,
            self.actor_memory_reserve_fraction,
            self.actor_memory_scope,
        )
        # Score current first for the below-target fallback, not as a pressure
        # cost incumbent: any feasible replacement takes precedence over it.
        # Subsequent small waves propagate only already completed incumbents.
        waves = [order[:1]] + [order[i : i + 8] for i in range(1, len(order), 8)]
        for wave in waves:
            bound = (
                SearchBound(
                    self._context_id,
                    self.slo.minimum_request_fraction,
                    best,
                    fraction_pruning=False,
                )
                if best and not (trigger == "pressure" and best.layout == view.layout)
                else None
            )
            if parallel and len(wave) > 1:
                chunks = tuple(
                    self.executor.map(
                        evaluate_chunk,
                        (
                            (payload + (bound,), wave[i : i + 2])
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
                    prediction = self.evaluate(view, forecasts, layout, bound=bound)
                    ordered[index] = prediction
                    work.append(prediction)
            # Original order keeps the current configuration as the work-ID anchor.
            best = select_prediction(
                tuple(p for p in ordered if p is not None),
                view.layout,
                self.slo.minimum_request_fraction,
                trigger=trigger,
            )
        self.execution_stats = dict(
            mode="bounded_process_pool" if parallel else "bounded_serial",
            considered=len(ordered),
            completed=sum(p.supported for p in ordered),
            pruned=sum(p.pruned for p in ordered),
            unsupported=sum(not p.supported and not p.pruned for p in ordered),
            evaluation_order=tuple(p.layout.key for p in work),
            workers_used=len(pids),
            worker_task_cpu_seconds=cpus,
            input_preparations=preparations,
        )
        return tuple(ordered)

    def evaluate_many(self, view, forecasts, layouts):
        spec = getattr(self.models, "worker_spec", None)
        if self.executor is None or spec is None:
            return tuple(self.evaluate(view, forecasts, layout) for layout in layouts)
        from .online_parallel import evaluate_chunk

        self.prepare_context(view, forecasts)
        payload = (
            spec,
            self.models.source_ids,
            self.config,
            self.handoff,
            self.slo,
            self.delay,
            self.prune_irreversible_slo,
            self.coalesce_decode,
            view,
            forecasts,
            self._context_id,
            self.actor_memory_reserve_fraction,
            self.actor_memory_scope,
        )
        indexed = tuple(enumerate(layouts))
        jobs = ((payload, indexed[i : i + 8]) for i in range(0, len(indexed), 8))
        # Small jobs balance unequal worker speeds; explicit indices retain
        # candidate order, including repeated layout entries.
        results = tuple(self.executor.map(evaluate_chunk, jobs))
        self.execution_stats = dict(
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

    def evaluate(
        self,
        view: OnlineObservation,
        forecasts: tuple[Forecast, ...],
        layout: OnlineLayout,
        *,
        bound: SearchBound | None = None,
    ) -> CandidatePrediction:
        self.prepare_context(view, forecasts)
        if (
            self.actor_memory_reserve_fraction
            and self.actor_memory_scope == "evaluation"
        ):
            models = EvaluationMemoryModels(
                self.models, layout, self.actor_memory_reserve_fraction, view.now
            )
            prediction = self._evaluate(
                view, forecasts, layout, bound=bound, models=models
            )
            memory = models.report
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
        if memory is not None and not memory.admissible:
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

    def _evaluate(self, view, forecasts, layout, *, bound=None, models=None):
        models = models or self.models
        self.prepare_context(view, forecasts)
        work_ids = self._work_ids
        if (
            view.actor.lifecycle is not None
            and view.actor.lifecycle.state == "stopped"
            and (layout.actor_dp, layout.actor_tp)
            != (view.layout.actor_dp, view.layout.actor_tp)
        ):
            return CandidatePrediction(
                layout,
                False,
                None,
                None,
                work_ids,
                "Stopped Actor retains its recovery DP/TP; only Rollout may change",
                self.source_ids,
            )
        if bound is not None and bound.context_id != self._context_id:
            bound = None
        if self.config.actor_lifecycle_enabled:
            # Constant target-residency lower bounds do not hold during sleep.
            bound = None
        costs, fractions, window_costs = [], [], []
        running_costs = []
        switching_costs, horizons, window_stats, window_results = [], [], [], []
        historical = self._historical
        coverage = "legacy_sampled_execution"
        known_batch = self._known_batch
        supply = (
            tuple(self.actor_supply(view, f, layout, models=models) for f in forecasts)
            if historical
            else ()
        )
        if historical:
            if known_batch:
                try:
                    if not models.actor(layout, known_batch):
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
                        actor_supply=supply,
                    )
                coverage = "known_complete_batch_capacity_checked"
        past = self._past
        if self.prune_irreversible_slo and not historical:
            failed = sum(
                r.completed_at is None
                and (
                    (r.first_token_at if r.first_token_at is not None else view.now)
                    - r.ready_at
                    > self.slo.ttft_seconds + 1e-12
                    or max(
                        r.maximum_tbt_seconds,
                        view.now - r.last_token_at
                        if r.last_token_at is not None
                        else 0.0,
                    )
                    > self.slo.tbt_seconds + 1e-12
                )
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
                simulation = OnlineSimulation.from_forecast(
                    view,
                    forecast,
                    models,
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
                if self.prune_irreversible_slo and not historical:
                    simulation._forecast_slo_guard = _ForecastSLOGuard(
                        past, self._remaining[index]
                    )
                if layout != view.layout:
                    simulation.decision_pending = True
                    simulation.schedule_at(
                        view.now + self.delay, "decision_apply", layout, priority=9
                    )
                if historical:
                    result = simulation.run(until=forecast.input_closes_at)
                    window_costs.append(result["total_gpu_seconds"])
                    if self.config.actor_lifecycle_enabled:
                        running_cost = (
                            result["total_gpu_seconds"]
                            - simulation.switching_gpu_seconds()
                        )
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
                    self.models.source_ids,
                    actor_supply=supply,
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
                and r.first_token_at - r.ready_at <= self.slo.ttft_seconds + 1e-12
                and r.maximum_tbt_seconds <= self.slo.tbt_seconds + 1e-12
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
            "Fixed serving window; score = switching GPU seconds + actual target residency integral; lifecycle enabled"
            if historical and self.config.actor_lifecycle_enabled
            else "Fixed serving window; score = switching GPU seconds + window seconds * target GPU count; target tail not scored"
            if historical
            else "Sampled arrivals close at horizon; all sampled serving and full training batches drain afterwards",
            self.models.source_ids,
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
            admission_credits_start=(
                (self.config.staleness + view.provider_version + 1)
                * self.config.batch_size
                - sum(e.admitted for e in view.episodes)
            )
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
