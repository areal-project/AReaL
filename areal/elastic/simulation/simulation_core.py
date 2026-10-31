# SPDX-License-Identifier: Apache-2.0

"""Shared event execution and state restoration for elastic training.

The private workload oracle stays in the environment. Policies receive immutable
observations; candidate branches receive only forecast workloads."""

from __future__ import annotations

import heapq
import math
from collections import Counter
from collections.abc import Callable
from concurrent.futures import Executor
from dataclasses import asdict, dataclass, field, replace
from functools import cached_property
from typing import Protocol

from areal.elastic.policy.basic.observations import (
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
from areal.elastic.policy.reference.cost_model._rollout.scheduling import policy_for
from areal.elastic.policy.reference.cost_model._rollout.types import (
    ServiceSlice,
)
from areal.elastic.policy.reference.cost_model.rollout import RolloutConfig
from areal.elastic.policy.reference.scenario import Request

from .online_inputs import ForecastInputs, WorkInputs
from .online_state import ReplicaState, RequestState
from .online_transitions import (
    ActorMemoryRejected,
    GPUAllocator,
    HandoffConfig,
    SerializedLink,
)


class SimulationModels(Protocol):
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
class SimulationConfig:
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


class SimulationCore:
    def __init__(
        self,
        episodes: tuple[EpisodeSample, ...],
        initial: OnlineLayout,
        models: SimulationModels,
        config: SimulationConfig,
        handoff: HandoffConfig,
        slo=None,
        *,
        input_closes_at: float,
        policy: object | None = None,
        event_sink: Callable[[dict], None] | None = None,
        start_time: float = 0.0,
        coalesce_decode: bool = False,
        candidate_executor: Executor | None = None,
        _work: WorkInputs | None = None,
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
        self.training_limit: int | None = None
        self.coalesce_decode = coalesce_decode
        self.candidate_executor = candidate_executor
        self.decode_steps_skipped = self.decode_segments = 0
        self.decode_plan_builds = self.decode_plan_hits = 0
        self.decode_batch_reuses = 0
        self.local_service_ticks = 0
        self.decode_run_reuses = 0
        self.control = None
        self.lifecycle = None
        if config.actor_lifecycle_enabled:
            from .online_lifecycle import ActorLifecycle

            self.lifecycle = ActorLifecycle(self, _lifecycle_view)
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
                    and self.publication == "idle"
                ):
                    paused = True
                    break
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
                    self.check_prediction_bounds()
            if self.status == "running" and not paused:
                self.status, self.reason = (
                    "complete",
                    "Input closed, serving drained and all full batches trained",
                )
        except (ValueError, ArithmeticError) as exc:
            self.status, self.reason = "unsupported", str(exc)
        return self.result()

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
        self.before_work()
        if self.publication == "idle":
            self.admit()
            self.start_tools()
            self.service()
            if self.lifecycle is not None:
                self.lifecycle.tick()
            self.start_actor()
        else:
            self.start_tools()  # Tool execution remains independent of GPU publication.
        self.after_work()
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
                    and not self.prediction_guard_active()
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
            and not self.prediction_guard_active()
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
            and not self.prediction_guard_active()
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

    def flush_decode(self):
        for deployment in self.deployments.values():
            for group in deployment.replicas.values():
                group.flush_decode()

    def result(self):
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
    def restore(
        cls,
        view: OnlineObservation,
        forecast: Forecast,
        models: SimulationModels,
        config: SimulationConfig,
        handoff: HandoffConfig,
        slo=None,
        *,
        coalesce_decode: bool = False,
        _prepared: ForecastInputs | None = None,
        **simulation_options,
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
            **simulation_options,
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

    def before_work(self):
        """Advance mode-specific transitions before admitting more work."""

    def after_work(self):
        """Observe progress and advance mode-specific control after execution."""

    def check_prediction_bounds(self):
        """Optional early termination of a candidate branch."""

    def reject_unrepairable_forecast_wait(self):
        """An outer controller may replace a capacity-limited Actor."""

    def deployment_released(self, identity, deployment):
        """Optional per-role handoff accounting."""

    def handle_transition_event(self, kind, data):
        raise ValueError(f"Unknown simulation event {kind}")

    def prediction_guard_active(self):
        return False

    def request_finished(self, view):
        """Optional request-level candidate scoring."""
