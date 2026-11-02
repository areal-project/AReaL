# SPDX-License-Identifier: Apache-2.0

"""Outer Online experiment environment with one version-owned control loop."""

from __future__ import annotations

import heapq
from collections import Counter
from collections.abc import Callable
from concurrent.futures import Executor
from dataclasses import asdict, replace

from areal.elastic.contracts import (
    ActorView,
    ActorWaitView,
    EpisodeSample,
    EpisodeView,
    Forecast,
    HandoffConfig,
    OnlineLayout,
    OnlineObservation,
    SimulationConfig,
    SimulationModels,
)
from areal.elastic.reconfigurator.reference.evaluator.cost_model.contracts import (
    ServiceSlice,
)
from areal.elastic.serving import request_slo_status
from areal.elastic.simulation_primitives import (
    GPUAllocator,
    ReplayInputs,
    SerializedLink,
    WorkInputs,
    advance_decode,
)

from .execution import SimulationExecution
from .reconfiguration import SimulationReconfiguration
from .state import _Deployment, _Episode, _Request

OnlineRunConfig = SimulationConfig
__all__ = ["OnlineSimulation", "OnlineRunConfig"]


class OnlineSimulation:
    """Run workload truth, expose observations and execute reconfiguration plans."""

    @classmethod
    def create(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    @property
    def layout(self):
        return self.reconfiguration.layout

    def request_handoff(self, target):
        return self.reconfiguration.request_handoff(target)

    def switching_gpu_seconds(self):
        return self.reconfiguration.switching_gpu_seconds()

    def poll_reconfigurator(self):
        if self.control is not None and not self.done:
            return self.control.poll()

    def handle_transition_event(self, kind, data):
        if kind == "decision_apply":
            # Explicitly scheduled experiment action; decisions use the shared controller.
            self.decision_pending = False
            self.request_handoff(data)
            return
        callbacks = {
            "reconfigurator": lambda: self.poll_reconfigurator(),
            "search_complete": lambda: self.control.complete(data),
            "verification_complete": lambda: self.control.verified(data),
            "decision_ready": lambda: self.control.ready(data),
        }
        if kind in callbacks:
            callbacks[kind]()
        else:
            self.reconfiguration.handle_event(kind, data)

    def prediction_guard_active(self):
        return False

    @classmethod
    def from_forecast(cls, view, forecast, models, config, handoff, slo, **kwargs):
        """Replay an observed snapshot against supplied work for independent checks."""
        sim = cls.restore(view, forecast, models, config, handoff, slo, **kwargs)
        if handoff.mode == "cold":
            sim.execution_layout = (
                handoff.native_source or sim.reconfiguration._physical_source()
            )
        return sim

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
        reconfigurator: object | None = None,
        prediction_models=None,
        measured_search_time=False,
        search_clock=None,
        event_sink: Callable[[dict], None] | None = None,
        start_time: float = 0.0,
        coalesce_decode: bool = False,
        candidate_executor: Executor | None = None,
        _work: WorkInputs | None = None,
    ):
        if config.batch_size % initial.actor_dp:
            raise ValueError("Actor DP must divide the episode batch")
        work = _work or WorkInputs.prepare(episodes, input_closes_at, start_time)
        self._truth = work.truth
        self.models, self.config, self.handoff, self.slo = (
            models,
            config,
            handoff,
            slo,
        )
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
        self.coalesce_decode = coalesce_decode
        self.candidate_executor = candidate_executor
        self.decode_steps_skipped = self.decode_segments = 0
        self.decode_plan_builds = self.decode_plan_hits = 0
        self.decode_batch_reuses = 0
        self.local_service_ticks = 0
        self.decode_run_reuses = 0
        self.control = None
        self._restoring_actor = None
        self.emit("allocation", leases=dict(self.pool.leases))
        self.execution = SimulationExecution(self)
        self.reconfiguration = SimulationReconfiguration(self)
        self.reconfiguration.initialize(initial)
        self.prediction_models = (
            self.models if prediction_models is None else prediction_models
        )
        self.measured_search_time = measured_search_time
        if reconfigurator is not None:
            clock = search_clock if measured_search_time else (lambda: 0.0)
            self.control = reconfigurator.bind(self, clock)
            self.schedule_at(
                self.now + reconfigurator.config.check_interval_seconds,
                "reconfigurator",
                priority=10,
            )

    def reconfiguration_allowed(self):
        return self._restoring_actor is None

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
        if when < self.now - 1e-09:
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
            self.execution.tick()
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
                if self.coalesce_decode:
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
                    and (self.pending_handoff is None)
                    and (self.publication == "idle")
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
                    self.execution.tick(
                        local_services
                        if service_only
                        and self.counters["request_complete"] == completed
                        else None
                    )
            if self.status == "running" and (not paused):
                self.status, self.reason = (
                    "complete",
                    "Input closed, serving drained and all full batches trained",
                )
        except (ValueError, ArithmeticError) as exc:
            self.status, self.reason = ("unsupported", str(exc))
        return self.result()

    def handle(self, kind: str, data):
        if kind == "arrival":
            self.execution.arrive(data)
        elif kind == "input_closed":
            self.input_closed = True
            self.emit("input_closed")
        elif kind == "service_complete":
            self.execution.finish_service(*data)
        elif kind == "tool_complete":
            self.execution.finish_tool(data)
        elif kind == "actor_phase_complete":
            self.execution.finish_actor_phase()
        elif kind == "publication_complete":
            self.execution.finish_publication(data)
        else:
            self.handle_transition_event(kind, data)

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
            tuple(self.deployments[self.rollout_id].busy.values())
            if self.rollout_id in self.deployments
            else (),
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
            result["control"] = dict(
                clock="measured" if self.measured_search_time else "zero",
                **self.control.stats,
            )
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
            seconds = models.service(view.layout, group.kv_tokens, slices)
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
