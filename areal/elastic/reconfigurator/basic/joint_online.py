# SPDX-License-Identifier: Apache-2.0

"""Online extension of the original aggregate Joint simulation.

Retains the single aggregate GPU channel, balanced token-work replica speedup,
sequential transition intervals and explicit residency of the original _JointRun.
Adds episode admission, current-work restoration and request timing within a
bounded window. A service cohort completes at a barrier; no continuous batching
or token-level interference is reconstructed. Component fits are never refitted.
"""

from __future__ import annotations

import heapq
import math
from collections import Counter
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Protocol

from areal.elastic.contracts import RequestView, ToolView
from areal.elastic.layout import ExecutionLayout, ReconfigurationPlanner
from areal.elastic.simulation_primitives import GPUAllocator, RequestState

from .core import fingerprint
from .cost_model._rollout.types import ServiceSlice
from .cost_model.aggregate_rollout import (
    RolloutConfig,
    RolloutCostModel,
    RolloutProfile,
    RolloutStage,
    RolloutWorkload,
)
from .cost_model.joint_transition import TransitionPhase
from .simulator import _JointRun


class ActorCatalog(Protocol):
    def actor_config(self, config): ...
    def predict(self, config, stats): ...


@dataclass
class _Episode:
    episode_id: str
    admitted: bool = False
    completed_at: float | None = None
    past: list = field(default_factory=list)
    current: str | None = None
    tool: ToolView | None = None


@dataclass
class _Request:
    view: RequestState


@dataclass
class _Segment:
    key: str
    initial: int
    outputs: int
    prefill_end: float
    first: float
    step: float
    version: int


@dataclass(frozen=True)
class _Profile:
    phases: tuple[TransitionPhase, ...]


class JointWindow(_JointRun):
    """A private forecast run; never owns or calls an experimental environment.

    The inherited schedule()/transition() implement the original Joint interval
    machinery. Online pump/dispatch restore episode-level facts and preserve the
    same aggregate service assumption when those facts are advanced.
    """

    version = "basic-joint-online-v1"

    def __init__(
        self,
        view,
        forecast,
        models,
        config,
        handoff,
        slo,
        *,
        _prepared=None,
        coalesce_decode=False,
    ):
        if view.busy:
            raise ValueError("Joint prediction requires a stable deployment boundary")
        self.view, self.forecast = view, forecast
        self.models, self.config, self.handoff, self.slo = models, config, handoff, slo
        self.samples = (
            dict(_prepared.work.truth)
            if _prepared is not None
            else {e.episode_id: e for e in forecast.episodes}
        )
        if len(self.samples) != len(forecast.episodes):
            raise ValueError("Duplicate forecast episode")
        self.now = view.now
        self.layout = view.layout
        self.training_version, self.provider = (
            view.training_version,
            view.provider_version,
        )
        self.consumed = view.consumed_episodes
        self.actor_id, self.rollout_id = 0, 1
        self.actor_running = view.actor.running
        self.episodes = {}
        self.order = [e.episode_id for e in view.episodes]
        for e in view.episodes:
            if e.episode_id not in self.samples:
                raise ValueError("Forecast must retain every observed episode")
            self.episodes[e.episode_id] = _Episode(
                e.episode_id,
                e.admitted,
                e.completed_at,
                list(e.completed_turns),
                e.current.request_id if e.current is not None else None,
                e.tool,
            )
        self.requests = {
            r.request_id: _Request(RequestState.copy(r)) for r in view.requests
        }
        for e in view.episodes:
            if e.current is not None and e.current.request_id not in self.requests:
                self.requests[e.current.request_id] = _Request(
                    RequestState.copy(e.current)
                )
        self.heap, self.sequence, self.events = [], 0, []
        self.counters = Counter()
        self.actor_cost = self.rollout_cost = 0.0
        self.allocation = (
            view.layout.actor_gpus if self.actor_running else 0,
            view.layout.rollout_gpus,
        )
        self.peak_gpus = sum(self.allocation)
        self.profiles = {}
        self.active_transition = None
        self.scenario = SimpleNamespace(
            max_actor_gpus=sum(handoff.node_gpus),
            max_rollout_gpus=sum(handoff.node_gpus),
            max_total_gpus=sum(handoff.node_gpus),
        )
        self.actor_batch = view.actor.batch
        self.actor_wait = view.actor.waiting
        self.actor_stages = ()
        self.actor_phase = view.actor.phase
        self.actor_started = view.actor.phase_started_at
        self.pending_publish = False
        self.training_records = []
        self._segments = []
        self.gpu_busy = 0
        self._inflight = set()
        self.tool_counts = Counter()
        self.pending_handoff = None
        self.decision_pending = False
        self._drain = set()
        self._input_closed = view.input_closed
        self.done = False
        self._handoff_cost = 0.0
        self._initial_actor = bool(view.actor.batch)
        self._forecast_slo_guard = self._candidate_guard = None
        self._checkpoint_version = handoff.native_checkpoint_version
        self.native_plan = None
        # A first request is ready at external arrival, including time spent
        # waiting for admission. Reuse any observed request identity/progress.
        for episode in self.episodes.values():
            if episode.completed_at is not None or episode.tool is not None:
                continue
            live = [
                r.view.request_id
                for r in self.requests.values()
                if r.view.episode_id == episode.episode_id
                and r.view.completed_at is None
            ]
            if live:
                episode.current = live[0]
            elif episode.current is None:
                self._new_request(
                    episode, self.samples[episode.episode_id].available_at
                )
        self._validate_source()
        for e in forecast.episodes:
            if e.episode_id not in self.episodes and not view.input_closed:
                self.schedule(
                    max(0.0, e.available_at - self.now), "arrival", e.episode_id
                )
        self.schedule(max(0.0, forecast.input_closes_at - self.now), "input_closed")
        for e in self.episodes.values():
            if e.tool is not None and e.tool.started_at is not None:
                self.tool_counts[e.tool.pool] += 1
                turn = self.samples[e.episode_id].turns[len(e.past)]
                self.schedule(
                    max(0.0, e.tool.started_at + turn.tool_seconds - self.now),
                    "tool_end",
                    e.episode_id,
                )
        for service in view.services:
            slices = tuple(ServiceSlice(*s) for s in service.slices)
            keys = {s.request_id for s in slices}
            kv = sum(
                self.requests[k].view.prefilled
                + max(0, self.requests[k].view.generated - 1)
                for k in keys
            )
            seconds = models.service(view.layout, kv, slices)
            self.gpu_busy += 1
            self._inflight.update(keys)
            self.schedule(
                max(0.0, service.started_at + seconds - self.now),
                "known_iteration",
                service,
            )
        if self.actor_batch:
            self.actor_stages = models.actor(view.layout, self._batch(self.actor_batch))
            if self.actor_phase >= len(self.actor_stages):
                raise ValueError("Observed Actor phase is outside the model")
            self.training_records.append(
                dict(episodes=self.actor_batch, restored=True, committed_at=None)
            )
            remaining = max(
                0.0,
                self.actor_started + self.actor_stages[self.actor_phase][1] - self.now,
            )
            self.schedule(remaining, "actor_phase_end")

    @property
    def training_version(self):
        return self._training_version

    @training_version.setter
    def training_version(self, value):
        self._training_version = value

    @property
    def provider_version(self):
        return self.provider

    @property
    def input_closed(self):
        return self._input_closed

    @input_closed.setter
    def input_closed(self, value):
        self._input_closed = value

    def _validate_source(self):
        if sum(self.allocation) > sum(self.handoff.node_gpus):
            raise ValueError("Source exceeds the declared GPU pool")
        self.execution_layout, self._source_pool = self._placement(
            self.layout, source=True
        )
        for request in self.requests.values():
            r = request.view
            turns = self.samples[r.episode_id].turns
            if r.turn >= len(turns) or r.generated > turns[r.turn].output_tokens:
                raise ValueError("Forecast contradicts observed request progress")

    def _placement(self, layout, *, source=False):
        h = self.handoff
        positions = tuple(
            (node, rank) for node, n in enumerate(h.node_gpus) for rank in range(n)
        )
        identifiers = h.native_devices or tuple(range(len(positions)))
        to_id = dict(zip(positions, identifiers, strict=True))
        to_position = dict(zip(identifiers, positions, strict=True))
        pool = GPUAllocator(h.node_gpus)
        if source and h.native_source is not None:
            physical = h.native_source
            if self.actor_running:
                pool.acquire_exact(
                    "actor",
                    "actor",
                    tuple(
                        to_position[d]
                        for d in physical.actor_devices[: physical.world_size]
                    ),
                )
            pool.acquire_exact(
                "rollout",
                "rollout",
                tuple(to_position[d] for d in physical.rollout_devices),
            )
            return physical, pool
        if source and self.view.leases:
            for owner, role, devices in self.view.leases:
                if owner not in (
                    self.view.actor_deployment,
                    self.view.rollout_deployment,
                ):
                    raise ValueError("Unexpected deployment in the source observation")
                pool.acquire_exact(role, role, devices)
            actor = tuple(to_id[p] for p in pool.leases.get("actor", (None, ()))[1])
            if not actor:
                probe = GPUAllocator(h.node_gpus, leases=dict(pool.leases))
                placed = probe.acquire(
                    "actor", "actor", layout.actor_gpus, tp=layout.actor_tp
                )
                if placed is None:
                    raise ValueError("Stopped Actor recovery does not fit the pool")
                actor = tuple(to_id[p] for p in placed)
            rollout = tuple(to_id[p] for p in pool.leases["rollout"][1])
            return ExecutionLayout(layout.actor_gpus, actor, rollout), pool
        if source:
            a = pool.acquire("actor", "actor", layout.actor_gpus, tp=layout.actor_tp)
            r = pool.acquire(
                "rollout", "rollout", layout.rollout_gpus, tp=layout.rollout_tp
            )
            if a is None or r is None:
                raise ValueError("Source does not fit the declared TP-local pool")
            physical = ExecutionLayout(
                layout.actor_gpus,
                tuple(to_id[p] for p in a),
                tuple(to_id[p] for p in r),
            )
            if not self.actor_running:
                pool.release("actor")
            return physical, pool
        if h.native_source is not None:
            target = next(
                (p.execution for p in h.native_placements if p.layout == layout), None
            )
            if target is None:
                raise ValueError("Native candidate lacks a physical mapping")
            return target, None
        pool = GPUAllocator(h.node_gpus, leases=dict(self._source_pool.leases))
        actor_changed = (layout.actor_dp, layout.actor_tp) != (
            self.layout.actor_dp,
            self.layout.actor_tp,
        )
        rollout_changed = (layout.rollout_replicas, layout.rollout_tp) != (
            self.layout.rollout_replicas,
            self.layout.rollout_tp,
        )
        if h.mode == "cold":
            for role, changed in (
                ("actor", actor_changed),
                ("rollout", rollout_changed),
            ):
                if changed and role in pool.leases:
                    pool.release(role)

        def acquire(role, changed, count, tp):
            if not changed and role in pool.leases:
                return pool.leases[role][1]
            return pool.acquire("target-" + role, role, count, tp=tp)

        a = acquire("actor", actor_changed, layout.actor_gpus, layout.actor_tp)
        r = acquire("rollout", rollout_changed, layout.rollout_gpus, layout.rollout_tp)
        if a is None or r is None:
            raise ValueError("No spare TP-local GPUs for the Joint target")
        return ExecutionLayout(
            layout.actor_gpus, tuple(to_id[p] for p in a), tuple(to_id[p] for p in r)
        ), pool

    def emit(self, kind, batch=-1, **data):
        self.events.append(dict(time=self.now, kind=kind, batch=batch, **data))
        self.counters[kind] += 1

    def schedule_at(self, when, kind, data=None, priority=10):
        if kind != "decision_apply":
            raise ValueError(
                "A Joint candidate accepts only its supplied configuration"
            )
        self.decision_pending = True
        self.schedule(max(0.0, when - self.now), kind, data)

    def _batch(self, ids):
        return tuple(tuple(self.episodes[key].past) for key in ids)

    def _new_request(self, episode, when):
        index = len(episode.past)
        turn = self.samples[episode.episode_id].turns[index]
        if (
            turn.input_tokens + turn.output_tokens > self.config.max_context_tokens
            or turn.output_tokens > self.config.max_new_tokens
        ):
            raise ValueError("Forecast exceeds the configured generation budget")
        key = f"{episode.episode_id}/request-{index}"
        if key in self.requests:
            raise ValueError("Duplicate predicted request")
        self.requests[key] = _Request(
            RequestState.copy(
                RequestView(key, episode.episode_id, index, turn.input_tokens, when)
            )
        )
        episode.current = key

    def _finish_request(self, key):
        r = self.requests[key].view
        if r.completed_at is not None:
            return
        r.completed_at = self.now
        e = self.episodes[r.episode_id]
        turn = self.samples[r.episode_id].turns[r.turn]
        self.emit("request_complete", request=key)
        if turn.tool_seconds is None:
            e.past.append(turn)
            e.current = None
            e.completed_at = self.now
        else:
            e.current = None
            e.tool = ToolView(turn.tool_pool, self.now, None)

    def _materialize(self, when):
        for segment in self._segments:
            r = self.requests[segment.key].view
            count = (
                0
                if when < segment.first - 1e-12
                else min(
                    segment.outputs,
                    1 + math.floor((when - segment.first + 1e-12) / segment.step),
                )
            )
            delta = segment.initial + count - r.generated
            if when + 1e-12 >= segment.prefill_end:
                r.prefilled = r.input_tokens
            if delta > 0:
                first_new = (
                    segment.first + (r.generated - segment.initial) * segment.step
                )
                end = segment.first + (count - 1) * segment.step
                gap = max(
                    segment.step if count > 1 else 0.0,
                    first_new - r.last_token_at if r.last_token_at is not None else 0.0,
                )
                if r.first_token_at is None:
                    r.first_token_at = segment.first
                r.add_tokens(delta, segment.version, end, gap)

    def _rates(self, rows):
        prefill_tokens = decode_tokens = 0
        prefill_seconds = decode_seconds = 0.0
        cfg = self.models.rollout_config(self.layout)
        for row in rows:
            r = row.view
            turn = self.samples[r.episode_id].turns[r.turn]
            remaining = turn.output_tokens - r.generated
            context = r.prefilled
            pending = r.input_tokens - context
            while pending:
                query = min(cfg.max_batch_tokens, pending)
                prefill_seconds += self.models.service(
                    self.layout,
                    context,
                    (ServiceSlice("joint", context, query, "prefill", False),),
                )
                pending -= query
                context += query
                prefill_tokens += query
            if remaining:
                context = r.input_tokens + max(0, r.generated - 1)
                seconds = self.models.service(
                    self.layout,
                    context,
                    (ServiceSlice("joint", context, 1, "decode", True),),
                )
                decode_seconds += remaining * seconds
                decode_tokens += remaining
        # Convert component predictions to explicit aggregate-rate priors. This is
        # algebraic normalization at the forecast context, not coefficient fitting.
        profile_config = RolloutConfig(
            "joint-components",
            "declared-hardware",
            "joint-online-v1",
            "fixed-sampling",
            self.layout.rollout_replicas,
            self.layout.rollout_tp,
            cfg.max_batch_sequences,
            cfg.max_batch_tokens,
            cfg.kv_capacity_tokens,
            1,
        )
        tokens = prefill_tokens + decode_tokens
        max_request = max(
            r.view.input_tokens
            + self.samples[r.view.episode_id].turns[r.view.turn].output_tokens
            for r in rows
        )
        profile = RolloutProfile(
            profile_config,
            prefill_tokens / prefill_seconds if prefill_tokens else 1.0,
            decode_tokens / decode_seconds if decode_tokens else 1.0,
            1,
            len(rows),
            1,
            max(1, tokens),
            max_request,
            fingerprint(
                self.models.source_ids,
                self.layout,
                tuple(r.view.snapshot() for r in rows),
            ),
        )
        stage = RolloutStage(prefill_tokens, decode_tokens, 0.0, len(rows))
        workload = RolloutWorkload((stage,), len(rows), max_request)
        estimate = RolloutCostModel((profile,)).predict(profile_config, workload)
        if not estimate.supported:
            raise ValueError(estimate.reason)
        width = min(self.layout.rollout_replicas, len(rows))
        return (
            prefill_seconds / width,
            decode_seconds / width,
            estimate.stages[0][0],
        )

    def _start_service(self):
        if self.gpu_busy or self.active_transition or self.pending_publish:
            return
        rows = [
            r
            for r in self.requests.values()
            if r.view.completed_at is None
            and r.view.request_id not in self._inflight
            and self.episodes[r.view.episode_id].admitted
        ]
        if self.pending_handoff is not None:
            rows = [r for r in rows if r.view.request_id in self._drain]
        if not rows:
            return
        cfg = self.models.rollout_config(self.layout)
        reservations = [0] * self.layout.rollout_replicas
        selected = []
        for r in sorted(rows, key=lambda r: (r.view.ready_at, r.view.request_id)):
            reserve = (
                r.view.input_tokens
                + min(
                    self.config.max_new_tokens,
                    self.config.max_context_tokens - r.view.input_tokens,
                )
                - 1
            )
            index = min(range(len(reservations)), key=reservations.__getitem__)
            if reserve > cfg.kv_capacity_tokens:
                raise ValueError("Joint request exceeds the per-replica KV budget")
            if reservations[index] + reserve > cfg.kv_capacity_tokens:
                continue
            if len(selected) >= cfg.max_batch_sequences * self.layout.rollout_replicas:
                break
            reservations[index] += reserve
            selected.append(r)
        if not selected:
            return
        prefill, decode, duration = self._rates(selected)
        maximum = max(
            self.samples[r.view.episode_id].turns[r.view.turn].output_tokens
            - r.view.generated
            for r in selected
        )
        tick = decode / maximum
        if tick <= 0 or duration <= 0:
            raise ValueError("Joint service requires positive component durations")
        self._segments = [
            _Segment(
                r.view.request_id,
                r.view.generated,
                self.samples[r.view.episode_id].turns[r.view.turn].output_tokens
                - r.view.generated,
                self.now + prefill,
                self.now + prefill + tick,
                tick,
                self.provider,
            )
            for r in selected
        ]
        self.gpu_busy = 1
        self._inflight = {r.view.request_id for r in selected}
        self.schedule(duration, "cohort_end")
        self.emit("rollout_service_start", requests=tuple(self._inflight))

    def _start_actor(self):
        if (
            self.actor_batch
            or self.active_transition
            or self.pending_handoff is not None
            or self.decision_pending
        ):
            return
        ids = tuple(self.order[self.consumed : self.consumed + self.config.batch_size])
        if len(ids) != self.config.batch_size or any(
            self.episodes[key].completed_at is None for key in ids
        ):
            return
        stages = self.models.actor(self.layout, self._batch(ids))
        if not stages:
            raise ValueError("Actor supplied no update stages")
        if not self.actor_running:
            seconds = (
                self.handoff.actor_startup_seconds
                + self.handoff.actor_warmup_seconds
                + self.handoff.actor_sync_setup_seconds
                + self.handoff.actor_state_bytes
                / (self.handoff.effective_bandwidth_GBps * 1e9)
            )
            self.profiles["wake"] = _Profile(
                (
                    TransitionPhase(
                        seconds, self.layout.actor_gpus, self.layout.rollout_gpus
                    ),
                )
            )
            self.transition("wake")
            return
        self.actor_batch, self.actor_stages, self.actor_phase, self.actor_started = (
            ids,
            stages,
            0,
            self.now,
        )
        self.actor_wait = None
        self.training_records.append(dict(episodes=ids, committed_at=None))
        self.emit("actor_start", episodes=ids, deployment=self.actor_id)
        self.schedule(stages[0][1], "actor_phase_end")

    def _cold_profile(self, target):
        physical, _ = self._placement(target)
        actor_changed = (target.actor_dp, target.actor_tp) != (
            self.layout.actor_dp,
            self.layout.actor_tp,
        )
        rollout_changed = (target.rollout_replicas, target.rollout_tp) != (
            self.layout.rollout_replicas,
            self.layout.rollout_tp,
        )
        plan = ReconfigurationPlanner().plan(
            self.execution_layout,
            physical,
            actor_running=self.actor_running,
            force_actor=actor_changed,
            force_rollout=rollout_changed,
        )
        self.native_plan = plan
        steps = list(plan.steps)
        if self._checkpoint_version != self.training_version:
            steps.insert(0, "checkpoint")
        actor, rollout = self.allocation
        times = dict(self.handoff.phase_seconds)
        phases = []
        for step in steps:
            if step == "drain_model_calls":
                continue
            if step == "rebuild_rollout":
                rollout = target.rollout_gpus
            if step == "restore_actor":
                actor = target.actor_gpus
            if step == "publish_weights":
                seconds = self.models.publication(target, (target,))
            elif step in {
                "close_admission",
                "set_layout",
                "rebind_runtime",
                "pause_rollout",
            }:
                seconds = times.get(step, 0.0)
            else:
                if step not in times:
                    raise ValueError(f"Cold transition timing unavailable for {step}")
                seconds = times[step]
            phases.append(TransitionPhase(seconds, actor, rollout))
            if step == "release_actor":
                actor = 0
            if step == "stop_rollout":
                rollout = 0
        return (
            _Profile(tuple(phases)),
            physical,
            "restore_actor" in plan.steps,
            "rebuild_rollout" in plan.steps,
        )

    def _begin_handoff(self):
        target = self.pending_handoff
        if target is None or self.actor_batch or self.pending_publish or self.gpu_busy:
            return
        if any(self.requests[k].view.completed_at is None for k in self._drain):
            return
        if (
            target.actor_gpus + target.rollout_gpus > 16
            or self.config.batch_size % target.actor_dp
        ):
            raise ValueError("Joint target exceeds the steady budget or logical batch")
        if self.handoff.mode == "cold":
            profile, physical, actor_changed, rollout_changed = self._cold_profile(
                target
            )
        else:
            physical, _ = self._placement(target)
            actor_changed = (target.actor_dp, target.actor_tp) != (
                self.layout.actor_dp,
                self.layout.actor_tp,
            )
            rollout_changed = (target.rollout_replicas, target.rollout_tp) != (
                self.layout.rollout_replicas,
                self.layout.rollout_tp,
            )
            h = self.handoff
            actor_time = (
                h.actor_startup_seconds
                + h.actor_warmup_seconds
                + h.actor_sync_setup_seconds
                + h.actor_state_bytes / (h.effective_bandwidth_GBps * 1e9)
                if actor_changed
                else 0.0
            )
            rollout_time = (
                h.rollout_startup_seconds
                + h.rollout_warmup_seconds
                + h.rollout_sync_setup_seconds
                + h.weight_bytes / (h.effective_bandwidth_GBps * 1e9)
                if rollout_changed
                else 0.0
            )
            old_actor = (
                self.layout.actor_gpus if actor_changed and self.actor_running else 0
            )
            old_rollout = self.layout.rollout_gpus if rollout_changed else 0
            seconds = actor_time + rollout_time
            profile = _Profile(
                (
                    TransitionPhase(
                        seconds,
                        target.actor_gpus + old_actor,
                        target.rollout_gpus + old_rollout,
                    ),
                )
            )
            self._handoff_cost += seconds * (old_actor + old_rollout)
        if not profile.phases:
            profile = _Profile((TransitionPhase(0.0, *self.allocation),))
        if any(
            not self.within_limits(p.actor_gpus, p.rollout_gpus) for p in profile.phases
        ):
            raise ValueError("Transition exceeds the declared GPU pool")
        self._target_physical = physical
        self._changed_roles = actor_changed, rollout_changed
        self.profiles["reconfigure"] = profile
        self.transition("reconfigure")
        self.peak_gpus = max(
            self.peak_gpus, *(p.actor_gpus + p.rollout_gpus for p in profile.phases)
        )

    def pump(self):
        if self.active_transition:
            return
        self._begin_handoff()
        if self.active_transition:
            return
        if self.pending_publish:
            if not self.gpu_busy:
                seconds = self.models.publication(self.layout, (self.layout,))
                self.profiles["publish"] = _Profile(
                    (TransitionPhase(seconds, *self.allocation),)
                )
                self.transition("publish")
            return
        if self.pending_handoff is None:
            active = sum(
                e.admitted and e.completed_at is None for e in self.episodes.values()
            )
            admitted = self.consumed + sum(
                e.admitted
                for e in (self.episodes[k] for k in self.order[self.consumed :])
            )
            credits = (
                self.config.batch_size * (self.config.staleness + self.provider + 1)
                - admitted
            )
            for key in self.order:
                e = self.episodes[key]
                if (
                    not e.admitted
                    and e.completed_at is None
                    and credits > 0
                    and active < self.config.client_concurrency
                ):
                    e.admitted = True
                    if e.current is None:
                        self._new_request(e, self.samples[key].available_at)
                    active += 1
                    credits -= 1
        capacities = dict(self.config.tool_capacities)
        for e in self.episodes.values():
            if (
                e.tool is not None
                and e.tool.started_at is None
                and self.tool_counts[e.tool.pool] < capacities[e.tool.pool]
            ):
                e.tool = ToolView(e.tool.pool, e.tool.ready_at, self.now)
                self.tool_counts[e.tool.pool] += 1
                turn = self.samples[e.episode_id].turns[len(e.past)]
                self.schedule(turn.tool_seconds, "tool_end", e.episode_id)
        self._start_actor()
        self._start_service()
        self.peak_gpus = max(self.peak_gpus, sum(self.allocation))
        if (
            self._input_closed
            and self.pending_handoff is None
            and not self.decision_pending
            and not self.actor_batch
            and not self.active_transition
            and not self.gpu_busy
            and all(e.completed_at is not None for e in self.episodes.values())
        ):
            self.done = True

    def _dispatch(self, kind, data):
        if kind == "arrival":
            if data in self.episodes:
                raise ValueError("Duplicate arrival")
            self.order.append(data)
            self.episodes[data] = _Episode(data)
            self._new_request(self.episodes[data], self.samples[data].available_at)
        elif kind == "input_closed":
            self._input_closed = True
        elif kind == "decision_apply":
            self.decision_pending = False
            if data != self.layout:
                self.pending_handoff = data
                if (data.rollout_replicas, data.rollout_tp) != (
                    self.layout.rollout_replicas,
                    self.layout.rollout_tp,
                ):
                    self._drain = {
                        k
                        for k, r in self.requests.items()
                        if r.view.completed_at is None
                        and self.episodes[r.view.episode_id].admitted
                    }
        elif kind == "known_iteration":
            self.gpu_busy -= 1
            for key, _, query, phase, emits in data.slices:
                r = self.requests[key].view
                if phase == "prefill":
                    r.prefilled += query
                if emits:
                    r.add_tokens(
                        1,
                        data.version,
                        self.now,
                        self.now - r.last_token_at
                        if r.last_token_at is not None
                        else 0.0,
                    )
                self._inflight.discard(key)
                if (
                    r.generated
                    == self.samples[r.episode_id].turns[r.turn].output_tokens
                ):
                    self._finish_request(key)
        elif kind == "cohort_end":
            self._materialize(self.now)
            for segment in self._segments:
                self._finish_request(segment.key)
            self._segments = []
            self._inflight.clear()
            self.gpu_busy = 0
        elif kind == "tool_end":
            e = self.episodes[data]
            self.tool_counts[e.tool.pool] -= 1
            e.past.append(self.samples[data].turns[len(e.past)])
            e.tool = None
            self._new_request(e, self.now)
        elif kind == "actor_phase_end":
            self.actor_phase += 1
            if self.actor_phase < len(self.actor_stages):
                self.actor_started = self.now
                self.schedule(self.actor_stages[self.actor_phase][1], "actor_phase_end")
            else:
                self.training_version += 1
                self.consumed += self.config.batch_size
                self.training_records[-1]["committed_at"] = self.now
                self.pending_publish = True
        elif kind == "allocation":
            self.allocation = data
        elif kind == "transition_end":
            self.active_transition = None
            if data == "publish":
                self.provider = self.training_version
                self.training_records[-1]["published_at"] = self.now
                self.actor_batch = ()
                self.actor_stages = ()
                self.pending_publish = False
                self.emit("weights_published", version=self.provider)
            elif data == "wake":
                self.actor_running = True
            elif data == "reconfigure":
                self.layout = self.pending_handoff
                self.execution_layout = self._target_physical
                self.actor_running = True
                self.allocation = (self.layout.actor_gpus, self.layout.rollout_gpus)
                if self._changed_roles[0]:
                    old = self.actor_id
                    self.actor_id += 2
                    self.emit("actor_cutover", source=old, target=self.actor_id)
                if self._changed_roles[1]:
                    self.rollout_id += 2
                self.pending_handoff = None
                self._drain.clear()
                self._checkpoint_version = self.training_version
                self.emit("handoff_complete", layout=self.layout.key)
            else:
                raise ValueError(f"Unknown Joint transition {data}")
        else:
            raise ValueError(f"Unknown Joint event {kind}")
        self.peak_gpus = max(self.peak_gpus, sum(self.allocation))

    def run(self, *, until=None, stop_after_handoff=False):
        count = 0
        self.pump()
        limit = min(
            until if until is not None else math.inf,
            self.view.now + self.config.max_simulated_seconds,
        )
        while not self.done:
            if (
                stop_after_handoff
                and self.pending_handoff is None
                and not self.decision_pending
            ):
                break
            if not self.heap:
                if until is None:
                    return self.result(
                        "incomplete", "No Joint event can advance pending work"
                    )
                next_time = limit
            else:
                next_time = min(limit, self.heap[0][0])
            self.actor_cost += (next_time - self.now) * self.allocation[0]
            self.rollout_cost += (next_time - self.now) * self.allocation[1]
            self.now = next_time
            self._materialize(self.now)
            while self.heap and self.heap[0][0] <= self.now + 1e-12:
                _, _, kind, data = heapq.heappop(self.heap)
                self._dispatch(kind, data)
                count += 1
                if count >= self.config.max_events:
                    return self.result("incomplete", "Joint event budget exhausted")
            self.pump()
            if self.now >= limit:
                break
        if (
            not self.done
            and until is None
            and not stop_after_handoff
            and self.now >= limit
        ):
            return self.result("incomplete", "Joint simulated time budget exhausted")
        return self.result(
            "complete" if self.done else "running", "Aggregate Joint forecast"
        )

    def result(self, status, reason):
        return dict(
            status=status,
            reason=reason,
            seconds=self.now - self.view.now,
            total_gpu_seconds=self.actor_cost + self.rollout_cost,
            actor_gpu_seconds=self.actor_cost,
            rollout_gpu_seconds=self.rollout_cost,
            peak_gpus=self.peak_gpus,
            completed_updates=self.training_version,
            retained_episodes=sum(
                e.completed_at is not None for e in self.episodes.values()
            )
            - self.consumed,
        )

    def switching_gpu_seconds(self):
        return self._handoff_cost
