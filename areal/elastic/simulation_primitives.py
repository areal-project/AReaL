# SPDX-License-Identifier: Apache-2.0

"""Shared request counters, GPU/link accounting and decode event acceleration."""

from __future__ import annotations

import heapq
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from types import MappingProxyType
from typing import Any, Protocol

from areal.elastic.contracts import (
    EpisodeSample,
    Forecast,
    OnlineObservation,
    RequestView,
    ServiceView,
    nonnegative,
)
from areal.elastic.reconfigurator.reference.evaluator.cost_model.contracts import (
    ServiceSlice,
)
from areal.elastic.reconfigurator.reference.evaluator.cost_model.rollout.scheduling import (
    scheduler_for,
)

_REQUEST_FIELDS = tuple(f.name for f in fields(RequestView))


@dataclass(slots=True)
class RequestState:
    request_id: str
    episode_id: str
    turn: int
    input_tokens: int
    ready_at: float
    prefilled: int = 0
    generated: int = 0
    first_token_at: float | None = None
    last_token_at: float | None = None
    maximum_tbt_seconds: float = 0.0
    completed_at: float | None = None
    deployment: int | None = None
    replica: int | None = None
    version_counts: tuple[tuple[int, int], ...] = ()

    @classmethod
    def copy(cls, view: RequestView) -> RequestState:
        return cls(**{name: getattr(view, name) for name in _REQUEST_FIELDS})

    def snapshot(self) -> RequestView:
        return RequestView(**{name: getattr(self, name) for name in _REQUEST_FIELDS})

    def add_tokens(self, count: int, version: int, at: float, gap: float) -> None:
        self.generated += count
        if self.first_token_at is None:
            self.first_token_at = at
        self.last_token_at = at
        self.maximum_tbt_seconds = max(self.maximum_tbt_seconds, gap)
        if self.version_counts and self.version_counts[-1][0] == version:
            old = self.version_counts[-1][1]
            self.version_counts = self.version_counts[:-1] + ((version, old + count),)
        else:
            versions = dict(self.version_counts)
            versions[version] = versions.get(version, 0) + count
            self.version_counts = tuple(sorted(versions.items()))


class RequestRecord(Protocol):
    view: RequestState
    reservation: int

    @property
    def kv_tokens(self) -> int: ...


@dataclass(frozen=True)
class _Run:
    event: tuple
    first_end: float
    first_seconds: float
    increment: float
    limit: int

    def end(self, steps: int) -> float:
        return (
            self.first_end
            + (steps - 1) * self.first_seconds
            + self.increment * steps * (steps - 1) / 2
        )

    def steps_before(self, boundary: float) -> int:
        low, high = 0, self.limit - 1
        target = boundary - 1e-9
        if high < 1 or self.first_end >= target:
            return 0
        if self.increment >= 0:
            distance = target - self.first_end
            a = self.first_seconds + self.increment / 2
            root = 2 * distance / (a + math.sqrt(a * a + 2 * self.increment * distance))
            steps = min(high, 1 + int(root))
            # The root is only an initial estimate. Preserve the exact same
            # strict boundary comparison as the reference binary search.
            while steps and self.end(steps) >= target:
                steps -= 1
            while steps < high and self.end(steps + 1) < target:
                steps += 1
            return steps
        # Keep boundary-adjacent iterations on the ordinary event path. This
        # also avoids changing simultaneous-event priority through roundoff.
        while low < high:
            middle = (low + high + 1) // 2
            if self.end(middle) < boundary - 1e-9:
                low = middle
            else:
                high = middle - 1
        return low


@dataclass
class _Plan:
    batch: ServiceView
    revision: int
    version: int
    first_seconds: float = 0.0
    increment: float = 0.0
    limit: int = 0
    continuation: bool = False
    run: _Run | None = None


@dataclass(slots=True)
class _DeferredDecode:
    plan: _Plan
    rows: tuple[RequestState, ...]
    version: int
    steps: int
    first_end: float
    last_end: float
    maximum_gap: float


@dataclass
class ReplicaState:
    rows: dict[str, RequestRecord] = field(default_factory=dict)
    reserved_tokens: int = 0
    kv_tokens: int = 0
    revision: int = 0
    decode_plan: _Plan | None = None
    pending: _DeferredDecode | None = None

    def defer_decode(
        self,
        plan: _Plan,
        steps: int,
        first_end: float,
        last_end: float,
        maximum_gap: float,
    ) -> None:
        """Summarize committed tokens; KV and event accounting stay immediate."""
        pending = self.pending
        if pending is not None and (
            pending.plan is not plan or pending.version != plan.version
        ):
            self.flush_decode()
            pending = None
        if pending is None:
            self.pending = _DeferredDecode(
                plan,
                tuple(self.rows[key].view for key, *_ in plan.batch.slices),
                plan.version,
                steps,
                first_end,
                last_end,
                maximum_gap,
            )
        else:
            pending.maximum_gap = max(
                pending.maximum_gap, maximum_gap, first_end - pending.last_end
            )
            pending.steps += steps
            pending.last_end = last_end

    def flush_decode(self) -> None:
        pending = self.pending
        if pending is None:
            return
        self.pending = None
        for view in pending.rows:
            gap = max(
                pending.maximum_gap,
                pending.first_end - view.last_token_at
                if view.last_token_at is not None
                else 0.0,
            )
            if view.first_token_at is None:
                view.first_token_at = pending.first_end
            view.add_tokens(pending.steps, pending.version, pending.last_end, gap)

    def add(self, request: RequestRecord) -> None:
        self.flush_decode()
        self.rows[request.view.request_id] = request
        self.reserved_tokens += request.view.input_tokens + request.reservation - 1
        self.kv_tokens += request.kv_tokens
        self.revision += 1

    def remove(self, request: RequestRecord) -> None:
        self.flush_decode()
        del self.rows[request.view.request_id]
        self.reserved_tokens -= request.view.input_tokens + request.reservation - 1
        self.kv_tokens -= request.kv_tokens
        self.revision += 1


@dataclass
class GPUAllocator:
    nodes: tuple[int, ...]
    leases: dict[str, tuple[str, tuple[tuple[int, int], ...]]] = field(
        default_factory=dict
    )
    gpu_seconds: dict[str, float] = field(
        default_factory=lambda: {"actor": 0.0, "rollout": 0.0}
    )
    now: float = 0.0
    peak_gpus: int = 0
    lease_gpu_seconds: dict[str, float] = field(default_factory=dict)

    def acquire(
        self, owner: str, role: str, count: int, *, tp: int = 1, spread: bool = False
    ) -> tuple[tuple[int, int], ...] | None:
        if owner in self.leases or role not in self.gpu_seconds:
            raise ValueError("A deployment needs a unique owner and known role")
        if tp not in (1, 2, 4, 8) or count % tp:
            raise ValueError("GPU count must contain complete node-local TP groups")
        occupied = {gpu for _, devices in self.leases.values() for gpu in devices}
        available = []
        for node, capacity in enumerate(self.nodes):
            free = tuple(
                (node, rank) for rank in range(capacity) if (node, rank) not in occupied
            )
            available.append(free)
            if len(free) >= count:
                return self._lease(owner, role, free[:count])
        # Legacy <=8-GPU roles stay node-local. Full candidates may spread
        # DP/PP/CP/EP across the declared nodes; every TP group stays local.
        if count > 8 or spread:
            chosen = []
            for free in available:
                take = min(count - len(chosen), len(free) // tp * tp)
                chosen.extend(free[:take])
                if len(chosen) == count:
                    return self._lease(owner, role, tuple(chosen))
        return None

    def acquire_exact(self, owner, role, devices):
        devices = tuple(devices)
        occupied = {d for _, lease in self.leases.values() for d in lease}
        if (
            owner in self.leases
            or role not in self.gpu_seconds
            or len(set(devices)) != len(devices)
            or any(
                d in occupied
                or node < 0
                or node >= len(self.nodes)
                or not 0 <= rank < self.nodes[node]
                for d in devices
                for node, rank in (d,)
            )
        ):
            raise ValueError(
                "Native target leases must fit the immutable reservation without overlap"
            )
        return self._lease(owner, role, devices)

    def _lease(self, owner, role, devices):
        self.leases[owner] = role, devices
        self.peak_gpus = max(
            self.peak_gpus, sum(len(gpus) for _, gpus in self.leases.values())
        )
        return devices

    def release(self, owner: str):
        del self.leases[owner]

    def count(self, role: str) -> int:
        return sum(
            len(devices) for name, devices in self.leases.values() if name == role
        )

    def advance(self, when: float):
        if when < self.now - 1e-12:
            raise ValueError("Resource accounting cannot move backwards")
        for owner, (role, devices) in self.leases.items():
            cost = (when - self.now) * len(devices)
            self.gpu_seconds[role] += cost
            self.lease_gpu_seconds[owner] = (
                self.lease_gpu_seconds.get(owner, 0.0) + cost
            )
        self.now = when


@dataclass
class SerializedLink:
    """Conservative shared-link prior: startup/state/publication transfers serialize.

    Reserving a fitted publication's complete wall span also serializes its host
    preparation. This is explicit headroom, not a measured network contention fit.
    """

    bandwidth_GBps: float
    available_at: float = 0.0

    def reserve(
        self,
        now: float,
        *,
        byte_count: int = 0,
        setup_seconds: float = 0.0,
        fitted_seconds: float | None = None,
    ) -> tuple[float, float]:
        seconds = (
            byte_count / (self.bandwidth_GBps * 1e9) + setup_seconds
            if fitted_seconds is None
            else fitted_seconds
        )
        nonnegative("transfer_seconds", seconds)
        start = max(now, self.available_at)
        self.available_at = start + seconds
        return start, self.available_at


@dataclass(frozen=True)
class WorkInputs:
    episodes: tuple[EpisodeSample, ...]
    truth: Mapping[str, EpisodeSample]
    heap: tuple[tuple, ...]
    sequence: int

    @classmethod
    def prepare(cls, episodes, closes_at, start, known=frozenset()):
        truth = {e.episode_id: e for e in episodes}
        if len(truth) != len(episodes):
            raise ValueError("Duplicate episode identity")
        if any(e.available_at > closes_at for e in episodes):
            raise ValueError("Input closes before an arrival")
        if closes_at < start - 1e-9:
            raise ValueError("An event cannot be scheduled in the past")
        events, sequence = [], 0
        for episode in episodes:
            if episode.available_at >= start:
                sequence += 1
                if episode.episode_id not in known:
                    heapq.heappush(
                        events,
                        (
                            episode.available_at,
                            1,
                            sequence,
                            "arrival",
                            episode.episode_id,
                        ),
                    )
        sequence += 1
        heapq.heappush(
            events, (max(start, closes_at), 1, sequence, "input_closed", None)
        )
        return cls(episodes, MappingProxyType(truth), tuple(events), sequence)


@dataclass(frozen=True)
class ReplayInputs:
    view: OnlineObservation
    forecast: Forecast
    work: WorkInputs
    requests: tuple[tuple[RequestView, int, tuple], ...]
    order: tuple[str, ...]
    active: int
    completed: int
    admission_cursor: int

    @classmethod
    def prepare(cls, view, forecast, max_new_tokens, max_context_tokens):
        order = tuple(e.episode_id for e in view.episodes)
        work = WorkInputs.prepare(
            forecast.episodes, forecast.input_closes_at, view.now, frozenset(order)
        )
        requests = []
        for request in view.requests:
            current = replace(
                request,
                deployment=1
                if request.deployment == view.rollout_deployment
                else request.deployment,
            )
            reservation = min(max_new_tokens, max_context_tokens - current.input_tokens)
            requests.append(
                (
                    current,
                    reservation,
                    tuple(getattr(current, n) for n in _REQUEST_FIELDS),
                )
            )
        return cls(
            view,
            forecast,
            work,
            tuple(requests),
            order,
            sum(e.admitted and e.completed_at is None for e in view.episodes),
            sum(e.completed_at is not None for e in view.episodes),
            next(
                (
                    i
                    for i, e in enumerate(view.episodes)
                    if not e.admitted and e.completed_at is None
                ),
                len(order),
            ),
        )


def advance_decode(sim: Any, until: float | None) -> None:
    """Skip only full decode iterations; keep the next GPU iteration in flight.

    All replicas share the earliest structural boundary. Independent decode
    completions need not wake the other replicas, but any request completion,
    prefill, arrival, tool, Actor/publication or handoff event does.
    """
    affine = getattr(sim.cost_models, "decode_affine", None)
    if (
        affine is None
        or sim.sink is not None
        or sim.publication != "idle"
        or sim.prediction_guard_active()
    ):
        return
    boundary = min(
        until if until is not None else math.inf, sim.config.max_simulated_seconds
    )
    runs = []
    for event in sim.heap:
        when, _, _, kind, data = event
        if kind != "service_complete":
            boundary = min(boundary, when)
            continue
        identity, replica = data
        deployment = sim.deployments[identity]
        batch = deployment.busy[replica]
        if when <= sim.now:
            boundary = min(boundary, when)
            continue
        group = deployment.replicas[replica]
        plan = group.decode_plan
        if (
            plan is None
            or plan.batch is not batch
            or plan.revision != group.revision
            or plan.version != deployment.version
        ):
            group.flush_decode()
            if not all(s[3] == "decode" for s in batch.slices):
                boundary = min(boundary, when)
                continue
            sim.decode_plan_builds += 1
            plan = _Plan(batch, group.revision, deployment.version)
            group.decode_plan = plan
            cfg = sim.cost_models.rollout_config(deployment.layout)
            # If this pure-decode batch contains every assigned row, no row
            # can enter or change priority before a membership revision.
            same_members = len(batch.slices) == len(group.rows)
            if not same_members:
                upcoming, _ = scheduler_for(cfg.backend).batch(
                    cfg, list(group.rows.values())
                )
                same_members = tuple(
                    (s.request_id, s.query_tokens, s.kind, s.emit_token)
                    for s in upcoming
                ) == tuple(
                    (key, query, phase, emits)
                    for key, _, query, phase, emits in batch.slices
                )
            if same_members:
                # Lengths belong to this simulator's workload, never the reconfigurator.
                selected = [sim.requests[s[0]] for s in batch.slices]
                remaining = min(
                    sim._truth[r.view.episode_id].turns[r.view.turn].output_tokens
                    - r.generated
                    for r in selected
                )
                plan.limit = min(
                    remaining,
                    (cfg.kv_capacity_tokens - group.kv_tokens) // len(selected),
                )
                plan.first_seconds, plan.increment = affine(
                    deployment.layout,
                    group.kv_tokens,
                    tuple(ServiceSlice(*s) for s in batch.slices),
                )
        else:
            sim.decode_plan_hits += 1
            if plan.run is not None and plan.run.event is event:
                # Another replica's intermediate GPU event does not change
                # this replica's analytic interval or its in-flight batch.
                sim.decode_run_reuses += 1
                runs.append(plan.run)
                boundary = min(boundary, plan.run.end(plan.run.limit))
                continue
        first, increment, limit = plan.first_seconds, plan.increment, plan.limit
        if limit < 1 or min(first, first + increment * (limit - 1)) <= 0:
            boundary = min(boundary, when)
            continue
        run = _Run(event, when, first, increment, limit)
        plan.run = run
        runs.append(run)
        boundary = min(boundary, run.end(limit))
    changes = [(run, run.steps_before(boundary)) for run in runs]
    changes = [(run, steps) for run, steps in changes if steps]
    skipped = sum(steps for _, steps in changes)
    if not skipped or sim.processed_events + skipped > sim.config.max_events:
        return
    replacements = {}
    # Preserve scheduling order for equal-time events across replicas. Every
    # replacement was scheduled after the events that remain in the heap.
    owners = {identity: index for index, identity in enumerate(sim.deployments)}
    changes.sort(
        key=lambda item: (
            item[0].end(item[1]),
            owners[item[0].event[4][0]],
            item[0].event[4][1],
        )
    )
    sim.sequence += skipped
    for index, (run, steps) in enumerate(changes):
        _, priority, sequence, kind, data = run.event
        identity, replica = data
        deployment = sim.deployments[identity]
        batch = deployment.busy[replica]
        group = deployment.replicas[replica]
        end = run.end(steps)
        internal_gap = (
            max(
                run.first_seconds + run.increment,
                run.first_seconds + run.increment * (steps - 1),
            )
            if steps > 1
            else 0.0
        )
        group.defer_decode(group.decode_plan, steps, run.first_end, end, internal_gap)
        group.kv_tokens += steps * len(batch.slices)
        deployment.busy[replica] = replace(
            batch,
            started_at=end,
            slices=tuple(
                (key, context + steps, query, phase, emits)
                for key, context, query, phase, emits in batch.slices
            ),
        )
        plan = group.decode_plan
        plan.batch = deployment.busy[replica]
        plan.first_seconds += plan.increment * steps
        plan.limit -= steps
        replacements[sequence] = (
            run.end(steps + 1),
            priority,
            sim.sequence - len(changes) + index + 1,
            kind,
            data,
        )
        sim.counters["token_generated"] += steps * len(batch.slices)
        sim.counters["service_batch"] += steps
    sim.heap = [replacements.get(event[2], event) for event in sim.heap]
    heapq.heapify(sim.heap)
    sim.processed_events += skipped
    sim.decode_steps_skipped += skipped
    sim.decode_segments += len(changes)
    sim.advance(max(run.end(steps) for run, steps in changes))
