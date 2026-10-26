# SPDX-License-Identifier: Apache-2.0

"""Coalesce affine, fixed-membership decode runs between structural events."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from areal.elastic.policy.basic.forecast import ServiceView
from areal.elastic.policy.reference.cost_model._rollout.scheduling import policy_for
from areal.elastic.policy.reference.cost_model._rollout.types import ServiceSlice

if TYPE_CHECKING:
    from .online_simulation import OnlineSimulation


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


def advance_decode(sim: OnlineSimulation, until: float | None) -> None:
    """Skip only full decode iterations; keep the next GPU iteration in flight.

    All replicas share the earliest structural boundary. Independent decode
    completions need not wake the other replicas, but any request completion,
    prefill, arrival, tool, Actor/publication or handoff event does.
    """
    affine = getattr(sim.models, "decode_affine", None)
    if (
        affine is None
        or sim.sink is not None
        or sim.publication != "idle"
        or sim._forecast_slo_guard is not None
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
            cfg = sim.models.rollout_config(deployment.layout)
            # If this pure-decode batch contains every assigned row, no row
            # can enter or change priority before a membership revision.
            same_members = len(batch.slices) == len(group.rows)
            if not same_members:
                upcoming, _ = policy_for(cfg.backend).batch(
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
                # Lengths belong to this simulator's workload, never the policy.
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
    sim.now = max(run.end(steps) for run, steps in changes)
    sim.pool.advance(sim.now)
