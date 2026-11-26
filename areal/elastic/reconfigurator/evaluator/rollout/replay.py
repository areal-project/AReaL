# SPDX-License-Identifier: Apache-2.0

"""Single-sided rollout replay for offered-task SLO."""

from collections import deque
from dataclasses import dataclass, field
from heapq import heapify, heappop, heappush
from itertools import count
from math import isfinite

from ..contracts import ServiceState
from ..workload import Request


@dataclass
class Call:
    group_id: str
    index: int
    order: int
    ready_at: float
    request: object
    prefilled: int = 0
    generated: int = 0
    first: float | None = None

    @property
    def kv_tokens(self):
        return self.prefilled + max(0, self.generated - 1)


@dataclass
class DecodeSpan:
    first_end: float
    first_seconds: float
    increment: float
    limit: int
    # Routing can add residents while the current decode batch is in flight.
    resident_count: int = 0

    def end(self, steps):
        return (
            self.first_end
            + (steps - 1) * self.first_seconds
            + self.increment * steps * (steps - 1) / 2
        )

    def steps_before(self, boundary):
        # Leave an ordinary iteration in flight and handle ties on the event heap.
        low, high = 0, self.limit - 1
        while low < high:
            middle = (low + high + 1) // 2
            if self.end(middle) < boundary - 1e-9:
                low = middle
            else:
                high = middle - 1
        return low


@dataclass
class Replica:
    rows: dict = field(default_factory=dict)
    busy: bool = False
    decode_span: DecodeSpan | None = None

    @property
    def kv_tokens(self):
        return sum(r.kv_tokens for r in self.rows.values())

    @property
    def reserved_tokens(self):
        return sum(
            r.request.input_tokens + r.request.output_tokens - 1
            for r in self.rows.values()
        )


def offered_fraction(forecast, rc, rollout_model, scheduler, slo, concurrency):
    """Score the first model call of every offered episode, including queued offers."""
    shapes = {e.episode_id: e.turns for e in forecast.episodes}
    offered = {e.episode_id: e.available_at for e in forecast.episodes}
    end = forecast.input_closes_at
    arrivals, waiting, active = deque(), {}, set()
    replicas = {i: Replica() for i in range(rc.replicas)}
    events, serial = [], count()
    passed = 0
    routing_dirty = True

    def push(at, kind, payload=None):
        heappush(events, (at, next(serial), kind, payload))

    def add_call(key, i, at):
        nonlocal routing_dirty
        routing_dirty = True
        turn = shapes[key][i]
        request = Request(f"turn-{i}", turn.input_tokens, turn.output_tokens)
        waiting[key] = Call(key, i, next(serial), at, request)

    for key, at in offered.items():
        push(at, "arrival", (key, at))
    push(end, "end")

    def simulate_step():
        nonlocal routing_dirty
        while arrivals and len(active) < concurrency:
            key, at = arrivals.popleft()
            active.add(key)
            add_call(key, 0, at)
        if routing_dirty:
            for key, call in tuple(waiting.items()):
                rid = scheduler.choose_replica(rc, call, replicas)
                if rid is not None:
                    replicas[rid].rows[key] = waiting.pop(key)
            routing_dirty = False
        for rid, replica in replicas.items():
            if replica.rows and not replica.busy:
                slices, keys = scheduler.batch(rc, list(replica.rows.values()))
                if not slices:
                    raise ValueError(
                        "Serving profile cannot schedule the pending tokens"
                    )
                replica.decode_span = None
                if all(s.kind == "decode" for s in slices):
                    remaining = min(
                        replica.rows[key].request.output_tokens
                        - replica.rows[key].generated
                        for key, _ in keys
                    )
                    duration, slope, steps = rollout_model.decode_interval(
                        rc, replica.kv_tokens, slices, remaining
                    )
                    if (
                        steps > 1
                        and isfinite(slope)
                        and duration + (steps - 1) * slope > 0
                    ):
                        replica.decode_span = DecodeSpan(
                            now + duration,
                            duration,
                            slope,
                            steps,
                            resident_count=len(replica.rows),
                        )
                else:
                    duration = rollout_model.estimate(
                        rc, ServiceState(replica.kv_tokens), tuple(slices)
                    ).duration.require_seconds()
                if not isfinite(duration) or duration <= 0:
                    raise ValueError("Unknown inference duration")
                replica.busy = True
                push(
                    now + duration,
                    "tokens",
                    (rid, tuple(zip(keys, slices))),
                )

    def advance_decode():
        # Stop at the earliest event that could change routing, batch membership,
        # policy version, or the cost model's segment on any replica.
        boundary = end
        spans = []
        for index, (when, _, kind, payload) in enumerate(events):
            if kind == "tokens":
                rid, items = payload
                replica = replicas[rid]
                span = replica.decode_span
                if span is not None and span.resident_count == len(replica.rows):
                    spans.append((index, rid, span, items))
                    boundary = min(boundary, span.end(span.limit))
                    continue
            boundary = min(boundary, when)

        moved = []
        for index, rid, span, items in spans:
            steps = span.steps_before(boundary)
            if steps:
                for (key, _), _ in items:
                    replicas[rid].rows[key].generated += steps
                moved.append((span.end(steps), rid, index))
                span.first_end = span.end(steps + 1)
                span.first_seconds += steps * span.increment
                span.limit -= steps
        if moved:
            # The next token would have been scheduled after the last skipped
            # completion. Replica order breaks ties as in simulate_step().
            for _, rid, index in sorted(moved):
                payload = events[index][3]
                events[index] = (
                    replicas[rid].decode_span.first_end,
                    next(serial),
                    "tokens",
                    payload,
                )
            heapify(events)

    while events:
        now = events[0][0]
        while events and events[0][0] == now:
            _, _, kind, payload = heappop(events)
            if kind == "arrival":
                arrivals.append(payload)
            elif kind == "tool":
                add_call(*payload, now)
            elif kind == "tokens":
                rid, items = payload
                replica = replicas[rid]
                replica.busy = False
                for (key, _), item in items:
                    call = replica.rows[key]
                    if item.kind == "prefill":
                        call.prefilled += item.query_tokens
                    if not item.emit_token:
                        continue
                    if call.generated == 0:
                        call.first = now
                    call.generated += 1
                    if call.generated != call.request.output_tokens:
                        continue
                    del replica.rows[key]
                    routing_dirty = True
                    if call.index == 0:
                        ttft = call.first - offered[key]
                        tpot = (now - call.first) / max(1, call.generated - 1)
                        passed += ttft <= slo.ttft_seconds and tpot <= slo.tpot_seconds
                    turn = shapes[key][call.index]
                    if turn.tool_seconds is None:
                        active.remove(key)
                    else:
                        push(now + turn.tool_seconds, "tool", (key, call.index + 1))
        if now >= end or not (events or arrivals or active):
            break
        simulate_step()
        advance_decode()
    return passed / len(offered)
