# SPDX-License-Identifier: Apache-2.0

"""AReaL event timeline, Actor dispatch and microbatch packing."""

from __future__ import annotations

import bisect
import math
from collections import deque
from dataclasses import dataclass, field, replace
from functools import cache
from heapq import heapify, heappop, heappush
from itertools import count
from math import isfinite

from areal.elastic.reconfigurator.evaluator.contracts import ActorConfig
from areal.elastic.reconfigurator.evaluator.workload import (
    Backlog,
    InflightCall,
    OnlineObservation,
    WorkloadStats,
)
from areal.elastic.reconfigurator.optimizer import Layout, Prediction

from ..actor.capacity import actor_capacity
from ..contracts import ServiceState


@dataclass(frozen=True)
class DispatchConfig:
    """Single-controller grouping of consecutive trajectories into atomic items."""

    group_size: int = 1

    def __post_init__(self) -> None:
        if type(self.group_size) is not int or self.group_size < 1:
            raise ValueError("Dispatch group_size must be a positive integer")


def align(value: int, multiple: int) -> int:
    return math.ceil(value / multiple) * multiple


def balanced_partition(lengths: list[int], groups: int) -> list[list[int]]:
    """AReaL balanced_greedy_partition: equal counts, then greedily balance tokens."""
    return [[lengths[i] for i in rank] for rank in _partition_indices(lengths, groups)]


def _partition_indices(lengths: list[int | float], groups: int) -> list[list[int]]:
    if len(lengths) < groups or len(lengths) % groups:
        raise ValueError("Atomic dispatch item count must be >= DP and divisible by DP")
    capacity = len(lengths) // groups
    result = [[] for _ in range(groups)]
    sums = [0] * groups
    for index in sorted(range(len(lengths)), key=lambda i: -lengths[i]):
        selected = min(
            (g for g in range(groups) if len(result[g]) < capacity),
            key=lambda g: sums[g],
        )
        result[selected].append(index)
        sums[selected] += lengths[index]
    return result


def dispatch_workload(
    workload: WorkloadStats,
    lengths: list[int],
    dp: int,
    dispatch: DispatchConfig = DispatchConfig(),
) -> list[list[int]]:
    if workload.per_dp_sequence_lengths is not None:
        if len(workload.per_dp_sequence_lengths) != dp:
            raise ValueError("per_dp_sequence_lengths must match config DP size")
        return [list(rank) for rank in workload.per_dp_sequence_lengths]
    if workload.trajectory_group_sizes is not None:
        trajectories, offset = ([], 0)
        for count in workload.trajectory_group_sizes:
            trajectories.append(lengths[offset : offset + count])
            offset += count
        weights = (
            list(workload.trajectory_dispatch_weights)
            if workload.trajectory_dispatch_weights is not None
            else [sum(t) for t in trajectories]
        )
        size = dispatch.group_size
        if len(trajectories) % size:
            raise ValueError("Trajectory count must divide dispatch group_size")
        grouped_weights = [
            sum(weights[i : i + size]) for i in range(0, len(weights), size)
        ]
        partitions = [
            [i * size + j for i in rank for j in range(size)]
            for rank in _partition_indices(grouped_weights, dp)
        ]
        return [[n for i in rank for n in trajectories[i]] for rank in partitions]
    if len(lengths) < dp:
        raise ValueError("Actual actor sequences are fewer than DP ranks")
    if len(lengths) % dp == 0:
        return balanced_partition(lengths, dp)
    ranks, totals = ([[] for _ in range(dp)], [0] * dp)
    for length in sorted(lengths, reverse=True):
        chosen = min(range(dp), key=lambda i: totals[i])
        ranks[chosen].append(length)
        totals[chosen] += length
    return ranks


def _ffd_indices(
    lengths: list[int], capacity: int, minimum: int, divisor: int = 1
) -> list[list[int]]:
    """Match runtime NumPy argsort, including its non-stable equal-length ties."""
    import numpy as np

    minimum = max(minimum, divisor)
    if any(n > capacity for n in lengths) or len(lengths) < minimum:
        raise ValueError(
            "Workload exceeds token capacity or has too few sequences for minimum microbatches"
        )
    while True:
        bins, ordered = ([], [])
        for item in np.argsort(-np.asarray(lengths, dtype=np.int64)).tolist():
            value = lengths[item]
            if len(ordered) < minimum or ordered[0][0] + value > capacity:
                index = len(bins)
                bins.append([item])
                bisect.insort(ordered, (value, index))
            else:
                stop = bisect.bisect_right(ordered, (capacity - value, len(lengths)))
                selected = min(range(stop), key=lambda j: len(bins[ordered[j][1]]))
                total, index = ordered.pop(selected)
                bins[index].append(item)
                bisect.insort(ordered, (total + value, index))
        if len(bins) % divisor == 0:
            return bins
        minimum += divisor - minimum % divisor
        if len(lengths) < minimum:
            raise ValueError("Cannot satisfy microbatch divisor with this workload")


def ffd(
    lengths: list[int], capacity: int, minimum: int, divisor: int = 1
) -> list[list[int]]:
    bins = _ffd_indices(lengths, capacity, minimum, divisor)
    return [[lengths[i] for i in group] for group in sorted(sorted(g) for g in bins)]


@dataclass(frozen=True)
class RankPacking:
    """Predicted pre-shard work before and after cross-DP count synchronization."""

    phase: str
    call_index: int
    dp_rank: int
    sequences: int
    real_tokens: int
    capacity_microbatch_lower_bound: int
    local_microbatches: int | None
    synchronized_microbatches: int

    def __post_init__(self) -> None:
        if self.phase not in {"recompute_logp", "train_step"}:
            raise ValueError("Unknown rank packing phase")
        for name in (
            "call_index",
            "dp_rank",
            "sequences",
            "real_tokens",
            "capacity_microbatch_lower_bound",
            "synchronized_microbatches",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < (
                0 if name in {"call_index", "dp_rank"} else 1
            ):
                raise ValueError(f"Invalid rank packing {name}")
        if self.local_microbatches is not None and (
            type(self.local_microbatches) is not int
            or not self.capacity_microbatch_lower_bound
            <= self.local_microbatches
            <= self.synchronized_microbatches
        ):
            raise ValueError(
                "Local microbatches must respect capacity and synchronized count"
            )
        if (
            not self.capacity_microbatch_lower_bound
            <= self.synchronized_microbatches
            <= self.sequences
        ):
            raise ValueError(
                "Synchronized count must respect capacity and nonempty sequences"
            )


@dataclass(frozen=True)
class Microbatch:
    real_tokens: int
    compute_tokens: int
    attention_length_squared: int
    padding_tokens: int

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value < 0
            for value in (
                self.real_tokens,
                self.compute_tokens,
                self.attention_length_squared,
                self.padding_tokens,
            )
        ):
            raise ValueError("Microbatch work must contain nonnegative integers")
        if (
            self.real_tokens < 1
            or self.compute_tokens < self.real_tokens
            or self.attention_length_squared < 1
        ):
            raise ValueError("Microbatch tokens and attention work must be positive")
        if self.padding_tokens > self.compute_tokens - self.real_tokens:
            raise ValueError(
                "Padding exceeds the difference between compute and real tokens"
            )


@dataclass(frozen=True)
class PackingPlan:
    logprob: tuple[tuple[Microbatch, ...], ...]
    train: tuple[tuple[tuple[Microbatch, ...], ...], ...]
    exact_lengths: bool
    exact_dispatch: bool
    assumptions: tuple[str, ...]
    count_source: str = "unspecified"
    rank_work: tuple[RankPacking, ...] = ()
    dispatch: DispatchConfig = DispatchConfig()

    def __post_init__(self) -> None:
        if self.count_source not in {"predicted", "observed", "unspecified"}:
            raise ValueError("Unknown packing count source")
        object.__setattr__(self, "logprob", tuple(tuple(rank) for rank in self.logprob))
        object.__setattr__(
            self,
            "train",
            tuple(tuple(tuple(rank) for rank in update) for update in self.train),
        )
        object.__setattr__(self, "assumptions", tuple(self.assumptions))
        object.__setattr__(self, "rank_work", tuple(self.rank_work))
        if self.rank_work:
            calls = {
                ("train_step", call, dp): rank
                for call, ranks in enumerate(self.train)
                for dp, rank in enumerate(ranks)
            }
            calls.update(
                {
                    ("recompute_logp", 0, dp): rank
                    for dp, rank in enumerate(self.logprob)
                }
            )
            keys = [(r.phase, r.call_index, r.dp_rank) for r in self.rank_work]
            if len(set(keys)) != len(keys) or set(keys) != calls.keys():
                raise ValueError(
                    "Rank diagnostics must cover each packing call exactly once"
                )
            for row, key in zip(self.rank_work, keys, strict=True):
                if row.real_tokens != sum(
                    m.real_tokens for m in calls[key]
                ) or row.synchronized_microbatches != len(calls[key]):
                    raise ValueError("Rank diagnostics disagree with packed work")

    @property
    def train_compute_tokens(self) -> int:
        return sum(
            mb.compute_tokens for update in self.train for rank in update for mb in rank
        )


def _pad(lengths: list[int], config: ActorConfig, hidden_size: int) -> Microbatch:
    p = config.parallel
    seq_alignment = p.tp * p.cp * 2 if p.cp > 1 else p.tp
    aligned = [align(n, seq_alignment) for n in lengths]
    total = sum(aligned)
    computed = (
        config.max_tokens_per_mb if config.pad_to_maximum else align(sum(lengths), 256)
    )
    computed = align(max(computed, total), math.lcm(seq_alignment, 256))
    if computed < total or computed % p.cp:
        raise ValueError("Packed microbatch cannot fit the token limit/layout")
    padding = computed - total
    return Microbatch(
        sum(lengths), computed, sum(n * n for n in aligned) + padding * padding, padding
    )


def _pack_ranks(
    shards: list[list[int]],
    config: ActorConfig,
    hidden_size: int,
    observed: int | None = None,
    *,
    trace: list[RankPacking] | None = None,
    phase: str = "train_step",
    call_index: int = 0,
):
    divisor = (
        config.parallel.pp
        if config.backend == "megatron" or config.parallel.pp > 1
        else config.n_mbs_divisor
    )
    minimum = max(config.minimum_microbatches, observed or 1)
    local_counts = None
    while True:
        grouped = [
            ffd(shard, config.max_tokens_per_mb, minimum, divisor) for shard in shards
        ]
        synchronized = max(len(bins) for bins in grouped)
        if local_counts is None:
            local_counts = [len(bins) for bins in grouped]
        if observed is not None and synchronized > observed:
            raise ValueError(
                "Observed microbatch count is incompatible with the workload/config"
            )
        if all(len(bins) == synchronized for bins in grouped):
            break
        minimum = synchronized
    if trace is not None:
        trace.extend(
            (
                RankPacking(
                    phase,
                    call_index,
                    dp,
                    len(shard),
                    sum(shard),
                    math.ceil(sum(shard) / config.max_tokens_per_mb),
                    None if observed is not None else local_counts[dp],
                    synchronized,
                )
                for dp, shard in enumerate(shards)
            )
        )
    return tuple(
        tuple(_pad(group, config, hidden_size) for group in groups)
        for groups in grouped
    )


def plan_workload(
    config: ActorConfig,
    workload: WorkloadStats,
    hidden_size: int,
    *,
    tail: bool = False,
    dispatch: DispatchConfig = DispatchConfig(),
) -> PackingPlan:
    if workload.train_microbatches is not None and config.ppo_n_minibatches != 1:
        raise ValueError(
            "Average num_micro_batches cannot identify counts of multiple PPO minibatches"
        )
    lengths = workload.lengths(tail=tail, capacity=config.max_tokens_per_mb)
    shards = dispatch_workload(workload, lengths, config.parallel.dp, dispatch)
    if len(shards[0]) % config.granularity:
        raise ValueError("Local sequence count must divide microbatch granularity")
    if config.granularity != 1:
        raise ValueError(
            "Grouped trajectories need a custom packing component; bundled model supports granularity=1"
        )
    rank_work = []
    logprob = (
        _pack_ranks(
            shards, config, hidden_size, trace=rank_work, phase="recompute_logp"
        )
        if config.compute_prox_logp
        else ()
    )
    parts = [ffd(shard, sum(shard), config.ppo_n_minibatches) for shard in shards]
    if any(len(groups) != config.ppo_n_minibatches for groups in parts):
        raise ValueError("PPO minibatch partition is inconsistent")
    train = tuple(
        _pack_ranks(
            [groups[i] for groups in parts],
            config,
            hidden_size,
            workload.train_microbatches,
            trace=rank_work,
            call_index=i,
        )
        for i in range(config.ppo_n_minibatches)
    )
    assumptions = []
    exact_lengths = (
        workload.sequence_lengths is not None
        or workload.per_dp_sequence_lengths is not None
    )
    exact_dispatch = workload.per_dp_sequence_lengths is not None or (
        workload.trajectory_group_sizes is not None
        and exact_lengths
        and workload.trajectory_dispatch_weights is not None
    )
    if workload.per_dp_sequence_lengths is not None:
        assumptions.append(
            "Given per-DP sequence lengths condition dispatch; new configurations must re-dispatch logical trajectories"
        )
    if not exact_lengths:
        assumptions.append(
            "Sequence distribution/DP packing inferred from summary statistics; exact lengths are optional"
        )
    if (
        workload.per_dp_sequence_lengths is None
        and workload.trajectory_group_sizes is None
    ):
        assumptions.append(
            "Unknown atomic trajectory grouping: dispatch uses a per-sequence proxy, not exact DP packing/feasibility; multi-turn trajectories can contain multiple rows"
        )
    elif (
        workload.per_dp_sequence_lengths is None
        and workload.trajectory_dispatch_weights is None
    ):
        assumptions.append(
            "Trajectory dispatch weights approximated by attended tokens; RTensor controller weights may include rectangular padding"
        )
    if workload.max_seq_len is None and (not exact_lengths):
        assumptions.append(
            "Unknown maximum sequence length bounded by max_tokens_per_mb"
        )
    if workload.train_microbatches is None:
        assumptions.append(
            "Synchronized engine microbatch counts inferred, not read from rank logs"
        )
    return PackingPlan(
        logprob,
        train,
        exact_lengths,
        exact_dispatch,
        tuple(assumptions),
        "predicted" if workload.train_microbatches is None else "observed",
        tuple(rank_work),
        dispatch,
    )


@dataclass(frozen=True)
class Observation:
    view: OnlineObservation
    current: Layout
    backlog: Backlog | None = None
    search_ready_at: float | None = None  # earliest time a decision can be applied


@dataclass
class Call:
    group_id: str
    index: int
    order: int
    ready_at: float
    request: object
    prefilled: int = 0
    generated: int = 0
    retained: int = 0
    first: float | None = None

    admitted_at: float | None = None

    @property
    def kv_tokens(self):
        return self.prefilled + max(0, self.generated - 1)

    def resume(self):
        """Re-prefill acknowledged output as context on a fresh backend."""
        self.retained += self.generated
        self.request = replace(
            self.request,
            input_tokens=self.request.input_tokens + self.generated,
            output_tokens=self.request.output_tokens - self.generated,
        )
        self.prefilled = self.generated = 0


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


def simulate(
    obs,
    forecast,
    cfg,
    layout,
    cut,
    pub,
    on_batch,
    *,
    batch_size,
    concurrency,
    actor_model,
    rollout_model,
    request_type,
    reserve,
    scheduler,
    slo,
    staleness,
    stop_at_publication=None,
    rollout_only=False,
):
    view, size = obs.view, batch_size
    if view.busy or view.provider_version != view.training_version:
        raise ValueError("Expected an idle Actor at the published update boundary")
    end, start = forecast.input_closes_at, view.now
    rc = cfg.rollout
    shapes = {e.episode_id: e.turns for e in forecast.episodes}

    @cache
    def actor_time(batch):
        estimate = actor_model.estimate_batch(cfg.actor, batch)
        peak, cap = estimate.peak_gib, actor_model.hardware.memory_gb * 1e9 / 2**30
        if peak > cap * (1 - reserve):
            raise ValueError(
                f"Batch exceeds Actor memory allowance (peak={peak:.2f} GiB, "
                f"allowance={cap * (1 - reserve):.2f} GiB, oom=False)"
            )
        duration = estimate.duration_seconds()
        if not isfinite(duration) or duration <= 0:
            raise ValueError("Unknown Actor duration")
        return duration

    templates = [episode.turns for episode in forecast.episodes]
    capacity = (
        actor_capacity(templates, size, actor_time, pub) if not rollout_only else None
    )
    deficit = len(templates) / (end - start) - capacity if capacity is not None else 0.0
    admission_failures = queue_failures = decode_failures = 0

    # All state below belongs to this candidate. Simultaneous completions are
    # applied before the next batch is chosen or new inference work is scheduled.
    ready = deque()
    arrivals, waiting = deque(), {}
    replicas = {i: Replica() for i in range(rc.replicas)}
    events, serial = [], count()
    active = set()
    accepted = 0
    passed = completed = updates = 0
    unknown_timing = actor_busy = False
    paused = cut.seconds > 0
    generation = 0
    now = start
    # Full-output reservations stay fixed until admission, completion or pause.
    routing_dirty = True

    def push(at, kind, payload=None):
        heappush(events, (at, next(serial), kind, payload))

    def add_call(key, i, at):
        nonlocal routing_dirty
        routing_dirty = True
        t = shapes[key][i]
        req = request_type(f"turn-{i}", t.input_tokens, t.output_tokens)
        call = Call(key, i, next(serial), at, req)
        waiting[key] = call
        return call

    # Only the next arrival bounds an interval. Preserve the original event
    # serials for equal timestamps even when input is supplied out of order.
    offers = iter(
        sorted(
            (
                e.available_at,
                next(serial),
                "arrival",
                (e.episode_id, e.available_at),
            )
            for e in forecast.episodes
        )
    )

    def enqueue_offer():
        event = next(offers, None)
        if event is not None:
            heappush(events, event)

    backlog = obs.backlog
    if backlog is not None:
        shapes.update(backlog.shapes)
        arrivals.extend(backlog.queued)
        ready.extend(backlog.ready)
        accepted = len(ready)
        passed, completed = backlog.passed_calls, backlog.completed_calls
        unknown_timing = backlog.unknown_timing
        for progress in backlog.running:
            key = progress.episode_id
            call = add_call(key, progress.turn, progress.ready_at)
            call.first = progress.first_token_at
            call.prefilled, call.generated = progress.prefilled, progress.generated
            call.retained = progress.retained
            call.request = replace(
                call.request,
                input_tokens=call.request.input_tokens + call.retained,
                output_tokens=call.request.output_tokens - call.retained,
            )
            active.add(key)
            if cut.seconds:
                call.resume()
            elif progress.replica is not None:
                replicas[progress.replica].rows[key] = waiting.pop(key)
        for key, turn, at in backlog.tools:
            active.add(key)
            push(at, "tool", (key, turn))
        if backlog.actor_event is not None:
            kind, remaining = backlog.actor_event
            accepted += size
            actor_busy = True
            paused = paused or kind == "published"
            push(start + cut.seconds + remaining, kind)

    enqueue_offer()
    if paused:
        push(start + cut.seconds, "resume")
    push(end, "end")

    def simulate_step():
        nonlocal routing_dirty
        limit = min(
            concurrency,
            cfg.rollout_concurrency or size,
        )
        while (
            arrivals
            and len(active) < limit
            and (
                rollout_only
                or accepted + len(active) < (updates + 1 + staleness) * size
            )
        ):
            key, at = arrivals.popleft()
            active.add(key)
            add_call(key, 0, at)
            waiting[key].admitted_at = now
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
                    (generation, rid, tuple(zip(keys, slices))),
                )

    def advance_decode():
        # Stop at the earliest event that could change routing, batch membership,
        # policy version, or the cost model's segment on any replica.
        boundary = end
        spans = []
        for index, (when, _, kind, payload) in enumerate(events):
            if kind == "tokens":
                gen, rid, items = payload
                replica = replicas.get(rid)
                span = replica.decode_span if gen == generation and replica else None
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

    while True:
        # Process all events at now, including newly scheduled zero-length tools.
        while events and events[0][0] <= now:
            _, _, kind, payload = heappop(events)
            if kind == "arrival":
                arrivals.append(payload)
                enqueue_offer()
            elif kind == "tool":
                add_call(*payload, now)
            elif kind == "actor":
                routing_dirty = True
                paused = True
                generation += 1
                for replica in replicas.values():
                    for key, call in replica.rows.items():
                        call.resume()
                        waiting[key] = call
                    replica.rows.clear()
                    replica.busy = False
                push(now + pub, "published")
            elif kind == "published":
                paused = actor_busy = False
                updates += 1
                if stop_at_publication is not None and now >= stop_at_publication:
                    if now >= end:
                        return obs, None
                    # Publishing flushed replica KV. Calls waiting for the
                    # new weights still own their acknowledged output.
                    running = tuple(
                        InflightCall(
                            c.group_id,
                            c.index,
                            c.ready_at,
                            c.first,
                            c.prefilled,
                            c.generated,
                            c.retained,
                        )
                        for c in waiting.values()
                    )
                    tools = tuple(
                        (*item, at) for at, _, event, item in events if event == "tool"
                    )
                    keys = set(active) | set(ready) | {k for k, _ in arrivals}
                    carry = Backlog(
                        {k: shapes[k] for k in keys},
                        tuple(arrivals),
                        running,
                        tools,
                        tuple(ready),
                        passed_calls=passed,
                        completed_calls=completed,
                        unknown_timing=unknown_timing,
                    )
                    unseen = {
                        item[0] for _, _, event, item in events if event == "arrival"
                    }
                    unseen.update(item[0] for _, _, _, item in offers)
                    rest = tuple(e for e in forecast.episodes if e.episode_id in unseen)
                    version = view.training_version + updates
                    next_view = replace(
                        view,
                        now=now,
                        training_version=version,
                        provider_version=version,
                    )
                    return (
                        Observation(next_view, layout, carry),
                        replace(forecast, episodes=rest),
                    )
            elif kind == "resume":
                paused = any(event[2] == "published" for event in events)
            elif kind == "tokens":
                gen, rid, items = payload
                if gen != generation:
                    continue
                replica = replicas[rid]
                replica.busy = False
                for (key, _), item in items:
                    call = replica.rows[key]
                    if item.kind == "prefill":
                        call.prefilled += item.query_tokens
                    if not item.emit_token:
                        continue
                    if call.generated + call.retained == 0:
                        call.first = now
                    call.generated += 1
                    if call.generated != call.request.output_tokens:
                        continue
                    del replica.rows[key]
                    routing_dirty = True
                    completed += 1
                    total = call.generated + call.retained
                    if call.first is None:
                        unknown_timing = True
                    else:
                        if call.first - call.ready_at > slo.ttft_seconds:
                            if (
                                call.admitted_at is not None
                                and call.admitted_at - call.ready_at > slo.ttft_seconds
                            ):
                                admission_failures += 1
                            else:
                                queue_failures += 1
                        elif (
                            total > 1
                            and (now - call.first) / (total - 1) > slo.tpot_seconds
                        ):
                            decode_failures += 1
                        else:
                            passed += 1
                    t = shapes[key][call.index]
                    if t.tool_seconds is None:
                        if not rollout_only:
                            ready.append(key)
                        active.remove(key)
                        accepted += 1
                    else:
                        push(now + t.tool_seconds, "tool", (key, call.index + 1))
        if now >= end:
            break
        if not paused:
            if not rollout_only and len(ready) >= size and not actor_busy:
                ids = tuple(ready.popleft() for _ in range(size))
                duration = actor_time(tuple(shapes[key] for key in ids))
                if on_batch is not None:
                    on_batch(now, ids)
                push(now + duration, "actor")
                actor_busy = True
            simulate_step()
            advance_decode()
        now = events[0][0]

    pending_calls = [
        *waiting.values(),
        *(c for r in replicas.values() for c in r.rows.values()),
    ]
    # Offered first calls held by permits still contribute to prospective TTFT.
    unadmitted = sum(end - at > slo.ttft_seconds for _, at in arrivals)
    admission_failures += unadmitted
    failures = unadmitted
    for call in pending_calls:
        if (call.first is not None or not (call.generated + call.retained)) and (
            (end if call.first is None else call.first) - call.ready_at
            > slo.ttft_seconds
        ):
            failures += 1
            if (
                call.admitted_at is not None
                and call.admitted_at - call.ready_at > slo.ttft_seconds
            ):
                admission_failures += 1
            else:
                queue_failures += 1
    decided = completed + failures
    return Prediction(
        layout,
        passed / decided if decided and not unknown_timing else None,
        max(
            (end - start) * layout.gpus,
            cut.gpu_seconds + (end - start - cut.seconds) * layout.gpus,
        ),
        deficit <= 1e-12,
        max(0.0, deficit),
        completed_updates=updates,
        pending_requests=len(pending_calls) + len(arrivals),
        completed_requests=completed,
        actor_capacity=capacity,
        admission_failures=admission_failures,
        rollout_queue_failures=queue_failures,
        rollout_decode_failures=decode_failures,
        unfinished_episodes=len(active) + len(arrivals),
    )
