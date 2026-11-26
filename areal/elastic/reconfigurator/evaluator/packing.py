# SPDX-License-Identifier: Apache-2.0

"""AReaL Actor dispatch and microbatch packing."""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

from .contracts import ActorConfig
from .workload import WorkloadStats


@dataclass(frozen=True)
class DispatchConfig:
    """Single-controller grouping of consecutive trajectories into atomic items."""

    group_size: int = 1

    def __post_init__(self) -> None:
        if type(self.group_size) is not int or self.group_size < 1:
            raise ValueError("Dispatch group_size must be a positive integer")


def align(value: int, multiple: int) -> int:
    return math.ceil(value / multiple) * multiple


def _partition_indices(lengths: list[int | float], groups: int) -> list[list[int]]:
    if len(lengths) < groups or len(lengths) % groups:
        raise ValueError("Atomic dispatch item count must be >= DP and divisible by DP")
    capacity = len(lengths) // groups
    result: list[list[int]] = [[] for _ in range(groups)]
    sums = [0.0] * groups
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
    if workload.trajectory_group_sizes is not None:
        trajectories, offset = [], 0
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
        return [[lengths[i] for i in rank] for rank in _partition_indices(lengths, dp)]
    ranks: list[list[int]] = [[] for _ in range(dp)]
    totals = [0] * dp
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
        bins: list[list[int]] = []
        ordered: list[tuple[int, int]] = []
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
class Microbatch:
    real_tokens: int
    compute_tokens: int
    attention_length_squared: int
    padding_tokens: int


@dataclass(frozen=True)
class PackingPlan:
    logprob: tuple[tuple[Microbatch, ...], ...]
    train: tuple[tuple[tuple[Microbatch, ...], ...], ...]


def _pad(lengths: list[int], config: ActorConfig) -> Microbatch:
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


def _pack_ranks(shards: list[list[int]], config: ActorConfig):
    divisor = (
        config.parallel.pp
        if config.backend == "megatron" or config.parallel.pp > 1
        else config.n_mbs_divisor
    )
    minimum = max(config.minimum_microbatches, 1)
    while True:
        grouped = [
            ffd(shard, config.max_tokens_per_mb, minimum, divisor) for shard in shards
        ]
        synchronized = max(len(bins) for bins in grouped)
        if all(len(bins) == synchronized for bins in grouped):
            break
        minimum = synchronized
    return tuple(tuple(_pad(group, config) for group in groups) for groups in grouped)


def plan_workload(
    config: ActorConfig,
    workload: WorkloadStats,
    *,
    dispatch: DispatchConfig = DispatchConfig(),
) -> PackingPlan:
    shards = dispatch_workload(
        workload, workload.lengths(), config.parallel.dp, dispatch
    )
    if len(shards[0]) % config.granularity:
        raise ValueError("Local sequence count must divide microbatch granularity")
    if config.granularity != 1:
        raise ValueError(
            "Grouped trajectories need a custom packing component; bundled model supports granularity=1"
        )
    logprob = _pack_ranks(shards, config) if config.compute_prox_logp else ()
    parts = [ffd(shard, sum(shard), config.ppo_n_minibatches) for shard in shards]
    if any(len(groups) != config.ppo_n_minibatches for groups in parts):
        raise ValueError("PPO minibatch partition is inconsistent")
    train = tuple(
        _pack_ranks([groups[i] for groups in parts], config)
        for i in range(config.ppo_n_minibatches)
    )
    return PackingPlan(logprob, train)
