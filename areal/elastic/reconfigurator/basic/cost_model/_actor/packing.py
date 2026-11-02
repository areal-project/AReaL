# SPDX-License-Identifier: Apache-2.0
# Initial analytical primitives frozen from cfdbb716; owned independently here.

"""Reproduce the observable parts of AReaL dispatch, PPO splitting and FFD.

Exact per-sequence lengths are optional. Summary-only predictions explicitly use
surrogate lengths; they never claim the inferred bins were observed in a trace.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

from .....contracts import WorkloadStats
from .config import ActorConfig


@dataclass(frozen=True)
class DispatchConfig:
    """Audited runtime path; controller weights are per trajectory before grouping.

    proxy preserves the old summary approximation without claiming exact dispatch.
    controller matches TrainController; redistribute matches multi-controller FFD.
    """

    mode: str = "proxy"
    group_size: int = 1

    def __post_init__(self) -> None:
        if self.mode not in {"proxy", "controller", "redistribute"}:
            raise ValueError("Unknown Expert dispatch mode")
        if type(self.group_size) is not int or self.group_size < 1:
            raise ValueError("Dispatch group_size must be a positive integer")
        if self.mode == "redistribute" and self.group_size != 1:
            raise ValueError("Redistribution does not support controller atomic groups")


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
        trajectories, offset = [], 0
        for count in workload.trajectory_group_sizes:
            trajectories.append(lengths[offset : offset + count])
            offset += count
        weights = (
            list(workload.trajectory_dispatch_weights)
            if workload.trajectory_dispatch_weights is not None
            else [sum(t) for t in trajectories]
        )
        if dispatch.mode == "redistribute":
            if len(trajectories) % dp:
                raise ValueError(
                    "Gathered trajectory count must divide DP; runtime requires equal pre-gather list sizes"
                )
            weights = [sum(t) for t in trajectories]
            if (
                workload.trajectory_dispatch_weights is not None
                and list(workload.trajectory_dispatch_weights) != weights
            ):
                raise ValueError(
                    "Redistribution uses attended tokens, not controller weights"
                )
            partitions = _ffd_indices(weights, int(1e12), dp)
            if len(partitions) != dp:
                raise ValueError(
                    "Redistribution produced a different number of DP ranks"
                )
        else:
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
    if dispatch.mode == "redistribute":
        return [
            [lengths[i] for i in rank] for rank in _ffd_indices(lengths, int(1e12), dp)
        ]
    if len(lengths) % dp == 0:
        return balanced_partition(lengths, dp)
    # Unknown multi-row trajectory grouping: n_seqs%DP need not be zero in the
    # real controller. Use a declared proxy, not a false infeasibility assertion.
    ranks, totals = [[] for _ in range(dp)], [0] * dp
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
        bins, ordered = [], []
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
    # Engine packing restores original row order; redistribution does not.
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
    logprob: tuple[tuple[Microbatch, ...], ...]  # DP ranks -> synchronized mbs
    train: tuple[
        tuple[tuple[Microbatch, ...], ...], ...
    ]  # PPO optimizer calls -> DP -> mbs
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
    seq_alignment = (
        (p.tp * p.cp * 2 if p.cp > 1 else p.tp) if config.backend == "megatron" else 1
    )
    aligned = [align(n, seq_alignment) for n in lengths]
    total = sum(aligned)
    # Shared utils.data uses 256 tokens; Archon additionally derives a 2MiB page.
    page = max((2 << 20) // hidden_size // 2, 1) if config.backend == "archon" else 256
    batch_alignment = (
        math.lcm(page, 2 * p.cp * p.tp) if config.backend == "archon" else page
    )
    computed = (
        config.max_tokens_per_mb
        if config.pad_to_maximum
        else align(align(sum(lengths), 256), batch_alignment)
    )
    if config.backend == "megatron":
        computed = align(max(computed, total), math.lcm(seq_alignment, 256))
    if computed < total or computed % (
        p.cp * (p.tp if config.backend == "archon" else 1)
    ):
        raise ValueError(
            "Aligned packed microbatch cannot fit the configured token limit/layout"
        )
    padding = computed - total
    # THD padding appends ONE real cu_seqlens segment before attention computation.
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
    # PPO first balances local sequence groups by count, then engine token-packs each.
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
        and (
            dispatch.mode == "redistribute"
            or (
                dispatch.mode == "controller"
                and workload.trajectory_dispatch_weights is not None
            )
        )
    )
    if dispatch.mode == "proxy" and workload.per_dp_sequence_lengths is None:
        assumptions.append(
            "Dispatch runtime path unknown; using balanced-greedy proxy, not exact rank assignment"
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
        and dispatch.mode != "redistribute"
    ):
        assumptions.append(
            "Trajectory dispatch weights approximated by attended tokens; RTensor controller weights may include rectangular padding"
        )
    if workload.max_seq_len is None and not exact_lengths:
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
