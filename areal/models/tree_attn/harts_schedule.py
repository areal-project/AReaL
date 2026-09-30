# SPDX-License-Identifier: Apache-2.0

"""Prefix-aware microbatch and DP-slot planning for tree training."""

from dataclasses import dataclass
from math import ceil

import numpy as np
import torch
import torch.distributed as dist

from areal.utils.data import TRANSPORT_DUMMY_KEY, concat_padded_tensors


@dataclass(frozen=True)
class TreeSchedule:
    """Microbatch rows indexed by DP replica, then synchronized slot."""

    slots: tuple[tuple[tuple[int, ...], ...], ...]
    compact_work: int
    slot_work: int
    max_replica_work: int


def _lcp(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    for i, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return i
    return min(len(left), len(right))


class _PrefixCosts:
    def __init__(self, sequences: list[tuple[int, ...]]):
        self.sequences = sequences
        self.order = sorted(range(len(sequences)), key=lambda i: (sequences[i], i))
        self.lcp = [0] + [
            _lcp(sequences[a], sequences[b])
            for a, b in zip(self.order[:-1], self.order[1:])
        ]
        self.prefix = [0]
        for i, row in enumerate(self.order):
            self.prefix.append(self.prefix[-1] + len(sequences[row]) - self.lcp[i])

    def interval(self, start: int, end: int) -> int:
        """Compact work for a half-open lexicographic interval."""
        return self.prefix[end] - self.prefix[start] + self.lcp[start]

    def group(self, rows: list[int] | tuple[int, ...]) -> int:
        ordered = sorted(rows, key=lambda i: (self.sequences[i], i))
        return sum(
            len(self.sequences[row])
            - (_lcp(self.sequences[ordered[i - 1]], self.sequences[row]) if i else 0)
            for i, row in enumerate(ordered)
        )


def _natural_partition(costs: _PrefixCosts, capacity: int) -> list[list[int]]:
    result: list[list[int]] = []
    pending = [(0, len(costs.order))]
    while pending:
        start, end = pending.pop()
        if costs.interval(start, end) <= capacity:
            result.append(costs.order[start:end])
            continue
        cut = min(
            range(start + 1, end),
            key=lambda mid: (
                costs.lcp[mid],
                max(costs.interval(start, mid), costs.interval(mid, end)),
                abs(costs.interval(start, mid) - costs.interval(mid, end)),
            ),
        )
        pending.extend(((cut, end), (start, cut)))
    return result


def _exact_partitions(
    costs: _PrefixCosts, capacity: int, targets: list[int]
) -> list[list[list[int]]]:
    n = len(costs.order)
    max_k = max(targets)
    inf = np.iinfo(np.int64).max // 4
    totals = np.full((max_k + 1, n + 1), inf, dtype=np.int64)
    peaks = np.full_like(totals, inf)
    previous = np.full((max_k + 1, n + 1), -1, dtype=np.int32)
    totals[0, 0] = peaks[0, 0] = 0
    prefix = np.asarray(costs.prefix, dtype=np.int64)
    lcp = np.asarray(costs.lcp, dtype=np.int64)
    for k in range(1, max_k + 1):
        for end in range(k, n + 1):
            starts = np.arange(k - 1, end)
            work = prefix[end] - prefix[starts] + lcp[starts]
            feasible = (work <= capacity) & (totals[k - 1, starts] < inf)
            if not feasible.any():
                continue
            starts = starts[feasible]
            work = work[feasible]
            candidate_total = totals[k - 1, starts] + work
            candidate_peak = np.maximum(peaks[k - 1, starts], work)
            winner = np.lexsort((starts, candidate_peak, candidate_total))[0]
            totals[k, end] = candidate_total[winner]
            peaks[k, end] = candidate_peak[winner]
            previous[k, end] = starts[winner]

    partitions = []
    for k in targets:
        if previous[k, n] < 0:
            continue
        end = n
        groups = []
        for level in range(k, 0, -1):
            start = int(previous[level, end])
            groups.append(costs.order[start:end])
            end = start
        partitions.append(groups[::-1])
    return partitions


def _evolve_partition(
    costs: _PrefixCosts, natural: list[list[int]], capacity: int, target: int
) -> list[list[int]] | None:
    groups = [list(group) for group in natural]
    while len(groups) != target:
        if len(groups) > target:
            options = [
                (
                    costs.group(groups[i])
                    + costs.group(groups[i + 1])
                    - costs.group(groups[i] + groups[i + 1]),
                    i,
                )
                for i in range(len(groups) - 1)
                if costs.group(groups[i] + groups[i + 1]) <= capacity
            ]
            if not options:
                return None
            _, i = max(options)
            groups[i : i + 2] = [groups[i] + groups[i + 1]]
        else:
            options = [
                (
                    costs.group(group[:cut])
                    + costs.group(group[cut:])
                    - costs.group(group),
                    i,
                    cut,
                )
                for i, group in enumerate(groups)
                for cut in range(1, len(group))
            ]
            if not options:
                return None
            _, i, cut = min(options)
            groups[i : i + 1] = [groups[i][:cut], groups[i][cut:]]
    return groups


def _greedy_partition(
    costs: _PrefixCosts, capacity: int, order: list[int], policy: str
) -> list[list[int]]:
    groups: list[list[int]] = []
    work: list[int] = []
    for row in order:
        options = []
        for i, group in enumerate(groups):
            new_work = costs.group(group + [row])
            if new_work <= capacity:
                score = new_work if policy == "best_fit" else work[i] - new_work
                options.append((score, -i, i, new_work))
        if options:
            _, _, i, new_work = max(options)
            groups[i].append(row)
            work[i] = new_work
        else:
            groups.append([row])
            work.append(len(costs.sequences[row]))
    return groups


def _align_groups(
    costs: _PrefixCosts, groups: list[list[int]], capacity: int, dp_size: int
) -> list[list[int]] | None:
    groups = [list(group) for group in groups]
    while len(groups) > dp_size:
        options = [
            (
                costs.group(groups[i])
                + costs.group(groups[j])
                - costs.group(groups[i] + groups[j]),
                i,
                j,
            )
            for i in range(len(groups))
            for j in range(i + 1, len(groups))
            if costs.group(groups[i] + groups[j]) <= capacity
        ]
        if not options:
            break
        _, i, j = max(options)
        groups[i] += groups[j]
        groups.pop(j)
    target = ceil(len(groups) / dp_size) * dp_size
    while len(groups) < target:
        options = [
            (
                costs.group(group[:-1])
                + len(costs.sequences[group[-1]])
                - costs.group(group),
                i,
            )
            for i, group in enumerate(groups)
            if len(group) > 1
        ]
        if not options:
            return None
        _, i = min(options)
        groups.append([groups[i].pop()])
    return groups


def _candidate_schedule(
    costs: _PrefixCosts, groups: list[list[int]], dp_size: int
) -> TreeSchedule:
    weighted = sorted(
        ((costs.group(group), tuple(group)) for group in groups), reverse=True
    )
    slots: list[list[tuple[int, ...]]] = [[] for _ in range(dp_size)]
    totals = [0] * dp_size
    slot_work = 0
    for start in range(0, len(weighted), dp_size):
        block = weighted[start : start + dp_size]
        slot_work += block[0][0]
        available = set(range(dp_size))
        for work, group in block:
            replica = min(available, key=lambda d: (totals[d], d))
            available.remove(replica)
            slots[replica].append(group)
            totals[replica] += work
    return TreeSchedule(
        tuple(tuple(replica) for replica in slots),
        sum(work for work, _ in weighted),
        slot_work,
        max(totals),
    )


def _refine_partition(
    costs: _PrefixCosts, groups: list[list[int]], capacity: int
) -> list[list[int]]:
    """Improve a large-batch partition near boundaries and sparse outliers."""
    groups = [list(group) for group in groups]

    def improve(i: int, j: int, left: list[int], right: list[int]) -> bool:
        if not left or not right:
            return False
        left_work, right_work = costs.group(left), costs.group(right)
        if left_work > capacity or right_work > capacity:
            return False
        if left_work + right_work >= costs.group(groups[i]) + costs.group(groups[j]):
            return False
        groups[i], groups[j] = left, right
        return True

    for _ in range(2):
        changed = False
        for i in range(len(groups) - 1):
            left, right = groups[i], groups[i + 1]
            boundary_changed = False
            for width in range(1, min(8, len(left) - 1) + 1):
                if improve(i, i + 1, left[:-width], left[-width:] + right):
                    changed = boundary_changed = True
                    break
            if boundary_changed:
                continue
            for width in range(1, min(8, len(right) - 1) + 1):
                if improve(i, i + 1, left + right[:width], right[width:]):
                    changed = True
                    break
        if not changed:
            break

    for _ in range(4):
        changed = False
        for i, source in enumerate(groups):
            if len(source) < 2:
                continue
            for row in sorted(source, key=lambda r: len(costs.sequences[r]))[:8]:
                for j, target in enumerate(groups):
                    if i != j and improve(
                        i, j, [r for r in source if r != row], target + [row]
                    ):
                        changed = True
                        break
                if changed:
                    break
            if changed:
                break
        if not changed:
            break
    return groups


def plan_tree_schedule(
    sequences: list[list[int] | tuple[int, ...]], capacity: int, dp_size: int
) -> TreeSchedule:
    """Plan compact microbatches and synchronized DP slots for one update."""
    if capacity < 1 or dp_size < 1:
        raise ValueError("tree capacity and DP size must be positive")
    if len(sequences) < dp_size:
        raise ValueError("tree scheduling needs at least one trajectory per DP rank")
    normalized = [tuple(sequence) for sequence in sequences]
    if any(len(sequence) > capacity for sequence in normalized):
        raise ValueError("a trajectory exceeds the tree microbatch capacity")
    costs = _PrefixCosts(normalized)
    natural = _natural_partition(costs, capacity)
    minimum = dp_size * max(1, ceil(costs.prefix[-1] / (capacity * dp_size)))
    maximum = max(
        minimum,
        min(
            dp_size * ceil(len(natural) / dp_size),
            dp_size * (len(sequences) // dp_size),
        ),
    )
    targets = list(range(minimum, maximum + 1, dp_size))
    if len(sequences) <= 512:
        candidates = _exact_partitions(costs, capacity, targets)
    else:
        if len(targets) > 32:
            sampled = sorted(
                set(
                    targets[:16]
                    + targets[16 :: max(1, len(targets) // 16)]
                    + targets[-1:]
                )
            )
        else:
            sampled = targets
        candidates = [
            groups
            for target in sampled
            if (groups := _evolve_partition(costs, natural, capacity, target))
            is not None
        ]

    lex = costs.order
    suffix = {
        row: len(normalized[row])
        - max(
            costs.lcp[i],
            costs.lcp[i + 1] if i + 1 < len(lex) else 0,
        )
        for i, row in enumerate(lex)
    }
    orders = [
        lex,
        lex[::-1],
        sorted(lex, key=lambda i: (-len(normalized[i]), i)),
        sorted(lex, key=lambda i: (-suffix[i], i)),
    ]
    greedy = []
    for order in orders:
        for policy in ("best_fit", "max_shared"):
            groups = _align_groups(
                costs,
                _greedy_partition(costs, capacity, order, policy),
                capacity,
                dp_size,
            )
            if groups is not None:
                greedy.append(groups)
    if greedy:
        candidates.append(
            min(
                greedy,
                key=lambda groups: (len(groups), sum(costs.group(g) for g in groups)),
            )
        )
    if not candidates:
        raise ValueError("no capacity-feasible tree schedule")
    if len(sequences) > 512:
        best = min(
            candidates,
            key=lambda groups: (
                (plan := _candidate_schedule(costs, groups, dp_size)).compact_work,
                plan.slot_work,
                plan.max_replica_work,
            ),
        )
        candidates.append(_refine_partition(costs, best, capacity))
    return min(
        (_candidate_schedule(costs, groups, dp_size) for groups in candidates),
        key=lambda plan: (plan.compact_work, plan.slot_work, plan.max_replica_work),
    )


def _select_rows(data: dict, rows: list[int], batch_size: int) -> dict:
    indices = torch.tensor(rows, dtype=torch.long)
    return {
        key: (
            value.index_select(0, indices)
            if torch.is_tensor(value) and value.ndim and value.shape[0] == batch_size
            else [value[i] for i in rows]
            if isinstance(value, list) and len(value) == batch_size
            else value
        )
        for key, value in data.items()
        if key != TRANSPORT_DUMMY_KEY
    }


def _reorder_received(
    received: list[tuple[list[int], dict]], assigned: list[int]
) -> dict:
    concatenated = concat_padded_tensors([batch for _, batch in received])
    arrival = [row for rows, _ in received for row in rows]
    positions = {row: i for i, row in enumerate(arrival)}
    return _select_rows(
        concatenated, [positions[row] for row in assigned], len(arrival)
    )


def schedule_tree_training_batch(
    data: dict, capacity: int, dp_group: dist.ProcessGroup | None
) -> tuple[dict, list[list[int]] | None]:
    """Exchange CPU rows so each DP rank receives its planned microbatch slots."""
    rank = dist.get_rank(dp_group) if dist.is_initialized() else 0
    world_size = dist.get_world_size(dp_group) if dist.is_initialized() else 1
    is_dummy = data.get(TRANSPORT_DUMMY_KEY, False)
    local_rows = 0 if is_dummy else data["input_ids"].shape[0]
    sequences = [
        tuple(ids[mask.bool()].tolist())
        for ids, mask in zip(
            data["input_ids"][:local_rows], data["attention_mask"][:local_rows]
        )
    ]
    gathered: list[list[tuple[int, ...]] | None] = [None] * world_size
    if world_size > 1:
        dist.all_gather_object(gathered, sequences, group=dp_group)
    else:
        gathered[0] = sequences
    all_sequences = [sequence for shard in gathered for sequence in shard]
    if len(all_sequences) < world_size:
        return data, None

    try:
        schedule = plan_tree_schedule(all_sequences, capacity, world_size)
    except ValueError as exc:
        if str(exc) != "no capacity-feasible tree schedule":
            raise
        # Equal nonempty slots can be impossible when capacity forces K > N.
        return data, None
    offsets = [0]
    for shard in gathered:
        offsets.append(offsets[-1] + len(shard))
    assigned = [[row for slot in replica for row in slot] for replica in schedule.slots]
    if world_size == 1:
        reordered = _select_rows(data, assigned[0], local_rows)
    else:
        global_ranks = dist.get_process_group_ranks(dp_group)
        received: list[tuple[list[int], dict]] = []
        for source in range(world_size):
            send = None
            if rank == source:
                send = []
                for destination in range(world_size):
                    global_rows = [
                        row
                        for row in assigned[destination]
                        if offsets[source] <= row < offsets[source + 1]
                    ]
                    payload = (
                        (
                            global_rows,
                            _select_rows(
                                data,
                                [row - offsets[source] for row in global_rows],
                                local_rows,
                            ),
                        )
                        if global_rows
                        else None
                    )
                    send.append(payload)
            result = [None]
            dist.scatter_object_list(
                result, send, src=global_ranks[source], group=dp_group
            )
            if result[0] is not None:
                received.append(result[0])
        reordered = _reorder_received(received, assigned[rank])

    groups = []
    offset = 0
    for slot in schedule.slots[rank]:
        groups.append(list(range(offset, offset + len(slot))))
        offset += len(slot)
    return reordered, groups
