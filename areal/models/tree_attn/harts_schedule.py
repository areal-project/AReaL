# SPDX-License-Identifier: Apache-2.0

"""Prefix-aware microbatch and DP-slot planning for tree training."""

from bisect import bisect_left
from dataclasses import dataclass
from heapq import heappop, heappush
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
        self.position = [0] * len(sequences)
        for position, row in enumerate(self.order):
            self.position[row] = position
        self.lcp_levels = [self.lcp]
        width = 2
        while width <= len(sequences):
            previous = self.lcp_levels[-1]
            half = width // 2
            self.lcp_levels.append(
                [
                    min(previous[i], previous[i + half])
                    for i in range(len(previous) - half)
                ]
            )
            width *= 2

    def shared(self, left: int, right: int) -> int:
        """LCP of two globally ordered rows, using a range minimum query."""
        a, b = self.position[left], self.position[right]
        if a == b:
            return len(self.sequences[left])
        if a > b:
            a, b = b, a
        a += 1
        level = (b - a + 1).bit_length() - 1
        width = 1 << level
        values = self.lcp_levels[level]
        return min(values[a], values[b - width + 1])

    def interval(self, start: int, end: int) -> int:
        """Compact work for a half-open lexicographic interval."""
        return self.prefix[end] - self.prefix[start] + self.lcp[start]

    def group(self, rows: list[int] | tuple[int, ...]) -> int:
        ordered = sorted(rows, key=self.position.__getitem__)
        return sum(
            len(self.sequences[row]) - (self.shared(ordered[i - 1], row) if i else 0)
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
    """Evolve the natural contiguous partition by cheapest legal cuts or merges."""
    intervals = [
        (costs.position[group[0]], costs.position[group[-1]] + 1) for group in natural
    ]
    if target > len(intervals):
        existing = {end for _, end in intervals[:-1]}
        cuts = sorted(
            (costs.lcp[cut], cut)
            for start, end in intervals
            for cut in range(start + 1, end)
        )
        required = target - len(intervals)
        if required > len(cuts):
            return None
        existing.update(cut for _, cut in cuts[:required])
        boundaries = [0, *sorted(existing), len(costs.order)]
        intervals = list(zip(boundaries[:-1], boundaries[1:]))
    while len(intervals) > target:
        options = [
            (costs.lcp[right_start], i)
            for i, ((start, _), (right_start, end)) in enumerate(
                zip(intervals[:-1], intervals[1:])
            )
            if costs.interval(start, end) <= capacity
        ]
        if not options:
            return None
        _, i = max(options)
        intervals[i : i + 2] = [(intervals[i][0], intervals[i + 1][1])]
    return [costs.order[start:end] for start, end in intervals]


def _sample_targets(targets: list[int], budget: int = 32) -> list[int]:
    if len(targets) <= budget:
        return targets
    dense = min(16, budget - 1)
    selected = set(targets[:dense])
    remaining = budget - dense
    for index in range(remaining):
        position = dense + round(index * (len(targets) - dense - 1) / (remaining - 1))
        selected.add(targets[position])
    return sorted(selected)


def _greedy_partition(
    costs: _PrefixCosts, capacity: int, order: list[int], policy: str
) -> list[list[int]]:
    groups: list[list[int]] = []
    positions: list[list[int]] = []
    work: list[int] = []
    for row in order:
        choice = None
        row_position = costs.position[row]
        for i, group_positions in enumerate(positions):
            insertion = bisect_left(group_positions, row_position)
            previous = (
                costs.order[group_positions[insertion - 1]] if insertion else None
            )
            following = (
                costs.order[group_positions[insertion]]
                if insertion < len(group_positions)
                else None
            )
            delta = len(costs.sequences[row])
            if previous is not None:
                delta -= costs.shared(previous, row)
            if following is not None:
                delta -= costs.shared(row, following)
            if previous is not None and following is not None:
                delta += costs.shared(previous, following)
            new_work = work[i] + delta
            if new_work <= capacity:
                score = new_work if policy == "best_fit" else -delta
                candidate = (score, -i, i, insertion, new_work)
                if choice is None or candidate > choice:
                    choice = candidate
        if choice is not None:
            _, _, i, insertion, new_work = choice
            groups[i].append(row)
            positions[i].insert(insertion, row_position)
            work[i] = new_work
        else:
            groups.append([row])
            positions.append([row_position])
            work.append(len(costs.sequences[row]))
    return groups


def _align_groups(
    costs: _PrefixCosts, groups: list[list[int]], capacity: int, dp_size: int
) -> list[list[int]] | None:
    active = {i: list(group) for i, group in enumerate(groups)}
    work = {i: costs.group(group) for i, group in active.items()}
    versions = dict.fromkeys(active, 0)
    merges: list[tuple[int, int, int, int, int]] = []

    def consider(i: int, j: int) -> None:
        if i > j:
            i, j = j, i
        combined = costs.group(active[i] + active[j])
        if combined <= capacity:
            saving = work[i] + work[j] - combined
            heappush(merges, (-saving, -i, -j, versions[i], versions[j]))

    indices = list(active)
    for offset, i in enumerate(indices):
        for j in indices[offset + 1 :]:
            consider(i, j)
    while len(active) > dp_size and merges:
        _, neg_i, neg_j, old_i, old_j = heappop(merges)
        i, j = -neg_i, -neg_j
        if i not in active or j not in active:
            continue
        if versions[i] != old_i or versions[j] != old_j:
            continue
        active[i].extend(active.pop(j))
        work[i] = costs.group(active[i])
        del work[j], versions[j]
        versions[i] += 1
        for other in active:
            if other != i:
                consider(i, other)
    groups = list(active.values())
    target = ceil(len(groups) / dp_size) * dp_size
    while len(groups) < target:
        options = []
        for i, group in enumerate(groups):
            if len(group) < 2:
                continue
            old_work = costs.group(group)
            for row in group:
                remaining = [other for other in group if other != row]
                increase = costs.group(remaining) + len(costs.sequences[row]) - old_work
                options.append((increase, i, row))
        if not options:
            return None
        _, i, row = min(options)
        groups[i].remove(row)
        groups.append([row])
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


def _balanced_natural_partition(
    natural: list[list[int]], dp_size: int
) -> list[list[int]] | None:
    """Retain the previous balanced natural plan as a comparison candidate."""
    groups = [list(group) for group in natural]
    target = ceil(len(groups) / dp_size) * dp_size
    while len(groups) < target:
        candidates = (i for i, group in enumerate(groups) if len(group) > 1)
        index = max(candidates, key=lambda i: (len(groups[i]), -i), default=None)
        if index is None:
            return None
        group = groups[index]
        cut = len(group) // 2
        groups[index : index + 1] = [group[:cut], group[cut:]]
    return groups


def _refine_partition(
    costs: _PrefixCosts, groups: list[list[int]], capacity: int
) -> list[list[int]]:
    """Bounded boundary and lexically local membership repair from Section 3.2."""
    groups = [sorted(group, key=costs.position.__getitem__) for group in groups]
    contiguous = (
        costs.position[groups[0][0]] == 0
        and all(
            costs.position[group[-1]] - costs.position[group[0]] + 1 == len(group)
            for group in groups
        )
        and all(
            costs.position[group[-1]] + 1 == costs.position[groups[i + 1][0]]
            for i, group in enumerate(groups[:-1])
        )
    )
    if contiguous:
        for _ in range(2):
            changed = False
            for i in range(len(groups) - 1):
                left, right = groups[i], groups[i + 1]
                current = (
                    costs.group(left) + costs.group(right),
                    max(costs.group(left), costs.group(right)),
                )
                best = current
                replacement = None
                for shift in range(-min(8, len(left) - 1), min(8, len(right) - 1) + 1):
                    if shift == 0:
                        continue
                    combined = left + right
                    cut = len(left) + shift
                    a, b = combined[:cut], combined[cut:]
                    a_work, b_work = costs.group(a), costs.group(b)
                    score = (a_work + b_work, max(a_work, b_work))
                    if a_work <= capacity and b_work <= capacity and score < best:
                        best, replacement = score, (a, b)
                if replacement is not None:
                    groups[i], groups[i + 1] = replacement
                    changed = True
            if not changed:
                break

    for _ in range(4):
        owner = {row: i for i, group in enumerate(groups) for row in group}
        work = [costs.group(group) for group in groups]
        best_saving = 0
        best_move = None
        for position, row in enumerate(costs.order):
            source = owner[row]
            if len(groups[source]) < 2:
                continue
            source_without = [other for other in groups[source] if other != row]
            source_work = costs.group(source_without)
            for neighbor in costs.order[
                max(0, position - 8) : min(len(costs.order), position + 9)
            ]:
                destination = owner[neighbor]
                if destination == source:
                    continue
                target_with = groups[destination] + [row]
                target_work = costs.group(target_with)
                if target_work > capacity:
                    continue
                saving = work[source] + work[destination] - source_work - target_work
                if saving > best_saving:
                    best_saving = saving
                    best_move = (source, destination, source_without, target_with)
        if best_move is not None:
            source, destination, source_without, target_with = best_move
            groups[source], groups[destination] = source_without, target_with
            continue

        best_swap = None
        for position, row in enumerate(costs.order):
            source = owner[row]
            for other in costs.order[position + 1 : position + 9]:
                destination = owner[other]
                if destination == source:
                    continue
                source_with = [
                    other if member == row else member for member in groups[source]
                ]
                target_with = [
                    row if member == other else member for member in groups[destination]
                ]
                source_work, target_work = (
                    costs.group(source_with),
                    costs.group(target_with),
                )
                if source_work > capacity or target_work > capacity:
                    continue
                saving = work[source] + work[destination] - source_work - target_work
                if saving > best_saving:
                    best_saving = saving
                    best_swap = (source, destination, source_with, target_with)
        if best_swap is None:
            break
        source, destination, source_with, target_with = best_swap
        groups[source], groups[destination] = source_with, target_with
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
        candidates = [
            groups
            for target in _sample_targets(targets)
            if (groups := _evolve_partition(costs, natural, capacity, target))
            is not None
        ]
        balanced_natural = _balanced_natural_partition(natural, dp_size)
        if balanced_natural is not None:
            candidates.append(balanced_natural)

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
    greedy_index = None
    if greedy:
        greedy_index = len(candidates)
        candidates.append(
            min(
                greedy,
                key=lambda groups: (len(groups), sum(costs.group(g) for g in groups)),
            )
        )
    if not candidates:
        raise ValueError("no capacity-feasible tree schedule")
    if greedy_index is not None:
        if len(sequences) <= 512:
            anchors = [greedy_index]
        else:
            anchors = [greedy_index]
            base = list(range(greedy_index))
            if base:
                cheapest = min(
                    base, key=lambda i: sum(costs.group(g) for g in candidates[i])
                )
                anchors.append(cheapest)
                remaining = [i for i in base if i != cheapest]
                if remaining:
                    anchors.append(
                        min(
                            remaining,
                            key=lambda i: (
                                abs(len(candidates[i]) - len(candidates[greedy_index])),
                                i,
                            ),
                        )
                    )
        candidates.extend(
            _refine_partition(costs, candidates[i], capacity) for i in anchors
        )
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
