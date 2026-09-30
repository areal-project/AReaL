"""Prefix-aware microbatch and DP-slot planning."""

from itertools import combinations

import torch
import torch.distributed as dist

from areal.api.cli_args import MicroBatchSpec
from areal.models.tree_attn.harts_schedule import (
    _reorder_received,
    plan_tree_schedule,
    schedule_tree_training_batch,
)
from areal.models.tree_attn.tree import build_packed_tree_batch


def _compact_work(sequences, rows):
    ordered = sorted(tuple(sequences[i]) for i in rows)
    prefixes = {
        sequence[:end] for sequence in ordered for end in range(1, len(sequence) + 1)
    }
    return len(prefixes)


def test_tree_schedule_groups_shared_prefixes_and_balances_slots():
    sequences = [[1, 2, 3], [1, 2, 4], [5, 6], [5, 7]]

    plan = plan_tree_schedule(sequences, capacity=4, dp_size=2)

    assert plan.slots == (((0, 1),), ((2, 3),))
    assert (plan.compact_work, plan.slot_work, plan.max_replica_work) == (7, 4, 4)


def test_tree_schedule_matches_exact_contiguous_compact_optimum():
    sequences = [
        [1, 1, 1, 1],
        [1, 1, 1, 2],
        [1, 1, 2],
        [2, 1, 1],
        [2, 1, 2],
        [3, 1],
    ]
    capacity = 6
    candidates = []
    for cuts in combinations(range(1, len(sequences)), 3):
        bounds = (0, *cuts, len(sequences))
        groups = [list(range(a, b)) for a, b in zip(bounds[:-1], bounds[1:])]
        work = [_compact_work(sequences, group) for group in groups]
        if max(work) <= capacity:
            candidates.append(sum(work))

    plan = plan_tree_schedule(sequences, capacity=capacity, dp_size=2)

    assert plan.compact_work <= min(candidates)
    rows = [row for replica in plan.slots for slot in replica for row in slot]
    assert sorted(rows) == list(range(len(sequences)))
    assert all(
        _compact_work(sequences, slot) <= capacity
        for replica in plan.slots
        for slot in replica
    )


def test_tree_schedule_large_search_covers_each_row_once():
    sequences = [[i // 4, i % 4, i] for i in range(520)]

    plan = plan_tree_schedule(sequences, capacity=32, dp_size=4)

    assert len({len(replica) for replica in plan.slots}) == 1
    rows = [row for replica in plan.slots for slot in replica for row in slot]
    assert sorted(rows) == list(range(len(sequences)))
    assert all(
        _compact_work(sequences, slot) <= 32
        for replica in plan.slots
        for slot in replica
    )


def test_tree_schedule_preserves_row_labels_through_packed_microbatches():
    sequences = [
        [1] * 70 + [2],
        [2] * 70 + [2],
        [1] * 70 + [3],
        [2] * 70 + [3],
    ]
    data = {
        "input_ids": torch.tensor(sequences),
        "attention_mask": torch.ones((4, 71), dtype=torch.bool),
        "labels": torch.tensor([[value] * 71 for value in (10, 20, 30, 40)]),
    }

    reordered, groups = schedule_tree_training_batch(data, 128, None)
    packed = build_packed_tree_batch(
        reordered,
        MicroBatchSpec(max_tokens_per_mb=128),
        group_indices=groups,
        compact_padding=True,
    )

    assert len(groups) == len(packed.mbs) == 2
    assert sorted(reordered["labels"][:, 0].tolist()) == [10, 20, 30, 40]
    assert packed.group_lens == [72, 72]
    assert all(len(mb["trie_node"].all_sequence_ids) == 2 for mb in packed.mbs)


def test_tree_schedule_reorders_cross_replica_rows_with_labels():
    received = [
        (
            [0, 2],
            {
                "input_ids": torch.tensor([[1, 2, 3], [8, 9, 1]]),
                "labels": torch.tensor([[10, 10, 10], [30, 30, 30]]),
                "loss_mask": torch.tensor([[1, 0, 0], [3, 0, 0]]),
                "logprobs": torch.tensor([[0.1, 0, 0], [0.3, 0, 0]]),
            },
        ),
        (
            [1, 3],
            {
                "input_ids": torch.tensor([[1, 2, 4], [8, 9, 2]]),
                "labels": torch.tensor([[20, 20, 20], [40, 40, 40]]),
                "loss_mask": torch.tensor([[2, 0, 0], [4, 0, 0]]),
                "logprobs": torch.tensor([[0.2, 0, 0], [0.4, 0, 0]]),
            },
        ),
    ]

    reordered = _reorder_received(received, [0, 1, 2, 3])

    assert reordered["input_ids"].tolist() == [
        [1, 2, 3],
        [1, 2, 4],
        [8, 9, 1],
        [8, 9, 2],
    ]
    assert reordered["labels"][:, 0].tolist() == [10, 20, 30, 40]
    assert reordered["loss_mask"][:, 0].tolist() == [1, 2, 3, 4]
    torch.testing.assert_close(
        reordered["logprobs"][:, 0],
        torch.tensor([0.1, 0.2, 0.3, 0.4]),
        rtol=0,
        atol=0,
    )


def test_tree_schedule_uses_local_packing_when_equal_slots_are_infeasible(
    monkeypatch,
):
    data = {
        "input_ids": torch.tensor([[1]]),
        "attention_mask": torch.ones((1, 1), dtype=torch.bool),
    }
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 4)

    def gather(target, source, group=None):
        target[:] = [[(1,)], [(2,)], [(3,)], [(4,), (5,)]]

    monkeypatch.setattr(dist, "all_gather_object", gather)

    scheduled, groups = schedule_tree_training_batch(data, 1, None)

    assert scheduled is data
    assert groups is None
