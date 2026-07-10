from unittest.mock import patch

import torch
import torch.distributed as dist

from areal.utils.stats_tracker import (
    DistributedStatsTracker,
    ReduceType,
    _StatMetadata,
)


def test_export_uses_key_sync_group_for_missing_per_key_stat():
    tracker = DistributedStatsTracker()
    dp_group = object()
    cp_dp_group = object()
    remote_metadata = {
        "n_tokens": _StatMetadata(ReduceType.SUM, None, True),
        "vocab_min_logits": _StatMetadata(
            ReduceType.AVG_MIN_MAX,
            "n_tokens",
            True,
        ),
    }
    all_reduce_calls = []

    def fake_get_world_size(group):
        return 2 if group is cp_dp_group else 1

    def fake_all_gather_object(output, local_metadata, group):
        if group is dp_group:
            assert local_metadata == {}
            output[:] = [local_metadata]
        else:
            assert group is cp_dp_group
            assert local_metadata == {}
            output[:] = [local_metadata, remote_metadata]

    def fake_all_reduce(tensor, group=None, op=None):
        all_reduce_calls.append((group, op))

    with (
        patch(
            "areal.utils.stats_tracker.dist.get_world_size",
            side_effect=fake_get_world_size,
        ),
        patch(
            "areal.utils.stats_tracker.dist.all_gather_object",
            side_effect=fake_all_gather_object,
        ),
        patch(
            "areal.utils.stats_tracker.dist.all_reduce",
            side_effect=fake_all_reduce,
        ),
    ):
        tracker.export(
            reduce_group=dp_group,
            key_sync_group=cp_dp_group,
            reset=False,
        )

    assert [group for group, _ in all_reduce_calls] == [cp_dp_group] * 5
    assert [op for _, op in all_reduce_calls] == [
        None,
        None,
        None,
        dist.ReduceOp.MIN,
        dist.ReduceOp.MAX,
    ]


def test_export_keeps_default_reduce_group_when_key_sync_group_is_larger():
    tracker = DistributedStatsTracker()
    dp_group = object()
    cp_dp_group = object()
    tracker.denominator(n_seqs=torch.ones(2, dtype=torch.bool))
    all_reduce_groups = []

    def fake_get_world_size(group):
        return 1

    def fake_all_gather_object(output, local_metadata, group):
        if group is dp_group:
            assert set(local_metadata) == {"n_seqs"}
            output[:] = [local_metadata]
        else:
            assert group is cp_dp_group
            assert local_metadata == {}
            output[:] = [local_metadata]

    def fake_all_reduce(tensor, group=None, op=None):
        all_reduce_groups.append(group)

    with (
        patch(
            "areal.utils.stats_tracker.dist.get_world_size",
            side_effect=fake_get_world_size,
        ),
        patch(
            "areal.utils.stats_tracker.dist.all_gather_object",
            side_effect=fake_all_gather_object,
        ),
        patch(
            "areal.utils.stats_tracker.dist.all_reduce",
            side_effect=fake_all_reduce,
        ),
    ):
        tracker.export(
            reduce_group=dp_group,
            key_sync_group=cp_dp_group,
            reset=False,
        )

    assert all_reduce_groups == [dp_group]
