from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist

from areal.utils.stats_tracker import (
    DistributedStatsTracker,
    ReduceType,
    _StatMetadata,
)


def test_compact_stats_match_masked_tensor_reductions_across_unequal_batches():
    tracker = DistributedStatsTracker()
    tracker.stat_compact(
        torch.tensor([True, False, True]),
        ReduceType.AVG_MIN_MAX,
        value=torch.tensor([2.0, float("nan"), 6.0]),
    )
    tracker.stat_compact(
        torch.tensor([True]),
        ReduceType.AVG_MIN_MAX,
        value=torch.tensor([12.0]),
    )
    assert tracker.export() == {
        "value/avg": 20 / 3,
        "value/min": 2,
        "value/max": 12,
    }


def test_compact_stats_no_valid_tokens_omit_avg_but_keep_sum():
    tracker = DistributedStatsTracker()
    mask = torch.tensor([False, False])
    tracker.stat_compact(mask, ReduceType.AVG_MIN_MAX, value=torch.ones(2))
    tracker.stat_compact(mask, ReduceType.SUM, count=torch.ones(2))
    assert tracker.export() == {"count": 0}


def test_compact_stats_cannot_reuse_a_normal_stat_key():
    tracker = DistributedStatsTracker()
    mask = torch.tensor([True])
    tracker.denominator(valid=mask)
    tracker.stat(denominator="valid", value=torch.ones(1))
    with pytest.raises(ValueError, match="non-compact stats"):
        tracker.stat_compact(mask, ReduceType.AVG, value=torch.ones(1))

    compact_tracker = DistributedStatsTracker()
    compact_tracker.stat_compact(mask, ReduceType.AVG, value=torch.ones(1))
    compact_tracker.denominator(valid=mask)
    with pytest.raises(ValueError, match="compact stats"):
        compact_tracker.stat(denominator="valid", value=torch.ones(1))


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


def test_export_scalar_key_missing_on_this_rank_does_not_crash():
    """A SCALAR key known only via metadata sync must still take part in the
    collective (aggregating over empty local values) and return 0.0 rather
    than NaN from a 0/0 division."""
    tracker = DistributedStatsTracker()
    dp_group = object()
    remote_metadata = {"loss_scalar": _StatMetadata(ReduceType.SCALAR, None, False)}
    all_reduce_groups = []

    def fake_get_world_size(group):
        return 2

    def fake_all_gather_object(output, local_metadata, group):
        assert group is dp_group
        assert local_metadata == {}  # this rank never recorded the scalar
        output[:] = [local_metadata, remote_metadata]

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
        result = tracker.export(reduce_group=dp_group, reset=False)

    # value + cnt are both reduced over the default group.
    assert all_reduce_groups == [dp_group, dp_group]
    assert result["loss_scalar"] == 0.0  # guarded 0/0, not NaN
    assert result["loss_scalar__count"] == 0


def test_all_reduce_moves_cpu_stat_to_nccl_device_before_reduction():
    """NCCL reductions must not receive CPU-backed rollout statistics."""
    tracker = DistributedStatsTracker()
    group = object()
    cpu_tensor = MagicMock(spec=torch.Tensor)
    cpu_tensor.device.type = "cpu"
    cuda_tensor = MagicMock(spec=torch.Tensor)
    cuda_tensor.device.type = "cuda"
    cpu_tensor.to.return_value = cuda_tensor
    platform = SimpleNamespace(
        communication_backend="nccl",
        device_type="cuda",
    )

    with (
        patch("areal.utils.stats_tracker.dist.get_backend", return_value="nccl"),
        patch("areal.utils.stats_tracker.current_platform", platform),
        patch("areal.utils.stats_tracker.dist.all_reduce") as mock_all_reduce,
    ):
        result = tracker._all_reduce(cpu_tensor, group=group)

    assert result is cuda_tensor
    cpu_tensor.to.assert_called_once_with("cuda")
    mock_all_reduce.assert_called_once_with(cuda_tensor, group=group)
