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


def test_weighted_mean_fractional_weights_preserve_partitioned_mean():
    tracker = DistributedStatsTracker()
    tracker.weighted_mean("loss", torch.tensor(2.0), torch.tensor(0.125))
    tracker.weighted_mean("loss", torch.tensor(8.0), torch.tensor(0.375))
    combined = DistributedStatsTracker()
    combined.weighted_mean("loss", torch.tensor(6.5), torch.tensor(0.5))

    assert tracker.export() == combined.export() == {"loss": 6.5}


def test_stat_weighted_average_requires_weighted_mean():
    tracker = DistributedStatsTracker()
    tracker.denominator(tokens=torch.ones(2, dtype=torch.bool))

    with pytest.raises(ValueError, match="Use weighted_mean"):
        tracker.stat(
            "tokens", reduce_type=ReduceType.WEIGHTED_AVG, loss=torch.tensor([2.0, 8.0])
        )

    assert tracker.export() == {"tokens": 2.0}


def test_weighted_mean_zero_weights_ignore_nonfinite_values():
    tracker = DistributedStatsTracker()
    for value in (float("nan"), float("inf"), -float("inf")):
        tracker.weighted_mean("loss", torch.tensor(value), torch.tensor(0.0))

    assert tracker.export(reset=False) == {}
    tracker.weighted_mean("loss", torch.tensor(3.0), torch.tensor(0.25))
    assert tracker.export() == {"loss": 3.0}


def test_weighted_mean_snapshots_detached_inputs_and_resets_scoped_key():
    tracker = DistributedStatsTracker("update")
    value = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    weight = torch.tensor(2, dtype=torch.int64)
    tracker.weighted_mean("loss/avg", value, weight)
    with torch.no_grad():
        value.fill_(99)
        weight.fill_(99)

    recorded = tracker.stats["update/loss/avg"][0]
    assert recorded.dtype == torch.float32
    assert not recorded.requires_grad
    assert tracker.export(reset=False) == {"update/loss/avg": 3.0}
    assert tracker.export(key="loss/avg") == {"update/loss/avg": 3.0}
    assert tracker.export() == {}
    tracker.weighted_mean("loss/avg", torch.tensor(5.0), torch.tensor(1.0))
    assert tracker.export() == {"update/loss/avg": 5.0}
    assert tracker.export() == {}


@pytest.mark.parametrize("argument", ["value", "weight"])
@pytest.mark.parametrize("invalid", [1.0, torch.ones(1), torch.tensor(1j)])
def test_weighted_mean_invalid_scalar_input_rejected(argument, invalid):
    tracker = DistributedStatsTracker()
    kwargs = dict(value=torch.tensor(1.0), weight=torch.tensor(1.0))
    kwargs[argument] = invalid

    with pytest.raises(ValueError, match=argument):
        tracker.weighted_mean("loss", **kwargs)
    assert tracker.export() == {}


@pytest.mark.parametrize("compact_first", [False, True])
def test_compact_and_weighted_stats_reject_same_key_without_corrupting_value(
    compact_first,
):
    tracker = DistributedStatsTracker()

    def compact():
        tracker.stat_compact(
            torch.ones(2, dtype=torch.bool),
            ReduceType.AVG,
            loss=torch.tensor([2.0, 4.0]),
        )

    def weighted():
        tracker.weighted_mean("loss", torch.tensor(7.0), torch.tensor(0.25))

    first, second = (compact, weighted) if compact_first else (weighted, compact)
    first()
    with pytest.raises(ValueError, match="compact stats"):
        second()
    assert tracker.export()["loss"] == (3.0 if compact_first else 7.0)
