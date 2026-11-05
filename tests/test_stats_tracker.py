import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

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


@pytest.mark.parametrize(
    "backend,adaptor_loaded", [("hccl", False), ("gloo", True), ("nccl", True)]
)
@pytest.mark.parametrize("local_value,remote_value", [(1.0, 9.0), (9.0, 1.0)])
def test_compact_stats_double_bypass_preserves_reduced_values(
    backend, adaptor_loaded, local_value, remote_value, monkeypatch
):
    """The adaptor reduces a float32 copy without writing float64 inputs back."""
    monkeypatch.setattr(
        "areal.utils.stats_tracker.current_platform",
        SimpleNamespace(communication_backend="gloo", device_type="cpu"),
    )
    tracker = DistributedStatsTracker()
    mask = torch.tensor([True])
    tracker.stat_compact(
        mask, ReduceType.AVG_MIN_MAX, value=torch.tensor([local_value])
    )
    tracker.stat_compact(mask, ReduceType.SUM, count=torch.ones(1))

    def all_reduce(tensor, *, group, op=None):
        if tensor.dtype == torch.float64:
            tensor = tensor.float()
        # The other rank has three samples, each with remote_value.
        if op == dist.ReduceOp.MIN:
            tensor.fill_(min(local_value, remote_value))
        elif op == dist.ReduceOp.MAX:
            tensor.fill_(max(local_value, remote_value))
        elif tensor.numel() == 2:
            tensor.add_(tensor.new_tensor([3 * remote_value, 3.0]))
        else:
            tensor.add_(3.0)

    monkeypatch.setattr(dist, "get_backend", lambda group: backend)
    monkeypatch.setattr(dist, "all_reduce", all_reduce)
    if adaptor_loaded:
        monkeypatch.setitem(
            sys.modules, "megatron_adaptor", ModuleType("megatron_adaptor")
        )
    else:
        monkeypatch.delitem(sys.modules, "megatron_adaptor", raising=False)
    group = object()

    result = tracker.export(key="value", reduce_group=group)
    result.update(tracker.export(key="count", reduce_group=group))

    assert result == {
        "value/avg": (local_value + 3 * remote_value) / 4,
        "value/min": 1.0,
        "value/max": 9.0,
        "count": 4.0,
    }


@pytest.mark.parametrize("backend", ["gloo", "nccl"])
def test_all_reduce_non_hccl_preserves_float64(backend, monkeypatch):
    monkeypatch.setattr(
        "areal.utils.stats_tracker.current_platform",
        SimpleNamespace(communication_backend="gloo", device_type="cpu"),
    )
    tracker = DistributedStatsTracker()
    tensor = torch.tensor([1e40], dtype=torch.float64)

    def all_reduce(tensor, *, group):
        assert tensor.dtype == torch.float64
        tensor.mul_(2)

    monkeypatch.setattr(dist, "get_backend", lambda group: backend)
    monkeypatch.setattr(dist, "all_reduce", all_reduce)
    monkeypatch.delitem(sys.modules, "megatron_adaptor", raising=False)

    result = tracker._all_reduce(tensor, group=object())

    torch.testing.assert_close(
        result, torch.tensor([2e40], dtype=torch.float64), rtol=0, atol=0
    )


def test_export_uses_key_sync_group_for_missing_per_key_stat():
    """Use Gloo peers for metadata and accelerator peers for reductions."""
    tracker = DistributedStatsTracker()
    dp_group = object()
    dp_gloo_group = object()
    cp_dp_group = object()
    cp_dp_gloo_group = object()
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
        if group is dp_gloo_group:
            assert local_metadata == {}
            output[:] = [local_metadata]
        else:
            assert group is cp_dp_gloo_group
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
            gloo_group=dp_gloo_group,
            key_sync_group=cp_dp_group,
            key_sync_gloo_group=cp_dp_gloo_group,
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
