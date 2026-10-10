# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from areal.engine.r3.asserts import R3Error
from areal.engine.r3.layout import prepare_native_replay_slabs


def _mb_input(
    *,
    orig_cu: list[int],
    padded_cu: list[int],
) -> SimpleNamespace:
    old_cu = torch.tensor(orig_cu, dtype=torch.int32)
    padded = torch.tensor(padded_cu, dtype=torch.int32)
    return SimpleNamespace(
        orig_mb={"cu_seqlens": old_cu},
        padded_mb={"cu_seqlens": padded},
        old_cu_seqlens=old_cu,
    )


def _patch_parallel(monkeypatch, *, tp_size: int = 1) -> None:
    monkeypatch.setattr(
        "areal.engine.megatron_utils.packed_context_parallel.mpu.get_context_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        "areal.engine.r3.layout._scatter_to_sequence_parallel_region",
        lambda tensor: tensor[: tensor.shape[0] // tp_size] if tp_size > 1 else tensor,
    )


def test_prepare_native_replay_slabs_packs_alignment_and_padding(monkeypatch):
    _patch_parallel(monkeypatch)
    routed = torch.arange(1, 1 + 2 * 4 * 3 * 2, dtype=torch.int32).reshape(2, 4, 3, 2)
    valid = torch.tensor([True, True])
    mb_input = _mb_input(orig_cu=[0, 3, 5], padded_cu=[0, 4, 8, 12])

    slabs = prepare_native_replay_slabs(
        routed,
        valid,
        mb_input,
        local_moe_indices=[0, 2],
    )

    assert not slabs.skip_replay
    assert len(slabs.slabs) == 2
    assert slabs.slabs[0].shape == (12, 2)
    torch.testing.assert_close(slabs.slabs[0][0], routed[0, 0, 0].long())
    torch.testing.assert_close(slabs.slabs[0][3], routed[0, 2, 0].long())
    torch.testing.assert_close(slabs.slabs[0][6], routed[1, 1, 0].long())
    torch.testing.assert_close(slabs.slabs[0][11], routed[1, 1, 0].long())
    torch.testing.assert_close(slabs.slabs[1][0], routed[0, 0, 2].long())


def test_prepare_native_replay_slabs_invalid_sample_skips(monkeypatch):
    _patch_parallel(monkeypatch)
    routed = torch.ones(2, 4, 1, 2, dtype=torch.int32)
    valid = torch.tensor([True, False])

    slabs = prepare_native_replay_slabs(
        routed,
        valid,
        _mb_input(orig_cu=[0, 3, 5], padded_cu=[0, 4, 8]),
        local_moe_indices=[0],
    )

    assert slabs.skip_replay
    assert slabs.reason == "invalid_sample"


def test_prepare_native_replay_slabs_trusts_preprocessed_routing_valid(
    monkeypatch,
):
    _patch_parallel(monkeypatch)
    routed = torch.ones(1, 4, 1, 2, dtype=torch.int32)
    routed[0, 1] = 0
    valid = torch.tensor([True])

    slabs = prepare_native_replay_slabs(
        routed,
        valid,
        _mb_input(orig_cu=[0, 4], padded_cu=[0, 4]),
        local_moe_indices=[0],
    )

    assert not slabs.skip_replay
    torch.testing.assert_close(slabs.slabs[0], routed[0, :, 0].long())


def test_prepare_native_replay_slabs_tp_scatter_preserves_layer_slices(monkeypatch):
    _patch_parallel(monkeypatch, tp_size=2)
    routed = torch.arange(1, 1 + 1 * 4 * 2 * 2, dtype=torch.int32).reshape(1, 4, 2, 2)

    slabs = prepare_native_replay_slabs(
        routed,
        torch.tensor([True]),
        _mb_input(orig_cu=[0, 4], padded_cu=[0, 4]),
        local_moe_indices=[1],
    )

    assert not slabs.skip_replay
    assert slabs.slabs[0].shape == (2, 2)
    torch.testing.assert_close(slabs.slabs[0], routed[0, :2, 1].long())


def test_prepare_native_replay_slabs_skips_tp_scatter_without_sequence_parallel(
    monkeypatch,
):
    _patch_parallel(monkeypatch, tp_size=2)
    routed = torch.arange(1, 1 + 1 * 4 * 2 * 2, dtype=torch.int32).reshape(1, 4, 2, 2)

    slabs = prepare_native_replay_slabs(
        routed,
        torch.tensor([True]),
        _mb_input(orig_cu=[0, 4], padded_cu=[0, 4]),
        local_moe_indices=[1],
        sequence_parallel=False,
    )

    assert not slabs.skip_replay
    assert slabs.slabs[0].shape == (4, 2)
    torch.testing.assert_close(slabs.slabs[0], routed[0, :, 1].long())


def test_prepare_native_replay_slabs_rejects_bad_local_moe_index(monkeypatch):
    _patch_parallel(monkeypatch)
    routed = torch.ones(1, 4, 2, 2, dtype=torch.int32)

    with pytest.raises(R3Error, match="out of routed_experts range"):
        prepare_native_replay_slabs(
            routed,
            torch.tensor([True]),
            _mb_input(orig_cu=[0, 4], padded_cu=[0, 4]),
            local_moe_indices=[2],
        )


@pytest.mark.parametrize("local_moe_indices", [[], ()])
def test_prepare_native_replay_slabs_no_local_routers_noops(local_moe_indices):
    slabs = prepare_native_replay_slabs(
        torch.ones(1, 2, 1, 2, dtype=torch.int32),
        torch.tensor([True]),
        _mb_input(orig_cu=[0, 2], padded_cu=[0, 2]),
        local_moe_indices=local_moe_indices,
    )

    assert not slabs.skip_replay
    assert slabs.slabs == []
