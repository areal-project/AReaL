"""Tests for the vectorized CP zigzag undo/redo index construction.

The vectorized builders in lightning_attention/kda_attention replaced a
per-sequence Python loop that called ``.item()`` on GPU tensors in the layer
forward hot path. These tests pin the vectorized output to the original loop
reference and cover the per-microbatch cache."""

import pytest
import torch

from areal.models.mcore.kda_attention import (
    _build_zigzag_undo_indices as kda_undo,
)
from areal.models.mcore.lightning_attention import (
    _build_zigzag_redo_indices,
    _build_zigzag_undo_indices,
    _get_zigzag_undo_redo_indices,
)


def _reference_undo_indices(
    total_len: int,
    cp_size: int,
    cu_seqlens: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor:
    """Original loop-based implementation (pre-vectorization)."""
    indices = torch.empty(total_len, dtype=torch.long, device=device)
    if cp_size <= 1:
        return torch.arange(total_len, dtype=torch.long, device=device)
    t_per_cp = total_len // cp_size

    if cu_seqlens is None:
        seq_bounds = [(0, total_len)]
    else:
        seq_bounds = [
            (int(cu_seqlens[i].item()), int(cu_seqlens[i + 1].item()))
            for i in range(cu_seqlens.shape[0] - 1)
        ]

    for cu_start, cu_end in seq_bounds:
        seq_len = cu_end - cu_start
        chunk = seq_len // (2 * cp_size)
        cu_s = cu_start // cp_size
        for j in range(cp_size):
            block_start = j * t_per_cp + cu_s
            base = torch.arange(chunk, device=device)
            dst_front = cu_start + j * chunk
            indices[dst_front : dst_front + chunk] = block_start + base
            dst_back = cu_start + seq_len - (j + 1) * chunk
            indices[dst_back : dst_back + chunk] = block_start + chunk + base
    return indices


CASES = [
    (2, [8, 12]),
    (2, [4]),
    (2, [16, 4, 8, 24]),
    (4, [16, 32]),
    (4, [8, 8, 8]),
    (8, [32, 64, 16]),
]


@pytest.mark.parametrize("cp_size,seq_lens", CASES)
@pytest.mark.parametrize("builder", [_build_zigzag_undo_indices, kda_undo])
def test_vectorized_matches_loop_reference(cp_size, seq_lens, builder):
    cu = torch.tensor([0] + list(torch.tensor(seq_lens).cumsum(0)))
    total = int(cu[-1])
    ref = _reference_undo_indices(total, cp_size, cu, torch.device("cpu"))
    out = builder(total, cp_size, cu, torch.device("cpu"))
    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("cp_size", [2, 4])
@pytest.mark.parametrize("builder", [_build_zigzag_undo_indices, kda_undo])
def test_no_cu_seqlens_single_sequence(cp_size, builder):
    total = 16 * cp_size
    ref = _reference_undo_indices(total, cp_size, None, torch.device("cpu"))
    out = builder(total, cp_size, None, torch.device("cpu"))
    torch.testing.assert_close(out, ref)


def test_undo_is_a_permutation_and_redo_inverts():
    cu = torch.tensor([0, 16, 40])
    undo = _build_zigzag_undo_indices(40, 4, cu, torch.device("cpu"))
    assert torch.equal(torch.sort(undo).values, torch.arange(40))
    redo = _build_zigzag_redo_indices(undo)
    x = torch.randn(40)
    torch.testing.assert_close(x[undo][redo], x)


def test_roundtrip_restores_zigzag_layout():
    """undo applied to the concatenated per-rank zigzag shards must yield
    the canonical sequence."""
    cp_size = 2
    cu = torch.tensor([0, 8, 20])
    total = 20
    canonical = torch.arange(total)
    # Build each rank's zigzag shard the way AReaL packs them.
    shards = []
    for rank in range(cp_size):
        rows = []
        for i in range(len(cu) - 1):
            seq = canonical[cu[i] : cu[i + 1]]
            half = len(seq) // (2 * cp_size)
            rows.append(seq[half * rank : half * (rank + 1)])
            rows.append(seq[len(seq) - half * (rank + 1) : len(seq) - half * rank])
        shards.append(torch.cat(rows))
    zigzag = torch.cat(shards)
    undo = _build_zigzag_undo_indices(total, cp_size, cu, torch.device("cpu"))
    torch.testing.assert_close(zigzag[undo], canonical)


def test_indivisible_sequence_raises():
    cu = torch.tensor([0, 6])  # 6 not divisible by 2*cp for cp=2
    with pytest.raises(ValueError, match="divisible"):
        _build_zigzag_undo_indices(6, 2, cu, torch.device("cpu"))


class TestUndoRedoCache:
    def test_cache_hits_for_same_microbatch(self):
        cu = torch.tensor([0, 8, 20])
        a = _get_zigzag_undo_redo_indices(20, 2, cu, torch.device("cpu"))
        b = _get_zigzag_undo_redo_indices(20, 2, cu, torch.device("cpu"))
        assert a[0] is b[0] and a[1] is b[1]

    def test_cache_scoped_to_cu_tensor_object(self):
        cu1 = torch.tensor([0, 8, 20])
        cu2 = torch.tensor([0, 8, 20])
        a = _get_zigzag_undo_redo_indices(20, 2, cu1, torch.device("cpu"))
        b = _get_zigzag_undo_redo_indices(20, 2, cu2, torch.device("cpu"))
        assert a[0] is not b[0]
        torch.testing.assert_close(a[0], b[0])

    def test_no_cu_seqlens_not_cached(self):
        a = _get_zigzag_undo_redo_indices(16, 2, None, torch.device("cpu"))
        b = _get_zigzag_undo_redo_indices(16, 2, None, torch.device("cpu"))
        assert a[0] is not b[0]
        torch.testing.assert_close(a[0], b[0])
