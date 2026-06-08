# SPDX-License-Identifier: Apache-2.0
"""Unit tests for reassemble_cp_packed_logprobs.

Validates that the reassembly operation is the correct inverse of
split_packed_seqs_for_context_parallel, and that gradients flow correctly
through the differentiable all-gather.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

try:
    import megatron.core  # noqa: F401

    HAS_MEGATRON = True
except ImportError:
    HAS_MEGATRON = False

requires_megatron = pytest.mark.skipif(
    not HAS_MEGATRON, reason="megatron-core not installed"
)


def _make_cu_seqlens(seq_lengths: list[int], device="cpu") -> torch.Tensor:
    """Build cu_seqlens from a list of sequence lengths."""
    cu = torch.zeros(len(seq_lengths) + 1, dtype=torch.long, device=device)
    for i, _len in enumerate(seq_lengths):
        cu[i + 1] = cu[i] + _len
    return cu


def _split_for_cp_rank(
    tensor: torch.Tensor, cu_seqlens: torch.Tensor, cp_rank: int, cp_size: int
) -> torch.Tensor:
    """Pure-Python reference implementation of split_packed_seqs_for_context_parallel."""
    input_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    batch_size = input_lens.shape[0]
    output_len = input_lens.sum().item() // cp_size

    splitted = torch.zeros(output_len, dtype=tensor.dtype, device=tensor.device)
    for i in range(batch_size):
        seqlen = input_lens[i] // cp_size
        half_seqlen = seqlen // 2
        start_idx = cu_seqlens[i] // cp_size

        d = tensor[cu_seqlens[i] : cu_seqlens[i + 1]]
        splitted[start_idx : start_idx + half_seqlen] = d[
            half_seqlen * cp_rank : half_seqlen * (cp_rank + 1)
        ]

        remain_start = input_lens[i] - half_seqlen * (cp_rank + 1)
        remain_end = input_lens[i] - half_seqlen * cp_rank
        remain_end = min(remain_end, d.shape[0])
        remain_len = remain_end - remain_start
        splitted[start_idx + half_seqlen : start_idx + half_seqlen + remain_len] = d[
            remain_start:remain_end
        ]
    return splitted


def _reassemble_reference(
    local_tensors: list[torch.Tensor], cu_seqlens: torch.Tensor, cp_size: int
) -> torch.Tensor:
    """Pure-Python reference reassembly (inverse of split) using advanced indexing."""
    input_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    batch_size = input_lens.shape[0]
    output_len = int(cu_seqlens[-1].item())
    local_len = output_len // cp_size

    # Build index mapping: indices[dst] = src in concatenated flat tensor
    indices = torch.empty(output_len, dtype=torch.long)

    for i in range(batch_size):
        seq_len = int(input_lens[i].item())
        chunk_size = seq_len // cp_size
        half_chunk = chunk_size // 2
        local_start = int(cu_seqlens[i].item()) // cp_size
        full_start = int(cu_seqlens[i].item())

        for j in range(cp_size):
            src_offset = j * local_len + local_start
            for k in range(half_chunk):
                indices[full_start + j * half_chunk + k] = src_offset + k
            for k in range(half_chunk):
                indices[full_start + seq_len - (j + 1) * half_chunk + k] = (
                    src_offset + half_chunk + k
                )

    gathered_flat = torch.cat(local_tensors, dim=0)
    return gathered_flat[indices]


class TestReassembleGradientFlow:
    """Test gradient correctness of the index-based reassembly (no megatron needed)."""

    @pytest.mark.parametrize("cp_size", [2, 4])
    def test_gradient_through_advanced_indexing(self, cp_size):
        """Verify grad flows back through cat + advanced indexing."""
        seq_lengths = [_len * 2 * cp_size for _len in [4, 6]]
        cu_seqlens = _make_cu_seqlens(seq_lengths)
        total_len = cu_seqlens[-1].item()

        original = torch.randn(total_len)
        splits = [
            _split_for_cp_rank(original, cu_seqlens, r, cp_size) for r in range(cp_size)
        ]

        # Simulate what reassemble_cp_packed_logprobs does internally:
        # dist_F.all_gather returns list of tensors, then cat + index
        splits_with_grad = [s.clone().requires_grad_(True) for s in splits]

        # Use the reference index building
        reassembled_with_grad = _reassemble_reference(
            splits_with_grad, cu_seqlens, cp_size
        )

        # Verify values match
        torch.testing.assert_close(reassembled_with_grad, original, rtol=0, atol=0)

        # Compute loss and backward
        loss = reassembled_with_grad.sum()
        loss.backward()

        # Each position in the original maps to exactly one source position,
        # so gradient for each source element should be 1.0
        for r in range(cp_size):
            assert splits_with_grad[r].grad is not None
            expected = torch.ones_like(splits_with_grad[r])
            torch.testing.assert_close(
                splits_with_grad[r].grad, expected, rtol=0, atol=0
            )

    @pytest.mark.parametrize("cp_size", [2, 4])
    def test_gradient_with_weighted_loss(self, cp_size):
        """Verify correct gradient with a non-uniform loss (weighted sum)."""
        seq_lengths = [_len * 2 * cp_size for _len in [3, 5]]
        cu_seqlens = _make_cu_seqlens(seq_lengths)
        total_len = cu_seqlens[-1].item()

        original = torch.randn(total_len)
        weights = torch.randn(total_len)
        splits = [
            _split_for_cp_rank(original, cu_seqlens, r, cp_size) for r in range(cp_size)
        ]

        splits_with_grad = [s.clone().requires_grad_(True) for s in splits]
        reassembled = _reassemble_reference(splits_with_grad, cu_seqlens, cp_size)

        loss = (reassembled * weights).sum()
        loss.backward()

        # The gradient at each source position should equal the weight at the
        # corresponding destination position
        all_grads = torch.cat([s.grad for s in splits_with_grad], dim=0)
        # Reconstruct what the gradient should be: weights permuted back to source order
        weight_splits = [
            _split_for_cp_rank(weights, cu_seqlens, r, cp_size) for r in range(cp_size)
        ]
        expected_grads = torch.cat(weight_splits, dim=0)

        torch.testing.assert_close(all_grads, expected_grads, rtol=1e-5, atol=1e-5)


class TestReassembleGradientCorrectness:
    """Verify gradient correctness of the cat + advanced-indexing reassembly."""

    @pytest.mark.parametrize("cp_size", [2, 4])
    def test_uniform_loss_gives_unit_gradient(self, cp_size):
        """With loss=sum(reassembled), each source position gets grad=1 (bijection)."""
        seq_lengths = [_len * 2 * cp_size for _len in [4, 3]]
        cu_seqlens = _make_cu_seqlens(seq_lengths)
        total_len = cu_seqlens[-1].item()

        original = torch.randn(total_len)
        splits = [
            _split_for_cp_rank(original, cu_seqlens, r, cp_size) for r in range(cp_size)
        ]

        splits_with_grad = [s.clone().requires_grad_(True) for s in splits]
        reassembled = _reassemble_reference(splits_with_grad, cu_seqlens, cp_size)

        loss = reassembled.sum()
        loss.backward()

        for r in range(cp_size):
            assert splits_with_grad[r].grad is not None
            torch.testing.assert_close(
                splits_with_grad[r].grad,
                torch.ones_like(splits_with_grad[r]),
                rtol=0,
                atol=0,
            )

    @pytest.mark.parametrize("cp_size", [2, 4])
    def test_no_grad_duplication_or_loss(self, cp_size):
        """Each source element contributes to exactly one output — no duplication."""
        seq_lengths = [_len * 2 * cp_size for _len in [5, 3]]
        cu_seqlens = _make_cu_seqlens(seq_lengths)
        total_len = cu_seqlens[-1].item()

        original = torch.randn(total_len)
        splits = [
            _split_for_cp_rank(original, cu_seqlens, r, cp_size) for r in range(cp_size)
        ]

        splits_with_grad = [s.clone().requires_grad_(True) for s in splits]
        reassembled = _reassemble_reference(splits_with_grad, cu_seqlens, cp_size)

        # Use squared loss so gradient magnitude varies per element
        loss = (reassembled**2).sum()
        loss.backward()

        # Grad for source[i] should be 2 * value_at_destination
        # Since it's a bijection, value_at_destination == source[i] (original value)
        all_grads = torch.cat([s.grad for s in splits_with_grad])
        all_values = torch.cat([s for s in splits_with_grad])
        expected = 2 * all_values.detach()
        torch.testing.assert_close(all_grads, expected, rtol=1e-5, atol=1e-5)

    @requires_megatron
    def test_advanced_indexing_is_bijective_permutation(self):
        """Confirm index tensor is a valid permutation (no duplicates)."""
        cp_size = 2
        seq_lengths = [_len * 2 * cp_size for _len in [4, 6, 3]]
        cu_seqlens = _make_cu_seqlens(seq_lengths)
        total_len = cu_seqlens[-1].item()
        local_len = total_len // cp_size

        from areal.engine.megatron_utils.packed_context_parallel import (
            _build_cp_reassemble_indices,
        )

        indices = _build_cp_reassemble_indices(cu_seqlens, cp_size)

        assert indices.shape == (total_len,)
        # All indices should be in [0, cp_size * local_len)
        assert (indices >= 0).all()
        assert (indices < cp_size * local_len).all()
        # No duplicates (bijection)
        assert indices.unique().shape[0] == total_len


class TestSplitReassembleRoundtrip:
    """Test that split → reassemble is an identity transformation."""

    @pytest.mark.parametrize("cp_size", [2, 4])
    @pytest.mark.parametrize(
        "seq_lengths",
        [
            [16],
            [32, 16],
            [64, 32, 16],
            [8, 8, 8, 8],
        ],
    )
    def test_roundtrip_single_sequence(self, cp_size, seq_lengths):
        """Split then reassemble should recover the original tensor."""
        # Ensure all seqlens are divisible by 2*cp_size
        aligned_lengths = [_len * 2 * cp_size for _len in seq_lengths]
        cu_seqlens = _make_cu_seqlens(aligned_lengths)
        total_len = cu_seqlens[-1].item()

        original = torch.randn(total_len)

        # Split for each CP rank
        splits = []
        for rank in range(cp_size):
            splits.append(_split_for_cp_rank(original, cu_seqlens, rank, cp_size))

        # Reassemble
        recovered = _reassemble_reference(splits, cu_seqlens, cp_size)

        torch.testing.assert_close(recovered, original, rtol=0, atol=0)

    @pytest.mark.parametrize("cp_size", [2, 4, 8])
    def test_roundtrip_large_batch(self, cp_size):
        """Test with larger batch to stress edge cases."""
        torch.manual_seed(42)
        batch_size = 16
        min_chunks = 1
        max_chunks = 8
        chunk_unit = 2 * cp_size

        lengths = [
            int(torch.randint(min_chunks, max_chunks + 1, (1,)).item()) * chunk_unit
            for _ in range(batch_size)
        ]
        cu_seqlens = _make_cu_seqlens(lengths)
        total_len = cu_seqlens[-1].item()
        original = torch.randn(total_len)

        splits = [
            _split_for_cp_rank(original, cu_seqlens, r, cp_size) for r in range(cp_size)
        ]
        recovered = _reassemble_reference(splits, cu_seqlens, cp_size)

        torch.testing.assert_close(recovered, original, rtol=0, atol=0)


@requires_megatron
class TestReassembleCpPackedLogprobs:
    """Test the actual distributed reassemble function with mocked dist."""

    @pytest.mark.parametrize("cp_size", [2, 4])
    def test_reassemble_matches_reference(self, cp_size):
        """reassemble_cp_packed_logprobs produces same result as reference."""
        seq_lengths = [_len * 2 * cp_size for _len in [8, 4, 6]]
        cu_seqlens = _make_cu_seqlens(seq_lengths)
        total_len = cu_seqlens[-1].item()

        original = torch.randn(total_len)

        splits = [
            _split_for_cp_rank(original, cu_seqlens, r, cp_size) for r in range(cp_size)
        ]

        for test_rank in range(cp_size):
            local_tensor = splits[test_rank].requires_grad_(True)

            with (
                patch(
                    "areal.engine.megatron_utils.packed_context_parallel.mpu"
                ) as mock_mpu,
                patch(
                    "areal.engine.megatron_utils.packed_context_parallel.dist_F"
                ) as mock_dist_F,
            ):
                mock_mpu.get_context_parallel_world_size.return_value = cp_size
                mock_group = MagicMock()
                mock_mpu.get_context_parallel_group.return_value = mock_group

                # Mock dist_F.all_gather to return all splits
                # (simulates the distributed all-gather)
                def fake_all_gather(tensor, group=None):
                    return [
                        s.detach().requires_grad_(r == test_rank)
                        for r, s in enumerate(splits)
                    ]

                mock_dist_F.all_gather.side_effect = fake_all_gather

                from areal.engine.megatron_utils.packed_context_parallel import (
                    reassemble_cp_packed_logprobs,
                )

                result = reassemble_cp_packed_logprobs(local_tensor, cu_seqlens)

            torch.testing.assert_close(result, original, rtol=1e-5, atol=1e-5)

    def test_noop_when_cp_size_1(self):
        """When cp_size=1, returns input unchanged."""
        tensor = torch.randn(32, requires_grad=True)
        cu_seqlens = _make_cu_seqlens([32])

        with patch(
            "areal.engine.megatron_utils.packed_context_parallel.mpu"
        ) as mock_mpu:
            mock_mpu.get_context_parallel_world_size.return_value = 1

            from areal.engine.megatron_utils.packed_context_parallel import (
                reassemble_cp_packed_logprobs,
            )

            result = reassemble_cp_packed_logprobs(tensor, cu_seqlens)

        assert result is tensor

    @pytest.mark.parametrize("cp_size", [2, 4])
    def test_gradient_flows_through_reassembly(self, cp_size):
        """Verify that gradients flow back through the all-gather and reassembly."""
        seq_lengths = [_len * 2 * cp_size for _len in [4, 6]]
        cu_seqlens = _make_cu_seqlens(seq_lengths)
        total_len = cu_seqlens[-1].item()

        original = torch.randn(total_len)
        splits = [
            _split_for_cp_rank(original, cu_seqlens, r, cp_size) for r in range(cp_size)
        ]

        test_rank = 0
        local_tensor = splits[test_rank].clone().requires_grad_(True)

        with (
            patch(
                "areal.engine.megatron_utils.packed_context_parallel.mpu"
            ) as mock_mpu,
            patch(
                "areal.engine.megatron_utils.packed_context_parallel.dist_F"
            ) as mock_dist_F,
        ):
            mock_mpu.get_context_parallel_world_size.return_value = cp_size
            mock_mpu.get_context_parallel_group.return_value = MagicMock()

            def fake_all_gather(tensor, group=None):
                result = []
                for r in range(cp_size):
                    if r == test_rank:
                        result.append(tensor)
                    else:
                        result.append(splits[r].detach())
                return result

            mock_dist_F.all_gather.side_effect = fake_all_gather

            from areal.engine.megatron_utils.packed_context_parallel import (
                reassemble_cp_packed_logprobs,
            )

            result = reassemble_cp_packed_logprobs(local_tensor, cu_seqlens)

        # Compute a loss and backward
        loss = result.sum()
        loss.backward()

        assert local_tensor.grad is not None
        # Gradient should be 1.0 for all positions that belong to test_rank
        # (since loss = sum(result) and each local position contributes once)
        expected_grad = torch.ones_like(local_tensor)
        torch.testing.assert_close(local_tensor.grad, expected_grad, rtol=0, atol=0)


@requires_megatron
class TestSplitReassembleConsistencyWithMegatron:
    """Test that our reference split matches the actual Megatron utility."""

    @pytest.mark.parametrize("cp_size", [2, 4])
    def test_split_matches_megatron_impl(self, cp_size):
        """Verify our reference split matches split_packed_seqs_for_context_parallel."""
        seq_lengths = [_len * 2 * cp_size for _len in [5, 3, 7]]
        cu_seqlens = _make_cu_seqlens(seq_lengths)
        total_len = cu_seqlens[-1].item()
        original = torch.randn(total_len)

        for rank in range(cp_size):
            reference = _split_for_cp_rank(original, cu_seqlens, rank, cp_size)

            with patch(
                "areal.engine.megatron_utils.packed_context_parallel.mpu"
            ) as mock_mpu:
                mock_mpu.get_context_parallel_world_size.return_value = cp_size
                mock_mpu.get_context_parallel_rank.return_value = rank

                from areal.engine.megatron_utils.packed_context_parallel import (
                    split_packed_seqs_for_context_parallel,
                )

                actual = split_packed_seqs_for_context_parallel(original, cu_seqlens)

            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
