import torch

from areal.engine.megatron_engine_r3_patch import _align_routed_experts_to_mask


class TestAlignRoutedExpertsToMask:
    """Tests for _align_routed_experts_to_mask: cu_seqlens-based alignment."""

    def test_same_shape_no_change(self):
        routed_experts = torch.randint(1, 64, (4, 10, 2, 3))
        cu_seqlens = torch.tensor([0, 10, 20, 30, 40], dtype=torch.long)
        max_seqlen = 10
        result = _align_routed_experts_to_mask(routed_experts, cu_seqlens, max_seqlen)
        torch.testing.assert_close(result, routed_experts)

    def test_seq_dim_shorter_right_pad(self):
        routed_experts = torch.randint(1, 64, (2, 5, 2, 3))
        cu_seqlens = torch.tensor([0, 5, 10], dtype=torch.long)
        max_seqlen = 8
        result = _align_routed_experts_to_mask(routed_experts, cu_seqlens, max_seqlen)
        assert result.shape == (2, 8, 2, 3)
        torch.testing.assert_close(result[:, :5, :, :], routed_experts)
        assert (result[:, 5:, :, :] == 0).all()

    def test_varying_seq_lens(self):
        bs, re_seqlen, num_layers, topk = 2, 10, 2, 3
        routed_experts = torch.randint(1, 64, (bs, re_seqlen, num_layers, topk))
        cu_seqlens = torch.tensor([0, 7, 12], dtype=torch.long)
        max_seqlen = 8
        result = _align_routed_experts_to_mask(routed_experts, cu_seqlens, max_seqlen)
        assert result.shape == (2, 8, 2, 3)
        torch.testing.assert_close(result[0, :7, :, :], routed_experts[0, :7, :, :])
        torch.testing.assert_close(result[1, :5, :, :], routed_experts[1, :5, :, :])
        assert (result[0, 7:, :, :] == 0).all()
        assert (result[1, 5:, :, :] == 0).all()

    def test_batch_dim_smaller_padded(self):
        routed_experts = torch.randint(1, 64, (3, 8, 2, 3))
        cu_seqlens = torch.tensor([0, 8, 16, 24, 32], dtype=torch.long)
        max_seqlen = 8
        result = _align_routed_experts_to_mask(routed_experts, cu_seqlens, max_seqlen)
        assert result.shape == (4, 8, 2, 3)
        torch.testing.assert_close(result[:3, :, :, :], routed_experts)
        assert (result[3:, :, :, :] == 0).all()

    def test_batch_dim_larger_truncated(self):
        routed_experts = torch.randint(1, 64, (5, 8, 2, 3))
        cu_seqlens = torch.tensor([0, 8, 16], dtype=torch.long)
        max_seqlen = 8
        result = _align_routed_experts_to_mask(routed_experts, cu_seqlens, max_seqlen)
        assert result.shape == (2, 8, 2, 3)
        torch.testing.assert_close(result, routed_experts[:2])

    def test_both_batch_and_seq_mismatch(self):
        routed_experts = torch.randint(1, 64, (2, 10, 2, 3))
        cu_seqlens = torch.tensor([0, 6, 12, 18, 24], dtype=torch.long)
        max_seqlen = 6
        result = _align_routed_experts_to_mask(routed_experts, cu_seqlens, max_seqlen)
        assert result.shape == (4, 6, 2, 3)
        torch.testing.assert_close(result[:2, :, :, :], routed_experts[:, :6, :, :])
        assert (result[2:, :, :, :] == 0).all()

    def test_zero_len_sequences(self):
        routed_experts = torch.randint(1, 64, (2, 8, 2, 3))
        cu_seqlens = torch.tensor([0, 0, 8], dtype=torch.long)
        max_seqlen = 8
        result = _align_routed_experts_to_mask(routed_experts, cu_seqlens, max_seqlen)
        assert result.shape == (2, 8, 2, 3)
        assert (result[0] == 0).all()
        torch.testing.assert_close(result[1], routed_experts[1])

    def test_max_seqlen_larger_than_re_seqlen(self):
        routed_experts = torch.randint(1, 64, (2, 5, 2, 3))
        cu_seqlens = torch.tensor([0, 5, 10], dtype=torch.long)
        max_seqlen = 10
        result = _align_routed_experts_to_mask(routed_experts, cu_seqlens, max_seqlen)
        assert result.shape == (2, 10, 2, 3)
        torch.testing.assert_close(result[:, :5, :, :], routed_experts)
        assert (result[:, 5:, :, :] == 0).all()
