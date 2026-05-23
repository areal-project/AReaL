import pytest
import torch

from areal.api.cli_args import RejectionSamplingConfig
from areal.utils.functional import (
    ppo_actor_loss_fn,
    sapo_loss_fn,
)


class TestPPOActorLossFnSequenceLevel:
    """Test cases for ppo_actor_loss_fn with importance_sampling_level='sequence'."""

    @pytest.fixture
    def basic_2d_data(self):
        """Basic 2D tensor test data for sequence-level importance sampling."""
        batch_size = 4
        seq_len = 8

        return {
            "logprobs": torch.randn(batch_size, seq_len),
            "proximal_logprobs": torch.randn(batch_size, seq_len),
            "old_logprobs": torch.randn(batch_size, seq_len),
            "advantages": torch.randn(batch_size, seq_len),
            "loss_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
            "eps_clip": 0.2,
        }

    @pytest.fixture
    def basic_1d_data(self):
        """Basic 1D tensor test data for sequence-level importance sampling (packed)."""
        # 3 sequences with lengths [8, 12, 12]
        total_tokens = 32
        cu_seqlens = torch.tensor([0, 8, 20, 32], dtype=torch.int32)

        return {
            "logprobs": torch.randn(total_tokens),
            "proximal_logprobs": torch.randn(total_tokens),
            "old_logprobs": torch.randn(total_tokens),
            "advantages": torch.randn(total_tokens),
            "loss_mask": torch.ones(total_tokens, dtype=torch.bool),
            "eps_clip": 0.2,
            "cu_seqlens": cu_seqlens,
        }

    def test_sequence_level_2d_shape(self, basic_2d_data):
        """Test that sequence-level loss returns correct shape for 2D tensors."""
        loss, stat = ppo_actor_loss_fn(
            logprobs=basic_2d_data["logprobs"],
            proximal_logprobs=basic_2d_data["proximal_logprobs"],
            old_logprobs=basic_2d_data["old_logprobs"],
            advantages=basic_2d_data["advantages"],
            eps_clip=basic_2d_data["eps_clip"],
            loss_mask=basic_2d_data["loss_mask"],
            importance_sampling_level="sequence",
        )

        # Loss should be a scalar
        assert loss.ndim == 0
        assert loss.dtype == torch.float32

        # Stats should have correct shapes
        assert stat["importance_weight"].shape == basic_2d_data["logprobs"].shape
        assert stat["approx_kl"].shape == basic_2d_data["logprobs"].shape
        assert stat["clip_mask"].shape == basic_2d_data["logprobs"].shape

    def test_sequence_level_1d_shape(self, basic_1d_data):
        """Test that sequence-level loss returns correct shape for 1D tensors."""
        loss, stat = ppo_actor_loss_fn(
            logprobs=basic_1d_data["logprobs"],
            proximal_logprobs=basic_1d_data["proximal_logprobs"],
            old_logprobs=basic_1d_data["old_logprobs"],
            advantages=basic_1d_data["advantages"],
            eps_clip=basic_1d_data["eps_clip"],
            loss_mask=basic_1d_data["loss_mask"],
            importance_sampling_level="sequence",
            cu_seqlens=basic_1d_data["cu_seqlens"],
        )

        # Loss should be a scalar
        assert loss.ndim == 0
        assert loss.dtype == torch.float32

        # Stats should have correct shapes
        assert stat["importance_weight"].shape == basic_1d_data["logprobs"].shape
        assert stat["approx_kl"].shape == basic_1d_data["logprobs"].shape
        assert stat["clip_mask"].shape == basic_1d_data["logprobs"].shape

    def test_sequence_level_geometric_mean_2d(self):
        """Test that geometric mean is computed correctly for 2D tensors."""
        batch_size = 2
        seq_len = 4

        # Create data where we can manually verify geometric mean
        logprobs = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0],
            ]
        )
        proximal_logprobs = torch.tensor(
            [
                [0.5, 0.5, 0.5, 0.5],
                [1.0, 1.0, 1.0, 1.0],
            ]
        )
        old_logprobs = torch.zeros(batch_size, seq_len)
        advantages = torch.ones(batch_size, seq_len)
        loss_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        loss, stat = ppo_actor_loss_fn(
            logprobs=logprobs,
            proximal_logprobs=proximal_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            eps_clip=0.2,
            loss_mask=loss_mask,
            importance_sampling_level="sequence",
        )

        # For first sequence: log_ratio = 1.0 - 0.5 = 0.5
        # Geometric mean = exp(0.5) ≈ 1.6487
        # For second sequence: log_ratio = 2.0 - 1.0 = 1.0
        # Geometric mean = exp(1.0) ≈ 2.7183

        expected_ratio_seq1 = torch.exp(torch.tensor(0.5))
        expected_ratio_seq2 = torch.exp(torch.tensor(1.0))

        # All tokens in a sequence should have the same ratio
        assert torch.allclose(
            stat["importance_weight"][0, :],
            expected_ratio_seq1.expand(seq_len),
            atol=1e-5,
        )
        assert torch.allclose(
            stat["importance_weight"][1, :],
            expected_ratio_seq2.expand(seq_len),
            atol=1e-5,
        )

    def test_sequence_level_geometric_mean_1d(self):
        """Test that geometric mean is computed correctly for 1D tensors (packed sequences)."""
        # Two sequences: [4 tokens, 4 tokens]
        total_tokens = 8
        cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32)

        # Sequence 1: log_ratio = 0.5, Sequence 2: log_ratio = 1.0
        logprobs = torch.tensor([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0])
        proximal_logprobs = torch.tensor([0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0])
        old_logprobs = torch.zeros(total_tokens)
        advantages = torch.ones(total_tokens)
        loss_mask = torch.ones(total_tokens, dtype=torch.bool)

        loss, stat = ppo_actor_loss_fn(
            logprobs=logprobs,
            proximal_logprobs=proximal_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            eps_clip=0.2,
            loss_mask=loss_mask,
            importance_sampling_level="sequence",
            cu_seqlens=cu_seqlens,
        )

        # Sequence 1: log_ratio = 1.0 - 0.5 = 0.5, Geometric mean = exp(0.5)
        expected_ratio_seq1 = torch.exp(torch.tensor(0.5))
        # Sequence 2: log_ratio = 2.0 - 1.0 = 1.0, Geometric mean = exp(1.0)
        expected_ratio_seq2 = torch.exp(torch.tensor(1.0))

        # All tokens in sequence 1 should have the same ratio
        assert torch.allclose(
            stat["importance_weight"][0:4], expected_ratio_seq1.expand(4), atol=1e-5
        )
        # All tokens in sequence 2 should have the same ratio
        assert torch.allclose(
            stat["importance_weight"][4:8], expected_ratio_seq2.expand(4), atol=1e-5
        )

    def test_sequence_level_advantage_averaging_2d(self):
        """Test that advantages are averaged per sequence for 2D tensors."""
        batch_size = 2
        seq_len = 4

        # Create advantages that vary per token
        advantages = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],  # sum = 10.0, mean = 2.5
                [0.5, 0.5, 0.5, 0.5],  # sum = 2.0, mean = 0.5
            ]
        )

        logprobs = torch.ones(batch_size, seq_len)
        proximal_logprobs = torch.ones(batch_size, seq_len)
        old_logprobs = torch.ones(batch_size, seq_len)
        loss_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        # We'll verify this by checking that the loss is computed correctly
        # The function should average advantages across the sequence dimension
        loss, stat = ppo_actor_loss_fn(
            logprobs=logprobs,
            proximal_logprobs=proximal_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            eps_clip=0.2,
            loss_mask=loss_mask,
            importance_sampling_level="sequence",
        )

        # The function should work without errors
        assert loss.ndim == 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

        # With ratio=1, the loss is the negative mean of advantages.
        # Seq1: mean_adv=2.5, 4 tokens → contributes -2.5 * 4 = -10.0
        # Seq2: mean_adv=0.5, 4 tokens → contributes -0.5 * 4 = -2.0
        # Total: (-10.0 + -2.0) / 8 = -1.5
        assert torch.allclose(loss, torch.tensor(-1.5))

    def test_sequence_level_with_mask_2d(self):
        """Test sequence-level with partial masking for 2D tensors."""
        batch_size = 2
        seq_len = 8

        # Create mask where only some tokens are valid
        loss_mask = torch.tensor(
            [
                [True, True, True, True, False, False, False, False],
                [True, True, True, True, True, True, False, False],
            ]
        )

        logprobs = torch.randn(batch_size, seq_len)
        proximal_logprobs = torch.randn(batch_size, seq_len)
        old_logprobs = torch.randn(batch_size, seq_len)
        advantages = torch.randn(batch_size, seq_len)

        loss, stat = ppo_actor_loss_fn(
            logprobs=logprobs,
            proximal_logprobs=proximal_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            eps_clip=0.2,
            loss_mask=loss_mask,
            importance_sampling_level="sequence",
        )

        # Masked positions should have zero importance weight
        assert torch.allclose(
            stat["importance_weight"][0, 4:], torch.zeros(4), atol=1e-6
        )
        assert torch.allclose(
            stat["importance_weight"][1, 6:], torch.zeros(2), atol=1e-6
        )

        # Valid positions in same sequence should have same ratio
        assert torch.allclose(
            stat["importance_weight"][0, 0], stat["importance_weight"][0, 3], atol=1e-5
        )

    def test_sequence_level_2d_advantage_average_ignores_masked_values(self):
        """Masked padding advantages must not change valid-token sequence loss."""
        loss_mask = torch.tensor([[True, True, False, False]])
        logprobs = torch.zeros(1, 4)
        proximal_logprobs = torch.zeros(1, 4)
        old_logprobs = torch.zeros(1, 4)
        clean_advantages = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        contaminated_advantages = torch.tensor([[1.0, 1.0, 10.0, 10.0]])

        clean_loss, clean_stat = ppo_actor_loss_fn(
            logprobs=logprobs,
            proximal_logprobs=proximal_logprobs,
            old_logprobs=old_logprobs,
            advantages=clean_advantages,
            eps_clip=0.2,
            loss_mask=loss_mask,
            importance_sampling_level="sequence",
        )
        contaminated_loss, contaminated_stat = ppo_actor_loss_fn(
            logprobs=logprobs,
            proximal_logprobs=proximal_logprobs,
            old_logprobs=old_logprobs,
            advantages=contaminated_advantages,
            eps_clip=0.2,
            loss_mask=loss_mask,
            importance_sampling_level="sequence",
        )

        assert torch.allclose(contaminated_loss, clean_loss)
        assert torch.allclose(
            contaminated_stat["loss"][loss_mask], clean_stat["loss"][loss_mask]
        )
        assert torch.allclose(
            contaminated_stat["loss"][~loss_mask],
            torch.zeros_like(contaminated_stat["loss"][~loss_mask]),
        )

    def test_sequence_level_with_mask_1d(self):
        """Test sequence-level with partial masking for 1D tensors (packed sequences)."""
        # Two sequences: [4 tokens, 4 tokens]
        total_tokens = 8
        cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32)

        # Sequence 1: fully valid, Sequence 2: half masked
        loss_mask = torch.tensor([True, True, True, True, True, True, False, False])

        logprobs = torch.randn(total_tokens)
        proximal_logprobs = torch.randn(total_tokens)
        old_logprobs = torch.randn(total_tokens)
        advantages = torch.randn(total_tokens)

        loss, stat = ppo_actor_loss_fn(
            logprobs=logprobs,
            proximal_logprobs=proximal_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            eps_clip=0.2,
            loss_mask=loss_mask,
            importance_sampling_level="sequence",
            cu_seqlens=cu_seqlens,
        )

        # Masked positions should have zero importance weight
        assert torch.allclose(stat["importance_weight"][6:8], torch.zeros(2), atol=1e-6)

        # All valid positions in sequence 1 should have same ratio
        seq1_ratios = stat["importance_weight"][:4]
        assert torch.allclose(seq1_ratios, seq1_ratios[0].expand(4), atol=1e-5)

        # All valid positions in sequence 2 should have same ratio
        seq2_valid_ratios = stat["importance_weight"][4:6]
        assert torch.allclose(
            seq2_valid_ratios, seq2_valid_ratios[0].expand(2), atol=1e-5
        )

    def test_sequence_level_batch_size_1_2d(self):
        """Test batch_size=1 edge case for 2D tensors."""
        batch_size = 1
        seq_len = 10

        logprobs = torch.randn(batch_size, seq_len)
        proximal_logprobs = torch.randn(batch_size, seq_len)
        old_logprobs = torch.randn(batch_size, seq_len)
        advantages = torch.randn(batch_size, seq_len)
        loss_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        loss, stat = ppo_actor_loss_fn(
            logprobs=logprobs,
            proximal_logprobs=proximal_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            eps_clip=0.2,
            loss_mask=loss_mask,
            importance_sampling_level="sequence",
        )

        # Should compute without errors
        assert loss.ndim == 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

        # All tokens in the single sequence should have same ratio
        ratios = stat["importance_weight"][0]
        assert torch.allclose(ratios, ratios[0].expand(seq_len), atol=1e-5)

    def test_sequence_level_batch_size_1_1d(self):
        """Test batch_size=1 edge case for 1D tensors (single sequence)."""
        total_tokens = 10
        cu_seqlens = torch.tensor([0, 10], dtype=torch.int32)

        logprobs = torch.randn(total_tokens)
        proximal_logprobs = torch.randn(total_tokens)
        old_logprobs = torch.randn(total_tokens)
        advantages = torch.randn(total_tokens)
        loss_mask = torch.ones(total_tokens, dtype=torch.bool)

        loss, stat = ppo_actor_loss_fn(
            logprobs=logprobs,
            proximal_logprobs=proximal_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            eps_clip=0.2,
            loss_mask=loss_mask,
            importance_sampling_level="sequence",
            cu_seqlens=cu_seqlens,
        )

        # Should compute without errors
        assert loss.ndim == 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

        # All tokens in the single sequence should have the same ratio
        ratios = stat["importance_weight"]
        assert torch.allclose(ratios, ratios[0].expand(total_tokens), atol=1e-5)

    def test_sequence_level_with_clipping_2d(self):
        """Test that clipping works correctly with sequence-level importance sampling."""
        batch_size = 2
        seq_len = 4

        # Create data that will trigger clipping
        logprobs = torch.tensor(
            [
                [2.0, 2.0, 2.0, 2.0],  # High ratio, will clip
                [-2.0, -2.0, -2.0, -2.0],  # Low ratio, will clip
            ]
        )
        proximal_logprobs = torch.zeros(batch_size, seq_len)
        old_logprobs = torch.zeros(batch_size, seq_len)
        advantages = torch.ones(batch_size, seq_len)
        loss_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        eps_clip = 0.2

        loss, stat = ppo_actor_loss_fn(
            logprobs=logprobs,
            proximal_logprobs=proximal_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            eps_clip=eps_clip,
            loss_mask=loss_mask,
            importance_sampling_level="sequence",
        )

        # Check that some clipping occurred
        assert stat["clip_mask"].any()

        # Verify loss is finite
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_sequence_level_with_dual_clip(self):
        """Test sequence-level with dual clip (c_clip parameter)."""
        batch_size = 2
        seq_len = 4

        logprobs = torch.randn(batch_size, seq_len)
        proximal_logprobs = torch.randn(batch_size, seq_len)
        old_logprobs = torch.randn(batch_size, seq_len)
        advantages = torch.randn(batch_size, seq_len)
        loss_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        loss, stat = ppo_actor_loss_fn(
            logprobs=logprobs,
            proximal_logprobs=proximal_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            eps_clip=0.2,
            loss_mask=loss_mask,
            c_clip=3.0,
            importance_sampling_level="sequence",
        )

        # Should have dual_clip_mask in stats
        assert "dual_clip_mask" in stat
        assert stat["dual_clip_mask"].shape == logprobs.shape

        # Verify loss is finite
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_sequence_level_with_rejection_sampling(self):
        """Test sequence-level with rejection sampling (replaces behave_imp_weight_cap)."""
        batch_size = 2
        seq_len = 4

        logprobs = torch.randn(batch_size, seq_len)
        proximal_logprobs = torch.randn(batch_size, seq_len)
        old_logprobs = torch.randn(batch_size, seq_len)
        advantages = torch.randn(batch_size, seq_len)
        loss_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        rs_config = RejectionSamplingConfig(metric="ratio", upper=10.0)
        loss, stat = ppo_actor_loss_fn(
            logprobs=logprobs,
            proximal_logprobs=proximal_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            eps_clip=0.2,
            loss_mask=loss_mask,
            rejection_sampling=rs_config,
            importance_sampling_level="sequence",
        )

        # Should have behavior stats
        assert "behave_imp_weight" in stat
        assert "behave_approx_kl" in stat
        assert "behave_mask" in stat
        assert "filtered_fraction" in stat

        # Verify loss is finite
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_no_rejection_sampling_excludes_behave_stats(self):
        """When rejection_sampling=None, stat should not contain behave keys."""
        batch_size = 2
        seq_len = 4

        logprobs = torch.randn(batch_size, seq_len)
        proximal_logprobs = torch.randn(batch_size, seq_len)
        old_logprobs = torch.randn(batch_size, seq_len)
        advantages = torch.randn(batch_size, seq_len)
        loss_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        loss, stat = ppo_actor_loss_fn(
            logprobs=logprobs,
            proximal_logprobs=proximal_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            eps_clip=0.2,
            loss_mask=loss_mask,
            rejection_sampling=None,
            importance_sampling_level="sequence",
        )

        # Core PPO stats should be present
        assert "importance_weight" in stat
        assert "approx_kl" in stat
        assert "clip_mask" in stat

        # Rejection sampling stats should NOT be present
        assert "behave_imp_weight" not in stat
        assert "behave_approx_kl" not in stat
        assert "behave_mask" not in stat
        assert "filtered_fraction" not in stat

        # Loss should be finite
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_sequence_level_vs_token_level_different(self):
        """Verify that sequence-level and token-level produce different results."""
        batch_size = 2
        seq_len = 4

        # Create non-uniform data within sequences (use smaller values to avoid clipping)
        logprobs = torch.tensor(
            [
                [0.1, 0.2, 0.05, 0.15],  # varying per token
                [0.15, 0.1, 0.2, 0.05],  # different pattern
            ]
        )
        proximal_logprobs = torch.zeros(batch_size, seq_len)
        old_logprobs = torch.zeros(batch_size, seq_len)
        # Non-uniform advantages to make the difference clear
        advantages = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [4.0, 3.0, 2.0, 1.0],
            ]
        )
        loss_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        # Compute sequence-level loss
        loss_seq, stat_seq = ppo_actor_loss_fn(
            logprobs=logprobs.clone(),
            proximal_logprobs=proximal_logprobs.clone(),
            old_logprobs=old_logprobs.clone(),
            advantages=advantages.clone(),
            eps_clip=0.2,
            loss_mask=loss_mask.clone(),
            importance_sampling_level="sequence",
        )

        # Compute token-level loss
        loss_tok, stat_tok = ppo_actor_loss_fn(
            logprobs=logprobs.clone(),
            proximal_logprobs=proximal_logprobs.clone(),
            old_logprobs=old_logprobs.clone(),
            advantages=advantages.clone(),
            eps_clip=0.2,
            loss_mask=loss_mask.clone(),
            importance_sampling_level="token",
        )

        # Results should be different
        assert not torch.allclose(loss_seq, loss_tok, atol=1e-6)
        assert not torch.allclose(
            stat_seq["importance_weight"], stat_tok["importance_weight"], atol=1e-6
        )

    def test_sequence_level_all_masked(self):
        """Test edge case where all tokens are masked."""
        batch_size = 2
        seq_len = 4

        logprobs = torch.randn(batch_size, seq_len)
        proximal_logprobs = torch.randn(batch_size, seq_len)
        old_logprobs = torch.randn(batch_size, seq_len)
        advantages = torch.randn(batch_size, seq_len)
        loss_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)

        loss, stat = ppo_actor_loss_fn(
            logprobs=logprobs,
            proximal_logprobs=proximal_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            eps_clip=0.2,
            loss_mask=loss_mask,
            importance_sampling_level="sequence",
        )

        # Loss should be zero or very small
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)

        # All importance weights should be zero (masked)
        assert torch.allclose(
            stat["importance_weight"],
            torch.zeros_like(stat["importance_weight"]),
            atol=1e-6,
        )

    def test_sequence_level_consistency_across_calls(self):
        """Test that multiple calls with same input produce same output."""
        batch_size = 2
        seq_len = 4

        logprobs = torch.randn(batch_size, seq_len)
        proximal_logprobs = torch.randn(batch_size, seq_len)
        old_logprobs = torch.randn(batch_size, seq_len)
        advantages = torch.randn(batch_size, seq_len)
        loss_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        # First call
        loss1, stat1 = ppo_actor_loss_fn(
            logprobs=logprobs.clone(),
            proximal_logprobs=proximal_logprobs.clone(),
            old_logprobs=old_logprobs.clone(),
            advantages=advantages.clone(),
            eps_clip=0.2,
            loss_mask=loss_mask.clone(),
            importance_sampling_level="sequence",
        )

        # Second call with same inputs
        loss2, stat2 = ppo_actor_loss_fn(
            logprobs=logprobs.clone(),
            proximal_logprobs=proximal_logprobs.clone(),
            old_logprobs=old_logprobs.clone(),
            advantages=advantages.clone(),
            eps_clip=0.2,
            loss_mask=loss_mask.clone(),
            importance_sampling_level="sequence",
        )

        # Results should be identical
        assert torch.allclose(loss1, loss2)
        assert torch.allclose(stat1["importance_weight"], stat2["importance_weight"])
        assert torch.equal(stat1["clip_mask"], stat2["clip_mask"])

    def test_sequence_level_1d_requires_cu_seqlens(self):
        """Test that cu_seqlens is required for 1D tensors with sequence-level."""
        total_tokens = 8

        logprobs = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        proximal_logprobs = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        old_logprobs = torch.zeros(total_tokens)
        advantages = torch.ones(total_tokens)
        loss_mask = torch.ones(total_tokens, dtype=torch.bool)

        # Should raise ValueError when cu_seqlens is not provided for 1D tensors
        with pytest.raises(
            ValueError,
            match="cu_seqlens is required for 1D tensors",
        ):
            ppo_actor_loss_fn(
                logprobs=logprobs,
                proximal_logprobs=proximal_logprobs,
                old_logprobs=old_logprobs,
                advantages=advantages,
                eps_clip=0.2,
                loss_mask=loss_mask,
                importance_sampling_level="sequence",
                cu_seqlens=None,
            )

    def test_sequence_level_1d_variable_lengths(self):
        """Test packed sequences with variable lengths."""
        # Three sequences with lengths [3, 5, 4]
        cu_seqlens = torch.tensor([0, 3, 8, 12], dtype=torch.int32)
        total_tokens = 12

        # Different log ratios for each sequence
        logprobs = torch.tensor(
            [
                1.0,
                1.0,
                1.0,  # seq 1: 3 tokens, log_ratio=0.5
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,  # seq 2: 5 tokens, log_ratio=1.0
                3.0,
                3.0,
                3.0,
                3.0,  # seq 3: 4 tokens, log_ratio=1.5
            ]
        )
        proximal_logprobs = torch.tensor(
            [
                0.5,
                0.5,
                0.5,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.5,
                1.5,
                1.5,
                1.5,
            ]
        )
        old_logprobs = torch.zeros(total_tokens)
        advantages = torch.ones(total_tokens)
        loss_mask = torch.ones(total_tokens, dtype=torch.bool)

        loss, stat = ppo_actor_loss_fn(
            logprobs=logprobs,
            proximal_logprobs=proximal_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            eps_clip=0.2,
            loss_mask=loss_mask,
            importance_sampling_level="sequence",
            cu_seqlens=cu_seqlens,
        )

        # Verify each sequence has its own ratio
        expected_ratio_seq1 = torch.exp(torch.tensor(0.5))
        expected_ratio_seq2 = torch.exp(torch.tensor(1.0))
        expected_ratio_seq3 = torch.exp(torch.tensor(1.5))

        assert torch.allclose(
            stat["importance_weight"][0:3], expected_ratio_seq1.expand(3), atol=1e-5
        )
        assert torch.allclose(
            stat["importance_weight"][3:8], expected_ratio_seq2.expand(5), atol=1e-5
        )
        assert torch.allclose(
            stat["importance_weight"][8:12], expected_ratio_seq3.expand(4), atol=1e-5
        )


class TestPPOActorLossFnEdgeCases:
    """Additional edge case tests for ppo_actor_loss_fn."""

    def test_invalid_importance_sampling_level(self):
        """Test that invalid importance_sampling_level raises ValueError."""
        batch_size = 2
        seq_len = 4

        logprobs = torch.randn(batch_size, seq_len)
        proximal_logprobs = torch.randn(batch_size, seq_len)
        old_logprobs = torch.randn(batch_size, seq_len)
        advantages = torch.randn(batch_size, seq_len)
        loss_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        with pytest.raises(ValueError, match="Invalid importance_sampling_level"):
            ppo_actor_loss_fn(
                logprobs=logprobs,
                proximal_logprobs=proximal_logprobs,
                old_logprobs=old_logprobs,
                advantages=advantages,
                eps_clip=0.2,
                loss_mask=loss_mask,
                importance_sampling_level="invalid",
            )

    def test_batch_size_1_with_full_mask(self):
        """Test batch_size=1 with complete sequence (no padding)."""
        batch_size = 1
        seq_len = 5

        logprobs = torch.randn(batch_size, seq_len)
        proximal_logprobs = torch.randn(batch_size, seq_len)
        old_logprobs = torch.randn(batch_size, seq_len)
        advantages = torch.randn(batch_size, seq_len)
        loss_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        # Test with sequence-level
        loss_seq, stat_seq = ppo_actor_loss_fn(
            logprobs=logprobs,
            proximal_logprobs=proximal_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            eps_clip=0.2,
            loss_mask=loss_mask,
            importance_sampling_level="sequence",
        )

        # Test with token-level for comparison
        loss_tok, stat_tok = ppo_actor_loss_fn(
            logprobs=logprobs,
            proximal_logprobs=proximal_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            eps_clip=0.2,
            loss_mask=loss_mask,
            importance_sampling_level="token",
        )

        # Both should produce valid finite losses
        assert not torch.isnan(loss_seq) and not torch.isinf(loss_seq)
        assert not torch.isnan(loss_tok) and not torch.isinf(loss_tok)

    def test_batch_size_1_with_partial_mask(self):
        """Test batch_size=1 with partial masking (some padding)."""
        batch_size = 1
        seq_len = 10

        logprobs = torch.randn(batch_size, seq_len)
        proximal_logprobs = torch.randn(batch_size, seq_len)
        old_logprobs = torch.randn(batch_size, seq_len)
        advantages = torch.randn(batch_size, seq_len)
        # First 7 tokens valid, last 3 masked (padding)
        loss_mask = torch.tensor(
            [[True, True, True, True, True, True, True, False, False, False]]
        )

        loss, stat = ppo_actor_loss_fn(
            logprobs=logprobs,
            proximal_logprobs=proximal_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            eps_clip=0.2,
            loss_mask=loss_mask,
            importance_sampling_level="sequence",
        )

        # Verify masked tokens have zero weight
        assert torch.allclose(
            stat["importance_weight"][0, 7:], torch.zeros(3), atol=1e-6
        )

        # Verify all valid tokens have same ratio (same sequence)
        valid_ratios = stat["importance_weight"][0, :7]
        assert torch.allclose(valid_ratios, valid_ratios[0].expand(7), atol=1e-5)


class TestSAPOLossFn:
    """Test cases for sapo_loss_fn (Soft Adaptive Policy Optimization)."""

    @pytest.fixture
    def basic_2d_data(self):
        """Basic 2D tensor test data for SAPO."""
        batch_size = 4
        seq_len = 8

        return {
            "logprobs": torch.randn(batch_size, seq_len),
            "old_logprobs": torch.randn(batch_size, seq_len),
            "advantages": torch.randn(batch_size, seq_len),
            "loss_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
            "tau_pos": 1.0,
            "tau_neg": 1.05,
        }

    @pytest.fixture
    def basic_1d_data(self):
        """Basic 1D tensor test data for SAPO (packed)."""
        # 3 sequences with lengths [8, 12, 12]
        total_tokens = 32
        cu_seqlens = torch.tensor([0, 8, 20, 32], dtype=torch.int32)

        return {
            "logprobs": torch.randn(total_tokens),
            "old_logprobs": torch.randn(total_tokens),
            "advantages": torch.randn(total_tokens),
            "loss_mask": torch.ones(total_tokens, dtype=torch.bool),
            "tau_pos": 1.0,
            "tau_neg": 1.05,
            "cu_seqlens": cu_seqlens,
        }

    def test_2d_shape(self, basic_2d_data):
        """Test that SAPO loss returns correct shape for 2D tensors."""
        loss, stat = sapo_loss_fn(
            logprobs=basic_2d_data["logprobs"],
            old_logprobs=basic_2d_data["old_logprobs"],
            advantages=basic_2d_data["advantages"],
            tau_pos=basic_2d_data["tau_pos"],
            tau_neg=basic_2d_data["tau_neg"],
            loss_mask=basic_2d_data["loss_mask"],
            importance_sampling_level="token",
        )

        # Loss should be a scalar
        assert loss.ndim == 0
        assert loss.dtype == torch.float32

        # Stats should have correct shapes
        assert stat["sapo_soft_gate"].shape == basic_2d_data["logprobs"].shape
        assert stat["sapo_scaled_gate_pos"].shape == basic_2d_data["logprobs"].shape
        assert stat["sapo_scaled_gate_neg"].shape == basic_2d_data["logprobs"].shape
        assert stat["clip_mask"].shape == basic_2d_data["logprobs"].shape

    def test_1d_shape(self, basic_1d_data):
        """Test that SAPO loss returns correct shape for 1D tensors."""
        loss, stat = sapo_loss_fn(
            logprobs=basic_1d_data["logprobs"],
            old_logprobs=basic_1d_data["old_logprobs"],
            advantages=basic_1d_data["advantages"],
            tau_pos=basic_1d_data["tau_pos"],
            tau_neg=basic_1d_data["tau_neg"],
            loss_mask=basic_1d_data["loss_mask"],
            importance_sampling_level="token",
        )

        # Loss should be a scalar
        assert loss.ndim == 0
        assert loss.dtype == torch.float32

        # Stats should have correct shapes
        assert stat["sapo_soft_gate"].shape == basic_1d_data["logprobs"].shape
        assert stat["sapo_scaled_gate_pos"].shape == basic_1d_data["logprobs"].shape
        assert stat["sapo_scaled_gate_neg"].shape == basic_1d_data["logprobs"].shape
        assert stat["clip_mask"].shape == basic_1d_data["logprobs"].shape

    def test_known_value(self):
        """Test SAPO loss with known input values."""
        # Simple case: ratio = 1.0, advantages = 1.0
        logprobs = torch.tensor([[1.0, 1.0]])
        old_logprobs = torch.tensor([[1.0, 1.0]])  # ratio = exp(0) = 1.0
        advantages = torch.tensor([[1.0, 1.0]])
        loss_mask = torch.ones(1, 2, dtype=torch.bool)
        tau_pos = 1.0

        loss, stat = sapo_loss_fn(
            logprobs=logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            tau_pos=tau_pos,
            tau_neg=tau_pos,
            loss_mask=loss_mask,
            importance_sampling_level="token",
        )

        # At ratio=1.0, sigmoid(tau * (1.0 - 1.0)) = sigmoid(0) = 0.5
        # gate = 0.5 * (4.0 / tau) = 0.5 * 4.0 = 2.0
        # loss = -gate * advantage = -2.0 * 1.0 = -2.0
        expected_loss = -2.0
        assert torch.allclose(loss, torch.tensor(expected_loss), atol=1e-5)

        # Check gate value
        expected_gate = 2.0
        assert torch.allclose(
            stat["sapo_soft_gate"],
            torch.tensor([[expected_gate, expected_gate]]),
            atol=1e-5,
        )

    def test_gate_selection_positive_advantages(self):
        """Test that positive gate is selected for positive advantages."""
        # Create data with positive advantages
        logprobs = torch.tensor([[0.5, 0.5]])
        old_logprobs = torch.tensor([[0.0, 0.0]])
        advantages = torch.tensor([[1.0, 2.0]])  # All positive
        loss_mask = torch.ones(1, 2, dtype=torch.bool)
        tau_pos = 1.0
        tau_neg = 2.0  # Different from tau_pos

        loss, stat = sapo_loss_fn(
            logprobs=logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            tau_pos=tau_pos,
            tau_neg=tau_neg,
            loss_mask=loss_mask,
            importance_sampling_level="token",
        )

        # For positive advantages, gate should match gate_pos
        assert torch.allclose(
            stat["sapo_soft_gate"], stat["sapo_scaled_gate_pos"], atol=1e-6
        )

    def test_gate_selection_negative_advantages(self):
        """Test that negative gate is selected for negative advantages."""
        # Create data with negative advantages
        logprobs = torch.tensor([[0.5, 0.5]])
        old_logprobs = torch.tensor([[0.0, 0.0]])
        advantages = torch.tensor([[-1.0, -2.0]])  # All negative
        loss_mask = torch.ones(1, 2, dtype=torch.bool)
        tau_pos = 1.0
        tau_neg = 2.0  # Different from tau_pos

        loss, stat = sapo_loss_fn(
            logprobs=logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            tau_pos=tau_pos,
            tau_neg=tau_neg,
            loss_mask=loss_mask,
            importance_sampling_level="token",
        )

        # For negative advantages, gate should match gate_neg
        assert torch.allclose(
            stat["sapo_soft_gate"], stat["sapo_scaled_gate_neg"], atol=1e-6
        )

    def test_gate_selection_mixed_advantages(self):
        """Test gate selection with mixed positive and negative advantages."""
        logprobs = torch.tensor([[0.5, 0.5, 0.5]])
        old_logprobs = torch.tensor([[0.0, 0.0, 0.0]])
        advantages = torch.tensor([[1.0, -1.0, 2.0]])  # Mixed signs
        loss_mask = torch.ones(1, 3, dtype=torch.bool)
        tau_pos = 1.0
        tau_neg = 2.0

        loss, stat = sapo_loss_fn(
            logprobs=logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            tau_pos=tau_pos,
            tau_neg=tau_neg,
            loss_mask=loss_mask,
            importance_sampling_level="token",
        )

        # Check that positive advantages use gate_pos
        assert torch.allclose(
            stat["sapo_soft_gate"][0, 0], stat["sapo_scaled_gate_pos"][0, 0], atol=1e-6
        )
        assert torch.allclose(
            stat["sapo_soft_gate"][0, 2], stat["sapo_scaled_gate_pos"][0, 2], atol=1e-6
        )

        # Check that negative advantage uses gate_neg
        assert torch.allclose(
            stat["sapo_soft_gate"][0, 1], stat["sapo_scaled_gate_neg"][0, 1], atol=1e-6
        )

    def test_zero_tau_raises_error(self):
        """Test that zero tau_pos raises ValueError."""
        logprobs = torch.tensor([[1.0]])
        old_logprobs = torch.tensor([[1.0]])
        advantages = torch.tensor([[1.0]])
        loss_mask = torch.ones(1, 1, dtype=torch.bool)

        with pytest.raises(ValueError, match="must be positive"):
            sapo_loss_fn(
                logprobs=logprobs,
                old_logprobs=old_logprobs,
                advantages=advantages,
                tau_pos=0.0,
                tau_neg=1.0,
                loss_mask=loss_mask,
                importance_sampling_level="token",
            )

    def test_negative_tau_raises_error(self):
        """Test that negative tau_neg raises ValueError."""
        logprobs = torch.tensor([[1.0]])
        old_logprobs = torch.tensor([[1.0]])
        advantages = torch.tensor([[1.0]])
        loss_mask = torch.ones(1, 1, dtype=torch.bool)

        with pytest.raises(ValueError, match="must be positive"):
            sapo_loss_fn(
                logprobs=logprobs,
                old_logprobs=old_logprobs,
                advantages=advantages,
                tau_pos=1.0,
                tau_neg=-1.0,
                loss_mask=loss_mask,
                importance_sampling_level="token",
            )

    def test_token_level_importance_sampling(self):
        """Test SAPO with token-level importance sampling."""
        batch_size = 2
        seq_len = 4

        logprobs = torch.randn(batch_size, seq_len)
        old_logprobs = torch.randn(batch_size, seq_len)
        advantages = torch.randn(batch_size, seq_len)
        loss_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        loss, stat = sapo_loss_fn(
            logprobs=logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            tau_pos=1.0,
            tau_neg=1.05,
            loss_mask=loss_mask,
            importance_sampling_level="token",
        )

        # Verify loss is finite
        assert not torch.isnan(loss) and not torch.isinf(loss)

        # Verify stat shapes
        assert stat["sapo_soft_gate"].shape == logprobs.shape
        assert stat["sapo_scaled_gate_pos"].shape == logprobs.shape
        assert stat["sapo_scaled_gate_neg"].shape == logprobs.shape

    def test_sequence_level_importance_sampling_2d(self):
        """Test SAPO with sequence-level importance sampling (2D)."""
        batch_size = 2
        seq_len = 4

        logprobs = torch.randn(batch_size, seq_len)
        old_logprobs = torch.randn(batch_size, seq_len)
        advantages = torch.randn(batch_size, seq_len)
        loss_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        loss, stat = sapo_loss_fn(
            logprobs=logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            tau_pos=1.0,
            tau_neg=1.05,
            loss_mask=loss_mask,
            importance_sampling_level="sequence",
        )

        # Verify loss is finite
        assert not torch.isnan(loss) and not torch.isinf(loss)

        # Verify stat shapes
        assert stat["sapo_soft_gate"].shape == logprobs.shape
        assert stat["sapo_scaled_gate_pos"].shape == logprobs.shape
        assert stat["sapo_scaled_gate_neg"].shape == logprobs.shape

    def test_sequence_level_importance_sampling_1d(self):
        """Test SAPO with sequence-level importance sampling (1D packed)."""
        # 2 sequences with lengths [4, 6]
        total_tokens = 10
        cu_seqlens = torch.tensor([0, 4, 10], dtype=torch.int32)

        logprobs = torch.randn(total_tokens)
        old_logprobs = torch.randn(total_tokens)
        advantages = torch.randn(total_tokens)
        loss_mask = torch.ones(total_tokens, dtype=torch.bool)

        loss, stat = sapo_loss_fn(
            logprobs=logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            tau_pos=1.0,
            tau_neg=1.05,
            loss_mask=loss_mask,
            importance_sampling_level="sequence",
            cu_seqlens=cu_seqlens,
        )

        # Verify loss is finite
        assert not torch.isnan(loss) and not torch.isinf(loss)

        # Verify stat shapes
        assert stat["sapo_soft_gate"].shape == logprobs.shape
        assert stat["sapo_scaled_gate_pos"].shape == logprobs.shape
        assert stat["sapo_scaled_gate_neg"].shape == logprobs.shape

    def test_gradient_normalization(self):
        """Test that 4/tau normalization provides consistent gradients."""
        logprobs = torch.tensor([[0.0]], requires_grad=True)
        old_logprobs = torch.tensor([[0.0]])
        advantages = torch.tensor([[1.0]])
        loss_mask = torch.ones(1, 1, dtype=torch.bool)

        # Test with tau=1.0
        loss1, _ = sapo_loss_fn(
            logprobs=logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            tau_pos=1.0,
            tau_neg=1.0,
            loss_mask=loss_mask,
            importance_sampling_level="token",
        )
        loss1.backward()
        grad1 = logprobs.grad.clone()
        logprobs.grad.zero_()

        # Test with tau=2.0
        loss2, _ = sapo_loss_fn(
            logprobs=logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            tau_pos=2.0,
            tau_neg=2.0,
            loss_mask=loss_mask,
            importance_sampling_level="token",
        )
        loss2.backward()
        grad2 = logprobs.grad.clone()

        # At ratio=1.0, gradient magnitude should be identical due to 4/tau normalization.
        assert torch.allclose(grad1, grad2, atol=1e-6)

    def test_loss_mask(self):
        """Test that loss_mask correctly excludes masked tokens."""
        logprobs = torch.randn(1, 4)
        old_logprobs = torch.randn(1, 4)
        advantages = torch.randn(1, 4)
        # Only first 2 tokens are valid
        loss_mask = torch.tensor([[True, True, False, False]])

        loss, stat = sapo_loss_fn(
            logprobs=logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            tau_pos=1.0,
            tau_neg=1.0,
            loss_mask=loss_mask,
            importance_sampling_level="token",
        )

        # Compute expected loss manually (only first 2 tokens)
        log_ratio = logprobs[:, :2] - old_logprobs[:, :2]
        ratio = torch.exp(log_ratio)
        gate = torch.sigmoid(1.0 * (ratio - 1.0)) * 4.0
        expected_loss = -(gate * advantages[:, :2]).sum() / 2

        assert torch.allclose(loss, expected_loss, atol=1e-5)
