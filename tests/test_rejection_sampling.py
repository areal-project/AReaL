import pytest
import torch

from areal.api.cli_args import RejectionSamplingConfig
from areal.utils.functional import apply_rejection_sampling


class TestRejectionSamplingConfig:
    """Tests for RejectionSamplingConfig validation."""

    def test_ratio_upper_must_exceed_one(self):
        """ratio metric with upper <= 1.0 should raise ValueError."""
        with pytest.raises(ValueError, match="upper must be > 1.0"):
            RejectionSamplingConfig(metric="ratio", upper=1.0)

    def test_ratio_lower_must_be_positive(self):
        """ratio metric with lower <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="lower must be positive"):
            RejectionSamplingConfig(metric="ratio", lower=-0.1, upper=5.0)

    def test_kl_upper_must_be_positive(self):
        """KL metrics with upper <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="upper must be positive"):
            RejectionSamplingConfig(metric="kl_k2", upper=0.0)

    def test_agg_warning_for_token_level(self):
        """agg != 'mean' with level='token' should warn."""
        with pytest.warns(UserWarning, match="agg=.*is ignored"):
            RejectionSamplingConfig(level="token", agg="max", metric="ratio", upper=5.0)

    def test_clamp_only_supports_ratio_metric(self):
        """action='clamp' with non-ratio metric should raise ValueError."""
        with pytest.raises(
            ValueError, match="action='clamp' only supports metric='ratio'"
        ):
            RejectionSamplingConfig(action="clamp", metric="kl_k2", upper=1.0)

    def test_clamp_sets_default_lower_to_zero(self):
        """action='clamp' without explicit lower should default to 0.0."""
        config = RejectionSamplingConfig(action="clamp", metric="ratio", upper=5.0)
        assert config.lower == 0.0


class TestRejectionSamplingMask:
    """Tests for apply_rejection_sampling with action='mask'."""

    def test_ratio_upper_bound_filters_high_ratio(self):
        """Token with ratio > upper should be filtered."""
        config = RejectionSamplingConfig(level="token", metric="ratio", upper=2.0)
        # ratio = exp(1) ~ 2.72 > 2.0
        proximal_logprobs = torch.tensor([[0.0, 1.0, 0.0]])
        old_logprobs = torch.tensor([[0.0, 0.0, 0.0]])
        loss_mask = torch.ones(1, 3)

        result = apply_rejection_sampling(
            proximal_logprobs,
            old_logprobs,
            loss_mask,
            cu_seqlens=None,
            config=config,
        )

        assert result.loss_mask[0, 0] == 1.0  # ratio = 1, keep
        assert result.loss_mask[0, 1] == 0.0  # ratio ~ 2.72 > 2.0, filter
        assert result.loss_mask[0, 2] == 1.0  # ratio = 1, keep
        assert result.filtered_fraction > 0

    def test_ratio_lower_bound_filters_low_ratio(self):
        """Token with ratio < lower should be filtered."""
        config = RejectionSamplingConfig(
            level="token", metric="ratio", lower=0.5, upper=2.0
        )
        # ratio = exp(-1) ~ 0.37 < 0.5
        proximal_logprobs = torch.tensor([[0.0, -1.0, 0.0]])
        old_logprobs = torch.tensor([[0.0, 0.0, 0.0]])
        loss_mask = torch.ones(1, 3)

        result = apply_rejection_sampling(
            proximal_logprobs,
            old_logprobs,
            loss_mask,
            cu_seqlens=None,
            config=config,
        )

        assert result.loss_mask[0, 0] == 1.0  # ratio = 1, keep
        assert result.loss_mask[0, 1] == 0.0  # ratio ~ 0.37 < 0.5, filter
        assert result.loss_mask[0, 2] == 1.0  # ratio = 1, keep

    def test_sequence_ratio_mask_uniform_weight(self):
        """Sequence-level ratio mask: behave_imp_weight = geometric mean for all tokens."""
        config = RejectionSamplingConfig(
            level="sequence", agg="mean", metric="ratio", upper=5.0
        )
        # log_ratios [0.0, 0.5, 1.0], geo_mean = exp(mean([0, 0.5, 1.0])) = exp(0.5)
        proximal_logprobs = torch.tensor([[0.0, 0.5, 1.0]])
        old_logprobs = torch.tensor([[0.0, 0.0, 0.0]])
        loss_mask = torch.ones(1, 3)

        result = apply_rejection_sampling(
            proximal_logprobs, old_logprobs, loss_mask, cu_seqlens=None, config=config
        )

        expected_weight = torch.exp(torch.tensor(0.5))
        torch.testing.assert_close(
            result.behave_imp_weight[0],
            expected_weight.expand(3),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_sequence_ratio_mask_uniform_weight_1d_packed(self):
        """1D packed: sequence-level ratio mask should also use uniform weight."""
        config = RejectionSamplingConfig(
            level="sequence", agg="mean", metric="ratio", upper=5.0
        )
        # Seq 0 (3 tokens): log_ratios [0.0, 0.6, 0.3], geo_mean = exp(0.3)
        # Seq 1 (2 tokens): log_ratios [0.0, 0.0], geo_mean = exp(0) = 1.0
        proximal_logprobs = torch.tensor([0.0, 0.6, 0.3, 0.0, 0.0])
        old_logprobs = torch.zeros(5)
        loss_mask = torch.ones(5)
        cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32)

        result = apply_rejection_sampling(
            proximal_logprobs,
            old_logprobs,
            loss_mask,
            cu_seqlens=cu_seqlens,
            config=config,
        )

        geo_mean_seq0 = torch.exp(torch.tensor(0.3))
        torch.testing.assert_close(
            result.behave_imp_weight[:3],
            geo_mean_seq0.expand(3),
            rtol=1e-5,
            atol=1e-5,
        )
        torch.testing.assert_close(
            result.behave_imp_weight[3:],
            torch.ones(2),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_sequence_kl_metric_keeps_per_token_weight(self):
        """Sequence-level KL metric: behave_imp_weight stays per-token (not uniform)."""
        config = RejectionSamplingConfig(
            level="sequence", agg="mean", metric="kl_k2", upper=5.0
        )
        # log_ratios [0.0, 1.0, 0.0] -> per-token ratios [1.0, e, 1.0]
        proximal_logprobs = torch.tensor([[0.0, 1.0, 0.0]])
        old_logprobs = torch.tensor([[0.0, 0.0, 0.0]])
        loss_mask = torch.ones(1, 3)

        result = apply_rejection_sampling(
            proximal_logprobs, old_logprobs, loss_mask, cu_seqlens=None, config=config
        )

        expected = torch.exp(torch.tensor([0.0, 1.0, 0.0]))
        torch.testing.assert_close(
            result.behave_imp_weight[0], expected, rtol=1e-5, atol=1e-5
        )

    def test_kl_k2_sequence_mean_keeps_clean_sequences(self):
        """Sequence-level mean KL K2 should keep sequences below threshold."""
        config = RejectionSamplingConfig(
            level="sequence", agg="mean", metric="kl_k2", upper=0.5
        )
        # Sequence 0: one token with log_ratio=1, KL_k2 = 0.5*1^2 = 0.5
        #   mean KL = (0 + 0.5 + 0) / 3 = 0.167 < 0.5, keep
        # Sequence 1: all zeros, mean KL = 0, keep
        proximal_logprobs = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        old_logprobs = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        loss_mask = torch.ones(2, 3)

        result = apply_rejection_sampling(
            proximal_logprobs,
            old_logprobs,
            loss_mask,
            cu_seqlens=None,
            config=config,
        )

        assert torch.all(result.loss_mask[0] == 1.0)
        assert torch.all(result.loss_mask[1] == 1.0)

    def test_kl_k2_sequence_mean_filters_stale_sequence(self):
        """Sequence with high mean KL should be fully filtered."""
        config = RejectionSamplingConfig(
            level="sequence", agg="mean", metric="kl_k2", upper=0.1
        )
        # Sequence 0: token with log_ratio=2, KL_k2 = 0.5*4 = 2.0
        #   mean KL = (0 + 2.0 + 0) / 3 = 0.667 > 0.1, filter entire sequence
        proximal_logprobs = torch.tensor([[0.0, 2.0, 0.0], [0.0, 0.0, 0.0]])
        old_logprobs = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        loss_mask = torch.ones(2, 3)

        result = apply_rejection_sampling(
            proximal_logprobs,
            old_logprobs,
            loss_mask,
            cu_seqlens=None,
            config=config,
        )

        assert torch.all(result.loss_mask[0] == 0.0)  # entire sequence filtered
        assert torch.all(result.loss_mask[1] == 1.0)  # clean sequence kept

    def test_packed_1d_format_with_cu_seqlens(self):
        """1D packed format should work with cu_seqlens."""
        config = RejectionSamplingConfig(
            level="sequence", agg="mean", metric="kl_k2", upper=0.1
        )
        # Two sequences packed: [seq0: 3 tokens, seq1: 2 tokens]
        proximal_logprobs = torch.tensor([0.0, 2.0, 0.0, 0.0, 0.0])
        old_logprobs = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
        loss_mask = torch.ones(5)
        cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32)

        result = apply_rejection_sampling(
            proximal_logprobs,
            old_logprobs,
            loss_mask,
            cu_seqlens=cu_seqlens,
            config=config,
        )

        # Sequence 0: mean KL > 0.1, entire sequence filtered
        assert torch.all(result.loss_mask[:3] == 0.0)
        # Sequence 1: mean KL = 0, kept
        assert torch.all(result.loss_mask[3:] == 1.0)

    def test_padding_tokens_not_counted(self):
        """Padding tokens (loss_mask=0) should not affect filtering."""
        config = RejectionSamplingConfig(
            level="sequence", agg="mean", metric="kl_k2", upper=0.5
        )
        # Only first token is valid, with KL_k2 = 0.5 * 2^2 = 2.0 > 0.5
        proximal_logprobs = torch.tensor([[2.0, 0.0, 0.0]])
        old_logprobs = torch.tensor([[0.0, 0.0, 0.0]])
        loss_mask = torch.tensor([[1.0, 0.0, 0.0]])  # only first token valid

        result = apply_rejection_sampling(
            proximal_logprobs,
            old_logprobs,
            loss_mask,
            cu_seqlens=None,
            config=config,
        )

        # mean KL = 2.0 / 1 = 2.0 > 0.5, filter
        assert result.loss_mask[0, 0] == 0.0


class TestRejectionSamplingClamp:
    """Tests for apply_rejection_sampling with action='clamp'."""

    def test_clamp_does_not_modify_loss_mask(self):
        """Clamp mode should never modify loss_mask."""
        config = RejectionSamplingConfig(
            level="token", metric="ratio", action="clamp", upper=2.0
        )
        proximal_logprobs = torch.tensor([[0.0, 5.0, 0.0]])  # ratio ~ 148.4
        old_logprobs = torch.tensor([[0.0, 0.0, 0.0]])
        loss_mask = torch.ones(1, 3)

        result = apply_rejection_sampling(
            proximal_logprobs,
            old_logprobs,
            loss_mask,
            cu_seqlens=None,
            config=config,
        )

        # loss_mask unchanged -- all tokens still participate
        assert torch.all(result.loss_mask == 1.0)

    def test_clamp_truncates_high_ratio(self):
        """Token with ratio > upper should have weight clamped to upper."""
        config = RejectionSamplingConfig(
            level="token", metric="ratio", action="clamp", upper=5.0
        )
        # ratio = exp(2) ~ 7.39 > 5.0
        proximal_logprobs = torch.tensor([[0.0, 2.0, 0.0]])
        old_logprobs = torch.tensor([[0.0, 0.0, 0.0]])
        loss_mask = torch.ones(1, 3)

        result = apply_rejection_sampling(
            proximal_logprobs,
            old_logprobs,
            loss_mask,
            cu_seqlens=None,
            config=config,
        )

        # Token 0: ratio=1.0, not clamped
        torch.testing.assert_close(
            result.behave_imp_weight[0, 0], torch.tensor(1.0), rtol=1e-5, atol=1e-5
        )
        # Token 1: ratio~7.39, clamped to 5.0
        torch.testing.assert_close(
            result.behave_imp_weight[0, 1], torch.tensor(5.0), rtol=1e-5, atol=1e-5
        )
        # Token 2: ratio=1.0, not clamped
        torch.testing.assert_close(
            result.behave_imp_weight[0, 2], torch.tensor(1.0), rtol=1e-5, atol=1e-5
        )

    def test_clamp_sequence_level(self):
        """Sequence-level clamp with ratio metric uses geometric mean as uniform weight.

        geo_mean = exp(mean(log_ratio)), broadcast to all tokens in the sequence.
        When geo_mean > upper, the uniform weight is clamped to upper for all tokens.
        """
        config = RejectionSamplingConfig(
            level="sequence",
            metric="ratio",
            action="clamp",
            agg="mean",
            upper=3.0,
        )
        # Sequence 0: log_ratios [0, 4, 0], geo_mean = exp(4/3) ≈ 3.79 > 3.0 -> clamp
        # Sequence 1: log_ratios [0, 0, 0], geo_mean = exp(0) = 1.0 <= 3.0 -> no clamp
        proximal_logprobs = torch.tensor([[0.0, 4.0, 0.0], [0.0, 0.0, 0.0]])
        old_logprobs = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        loss_mask = torch.ones(2, 3)

        result = apply_rejection_sampling(
            proximal_logprobs,
            old_logprobs,
            loss_mask,
            cu_seqlens=None,
            config=config,
        )

        # Sequence 0: geo_mean ≈ 3.79 > 3.0, all tokens get clamped uniform weight = 3.0
        torch.testing.assert_close(
            result.behave_imp_weight[0],
            torch.tensor([3.0, 3.0, 3.0]),
            rtol=1e-5,
            atol=1e-5,
        )
        # Sequence 1: geo_mean = 1.0 <= 3.0, uniform weight = 1.0 (no clamp)
        torch.testing.assert_close(
            result.behave_imp_weight[1],
            torch.ones(3),
            rtol=1e-5,
            atol=1e-5,
        )
        # loss_mask unchanged for both sequences
        assert torch.all(result.loss_mask == 1.0)

    def test_clamp_reports_clamped_fraction(self):
        """filtered_fraction should report proportion of clamped tokens."""
        config = RejectionSamplingConfig(
            level="token", metric="ratio", action="clamp", upper=2.0
        )
        # 1 of 3 tokens has ratio > 2.0
        proximal_logprobs = torch.tensor([[0.0, 1.0, 0.0]])  # ratios: 1.0, 2.72, 1.0
        old_logprobs = torch.tensor([[0.0, 0.0, 0.0]])
        loss_mask = torch.ones(1, 3)

        result = apply_rejection_sampling(
            proximal_logprobs,
            old_logprobs,
            loss_mask,
            cu_seqlens=None,
            config=config,
        )

        assert result.filtered_fraction > 0  # at least 1 token clamped


class TestBackwardCompatibility:
    """Verify new config reproduces old behave_imp_weight_cap behavior."""

    def test_equivalent_to_legacy_token_mask(self):
        """New ratio/token/mask config should match old token_mask behavior."""
        config = RejectionSamplingConfig(level="token", metric="ratio", upper=5.0)

        proximal_logprobs = torch.tensor([[0.0, 2.0, -0.5]])  # ratios: 1.0, 7.39, 0.61
        old_logprobs = torch.tensor([[0.0, 0.0, 0.0]])
        loss_mask = torch.ones(1, 3)

        result = apply_rejection_sampling(
            proximal_logprobs,
            old_logprobs,
            loss_mask,
            cu_seqlens=None,
            config=config,
        )

        # ratio=7.39 > 5.0 -> filtered, others kept
        assert result.loss_mask[0, 0] == 1.0
        assert result.loss_mask[0, 1] == 0.0
        assert result.loss_mask[0, 2] == 1.0

    def test_equivalent_to_legacy_token_truncate(self):
        """New ratio/token/clamp config should match old token_truncate behavior."""
        config = RejectionSamplingConfig(
            level="token", metric="ratio", action="clamp", upper=5.0
        )

        proximal_logprobs = torch.tensor([[0.0, 2.0, -0.5]])  # ratios: 1.0, 7.39, 0.61
        old_logprobs = torch.tensor([[0.0, 0.0, 0.0]])
        loss_mask = torch.ones(1, 3)

        result = apply_rejection_sampling(
            proximal_logprobs,
            old_logprobs,
            loss_mask,
            cu_seqlens=None,
            config=config,
        )

        # All tokens kept in loss_mask
        assert torch.all(result.loss_mask == 1.0)
        # ratio=7.39 > 5.0 -> clamped to 5.0
        torch.testing.assert_close(
            result.behave_imp_weight[0, 1], torch.tensor(5.0), rtol=1e-5, atol=1e-5
        )
        # ratio=0.61 within [0, 5.0] -> not clamped
        torch.testing.assert_close(
            result.behave_imp_weight[0, 2],
            torch.exp(torch.tensor(-0.5)),
            rtol=1e-5,
            atol=1e-5,
        )


class TestKLK1Metric:
    """Tests for kl_k1 metric (forward KL unbiased estimator, can be negative)."""

    def test_kl_k1_can_be_negative(self):
        """kl_k1 = log(r) can be negative when proximal < old."""
        config = RejectionSamplingConfig(
            level="token", metric="kl_k1", upper=1.0, lower=-0.5
        )
        # log_ratio = -1.0, so kl_k1 = -1.0 < lower=-0.5 -> filtered
        # log_ratio = -0.3, so kl_k1 = -0.3, within [-0.5, 1.0] -> kept
        proximal_logprobs = torch.tensor([[0.0, -1.0, -0.3]])
        old_logprobs = torch.tensor([[0.0, 0.0, 0.0]])
        loss_mask = torch.ones(1, 3)

        result = apply_rejection_sampling(
            proximal_logprobs,
            old_logprobs,
            loss_mask,
            cu_seqlens=None,
            config=config,
        )

        assert result.loss_mask[0, 0] == 1.0  # kl_k1=0, within bounds
        assert result.loss_mask[0, 1] == 0.0  # kl_k1=-1.0 < -0.5, filtered
        assert result.loss_mask[0, 2] == 1.0  # kl_k1=-0.3, within bounds

    def test_kl_k1_filters_high_positive(self):
        """kl_k1 with high positive value should be filtered."""
        config = RejectionSamplingConfig(level="token", metric="kl_k1", upper=0.5)
        # log_ratio = 1.0, kl_k1 = 1.0 > 0.5 -> filtered
        proximal_logprobs = torch.tensor([[0.0, 1.0]])
        old_logprobs = torch.tensor([[0.0, 0.0]])
        loss_mask = torch.ones(1, 2)

        result = apply_rejection_sampling(
            proximal_logprobs,
            old_logprobs,
            loss_mask,
            cu_seqlens=None,
            config=config,
        )

        assert result.loss_mask[0, 0] == 1.0  # kl_k1=0
        assert result.loss_mask[0, 1] == 0.0  # kl_k1=1.0 > 0.5

    def test_kl_k1_sequence_level(self):
        """kl_k1 should work at sequence level with mean aggregation."""
        config = RejectionSamplingConfig(
            level="sequence", agg="mean", metric="kl_k1", upper=0.5
        )
        # Seq 0: log_ratios [0, 1.0, -0.5], kl_k1 values [0, 1.0, -0.5]
        #   mean = (0 + 1.0 + (-0.5)) / 3 = 0.167 < 0.5, keep
        # Seq 1: log_ratios [2.0, 2.0, 0], kl_k1 values [2.0, 2.0, 0]
        #   mean = (2.0 + 2.0 + 0) / 3 = 1.33 > 0.5, filter
        proximal_logprobs = torch.tensor([[0.0, 1.0, -0.5], [2.0, 2.0, 0.0]])
        old_logprobs = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        loss_mask = torch.ones(2, 3)

        result = apply_rejection_sampling(
            proximal_logprobs,
            old_logprobs,
            loss_mask,
            cu_seqlens=None,
            config=config,
        )

        assert torch.all(result.loss_mask[0] == 1.0)  # seq 0 kept
        assert torch.all(result.loss_mask[1] == 0.0)  # seq 1 filtered


class TestKLK3Metric:
    """Tests for kl_k3 metric (exact forward KL estimator, non-negative)."""

    def test_kl_k3_is_non_negative(self):
        """kl_k3 = exp(-log_ratio) - 1 - (-log_ratio) should be >= 0."""
        config = RejectionSamplingConfig(level="token", metric="kl_k3", upper=0.5)
        # log_ratio = 0.5, kl_k3 = exp(-0.5) - 1 - (-0.5) = 0.6065 - 0.5 = 0.1065
        # log_ratio = -0.5, kl_k3 = exp(0.5) - 1 - 0.5 = 1.6487 - 1.5 = 0.1487
        # Both below upper=0.5 -> kept
        proximal_logprobs = torch.tensor([[0.5, -0.5, 0.0]])
        old_logprobs = torch.tensor([[0.0, 0.0, 0.0]])
        loss_mask = torch.ones(1, 3)

        result = apply_rejection_sampling(
            proximal_logprobs,
            old_logprobs,
            loss_mask,
            cu_seqlens=None,
            config=config,
        )

        assert torch.all(result.loss_mask == 1.0)

    def test_kl_k3_filters_stale_tokens(self):
        """kl_k3 with large divergence should exceed threshold."""
        config = RejectionSamplingConfig(level="token", metric="kl_k3", upper=0.5)
        # log_ratio = 2.0, kl_k3 = exp(-2) - 1 - (-2) = 0.1353 + 1 = 1.1353
        # kl_k3 = 1.1353 > 0.5 -> filtered
        proximal_logprobs = torch.tensor([[0.0, 2.0]])
        old_logprobs = torch.tensor([[0.0, 0.0]])
        loss_mask = torch.ones(1, 2)

        result = apply_rejection_sampling(
            proximal_logprobs,
            old_logprobs,
            loss_mask,
            cu_seqlens=None,
            config=config,
        )

        assert result.loss_mask[0, 0] == 1.0  # kl_k3=0
        assert result.loss_mask[0, 1] == 0.0  # kl_k3 > 0.5


class TestAggregationMethods:
    """Tests for sum and max aggregation methods (Issue 10)."""

    def test_sequence_sum_is_length_sensitive(self):
        """Sum aggregation should be sensitive to sequence length."""
        config = RejectionSamplingConfig(
            level="sequence", agg="sum", metric="kl_k2", upper=1.0
        )
        # Seq 0: 4 tokens each with kl_k2 = 0.5*0.5^2 = 0.125
        #   sum = 4 * 0.125 = 0.5 < 1.0, keep
        # Seq 1: 4 tokens each with kl_k2 = 0.5*1.0^2 = 0.5
        #   sum = 4 * 0.5 = 2.0 > 1.0, filter
        proximal_logprobs = torch.tensor([[0.5, 0.5, 0.5, 0.5], [1.0, 1.0, 1.0, 1.0]])
        old_logprobs = torch.zeros(2, 4)
        loss_mask = torch.ones(2, 4)

        result = apply_rejection_sampling(
            proximal_logprobs,
            old_logprobs,
            loss_mask,
            cu_seqlens=None,
            config=config,
        )

        assert torch.all(result.loss_mask[0] == 1.0)  # sum < 1.0, kept
        assert torch.all(result.loss_mask[1] == 0.0)  # sum > 1.0, filtered

    def test_sequence_max_filters_single_high_token(self):
        """Max aggregation should filter based on worst token in sequence."""
        config = RejectionSamplingConfig(
            level="sequence", agg="max", metric="ratio", upper=3.0
        )
        # Seq 0: ratios [1, 1, exp(2)~7.39], max=7.39 > 3.0 -> filter entire seq
        # Seq 1: ratios [1, exp(0.5)~1.65, 1], max=1.65 < 3.0 -> keep
        proximal_logprobs = torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.5, 0.0]])
        old_logprobs = torch.zeros(2, 3)
        loss_mask = torch.ones(2, 3)

        result = apply_rejection_sampling(
            proximal_logprobs,
            old_logprobs,
            loss_mask,
            cu_seqlens=None,
            config=config,
        )

        assert torch.all(result.loss_mask[0] == 0.0)  # max > 3.0, filtered
        assert torch.all(result.loss_mask[1] == 1.0)  # max < 3.0, kept

    def test_packed_sum_aggregation(self):
        """Sum aggregation should work with 1D packed format."""
        config = RejectionSamplingConfig(
            level="sequence", agg="sum", metric="kl_k2", upper=0.5
        )
        # Seq 0 (3 tokens): log_ratios [0, 1.0, 0], kl_k2 = [0, 0.5, 0], sum=0.5
        # Seq 1 (2 tokens): log_ratios [2.0, 0], kl_k2 = [2.0, 0], sum=2.0 > 0.5
        proximal_logprobs = torch.tensor([0.0, 1.0, 0.0, 2.0, 0.0])
        old_logprobs = torch.zeros(5)
        loss_mask = torch.ones(5)
        cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32)

        result = apply_rejection_sampling(
            proximal_logprobs,
            old_logprobs,
            loss_mask,
            cu_seqlens=cu_seqlens,
            config=config,
        )

        assert torch.all(result.loss_mask[:3] == 1.0)  # seq 0: sum=0.5, kept
        assert torch.all(result.loss_mask[3:] == 0.0)  # seq 1: sum=2.0 > 0.5, filtered

    def test_packed_max_aggregation(self):
        """Max aggregation should work with 1D packed format."""
        config = RejectionSamplingConfig(
            level="sequence", agg="max", metric="ratio", upper=3.0
        )
        # Seq 0 (3 tokens): ratios [1, exp(2)~7.39, 1], max=7.39 > 3.0 -> filter
        # Seq 1 (2 tokens): ratios [1, 1], max=1 <= 3.0 -> keep
        proximal_logprobs = torch.tensor([0.0, 2.0, 0.0, 0.0, 0.0])
        old_logprobs = torch.zeros(5)
        loss_mask = torch.ones(5)
        cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32)

        result = apply_rejection_sampling(
            proximal_logprobs,
            old_logprobs,
            loss_mask,
            cu_seqlens=cu_seqlens,
            config=config,
        )

        assert torch.all(result.loss_mask[:3] == 0.0)  # seq 0: max > 3.0
        assert torch.all(result.loss_mask[3:] == 1.0)  # seq 1: max <= 3.0


class TestEdgeCases:
    """Tests for edge cases (Issue 15)."""

    def test_empty_loss_mask(self):
        """All-zero loss_mask should produce no filtering and zero fraction."""
        config = RejectionSamplingConfig(level="token", metric="ratio", upper=2.0)
        proximal_logprobs = torch.tensor([[5.0, 5.0, 5.0]])  # huge ratios
        old_logprobs = torch.zeros(1, 3)
        loss_mask = torch.zeros(1, 3)  # all masked

        result = apply_rejection_sampling(
            proximal_logprobs,
            old_logprobs,
            loss_mask,
            cu_seqlens=None,
            config=config,
        )

        # loss_mask stays all-zero (nothing to filter)
        assert torch.all(result.loss_mask == 0.0)
        # behave_imp_weight should be zeroed where mask is 0
        assert torch.all(result.behave_imp_weight == 0.0)
        assert result.filtered_fraction == 0.0

    def test_single_token_sequence(self):
        """Single-token sequences should work correctly."""
        config = RejectionSamplingConfig(
            level="sequence", agg="mean", metric="ratio", upper=2.0
        )
        proximal_logprobs = torch.tensor([[1.0]])  # ratio=exp(1)~2.72 > 2.0
        old_logprobs = torch.tensor([[0.0]])
        loss_mask = torch.ones(1, 1)

        result = apply_rejection_sampling(
            proximal_logprobs,
            old_logprobs,
            loss_mask,
            cu_seqlens=None,
            config=config,
        )

        assert result.loss_mask[0, 0] == 0.0  # filtered

    def test_all_tokens_filtered(self):
        """When all tokens exceed threshold, everything should be filtered."""
        config = RejectionSamplingConfig(level="token", metric="ratio", upper=1.5)
        # All ratios > 1.5
        proximal_logprobs = torch.tensor([[1.0, 2.0, 3.0]])
        old_logprobs = torch.zeros(1, 3)
        loss_mask = torch.ones(1, 3)

        result = apply_rejection_sampling(
            proximal_logprobs,
            old_logprobs,
            loss_mask,
            cu_seqlens=None,
            config=config,
        )

        assert torch.all(result.loss_mask == 0.0)
        assert result.filtered_fraction == 1.0

    def test_clamp_with_lower_bound(self):
        """Clamp mode with explicit lower bound should clamp from both sides."""
        config = RejectionSamplingConfig(
            level="token", metric="ratio", action="clamp", upper=3.0, lower=0.5
        )
        # ratio = exp(-2) ~ 0.135 < 0.5 -> clamped to 0.5
        # ratio = exp(2) ~ 7.39 > 3.0 -> clamped to 3.0
        # ratio = exp(0) = 1.0, within bounds -> unchanged
        proximal_logprobs = torch.tensor([[-2.0, 2.0, 0.0]])
        old_logprobs = torch.zeros(1, 3)
        loss_mask = torch.ones(1, 3)

        result = apply_rejection_sampling(
            proximal_logprobs,
            old_logprobs,
            loss_mask,
            cu_seqlens=None,
            config=config,
        )

        torch.testing.assert_close(
            result.behave_imp_weight[0, 0], torch.tensor(0.5), rtol=1e-5, atol=1e-5
        )
        torch.testing.assert_close(
            result.behave_imp_weight[0, 1], torch.tensor(3.0), rtol=1e-5, atol=1e-5
        )
        torch.testing.assert_close(
            result.behave_imp_weight[0, 2], torch.tensor(1.0), rtol=1e-5, atol=1e-5
        )

    def test_ratio_exactly_at_upper_bound(self):
        """Token with ratio exactly equal to upper should pass (<=)."""
        config = RejectionSamplingConfig(level="token", metric="ratio", upper=2.0)
        # log_ratio = ln(2.0) ~ 0.6931
        log_2 = torch.tensor(2.0).log()
        proximal_logprobs = torch.tensor([[log_2.item()]])
        old_logprobs = torch.tensor([[0.0]])
        loss_mask = torch.ones(1, 1)

        result = apply_rejection_sampling(
            proximal_logprobs,
            old_logprobs,
            loss_mask,
            cu_seqlens=None,
            config=config,
        )

        assert result.loss_mask[0, 0] == 1.0  # exactly at boundary, kept

    def test_nan_from_inf_logprobs(self):
        """Non-finite log-probs (both -inf) should not produce NaN."""
        config = RejectionSamplingConfig(level="token", metric="ratio", upper=5.0)
        proximal_logprobs = torch.tensor([[0.0, float("-inf")]])
        old_logprobs = torch.tensor([[0.0, float("-inf")]])
        loss_mask = torch.ones(1, 2)

        result = apply_rejection_sampling(
            proximal_logprobs,
            old_logprobs,
            loss_mask,
            cu_seqlens=None,
            config=config,
        )

        # No NaN in outputs
        assert not torch.isnan(result.behave_imp_weight).any()
        assert not torch.isnan(result.loss_mask).any()

    def test_all_masked_sequence_with_max_agg(self):
        """Sequence with all tokens masked should pass bounds check with max agg."""
        config = RejectionSamplingConfig(
            level="sequence", agg="max", metric="ratio", upper=2.0
        )
        # Seq 0: all masked, should be treated as in-bounds
        # Seq 1: valid tokens, ratio=1.0 within bounds
        proximal_logprobs = torch.tensor([[5.0, 5.0], [0.0, 0.0]])
        old_logprobs = torch.zeros(2, 2)
        loss_mask = torch.tensor([[0.0, 0.0], [1.0, 1.0]])

        result = apply_rejection_sampling(
            proximal_logprobs,
            old_logprobs,
            loss_mask,
            cu_seqlens=None,
            config=config,
        )

        # Seq 0: all-masked, loss_mask stays all-zero (no change)
        assert torch.all(result.loss_mask[0] == 0.0)
        # Seq 1: within bounds, kept
        assert torch.all(result.loss_mask[1] == 1.0)

    def test_shape_mismatch_raises(self):
        """Mismatched tensor shapes should raise ValueError."""
        config = RejectionSamplingConfig(level="token", metric="ratio", upper=5.0)
        proximal_logprobs = torch.tensor([[0.0, 1.0]])
        old_logprobs = torch.tensor([[0.0, 1.0, 2.0]])
        loss_mask = torch.ones(1, 2)

        with pytest.raises(ValueError, match="shape"):
            apply_rejection_sampling(
                proximal_logprobs,
                old_logprobs,
                loss_mask,
                cu_seqlens=None,
                config=config,
            )


class TestConfigValidation:
    """Tests for new config validation rules (Issues 1, 9)."""

    def test_lower_greater_than_upper_raises(self):
        """lower > upper should raise ValueError."""
        with pytest.raises(ValueError, match="lower.*cannot be greater than upper"):
            RejectionSamplingConfig(metric="ratio", lower=3.0, upper=2.0)

    def test_invalid_level_raises(self):
        """Invalid level should raise ValueError."""
        with pytest.raises(ValueError, match="level must be one of"):
            RejectionSamplingConfig(level="invalid")

    def test_invalid_action_raises(self):
        """Invalid action should raise ValueError."""
        with pytest.raises(ValueError, match="action must be one of"):
            RejectionSamplingConfig(action="invalid")

    def test_invalid_metric_raises(self):
        """Invalid metric should raise ValueError."""
        with pytest.raises(ValueError, match="metric must be one of"):
            RejectionSamplingConfig(metric="invalid")

    def test_invalid_agg_raises(self):
        """Invalid agg should raise ValueError."""
        with pytest.raises(ValueError, match="agg must be one of"):
            RejectionSamplingConfig(agg="invalid")
