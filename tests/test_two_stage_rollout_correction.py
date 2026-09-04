# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from areal.api.cli_args import PPOActorConfig, RejectionSamplingConfig
from areal.utils.functional import ppo_actor_loss_fn

# Three packed sequences of four tokens. pi_behave is uniform, so the ratio of
# each token is exp(proximal_logprobs).
#   seq 0: every ratio 1.0            -> geometric mean 1.0, inside the band
#   seq 1: every ratio 1.5            -> geometric mean 1.5, outside the band
#   seq 2: ratios 3.0, 1/3, 1.0, 1.0  -> geometric mean 1.0, inside the band,
#                                        but one token is far out on its own
CU_SEQLENS = torch.tensor([0, 4, 8, 12], dtype=torch.int32)
PROXIMAL = torch.log(
    torch.tensor([1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 1.5, 1.5, 3.0, 1.0 / 3.0, 1.0, 1.0])
)
BEHAVE = torch.zeros(12)
LOSS_MASK = torch.ones(12, dtype=torch.bool)

TOKEN_IS = RejectionSamplingConfig(
    level="token", action="clamp", metric="ratio", upper=2.0
)
GEO_RS = RejectionSamplingConfig(
    level="sequence",
    action="mask",
    metric="ratio",
    agg="mean",
    lower=0.98,
    upper=1.02,
)


def run_loss(rejection_sampling=None, importance_sampling=None):
    """Run the actor loss on the fixture above and return its stats."""
    logprobs = PROXIMAL.clone().requires_grad_(True)
    _, stat = ppo_actor_loss_fn(
        logprobs=logprobs,
        proximal_logprobs=PROXIMAL,
        old_logprobs=BEHAVE,
        advantages=torch.ones(12),
        eps_clip=0.2,
        loss_mask=LOSS_MASK,
        rejection_sampling=rejection_sampling,
        importance_sampling=importance_sampling,
        cu_seqlens=CU_SEQLENS,
    )
    return stat


class TestSingleStageUnchanged:
    """One configured stage keeps supplying both the weight and the mask."""

    def test_sequence_stage_alone_uses_the_geometric_mean_weight(self):
        stat = run_loss(rejection_sampling=GEO_RS)
        weight = stat["behave_imp_weight"]
        # Every surviving token carries its sequence's geometric mean, which is
        # 1.0 for both sequences that stay.
        assert weight[0].item() == pytest.approx(1.0)
        assert weight[8].item() == pytest.approx(1.0)

    def test_token_stage_alone_uses_the_clamped_token_weight(self):
        stat = run_loss(rejection_sampling=TOKEN_IS)
        weight = stat["behave_imp_weight"]
        assert weight[8].item() == pytest.approx(2.0)
        assert weight[9].item() == pytest.approx(1.0 / 3.0)

    def test_clamp_stage_alone_still_reports_the_clamped_fraction(self):
        """A single clamp stage keeps reporting what it clamped, not what it masked."""
        stat = run_loss(rejection_sampling=TOKEN_IS)
        assert stat["filtered_fraction"] == pytest.approx(1 / 12)

    def test_no_stage_reports_no_correction(self):
        stat = run_loss()
        assert "behave_imp_weight" not in stat
        assert "filtered_fraction" not in stat


class TestTwoStage:
    """rejection_sampling masks, importance_sampling weighs."""

    def test_weight_comes_from_the_importance_sampling_stage(self):
        stat = run_loss(rejection_sampling=GEO_RS, importance_sampling=TOKEN_IS)
        weight = stat["behave_imp_weight"]
        # Token-level, clamped at 2.0 -- not the sequence geometric mean.
        assert weight[8].item() == pytest.approx(2.0)
        assert weight[9].item() == pytest.approx(1.0 / 3.0)

    def test_mask_comes_from_the_rejection_sampling_stage(self):
        stat = run_loss(rejection_sampling=GEO_RS, importance_sampling=TOKEN_IS)
        kept = stat["behave_mask"]
        # Sequence 1 leaves the band and goes as a whole; the other two stay.
        assert kept[0:4].all()
        assert not kept[4:8].any()
        assert kept[8:12].all()

    def test_each_stage_reports_its_own_fraction(self):
        stat = run_loss(rejection_sampling=GEO_RS, importance_sampling=TOKEN_IS)
        # The mask stage drops one of three sequences ...
        assert stat["filtered_fraction"] == pytest.approx(4 / 12)
        # ... and the weight stage clamps one of twelve tokens.
        assert stat["is_filtered_fraction"] == pytest.approx(1 / 12)

    def test_two_stage_loss_and_backward_match_corrected_token_weights(self):
        """Apply token weights and sequence rejection to the actual loss and gradient."""
        logprobs = PROXIMAL.clone().requires_grad_(True)
        loss, _ = ppo_actor_loss_fn(
            logprobs=logprobs,
            proximal_logprobs=PROXIMAL,
            old_logprobs=BEHAVE,
            advantages=torch.ones_like(logprobs),
            eps_clip=0.2,
            loss_mask=LOSS_MASK,
            rejection_sampling=GEO_RS,
            importance_sampling=TOKEN_IS,
            cu_seqlens=CU_SEQLENS,
        )

        # theta == proximal and advantages == 1: PPO is unclipped. Geo-RS
        # removes sequence 1, Token-TIS caps 3 at 2, and the denominator stays 12.
        # Surviving weights sum to 4 + (2 + 1/3 + 1 + 1) = 25/3.
        torch.testing.assert_close(
            loss, loss.new_tensor(-25 / 36), rtol=1e-6, atol=1e-7
        )
        loss.backward()
        expected_gradient = (
            -logprobs.new_tensor([1, 1, 1, 1, 0, 0, 0, 0, 2, 1 / 3, 1, 1]) / 12
        )
        torch.testing.assert_close(
            logprobs.grad, expected_gradient, rtol=1e-6, atol=1e-7
        )

    def test_masking_stage_verdicts_are_intersected(self):
        """A weight stage that masks narrows the mask further."""
        masking_is = RejectionSamplingConfig(
            level="token", action="mask", metric="ratio", upper=2.0
        )
        stat = run_loss(rejection_sampling=GEO_RS, importance_sampling=masking_is)
        kept = stat["behave_mask"]
        assert not kept[4:8].any()  # dropped by the sequence stage
        assert not kept[8]  # ratio 3.0 > 2.0, dropped by the token stage
        assert kept[9:12].all()


class TestConfigValidation:
    def test_importance_sampling_requires_rejection_sampling(self):
        with pytest.raises(ValueError, match="importance_sampling requires"):
            PPOActorConfig(
                path="dummy",
                use_decoupled_loss=True,
                importance_sampling=TOKEN_IS,
            )

    def test_both_stages_are_accepted(self):
        config = PPOActorConfig(
            path="dummy",
            use_decoupled_loss=True,
            rejection_sampling=GEO_RS,
            importance_sampling=TOKEN_IS,
        )
        assert config.importance_sampling is TOKEN_IS
        assert config.rejection_sampling is GEO_RS

    def test_rejection_sampling_alone_is_still_accepted(self):
        config = PPOActorConfig(
            path="dummy",
            use_decoupled_loss=True,
            rejection_sampling=GEO_RS,
        )
        assert config.importance_sampling is None
