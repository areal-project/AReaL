# SPDX-License-Identifier: Apache-2.0

"""Independent numerical, gradient, padding and configuration contracts for SC."""

import math

import pytest
import torch

from areal.utils.data import concat_padded_tensors
from areal.utils.score_centering import (
    _build_weight_fn,
    _TopKLogProbsFromLogits,
    gather_train_head_logp,
    score_centering_correction,
)


class TestTopKLogProbsFromLogits:
    """Compare real chunk boundaries and repeated IDs with native autograd."""

    @pytest.mark.parametrize("chunk_size", [1, 3, 7, 16])
    def test_chunks_match_native_values_and_weighted_gradients(self, chunk_size):
        """All rows, duplicate IDs and mixed upstream signs must be preserved."""
        values = torch.linspace(-2, 2, 35, dtype=torch.float64).reshape(7, 5)
        ids = torch.tensor(
            [
                [0, 0, 4],
                [1, 3, 1],
                [4, 2, 0],
                [2, 2, 2],
                [3, 1, 4],
                [0, 4, 0],
                [4, 4, 1],
            ]
        )
        upstream = torch.linspace(-1, 1, 21, dtype=torch.float64).reshape(7, 3)
        actual_logits = values.clone().requires_grad_()
        reference_logits = values.clone().requires_grad_()

        actual = _TopKLogProbsFromLogits.apply(actual_logits, ids, chunk_size)
        reference = reference_logits.log_softmax(-1).gather(-1, ids)
        actual.backward(upstream)
        reference.backward(upstream)

        torch.testing.assert_close(actual, reference, rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(
            actual_logits.grad, reference_logits.grad, rtol=1e-10, atol=1e-12
        )

    def test_bf16_values_and_gradients_match_promoted_reference(self):
        """Low-precision logits must use fp32 normalization and valid gradients."""
        values = torch.linspace(-2, 2, 35).reshape(7, 5).to(torch.bfloat16)
        ids = torch.tensor([[0, 0, 4]]).expand(7, -1)
        upstream = torch.linspace(-1, 1, 21).reshape(7, 3)
        actual_logits = values.clone().requires_grad_()
        reference_logits = values.float().requires_grad_()

        actual = _TopKLogProbsFromLogits.apply(actual_logits, ids, 3)
        reference = reference_logits.log_softmax(-1).gather(-1, ids)
        actual.backward(upstream)
        reference.backward(upstream)

        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            actual_logits.grad.float(),
            reference_logits.grad,
            rtol=2e-2,
            atol=2e-3,
        )


class TestScoreCenteringCorrection:
    """Pin SC values and stopped-coefficient gradients to hand-derived results."""

    @pytest.mark.parametrize(
        "mode,coefficients,expected_gradient",
        [
            ("none", (1 / 3, 1 / 3), [1 / 5, -1 / 5, 0]),
            ("tis", (1 / 15, 1 / 6), [1 / 50, -7 / 100, 1 / 20]),
            ("mis", (1 / 5, 1 / 2), [3 / 50, -21 / 100, 3 / 20]),
        ],
    )
    @pytest.mark.parametrize("head_ids", [[2, 0], [0, 2]])
    @pytest.mark.parametrize("padding_ids", [[], [1, 2]])
    def test_known_distributions_match_value_gradient_and_padding_invariance(
        self, mode, coefficients, expected_gradient, head_ids, padding_ids
    ):
        """p=(.2,.3,.5), q=(.4,.1,.5) have head masses .7/.9 and rho=1/3."""
        logits = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64).log()
        logits.requires_grad_()
        ids = torch.tensor(head_ids + padding_ids)
        sampler = [math.log({0: 0.4, 2: 0.5}[i]) for i in head_ids]
        sampler += [float("-inf")] * len(padding_ids)
        sampler_logps = torch.tensor(sampler, dtype=torch.float64).requires_grad_()
        trainer_logps = logits.log_softmax(-1).gather(-1, ids)
        # These coefficients are fixed fractions derived on paper, not from
        # score_centering_correction or its weighting helpers.
        expected = coefficients[0] * math.log(0.2) + coefficients[1] * math.log(0.5)

        correction, s_bar, q_mass, p_mass, rho = score_centering_correction(
            trainer_logps,
            sampler_logps,
            sc_mode=mode,
            sc_clip_low=0.5,
            sc_clip_high=2.0,
        )
        correction.backward()

        for actual, expected_value in [
            (correction, expected),
            (s_bar, 0.4 * math.log(0.2) + 0.5 * math.log(0.5)),
            (q_mass, 0.9),
            (p_mass, 0.7),
            (rho, 1 / 3),
        ]:
            torch.testing.assert_close(
                actual,
                torch.tensor(expected_value, dtype=torch.float64),
                rtol=1e-10,
                atol=1e-12,
            )
        torch.testing.assert_close(
            logits.grad,
            torch.tensor(expected_gradient, dtype=torch.float64),
            rtol=1e-10,
            atol=1e-12,
        )
        assert sampler_logps.grad is None

    @pytest.mark.parametrize("mode", ["none", "tis", "mis"])
    def test_on_policy_is_exact_noop_value_and_gradient(self, mode):
        """Equal distributions must yield zero correction and zero gradient."""
        logits = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64).log()
        logits.requires_grad_()
        head = logits.log_softmax(-1)[torch.tensor([2, 1])]
        sampler = head.detach().clone().requires_grad_()

        correction, _, q_mass, p_mass, rho = score_centering_correction(
            head, sampler, sc_mode=mode, sc_clip_low=0.5, sc_clip_high=2.0
        )
        correction.backward()

        torch.testing.assert_close(
            correction, torch.zeros_like(correction), rtol=0, atol=0
        )
        torch.testing.assert_close(
            logits.grad, torch.zeros_like(logits), rtol=0, atol=0
        )
        torch.testing.assert_close(q_mass, p_mass, rtol=0, atol=0)
        torch.testing.assert_close(rho, torch.ones_like(rho), rtol=0, atol=0)
        assert sampler.grad is None

    @pytest.mark.parametrize("mode", ["none", "tis", "mis"])
    def test_all_padding_has_zero_mass_value_and_gradient(self, mode):
        """Absent sampler slots must contribute neither probability nor gradient."""
        logits = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64).log()
        logits.requires_grad_()
        trainer = logits.log_softmax(-1)[torch.tensor([1, 1, 0])]
        sampler = torch.full((3,), float("-inf"), dtype=torch.float64)

        correction, s_bar, q_mass, p_mass, rho = score_centering_correction(
            trainer, sampler, sc_mode=mode, sc_clip_low=0.5, sc_clip_high=2.0
        )
        correction.backward()

        for value in (correction, s_bar, q_mass, p_mass):
            torch.testing.assert_close(value, torch.zeros_like(value), rtol=0, atol=0)
        torch.testing.assert_close(rho, torch.ones_like(rho), rtol=0, atol=0)
        torch.testing.assert_close(
            logits.grad, torch.zeros_like(logits), rtol=0, atol=0
        )


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("none", [1, 1, 1, 1, 1]),
        ("tis", [0.5, 0.5, 1, 2, 2]),
        ("mis", [0, 0.5, 1, 2, 0]),
    ],
)
def test_weighting_boundaries_match_literal_table(mode, expected):
    """Check below, on and above each boundary against independent constants."""
    ratios = torch.tensor([0.25, 0.5, 1, 2, 3], dtype=torch.float64)
    actual = _build_weight_fn(mode, 0.5, 2.0)(ratios)
    torch.testing.assert_close(
        actual, torch.tensor(expected, dtype=torch.float64), rtol=0, atol=0
    )


class TestGatherTrainHeadLogp:
    """Compare the public gather operation with native PyTorch probabilities."""

    @pytest.mark.parametrize("temperature", [0.7, 1.0, 1.3])
    def test_temperature_scaled_gather_matches_native_values_and_gradients(
        self, temperature
    ):
        """Temperature must scale both the selected log-probabilities and gradients."""
        values = torch.linspace(-2, 2, 30, dtype=torch.float64).reshape(2, 3, 5)
        ids = torch.tensor([[[0, 0], [1, 4], [3, 2]], [[4, 1], [2, 2], [1, 3]]])
        upstream = torch.linspace(-1, 1, 12, dtype=torch.float64).reshape(2, 3, 2)
        logits = values.clone().requires_grad_()
        reference_logits = values.clone().requires_grad_()

        actual = gather_train_head_logp(logits, ids, temperature=temperature)
        expected = (reference_logits / temperature).log_softmax(-1).gather(-1, ids)
        actual.backward(upstream)
        expected.backward(upstream)

        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(
            logits.grad, reference_logits.grad, rtol=1e-10, atol=1e-12
        )


class TestConcatPaddedTensorsPadding:
    """Check real sequence padding and preservation of unpadded values."""

    @pytest.mark.parametrize("key", ["logprobs", "ref_logprobs"])
    def test_normal_logprobs_keep_zero_padding(self, key):
        """Shared log-probability tensors use zero padding for masked arithmetic."""
        first = torch.tensor([[-1.0, -2.0, -3.0]])
        second = torch.tensor([[-4.0, -5.0, -6.0, -7.0, -8.0]])
        actual = concat_padded_tensors([{key: first}, {key: second}])[key]
        expected = torch.tensor(
            [[-1, -2, -3, 0, 0], [-4, -5, -6, -7, -8]], dtype=torch.float32
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_sampler_padding_has_zero_probability_mass(self):
        """Only absent sampler slots become -inf; valid values stay unchanged."""
        first = torch.tensor([[[0.5, 0.4]]]).log()
        second = torch.tensor([[[0.6, 0.3], [0.7, 0.2]]]).log()
        actual = concat_padded_tensors(
            [{"sampler_topk_logps": first}, {"sampler_topk_logps": second}]
        )["sampler_topk_logps"]
        expected_mass = torch.tensor([[[0.5, 0.4], [0, 0]], [[0.6, 0.3], [0.7, 0.2]]])
        torch.testing.assert_close(actual.exp(), expected_mass, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(actual[:1, :1], first, rtol=0, atol=0)
        torch.testing.assert_close(actual[1:], second, rtol=0, atol=0)
        assert torch.isneginf(actual[0, 1]).all()


class TestSCConfigValidation:
    """Verify PPOActorConfig validation for Score Centering fields."""

    def test_sc_loss_mode_invalid_raises(self):
        """Invalid sc_loss_mode should raise ValueError."""
        from areal.api.cli_args import PPOActorConfig

        with pytest.raises(ValueError, match="sc_loss_mode"):
            PPOActorConfig(path="dummy", sc_loss_mode="invalid")

    def test_sc_loss_mode_reinforce_default(self):
        """Default sc_loss_mode should be 'reinforce'."""
        from areal.api.cli_args import PPOActorConfig

        config = PPOActorConfig(path="dummy")
        assert config.sc_loss_mode == "reinforce"

    def test_sc_loss_mode_ppo_additive_valid(self):
        """sc_loss_mode='ppo_additive' should be accepted."""
        from areal.api.cli_args import PPOActorConfig

        config = PPOActorConfig(path="dummy", sc_loss_mode="ppo_additive")
        assert config.sc_loss_mode == "ppo_additive"

    def test_sc_mode_invalid_raises(self):
        """Invalid sc_mode should raise ValueError."""
        from areal.api.cli_args import PPOActorConfig

        with pytest.raises(ValueError, match="sc_mode"):
            PPOActorConfig(path="dummy", use_score_centering=True, sc_mode="invalid")

    def test_sc_reinforce_baseline_with_sc_on_raises(self):
        """sc_reinforce_baseline=True + use_score_centering=True should raise."""
        from areal.api.cli_args import PPOActorConfig

        with pytest.raises(ValueError, match="sc_reinforce_baseline"):
            PPOActorConfig(
                path="dummy",
                use_score_centering=True,
                sc_reinforce_baseline=True,
                sc_top_k=10,
            )

    def test_tis_mode_requires_finite_clips(self):
        """tis mode with infinite clips should raise (two-sided clamp)."""
        from areal.api.cli_args import PPOActorConfig

        with pytest.raises(ValueError, match="tis.*finite"):
            PPOActorConfig(
                path="dummy",
                use_score_centering=True,
                sc_mode="tis",
                sc_clip_high=float("inf"),
                sc_top_k=10,
            )

    def test_tis_mode_requires_finite_low(self):
        """tis mode with infinite sc_clip_low should raise."""
        from areal.api.cli_args import PPOActorConfig

        with pytest.raises(ValueError, match="tis.*finite"):
            PPOActorConfig(
                path="dummy",
                use_score_centering=True,
                sc_mode="tis",
                sc_clip_low=float("-inf"),
                sc_clip_high=5.0,
                sc_top_k=10,
            )

    def test_mis_mode_requires_finite_clips(self):
        """mis mode with infinite clips should raise."""
        from areal.api.cli_args import PPOActorConfig

        with pytest.raises(ValueError, match="mis.*finite"):
            PPOActorConfig(
                path="dummy",
                use_score_centering=True,
                sc_mode="mis",
                sc_clip_high=float("inf"),
                sc_top_k=10,
            )

    def test_sapo_and_reinforce_baseline_mutual_exclusion(self):
        """use_sapo_loss=True + sc_reinforce_baseline=True should raise."""
        from areal.api.cli_args import PPOActorConfig

        with pytest.raises(ValueError, match="sc_reinforce_baseline.*use_sapo_loss"):
            PPOActorConfig(
                path="dummy",
                use_sapo_loss=True,
                sc_reinforce_baseline=True,
            )

    def test_sc_off_does_not_validate_sc_mode(self):
        """SC-off with invalid sc_mode should NOT raise (inert config)."""
        from areal.api.cli_args import PPOActorConfig

        # sc_mode='invalid' but use_score_centering=False — should not raise
        config = PPOActorConfig(
            path="dummy",
            use_score_centering=False,
            sc_mode="invalid",
        )
        assert config.sc_mode == "invalid"
