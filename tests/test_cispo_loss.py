# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from areal.api.cli_args import PPOActorConfig, RejectionSamplingConfig
from areal.utils.functional import apply_rejection_sampling, cispo_loss_fn

BANDS = [(0.2, 0.28), (1.0, 4.0)]


def _inputs():
    log_ratio = torch.tensor([-2.0, -0.5, 0.0, 0.3, 1.5, 0.1])
    logprobs = torch.tensor([-1.0, -2.0, -0.3, -1.2, -0.7, -2.5], requires_grad=True)
    proximal = logprobs.detach() - log_ratio  # logprobs - proximal == log_ratio
    advantages = torch.tensor([1.0, -2.0, 0.5, 3.0, -1.0, 2.0])
    loss_mask = torch.tensor([1, 1, 1, 1, 1, 0], dtype=torch.bool)
    return logprobs, proximal, advantages, loss_mask, log_ratio


@pytest.mark.parametrize("eps_clip,eps_clip_higher", BANDS)
def test_cispo_closed_form_value_and_clip_mask(eps_clip, eps_clip_higher):
    logprobs, proximal, advantages, loss_mask, log_ratio = _inputs()

    loss, stat = cispo_loss_fn(
        logprobs=logprobs,
        proximal_logprobs=proximal,
        advantages=advantages,
        eps_clip=eps_clip,
        eps_clip_higher=eps_clip_higher,
        loss_mask=loss_mask,
    )

    ratio = torch.exp(log_ratio)
    ratio_clipped = ratio.clamp(1.0 - eps_clip, 1.0 + eps_clip_higher)
    per_token = -ratio_clipped * advantages * logprobs.detach()
    expected = (
        torch.where(loss_mask, per_token, torch.zeros_like(per_token)).sum()
        / loss_mask.count_nonzero()
    )

    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(stat["importance_weight"], ratio)
    expected_clip = (ratio != ratio_clipped) & loss_mask
    assert torch.equal(stat["clip_mask"], expected_clip)
    assert not stat["dual_clip_mask"].any()


@pytest.mark.parametrize("eps_clip,eps_clip_higher", BANDS)
def test_cispo_gradient_routes_through_logprobs_only(eps_clip, eps_clip_higher):
    logprobs, proximal, advantages, loss_mask, log_ratio = _inputs()
    proximal = proximal.clone().requires_grad_(True)

    loss, _ = cispo_loss_fn(
        logprobs=logprobs,
        proximal_logprobs=proximal,
        advantages=advantages,
        eps_clip=eps_clip,
        eps_clip_higher=eps_clip_higher,
        loss_mask=loss_mask,
    )
    loss.backward()

    ratio = torch.exp(log_ratio)
    ratio_clipped = ratio.clamp(1.0 - eps_clip, 1.0 + eps_clip_higher)
    n = loss_mask.count_nonzero()
    expected_grad = torch.where(
        loss_mask, -ratio_clipped * advantages / n, torch.zeros_like(ratio)
    )
    torch.testing.assert_close(logprobs.grad, expected_grad)
    assert proximal.grad is None


def test_cispo_rejects_nonpositive_eps_clip_higher():
    logprobs, proximal, advantages, loss_mask, _ = _inputs()
    for bad in (None, 0.0, -1.0):
        with pytest.raises(ValueError, match="eps_clip_higher"):
            cispo_loss_fn(
                logprobs=logprobs,
                proximal_logprobs=proximal,
                advantages=advantages,
                eps_clip=1.0,
                eps_clip_higher=bad,
                loss_mask=loss_mask,
            )


def test_cispo_decoupled_applies_behave_imp_weight():
    eps_clip, eps_clip_higher = 1.0, 4.0
    logprobs, proximal, advantages, loss_mask, log_ratio = _inputs()
    behave_log_ratio = torch.tensor([0.2, -0.4, 0.0, 0.1, -0.3, 0.5])
    old_logprobs = proximal - behave_log_ratio  # proximal - old == behave_log_ratio
    rs = RejectionSamplingConfig(
        level="token", action="clamp", metric="ratio", upper=100.0
    )

    proximal_leaf = proximal.clone().requires_grad_(True)
    old_leaf = old_logprobs.clone().requires_grad_(True)

    loss, stat = cispo_loss_fn(
        logprobs=logprobs,
        proximal_logprobs=proximal_leaf,
        advantages=advantages,
        eps_clip=eps_clip,
        eps_clip_higher=eps_clip_higher,
        loss_mask=loss_mask,
        old_logprobs=old_leaf,
        rejection_sampling=rs,
    )

    ratio = torch.exp(log_ratio)
    ratio_clipped = ratio.clamp(1.0 - eps_clip, 1.0 + eps_clip_higher)
    behave_w = torch.exp(behave_log_ratio)
    n = loss_mask.count_nonzero()
    per_token = -ratio_clipped * advantages * logprobs.detach() * behave_w
    expected = torch.where(loss_mask, per_token, torch.zeros_like(per_token)).sum() / n
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(
        stat["behave_imp_weight"][loss_mask], behave_w[loss_mask]
    )
    assert "behave_mask" in stat and "behave_approx_kl" in stat

    loss.backward()
    expected_grad = torch.where(
        loss_mask, -ratio_clipped * advantages * behave_w / n, torch.zeros_like(ratio)
    )
    torch.testing.assert_close(logprobs.grad, expected_grad)
    assert proximal_leaf.grad is None
    assert old_leaf.grad is None


def test_cispo_respects_sequence_loss_aggregation():
    logprobs = torch.tensor([[-2.0, -2.0], [-4.0, -1.0]], requires_grad=True)
    proximal = logprobs.detach()
    advantages = torch.ones_like(logprobs)
    loss_mask = torch.tensor([[1, 1], [1, 0]], dtype=torch.bool)

    loss, _ = cispo_loss_fn(
        logprobs=logprobs,
        proximal_logprobs=proximal,
        advantages=advantages,
        eps_clip=1.0,
        eps_clip_higher=4.0,
        loss_mask=loss_mask,
        loss_aggregation="seq_mean",
    )

    torch.testing.assert_close(loss, torch.tensor(3.0))


def test_cispo_prompt_mean_accepts_partial_group_sizes():
    logprobs = torch.tensor(
        [[-2.0, -2.0], [-2.0, -2.0], [-10.0, -10.0]], requires_grad=True
    )
    proximal = logprobs.detach()
    advantages = torch.ones_like(logprobs)
    loss_mask = torch.ones_like(logprobs, dtype=torch.bool)

    loss, _ = cispo_loss_fn(
        logprobs=logprobs,
        proximal_logprobs=proximal,
        advantages=advantages,
        eps_clip=1.0,
        eps_clip_higher=4.0,
        loss_mask=loss_mask,
        loss_aggregation="prompt_mean",
        group_size=2,
        group_sizes=[2, 1],
    )

    torch.testing.assert_close(loss, torch.tensor(6.0))


def test_rejection_sampling_preserves_infinite_log_ratios():
    rs = RejectionSamplingConfig(
        level="token", action="mask", metric="ratio", upper=10.0
    )
    proximal = torch.tensor([0.0, -torch.inf, -torch.inf, torch.inf])
    old = torch.tensor([-torch.inf, -torch.inf, 0.0, -torch.inf])
    loss_mask = torch.ones(4, dtype=torch.bool)

    result = apply_rejection_sampling(
        proximal_logprobs=proximal,
        old_logprobs=old,
        loss_mask=loss_mask,
        cu_seqlens=None,
        config=rs,
    )

    assert torch.equal(result.loss_mask, torch.tensor([False, True, True, False]))


def test_cispo_config_validation():
    with pytest.raises(ValueError, match="eps_clip_higher"):
        PPOActorConfig(use_cispo_loss=True, eps_clip_higher=None)
    with pytest.raises(ValueError, match="mutually exclusive"):
        PPOActorConfig(use_cispo_loss=True, use_sapo_loss=True, eps_clip_higher=4.0)
    with pytest.raises(ValueError, match="importance_sampling_level"):
        PPOActorConfig(
            use_cispo_loss=True,
            eps_clip_higher=4.0,
            importance_sampling_level="sequence",
        )
    PPOActorConfig(
        use_cispo_loss=True,
        eps_clip_higher=4.0,
        loss_aggregation="seq_mean",
    )
    PPOActorConfig(use_cispo_loss=True, eps_clip=1.0, eps_clip_higher=4.0)
