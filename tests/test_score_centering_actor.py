# SPDX-License-Identifier: Apache-2.0

"""Actor objectives and gradients checked against independent analytic results."""

import math

import pytest
import torch

from areal.trainer.ppo import actor as actor_module
from areal.utils.functional.loss_aggregation import PolicyGradientReduction
from areal.utils.stats_tracker import DistributedStatsTracker


@pytest.mark.parametrize(
    "advantage,old_probability,ppo_value,ppo_gradient",
    [
        (2.0, 0.25, -2.4, [0, 0, 0]),
        (2.0, 0.5, -2.0, [0.4, 0.6, -1.0]),
        (2.0, 0.8, -1.25, [0.25, 0.375, -0.625]),
        (-2.0, 0.25, 4.0, [-0.8, -1.2, 2.0]),
        (-2.0, 0.5, 2.0, [-0.4, -0.6, 1.0]),
        (-2.0, 0.8, 1.6, [0, 0, 0]),
    ],
)
@pytest.mark.parametrize(
    "use_sc,reinforce_baseline,loss_mode,metric_code",
    [
        (False, False, "reinforce", 0),
        (True, False, "reinforce", 1),
        (False, True, "reinforce", 2),
        (True, False, "ppo_additive", 3),
    ],
)
def test_actor_objective_and_logits_gradient_match_analytic_reference(
    monkeypatch,
    advantage,
    old_probability,
    ppo_value,
    ppo_gradient,
    use_sc,
    reinforce_baseline,
    loss_mode,
    metric_code,
):
    """Check both advantage signs, active PPO clipping, and all four loss paths."""
    logits = torch.tensor([[[0.2, 0.3, 0.5]]]).log().requires_grad_()
    logprobs = logits.log_softmax(-1)[..., 2]
    # H=[2,0] is the sampler's true top-2 for q=(.4,.1,.5).
    sampler = torch.tensor([[[0.5, 0.4]]]).log().requires_grad_()
    advantage_tensor = torch.tensor([[advantage]])
    if reinforce_baseline or (use_sc and loss_mode == "reinforce"):
        advantage_tensor.requires_grad_()
    data = {
        "logprobs": torch.tensor([[math.log(old_probability)]]),
        "prox_logp": torch.tensor([[math.log(old_probability)]]),
        "advantages": advantage_tensor,
        "loss_mask": torch.ones(1, 1, dtype=torch.bool),
    }
    if use_sc:
        data.update(
            logits=logits,
            sampler_topk_logps=sampler,
            sampler_topk_token_ids=torch.tensor([[[2, 0]]]),
        )
    tracker = DistributedStatsTracker()
    monkeypatch.setattr(actor_module, "stats_tracker", tracker)

    loss = actor_module.grpo_loss_fn(
        logprobs,
        torch.zeros_like(logprobs),
        data,
        eps_clip=0.2,
        eps_clip_higher=None,
        c_clip=None,
        use_score_centering=use_sc,
        sc_reinforce_baseline=reinforce_baseline,
        sc_loss_mode=loss_mode,
    )
    loss.backward()

    if reinforce_baseline or (use_sc and loss_mode == "reinforce"):
        expected_value = -advantage * math.log(0.5)
        expected_gradient = advantage * torch.tensor([0.2, 0.3, -0.5])
        assert advantage_tensor.grad is None
    else:
        expected_value = ppo_value
        expected_gradient = torch.tensor(ppo_gradient, dtype=torch.float32)
    if use_sc:
        # Closed-form SC constants, derived independently of production helpers.
        expected_value += advantage * math.log(0.1) / 3
        expected_gradient += advantage * torch.tensor([0.2, -0.2, 0])
    torch.testing.assert_close(loss, torch.tensor(expected_value), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        logits.grad[0, 0], expected_gradient, rtol=1e-5, atol=1e-6
    )
    assert sampler.grad is None
    metrics = tracker.export()
    assert metrics["loss_mode"] == metric_code
    assert metrics["sc_active"] == int(use_sc)


@pytest.mark.parametrize(
    "mode,expected_scale,coefficients",
    [
        ("token-mean", 1 / 3, [[2 / 3, 0, 0], [1 / 3, -2 / 3, 0]]),
        ("seq-mean", 3 / 4, [[1, 0, 0], [1 / 4, -1 / 2, 0]]),
    ],
)
@pytest.mark.parametrize("masked_probability", [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])
def test_masked_tokens_and_unequal_lengths_preserve_expected_loss_and_gradient(
    monkeypatch, mode, expected_scale, coefficients, masked_probability
):
    """Masked values cannot affect the independently weighted valid-token result."""
    mask = torch.tensor([[True, False, False], [True, True, False]])
    probabilities = torch.tensor([0.2, 0.3, 0.5]).expand(2, 3, 3).clone()
    probabilities[~mask] = torch.tensor(masked_probability)
    logits = probabilities.log().requires_grad_()
    logprobs = logits.log_softmax(-1)[..., 2]
    sampler = torch.tensor([0.5, 0.4]).expand(2, 3, 2).clone().log().requires_grad_()
    data = {
        "logprobs": logprobs.detach(),
        "prox_logp": logprobs.detach(),
        "advantages": torch.tensor([[2.0, 77.0, -99.0], [1.0, -2.0, 88.0]]),
        "loss_mask": mask,
        "logits": logits,
        "sampler_topk_logps": sampler,
        "sampler_topk_token_ids": torch.tensor([2, 0]).expand(2, 3, 2),
    }
    monkeypatch.setattr(actor_module, "stats_tracker", DistributedStatsTracker())

    loss = actor_module.grpo_loss_fn(
        logprobs,
        torch.zeros_like(logprobs),
        data,
        eps_clip=0.2,
        eps_clip_higher=None,
        c_clip=None,
        use_score_centering=True,
        pg_reduction=PolicyGradientReduction(mode=mode),
    )
    loss.backward()

    unit_loss = -math.log(0.5) + math.log(0.1) / 3
    expected_gradient = torch.tensor(coefficients).unsqueeze(-1) * torch.tensor(
        [0.4, 0.1, -0.5]
    )
    torch.testing.assert_close(
        loss, torch.tensor(expected_scale * unit_loss), rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(logits.grad, expected_gradient, rtol=1e-5, atol=1e-6)
    assert sampler.grad is None


@pytest.mark.parametrize(
    "missing", ["logits", "sampler_topk_logps", "sampler_topk_token_ids"]
)
def test_enabled_sc_requires_all_training_inputs(monkeypatch, missing):
    """An incomplete SC request must fail instead of optimizing a fallback loss."""
    logits = torch.tensor([[[0.2, 0.3, 0.5]]]).log().requires_grad_()
    logprobs = logits.log_softmax(-1)[..., 2]
    data = {
        "logprobs": logprobs.detach(),
        "prox_logp": logprobs.detach(),
        "advantages": torch.ones(1, 1),
        "loss_mask": torch.ones(1, 1, dtype=torch.bool),
        "logits": logits,
        "sampler_topk_logps": torch.tensor([[[0.5, 0.4]]]).log(),
        "sampler_topk_token_ids": torch.tensor([[[2, 0]]]),
    }
    data.pop(missing)
    monkeypatch.setattr(actor_module, "stats_tracker", DistributedStatsTracker())

    with pytest.raises(RuntimeError, match=f"{missing}=MISSING"):
        actor_module.grpo_loss_fn(
            logprobs,
            torch.zeros_like(logprobs),
            data,
            eps_clip=0.2,
            eps_clip_higher=None,
            c_clip=None,
            use_score_centering=True,
        )
