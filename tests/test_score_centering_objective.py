# SPDX-License-Identifier: Apache-2.0
"""The score-centering objectives, checked against closed-form token means."""

import pytest
import torch

from areal.trainer.ppo.actor import grpo_loss_fn
from areal.utils.functional.loss_aggregation import PolicyGradientReduction


def _reduction() -> PolicyGradientReduction:
    return PolicyGradientReduction("token-mean")


def _inputs(advantages: torch.Tensor) -> dict:
    logprobs = torch.tensor([[-1.0, -2.0]], dtype=torch.float32)
    return {
        "logprobs": logprobs.clone(),
        "prox_logp": logprobs.clone(),
        "advantages": advantages.clone(),
        "loss_mask": torch.ones(1, 2, dtype=torch.bool),
    }


def _call(logprobs, data, **kwargs):
    return grpo_loss_fn(
        logprobs,
        torch.zeros_like(logprobs),
        data,
        eps_clip=0.2,
        eps_clip_higher=None,
        c_clip=None,
        pg_reduction=_reduction(),
        **kwargs,
    )


def test_plan_a_is_reinforce_plus_detached_correction():
    data = _inputs(torch.tensor([[2.0, -1.0]], dtype=torch.float32))
    logprobs = data["logprobs"].clone().requires_grad_(True)
    correction = torch.tensor([[0.5, -0.5]], dtype=torch.float32)

    loss = _call(
        logprobs,
        data,
        use_score_centering=True,
        sc_correction=correction,
        sc_loss_mode="reinforce",
    )

    expected = (data["advantages"] * (correction - logprobs.detach())).mean()
    torch.testing.assert_close(loss, expected, rtol=1e-6, atol=1e-6)


def test_plan_b_adds_only_the_correction_to_ppo():
    data = _inputs(torch.tensor([[2.0, -1.0]], dtype=torch.float32))
    logprobs = data["logprobs"].clone().requires_grad_(True)
    correction = torch.tensor([[0.5, -0.5]], dtype=torch.float32)

    ppo_loss = _call(logprobs, data, use_score_centering=False)
    plan_b = _call(
        logprobs,
        data,
        use_score_centering=True,
        sc_correction=correction,
        sc_loss_mode="ppo_additive",
    )

    expected_delta = (data["advantages"] * correction).mean()
    torch.testing.assert_close(plan_b - ppo_loss, expected_delta, rtol=1e-6, atol=1e-6)


def test_reinforce_baseline_is_bare_reinforce():
    data = _inputs(torch.tensor([[2.0, -1.0]], dtype=torch.float32))
    logprobs = data["logprobs"].clone().requires_grad_(True)

    loss = _call(logprobs, data, use_score_centering=False, sc_reinforce_baseline=True)

    expected = (-data["advantages"] * logprobs.detach()).mean()
    torch.testing.assert_close(loss, expected, rtol=1e-6, atol=1e-6)


def test_plan_a_requires_the_engine_correction():
    data = _inputs(torch.tensor([[2.0, -1.0]], dtype=torch.float32))
    logprobs = data["logprobs"].clone().requires_grad_(True)

    with pytest.raises(ValueError, match="sc_correction"):
        _call(logprobs, data, use_score_centering=True, sc_correction=None)
