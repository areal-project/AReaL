# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from areal.api.cli_args import PPOActorConfig
from areal.trainer.ppo.actor import grpo_loss_fn
from areal.utils import stats_tracker


def _make_dummy_input():
    bs, seqlen = 2, 4
    return {
        "logprobs": torch.tensor([[-0.5, -0.6, -0.7, -0.8], [-0.4, -0.3, -0.2, -0.1]]),
        "advantages": torch.tensor([[1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0]]),
        "loss_mask": torch.tensor([[0, 1, 1, 1], [0, 1, 1, 1]], dtype=torch.int32),
        "prox_logp": torch.tensor([[-0.5, -0.6, -0.7, -0.8], [-0.4, -0.3, -0.2, -0.1]]),
    }


def test_ppo_actor_config_entropy_coeff_validation():
    # Default is 0.0
    cfg = PPOActorConfig()
    assert cfg.entropy_coeff == 0.0

    # Positive value is accepted
    cfg_pos = PPOActorConfig(entropy_coeff=0.05)
    assert cfg_pos.entropy_coeff == 0.05

    # Negative value is rejected
    with pytest.raises(ValueError, match="entropy_coeff must be non-negative"):
        PPOActorConfig(entropy_coeff=-0.01)


def test_grpo_loss_fn_entropy_bonus_gradient():
    input_data = _make_dummy_input()
    logprobs = input_data["logprobs"].clone().requires_grad_(True)
    entropy = torch.tensor([[1.2, 1.5, 1.8, 1.1], [0.9, 1.0, 1.3, 1.4]], requires_grad=True)

    loss_mask = input_data["loss_mask"].bool()
    valid_tokens = loss_mask.sum().item()
    entropy_coeff = 0.02

    loss = grpo_loss_fn(
        logprobs=logprobs,
        entropy=entropy,
        input_data=input_data,
        eps_clip=0.2,
        eps_clip_higher=None,
        c_clip=None,
        entropy_coeff=entropy_coeff,
    )

    loss.backward()

    # Verify entropy has non-zero gradients
    assert entropy.grad is not None
    # For valid tokens, grad of loss w.r.t entropy is: - entropy_coeff / valid_tokens
    expected_grad = torch.where(
        loss_mask,
        torch.tensor(-entropy_coeff / valid_tokens, dtype=torch.float32),
        torch.tensor(0.0, dtype=torch.float32),
    )
    torch.testing.assert_close(entropy.grad, expected_grad, rtol=1e-5, atol=1e-5)


def test_grpo_loss_fn_entropy_zero_is_detached():
    input_data = _make_dummy_input()
    logprobs = input_data["logprobs"].clone().requires_grad_(True)
    entropy = torch.tensor([[1.2, 1.5, 1.8, 1.1], [0.9, 1.0, 1.3, 1.4]], requires_grad=True)

    loss = grpo_loss_fn(
        logprobs=logprobs,
        entropy=entropy,
        input_data=input_data,
        eps_clip=0.2,
        eps_clip_higher=None,
        c_clip=None,
        entropy_coeff=0.0,
    )

    loss.backward()

    # When entropy_coeff=0.0, entropy is detached and receives no gradients
    assert entropy.grad is None


def test_grpo_loss_fn_entropy_bonus_modifies_loss():
    input_data = _make_dummy_input()
    logprobs = input_data["logprobs"].clone()
    entropy = torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]])

    loss_zero = grpo_loss_fn(
        logprobs=logprobs,
        entropy=entropy,
        input_data=input_data,
        eps_clip=0.2,
        eps_clip_higher=None,
        c_clip=None,
        entropy_coeff=0.0,
    )

    coeff = 0.05
    loss_with_bonus = grpo_loss_fn(
        logprobs=logprobs,
        entropy=entropy,
        input_data=input_data,
        eps_clip=0.2,
        eps_clip_higher=None,
        c_clip=None,
        entropy_coeff=coeff,
    )

    # Valid tokens are index 1, 2, 3 in each row
    # row 0: 2.0, 3.0, 4.0; row 1: 3.0, 4.0, 5.0 -> sum = 21.0, valid_tokens = 6 -> mean = 3.5
    # entropy_loss = -3.5
    # diff should be coeff * (-3.5) = -0.175
    expected_diff = coeff * (-3.5)
    actual_diff = (loss_with_bonus - loss_zero).item()
    assert pytest.approx(actual_diff, abs=1e-5) == expected_diff
