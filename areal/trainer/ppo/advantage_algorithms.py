# SPDX-License-Identifier: Apache-2.0

"""Temporal estimators and local process shaping for PPO advantages."""

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from areal.trainer.ppo.gae import (
    _compute_token_level_gae,
    _compute_turn_level_gae,
    compute_gae_lambda,
)


@dataclass(frozen=True)
class TemporalInputs:
    """Next-token-aligned reward channels and explicit trajectory bootstraps.

    ``rewards`` contains only outcome rewards. ``process_rewards`` contains
    process rewards folded onto turn ends, while ``kl_rewards`` contains token
    KL penalties. Keeping channels separate preserves the established addition
    order and prevents turn GAE from including KL in critic targets.

    Callable lambda functions receive lengths in the estimator's timestep unit.
    """

    rewards: torch.Tensor
    values: torch.Tensor
    loss_mask: torch.Tensor
    seq_no_eos_mask: torch.Tensor
    bootstrap_values: torch.Tensor
    discount: float
    gae_lambda: float | torch.Tensor | Callable
    gae_lambda_kwargs: Mapping[str, Any] | None = None
    turn_ids: torch.Tensor | None = None
    kl_rewards: torch.Tensor | None = None
    process_rewards: torch.Tensor | None = None


@dataclass(frozen=True)
class TemporalResult:
    advantages: torch.Tensor
    returns: torch.Tensor
    total_rewards: torch.Tensor


def token_gae(inputs: TemporalInputs) -> TemporalResult:
    """Estimate token GAE, including KL in both actor and critic targets."""
    rewards = inputs.rewards
    if inputs.kl_rewards is not None:
        rewards = inputs.kl_rewards + rewards
    if inputs.process_rewards is not None:
        rewards = rewards + inputs.process_rewards
    advantages, returns = _compute_token_level_gae(
        rewards=rewards,
        values=inputs.values,
        loss_mask=inputs.loss_mask,
        seq_no_eos_mask=inputs.seq_no_eos_mask,
        bootstrap_values=inputs.bootstrap_values,
        discount=inputs.discount,
        gae_lambda=compute_gae_lambda(
            inputs.gae_lambda,
            inputs.loss_mask,
            inputs.turn_ids,
            timestep_unit="token",
            kwargs=inputs.gae_lambda_kwargs,
        ),
    )
    return TemporalResult(advantages=advantages, returns=returns, total_rewards=rewards)


def turn_gae(inputs: TemporalInputs) -> TemporalResult:
    """Estimate turn GAE, applying token KL only to actor advantages."""
    if inputs.turn_ids is None:
        raise ValueError("turn_gae requires rollout data to include 'turn_ids'")
    rewards = inputs.rewards
    if inputs.process_rewards is not None:
        rewards = rewards + inputs.process_rewards
    advantages, returns = _compute_turn_level_gae(
        rewards=rewards,
        values=inputs.values,
        loss_mask=inputs.loss_mask,
        turn_ids=inputs.turn_ids,
        seq_no_eos_mask=inputs.seq_no_eos_mask,
        bootstrap_values=inputs.bootstrap_values,
        discount=inputs.discount,
        gae_lambda=compute_gae_lambda(
            inputs.gae_lambda,
            inputs.loss_mask,
            inputs.turn_ids,
            timestep_unit="turn",
            kwargs=inputs.gae_lambda_kwargs,
        ),
    )
    if inputs.kl_rewards is not None:
        advantages = advantages + inputs.kl_rewards
        rewards = rewards + inputs.kl_rewards
    return TemporalResult(advantages=advantages, returns=returns, total_rewards=rewards)


def _shape_advantages_with_gvpo(
    advantages: torch.Tensor,
    token_advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    negative_scale: float,
    zero_penalty: float,
    zero_eps: float,
) -> torch.Tensor:
    """Shape outcome advantages where a negative process signal marks failure."""
    for name, value in (
        ("negative_scale", negative_scale),
        ("zero_penalty", zero_penalty),
        ("zero_eps", zero_eps),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"GVPO {name} must be finite and non-negative, got {value}"
            )

    failed_mask = (token_advantages < 0) & loss_mask.bool()
    negative_mask = failed_mask & (advantages < -zero_eps)
    zero_mask = failed_mask & (advantages.abs() <= zero_eps)
    positive_mask = failed_mask & (advantages > zero_eps)

    shaped = torch.where(
        negative_mask,
        advantages * (1.0 + negative_scale),
        advantages,
    )
    shaped = torch.where(zero_mask, torch.full_like(shaped, -zero_penalty), shaped)
    return shaped.masked_fill(positive_mask, 0.0)


def _shape_advantages_with_process_weighting(
    advantages: torch.Tensor,
    process_rewards: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Combine outcome advantages with process rewards in ``[0, 1]``."""
    active_mask = loss_mask.bool()
    rewards_in_range = (process_rewards >= 0) & (process_rewards <= 1)
    torch._assert_async(
        torch.all(rewards_in_range | ~active_mask),
        "Process-weighted advantage shaping requires process rewards in [0, 1]",
    )

    shaped = torch.where(
        advantages >= 0,
        advantages * process_rewards,
        torch.where(process_rewards > 0, process_rewards, advantages),
    )
    return torch.where(active_mask, shaped, advantages)


@dataclass(frozen=True)
class ProcessAdvantageInputs:
    """Normalized advantages and next-token-aligned direct process rewards."""

    advantages: torch.Tensor
    token_local_rewards: torch.Tensor | None
    loss_mask: torch.Tensor
    token_rewards_as_adv: bool = True
    mask_no_eos_with_zero: bool = False


def additive(inputs: ProcessAdvantageInputs) -> torch.Tensor:
    if inputs.token_local_rewards is None:
        return inputs.advantages
    return inputs.advantages + inputs.token_local_rewards


def gvpo(
    inputs: ProcessAdvantageInputs,
    *,
    negative_scale: float = 0.2,
    zero_penalty: float = 0.4,
    zero_eps: float = 1e-6,
) -> torch.Tensor:
    if not inputs.token_rewards_as_adv:
        raise ValueError("'gvpo' advantage shaping requires token_rewards_as_adv=True")
    for name, value in (
        ("negative_scale", negative_scale),
        ("zero_penalty", zero_penalty),
        ("zero_eps", zero_eps),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"GVPO {name} must be finite and non-negative, got {value}"
            )
    if inputs.token_local_rewards is None:
        return inputs.advantages
    return _shape_advantages_with_gvpo(
        inputs.advantages,
        inputs.token_local_rewards,
        inputs.loss_mask,
        negative_scale=negative_scale,
        zero_penalty=zero_penalty,
        zero_eps=zero_eps,
    )


def process_weighted(inputs: ProcessAdvantageInputs) -> torch.Tensor:
    if not inputs.token_rewards_as_adv:
        raise ValueError(
            "'process_weighted' advantage shaping requires token_rewards_as_adv=True"
        )
    if inputs.mask_no_eos_with_zero:
        raise ValueError(
            "'process_weighted' advantage shaping is incompatible with mask_no_eos_with_zero=True"
        )
    if inputs.token_local_rewards is None:
        return inputs.advantages
    return _shape_advantages_with_process_weighting(
        inputs.advantages, inputs.token_local_rewards, inputs.loss_mask
    )
