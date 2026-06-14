# SPDX-License-Identifier: Apache-2.0
"""GRPO advantage and step-level policy loss for diffusion RL.

Two pieces live here:

1. ``compute_group_advantages`` -- group-relative advantage normalization. This
   is modality-agnostic (it operates on scalar rewards), so it reuses the same
   normalization primitive (:func:`areal.utils.functional.masked_normalization`)
   that the LLM GRPO path uses.

2. ``diffusion_grpo_loss_fn`` -- the diffusion-specific policy loss. Unlike the
   LLM case (token-level ``logprob * advantage``), the diffusion trajectory is a
   sequence of *denoising steps*; each UNet prediction yields one step log-prob.
   The whole trajectory shares a single terminal advantage (the reward can only
   be computed from the final image), so the loss is
   ``- mean_t( logprob_t ) * advantage`` averaged over the batch. This mirrors
   the ddpo-pytorch / TRL ``DDPOTrainer`` formulation.

The loss callable is shaped to be consumed by
``TrainEngine.train_batch(input_, loss_fn, loss_weight_fn)``: it accepts the
model output and the input dict, and returns a scalar loss.
"""

from __future__ import annotations

import torch

from areal.utils import logging
from areal.utils.functional import masked_normalization

logger = logging.getLogger("DiffusionGRPOLoss")


def compute_group_advantages(
    rewards: torch.Tensor,
    group_size: int,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Group-relative advantage: normalize rewards within each prompt group.

    Args:
        rewards: 1-D tensor of scalar rewards, shape ``[num_prompts * group_size]``,
            laid out group-contiguously (all samples of prompt 0, then prompt 1, ...).
        group_size: Number of samples per prompt group.
        eps: Numerical stabilizer for the per-group std.

    Returns:
        Advantages with the same shape as ``rewards``, normalized within each
        group (zero mean, unit variance per group).
    """
    if rewards.ndim != 1:
        raise ValueError(f"rewards must be 1-D, got shape {tuple(rewards.shape)}")
    if rewards.numel() % group_size != 0:
        raise ValueError(
            f"rewards length {rewards.numel()} not divisible by group_size {group_size}"
        )

    grouped = rewards.view(-1, group_size)
    # Per-group normalization; disable distributed all-reduce because grouping is
    # local to the rollout buffer in Phase 1 (single process). ``dim`` must be a
    # tuple: masked_normalization iterates over it.
    normalized = masked_normalization(grouped, dim=(1,), eps=eps, all_reduce=False)
    return normalized.reshape(-1)


def diffusion_grpo_loss_fn(
    step_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Step-level GRPO policy loss for a batch of diffusion trajectories.

    Args:
        step_logprobs: Per-step log-probs, shape ``[batch, num_steps]``. These
            must be *differentiable* (recomputed under grad during training,
            not the detached rollout-time values).
        advantages: Per-trajectory advantages, shape ``[batch]``. Broadcast
            across all steps of a trajectory.
        loss_mask: Optional ``[batch, num_steps]`` mask for variable-length
            trajectories. ``None`` means all steps are valid.

    Returns:
        Scalar policy loss (to be minimized).
    """
    if step_logprobs.ndim != 2:
        raise ValueError(
            f"step_logprobs must be [batch, num_steps], got "
            f"{tuple(step_logprobs.shape)}"
        )
    if advantages.ndim != 1 or advantages.shape[0] != step_logprobs.shape[0]:
        raise ValueError(
            f"advantages must be [batch] matching step_logprobs batch dim; "
            f"got advantages {tuple(advantages.shape)} vs "
            f"step_logprobs {tuple(step_logprobs.shape)}"
        )

    # Mean log-prob per trajectory over its denoising steps.
    if loss_mask is None:
        traj_logprob = step_logprobs.mean(dim=1)
    else:
        denom = loss_mask.sum(dim=1).clamp_min(1.0)
        traj_logprob = (step_logprobs * loss_mask).sum(dim=1) / denom

    # Policy gradient surrogate: maximize advantage-weighted log-prob.
    loss = -(traj_logprob * advantages).mean()
    return loss
