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
   be computed from the final image). When ``old_logprobs`` is supplied we use a
   PPO-style clipped ratio surrogate ``min(r*A, clip(r)*A)`` with
   ``r = exp(new_logprob - old_logprob)``; when it is ``None`` we fall back to
   the plain REINFORCE surrogate ``- mean_t(logprob_t) * advantage``. Both
   mirror the ddpo-pytorch / TRL ``DDPOTrainer`` formulation. The PPO form is
   what enables off-policy reuse in Phase 2; in Phase 1 the single on-policy
   step has ``new == old`` so the ratio is ~1 and the two forms coincide.

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
    old_logprobs: torch.Tensor | None = None,
    clip_eps: float = 0.2,
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
        old_logprobs: Optional detached rollout-time per-step log-probs, same
            shape as ``step_logprobs``. When provided, a PPO-style clipped ratio
            surrogate is used; when ``None``, plain REINFORCE is used.
        clip_eps: PPO clip range (only used when ``old_logprobs`` is provided).

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
    if old_logprobs is not None and old_logprobs.shape != step_logprobs.shape:
        raise ValueError(
            f"old_logprobs must match step_logprobs shape; got "
            f"{tuple(old_logprobs.shape)} vs {tuple(step_logprobs.shape)}"
        )

    # Mean log-prob per trajectory over its denoising steps.
    if loss_mask is None:
        traj_logprob = step_logprobs.mean(dim=1)
    else:
        denom = loss_mask.sum(dim=1).clamp_min(1.0)
        traj_logprob = (step_logprobs * loss_mask).sum(dim=1) / denom

    if old_logprobs is None:
        # Plain REINFORCE surrogate: maximize advantage-weighted log-prob.
        loss = -(traj_logprob * advantages).mean()
        return loss

    # PPO-style clipped ratio surrogate. old_logprobs are the detached
    # rollout-time values; the ratio is per-trajectory (mean over steps).
    if loss_mask is None:
        old_traj_logprob = old_logprobs.mean(dim=1)
    else:
        denom = loss_mask.sum(dim=1).clamp_min(1.0)
        old_traj_logprob = (old_logprobs * loss_mask).sum(dim=1) / denom
    old_traj_logprob = old_traj_logprob.detach()

    ratio = torch.exp(traj_logprob - old_traj_logprob)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    loss = -torch.minimum(surr1, surr2).mean()
    return loss
