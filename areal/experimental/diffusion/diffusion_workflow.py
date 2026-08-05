# SPDX-License-Identifier: Apache-2.0
"""Rollout workflow for diffusion RL post-training.

This subclasses :class:`areal.api.workflow_api.RolloutWorkflow`, whose
``arun_episode`` returns a loose ``dict[str, Any] | None``. That looseness is
the key enabler for a non-invasive design: we pack the diffusion trajectory
(latents / timesteps / step log-probs / rewards / advantages) into the returned
dict without touching any base-class signature or the token-centric LLM path.

The returned dict is consumed by
:func:`areal.experimental.diffusion.diffusion_loss.diffusion_grpo_loss_fn`
during the training step.

Note: the ``engine`` argument is a
:class:`~areal.experimental.diffusion.diffusion_engine.DiffusionInferenceEngine`,
which deliberately does NOT inherit ``InferenceEngine`` (see that module's
docstring). Phase 1 therefore drives this workflow with a synchronous rollout
loop rather than the async ``engine.submit`` scheduler.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

from areal import workflow_context
from areal.api.workflow_api import RolloutWorkflow
from areal.experimental.diffusion.diffusion_api import DiffusionModelRequest
from areal.experimental.diffusion.diffusion_loss import compute_group_advantages
from areal.utils import logging, stats_tracker

if TYPE_CHECKING:
    from areal.experimental.diffusion.diffusion_engine import (
        DiffusionInferenceEngine,
    )

logger = logging.getLogger("DiffusionRolloutWorkflow")


class DiffusionRolloutWorkflow(RolloutWorkflow):
    """Generate a group of denoising trajectories and score them.

    For each prompt the workflow samples ``group_size`` trajectories, scores the
    final images with ``reward_fn``, and computes group-relative advantages. The
    resulting trajectory tensors are returned for the training step.

    Args:
        reward_fn: Callable ``(prompt, image, **kwargs) -> float``. In Phase 1
            this is invoked synchronously in the main process (see
            ``aesthetic_reward`` for the rationale).
        group_size: Number of samples per prompt (the GRPO group).
        num_inference_steps: Denoising steps per trajectory.
        guidance_scale: Classifier-free guidance scale.
        eta: DDIM stochasticity coefficient (must be > 0 for usable log-probs).
        height: Output image height.
        width: Output image width.
    """

    def __init__(
        self,
        reward_fn: Callable[..., float],
        group_size: int = 8,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        eta: float = 1.0,
        height: int = 512,
        width: int = 512,
    ):
        if eta <= 0:
            raise ValueError(
                f"eta must be > 0 for policy-gradient training (no log-prob "
                f"signal otherwise); got {eta}"
            )
        self.reward_fn = reward_fn
        self.group_size = group_size
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.eta = eta
        self.height = height
        self.width = width

    async def arun_episode(
        self, engine: DiffusionInferenceEngine, data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Sample a GRPO group for one prompt and return its trajectory tensors.

        Args:
            engine: The diffusion inference engine.
            data: A data item containing at least a ``"prompt"`` field.

        Returns:
            A dict carrying the batched trajectory:
            ``old_step_logprobs`` ``[group_size, num_steps]`` (rollout-time,
            detached -- the PPO "old" policy log-probs),
            ``advantages`` ``[group_size]``, ``rewards`` ``[group_size]``,
            ``latents`` (list of per-sample latent trajectories), ``timesteps``,
            and ``prompts``. The training step recomputes differentiable
            log-probs from ``latents``/``timesteps`` via
            ``engine.recompute_step_logprobs`` (the rollout-time values are
            detached and cannot backprop). Returns ``None`` if the prompt is
            missing.
        """
        prompt = data.get("prompt")
        if prompt is None:
            logger.warning("Data item missing 'prompt'; skipping episode.")
            return None

        responses = []
        for _ in range(self.group_size):
            req = DiffusionModelRequest(
                rid=str(uuid.uuid4()),
                prompt=prompt,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                eta=self.eta,
                height=self.height,
                width=self.width,
            )
            resp = await engine.agenerate(req)
            responses.append(resp)

        # ---- Reward scoring (synchronous, main process) ----
        rewards = torch.tensor(
            [float(self.reward_fn(prompt=prompt, image=r.image)) for r in responses],
            dtype=torch.float32,
        )

        # ---- Group-relative advantages ----
        advantages = compute_group_advantages(rewards, group_size=self.group_size)

        # ---- Stack per-step log-probs into [group_size, num_steps] ----
        step_logprobs = torch.stack(
            [torch.stack(r.step_logprobs).reshape(-1) for r in responses], dim=0
        )

        stats_tracker.get(workflow_context.stat_scope()).scalar(
            diffusion_reward_mean=float(rewards.mean()),
            diffusion_reward_std=float(rewards.std()),
        )

        return {
            "prompts": [prompt] * self.group_size,
            # Rollout-time (detached) log-probs: the PPO "old" policy. The
            # training step recomputes the differentiable "new" log-probs from
            # the recorded latents/timesteps via recompute_step_logprobs.
            "old_step_logprobs": step_logprobs,
            "advantages": advantages,
            "rewards": rewards,
            "latents": [r.latents for r in responses],
            "timesteps": [r.timesteps for r in responses],
        }
