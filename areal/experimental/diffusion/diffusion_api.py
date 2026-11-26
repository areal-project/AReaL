# SPDX-License-Identifier: Apache-2.0
"""Data contracts for diffusion RL post-training.

These dataclasses are intentionally kept separate from the token-centric
``ModelRequest`` / ``ModelResponse`` in :mod:`areal.api.io_struct`. Diffusion
trajectories (latents, timesteps, per-step log-probs) do not map naturally onto
token sequences, so polluting the LLM path with diffusion-specific fields would
break the existing abstractions. Phase 1 therefore defines parallel structs that
live entirely under ``areal.experimental.diffusion``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    from PIL.Image import Image


@dataclass
class DiffusionModelRequest:
    """A single text-to-image generation request for RL rollout.

    Mirrors the role of :class:`areal.api.io_struct.ModelRequest` but carries
    diffusion sampling parameters instead of token / generation hyperparameters.

    Args:
        rid: Unique request id (used for tracing / grouping).
        prompt: Text prompt fed to the text encoder.
        num_inference_steps: Number of denoising steps. Each step produces one
            recorded log-prob, so this also defines the trajectory length.
        guidance_scale: Classifier-free guidance scale.
        eta: DDIM stochasticity coefficient. ``eta > 0`` turns sampling into a
            stochastic process whose transition kernel has a tractable Gaussian
            log-density -- this is what makes per-step log-probs (and therefore
            policy-gradient training) possible. ``eta == 0`` is deterministic
            DDIM and yields no usable log-prob signal.
        height: Output image height in pixels.
        width: Output image width in pixels.
        seed: Optional RNG seed for reproducible sampling.
        metadata: Free-form per-request metadata propagated to the response.
    """

    rid: str
    prompt: str
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    eta: float = 1.0
    height: int = 512
    width: int = 512
    seed: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.num_inference_steps <= 0:
            raise ValueError(
                f"num_inference_steps must be positive, got {self.num_inference_steps}"
            )
        if not 0.0 <= self.eta <= 1.0:
            raise ValueError(f"eta must be in [0, 1], got {self.eta}")
        if self.height <= 0 or self.width <= 0:
            raise ValueError(
                f"height/width must be positive, got {self.height}x{self.width}"
            )


@dataclass
class DiffusionModelResponse:
    """The denoising trajectory produced for one :class:`DiffusionModelRequest`.

    Mirrors the role of :class:`areal.api.io_struct.ModelResponse`. The fields
    ``latents`` / ``timesteps`` / ``step_logprobs`` together form the
    trajectory consumed by the step-level GRPO loss in
    :mod:`areal.experimental.diffusion.diffusion_loss`.

    Args:
        prompt: The prompt that generated this trajectory.
        image: Final decoded image (PIL image), used for reward scoring.
        latents: Per-step latent tensors along the denoising trajectory. Length
            equals ``num_steps + 1`` (initial noise plus one latent per step).
        timesteps: Scheduler timesteps visited during denoising. Length equals
            ``num_steps``.
        step_logprobs: Per-step log-probability of the sampled transition under
            the policy (the DDIM-with-eta Gaussian kernel). Length equals
            ``num_steps``. Shape per element: scalar (summed over latent dims).
        version: Weight version that produced this trajectory (for async
            staleness control in Phase 2).
        latency: Wall-clock sampling latency in seconds.
        metadata: Metadata propagated from the originating request.
    """

    prompt: str
    image: Image
    latents: list[torch.Tensor]
    timesteps: list[int]
    step_logprobs: list[torch.Tensor]
    version: int = -1
    latency: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_steps(self) -> int:
        """Number of denoising steps in this trajectory."""
        return len(self.step_logprobs)
