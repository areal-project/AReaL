# SPDX-License-Identifier: Apache-2.0
"""Inference engine for diffusion RL rollout.

Design note -- why this does NOT subclass ``InferenceEngine``:

``areal.api.engine_api.InferenceEngine.agenerate`` has the concrete signature
``agenerate(req: ModelRequest) -> ModelResponse``. If we subclassed it and
narrowed the parameter to ``DiffusionModelRequest``, any caller typed against
``InferenceEngine`` would receive an unexpected type -- a Liskov substitution
violation. We therefore implement a *parallel* engine with its own
``agenerate(req: DiffusionModelRequest) -> DiffusionModelResponse``.

The trade-off: this engine cannot be plugged into the async
``engine.submit(workflow=...)`` scheduler (which is typed against
``InferenceEngine``). Phase 1 drives it with a synchronous loop instead; Phase 2
will introduce an adapter to bridge into the async path. This is a known,
deliberately surfaced piece of technical debt.

Heavy optional dependencies (``diffusers``, ``torch``) are imported lazily
inside methods so that importing this module never drags in the diffusion stack.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

from areal.experimental.diffusion.diffusion_api import (
    DiffusionModelRequest,
    DiffusionModelResponse,
)
from areal.utils import logging

if TYPE_CHECKING:
    import torch

logger = logging.getLogger("DiffusionInferenceEngine")


class DiffusionInferenceEngine:
    """Wraps a diffusers Stable Diffusion pipeline for RL rollout.

    Phase 1 targets single-GPU SD1.5 + LoRA. The engine performs DDIM-with-eta
    sampling and records the per-step log-probability of each sampled transition,
    which downstream code turns into a policy-gradient signal.

    Args:
        model_path: HuggingFace model id or local path to the SD1.5 checkpoint.
        dtype: Torch dtype for the pipeline (``"float16"`` / ``"float32"`` /
            ``"bfloat16"``).
        device: Device to place the pipeline on (e.g. ``"cuda"``).
        lora_rank: Rank of the LoRA adapter injected into the UNet. Set to 0 to
            train the full UNet (not recommended for Phase 1).
    """

    def __init__(
        self,
        model_path: str,
        dtype: str = "float16",
        device: str = "cuda",
        lora_rank: int = 8,
    ):
        self.model_path = model_path
        self.dtype_str = dtype
        self.device = device
        self.lora_rank = lora_rank

        self.pipe = None
        self.unet = None
        self.scheduler = None
        self._version = 0

    def initialize(self):
        """Load the SD pipeline and inject a trainable LoRA adapter into the UNet."""
        import torch
        from diffusers import DDIMScheduler, StableDiffusionPipeline

        dtype = getattr(torch, self.dtype_str)
        pipe = StableDiffusionPipeline.from_pretrained(
            self.model_path, torch_dtype=dtype
        )
        # DDIM scheduler is required for the eta-parameterized stochastic sampler.
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(self.device)

        # Freeze everything except the LoRA adapter on the UNet.
        pipe.vae.requires_grad_(False)
        pipe.text_encoder.requires_grad_(False)
        pipe.unet.requires_grad_(False)

        if self.lora_rank > 0:
            from peft import LoraConfig

            lora_config = LoraConfig(
                r=self.lora_rank,
                lora_alpha=self.lora_rank,
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )
            pipe.unet.add_adapter(lora_config)
            # Re-enable grad only for LoRA params.
            for name, param in pipe.unet.named_parameters():
                param.requires_grad_("lora" in name)

        self.pipe = pipe
        self.unet = pipe.unet
        self.scheduler = pipe.scheduler
        logger.info(
            f"Initialized diffusion engine: model={self.model_path}, "
            f"dtype={self.dtype_str}, lora_rank={self.lora_rank}"
        )

    async def agenerate(self, req: DiffusionModelRequest) -> DiffusionModelResponse:
        """Sample one denoising trajectory and record per-step log-probs.

        The method is ``async`` to mirror the inference-engine contract, but
        Phase 1 runs the denoising loop synchronously (diffusers is not async).

        Args:
            req: The generation request describing prompt and sampling params.

        Returns:
            A :class:`DiffusionModelResponse` carrying the final image plus the
            full latent / timestep / log-prob trajectory.
        """
        if self.pipe is None:
            raise RuntimeError("Engine not initialized; call initialize() first.")

        import torch

        start = time.perf_counter()
        device = self.device
        dtype = getattr(torch, self.dtype_str)

        generator = None
        if req.seed is not None:
            generator = torch.Generator(device=device).manual_seed(req.seed)

        # ---- Text conditioning (with classifier-free guidance) ----
        prompt_embeds, negative_embeds = self._encode_prompt(req.prompt)
        text_embeddings = torch.cat([negative_embeds, prompt_embeds], dim=0)

        # ---- Initial latent noise ----
        num_channels = self.unet.config.in_channels
        latent_h = req.height // self.pipe.vae_scale_factor
        latent_w = req.width // self.pipe.vae_scale_factor
        latents = torch.randn(
            (1, num_channels, latent_h, latent_w),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        latents = latents * self.scheduler.init_noise_sigma

        self.scheduler.set_timesteps(req.num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        latents_traj: list[torch.Tensor] = [latents.detach().cpu()]
        step_logprobs: list[torch.Tensor] = []
        visited_timesteps: list[int] = []

        for t in timesteps:
            # Duplicate latents for CFG (unconditional + conditional).
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input, t, encoder_hidden_states=text_embeddings
                ).sample

            noise_uncond, noise_cond = noise_pred.chunk(2)
            noise_pred = noise_uncond + req.guidance_scale * (noise_cond - noise_uncond)

            latents, logprob = self._ddim_step_with_logprob(
                noise_pred, t, latents, eta=req.eta, generator=generator
            )
            latents_traj.append(latents.detach().cpu())
            step_logprobs.append(logprob.detach().cpu())
            visited_timesteps.append(int(t))

        image = self._decode_latents(latents)
        latency = time.perf_counter() - start

        return DiffusionModelResponse(
            prompt=req.prompt,
            image=image,
            latents=latents_traj,
            timesteps=visited_timesteps,
            step_logprobs=step_logprobs,
            version=self._version,
            latency=latency,
            metadata=dict(req.metadata),
        )

    def _ddim_coeffs(self, timestep: torch.Tensor):
        """Recover the per-timestep DDIM coefficients.

        Pulled out so that both the rollout sampler and the gradient-enabled
        recompute path use *exactly* the same scheduler math (no divergence).
        Returns ``(alpha_prod_t, alpha_prod_t_prev, beta_prod_t)``.
        """
        import torch

        scheduler = self.scheduler
        prev_timestep = (
            timestep
            - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
        )
        # Keep everything on the timestep's device and branch with ``torch.where``
        # instead of a Python ``if`` on a GPU tensor: evaluating ``prev_timestep >= 0``
        # in a Python conditional forces a device-to-host sync on every denoising
        # step, which is a hot-path stall during both rollout and training.
        alphas_cumprod = scheduler.alphas_cumprod.to(device=timestep.device)
        alpha_prod_t = alphas_cumprod[timestep]
        prev_timestep_clipped = prev_timestep.clamp(min=0)
        alpha_prod_t_prev = torch.where(
            prev_timestep >= 0,
            alphas_cumprod[prev_timestep_clipped],
            torch.as_tensor(
                scheduler.final_alpha_cumprod,
                device=timestep.device,
                dtype=alphas_cumprod.dtype,
            ),
        )
        beta_prod_t = 1 - alpha_prod_t
        return alpha_prod_t, alpha_prod_t_prev, beta_prod_t

    @staticmethod
    def _ddim_mean_and_std(
        noise_pred: torch.Tensor,
        x_t: torch.Tensor,
        alpha_prod_t,
        alpha_prod_t_prev,
        beta_prod_t,
        eta: float,
    ):
        """Compute the Gaussian transition mean ``mu`` and std ``sigma``.

        This is the single source of truth for the DDIM-with-eta kernel; it is
        shared between sampling (``_ddim_step_with_logprob``) and the
        gradient-enabled recompute (``_ddim_logprob_given_samples``) so that the
        two can never drift apart numerically.
        """
        # ---- Predicted original sample x_0 (epsilon prediction) ----
        pred_original = (x_t - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (
            0.5
        )
        # ---- Stochastic noise std controlled by eta ----
        variance = (
            (1 - alpha_prod_t_prev)
            / (1 - alpha_prod_t)
            * (1 - alpha_prod_t / alpha_prod_t_prev)
        )
        # Clamp before the square root: in float16/bfloat16, underflow or rounding
        # can push ``variance`` (or the direction term below) slightly negative,
        # and ``(<0) ** 0.5`` yields NaN that then propagates through the whole
        # trajectory and destabilizes training.
        std_dev_t = eta * variance.clamp(min=0.0) ** (0.5)
        # ---- Direction pointing to x_t (uses the std-corrected coefficient) ----
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2).clamp(
            min=0.0
        ) ** (0.5) * noise_pred
        mean = alpha_prod_t_prev ** (0.5) * pred_original + pred_sample_direction
        return mean, std_dev_t

    @staticmethod
    def _gaussian_logprob(x: torch.Tensor, mean: torch.Tensor, std) -> torch.Tensor:
        """Per-sample Gaussian log-density, averaged over channel/spatial dims.

        ``log N(x; mu, sigma^2) = -0.5*((x-mu)^2/sigma^2 + log(2*pi*sigma^2))``.
        Differentiable w.r.t. ``mean`` (and therefore w.r.t. ``noise_pred`` /
        the UNet params when ``mean`` carries grad).
        """
        import torch

        std = std + 1e-8
        # Use ``torch.log(std)`` rather than ``math.log(float(std))``: calling
        # ``float()`` on a CUDA tensor forces a device-to-host sync every step
        # (and breaks outright once ``std`` is non-scalar). Keep it on-device.
        log_prob = -0.5 * (
            ((x - mean) ** 2) / (std**2) + 2 * torch.log(std) + math.log(2 * math.pi)
        )
        return log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    def _ddim_step_with_logprob(
        self,
        noise_pred: torch.Tensor,
        timestep: torch.Tensor,
        latents: torch.Tensor,
        eta: float,
        generator=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One DDIM-with-eta step that also returns the transition log-prob.

        Follows the stochastic DDIM formulation used by ddpo-pytorch: the
        forward transition ``p(x_{t-1} | x_t)`` is Gaussian with mean given by
        the deterministic DDIM update and standard deviation ``sigma_t`` set by
        ``eta``. We sample ``x_{t-1}`` from this Gaussian and evaluate its
        log-density, summed over all latent dimensions.

        Used during rollout (under ``no_grad``); the gradient-enabled twin is
        :meth:`_ddim_logprob_given_samples`, which reuses the same kernel math.
        """
        import torch

        alpha_prod_t, alpha_prod_t_prev, beta_prod_t = self._ddim_coeffs(timestep)
        prev_sample_mean, std_dev_t = self._ddim_mean_and_std(
            noise_pred, latents, alpha_prod_t, alpha_prod_t_prev, beta_prod_t, eta
        )

        # ---- Sample x_{t-1} ~ N(mean, std^2 I) ----
        noise = torch.randn(
            latents.shape,
            generator=generator,
            device=latents.device,
            dtype=latents.dtype,
        )
        prev_sample = prev_sample_mean + std_dev_t * noise
        log_prob = self._gaussian_logprob(prev_sample, prev_sample_mean, std_dev_t)
        return prev_sample, log_prob

    def _ddim_logprob_given_samples(
        self,
        noise_pred: torch.Tensor,
        timestep: torch.Tensor,
        x_t: torch.Tensor,
        x_prev: torch.Tensor,
        eta: float,
    ) -> torch.Tensor:
        """Gradient-enabled log-prob of an *already sampled* transition.

        Teacher-forcing: given the recorded ``x_t`` (input latent) and the
        actually sampled ``x_prev`` (the next latent), recompute the Gaussian
        kernel mean from a fresh ``noise_pred`` (carrying grad through the UNet)
        and evaluate ``log p(x_prev | x_t)``. Because ``x_prev`` is treated as a
        fixed observation while ``mean`` depends on the UNet, the returned
        log-prob is differentiable w.r.t. the LoRA parameters.

        Reuses the exact same kernel math as the sampler via
        ``_ddim_mean_and_std`` -- so the recomputed log-prob is consistent with
        the rollout-time value (modulo the policy weight update).
        """
        alpha_prod_t, alpha_prod_t_prev, beta_prod_t = self._ddim_coeffs(timestep)
        mean, std_dev_t = self._ddim_mean_and_std(
            noise_pred, x_t, alpha_prod_t, alpha_prod_t_prev, beta_prod_t, eta
        )
        return self._gaussian_logprob(x_prev, mean, std_dev_t)

    def iter_step_logprobs(
        self,
        latents_traj: list[torch.Tensor],
        timesteps: list[int],
        prompt: str,
        eta: float,
        guidance_scale: float,
    ):
        """Yield the per-step differentiable log-prob *one denoising step at a time*.

        This is the memory-efficient twin of :meth:`recompute_step_logprobs`.
        Instead of building the whole trajectory's autograd graph and stacking
        all step log-probs (peak memory grows linearly with ``num_steps`` and
        OOMs on a single GPU), this generator yields each step's scalar log-prob
        as soon as it is computed. The caller is expected to immediately
        ``backward()`` the (advantage-weighted) step loss and let autograd free
        that step's graph before the next step is produced.

        Mathematically this enables *exact* gradient accumulation for the
        REINFORCE surrogate, because the trajectory loss
        ``-(1/T) * sum_t logprob_t * advantage`` is linear in the per-step
        log-probs: accumulating ``grad[-(advantage/T) * logprob_t]`` over steps
        equals ``grad`` of the summed loss. Peak activation memory stays at
        roughly a single step instead of ``T`` steps.

        Args:
            latents_traj: Recorded latent trajectory, length ``num_steps+1``
                (``latents_traj[i]`` is the input ``x_t`` of step ``i``;
                ``latents_traj[i+1]`` is the sampled ``x_prev``). May live on CPU.
            timesteps: Scheduler timesteps visited, length ``num_steps``.
            prompt: Prompt used for this trajectory (for CFG conditioning).
            eta: DDIM stochasticity coefficient used during rollout.
            guidance_scale: CFG scale used during rollout.

        Yields:
            For each denoising step, a differentiable scalar (``()``-shaped)
            log-prob tensor carrying grad to the LoRA parameters.
        """
        if self.pipe is None:
            raise RuntimeError("Engine not initialized; call initialize() first.")

        import torch

        device = self.device
        dtype = getattr(torch, self.dtype_str)

        prompt_embeds, negative_embeds = self._encode_prompt(prompt)
        text_embeddings = torch.cat([negative_embeds, prompt_embeds], dim=0)

        self.scheduler.set_timesteps(len(timesteps), device=device)

        for i, t in enumerate(timesteps):
            x_t = latents_traj[i].to(device=device, dtype=dtype)
            x_prev = latents_traj[i + 1].to(device=device, dtype=dtype)
            t_tensor = torch.as_tensor(t, device=device)

            latent_model_input = torch.cat([x_t] * 2)
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t_tensor
            )
            # NOTE: grad ENABLED here (no torch.no_grad) so log_prob flows to LoRA.
            noise_pred = self.unet(
                latent_model_input, t_tensor, encoder_hidden_states=text_embeddings
            ).sample
            noise_uncond, noise_cond = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

            logprob = self._ddim_logprob_given_samples(
                noise_pred, t_tensor, x_t, x_prev, eta
            )
            yield logprob.reshape(())

    def recompute_step_logprobs(
        self,
        latents_traj: list[torch.Tensor],
        timesteps: list[int],
        prompt: str,
        eta: float,
        guidance_scale: float,
    ) -> torch.Tensor:
        """Recompute the per-step log-probs of a recorded trajectory *under grad*.

        This is the fix for the "gradient backflow gap": the rollout-time
        log-probs are detached, so the training step must re-evaluate each
        denoising transition through the (now LoRA-updated) UNet to obtain a
        differentiable log-prob.

        .. warning::
            This builds the autograd graph for the *entire* trajectory at once,
            so peak memory grows linearly with ``num_steps`` and can OOM on a
            single GPU. For training prefer :meth:`iter_step_logprobs` with
            per-step gradient accumulation. This method is kept for parity tests
            and small-trajectory use.

        Args:
            latents_traj: The recorded latent trajectory, length ``num_steps+1``
                (``latents_traj[i]`` is the input ``x_t`` of step ``i``;
                ``latents_traj[i+1]`` is the sampled ``x_prev``). May live on
                CPU; tensors are moved to the engine device.
            timesteps: The scheduler timesteps visited, length ``num_steps``.
            prompt: The prompt used for this trajectory (for CFG conditioning).
            eta: DDIM stochasticity coefficient used during rollout.
            guidance_scale: CFG scale used during rollout.

        Returns:
            A differentiable ``[num_steps]`` tensor of step log-probs.
        """
        import torch

        logprobs = list(
            self.iter_step_logprobs(
                latents_traj, timesteps, prompt, eta, guidance_scale
            )
        )
        return torch.stack(logprobs)

    def _encode_prompt(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode prompt and an empty negative prompt for CFG."""
        import torch

        tokenizer = self.pipe.tokenizer
        text_encoder = self.pipe.text_encoder

        def _embed(text: str) -> torch.Tensor:
            tokens = tokenizer(
                text,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                return text_encoder(tokens.input_ids.to(self.device))[0]

        return _embed(prompt), _embed("")

    def _decode_latents(self, latents: torch.Tensor):
        """Decode latents into a PIL image via the VAE."""
        import torch

        latents = latents / self.pipe.vae.config.scaling_factor
        with torch.no_grad():
            image = self.pipe.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()[0]
        from PIL import Image

        return Image.fromarray((image * 255).round().astype("uint8"))

    def set_version(self, version: int):
        """Set the current weight version tag."""
        self._version = version

    def get_version(self) -> int:
        """Return the current weight version tag."""
        return self._version

    def save(self, path: str):
        """Save the LoRA adapter weights to ``path``."""
        if self.unet is None:
            raise RuntimeError("Engine not initialized.")
        self.unet.save_pretrained(path)
        logger.info(f"Saved LoRA adapter to {path}")

    def load(self, path: str):
        """Load LoRA adapter weights from ``path``."""
        if self.unet is None:
            raise RuntimeError("Engine not initialized.")
        self.unet.load_adapter(path)
        logger.info(f"Loaded LoRA adapter from {path}")
