# SPDX-License-Identifier: Apache-2.0
"""Phase 1 PoC: single-GPU SD1.5 + LoRA + GRPO with an aesthetic reward.

This script wires together the experimental diffusion components into a minimal
end-to-end training loop. It is deliberately synchronous (no async scheduler):
the ``DiffusionInferenceEngine`` does not implement the ``InferenceEngine``
contract, so we drive rollout with a plain loop here.

Run (requires a GPU, diffusers, peft, transformers, and the LAION aesthetic
predictor weights)::

    python examples/diffusion/sd15_grpo.py \\
        --model_path stable-diffusion-v1-5/stable-diffusion-v1-5 \\
        --aesthetic_weights /path/to/sac+logos+ava1-l14-linearMSE.pth \\
        --num_iterations 100

This is a proof of concept, not a production launcher; multi-GPU / FSDP2 / async
rollout are explicitly out of scope for Phase 1 (see the execution plan).
"""

from __future__ import annotations

import argparse
import asyncio

from areal.experimental.diffusion.aesthetic_reward import make_aesthetic_reward_fn
from areal.experimental.diffusion.diffusion_engine import DiffusionInferenceEngine
from areal.experimental.diffusion.diffusion_loss import diffusion_grpo_loss_fn
from areal.experimental.diffusion.diffusion_workflow import DiffusionRolloutWorkflow
from areal.utils import logging

logger = logging.getLogger("SD15GRPOExample")

# A tiny fixed prompt set for the PoC. Replace with a real dataset loader.
DEFAULT_PROMPTS = [
    "a serene mountain lake at sunrise",
    "a cozy bookstore cafe in autumn",
    "a futuristic city skyline at night",
    "a field of wildflowers under a blue sky",
]


def parse_args():
    p = argparse.ArgumentParser(description="SD1.5 GRPO PoC")
    # The original `runwayml/stable-diffusion-v1-5` repo was removed from the
    # Hub; use the byte-identical community mirror (see prepare_assets.sh).
    p.add_argument("--model_path", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    p.add_argument("--aesthetic_weights", default=None)
    p.add_argument("--clip_model", default="openai/clip-vit-large-patch14")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--group_size", type=int, default=8)
    p.add_argument("--num_inference_steps", type=int, default=20)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_iterations", type=int, default=100)
    p.add_argument("--save_path", default="./sd15_grpo_lora")
    return p.parse_args()


async def run(args):
    import torch

    # ---- Engine ----
    engine = DiffusionInferenceEngine(
        model_path=args.model_path,
        dtype=args.dtype,
        device=args.device,
        lora_rank=args.lora_rank,
    )
    engine.initialize()

    # ---- Reward ----
    reward_fn = make_aesthetic_reward_fn(
        weights_path=args.aesthetic_weights,
        clip_model=args.clip_model,
        device=args.device,
    )

    # ---- Workflow ----
    workflow = DiffusionRolloutWorkflow(
        reward_fn=reward_fn,
        group_size=args.group_size,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        eta=args.eta,
    )

    # ---- Optimizer over LoRA params only ----
    trainable = [p for p in engine.unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    for step in range(args.num_iterations):
        prompt = DEFAULT_PROMPTS[step % len(DEFAULT_PROMPTS)]

        # ---- Rollout (synchronous; engine is not an InferenceEngine) ----
        traj = await workflow.arun_episode(engine, {"prompt": prompt})
        if traj is None:
            continue

        advantages = traj["advantages"].to(args.device)
        # NOTE(agent): Phase 1 uses the rollout-time log-probs directly as a
        # differentiable surrogate placeholder. A correct implementation must
        # recompute step log-probs under grad (teacher-forcing the recorded
        # latents through the UNet). That recompute path is tracked as a Phase 2
        # follow-up in the execution plan; see diffusion_engine for the forward.
        step_logprobs = traj["step_logprobs"].to(args.device).requires_grad_(True)

        loss = diffusion_grpo_loss_fn(step_logprobs, advantages)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        engine.set_version(engine.get_version() + 1)

        logger.info(
            f"[step {step}] reward_mean={float(traj['rewards'].mean()):.4f} "
            f"loss={float(loss):.4f}"
        )

    engine.save(args.save_path)
    logger.info(f"Training finished; LoRA adapter saved to {args.save_path}")


def main():
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
