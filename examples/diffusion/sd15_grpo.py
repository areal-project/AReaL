# SPDX-License-Identifier: Apache-2.0
"""Phase 1 PoC: single-GPU SD1.5 + LoRA + GRPO with an aesthetic reward.

This script wires together the experimental diffusion components into a minimal
end-to-end training loop. It is deliberately synchronous (no async scheduler):
the ``DiffusionInferenceEngine`` does not implement the ``InferenceEngine``
contract, so we drive rollout with a plain loop here.

Run (requires a GPU, diffusers, peft, transformers, and the LAION aesthetic
predictor weights)::

    bash examples/diffusion/prepare_assets.sh        # fetch SD1.5 + CLIP + reward
    python examples/diffusion/sd15_grpo.py \\
        --num_iterations 100

With no --model_path / --clip_model / --aesthetic_weights flags the script
picks up the local copies that prepare_assets.sh writes under
``./assets/diffusion`` (see the argparse defaults below).

This is a proof of concept, not a production launcher; multi-GPU / FSDP2 / async
rollout are explicitly out of scope for Phase 1 (see the execution plan).
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from areal.experimental.diffusion.aesthetic_reward import make_aesthetic_reward_fn
from areal.experimental.diffusion.diffusion_engine import DiffusionInferenceEngine
from areal.experimental.diffusion.diffusion_loss import diffusion_grpo_loss_fn
from areal.experimental.diffusion.diffusion_workflow import DiffusionRolloutWorkflow
from areal.utils import logging

logger = logging.getLogger("SD15GRPOExample")

# Fallback prompt set when --prompt_file is not given. For real training pass a
# prompt file (one prompt per line); see examples/diffusion/prompts/.
DEFAULT_PROMPTS = [
    "a serene mountain lake at sunrise",
    "a cozy bookstore cafe in autumn",
    "a futuristic city skyline at night",
    "a field of wildflowers under a blue sky",
]


def load_prompts(prompt_file: str | None) -> list[str]:
    """Load prompts from a text file (one per line), else use the fallback set.

    Blank lines and lines starting with ``#`` are ignored.
    """
    if prompt_file is None:
        logger.warning(
            "No --prompt_file given; using %d built-in fallback prompts. "
            "Pass a prompt file for real training.",
            len(DEFAULT_PROMPTS),
        )
        return list(DEFAULT_PROMPTS)
    path = Path(prompt_file)
    if not path.is_file():
        raise FileNotFoundError(f"--prompt_file not found: {prompt_file}")
    prompts = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not prompts:
        raise ValueError(f"--prompt_file {prompt_file} contains no usable prompts")
    logger.info("Loaded %d prompts from %s", len(prompts), prompt_file)
    return prompts


def parse_args():
    p = argparse.ArgumentParser(description="SD1.5 GRPO PoC")
    p.add_argument(
        "--model_path",
        default="./assets/diffusion/stable-diffusion-v1-5",
        help="SD1.5 pipeline: local dir (default, as written by "
        "prepare_assets.sh) or a HuggingFace/ModelScope hub id.",
    )
    p.add_argument(
        "--aesthetic_weights",
        default="./assets/diffusion/sac+logos+ava1-l14-linearMSE.pth",
        help="LAION aesthetic MLP head (.pth). Defaults to the prepare_assets.sh "
        "download location.",
    )
    p.add_argument(
        "--clip_model",
        default="./assets/diffusion/clip-vit-large-patch14",
        help="CLIP ViT-L/14 backbone for the aesthetic reward: local dir "
        "(default) or a HuggingFace hub id.",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help=(
            "Compute dtype for the diffusion pipeline. Default float32 is the "
            "recommended/stable setting for REINFORCE fine-tuning: fp16 has a narrow "
            "dynamic range and the UNet forward overflows to NaN after the first LoRA "
            "update (rewards then collapse to the fixed aesthetic score of a NaN image). "
            "On a 48GB GPU fp32 peaks at ~12GB, so there is no memory reason to use fp16."
        ),
    )
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--group_size", type=int, default=8)
    p.add_argument("--num_inference_steps", type=int, default=20)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Gradient clipping max-norm for LoRA params. REINFORCE log-prob "
        "gradients are large/high-variance; clipping is required for numerical "
        "stability (without it the policy diverges to NaN within a few steps).",
    )
    p.add_argument("--num_iterations", type=int, default=100)
    p.add_argument(
        "--reinforce_stepwise",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use memory-efficient per-step gradient accumulation (REINFORCE; "
        "exact, ~single-step peak memory). Disable (--no-reinforce_stepwise) to "
        "fall back to the whole-trajectory PPO-clip path (much higher memory).",
    )
    p.add_argument(
        "--prompt_file",
        default=None,
        help="Text file with one training prompt per line (# comments allowed). "
        "If omitted, a tiny built-in fallback prompt set is used.",
    )
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

    # ---- Prompts ----
    prompts = load_prompts(args.prompt_file)

    # ---- Optimizer over LoRA params only ----
    trainable = [p for p in engine.unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    for step in range(args.num_iterations):
        prompt = prompts[step % len(prompts)]

        # ---- Rollout (synchronous; engine is not an InferenceEngine) ----
        traj = await workflow.arun_episode(engine, {"prompt": prompt})
        if traj is None:
            continue

        advantages = traj["advantages"].to(args.device)
        old_step_logprobs = traj["old_step_logprobs"].to(args.device)

        # ---- Teacher-forcing recompute of differentiable step log-probs ----
        # The rollout-time log-probs are detached and cannot backprop. We re-run
        # each recorded latent trajectory through the (LoRA-updated) UNet UNDER
        # grad to obtain new, differentiable per-step log-probs. This is the fix
        # for the gradient backflow gap.
        num_samples = len(traj["prompts"])
        optimizer.zero_grad()

        if args.reinforce_stepwise:
            # ---- Memory-efficient REINFORCE via per-step gradient accumulation ----
            # The REINFORCE trajectory loss is linear in the per-step log-probs:
            #   L = mean_s( -(1/T_s) * sum_t logprob_{s,t} * A_s )
            #     = sum_s sum_t [ -(A_s / (G * T_s)) * logprob_{s,t} ]
            # so we can backward each step immediately with weight
            # -(A_s / (G * T_s)) and let autograd free that step's graph. Peak
            # activation memory stays ~one denoising step instead of T steps,
            # while the accumulated gradient is mathematically identical to
            # backprop-ing the full summed loss. (PPO clip is non-linear in the
            # step mean, so this exact-accumulation path is REINFORCE-only.)
            total_loss_val = 0.0
            for s in range(num_samples):
                adv_s = float(advantages[s])
                T_s = len(traj["timesteps"][s])
                weight = -adv_s / (num_samples * T_s)
                for logprob_t in engine.iter_step_logprobs(
                    latents_traj=traj["latents"][s],
                    timesteps=traj["timesteps"][s],
                    prompt=traj["prompts"][s],
                    eta=args.eta,
                    guidance_scale=args.guidance_scale,
                ):
                    step_loss = weight * logprob_t
                    step_loss.backward()
                    total_loss_val += float(step_loss.detach())
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable, max_norm=args.max_grad_norm
            )
            if torch.isfinite(grad_norm):
                optimizer.step()
            else:
                logger.warning(
                    f"[step {step}] non-finite grad norm ({float(grad_norm)}); "
                    "skipping optimizer step to protect the policy."
                )
            loss_val = total_loss_val
        else:
            # ---- Original whole-trajectory path (PPO clip; high peak memory) ----
            new_logprobs_per_sample = []
            for s in range(num_samples):
                lp = engine.recompute_step_logprobs(
                    latents_traj=traj["latents"][s],
                    timesteps=traj["timesteps"][s],
                    prompt=traj["prompts"][s],
                    eta=args.eta,
                    guidance_scale=args.guidance_scale,
                )
                new_logprobs_per_sample.append(lp)
            new_step_logprobs = torch.stack(new_logprobs_per_sample, dim=0)

            loss = diffusion_grpo_loss_fn(
                new_step_logprobs,
                advantages,
                old_logprobs=old_step_logprobs,
                clip_eps=args.clip_eps,
            )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable, max_norm=args.max_grad_norm
            )
            if torch.isfinite(grad_norm):
                optimizer.step()
            else:
                logger.warning(
                    f"[step {step}] non-finite grad norm ({float(grad_norm)}); "
                    "skipping optimizer step to protect the policy."
                )
            loss_val = float(loss)

        engine.set_version(engine.get_version() + 1)

        logger.info(
            f"[step {step}] reward_mean={float(traj['rewards'].mean()):.4f} "
            f"loss={loss_val:.4f}"
        )

    engine.save(args.save_path)
    logger.info(f"Training finished; LoRA adapter saved to {args.save_path}")


def main():
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
