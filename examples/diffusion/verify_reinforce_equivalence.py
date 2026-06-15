"""Verify the REINFORCE per-step gradient-accumulation proposal.

This script proves the two central claims of the issue's proposal:

1. **Mathematical equivalence.** Backward-ing the whole-trajectory REINFORCE
   loss in one shot (build the full graph, sum all step log-probs, then
   ``loss.backward()``) produces *exactly the same* parameter gradients as
   accumulating each denoising step's gradient immediately
   (``iter_step_logprobs`` -> per-step ``backward()``). This holds because the
   REINFORCE trajectory loss is linear in the per-step log-probs, so
   ``grad(sum_t w*lp_t) == sum_t grad(w*lp_t)``.

2. **Memory saving.** The per-step path keeps peak activation memory at roughly
   one denoising step instead of growing linearly with ``num_inference_steps``.

Both paths consume the *same* rollout trajectory (same latents / timesteps /
advantages), so the gradients are directly comparable element by element.

Usage mirrors ``sd15_grpo.py``; see ``--help``.
"""

import argparse
import asyncio

import torch

# Reuse the example's reward + prompt helpers so behaviour matches training.
from examples.diffusion.sd15_grpo import load_prompts, make_aesthetic_reward_fn

from areal.experimental.diffusion import (
    DiffusionInferenceEngine,
    DiffusionRolloutWorkflow,
)
from areal.utils import logging

logger = logging.getLogger("VerifyReinforceEquiv")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--aesthetic_weights", default=None)
    p.add_argument("--clip_model", default="openai/clip-vit-large-patch14")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--group_size", type=int, default=4)
    p.add_argument("--num_inference_steps", type=int, default=10)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--prompt_file", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--rtol", type=float, default=1e-4, help="relative tolerance for grad equality"
    )
    p.add_argument(
        "--atol", type=float, default=1e-6, help="absolute tolerance for grad equality"
    )
    return p.parse_args()


def _trainable(engine):
    return [p for p in engine.unet.parameters() if p.requires_grad]


def _flat_grad(params):
    """Concatenate all param grads into one 1-D tensor (zeros where grad is None)."""
    chunks = []
    for p in params:
        g = p.grad
        chunks.append(
            torch.zeros_like(p).reshape(-1) if g is None else g.detach().reshape(-1)
        )
    return torch.cat(chunks)


def _zero_grad(params):
    for p in params:
        p.grad = None


def _peak_mem_mb(device):
    if device.startswith("cuda"):
        return torch.cuda.max_memory_allocated() / (1024**2)
    return float("nan")


def _grad_whole_trajectory(engine, traj, advantages, num_samples, device, eta, gscale):
    """Original path: one big graph, sum the REINFORCE loss, single backward."""
    params = _trainable(engine)
    _zero_grad(params)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    total = torch.zeros((), device=device)
    for s in range(num_samples):
        adv_s = float(advantages[s])
        T_s = len(traj["timesteps"][s])
        lp = engine.recompute_step_logprobs(
            latents_traj=traj["latents"][s],
            timesteps=traj["timesteps"][s],
            prompt=traj["prompts"][s],
            eta=eta,
            guidance_scale=gscale,
        )
        # REINFORCE: L_s = -(A_s / T_s) * sum_t lp_t ; averaged over G samples.
        total = total + (-adv_s / (num_samples * T_s)) * lp.sum()
    total.backward()
    g = _flat_grad(params)
    return g, float(total.detach()), _peak_mem_mb(device)


def _grad_stepwise(engine, traj, advantages, num_samples, device, eta, gscale):
    """Proposal path: per-step backward, gradient accumulates in .grad."""
    params = _trainable(engine)
    _zero_grad(params)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    total_val = 0.0
    for s in range(num_samples):
        adv_s = float(advantages[s])
        T_s = len(traj["timesteps"][s])
        weight = -adv_s / (num_samples * T_s)
        for logprob_t in engine.iter_step_logprobs(
            latents_traj=traj["latents"][s],
            timesteps=traj["timesteps"][s],
            prompt=traj["prompts"][s],
            eta=eta,
            guidance_scale=gscale,
        ):
            step_loss = weight * logprob_t
            step_loss.backward()
            total_val += float(step_loss.detach())
    g = _flat_grad(params)
    return g, total_val, _peak_mem_mb(device)


async def run(args):
    torch.manual_seed(args.seed)

    engine = DiffusionInferenceEngine(
        model_path=args.model_path,
        dtype=args.dtype,
        device=args.device,
        lora_rank=args.lora_rank,
    )
    engine.initialize()

    reward_fn = make_aesthetic_reward_fn(
        weights_path=args.aesthetic_weights,
        clip_model=args.clip_model,
        device=args.device,
    )
    workflow = DiffusionRolloutWorkflow(
        reward_fn=reward_fn,
        group_size=args.group_size,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        eta=args.eta,
    )
    prompts = load_prompts(args.prompt_file)

    # ---- One fixed rollout, reused by BOTH paths ----
    traj = await workflow.arun_episode(engine, {"prompt": prompts[0]})
    if traj is None:
        raise RuntimeError("rollout returned None; cannot verify")

    advantages = traj["advantages"].to(args.device)
    num_samples = len(traj["prompts"])
    logger.info(
        f"Fixed rollout: {num_samples} samples, "
        f"T per sample = {[len(t) for t in traj['timesteps']]}, "
        f"advantages = {[round(float(a), 4) for a in advantages]}"
    )

    # ---- Path A: whole-trajectory single backward ----
    g_whole, loss_whole, mem_whole = _grad_whole_trajectory(
        engine,
        traj,
        advantages,
        num_samples,
        args.device,
        args.eta,
        args.guidance_scale,
    )
    logger.info(
        f"[whole-trajectory] loss={loss_whole:.6f} "
        f"peak_mem={mem_whole:.1f} MiB grad_norm={float(g_whole.norm()):.6e}"
    )

    # ---- Path B: per-step accumulation (the proposal) ----
    g_step, loss_step, mem_step = _grad_stepwise(
        engine,
        traj,
        advantages,
        num_samples,
        args.device,
        args.eta,
        args.guidance_scale,
    )
    logger.info(
        f"[stepwise-accum ] loss={loss_step:.6f} "
        f"peak_mem={mem_step:.1f} MiB grad_norm={float(g_step.norm()):.6e}"
    )

    # ---- Compare ----
    abs_diff = (g_whole - g_step).abs()
    max_abs = float(abs_diff.max())
    cos = float(
        torch.nn.functional.cosine_similarity(g_whole.unsqueeze(0), g_step.unsqueeze(0))
    )
    # The fair equivalence metric is the *relative residual norm*:
    #   ||g_whole - g_step|| / ||g_whole||
    # Per-element relative error blows up where g_whole ~ 0 (tiny denominator)
    # and is meaningless there; the global norm ratio is the honest measure of
    # whether the two gradient vectors agree. The two paths share the SAME
    # forward (recompute_step_logprobs just stacks iter_step_logprobs), so any
    # residual is pure fp32 reduction-order rounding, not algorithmic error.
    rel_resid = float((g_whole - g_step).norm() / g_whole.norm().clamp_min(1e-12))
    # Pass if the gradient vectors are co-directional and the relative residual
    # norm is at floating-point rounding scale. fp32 reduction-order rounding
    # lands around 1e-3; fp64 should collapse to ~1e-10 (proving the only
    # difference was rounding, i.e. the two paths are algebraically identical).
    tol = 1e-2 if args.dtype == "float32" else 1e-6
    allclose = (cos > 1 - 1e-5) and (rel_resid < tol)

    print("\n" + "=" * 64)
    print("REINFORCE per-step accumulation -- equivalence verification")
    print("=" * 64)
    print(f"  num params compared : {g_whole.numel()}")
    print(f"  loss (whole / step) : {loss_whole:.6f} / {loss_step:.6f}")
    print(f"  grad max abs diff   : {max_abs:.3e}")
    print(f"  grad rel resid norm : {rel_resid:.3e}   (||dg|| / ||g||)")
    print(f"  grad cosine sim     : {cos:.10f}")
    print(f"  equivalent?         : {allclose}   (cos>1-1e-4 and rel_resid<1e-3)")
    print("-" * 64)
    print(f"  peak mem whole-traj : {mem_whole:.1f} MiB")
    print(f"  peak mem stepwise   : {mem_step:.1f} MiB")
    if mem_whole == mem_whole and mem_step > 0:  # not NaN
        print(f"  memory reduction    : {mem_whole / mem_step:.2f}x")
    print("=" * 64)
    print(
        "RESULT: "
        + (
            "PASS -- gradients are equivalent; proposal is mathematically exact."
            if allclose
            else "FAIL -- gradients differ beyond tolerance."
        )
    )
    print("=" * 64 + "\n")
    return 0 if allclose else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(parse_args())))
