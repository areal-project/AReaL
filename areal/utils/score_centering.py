# SPDX-License-Identifier: Apache-2.0
"""Score Centering (SC) correction for off-policy RL.

Implements the additive drift correction from "Score Centering Stabilizes
Off-policy Reinforcement Learning" (arXiv:2609.20807, Marek & Ryabinin).

The sampler's top-k head is exact; its tail is modeled as the trainer's tail
rescaled to the sampler's tail mass, so the centering term only needs the k
head log-probs of the trainer. This mirrors the reference implementation in
VeRL (verl/trainer/ppo/score_centering.py) and Slime (PR #2405).

Core idea: subtract the expected trainer score under the sampler distribution:
    correction = sum_H sg[q*w - alpha*p] * log p   (additive, low variance)

where rho = (1 - q_mass) / (1 - p_mass) models the tail mass ratio, alpha =
rho * weight_fn(1/rho) is the tail IS weight, and w = weight_fn(p/q) is the
head IS weight. When weight_fn = identity (pure SC, no IS), the correction
reduces to the exact top-k centering term. On-policy (p=q) -> correction = 0
(residual = 0). Gradient flows only through the trainer logits; all sampler
quantities and the residual are detached.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

# Padding log-prob for an absent top-k slot. ``exp(-inf) == 0`` so padded slots
# contribute no probability mass; padding with ``0.0`` would mean ``exp(0) == 1``
# and silently inflate ``q_mass``.
NEG_INF_PAD = float("-inf")

# Chunk size for _TopKLogProbsFromLogits: bounds the fp32 workspace to
# (chunk_size, vocab) instead of (N, vocab). Matches VeRL default.
_TOPK_LOG_PROB_CHUNK_SIZE = 4096


def pad_topk_ids(token_ids: list[int], k: int) -> list[int]:
    """Right-pad a top-k token-id row to length *k* with the padding id 0."""
    ids = list(token_ids[:k])
    return ids + [0] * max(0, k - len(ids))


def pad_topk_logps(logps: list[float], k: int) -> list[float]:
    """Right-pad a top-k log-prob row to length *k* with ``-inf`` (q = 0)."""
    ps = list(logps[:k])
    return ps + [NEG_INF_PAD] * max(0, k - len(ps))


def check_score_centering_sp_supported(use_score_centering: bool, sp_size: int) -> None:
    """Fail fast when Score Centering is combined with Ulysses sequence parallel.

    ``fsdp_engine`` injects the trainer ``logits`` for the *local* SP slice,
    while ``logprobs``/``loss_mask`` are all-gathered to the full sequence first.
    With ``sp_size > 1`` the injected logits therefore have a different sequence
    length than the sampler data, so SC cannot run correctly.
    """
    if use_score_centering and sp_size > 1:
        raise NotImplementedError(
            "Score Centering is not supported with Ulysses sequence parallel "
            f"(sp_size={sp_size}): the injected trainer logits are the local SP "
            "slice, not the full sequence. Set actor.use_score_centering=False "
            "or keep sp_size == 1."
        )


class _TopKLogProbsFromLogits(torch.autograd.Function):
    """log_softmax(logits).gather(ids) with a bounded fp32 workspace.

    Equivalent to ``torch.log_softmax(logits, dim=-1).gather(-1, token_ids)``
    but never materializes the full (N, V) softmax at once: the normalizer is
    recomputed per chunk in forward and backward, bounding the extra fp32
    workspace to (chunk_size, V).

    Only the original logits, the token_ids and two scalars per row (maxima,
    denominators) are saved for backward; the softmax is recomputed per chunk.

    Ported from VeRL (verl/trainer/ppo/score_centering.py).
    """

    @staticmethod
    def forward(ctx, logits, token_ids, chunk_size):
        n = logits.shape[0]
        chunk_size = chunk_size if chunk_size > 0 else max(n, 1)
        # Use the input dtype, but promote low-precision floats (bf16/fp16) to
        # fp32 for numerical safety. fp64 is preserved for test precision.
        work_dtype = (
            logits.dtype
            if logits.dtype in (torch.float32, torch.float64)
            else torch.float32
        )
        output = torch.empty(token_ids.shape, device=logits.device, dtype=work_dtype)
        maxima = torch.empty((n, 1), device=logits.device, dtype=work_dtype)
        denominators = torch.empty_like(maxima)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            work = logits[start:end].to(dtype=work_dtype, copy=True)
            maximum = work.max(dim=-1, keepdim=True).values
            targets = work.gather(-1, token_ids[start:end])
            work.sub_(maximum).exp_()
            denominator = work.sum(dim=-1, keepdim=True)
            output[start:end] = (targets - maximum) - denominator.log()
            maxima[start:end] = maximum
            denominators[start:end] = denominator
            del work
        ctx.save_for_backward(logits, token_ids, maxima, denominators)
        ctx.chunk_size = chunk_size
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        logits, token_ids, maxima, denominators = ctx.saved_tensors
        n = logits.shape[0]
        work_dtype = maxima.dtype  # same as forward's work_dtype
        grad_input = torch.empty_like(logits)
        for start in range(0, n, ctx.chunk_size):
            end = min(start + ctx.chunk_size, n)
            work = logits[start:end].to(dtype=work_dtype, copy=True)
            work.sub_(maxima[start:end]).exp_().div_(denominators[start:end])
            grad = grad_output[start:end].to(dtype=work_dtype)
            work.mul_(-grad.sum(dim=-1, keepdim=True))
            work.scatter_add_(-1, token_ids[start:end], grad)
            grad_input[start:end] = work
            del work
        return grad_input, None, None


def gather_train_head_logp(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Gather log-probs of *token_ids* from trainer *logits*.

    Uses a chunked fp32 autograd function that never materializes the full
    (N, V) softmax, bounding memory to (chunk_size, V). Equivalent to
    ``torch.log_softmax(logits, dim=-1).gather(-1, token_ids)`` but memory-safe.

    Args:
        logits: ``[..., vocab]`` — raw trainer logits.
        token_ids: ``[..., k]`` — top-k token IDs sampled by the sampler.
        temperature: softmax temperature applied to logits.

    Returns:
        ``[..., k]`` — log-prob of each token under the trainer distribution.
        Gradient flows through logits only.
    """
    if temperature != 1.0:
        logits = logits / temperature
    # Reshape to (N, vocab) / (N, k) for the chunked autograd function.
    orig_shape = logits.shape[:-1]
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_ids = token_ids.reshape(-1, token_ids.shape[-1]).long()
    flat_logp = _TopKLogProbsFromLogits.apply(
        flat_logits, flat_ids, _TOPK_LOG_PROB_CHUNK_SIZE
    )
    return flat_logp.reshape(*orig_shape, token_ids.shape[-1])


def _build_weight_fn(
    sc_mode: str, sc_clip_low: float, sc_clip_high: float
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build the IS weight function shared by head and tail (after VeRL).

    - "none": identity (pure SC, no IS) -> weight_fn(r) = 1
    - "tis": upper-clamp -> weight_fn(r) = r.clamp(max=sc_clip_high)
    - "mis": band-mask (IcePop) -> weight_fn(r) = where(low<=r<=high, r, 0)
    """
    if sc_mode == "none":
        return torch.ones_like
    if sc_mode == "tis":
        return lambda ratio: ratio.clamp(max=sc_clip_high)
    if sc_mode == "mis":
        low, high = sc_clip_low, sc_clip_high
        return lambda ratio: torch.where(
            (ratio >= low) & (ratio <= high), ratio, torch.zeros_like(ratio)
        )
    raise ValueError(f"sc_mode must be 'none', 'tis', or 'mis'; got {sc_mode!r}.")


def score_centering_correction(
    train_head_logp: torch.Tensor,
    sampler_topk_logps: torch.Tensor,
    sampler_topk_mask: torch.Tensor | None = None,
    sc_mode: str = "none",
    sc_clip_low: float = 0.0,
    sc_clip_high: float = float("inf"),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the additive Score Centering correction.

    The sampler's tail (outside the top-k head H) is modeled as the trainer's
    tail rescaled by ``alpha = rho * weight_fn(1/rho)`` with
    ``rho = (1 - q_mass) / (1 - p_mass)``, so the correction only needs the k
    head log-probs of both distributions.

        correction = sum_H sg[q*w - alpha*p] * log p     (additive, low variance)

    On-policy (p=q) -> residual = 0 -> correction = 0 (exact no-op, both value
    and gradient). Gradient flows only through *train_head_logp*; all sampler
    quantities and the residual are detached.

    Args:
        train_head_logp: ``[batch, seq, k]`` — trainer log-probs for the
            sampler's top-k tokens (from :func:`gather_train_head_logp`).
            **Gradient flows through this tensor only.**
        sampler_topk_logps: ``[batch, seq, k]`` — sampler log-probs for the
            same top-k tokens. **Detached** (no gradient to inference engine).
        sampler_topk_mask: ``[batch, seq, k]`` — optional mask (1 = valid).
        sc_mode: "none" (pure SC) | "tis" (truncated IS) | "mis" (mixture IS).
        sc_clip_low / sc_clip_high: bounds for IS ratio clipping (tis/mis).

    Returns:
        ``(correction, s_bar, q_mass, p_mass, rho)`` each ``[batch, seq]``.

        - correction: centering term to subtract from the loss (has gradient).
        - s_bar: ``sum_v q_v * s_p(v)`` (expected trainer score under sampler,
          detached). Kept for backward-compatible stats reporting.
        - q_mass: ``sum_v q_v`` (sampler head mass, detached).
        - p_mass: ``sum_v p_v`` (trainer head mass over the sampler top-k,
          detached). Used to detect pad-slot pollution of the trainer head.
        - rho: ``(1 - q_mass) / (1 - p_mass)`` tail-mass ratio (detached).
          A value of ~0 signals tail-model degeneration.
    """

    # Promote low-precision floats (bf16/fp16) to fp32 for numerical safety;
    # keep fp32 and fp64 as-is (fp64 needed for test precision).
    def _promote(t: torch.Tensor) -> torch.Tensor:
        return t.float() if t.dtype in (torch.bfloat16, torch.float16) else t

    train_head_logp = _promote(train_head_logp)
    q_logp = _promote(sampler_topk_logps.detach())  # [batch, seq, k]

    q = torch.exp(q_logp)  # [batch, seq, k]

    # Self-evident mask: -inf logprob slots are invalid (pad). Fall back to
    # finiteness of the sampler logprob when no explicit mask is provided.
    if sampler_topk_mask is not None:
        valid = sampler_topk_mask.detach().bool()
    else:
        valid = torch.isfinite(q_logp)
    mask = valid.float()  # [batch, seq, k]

    # Apply mask: a masked slot must contribute zero mass, not exp(0)=1.
    q = q * mask

    weight_fn = _build_weight_fn(sc_mode, sc_clip_low, sc_clip_high)

    with torch.no_grad():
        p = train_head_logp.exp() * mask  # [batch, seq, k] — also masked
        p_mass = p.sum(-1)  # [batch, seq]
        q_mass = q.sum(-1)  # [batch, seq]

        # Tail mass ratio: rho = q_tail / p_tail = (1-q_mass) / (1-p_mass).
        # Clamp tails to eps to avoid div-by-zero when head covers most mass.
        eps = 1e-6
        rho = (1 - q_mass).clamp_min(eps) / (1 - p_mass).clamp_min(eps)
        alpha = rho * weight_fn(1.0 / rho)  # [batch, seq]

        # Head IS weight: w = weight_fn(p/q). In pure SC (weight_fn=ones), w=1.
        # Use a safe ratio (replace inf/NaN in pad slots with 0 before weight_fn).
        ratio = torch.where(
            valid, (train_head_logp - q_logp).exp(), torch.zeros_like(q_logp)
        )
        head_weights = weight_fn(ratio) * mask  # [batch, seq, k]

        # Residual: weighted sampler head minus weighted trainer tail estimate.
        # Pad slots are exactly zero: q=0, head_weights=0, p=0, so residual=0.
        residual = q * head_weights - alpha.unsqueeze(-1) * p  # [batch, seq, k]

    # Gradient flows only through train_head_logp (residual is detached).
    correction = (residual * train_head_logp).sum(-1)  # [batch, seq]

    # s_bar kept for backward-compatible stats reporting (detached).
    s_bar = (q * train_head_logp.detach()).sum(-1)  # [batch, seq]

    return correction, s_bar, q_mass, p_mass, rho
