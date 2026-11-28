# SPDX-License-Identifier: Apache-2.0
"""Linear speculative rejection sampling on the retained target support."""

import torch
import triton
import triton.language as tl


@triton.jit
def _final_weights(
    target, draft, ids, columns, K: tl.constexpr, rejected, vocab_stride: tl.constexpr
):
    valid = columns < K
    p = tl.load(target + columns, mask=valid, other=0.0)
    if rejected:
        token = tl.load(ids + columns, mask=valid, other=0)
        q = tl.load(draft + token * vocab_stride, mask=valid, other=0.0)
        p = tl.maximum(p - tl.where(q == q, q, 0.0), 0.0)
    return p.to(tl.float64)


@triton.jit
def _sample_chain(
    proposals,
    ids,
    target,
    draft,
    uniforms,
    accepted_out,
    bonus_out,
    GAMMA: tl.constexpr,
    K: tl.constexpr,
    draft_batch_stride,
    draft_row_stride,
    draft_vocab_stride: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch = tl.program_id(0).to(tl.int64)
    ids += batch * (GAMMA + 1) * K
    target += batch * (GAMMA + 1) * K
    draft += batch * draft_batch_stride
    uniforms += batch * (GAMMA + 1)
    lanes = tl.arange(0, BLOCK)
    accepted = 0
    keep = True
    while (accepted < GAMMA) & keep:
        step = accepted
        token = tl.load(proposals + batch * GAMMA + step)
        p = 0.0
        for start in range(0, K, BLOCK):
            col = start + lanes
            candidate = tl.load(ids + step * K + col, mask=col < K, other=-1)
            prob = tl.load(
                target + step * K + col,
                mask=(col < K) & (candidate == token),
                other=0.0,
            )
            p += tl.sum(prob, axis=0)
        q = tl.load(draft + step * draft_row_stride + token * draft_vocab_stride)
        keep = tl.load(uniforms + step) * q < p
        accepted += keep.to(tl.int32)
    tl.store(accepted_out + batch, accepted)

    target += accepted * K
    ids += accepted * K
    # The bonus row has no corresponding draft row.
    draft += tl.minimum(accepted, GAMMA - 1) * draft_row_stride
    rejected = accepted < GAMMA
    mass = tl.full((), 0, tl.float64)
    for start in range(0, K, BLOCK):
        weights = _final_weights(
            target, draft, ids, start + lanes, K, rejected, draft_vocab_stride
        )
        mass += tl.sum(weights, axis=0)
    # For a numerically empty residual, draw from the original target.
    if mass == 0.0:
        rejected = False
        for start in range(0, K, BLOCK):
            p = tl.load(target + start + lanes, mask=start + lanes < K, other=0.0)
            mass += tl.sum(p.to(tl.float64), axis=0)

    ticket = tl.load(uniforms + GAMMA) * mass
    cumulative = tl.full((), 0, tl.float64)
    chosen = K
    last_positive = -1
    for start in range(0, K, BLOCK):
        col = start + lanes
        weights = _final_weights(
            target, draft, ids, col, K, rejected, draft_vocab_stride
        )
        cdf = cumulative + tl.cumsum(weights, axis=0)
        positive = weights > 0.0
        chosen = tl.minimum(
            chosen, tl.min(tl.where(positive & (cdf > ticket), col, K), axis=0)
        )
        last_positive = tl.maximum(
            last_positive, tl.max(tl.where(positive, col, -1), axis=0)
        )
        cumulative += tl.sum(weights, axis=0)
    # Even a rounded upper endpoint belongs to a positive-mass candidate.
    column = tl.where(chosen == K, last_positive, chosen)
    tl.store(bonus_out + batch, tl.load(ids + column))


def sample_chain(proposals, policy, draft_probs, uniforms):
    bs, gamma = proposals.shape
    k = policy.rows.ids.shape[-1]
    accepted = torch.empty(bs, dtype=torch.int32, device=proposals.device)
    bonus = torch.empty(bs, dtype=torch.int64, device=proposals.device)
    _sample_chain[(bs,)](
        proposals.contiguous(),
        policy.rows.ids.contiguous(),
        policy.logprobs.exp().contiguous(),
        draft_probs,
        uniforms.contiguous(),
        accepted,
        bonus,
        gamma,
        k,
        *draft_probs.stride(),
        BLOCK=min(triton.next_power_of_2(k), 1024),
    )
    return accepted, bonus
