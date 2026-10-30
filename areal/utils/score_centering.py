# SPDX-License-Identifier: Apache-2.0
"""Head/tail score centering with ordinary PyTorch autograd.

The detached head/tail model follows Score Centering (arXiv:2609.20807)
and the verl/slime references. The surrogate uses REINFORCE with optional truncated importance sampling.
Full logits remain live; log-probabilities are computed in checkpointed chunks.
"""

import math

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop


def validate_sampling(config) -> None:
    """Reject sampling transformations not represented by the collected heads."""
    if (
        not isinstance(config.score_centering_top_k, int)
        or isinstance(config.score_centering_top_k, bool)
        or not 1 <= config.score_centering_top_k <= 1024
    ):
        raise ValueError("Score-centering head width must be an integer in [1, 1024].")
    if (
        config.temperature != 1.0
        or config.greedy
        or config.top_p != 1.0
        or config.top_k not in (-1, int(1e8))
        or config.frequency_penalty != 0
        or config.use_beam_search
    ):
        raise ValueError(
            "Score centering requires temperature=1, untruncated stochastic sampling, and no penalties."
        )


def validate_heads(ids, logprobs, count: int, width: int) -> None:
    """Validate CPU transport metadata before it reaches the training hot path."""
    if ids is None or logprobs is None or len(ids) != count or len(logprobs) != count:
        raise ValueError("Missing or misaligned SGLang sampler heads.")
    for row_ids, row_logprobs in zip(ids, logprobs):
        if len(row_ids) != width or len(row_logprobs) != width:
            raise ValueError("Sampler head width differs from the requested width.")
        if (
            any(not isinstance(i, int) or isinstance(i, bool) or i < 0 for i in row_ids)
            or len(set(row_ids)) != width
        ):
            raise ValueError("Sampler head IDs must be distinct nonnegative integers.")
        if any(
            not isinstance(x, (float, int))
            or isinstance(x, bool)
            or not math.isfinite(x)
            or x > 0
            for x in row_logprobs
        ):
            raise ValueError("Sampler head logprobs must be finite and nonpositive.")
        if sum(math.exp(x) for x in row_logprobs) > 1 + 1e-5:
            raise ValueError("Sampler head probability mass exceeds one.")


def parse_sampler_heads(meta: dict, output_tokens: list[int]):
    """Read SGLang's standard [logprob, token_id, text] top-logprob rows."""
    rows = meta.get("output_top_logprobs")
    if not rows:
        return None, None
    try:
        ids = [[entry[1] for entry in row] for row in rows]
        logprobs = [[entry[0] for entry in row] for row in rows]
        validate_heads(ids, logprobs, len(output_tokens), len(ids[0]))
    except (IndexError, TypeError) as exc:
        raise ValueError("Malformed SGLang sampler heads.") from exc
    sampled = meta["output_token_logprobs"]
    for row_ids, row_logprobs, sample in zip(ids, logprobs, sampled):
        if sample[1] in row_ids:
            head_logp = row_logprobs[row_ids.index(sample[1])]
            if not math.isclose(head_logp, sample[0], rel_tol=1e-5, abs_tol=1e-5):
                raise ValueError(
                    "Sampled-token and head logprobs describe different distributions."
                )
    return ids, logprobs


def trajectory_heads(response, width: int) -> dict[str, torch.Tensor]:
    """Pad prompt positions; keep sampler heads aligned to generated tokens."""
    validate_heads(
        response.sampler_head_ids,
        response.sampler_head_logprobs,
        response.output_len,
        width,
    )
    length = response.input_len + response.output_len
    ids = torch.zeros((length, width), dtype=torch.int64)
    logprobs = torch.full((length, width), -torch.inf, dtype=torch.float32)
    if response.output_len:
        ids[response.input_len :] = torch.tensor(
            response.sampler_head_ids, dtype=torch.int64
        )
        logprobs[response.input_len :] = torch.tensor(
            response.sampler_head_logprobs, dtype=torch.float32
        )
    return {"sampler_head_ids": ids, "sampler_head_logprobs": logprobs}


def selected_logprobs(
    logits: torch.Tensor,
    ids: torch.Tensor,
    chunk_size: int = 256,
    compute_entropy: bool = True,
):
    """Selected log-probabilities using ordinary PyTorch autograd.

    Checkpoint row chunks to avoid saving full FP32 log-probabilities for
    backward. Full input logits remain live. Entropy is diagnostic-only.
    """
    if logits.ndim != 2 or ids.ndim != 2 or logits.shape[0] != ids.shape[0]:
        raise ValueError("Expected logits [N,V] and token IDs [N,K].")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32

    def compute(chunk, indices):
        logp = chunk.to(dtype).log_softmax(-1)
        selected = logp.gather(-1, indices.long())
        entropy = -(logp.exp() * logp).sum(-1).detach() if compute_entropy else None
        return selected, entropy

    if logits.shape[0] == 0:
        return compute(logits, ids)
    outputs = [
        checkpoint(
            compute,
            logits[start : start + chunk_size],
            ids[start : start + chunk_size],
            use_reentrant=False,
        )
        if torch.is_grad_enabled()
        else compute(
            logits[start : start + chunk_size], ids[start : start + chunk_size]
        )
        for start in range(0, logits.shape[0], chunk_size)
    ]
    selected = torch.cat([output[0] for output in outputs])
    entropy = torch.cat([output[1] for output in outputs]) if compute_entropy else None
    return selected, entropy


def vocab_parallel_selected_logprobs(
    logits: torch.Tensor,
    ids: torch.Tensor,
    *,
    tp_group: dist.ProcessGroup | None,
    vocab_size: int,
    chunk_size: int = 256,
):
    """Selected probabilities from contiguous, equally sized vocabulary shards.

    Token rows and IDs must be replicated within the explicit TP group. Each rank
    evaluates the same loss, so SUM collectives have an identity local backward
    (an autograd SUM backward would incorrectly multiply gradients by TP size).
    Padding beyond the model vocabulary is excluded, matching the sampler.
    No full-vocabulary gather or custom autograd kernel is needed.
    """
    if logits.ndim != 2 or ids.ndim != 2 or logits.shape[0] != ids.shape[0]:
        raise ValueError("Expected logits [N,V_local] and token IDs [N,K].")
    world_size = dist.get_world_size(tp_group) if tp_group is not None else 1
    rank = dist.get_rank(tp_group) if tp_group is not None else 0
    width = logits.shape[1]
    if chunk_size <= 0 or not 0 < vocab_size <= width * world_size:
        raise ValueError("Invalid chunk size or vocabulary size.")
    torch._assert_async(
        ((ids >= 0) & (ids < vocab_size)).all(),
        "Selected token IDs must be within the model vocabulary.",
    )
    start = rank * width
    valid_vocab = torch.arange(width, device=logits.device) + start < vocab_size
    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32

    def reduce_sum(value):
        if world_size == 1:
            return value
        total = value.detach().clone()
        dist.all_reduce(total, group=tp_group)
        return total + (value - value.detach())

    def compute(chunk, indices):
        x = chunk.to(dtype).masked_fill(~valid_vocab, -torch.inf)
        maximum = x.detach().amax(-1, keepdim=True)
        if world_size > 1:
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=tp_group)
        shifted = x - maximum
        denominator = reduce_sum(shifted.exp().sum(-1, keepdim=True))
        local_ids = indices.long() - start
        owned = (local_ids >= 0) & (local_ids < width) & (indices < vocab_size)
        selected = shifted.gather(-1, local_ids.clamp(0, width - 1))
        selected = reduce_sum(selected.masked_fill(~owned, 0))
        logp = selected - denominator.log()
        with torch.no_grad():
            local_logp = shifted - denominator.log()
            terms = local_logp.exp() * local_logp.masked_fill(~valid_vocab, 0)
            entropy = -terms.sum(-1)
            if world_size > 1:
                dist.all_reduce(entropy, group=tp_group)
        return logp, entropy

    if logits.shape[0] == 0:
        return logits.to(dtype).sum(-1, keepdim=True).expand_as(ids), logits.to(
            dtype
        ).sum(-1).detach()
    # All TP ranks must replay every collective in identical order in backward.
    with set_checkpoint_early_stop(False):
        outputs = [
            checkpoint(
                compute,
                logits[i : i + chunk_size],
                ids[i : i + chunk_size],
                use_reentrant=False,
            )
            if torch.is_grad_enabled()
            else compute(logits[i : i + chunk_size], ids[i : i + chunk_size])
            for i in range(0, logits.shape[0], chunk_size)
        ]
    return tuple(torch.cat([output[j] for output in outputs]) for j in range(2))


def score_centering_correction(
    train_head_logp, sampler_head_logp, eps=1e-6, *, tis_cap=None
):
    """Only the final score factor is differentiated; coefficients are detached."""
    with torch.no_grad():
        p, q = train_head_logp.exp(), sampler_head_logp.exp()
        # Match the paper: only the denominator needs an epsilon floor.
        # A sampler with zero tail mass must not acquire artificial tail mass.
        rho = (1 - q.sum(-1)).clamp_min(0) / (1 - p.sum(-1)).clamp_min(eps)
        if tis_cap is None:
            residual = q - rho.unsqueeze(-1) * p
        else:
            # Paper Eq. 12: center the weighted scores. These algebraic forms
            # avoid dividing by zero q or multiplying zero tail mass by infinity.
            weighted_head_mass = torch.minimum(p, tis_cap * q)
            alpha = (tis_cap * rho).clamp_max(1)
            residual = weighted_head_mass - alpha.unsqueeze(-1) * p
    return (residual * train_head_logp).sum(-1)


def score_centering_logprobs(
    logits,
    labels,
    head_ids,
    sampler_head_logp,
    chunk_size=256,
    *,
    tp_group=None,
    vocab_size=None,
    tis_cap=None,
):
    """Compute sampled log-probabilities, diagnostic entropy, and SC correction."""
    if head_ids.shape != sampler_head_logp.shape:
        raise ValueError("Sampler IDs and logprobs must have matching shapes.")
    ids = torch.cat((labels.reshape(-1, 1), head_ids), dim=-1)
    if vocab_size is None:
        if tp_group is not None:
            raise ValueError("Vocabulary-parallel score centering requires vocab_size.")
        selected, entropy = selected_logprobs(logits, ids, chunk_size)
    else:
        selected, entropy = vocab_parallel_selected_logprobs(
            logits, ids, tp_group=tp_group, vocab_size=vocab_size, chunk_size=chunk_size
        )
    correction = score_centering_correction(
        selected[:, 1:], sampler_head_logp, tis_cap=tis_cap
    )
    return selected[:, 0], entropy, correction


def score_centering_tis_weights(train_logp, sampler_logp, cap):
    """Detached token-wise min(p/q, cap), using actual sampled-token logprobs."""
    with torch.no_grad():
        # Clamp in log space before exponentiating, including extreme mismatch.
        return (train_logp - sampler_logp).clamp_max(math.log(cap)).exp().clamp_max(cap)


def resolve_score_centering_tis_cap(rejection_sampling):
    """Use AReaL's existing TIS configuration for the weighted SC objective.

    SC uses current-trainer / sampler ratios, not decoupled PPO's proximal
    ratios. Other rejection rules need their own weighted-centering derivation.
    """
    if rejection_sampling is None:
        return None
    rs = rejection_sampling
    if (
        rs.level != "token"
        or rs.metric != "ratio"
        or rs.action != "clamp"
        or rs.lower not in (None, 0.0)
    ):
        raise ValueError(
            "Score centering supports rejection_sampling only with "
            "level='token', metric='ratio', action='clamp', and no lower bound."
        )
    if (
        isinstance(rs.upper, bool)
        or not math.isfinite(rs.upper)
        or not 1 < rs.upper <= torch.finfo(torch.float32).max
    ):
        raise ValueError(
            "Score-centering TIS upper must be finite, greater than 1, and FP32-representable."
        )
    return rs.upper
