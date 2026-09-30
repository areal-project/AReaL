# SPDX-License-Identifier: Apache-2.0

"""Compact-logit log probabilities for semantic rollout positions."""

import torch
import torch.distributed as dist


class _MappedLogprobsEntropy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        logits: torch.Tensor,
        rows: torch.Tensor,
        labels: torch.Tensor,
        temperature: float,
        tp_group: dist.ProcessGroup | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tp_size = dist.get_world_size(tp_group) if tp_group is not None else 1
        tp_rank = dist.get_rank(tp_group) if tp_size > 1 else 0
        width = logits.shape[-1]
        unique_rows, inverse = torch.unique(rows, sorted=True, return_inverse=True)
        scaled = logits.index_select(0, unique_rows).float() / temperature

        row_max = scaled.max(dim=-1).values
        if tp_size > 1:
            dist.all_reduce(row_max, op=dist.ReduceOp.MAX, group=tp_group)
        exp_logits = (scaled - row_max[:, None]).exp()
        row_sum = exp_logits.sum(dim=-1)
        if tp_size > 1:
            dist.all_reduce(row_sum, op=dist.ReduceOp.SUM, group=tp_group)
        probabilities = exp_logits / row_sum[:, None]
        log_normalizer = row_max + row_sum.log()

        local_labels = labels - tp_rank * width
        owned = (local_labels >= 0) & (local_labels < width)
        target_logits = scaled[inverse, local_labels.clamp(0, width - 1)]
        target_logits = torch.where(owned, target_logits, 0.0)
        if tp_size > 1:
            dist.all_reduce(target_logits, op=dist.ReduceOp.SUM, group=tp_group)
        logprobs = target_logits - log_normalizer[inverse]

        expected_logits = (probabilities * scaled).sum(dim=-1)
        if tp_size > 1:
            dist.all_reduce(expected_logits, op=dist.ReduceOp.SUM, group=tp_group)
        entropy = log_normalizer - expected_logits

        ctx.save_for_backward(
            probabilities,
            scaled,
            log_normalizer,
            entropy,
            unique_rows,
            inverse,
            local_labels,
            owned,
        )
        ctx.input_shape = logits.shape
        ctx.input_dtype = logits.dtype
        ctx.temperature = temperature
        return logprobs, entropy[inverse]

    @staticmethod
    def backward(
        ctx, grad_logprobs: torch.Tensor | None, grad_entropy: torch.Tensor | None
    ):
        (
            probabilities,
            scaled,
            log_normalizer,
            entropy,
            unique_rows,
            inverse,
            local_labels,
            owned,
        ) = ctx.saved_tensors
        grad_unique = torch.zeros_like(probabilities)
        if grad_logprobs is not None:
            row_weights = torch.zeros_like(entropy).index_add_(
                0, inverse, grad_logprobs.float()
            )
            grad_unique -= probabilities * row_weights[:, None]
            grad_unique.index_put_(
                (inverse[owned], local_labels[owned]),
                grad_logprobs.float()[owned],
                accumulate=True,
            )
        if grad_entropy is not None:
            entropy_weights = torch.zeros_like(entropy).index_add_(
                0, inverse, grad_entropy.float()
            )
            grad_unique -= (
                probabilities
                * (scaled - log_normalizer[:, None] + entropy[:, None])
                * entropy_weights[:, None]
            )

        grad_input = torch.zeros(
            ctx.input_shape, device=scaled.device, dtype=torch.float32
        )
        grad_input.index_add_(0, unique_rows, grad_unique / ctx.temperature)
        return grad_input.to(ctx.input_dtype), None, None, None, None


def mapped_logprobs_entropy(
    logits: torch.Tensor,
    rows: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    tp_group: dist.ProcessGroup | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate semantic labels from compact logits without expanded vocab rows."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if rows.numel() == 0:
        empty = torch.empty(0, device=logits.device, dtype=torch.float32)
        return empty, empty
    return _MappedLogprobsEntropy.apply(logits, rows, labels, temperature, tp_group)
