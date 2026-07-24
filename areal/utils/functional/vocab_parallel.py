# SPDX-License-Identifier: Apache-2.0

import functools
import math
import os
from collections.abc import Callable
from typing import TypeVar

import torch
from torch import distributed as dist

from areal.infra.platforms import is_npu_available

T = TypeVar("T", torch.Tensor, tuple[torch.Tensor, torch.Tensor])


def _gather_logprobs(
    logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0
):
    log_probs = torch.nn.functional.log_softmax(logits.float() / temperature, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return log_probs_labels


def _gather_logprobs_entropy(
    logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0
):
    log_probs = torch.nn.functional.log_softmax(logits.float() / temperature, dim=-1)
    entropy = -torch.sum(log_probs.exp() * log_probs, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return log_probs_labels, entropy


def _should_use_torch_compile() -> bool:
    return not is_npu_available


if _should_use_torch_compile():
    _gather_logprobs = torch.compile(_gather_logprobs)
    _gather_logprobs_entropy = torch.compile(_gather_logprobs_entropy)


def _chunked_apply(
    fn: Callable[[torch.Tensor, torch.Tensor], T],
    logits: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int = 1024,
) -> T:
    """Apply a function in chunks along the first dimension to reduce peak memory."""
    total_seqlen = logits.shape[0]
    assert total_seqlen > 0, "Input logits must have at least one element"
    results: list = []

    for i in range(0, total_seqlen, chunk_size):
        end_idx = min(i + chunk_size, total_seqlen)
        chunk_result = fn(logits[i:end_idx], labels[i:end_idx])
        results.append(chunk_result)

    # Handle single tensor vs tuple of tensors
    if isinstance(results[0], tuple):
        num_outputs = len(results[0])
        return tuple(torch.cat([r[i] for r in results]) for i in range(num_outputs))
    return torch.cat(results)


def _resolve_chunk_size(chunk_size: int) -> int:
    env_chunk_size = os.getenv("AREAL_LOGPROBS_CHUNK_SIZE")
    if env_chunk_size is None:
        return chunk_size
    try:
        resolved = int(env_chunk_size)
    except ValueError as e:
        raise ValueError(
            "AREAL_LOGPROBS_CHUNK_SIZE must be a positive integer, "
            f"got {env_chunk_size!r}"
        ) from e
    if resolved <= 0:
        raise ValueError(
            "AREAL_LOGPROBS_CHUNK_SIZE must be a positive integer, "
            f"got {env_chunk_size!r}"
        )
    return resolved


def _chunked_gather_logprobs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    chunk_size: int = 1024,
) -> torch.Tensor:
    fn = functools.partial(_gather_logprobs, temperature=temperature)
    return _chunked_apply(fn, logits, labels, chunk_size)


def _chunked_gather_logprobs_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    chunk_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    fn = functools.partial(_gather_logprobs_entropy, temperature=temperature)
    return _chunked_apply(fn, logits, labels, chunk_size)


class _VocabParallelLogProbs(torch.autograd.Function):
    """Compute log probabilities when logits are sharded on the vocab dimension.

    Given sharded logits [..., vocab_size/tp] and labels [...], computes:
        logprobs[i] = logits[i, labels[i]] - log(sum(exp(logits[i, :])))

    The input can have arbitrary leading dimensions (e.g., [batch, seq_len] or just
    [seq_len]). The labels indices are global (0 to vocab_size-1), and each TP rank
    only holds a partition of the vocabulary.

    Memory Optimization:
        Following Megatron's cross_entropy pattern, we use in-place operations to
        minimize memory allocations. The key optimization is in backward():

        - The gradient formula is: grad = one_hot(labels) - softmax
        - Since this only requires subtracting 1 at the label position and scaling,
          we can directly reuse the saved softmax tensor as grad_input (in-place).
        - This avoids allocating a new [*, vocab/tp] tensor for gradients.

        Forward saves only ONE large tensor:
        - softmax: [*, vocab/tp] - unavoidable for gradient computation

        Backward allocates NO new large tensors:
        - Reuses softmax directly as grad_input via in-place modifications

    Note:
        This implementation uses in-place operations on saved tensors for memory
        efficiency. As a result, it does NOT support:
        - `retain_graph=True` in backward()
        - Higher-order gradients (e.g., torch.autograd.grad with create_graph=True)

        These limitations are acceptable for typical RL training where only
        first-order gradients are needed and each backward is called once.
    """

    @staticmethod
    def forward(
        ctx,
        vocab_parallel_logits: torch.Tensor,
        labels: torch.Tensor,
        tp_group: dist.ProcessGroup,
    ) -> torch.Tensor:
        # Get TP rank info
        tp_rank = dist.get_rank(tp_group)

        # Calculate vocab partition boundaries for this rank
        partition_vocab_size = vocab_parallel_logits.size(-1)
        vocab_start_index = tp_rank * partition_vocab_size
        vocab_end_index = vocab_start_index + partition_vocab_size

        # Step 1: Numerical stability - subtract max
        logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=tp_group)

        # In-place subtraction following Megatron pattern
        normalized_logits = vocab_parallel_logits - logits_max

        # Step 2: Compute exp in-place and sum across all ranks
        exp_logits = normalized_logits.exp()
        sum_exp_logits = exp_logits.sum(dim=-1, keepdim=True)
        dist.all_reduce(sum_exp_logits, op=dist.ReduceOp.SUM, group=tp_group)

        # Step 3: Get the logit value at labels position
        labels_mask = (labels < vocab_start_index) | (labels >= vocab_end_index)
        masked_labels = labels.clone() - vocab_start_index
        masked_labels[labels_mask] = 0

        logits_2d = normalized_logits.view(-1, partition_vocab_size)
        masked_labels_1d = masked_labels.view(-1)
        arange_1d = torch.arange(logits_2d.size(0), device=logits_2d.device)

        predicted_logits_1d = logits_2d[arange_1d, masked_labels_1d]
        predicted_logits = predicted_logits_1d.view_as(labels)
        predicted_logits[labels_mask] = 0.0
        dist.all_reduce(predicted_logits, op=dist.ReduceOp.SUM, group=tp_group)

        # Step 4: Compute log probability
        log_sum_exp = sum_exp_logits.squeeze(-1).log()
        logprobs = predicted_logits - log_sum_exp

        # Step 5: Compute softmax in-place for backward (reuse exp_logits memory)
        softmax = exp_logits.div_(sum_exp_logits)
        ctx.save_for_backward(softmax, labels_mask, masked_labels_1d)

        return logprobs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        softmax, labels_mask, masked_labels_1d = ctx.saved_tensors

        # Gradient of logprobs w.r.t. logits: one_hot(labels) - softmax
        # Following Megatron's pattern: use softmax directly as grad_input base
        # and modify in-place where possible
        partition_vocab_size = softmax.size(-1)

        # Use softmax as the gradient base (will be modified)
        grad_input = softmax
        grad_2d = grad_input.view(-1, partition_vocab_size)
        arange_1d = torch.arange(grad_2d.size(0), device=grad_2d.device)

        # Subtract 1 at labels position (only for labels in this partition)
        # This gives: softmax - one_hot(labels)
        update_mask = ~labels_mask.view(-1)
        grad_2d[arange_1d, masked_labels_1d] -= update_mask.float()

        # Scale by grad_output (in-place)
        # Note: we want -(softmax - one_hot) = one_hot - softmax for logprobs gradient
        grad_input.mul_(grad_output.unsqueeze(-1))
        grad_input.neg_()

        return grad_input, None, None


class _VocabParallelLogProbsEntropy(torch.autograd.Function):
    """Compute both log probabilities and entropy when logits are sharded.

    Input tensors can have arbitrary leading dimensions:
        - logits: [..., vocab_size/tp]
        - labels: [...]

    This combines the computation to share intermediate results (softmax, sum_exp, etc.)
    and reduce redundant all-reduce operations compared to calling logprobs and entropy
    separately.

    Memory Optimization:
        Forward saves only ONE large tensor (softmax) plus a few small scalars.
        The entropy gradient is algebraically rewritten to avoid saving original logits:

            grad_entropy = softmax * (E[x] - x)
                         = softmax * (E[x] - log(softmax) - log(Z))
                         = softmax * (E[x] - log(Z)) - softmax * log(softmax)

        where E[x] = sum(softmax * logits) and log(Z) = log(sum(exp(logits))).

        Why we CANNOT reuse softmax in-place (unlike _VocabParallelLogProbs):
            The combined gradient requires multiple reads of the original softmax:

            1. grad_input = softmax * (E[x] - log(Z))   # first read
            2. grad_input -= xlogy(softmax, softmax)    # second read
            3. grad_input -= softmax * grad_logprobs    # third read

            If we modified softmax in step 1, steps 2 and 3 would get wrong values.
            In contrast, _VocabParallelLogProbs only needs: grad = softmax - one_hot,
            which can be done by subtracting 1 at one position then scaling - a single
            pass that allows full in-place reuse.

        Backward allocates ONE new large tensor:
            - grad_input: [*, vocab/tp] - created via `softmax * mean_x_minus_log_z`

        Memory comparison (seq=8192, vocab=152K, tp=2, fp32):
            - Naive approach: save both logits and softmax = ~4.7GB
            - Our approach: save only softmax = ~2.3GB (50% reduction in forward)
            - Backward: +2.3GB temporary for grad_input (unavoidable for correctness)

    Note:
        This implementation does NOT support:
        - `retain_graph=True` in backward()
        - Higher-order gradients (e.g., torch.autograd.grad with create_graph=True)

        These limitations are acceptable for typical RL training where only
        first-order gradients are needed and each backward is called once.

    Returns:
        logprobs: [...] log probability of labels tokens
        entropy: [...] entropy of the distribution
    """

    @staticmethod
    def forward(
        ctx,
        vocab_parallel_logits: torch.Tensor,
        labels: torch.Tensor,
        tp_group: dist.ProcessGroup,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Get TP rank info
        tp_rank = dist.get_rank(tp_group)
        partition_vocab_size = vocab_parallel_logits.size(-1)
        vocab_start_index = tp_rank * partition_vocab_size
        vocab_end_index = vocab_start_index + partition_vocab_size

        # Step 1: Numerical stability - subtract max (shared)
        logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=tp_group)

        # In-place subtraction following Megatron pattern
        normalized_logits = vocab_parallel_logits - logits_max

        # Step 2: Compute exp and sum_exp (shared)
        # Use in-place exp to reuse memory
        exp_logits = normalized_logits.exp()
        sum_exp_logits = exp_logits.sum(dim=-1, keepdim=True)
        dist.all_reduce(sum_exp_logits, op=dist.ReduceOp.SUM, group=tp_group)

        # Step 3: Compute softmax in-place (shared)
        # After this, exp_logits becomes softmax
        softmax = exp_logits.div_(sum_exp_logits)

        # Step 4: For logprobs - get labels logit
        labels_mask = (labels < vocab_start_index) | (labels >= vocab_end_index)
        masked_labels = labels.clone() - vocab_start_index
        masked_labels[labels_mask] = 0

        logits_2d = normalized_logits.view(-1, partition_vocab_size)
        masked_labels_1d = masked_labels.view(-1)
        arange_1d = torch.arange(logits_2d.size(0), device=logits_2d.device)

        predicted_logits_1d = logits_2d[arange_1d, masked_labels_1d]
        predicted_logits = predicted_logits_1d.view_as(labels)
        predicted_logits[labels_mask] = 0.0
        dist.all_reduce(predicted_logits, op=dist.ReduceOp.SUM, group=tp_group)

        # Step 5: For entropy - compute sum(softmax * logits).
        # vecdot performs the multiply-reduce without materializing another
        # [*, vocab/tp] tensor, which matters when profiling near memory limits.
        sum_softmax_times_logits = torch.linalg.vecdot(
            softmax, vocab_parallel_logits, dim=-1
        ).unsqueeze(-1)
        dist.all_reduce(sum_softmax_times_logits, op=dist.ReduceOp.SUM, group=tp_group)

        # Step 6: Compute final results
        log_sum_exp = sum_exp_logits.log()
        logprobs = predicted_logits - log_sum_exp.squeeze(-1)
        # entropy = log(Z) - E[x] = (max + log(sum_exp)) - sum_softmax_times_logits
        entropy = (logits_max + log_sum_exp - sum_softmax_times_logits).squeeze(-1)

        # Compute log(Z) for backward (small tensor: [*, 1])
        # log(Z) = max + log(sum_exp)
        log_z = logits_max + log_sum_exp

        # Save for backward - only ONE large tensor (softmax) instead of two
        # Memory savings: ~2.3GB for typical configs (seq=8192, vocab=152K, tp=2)
        ctx.save_for_backward(
            softmax,  # [*, vocab/tp] - the only large tensor
            sum_softmax_times_logits,  # [*, 1] - small
            log_z,  # [*, 1] - small
            labels_mask,  # [*] - small (bool)
            masked_labels_1d,  # [N] - small (int64)
        )
        ctx.partition_vocab_size = partition_vocab_size

        return logprobs, entropy

    @staticmethod
    def backward(ctx, grad_logprobs: torch.Tensor, grad_entropy: torch.Tensor) -> tuple:
        (
            softmax,
            sum_softmax_times_logits,
            log_z,
            labels_mask,
            masked_labels_1d,
        ) = ctx.saved_tensors
        partition_vocab_size = ctx.partition_vocab_size

        # Memory-optimized backward using in-place operations.
        # We compute gradients directly on softmax tensor to avoid extra allocations.
        #
        # Total gradient = grad_logprobs * (one_hot - softmax) + grad_entropy * softmax * (E[x] - x)
        #
        # Strategy: First compute entropy gradient (needs original softmax values),
        # then add logprobs gradient.

        # Step 1: Compute entropy gradient contribution
        # grad_entropy_contrib = softmax * ((E[x] - log(Z)) - log(softmax))
        #                     = softmax * (mean_x - log_z) - softmax * log(softmax)
        # Note: torch.xlogy handles 0 * log(0) = 0 correctly
        mean_x_minus_log_z = sum_softmax_times_logits - log_z  # [*, 1] small tensor

        # The gradient is computed in a single large tensor, grad_input, to minimize
        # peak memory usage. It is initialized here and then modified in-place.
        # Compute: softmax * (mean_x - log_z) - xlogy(softmax, softmax)
        # First: grad_input = softmax * mean_x_minus_log_z (broadcast, creates new tensor)
        grad_input = softmax * mean_x_minus_log_z
        # Subtract xlogy term in-place
        grad_input.sub_(torch.xlogy(softmax, softmax))
        # Scale by grad_entropy in-place
        grad_input.mul_(grad_entropy.unsqueeze(-1))

        # Step 2: Add logprobs gradient contribution
        # grad_logprobs_contrib = grad_logprobs * (one_hot(labels) - softmax)
        #                      = -grad_logprobs * softmax + grad_logprobs * one_hot
        # Add -softmax * grad_logprobs term without materializing the product.
        grad_input.addcmul_(softmax, grad_logprobs.unsqueeze(-1), value=-1)

        # Add one_hot * grad_logprobs at labels positions (only for labels in this partition)
        grad_2d = grad_input.view(-1, partition_vocab_size)
        arange_1d = torch.arange(grad_2d.size(0), device=grad_2d.device)
        update_mask = ~labels_mask.view(-1)
        grad_2d[arange_1d, masked_labels_1d] += update_mask * grad_logprobs.view(-1)

        return grad_input, None, None


def _all_reduce_if_needed(
    tensor: torch.Tensor,
    op: dist.ReduceOp.RedOpType,
    tp_group: dist.ProcessGroup | None,
) -> None:
    if tp_group is not None and dist.get_world_size(tp_group) > 1:
        dist.all_reduce(tensor, op=op, group=tp_group)


class _InplaceVocabParallelLogProbsEntropy(torch.autograd.Function):
    """Megatron-compatible destructive logprob and entropy computation.

    The input must be a contiguous CUDA FP32 tensor dedicated to loss computation.
    Forward overwrites logits with softmax probabilities, and backward overwrites the
    same storage with dlogits. Chunking happens inside this autograd node so it cannot
    create full-size SliceBackward buffers.
    """

    @staticmethod
    def forward(
        ctx,
        vocab_parallel_logits: torch.Tensor,
        labels: torch.Tensor,
        tp_group: dist.ProcessGroup | None,
        temperature: float,
        chunk_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from areal.utils.functional.vocab_parallel_kernels import (
            fused_exp_sum_inplace,
            fused_normalize_inplace,
            vocab_tile_count,
        )

        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        partition_vocab_size = vocab_parallel_logits.size(-1)
        logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)
        labels_1d = labels.reshape(-1)
        if logits_2d.size(0) != labels_1d.numel():
            raise ValueError(
                "logits leading dimensions must match labels: "
                f"got {tuple(vocab_parallel_logits.shape)} and {tuple(labels.shape)}"
            )

        tp_rank = dist.get_rank(tp_group) if tp_group is not None else 0
        vocab_start_index = tp_rank * partition_vocab_size
        local_targets = labels_1d - vocab_start_index
        target_mask = (local_targets < 0) | (local_targets >= partition_vocab_size)
        local_targets.masked_fill_(target_mask, -1)

        num_tokens = logits_2d.size(0)
        logprobs = torch.empty(num_tokens, dtype=torch.float32, device=logits_2d.device)
        entropy = torch.empty_like(logprobs)
        workspace_rows = min(chunk_size, num_tokens)
        partial_reduction = torch.empty(
            (2, workspace_rows, vocab_tile_count(partition_vocab_size)),
            dtype=torch.float32,
            device=logits_2d.device,
        )
        reduced_values = torch.empty(
            (3, workspace_rows), dtype=torch.float32, device=logits_2d.device
        )
        inv_temperature = 1.0 / temperature

        for start in range(0, num_tokens, chunk_size):
            end = min(start + chunk_size, num_tokens)
            rows = end - start
            chunk_logits = logits_2d[start:end]
            chunk_targets = local_targets[start:end]

            logits_max = chunk_logits.max(dim=-1).values
            _all_reduce_if_needed(logits_max, dist.ReduceOp.MAX, tp_group)

            row_indices = torch.arange(rows, device=logits_2d.device)
            predicted_logits = chunk_logits[
                row_indices, chunk_targets.clamp_min(0)
            ].masked_fill_(chunk_targets < 0, 0.0)
            predicted_logits.sub_(logits_max).mul_(inv_temperature)
            predicted_logits.masked_fill_(chunk_targets < 0, 0.0)

            partial_sums = partial_reduction[0, :rows]
            partial_weighted_sums = partial_reduction[1, :rows]
            fused_exp_sum_inplace(
                chunk_logits,
                logits_max,
                partial_sums,
                partial_weighted_sums,
                inv_temperature,
            )
            reduced = reduced_values[:, :rows]
            reduced[0].copy_(predicted_logits)
            reduced[1].copy_(partial_sums.sum(dim=-1))
            reduced[2].copy_(partial_weighted_sums.sum(dim=-1))
            if rows < workspace_rows:
                reduced_values[:, rows:].zero_()
            _all_reduce_if_needed(reduced_values, dist.ReduceOp.SUM, tp_group)

            sum_exp = reduced[1]
            log_sum_exp = sum_exp.log()
            logprobs[start:end] = reduced[0] - log_sum_exp
            entropy[start:end] = log_sum_exp - reduced[2] / sum_exp
            fused_normalize_inplace(chunk_logits, sum_exp)

        logprobs_output = logprobs.view_as(labels)
        entropy_output = entropy.view_as(labels)
        ctx.set_materialize_grads(False)
        ctx.mark_non_differentiable(entropy_output)
        torch.autograd.graph.increment_version(vocab_parallel_logits)
        ctx.save_for_backward(vocab_parallel_logits, local_targets)
        ctx.temperature = temperature
        ctx.chunk_size = chunk_size
        return logprobs_output, entropy_output

    @staticmethod
    def backward(
        ctx,
        grad_logprobs: torch.Tensor | None,
        grad_entropy: torch.Tensor | None,
    ) -> tuple:
        from areal.utils.functional.vocab_parallel_kernels import (
            fused_softmax_backward_inplace,
        )

        del grad_entropy
        softmax, local_targets = ctx.saved_tensors
        partition_vocab_size = softmax.size(-1)
        softmax_2d = softmax.view(-1, partition_vocab_size)
        grad_logprobs_1d = (
            grad_logprobs.reshape(-1).contiguous()
            if grad_logprobs is not None
            else None
        )
        num_tokens = softmax_2d.size(0)
        for start in range(0, num_tokens, ctx.chunk_size):
            end = min(start + ctx.chunk_size, num_tokens)
            fused_softmax_backward_inplace(
                softmax_2d[start:end],
                local_targets[start:end],
                grad_logprobs_1d[start:end] if grad_logprobs_1d is not None else None,
                1.0 / ctx.temperature,
            )

        torch.autograd.graph.increment_version(softmax)
        return softmax, None, None, None, None


def _vocab_parallel_logprobs(
    vocab_parallel_logits: torch.Tensor,
    labels: torch.Tensor,
    tp_group: dist.ProcessGroup,
    temperature: float = 1.0,
) -> torch.Tensor:
    if temperature != 1.0:
        logits = vocab_parallel_logits.float() / temperature
    else:
        logits = vocab_parallel_logits.float()
    return _VocabParallelLogProbs.apply(logits, labels, tp_group)


def _vocab_parallel_logprobs_entropy(
    vocab_parallel_logits: torch.Tensor,
    labels: torch.Tensor,
    tp_group: dist.ProcessGroup,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if temperature != 1.0:
        logits = vocab_parallel_logits.float() / temperature
    else:
        logits = vocab_parallel_logits.float()
    return _VocabParallelLogProbsEntropy.apply(logits, labels, tp_group)


def _inplace_vocab_parallel_logprobs_entropy(
    vocab_parallel_logits: torch.Tensor,
    labels: torch.Tensor,
    tp_group: dist.ProcessGroup | None,
    temperature: float = 1.0,
    chunk_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Destructively reuse CUDA FP32 logits as softmax and dlogits storage."""
    logprobs, entropy = _InplaceVocabParallelLogProbsEntropy.apply(
        vocab_parallel_logits, labels, tp_group, temperature, chunk_size
    )
    return logprobs, entropy


def gather_logprobs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    tp_group: dist.ProcessGroup | None = None,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Compute log probabilities with optional vocab parallelism for FSDP.

    Args:
        logits: Model logits with shape [..., vocab_size] or [..., vocab_size/tp]
            when tensor parallelism is enabled.
        labels: Token indices with shape [...] for which to compute log probabilities.
        temperature: Softmax temperature scaling. Default is 1.0.
        tp_group: If provided with tp_size > 1, uses vocab-parallel computation
            to avoid gathering the full vocab dimension across TP ranks.
        chunk_size: Chunk size for memory-efficient processing along the sequence
            dimension. Default is 1024.

    Returns:
        Log probabilities at the label positions with shape [...].
    """
    chunk_size = _resolve_chunk_size(chunk_size)
    if tp_group is not None and dist.get_world_size(tp_group) > 1:
        fn = functools.partial(
            _vocab_parallel_logprobs,
            tp_group=tp_group,
            temperature=temperature,
        )
        return _chunked_apply(fn, logits, labels, chunk_size)

    return _chunked_gather_logprobs(logits, labels, temperature, chunk_size)


def gather_logprobs_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    tp_group: dist.ProcessGroup | None = None,
    chunk_size: int = 1024,
    reuse_logits: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute log probabilities and entropy with optional vocab parallelism for FSDP.

    This function computes both values in a single pass, sharing intermediate results
    (softmax, sum_exp, etc.) to reduce redundant computation and all-reduce operations.

    Args:
        logits: Model logits with shape [..., vocab_size] or [..., vocab_size/tp]
            when tensor parallelism is enabled.
        labels: Token indices with shape [...] for which to compute log probabilities.
        temperature: Softmax temperature scaling. Default is 1.0.
        tp_group: If provided with tp_size > 1, uses vocab-parallel computation
            to avoid gathering the full vocab dimension across TP ranks.
        chunk_size: Chunk size for memory-efficient processing along the sequence
            dimension. Default is 1024.
        reuse_logits: Destructively reuse contiguous CUDA FP32 logits as the softmax
            and dlogits buffer. Intended for Megatron training after all other logits
            consumers have run. Entropy is non-differentiable on this path. Other
            devices and dtypes fall back to the regular path.

    Returns:
        A tuple of (logprobs, entropy):
            - logprobs: Log probabilities at the label positions with shape [...].
            - entropy: Entropy of the probability distribution with shape [...].
    """
    chunk_size = _resolve_chunk_size(chunk_size)
    if reuse_logits:
        from areal.utils.functional.vocab_parallel_kernels import (
            reusable_vocab_parallel_logits,
        )

        workspace = reusable_vocab_parallel_logits(logits)
        if workspace is not None:
            return _inplace_vocab_parallel_logprobs_entropy(
                workspace,
                labels,
                tp_group=tp_group,
                temperature=temperature,
                chunk_size=chunk_size,
            )
    if tp_group is not None and dist.get_world_size(tp_group) > 1:
        fn = functools.partial(
            _vocab_parallel_logprobs_entropy,
            tp_group=tp_group,
            temperature=temperature,
        )
        return _chunked_apply(fn, logits, labels, chunk_size)

    return _chunked_gather_logprobs_entropy(logits, labels, temperature, chunk_size)
