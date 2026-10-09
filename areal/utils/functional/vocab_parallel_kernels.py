# SPDX-License-Identifier: Apache-2.0

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


_BLOCK_SIZE = 1024


if TRITON_AVAILABLE:

    @triton.jit
    def _exp_sum_inplace_kernel(
        logits,
        row_max,
        partial_sums,
        partial_weighted_sums,
        n_rows,
        n_cols,
        n_tiles,
        inv_temperature,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(axis=0)
        tile = tl.program_id(axis=1)
        cols = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = (row < n_rows) & (cols < n_cols)
        offsets = row * n_cols + cols

        values = tl.load(logits + offsets, mask=mask, other=-float("inf"))
        maximum = tl.load(row_max + row)
        shifted = (values - maximum) * inv_temperature
        exponentials = tl.exp(shifted)
        tl.store(logits + offsets, exponentials, mask=mask)
        tl.store(
            partial_sums + row * n_tiles + tile,
            tl.sum(tl.where(mask, exponentials, 0.0), axis=0),
        )
        tl.store(
            partial_weighted_sums + row * n_tiles + tile,
            tl.sum(tl.where(mask, shifted * exponentials, 0.0), axis=0),
        )

    @triton.jit
    def _normalize_inplace_kernel(
        logits,
        sum_exp,
        n_rows,
        n_cols,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(axis=0)
        tile = tl.program_id(axis=1)
        cols = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = (row < n_rows) & (cols < n_cols)
        offsets = row * n_cols + cols

        exponentials = tl.load(logits + offsets, mask=mask, other=0.0)
        denominator = tl.load(sum_exp + row)
        probabilities = exponentials / denominator
        tl.store(logits + offsets, probabilities, mask=mask)

    @triton.jit
    def _softmax_backward_inplace_kernel(
        softmax,
        local_targets,
        grad_logprobs,
        n_rows,
        n_cols,
        inv_temperature,
        HAS_LOGPROBS_GRAD: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(axis=0)
        tile = tl.program_id(axis=1)
        cols = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = (row < n_rows) & (cols < n_cols)
        offsets = row * n_cols + cols
        probabilities = tl.load(softmax + offsets, mask=mask, other=0.0)
        gradient = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

        if HAS_LOGPROBS_GRAD:
            logprob_gradient = tl.load(grad_logprobs + row)
            target = tl.load(local_targets + row)
            gradient = -probabilities * logprob_gradient
            gradient += tl.where(
                mask & (target >= 0) & (cols == target), logprob_gradient, 0.0
            )

        tl.store(softmax + offsets, gradient * inv_temperature, mask=mask)

    @triton.jit
    def _logits_backward_inplace_kernel(
        logits,
        row_max,
        sum_exp,
        local_targets,
        grad_logprobs,
        n_rows,
        n_cols,
        inv_temperature,
        logit_scale,
        HAS_LOGPROBS_GRAD: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(axis=0)
        tile = tl.program_id(axis=1)
        cols = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = (row < n_rows) & (cols < n_cols)
        offsets = row * n_cols + cols

        values = tl.load(logits + offsets, mask=mask, other=-float("inf"))
        maximum = tl.load(row_max + row)
        denominator = tl.load(sum_exp + row)
        probabilities = tl.exp((values - maximum) * inv_temperature) / denominator
        gradient = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

        if HAS_LOGPROBS_GRAD:
            logprob_gradient = tl.load(grad_logprobs + row)
            target = tl.load(local_targets + row)
            gradient = -probabilities * logprob_gradient
            gradient += tl.where(
                mask & (target >= 0) & (cols == target), logprob_gradient, 0.0
            )

        tl.store(
            logits + offsets,
            gradient * inv_temperature * logit_scale,
            mask=mask,
        )


def _check_input(tensor: torch.Tensor) -> None:
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for fused vocab-parallel kernels")
    if not tensor.is_cuda or tensor.dtype is not torch.float32:
        raise ValueError("Fused vocab-parallel kernels require CUDA FP32 tensors")
    if tensor.ndim != 2 or not tensor.is_contiguous():
        raise ValueError("Fused vocab-parallel kernels require contiguous 2D tensors")


def fused_exp_sum_inplace(
    logits: torch.Tensor,
    row_max: torch.Tensor,
    partial_sums: torch.Tensor,
    partial_weighted_sums: torch.Tensor,
    inv_temperature: float,
) -> None:
    """Overwrite logits with shifted exponentials and emit deterministic tile sums."""
    _check_input(logits)
    n_rows, n_cols = logits.shape
    n_tiles = triton.cdiv(n_cols, _BLOCK_SIZE)
    _exp_sum_inplace_kernel[(n_rows, n_tiles)](
        logits,
        row_max,
        partial_sums,
        partial_weighted_sums,
        n_rows,
        n_cols,
        n_tiles,
        inv_temperature,
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=8,
        num_stages=1,
    )


def fused_normalize_inplace(
    logits: torch.Tensor,
    sum_exp: torch.Tensor,
) -> None:
    """Overwrite exponentials with normalized softmax probabilities."""
    _check_input(logits)
    n_rows, n_cols = logits.shape
    n_tiles = triton.cdiv(n_cols, _BLOCK_SIZE)
    _normalize_inplace_kernel[(n_rows, n_tiles)](
        logits,
        sum_exp,
        n_rows,
        n_cols,
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=8,
        num_stages=1,
    )


def fused_softmax_backward_inplace(
    softmax: torch.Tensor,
    local_targets: torch.Tensor,
    grad_logprobs: torch.Tensor | None,
    inv_temperature: float,
) -> None:
    """Overwrite softmax with the selected-token logprob gradient."""
    _check_input(softmax)
    n_rows, n_cols = softmax.shape
    n_tiles = triton.cdiv(n_cols, _BLOCK_SIZE)
    dummy = softmax
    _softmax_backward_inplace_kernel[(n_rows, n_tiles)](
        softmax,
        local_targets,
        grad_logprobs if grad_logprobs is not None else dummy,
        n_rows,
        n_cols,
        inv_temperature,
        HAS_LOGPROBS_GRAD=grad_logprobs is not None,
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=8,
        num_stages=1,
    )


def fused_logits_backward_inplace(
    logits: torch.Tensor,
    row_max: torch.Tensor,
    sum_exp: torch.Tensor,
    local_targets: torch.Tensor,
    grad_logprobs: torch.Tensor | None,
    inv_temperature: float,
    logit_scale: float = 1.0,
) -> None:
    """Overwrite recomputed logits with selected-token logprob gradients."""
    _check_input(logits)
    n_rows, n_cols = logits.shape
    n_tiles = triton.cdiv(n_cols, _BLOCK_SIZE)
    dummy = logits
    _logits_backward_inplace_kernel[(n_rows, n_tiles)](
        logits,
        row_max,
        sum_exp,
        local_targets,
        grad_logprobs if grad_logprobs is not None else dummy,
        n_rows,
        n_cols,
        inv_temperature,
        logit_scale,
        HAS_LOGPROBS_GRAD=grad_logprobs is not None,
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=8,
        num_stages=1,
    )


def reusable_vocab_parallel_logits(logits: torch.Tensor) -> torch.Tensor | None:
    """Return logits when they exclusively cover their underlying storage."""
    workspace = logits
    if logits._base is not None:
        base = logits._base
        if (
            not logits.is_contiguous()
            or logits.storage_offset() != 0
            or logits.numel() != base.numel()
            or not base.is_contiguous()
        ):
            return None
        workspace = base

    if (
        not TRITON_AVAILABLE
        or not workspace.is_cuda
        or workspace.dtype is not torch.float32
        or not workspace.is_contiguous()
        or workspace.is_leaf
    ):
        return None
    return workspace


def fused_vocab_parallel_available(logits: torch.Tensor) -> bool:
    return reusable_vocab_parallel_logits(logits) is not None


def vocab_tile_count(vocab_size: int) -> int:
    return (vocab_size + _BLOCK_SIZE - 1) // _BLOCK_SIZE
