# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from areal.models.tree_attn.module_archon import TreeAttentionMeta

__all__ = ["varlen_attn", "VarlenAttentionWrapper"]

# NPU npu_fusion_attention pre_tockens/next_tockens upper bound (INT32 max ~2.1B).
_MAX_SEQ_TOKENS = 2147483647


def _is_npu_device(tensor: torch.Tensor) -> bool:
    """Check if tensor is on NPU device."""
    return tensor.device.type == "npu"


def _default_scale(head_dim: int, scale: float | None) -> float:
    """Return the attention scale, defaulting to 1/sqrt(head_dim)."""
    return scale if scale is not None else 1.0 / (head_dim**0.5)


def _cu_seqlens_to_actual(cu_seqlens: torch.Tensor) -> list[int]:
    """Convert cumulative sequence lengths [0, s1, s1+s2, ...] to actual lengths [s1, s1+s2, ...]."""
    return cu_seqlens[1:].tolist()


def _get_npu_sparse_config(
    is_causal: bool,
    max_q: int,
    max_k: int,
    device: torch.device,
) -> tuple[torch.Tensor | None, int]:
    """Return (atten_mask, sparse_mode) for NPU varlen attention.

    When is_causal=True, builds an explicit bool causal mask and uses sparse_mode=1
    (allMask). When False, returns (None, 0) for default (no mask) behavior.
    """
    if is_causal:
        return _make_causal_mask_npu(max_q, max_k, device), 1
    return None, 0


# ── Custom Op: Forward ───────────────────────────────────────────────


@torch.library.custom_op("areal::_varlen_attn", mutates_args={})
def _varlen_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seq_q: torch.Tensor,
    cu_seq_k: torch.Tensor,
    max_q: int,
    max_k: int,
    is_causal: bool = False,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Internal custom op calling Flash Attention kernel.

    Dispatches to CUDA (_flash_attention_forward) or NPU (npu_fusion_attention)
    based on the device of the input tensors.

    Args:
        query: Query tensor, shape (T_q, H, D)
        key: Key tensor, shape (T_k, H, D)
        value: Value tensor, shape (T_k, H, D)
        cu_seq_q: Cumulative sequence lengths for queries, shape (N+1,)
        cu_seq_k: Cumulative sequence lengths for keys, shape (N+1,)
        max_q: Maximum query sequence length
        max_k: Maximum key sequence length
        is_causal: Whether to apply causal masking
        scale: Optional scale factor for attention scores

    Returns:
        output: Attention output, shape (T_q, H, D)
        lse_or_max: CUDA: softmax_lse [H, T_q]; NPU: softmax_max [T_q, H, S]
        softmax_sum: CUDA: placeholder zeros [2]; NPU: softmax_sum [T_q, H, S]
        aux: CUDA: placeholder zeros [2]; NPU: [seed, offset] [2]
    """
    if _is_npu_device(query):
        return _varlen_attn_npu(
            query, key, value, cu_seq_q, cu_seq_k, max_q, max_k, is_causal, scale
        )

    # CUDA path: _flash_attention_forward
    output, softmax_lse, rng_state, _, _ = torch.ops.aten._flash_attention_forward(
        query,
        key,
        value,
        cu_seq_q,
        cu_seq_k,
        max_q,
        max_k,
        0.0,  # dropout_p hardcoded to 0.0
        is_causal,
        return_debug_mask=False,
        scale=scale,
    )
    # CUDA path keeps softmax_lse in its native [H, T_q] shape — no format conversion.
    # NPU path returns [T_q, H, S] shaped tensors. custom_op requires a fixed *count*
    # of returns, but shapes may differ per device. _backward dispatches by device and
    # handles each format accordingly.
    softmax_sum = torch.zeros(2, dtype=torch.float, device=query.device)
    aux = torch.zeros(2, device=query.device)
    return output, softmax_lse, softmax_sum, aux


def _make_causal_mask_npu(
    max_q: int,
    max_k: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Build a bool causal mask for npu_fusion_attention varlen mode.

    NPU atten_mask semantics: 1 = masked out (not attend), 0 = attend.
    For causal attention, the upper triangle is masked (1), lower triangle is kept (0).

    In varlen mode, the mask shape is [maxSq, maxSkv] (SS format) and is applied
    per-sequence: each sequence uses the top-left Lq x Lkv submatrix where L is the
    sequence length. Sequence isolation is handled by actual_seq_qlen/kvlen, not the mask.

    Returns None for non-causal attention (no mask needed).
    """
    return torch.triu(
        torch.ones(max_q, max_k, dtype=torch.bool, device=device), diagonal=1
    )


def _varlen_attn_npu(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seq_q: torch.Tensor,
    cu_seq_k: torch.Tensor,
    max_q: int,
    max_k: int,
    is_causal: bool,
    scale: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """NPU forward using torch_npu.npu_fusion_attention (TND varlen layout).

    NOTE: When atten_mask=None, npu_fusion_attention ignores sparse_mode and
    computes full (non-causal) attention regardless of the sparse_mode value.
    Therefore we must pass an explicit causal mask when is_causal=True, using
    sparse_mode=1 (allMask) which accepts an arbitrary [maxSq, maxSkv] bool mask.
    sparse_mode=3 (rightDownCausal) requires a fixed [2048, 2048] compressed mask
    which is incompatible with dynamic max_seqlen in varlen scenarios.
    """
    import torch_npu

    head_num = query.size(1)
    scale_val = _default_scale(query.size(-1), scale)

    actual_seq_qlen = _cu_seqlens_to_actual(cu_seq_q)
    actual_seq_kvlen = _cu_seqlens_to_actual(cu_seq_k)

    atten_mask, sparse_mode = _get_npu_sparse_config(
        is_causal, max_q, max_k, query.device
    )

    output, softmax_max, softmax_sum, _, seed, offset, _ = (
        torch_npu.npu_fusion_attention(
            query,
            key,
            value,
            head_num=head_num,
            input_layout="TND",
            pse=None,
            padding_mask=None,
            atten_mask=atten_mask,
            scale=scale_val,
            keep_prob=1.0,
            pre_tockens=_MAX_SEQ_TOKENS,
            next_tockens=_MAX_SEQ_TOKENS,
            inner_precise=0,
            prefix=None,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_kvlen,
            sparse_mode=sparse_mode,
        )
    )

    # Keep softmax_max/softmax_sum in their original shape [T, H, S]
    # and encode seed/offset as tensor for autograd
    aux = torch.tensor([seed, offset], dtype=torch.int64, device=query.device)

    return output, softmax_max, softmax_sum, aux


@_varlen_attn.register_fake
def _varlen_attn_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seq_q: torch.Tensor,
    cu_seq_k: torch.Tensor,
    max_q: int,
    max_k: int,
    is_causal: bool = False,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fake implementation for meta tensor computation and tracing."""
    output = torch.empty_like(query)

    # For varlen path: logsumexp shape is (num_heads, total_q)
    total_q = query.size(0)
    num_heads = query.size(1)
    logsumexp = torch.empty(
        (num_heads, total_q), dtype=torch.float, device=query.device
    )

    rng_state = torch.empty((2,), dtype=torch.uint64, device=query.device)

    return output, logsumexp, rng_state


# ── Custom Op: Backward ──────────────────────────────────────────────


@torch.library.custom_op("areal::_varlen_attn_backward", mutates_args={})
def _varlen_attn_backward(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    lse_or_max: torch.Tensor,
    softmax_sum: torch.Tensor,
    aux: torch.Tensor,
    cu_seq_q: torch.Tensor,
    cu_seq_k: torch.Tensor,
    max_q: int,
    max_k: int,
    is_causal: bool,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward pass for varlen attention, dispatching to CUDA or NPU."""
    if _is_npu_device(query):
        return _varlen_attn_backward_npu(
            grad_out,
            query,
            key,
            value,
            out,
            lse_or_max,
            softmax_sum,
            aux,
            cu_seq_q,
            cu_seq_k,
            max_q,
            max_k,
            is_causal,
            scale,
        )

    # CUDA path: lse_or_max is already softmax_lse [H, T_q] — use directly
    rng_state = torch.zeros((2,), dtype=torch.uint64, device=query.device)
    unused = torch.empty(0, device=query.device)

    dq, dk, dv = torch.ops.aten._flash_attention_backward(
        grad_out,
        query,
        key,
        value,
        out,
        lse_or_max,
        cu_seq_q,
        cu_seq_k,
        max_q,
        max_k,
        0.0,  # dropout_p
        is_causal,
        rng_state,
        unused,
        scale=scale,
    )
    return dq, dk, dv


def _varlen_attn_backward_npu(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    softmax_max: torch.Tensor,
    softmax_sum: torch.Tensor,
    aux: torch.Tensor,
    cu_seq_q: torch.Tensor,
    cu_seq_k: torch.Tensor,
    max_q: int,
    max_k: int,
    is_causal: bool,
    scale: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """NPU backward using torch_npu.npu_fusion_attention_grad.

    Must pass the same causal mask and sparse_mode as the forward pass.
    NOTE (#2): The causal mask is rebuilt here rather than cached from forward.
    Rebuilding is *better* than caching for memory: the mask is created, used, and
    freed in each pass. Caching via ctx would hold the mask in memory throughout
    the forward-to-backward gap, increasing peak memory by one mask size
    (e.g. [65536, 65536] bool = 512 MB). The mask itself is cheap to build
    (torch.triu of ones, pure memory init).
    """
    import torch_npu

    head_num = query.size(1)
    scale_val = _default_scale(query.size(-1), scale)

    seed = int(aux[0].item())
    offset = int(aux[1].item())

    actual_seq_qlen = _cu_seqlens_to_actual(cu_seq_q)
    actual_seq_kvlen = _cu_seqlens_to_actual(cu_seq_k)

    atten_mask, sparse_mode = _get_npu_sparse_config(
        is_causal, max_q, max_k, query.device
    )

    dq, dk, dv, _, _ = torch_npu.npu_fusion_attention_grad(
        query,
        key,
        value,
        grad_out,  # dy
        head_num,
        "TND",
        softmax_max=softmax_max,
        softmax_sum=softmax_sum,
        attention_in=out,
        atten_mask=atten_mask,
        scale_value=scale_val,
        keep_prob=1.0,
        pre_tockens=_MAX_SEQ_TOKENS,
        next_tockens=_MAX_SEQ_TOKENS,
        inner_precise=0,
        seed=seed,
        offset=offset,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
        sparse_mode=sparse_mode,
    )

    return dq, dk, dv


@_varlen_attn_backward.register_fake
def _varlen_attn_backward_fake(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    cu_seq_q: torch.Tensor,
    cu_seq_k: torch.Tensor,
    max_q: int,
    max_k: int,
    is_causal: bool,
    rng_state: torch.Tensor,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fake implementation for backward tracing."""
    grad_query = torch.empty_like(query)
    grad_key = torch.empty_like(key)
    grad_value = torch.empty_like(value)
    return grad_query, grad_key, grad_value


# ── Autograd Registration ────────────────────────────────────────────


def _setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
    """Save tensors for backward pass."""
    query, key, value, cu_seq_q, cu_seq_k, max_q, max_k, is_causal, scale = inputs
    out, lse_or_max, softmax_sum, aux = output

    ctx.save_for_backward(
        query, key, value, cu_seq_q, cu_seq_k, out, lse_or_max, softmax_sum, aux
    )

    ctx.max_q = max_q
    ctx.max_k = max_k
    ctx.is_causal = is_causal
    ctx.scale = scale


def _backward(
    ctx: Any,
    grad_out: torch.Tensor,
    grad_lse: torch.Tensor,
    grad_sum: torch.Tensor,
    grad_aux: torch.Tensor,
) -> tuple[torch.Tensor | None, ...]:
    """Compute gradients for backward pass."""
    # grad_sum and grad_aux correspond to non-differentiable outputs (softmax_sum, aux).
    # PyTorch autograd passes zero tensors (not None) for these; they carry no gradient
    # signal and are safely ignored.

    (
        query,
        key,
        value,
        cu_seq_q,
        cu_seq_k,
        out,
        lse_or_max,
        softmax_sum,
        aux,
    ) = ctx.saved_tensors

    max_q = ctx.max_q
    max_k = ctx.max_k
    is_causal = ctx.is_causal
    scale = ctx.scale

    dq, dk, dv = torch.ops.areal._varlen_attn_backward(
        grad_out,
        query,
        key,
        value,
        out,
        lse_or_max,
        softmax_sum,
        aux,
        cu_seq_q,
        cu_seq_k,
        max_q,
        max_k,
        is_causal,
        scale,
    )
    # Return gradients for all inputs (None for non-tensor inputs)
    return dq, dk, dv, None, None, None, None, None, None


_varlen_attn.register_autograd(_backward, setup_context=_setup_context)


# ── Public API ────────────────────────────────────────────────────────


def varlen_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seq_q: torch.Tensor,
    cu_seq_k: torch.Tensor,
    max_q: int,
    max_k: int,
    is_causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """Compute variable-length attention using Flash Attention.

    This function is similar to scaled_dot_product_attention but optimized for
    variable-length sequences using cumulative sequence position tensors.

    Args:
        query: Query tensor, shape (T_q, H, D)
        key: Key tensor, shape (T_k, H, D)
        value: Value tensor, shape (T_k, H, D)
        cu_seq_q: Cumulative sequence positions for queries, shape (N+1,)
            Example: [0, 100, 250, 538] for 3 sequences with lengths 100, 150, 288
        cu_seq_k: Cumulative sequence positions for keys/values, shape (N+1,)
        max_q: Maximum query sequence length in the batch
        max_k: Maximum key/value sequence length in the batch
        is_causal: If True, applies causal masking (default: False)
        scale: Optional scaling factor for attention scores.
               Defaults to 1/sqrt(head_dim).

    Returns:
        Attention output tensor, shape (T_q, H, D)

    Shape legend:
        - N: Number of sequences in the batch
        - T_q: Total query tokens (sum of all query sequence lengths)
        - T_k: Total key/value tokens (sum of all key/value sequence lengths)
        - H: Number of attention heads
        - D: Head dimension
    """
    out, _, _, _ = torch.ops.areal._varlen_attn(
        query, key, value, cu_seq_q, cu_seq_k, max_q, max_k, is_causal, scale
    )
    return out


# ── Wrapper for Archon Engine ─────────────────────────────────────────


class VarlenAttentionWrapper(nn.Module):
    """Wrapper adapting varlen_attn for Archon Engine's 4D tensor format.

    Archon Engine uses 4D tensors [batch, heads, seq_len, head_dim],
    while varlen_attn expects 3D tensors [total_tokens, heads, head_dim].
    This wrapper handles the shape conversion.

    For packed sequences in Archon, batch is always 1.
    """

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        scale: float | None = None,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        tree_attn_meta: TreeAttentionMeta | None = None,
    ) -> torch.Tensor:
        """Compute attention with varlen_attn.

        Args:
            q: Query tensor, shape [batch, heads, seq_len, head_dim]
            k: Key tensor, shape [batch, heads, seq_len, head_dim]
            v: Value tensor, shape [batch, heads, seq_len, head_dim]
            scale: Optional scale factor for attention scores
            cu_seqlens: Cumulative sequence lengths, shape [num_seqs + 1]
            max_seqlen: Maximum sequence length
            tree_attn_meta: Unused. Accepted for interface compatibility with
                TreeAttentionWrapper.

        Returns:
            Attention output, shape [batch, heads, seq_len, head_dim]
        """
        # Input: [batch, heads, seq_len, head_dim]
        # varlen_attn expects: [total_tokens, heads, head_dim]
        batch, n_heads, seq_len, head_dim = q.shape

        # For packed sequences, batch should be 1
        assert batch == 1, (
            f"VarlenAttentionWrapper expects batch=1 for packed sequences, "
            f"got batch={batch}"
        )

        # Transpose: [1, H, T, D] -> [T, H, D]
        q_3d = q.squeeze(0).transpose(0, 1).contiguous()
        k_3d = k.squeeze(0).transpose(0, 1).contiguous()
        v_3d = v.squeeze(0).transpose(0, 1).contiguous()

        # Ensure cu_seqlens is int32 (required by flash_attn)
        cu_seqlens_i32 = cu_seqlens.to(torch.int32)

        # Call varlen_attn (self-attention: q and k have same cu_seqlens)
        out = varlen_attn(
            q_3d,
            k_3d,
            v_3d,
            cu_seqlens_i32,
            cu_seqlens_i32,
            max_seqlen,
            max_seqlen,
            is_causal=True,
            scale=scale,
        )

        # Transpose back: [T, H, D] -> [1, H, T, D]
        return out.transpose(0, 1).unsqueeze(0)
