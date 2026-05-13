# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from areal.models.tree_attn.module_archon import TreeAttentionMeta

__all__ = ["varlen_attn", "VarlenAttentionWrapper"]


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Internal custom op calling Flash Attention kernel.

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
        softmax_lse: Log-sum-exp of attention scores
        rng_state: RNG state (unused, dropout=0)
    """
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
    # Placeholder rng_state since dropout is disabled
    rng_state_ = torch.zeros((2,), dtype=torch.uint64, device=query.device)
    return output, softmax_lse, rng_state_


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
    lse: torch.Tensor,
    cu_seq_q: torch.Tensor,
    cu_seq_k: torch.Tensor,
    max_q: int,
    max_k: int,
    is_causal: bool,
    rng_state: torch.Tensor,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward pass calling Flash Attention backward kernel."""
    unused = torch.empty(0, device=query.device)

    dq, dk, dv = torch.ops.aten._flash_attention_backward(
        grad_out,
        query,
        key,
        value,
        out,
        lse,
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
    out, lse, rng_state = output

    ctx.save_for_backward(query, key, value, cu_seq_q, cu_seq_k, out, lse, rng_state)

    ctx.max_q = max_q
    ctx.max_k = max_k
    ctx.is_causal = is_causal
    ctx.scale = scale


def _backward(
    ctx: Any, grad_out: torch.Tensor, grad_lse: torch.Tensor, grad_rng: torch.Tensor
) -> tuple[torch.Tensor | None, ...]:
    """Compute gradients for backward pass."""
    query, key, value, cu_seq_q, cu_seq_k, out, lse, rng_state = ctx.saved_tensors

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
        lse,
        cu_seq_q,
        cu_seq_k,
        max_q,
        max_k,
        is_causal,
        rng_state,
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
    out, _, _ = torch.ops.areal._varlen_attn(
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
        cu_seqlens_k: torch.Tensor | None = None,
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
        cu_seqlens_q_i32 = cu_seqlens.to(torch.int32)

        if cu_seqlens_k is None:
            cu_seqlens_k_i32 = cu_seqlens_q_i32
        else:
            cu_seqlens_k_i32 = cu_seqlens_k.to(torch.int32)

        # Call varlen_attn (self-attention: q and k have same cu_seqlens)
        out = varlen_attn(
            q_3d,
            k_3d,
            v_3d,
            cu_seqlens_q_i32,
            cu_seqlens_k_i32,
            max_seqlen,
            max_seqlen,
            is_causal=True,
            scale=scale,
        )

        # Transpose back: [T, H, D] -> [1, H, T, D]
        return out.transpose(0, 1).unsqueeze(0)
