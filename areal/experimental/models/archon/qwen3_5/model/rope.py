# SPDX-License-Identifier: Apache-2.0

import torch
from torch.distributed.tensor import DTensor


def _maybe_to_local(x: torch.Tensor) -> torch.Tensor:
    """Convert DTensor to local tensor if needed."""
    if isinstance(x, DTensor):
        return x.to_local()
    return x


def precompute_rope_cache(
    head_dim: int,
    max_seq_len: int,
    partial_rotary_factor: float = 1.0,
    base: float = 10_000.0,
) -> torch.Tensor:
    """Precompute partial RoPE cos/sin cache.

    Args:
        head_dim: Full head dimension.
        max_seq_len: Maximum sequence length.
        partial_rotary_factor: Fraction of head_dim for RoPE (e.g. 0.25).
        base: RoPE base frequency.

    Returns:
        RoPE cache tensor of shape ``[max_seq_len, rotary_dim * 2]`` where
        ``rotary_dim = int(head_dim * partial_rotary_factor)``.
        First half is cos, second half is sin.
    """
    rotary_dim = int(head_dim * partial_rotary_factor)
    freqs = 1.0 / (
        base
        ** (torch.arange(0, rotary_dim, 2)[: (rotary_dim // 2)].float() / rotary_dim)
    )
    t = torch.arange(max_seq_len, dtype=freqs.dtype)
    idx_theta = torch.outer(t, freqs).float()
    freqs = torch.cat([idx_theta, idx_theta], dim=-1)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def reshape_for_broadcast(
    rope_cache: torch.Tensor,
    x: torch.Tensor,
    positions: torch.Tensor | None = None,
    rotary_dim: int | None = None,
) -> torch.Tensor:
    """Reshape RoPE cache for broadcasting with input tensor.

    Args:
        rope_cache: RoPE tensor of shape ``[max_seqlen, rotary_dim * 2]``.
        x: Target tensor, shape ``[bsz, seq_len, num_heads, head_dim]``.
        positions: Position indices, shape ``(1, seqlen)`` or ``(bz, seqlen)``.
        rotary_dim: The rotary dimension (may differ from head_dim for partial RoPE).
            If *None*, falls back to ``rope_cache.shape[-1] // 2``.

    Returns:
        Reshaped RoPE tensor suitable for broadcasting.
    """
    if rotary_dim is None:
        rotary_dim = rope_cache.shape[-1] // 2
    bz, seqlen, _, _ = x.shape
    rope_width = rotary_dim * 2  # cos + sin

    if positions is None:
        rope_cache = rope_cache[0:seqlen]
        return rope_cache.view(-1, seqlen, 1, rope_width)
    elif positions.size(0) == 1:
        rope_cache = rope_cache[positions.squeeze(0)]
        return rope_cache.view(-1, seqlen, 1, rope_width)
    else:
        rope_cache_expanded = rope_cache[None, :, None, :].expand(bz, -1, -1, -1)
        rope_cache = torch.gather(
            rope_cache_expanded,
            dim=1,
            index=positions.view(bz, seqlen, 1, 1).expand(bz, seqlen, 1, rope_width),
        )
        return rope_cache


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    rope_cache: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply partial rotary position embedding to query and key tensors.

    Only the first ``rotary_dim`` dimensions participate in rotation;
    the remaining dimensions pass through unchanged (bit-exact).

    Args:
        xq: Query tensor, shape ``[bsz, seq_len, num_heads, head_dim]``.
        xk: Key tensor, shape ``[bsz, seq_len, num_kv_heads, head_dim]``.
        rope_cache: RoPE cache, shape ``[max_seq_len, rotary_dim * 2]``.
        positions: Position indices.

    Returns:
        Tuple of (rotated_q, rotated_k) with same shapes as inputs.
    """
    rope_cache = _maybe_to_local(rope_cache)
    if positions is not None:
        positions = _maybe_to_local(positions)

    rotary_dim = rope_cache.shape[-1] // 2
    rope_cache = reshape_for_broadcast(rope_cache, xq, positions, rotary_dim)

    cos = rope_cache[..., :rotary_dim].to(dtype=xq.dtype, device=xq.device)
    sin = rope_cache[..., rotary_dim:].to(dtype=xq.dtype, device=xq.device)

    # Split: only first rotary_dim dims participate in rotation
    xq_rot, xq_pass = xq[..., :rotary_dim], xq[..., rotary_dim:]
    xk_rot, xk_pass = xk[..., :rotary_dim], xk[..., rotary_dim:]

    xq_out = torch.cat([(xq_rot * cos) + (rotate_half(xq_rot) * sin), xq_pass], dim=-1)
    xk_out = torch.cat([(xk_rot * cos) + (rotate_half(xk_rot) * sin), xk_pass], dim=-1)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to match number of query heads.

    Args:
        x: Input tensor of shape ``[bs, slen, n_kv_heads, head_dim]``.
        n_rep: Number of repetitions.

    Returns:
        Tensor of shape ``[bs, slen, n_kv_heads * n_rep, head_dim]``.
    """
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        torch.unsqueeze(x, dim=3)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


__all__ = [
    "precompute_rope_cache",
    "rotate_half",
    "reshape_for_broadcast",
    "apply_rotary_emb",
    "repeat_kv",
]
