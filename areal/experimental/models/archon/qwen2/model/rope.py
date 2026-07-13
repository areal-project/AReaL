# SPDX-License-Identifier: Apache-2.0

# Adapted from torchtitan: torchtitan/models/qwen3/model/model.py

import torch
from torch.distributed.tensor import DTensor


def _maybe_to_local(x: torch.Tensor) -> torch.Tensor:
    """Convert DTensor to local tensor if needed."""
    if isinstance(x, DTensor):
        return x.to_local()
    return x


def precompute_rope_cache(
    dim: int, max_seq_len: int, base: float = 10_000.0
) -> torch.Tensor:
    """Precompute RoPE cos/sin cache.

    Args:
        dim: Head dimension.
        max_seq_len: Maximum sequence length.
        base: RoPE base frequency. Defaults to 10_000.0 (Qwen2 default).

    Returns:
        RoPE cache tensor of shape [max_seq_len, dim * 2].
        First half is cos, second half is sin.
    """
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # Create position indexes `[0, 1, ..., max_seq_len - 1]`
    t = torch.arange(max_seq_len, dtype=freqs.dtype, device=freqs.device)

    # Outer product of theta and position index; output tensor has
    # a shape of [max_seq_len, dim // 2]
    idx_theta = torch.outer(t, freqs).float()

    # We cache the cos and sin embeddings instead of the IDs. This helps
    # ensure we have correct behavior when training with bf16
    # Size: [max_seq_len, (dim * 2)]
    freqs = torch.cat([idx_theta, idx_theta], dim=-1)
    rope_cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
    return rope_cache


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def reshape_for_broadcast(
    rope_cache: torch.Tensor, x: torch.Tensor, positions: torch.Tensor | None = None
) -> torch.Tensor:
    """Reshape RoPE cache for broadcasting with input tensor.

    This function reshapes the RoPE cache to have the same shape as the target tensor 'x'
    for the purpose of broadcasting during element-wise operations.

    The input rope_cache tensor is assumed to be of shape (max_seqlen, head_dim * 2),
    and the first seqlen elements will be sliced, but dim must match x.

    Args:
        rope_cache: RoPE tensor (cos and sin) to be reshaped.
        x: Target tensor for broadcasting compatibility.
            Shape is [bsz, seq_len, num_heads, head_dim].
        positions: Position indices used to access/shuffle RoPE cache.
            Shape is (1, seqlen) or (bz, seqlen). Defaults to None.

    Returns:
        Reshaped RoPE tensor.
    """
    ndim = x.ndim
    assert ndim > 1
    bz, seqlen, _, head_dim = x.shape
    if positions is None:
        rope_cache = rope_cache[0:seqlen]
        shape = [-1, seqlen, 1, head_dim * 2]
        return rope_cache.view(*shape)
    elif positions.size(0) == 1:
        rope_cache = rope_cache[positions.squeeze(0)]
        shape = [-1, seqlen, 1, head_dim * 2]
        return rope_cache.view(*shape)
    else:
        rope_cache_expanded = rope_cache[None, :, None, :].expand(bz, -1, -1, -1)
        rope_cache = torch.gather(
            rope_cache_expanded,
            dim=1,
            index=positions.view(bz, seqlen, 1, 1).expand(bz, seqlen, 1, head_dim * 2),
        )
        return rope_cache


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    rope_cache: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key tensors.

    Args:
        xq: Query tensor, shape [bsz, seq_len, num_heads, head_dim].
        xk: Key tensor, shape [bsz, seq_len, num_kv_heads, head_dim].
        rope_cache: RoPE cache tensor, shape [max_seq_len, head_dim * 2].
        positions: Position indices, shape [1, seq_len] or [bsz, seq_len].
            Defaults to None (sequential positions).

    Returns:
        Tuple of (rotated_q, rotated_k) with same shapes as inputs.
    """
    head_dim = xq.shape[-1]

    # Convert DTensor to local tensor for compatibility with local Q/K tensors
    rope_cache = _maybe_to_local(rope_cache)
    if positions is not None:
        positions = _maybe_to_local(positions)

    rope_cache = reshape_for_broadcast(rope_cache, xq, positions)

    # [bsz, seq_len, 1, head_dim]
    cos = rope_cache[..., :head_dim].to(dtype=xq.dtype, device=xq.device)
    sin = rope_cache[..., head_dim:].to(dtype=xq.dtype, device=xq.device)

    # xq:  [bsz, seq_len, num_heads, head_dim]
    # xk:  [bsz, seq_len, num_kv_heads, head_dim]
    xq_out = (xq * cos) + (rotate_half(xq) * sin)
    xk_out = (xk * cos) + (rotate_half(xk) * sin)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to match number of query heads.

    Equivalent to torch.repeat_interleave(x, dim=2, repeats=n_rep).

    Args:
        x: Input tensor of shape [bs, slen, n_kv_heads, head_dim].
        n_rep: Number of repetitions.

    Returns:
        Tensor of shape [bs, slen, n_kv_heads * n_rep, head_dim].
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
