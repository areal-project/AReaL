# SPDX-License-Identifier: Apache-2.0
"""Apply and remove the rotary position transform exactly as Hugging Face Qwen2 stores keys.

Hugging Face caches keys after rotation:
    stored = k * cos + rotate_half(k) * sin
with, for head dimension ``d`` and position ``p``,
    inv_freq[i] = theta ** (-2 i / d), i in [0, d / 2), emb = cat(freqs, freqs),
    cos = cos(p * emb), sin = sin(p * emb).
``rotate_half`` is a quarter-turn on each (i, i + d/2) pair, so the inverse is
    k = stored * cos - rotate_half(stored) * sin.
Contract: ``x`` is ``[batch, heads, tokens, d]`` with ``d`` even; ``positions`` is an integer
``[batch, tokens]`` tensor. Only the default rope type (no scaling) is supported.
Angles are computed in float32, as Hugging Face does, unless ``dtype`` is float64.
"""

import torch


def rotary_cos_sin(
    positions: torch.Tensor, *, head_dim: int, theta: float, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``cos`` and ``sin`` of shape ``[batch, tokens, head_dim]`` for the positions."""
    if positions.ndim != 2:
        raise ValueError(
            f"positions must be [batch, tokens], got shape {tuple(positions.shape)}"
        )
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for rotate_half, got {head_dim}")
    if theta <= 0:
        raise ValueError(f"theta must be positive, got {theta}")
    angle_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    exponent = torch.arange(0, head_dim, 2, dtype=angle_dtype, device=positions.device)
    inv_freq = 1.0 / (theta ** (exponent / head_dim))
    freqs = positions.to(angle_dtype)[:, :, None] * inv_freq[None, None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Quarter-turn on each (i, i + d/2) pair, identical to the Hugging Face helper."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` ``[batch, heads, tokens, d]`` by ``cos``/``sin`` ``[batch, tokens, d]``."""
    _assert_broadcastable(x, cos)
    return x * cos.unsqueeze(1) + rotate_half(x) * sin.unsqueeze(1)


def remove_rotary(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Undo ``apply_rotary``; exact in exact arithmetic because the rotation is orthogonal."""
    _assert_broadcastable(x, cos)
    return x * cos.unsqueeze(1) - rotate_half(x) * sin.unsqueeze(1)


def _assert_broadcastable(x: torch.Tensor, cos: torch.Tensor) -> None:
    if x.ndim != 4 or cos.ndim != 3:
        raise ValueError(
            "expected x [batch, heads, tokens, d] and cos [batch, tokens, d], got "
            f"{tuple(x.shape)} and {tuple(cos.shape)}"
        )
    if (
        x.shape[0] != cos.shape[0]
        or x.shape[2] != cos.shape[1]
        or x.shape[3] != cos.shape[2]
    ):
        raise ValueError(
            f"x {tuple(x.shape)} and cos {tuple(cos.shape)} disagree on batch, tokens or d"
        )
