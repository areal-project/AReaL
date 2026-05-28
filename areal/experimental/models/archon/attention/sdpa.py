# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

if TYPE_CHECKING:
    from areal.models.tree_attn.module_archon import TreeAttentionMeta

__all__ = ["SDPAWrapper", "create_block_causal_mask_2d"]


def create_block_causal_mask_2d(
    cu_seqlens: torch.Tensor,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create a 2D block-diagonal causal attention mask for SDPA.

    For packed sequences, creates a mask where:
    - Within each sequence: standard causal masking (lower triangular)
    - Across sequences: no attention allowed

    Args:
        cu_seqlens: Cumulative sequence lengths, shape [num_seqs + 1].
            For example, [0, 3, 5, 7] means 3 sequences with lengths 3, 2, 2.
        seq_len: Total sequence length (should equal cu_seqlens[-1]).
        device: Target device for the mask tensor.
        dtype: Target dtype (float mask with 0.0 and -inf).

    Returns:
        Attention mask of shape [seq_len, seq_len].
        0.0 = attend, -inf = mask out.

    Example for cu_seqlens=[0, 3, 5, 7]::

        [  0, -inf, -inf, | -inf, -inf, | -inf, -inf]
        [  0,   0 , -inf, | -inf, -inf, | -inf, -inf]
        [  0,   0 ,   0 , | -inf, -inf, | -inf, -inf]
        [-inf, -inf, -inf, |   0, -inf, | -inf, -inf]
        [-inf, -inf, -inf, |   0,   0 , | -inf, -inf]
        [-inf, -inf, -inf, | -inf, -inf, |   0, -inf]
        [-inf, -inf, -inf, | -inf, -inf, |   0,   0 ]
    """
    positions = torch.arange(seq_len, device=device)
    seq_ids = torch.searchsorted(cu_seqlens, positions, side="right") - 1

    # same_seq: query and key must be in the same sequence
    # causal: key position <= query position
    same_seq = seq_ids.unsqueeze(1) == seq_ids.unsqueeze(0)
    causal = positions.unsqueeze(0) <= positions.unsqueeze(1)

    mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
    mask = mask.masked_fill(same_seq & causal, 0.0)
    return mask


class SDPAWrapper(nn.Module):
    """SDPA wrapper for packed sequences with block-diagonal causal mask.

    This wrapper supports packed sequences by creating a block-diagonal
    attention mask that combines causal masking with document boundary
    masking.
    """

    # Backend priority order
    DEFAULT_BACKENDS = [
        SDPBackend.CUDNN_ATTENTION,
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.MATH,
    ]

    def __init__(self, sdpa_backends: list[SDPBackend] | None = None):
        super().__init__()
        self.sdpa_backends = (
            sdpa_backends if sdpa_backends is not None else self.DEFAULT_BACKENDS
        )

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
        """Compute attention with block-diagonal causal mask.

        Args:
            q: Query tensor, shape [batch, heads, seq_len, head_dim]
            k: Key tensor, shape [batch, heads, seq_len, head_dim]
            v: Value tensor, shape [batch, heads, seq_len, head_dim]
            scale: Optional scale factor for attention scores.
            cu_seqlens: Cumulative sequence lengths, shape [num_seqs + 1].
            max_seqlen: Maximum sequence length (unused, for API compatibility).
            tree_attn_meta: Unused. Accepted for interface compatibility with
                TreeAttentionWrapper.

        Returns:
            Attention output, shape [batch, heads, seq_len, head_dim]
        """
        seq_len = q.shape[2]
        # TODO: Mask should be precomputed and passed in, not computed here.
        attn_mask = create_block_causal_mask_2d(cu_seqlens, seq_len, q.device, q.dtype)

        with sdpa_kernel(self.sdpa_backends, set_priority=True):
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                scale=scale,
                is_causal=False,
                enable_gqa=q.size(1) != k.size(1),
            )
