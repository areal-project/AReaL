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
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    q_len: int,
    k_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create a 2D block-diagonal causal attention mask for SDPA.

    For packed sequences, creates a mask where:
    - Within each sequence: standard causal masking (lower triangular)
    - Across sequences: no attention allowed

    Args:
        cu_seqlens_q: Query cumulative sequence lengths, shape [num_seqs + 1].
        cu_seqlens_k: Key cumulative sequence lengths, shape [num_seqs + 1].
            For self-attention, ``cu_seqlens_q == cu_seqlens_k``.
            For KV-cache attention, K can be longer than Q for each sequence.
        q_len: Total number of query tokens.
        k_len: Total number of key/value tokens.
        device: Target device for the mask tensor.
        dtype: Target dtype (float mask with 0.0 and -inf).

    Returns:
        Attention mask of shape [q_len, k_len].
        0.0 = attend, -inf = mask out.

    Examples:
        1) Standard packed self-attention (q_len == k_len)::

            cu_q = cu_k = [0, 3]
            # sequence length = 3
            # allowed key positions per query row:
            # q0 -> [0]
            # q1 -> [0, 1]
            # q2 -> [0, 1, 2]

        2) Right-aligned KV-cache attention (q_len < k_len)::

            cu_q = [0, 4]
            cu_k = [0, 6]
            # q tokens correspond to the last 4 positions in key timeline.
            # Local alignment offset = k_seq_len - q_seq_len = 2
            # allowed key positions per query row:
            # q0 -> [0, 1, 2]
            # q1 -> [0, 1, 2, 3]
            # q2 -> [0, 1, 2, 3, 4]
            # q3 -> [0, 1, 2, 3, 4, 5]
    """
    if cu_seqlens_q.numel() != cu_seqlens_k.numel():
        raise ValueError(
            "cu_seqlens_q and cu_seqlens_k must have same number of sequences, "
            f"got {cu_seqlens_q.numel()} vs {cu_seqlens_k.numel()}."
        )

    q_positions = torch.arange(q_len, device=device)
    k_positions = torch.arange(k_len, device=device)
    cu_q = cu_seqlens_q.to(device)
    cu_k = cu_seqlens_k.to(device)
    q_seq_ids = torch.searchsorted(cu_q, q_positions, side="right") - 1
    k_seq_ids = torch.searchsorted(cu_k, k_positions, side="right") - 1

    # Query/key must belong to the same packed sequence.
    same_seq = q_seq_ids.unsqueeze(1) == k_seq_ids.unsqueeze(0)

    # Sequence-local token indices.
    q_local = q_positions - cu_q[q_seq_ids]
    k_local = k_positions - cu_k[k_seq_ids]

    # Right-align Q inside K for KV-cache style attention:
    # q_abs = (k_seq_len - q_seq_len) + q_local.
    q_seq_lens = cu_q[q_seq_ids + 1] - cu_q[q_seq_ids]
    k_seq_lens = cu_k[q_seq_ids + 1] - cu_k[q_seq_ids]
    right_offset = (k_seq_lens - q_seq_lens).clamp(min=0)

    # Causal condition: key local index <= aligned query absolute index.
    causal = k_local.unsqueeze(0) <= (q_local + right_offset).unsqueeze(1)

    mask = torch.full((q_len, k_len), float("-inf"), device=device, dtype=dtype)
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
        cu_seqlens_k: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute attention with block-diagonal causal mask.

        Args:
            q: Query tensor, shape [batch, heads, seq_len, head_dim]
            k: Key tensor, shape [batch, heads, seq_len, head_dim]
            v: Value tensor, shape [batch, heads, seq_len, head_dim]
            scale: Optional scale factor for attention scores.
            cu_seqlens: Query cumulative sequence lengths, shape [num_seqs + 1].
            max_seqlen: Maximum sequence length (unused, for API compatibility).
            tree_attn_meta: Unused. Accepted for interface compatibility with
                TreeAttentionWrapper.
            cu_seqlens_k: Optional key cumulative sequence lengths. If not set,
                defaults to ``cu_seqlens``.

        Returns:
            Attention output, shape [batch, heads, seq_len, head_dim]
        """
        q_len = q.shape[2]
        k_len = k.shape[2]
        if cu_seqlens_k is None:
            cu_seqlens_k = cu_seqlens
        # TODO: Mask should be precomputed and passed in, not computed here.
        attn_mask = create_block_causal_mask_2d(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens_k,
            q_len=q_len,
            k_len=k_len,
            device=q.device,
            dtype=q.dtype,
        )

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
