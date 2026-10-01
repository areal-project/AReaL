# SPDX-License-Identifier: Apache-2.0

"""Tree attention module for Archon engine.

This module provides a TreeAttentionWrapper that has the same interface as
VarlenAttentionWrapper but uses tree attention (flex attention or Triton kernel)
instead of standard varlen attention.

The wrapper is designed to be a drop-in replacement for VarlenAttentionWrapper
when tree training is enabled. Set attn_type="tree" in model_args to use it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from areal.models.tree_attn.constants import USE_TRITON_TREE_ATTN
from areal.models.tree_attn.tree import (
    build_block_mask_from_trie,
    build_triton_attn_data_from_trie,
)
from areal.models.tree_attn.triton_kernel import TRITON_AVAILABLE, TreeAttentionData
from areal.utils import logging

if TYPE_CHECKING:
    from areal.models.tree_attn.tree import TrieNode

logger = logging.getLogger("TreeAttentionWrapper")

__all__ = [
    "TreeAttentionMeta",
    "TreeAttentionWrapper",
]


@dataclass
class TreeAttentionMeta:
    """Tree attention metadata for one microbatch.

    Exactly one of block_mask or triton_data must be set,
    depending on the backend (flex attention vs Triton kernel).
    """

    block_mask: BlockMask | None = None
    triton_data: TreeAttentionData | None = None

    def __post_init__(self):
        if (self.block_mask is None) == (self.triton_data is None):
            raise ValueError("Exactly one of block_mask or triton_data must be set.")

    @classmethod
    def from_trie(
        cls, trie_node: TrieNode, padded_size: int, device: torch.device
    ) -> TreeAttentionMeta:
        """Build tree attention metadata from a trie node.

        Automatically selects the Triton kernel or flex attention backend.
        """
        if USE_TRITON_TREE_ATTN and TRITON_AVAILABLE:
            return cls(
                triton_data=build_triton_attn_data_from_trie(trie_node, padded_size)
            )
        return cls(
            block_mask=build_block_mask_from_trie(trie_node, padded_size, device)
        )


class TreeAttentionWrapper(nn.Module):
    """Attention wrapper for tree training in Archon Engine.

    This wrapper has the same interface as VarlenAttentionWrapper but uses
    tree attention (flex attention with BlockMask or Triton tree attention)
    instead of standard flash attention.

    The wrapper expects a ``tree_attn_meta`` (:class:`TreeAttentionMeta`) to be
    passed as a keyword argument during forward.

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
        """Compute tree attention.

        Args:
            q: Query tensor, shape [batch, heads, seq_len, head_dim]
            k: Key tensor, shape [batch, heads, seq_len, head_dim]
            v: Value tensor, shape [batch, heads, seq_len, head_dim]
            scale: Optional scale factor for attention scores.
            cu_seqlens: Cumulative sequence lengths (unused for tree attention,
                kept for API compatibility with VarlenAttentionWrapper).
            max_seqlen: Maximum sequence length (unused for tree attention,
                kept for API compatibility with VarlenAttentionWrapper).
            tree_attn_meta: Tree attention metadata containing either a
                BlockMask (flex attention) or TreeAttentionData (Triton).
            cu_seqlens_k: Unused. Accepted for interface compatibility with
                VarlenAttentionWrapper.

        Returns:
            Attention output, shape [batch, heads, seq_len, head_dim]
        """
        # Input: [batch, heads, seq_len, head_dim]
        batch, n_heads, seq_len, head_dim = q.shape

        # For packed sequences, batch should be 1
        assert batch == 1, (
            f"TreeAttentionWrapper expects batch=1 for packed sequences, "
            f"got batch={batch}"
        )

        assert tree_attn_meta is not None, (
            "TreeAttentionWrapper requires tree_attn_meta. "
            "For standard attention, use VarlenAttentionWrapper instead."
        )

        if tree_attn_meta.triton_data is not None:
            from areal.models.tree_attn.triton_kernel import tree_attention

            triton_data = tree_attn_meta.triton_data
            return tree_attention(
                q,
                k,
                v,
                triton_data.packed_mask,
                triton_data.kv_indices,
                triton_data.kv_offsets,
                triton_data.q_indices,
                triton_data.q_offsets,
                sm_scale=scale,
            )

        assert tree_attn_meta.block_mask is not None
        enable_gqa = q.shape[1] != k.shape[1]
        return flex_attention(
            q,
            k,
            v,
            block_mask=tree_attn_meta.block_mask,
            scale=scale,
            enable_gqa=enable_gqa,
        )
