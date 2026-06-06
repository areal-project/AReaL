# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Partitioned Rotary Positional Embedding with Periodic Support.

This implements a hybrid positional encoding scheme where:
- Top-left 0.8 * 0.8 of the matrix uses standard RoPE encoding with local position (S = I % P)
- Bottom-right 0.2 * 0.2 uses custom sinusoidal encoding: R_ij = sin(2πijB/n) / sqrt(n)
  where B = I // P is the block index
- Other regions are zero
- P is the period parameter
"""

import math

import torch

from .base import RotaryEmbedding


class PartitionedRotaryEmbedding(RotaryEmbedding):
    """Partitioned RoPE with hybrid encoding in different matrix regions and periodic support.
    
    Args:
        period: Period P for partitioning positions into blocks.
                Position I is split into block index B = I // P and local position S = I % P.
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
        period: int = 512,
    ) -> None:
        # Calculate partition dimensions
        self.rope_dim = int(rotary_dim * 0.8)  # 80% uses standard RoPE
        self.custom_dim_start = int(rotary_dim * 0.8)
        self.custom_dim = rotary_dim - self.rope_dim  # 20% uses custom encoding
        self.period = period
        
        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )

    def _compute_inv_freq(self, base: float) -> torch.Tensor:
        """Compute inverse frequency for the RoPE portion (first 80%)."""
        # Only compute for the RoPE dimensions (0.8 of total)
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, self.rope_dim, 2, dtype=torch.float) / self.rope_dim
            )
        )
        return inv_freq

    def _compute_custom_encoding(self) -> torch.Tensor:
        """Compute custom sinusoidal encoding for bottom-right region with block indices.
        
        R_ij = sin(2π * i * j * B / n) / sqrt(n)
        where n = custom_dim (0.2 * rotary_dim) and B is the block index
        
        For each position I:
        - B = I // period (block index)
        - The encoding uses B in the formula
        """
        n = self.custom_dim
        if n == 0:
            return torch.zeros(
                self.max_position_embeddings, 0, dtype=torch.float
            )
        
        # Create position indices (i) for positions 0 to max_position_embeddings - 1
        positions = torch.arange(self.max_position_embeddings, dtype=torch.float)
        
        # Compute block indices: B = I // period
        block_indices = (positions / self.period).floor()  # Shape: [max_pos]
        
        # Create dimension indices (j) from 0 to n - 1
        j = torch.arange(n, dtype=torch.float).unsqueeze(0)  # Shape: [1, n]
        i = torch.arange(n, dtype=torch.float).unsqueeze(1)  # Shape: [n, 1]
        
        # Expand block_indices for broadcasting: [max_pos, 1, 1]
        B = block_indices.unsqueeze(1).unsqueeze(2)
        
        # Compute R_ij = sin(2π * i * j * B / n) / sqrt(n)
        # i: [n, 1], j: [1, n], B: [max_pos, 1, 1]
        # Result: [max_pos, n, n] -> we need [max_pos, n] so we extract diagonal or compress
        # Actually, we need to apply this as a transformation, let's store per-position encoding
        
        # For each position, compute the encoding matrix element-wise
        # We'll store a linearized version: for each position, store n values
        # These represent the encoding for that position
        encoding_list = []
        for pos_idx in range(self.max_position_embeddings):
            B_val = block_indices[pos_idx].item()
            # For this position, compute encoding for each dimension j
            j_vals = torch.arange(n, dtype=torch.float)
            # Using i=j for diagonal encoding elements
            encoding_pos = torch.sin(2 * math.pi * j_vals * j_vals * B_val / n) / math.sqrt(n)
            encoding_list.append(encoding_pos)
        
        encoding = torch.stack(encoding_list, dim=0)  # [max_pos, n]
        
        return encoding

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute combined cos/sin cache with partitioned periodic encoding.
        
        For positions I:
        - Local position S = I % period (used for first 80% RoPE encoding)
        - Block index B = I // period (used for last 20% custom encoding)
        """
        # For the RoPE portion (80%), we use local position S = I % period
        # Compute frequencies for local positions within a period
        inv_freq = self._compute_inv_freq(self.base)
        
        # Generate positions 0 to max_position_embeddings - 1
        all_positions = torch.arange(self.max_position_embeddings, dtype=torch.float)
        
        # Compute local positions S = I % period
        local_positions = all_positions % self.period
        
        # Compute RoPE encoding using local positions
        freqs = torch.einsum("i,j -> ij", local_positions, inv_freq)
        cos_rope = freqs.cos()
        sin_rope = freqs.sin()
        
        # Standard RoPE cache (concatenated cos and sin)
        rope_cache = torch.cat((cos_rope, sin_rope), dim=-1)
        
        # Custom encoding for last 20% (uses block index B)
        custom_encoding = self._compute_custom_encoding()
        
        # For the custom part, we need to create cos/sin-like representation
        # We'll use the encoding directly as "cos" and zeros as "sin"
        # This allows us to reuse the standard rotation application logic
        custom_cos = custom_encoding
        custom_sin = torch.zeros_like(custom_encoding)
        
        # Combine RoPE cache and custom encoding cache
        # Format: [rope_cos, custom_cos, rope_sin, custom_sin]
        combined_cache = torch.cat(
            (rope_cache, custom_cos, custom_sin), dim=-1
        )
        
        return combined_cache

    def get_cos_sin(self, seqlen: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract cos and sin from the combined cache."""
        cos_sin = self.cos_sin_cache[:seqlen]
        
        # Split the cache back into components
        # rope_dim gives us the size after doubling (cos+sin for rope)
        rope_cache_size = self.rope_dim  # This is inv_freq size, doubled in cache
        custom_size = self.custom_dim
        
        # Extract components
        rope_cos_sin = cos_sin[:, :rope_cache_size * 2]
        custom_cos = cos_sin[:, rope_cache_size * 2:rope_cache_size * 2 + custom_size]
        custom_sin = cos_sin[:, rope_cache_size * 2 + custom_size:]
        
        # Standard RoPE cos/sin
        rope_cos, rope_sin = rope_cos_sin.chunk(2, dim=-1)
        
        # Combine: [rope_cos, custom_cos] and [rope_sin, custom_sin]
        cos = torch.cat((rope_cos, custom_cos), dim=-1)
        sin = torch.cat((rope_sin, custom_sin), dim=-1)
        
        return cos, sin

    def extra_repr(self) -> str:
        s = super().extra_repr()
        s += f", rope_dim={self.rope_dim}, custom_dim={self.custom_dim}"
        s += f", period={self.period}"
        s += " (partitioned: 80% RoPE(local) + 20% custom(block))"
        return s
