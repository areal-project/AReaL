# SPDX-License-Identifier: Apache-2.0

"""Lightning Attention module for megatron-core, using fla (flash-linear-attention) Triton kernels.

BailingMoeV2_5 uses a heterogeneous architecture where most layers use Lightning Attention
(linear attention with learned decay) and every `layer_group_size`-th layer uses standard MLA.

This module implements Lightning Attention as a megatron-core compatible module that can be
used in TransformerLayer specs alongside MLASelfAttention.

Reference:
    - fla library: https://github.com/sustcsonglin/flash-linear-attention
    - API: fla.ops.simple_gla.chunk_simple_gla(q, k, v, g_gamma=..., scale=...)
    - Input shapes: [B, T, H, K] (batch, seq_len, num_heads, head_dim)

Key differences from MLA layers:
    - No GQA: all heads have independent Q, K, V
    - attn_head_dim may differ from MLA's qk_nope_head_dim/v_head_dim
    - Has gate projection (g_proj) + gate norm (g_norm) for output gating
    - partial_rotary_factor applies to attn_head_dim (not qk_rope_head_dim)

TP support:
    Under tensor parallelism, attention heads are split across TP ranks.
    The g_gamma decay is computed using global head count and global head indices
    to ensure correct per-head decay regardless of TP degree.
"""

import copy
import math
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
from megatron.core import parallel_state as mpu
from megatron.core.models.common.embeddings.rotary_pos_embedding import (
    apply_rotary_pos_emb,
)
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from torch.distributed._functional_collectives import all_to_all_single_autograd

from areal.utils import logging

logger = logging.getLogger("LightningAttention")


def _get_tp_world_size() -> int:
    """Get tensor model parallel world size, with fallback for uninitialized mpu."""
    try:
        if mpu.model_parallel_is_initialized():
            return mpu.get_tensor_model_parallel_world_size()
    except (RuntimeError, AttributeError):
        pass
    return 1


def _get_tp_rank() -> int:
    """Get tensor model parallel rank, with fallback for uninitialized mpu."""
    try:
        if mpu.model_parallel_is_initialized():
            return mpu.get_tensor_model_parallel_rank()
    except (RuntimeError, AttributeError):
        pass
    return 0


def _get_cp_world_size() -> int:
    """Get context parallel world size, with fallback for uninitialized mpu."""
    try:
        if mpu.model_parallel_is_initialized():
            return mpu.get_context_parallel_world_size()
    except (RuntimeError, AttributeError):
        pass
    return 1


def _get_cp_rank() -> int:
    """Get context parallel rank, with fallback for uninitialized mpu."""
    try:
        if mpu.model_parallel_is_initialized():
            return mpu.get_context_parallel_rank()
    except (RuntimeError, AttributeError):
        pass
    return 0


def _get_cp_group():
    """Get context parallel process group, with fallback for uninitialized mpu."""
    try:
        if mpu.model_parallel_is_initialized():
            return mpu.get_context_parallel_group()
    except (RuntimeError, AttributeError):
        pass
    return None


def _build_alibi_slopes(n_attention_heads: int) -> torch.Tensor:
    """Build ALiBi-style geometric slopes for Lightning Attention decay.

    For power-of-2 head counts: slopes are geometric sequence starting from
    2^(-(2^-(log2(n)-3))) with the same ratio.
    For non-power-of-2: uses closest power-of-2 with interleaved extras.

    Returns:
        Tensor of shape [n_attention_heads] with float32 slopes.
    """

    def _get_slopes(n):
        def _get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return _get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                _get_slopes_power_of_2(closest_power_of_2)
                + _get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
            )

    return torch.tensor(_get_slopes(n_attention_heads), dtype=torch.float32)


# ---------------------------------------------------------------------------
# Context Parallelism (CP) communication primitives
# ---------------------------------------------------------------------------


def _all_to_all_cp2hp(input_: torch.Tensor, cp_group) -> torch.Tensor:
    """All-to-all: context-parallel to head-parallel with autograd support.

    [S/CP, B, H_local, D] -> [S, B, H_local/CP, D]
    """
    cp_size = dist.get_world_size(group=cp_group)
    seq_len, batch, num_heads, head_dim = input_.shape

    input_ = input_.view(seq_len, batch, cp_size, num_heads // cp_size, head_dim)
    input_ = input_.permute(2, 0, 1, 3, 4).contiguous()
    stacked_shape = input_.shape

    flat = input_.reshape(-1)
    received = all_to_all_single_autograd(flat, None, None, group=cp_group)
    output = received.reshape(stacked_shape)

    output = output.reshape(cp_size * seq_len, batch, num_heads // cp_size, head_dim)
    return output


def _all_to_all_hp2cp(input_: torch.Tensor, cp_group) -> torch.Tensor:
    """All-to-all: head-parallel to context-parallel with autograd support.

    [S, B, H_local/CP, D] -> [S/CP, B, H_local, D]
    """
    cp_size = dist.get_world_size(group=cp_group)
    full_seq, batch, heads_per_cp, head_dim = input_.shape
    seq_per_cp = full_seq // cp_size

    input_ = input_.view(
        cp_size, seq_per_cp, batch, heads_per_cp, head_dim
    ).contiguous()
    stacked_shape = input_.shape

    flat = input_.reshape(-1)
    received = all_to_all_single_autograd(flat, None, None, group=cp_group)
    output = received.reshape(stacked_shape)

    output = output.permute(1, 2, 0, 3, 4).contiguous()
    output = output.reshape(seq_per_cp, batch, cp_size * heads_per_cp, head_dim)
    return output


# ---------------------------------------------------------------------------
# Zigzag load-balancing undo / redo for CP
# ---------------------------------------------------------------------------


def _build_zigzag_undo_indices(
    total_len: int,
    cp_size: int,
    cu_seqlens: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor:
    """Build index tensor to undo zigzag reordering after all-to-all cp2hp.

    After cp2hp, the full sequence has tokens in zigzag-interleaved order
    (rank 0's front+back chunks, then rank 1's, ...).  This builds an index
    tensor so that ``sequential = zigzag[indices]`` restores sequential order.

    Supports both packed sequences (per-sequence zigzag via cu_seqlens) and
    fixed-length BSHD format (cu_seqlens=None -> single global sequence).
    """
    indices = torch.empty(total_len, dtype=torch.long, device=device)
    t_per_cp = total_len // cp_size

    if cu_seqlens is None:
        seq_bounds = [(0, total_len)]
    else:
        seq_bounds = [
            (cu_seqlens[i].item(), cu_seqlens[i + 1].item())
            for i in range(cu_seqlens.shape[0] - 1)
        ]

    for cu_start, cu_end in seq_bounds:
        seq_len = cu_end - cu_start
        chunk = seq_len // (2 * cp_size)
        cu_s = cu_start // cp_size

        for j in range(cp_size):
            block_start = j * t_per_cp + cu_s
            base = torch.arange(chunk, device=device)

            dst_front = cu_start + j * chunk
            indices[dst_front : dst_front + chunk] = block_start + base

            dst_back = cu_start + seq_len - (j + 1) * chunk
            indices[dst_back : dst_back + chunk] = block_start + chunk + base

    return indices


def _build_zigzag_redo_indices(undo_indices: torch.Tensor) -> torch.Tensor:
    """Build inverse permutation of undo indices (sequential -> zigzag)."""
    redo = torch.empty_like(undo_indices)
    redo[undo_indices] = torch.arange(len(undo_indices), device=undo_indices.device)
    return redo


@dataclass
class LightningAttentionSubmodules:
    """Submodule specs for Lightning Self-Attention layer."""

    linear_qkv: ModuleSpec | type = None
    linear_gate: ModuleSpec | type = None
    linear_proj: ModuleSpec | type = None


class LightningCoreAttention(MegatronModule):
    """Core Lightning Attention computation using fla's chunk_simple_gla kernel.

    Handles tensor layout conversion between megatron-core format and fla format:
    - megatron-core: [S, B, num_heads_local, head_dim]
    - fla kernel: [B, T, num_heads_local, head_dim]

    The g_gamma (per-head log decay) is pre-computed using ALiBi-style geometric slopes
    scaled by layer position, matching the Megatron-LM reference implementation.

    Formula: g_gamma = -alibi_slopes(H_global) * (1 - layer_idx/(num_layers-1) + 1e-5)
    where layer_idx is 0-indexed. This matches SGLang (theta) and HybridEngine.
    Then TP-sliced: g_gamma_local = g_gamma[tp_rank*H_local : (tp_rank+1)*H_local]
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_head_dim: int,
    ):
        super().__init__(config=config)
        # megatron layer_number is 1-indexed; convert to 0-indexed
        self.layer_idx = layer_number - 1
        self.num_layers = config.num_layers
        self.attn_head_dim = attn_head_dim
        self.scale = 1.0 / math.sqrt(attn_head_dim)

        # TP-aware head count
        num_heads_global = config.num_attention_heads
        tp_size = _get_tp_world_size()
        tp_rank = _get_tp_rank()
        self.num_heads_local = num_heads_global // tp_size

        # Pre-compute g_gamma using ALiBi geometric slopes.
        # Formula: 1 - layer_idx/(num_layers-1) + 1e-5  (layer_idx 0-indexed)
        # Matches SGLang (theta/SGLang/hybrid_linear_attn_backend.py:658) and HybridEngine.
        # Note: HF modeling_bailing_moe_v2_5.py uses (layer_idx-1)/(N-1) which gives
        # slightly different values (layer 0 scale 1.0526 vs 1.0000 here).
        alibi_slopes = _build_alibi_slopes(num_heads_global)
        layer_scale = 1.0 - self.layer_idx / max(self.num_layers - 1, 1) + 1e-5
        g_gamma_global = -alibi_slopes * layer_scale
        # TP-slice to this rank's local heads
        head_offset = tp_rank * self.num_heads_local
        g_gamma = g_gamma_global[
            head_offset : head_offset + self.num_heads_local
        ].contiguous()
        self.register_buffer("g_gamma", g_gamma, persistent=False)

    def forward(self, query, key, value, cu_seqlens=None, cp_rank=None):
        """Forward pass for Lightning Attention core computation.

        Args:
            query: [S, B, num_heads_local, head_dim]
            key: [S, B, num_heads_local, head_dim]
            value: [S, B, num_heads_local, head_dim]
            cu_seqlens: Cumulative sequence lengths for packed sequences, or None.
            cp_rank: CP rank for g_gamma slicing when CP > 1, or None.

        Returns:
            output: [S, B, num_heads_local, head_dim]
        """
        try:
            from fla.ops.simple_gla import chunk_simple_gla
        except ImportError:
            raise ImportError(
                "flash-linear-attention (fla) is required for Lightning Attention. "
                "Install with: pip install flash-linear-attention>=0.3.0"
            )

        # Convert from megatron layout [S, B, H, D] to fla layout [B, T, H, D]
        q = query.permute(1, 0, 2, 3).contiguous()  # [B, T, H, D]
        k = key.permute(1, 0, 2, 3).contiguous()  # [B, T, H, D]
        v = value.permute(1, 0, 2, 3).contiguous()  # [B, T, H, D]

        # CP-slice g_gamma when running with context parallelism
        g_gamma = self.g_gamma
        if cp_rank is not None:
            cp_size = _get_cp_world_size()
            heads_per_cp = self.num_heads_local // cp_size
            g_gamma = g_gamma[cp_rank * heads_per_cp : (cp_rank + 1) * heads_per_cp]

        # Call fla kernel with pre-computed ALiBi-based g_gamma
        # g_gamma (per-head data-independent decay, shape [H]) is mathematically equivalent
        # to g (per-token, shape [B,T,H]) when the decay is constant across time.
        # g_gamma is more efficient (no extra memory, no cumsum kernel).
        # Reference: Megatron-LM attention.py:2188
        output, _ = chunk_simple_gla(
            q=q,
            k=k,
            v=v,
            g_gamma=g_gamma,
            scale=self.scale,
            cu_seqlens=cu_seqlens,
        )
        # output shape: [B, T, H, D]

        # Convert back to megatron layout [S, B, H, D]
        output = output.permute(1, 0, 2, 3).contiguous()

        return output


class GroupRMSNorm(nn.Module):
    """Group RMSNorm applied per group of heads.

    Used for gate normalization in Lightning Attention.
    Applies RMSNorm independently to each group of heads.

    Under TP, num_heads should be the LOCAL (per-partition) head count.
    """

    def __init__(
        self, num_heads: int, head_dim: int, num_groups: int, eps: float = 1e-6
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_groups = num_groups
        self.group_size = num_heads // num_groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_heads * head_dim))

    def forward(self, x):
        """Apply group RMSNorm.

        Args:
            x: [..., num_heads, head_dim]

        Returns:
            Normalized tensor with same shape
        """
        original_shape = x.shape
        # Reshape to [..., num_groups, group_size, head_dim]
        x = x.view(
            *original_shape[:-2], self.num_groups, self.group_size, self.head_dim
        )
        # Compute RMS per group
        rms = x.float().pow(2).mean(dim=(-2, -1), keepdim=True).add(self.eps).rsqrt()
        x = (x.float() * rms).to(x.dtype)
        # Reshape back and apply weight
        x = x.view(*original_shape)
        weight = self.weight.view(self.num_heads, self.head_dim)
        return x * weight


class LightningSelfAttention(MegatronModule):
    """Lightning Self-Attention layer compatible with megatron-core TransformerLayer.

    Architecture:
    - Fused QKV projection (query_key_value) in megatron interleaved format
    - Q/K RMSNorm
    - RoPE (applied to first rotary_dim dimensions)
    - Lightning Attention kernel (via fla chunk_simple_gla)
    - Gate: sigmoid(g_norm(g_proj(hidden_states))) * attention_output
    - Output projection

    Weight mapping (mcore -> HF):
    - linear_qkv.weight -> attention.query_key_value.weight (interleaved [H,3,D] format)
    - linear_gate.weight -> attention.g_proj.weight
    - gate_norm.weight -> attention.g_norm.weight
    - linear_proj.weight -> attention.dense.weight
    - q_layernorm.weight -> attention.query_layernorm.weight
    - k_layernorm.weight -> attention.key_layernorm.weight

    TP support:
    - linear_qkv/linear_gate: ColumnParallelLinear (split output by TP)
    - linear_proj: RowParallelLinear (split input by TP)
    - gate_norm: weight split by TP (num_heads_local * head_dim)
    - q_layernorm/k_layernorm: per-head norm, no TP split needed
    - core_attention: g_gamma computed with global H and global head indices

    This module is instantiated via ModuleSpec with params:
    - attn_head_dim: int (e.g., 256)
    - partial_rotary_factor: float (e.g., 0.5)
    - linear_attn_norm_group_size: int (e.g., 8)
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: LightningAttentionSubmodules,
        layer_number: int,
        attn_mask_type=None,
        attn_head_dim: int = 256,
        partial_rotary_factor: float = 0.5,
        linear_attn_norm_group_size: int = 1,
        **kwargs,
    ):
        super().__init__(config=config)
        self.config = config
        # HACK: Lightning Attention needs multi_latent_attention=False for correct RoPE.
        # When True, apply_rotary_pos_emb deinterleaves dims (x[...,0::2], x[...,1::2])
        # before rotation, which is wrong for Lightning Attention's standard layout.
        # Reference: Megatron-LM attention.py:1706
        self.rope_config = copy.copy(config)
        self.rope_config.multi_latent_attention = False
        # Lightning uses its own RotaryEmbedding with SBHD format (no cu_seqlens),
        # so the unfused path is sufficient and avoids TE fused RoPE dependencies.
        self.rope_config.apply_rope_fusion = False
        self.layer_number = layer_number
        self.num_attention_heads = config.num_attention_heads
        self.attn_head_dim = attn_head_dim
        self.hidden_size = config.hidden_size
        self.partial_rotary_factor = partial_rotary_factor
        self.rotary_dim = int(attn_head_dim * partial_rotary_factor)

        # TP-aware local head count for forward reshapes
        tp_size = _get_tp_world_size()
        self.num_heads_per_partition = self.num_attention_heads // tp_size

        # Megatron interleaved QKV: for each head [q_i, k_i, v_i] (no GQA)
        # Total output: num_heads * 3 * head_dim (global; ColumnParallelLinear splits internally)
        self.qkv_size = self.num_attention_heads * attn_head_dim * 3

        # Fused QKV projection
        self.linear_qkv = build_module(
            submodules.linear_qkv,
            self.hidden_size,
            self.qkv_size,
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=getattr(config, "add_bias_linear", False),
            skip_bias_add=False,
            is_expert=False,
        )

        # Gate projection: hidden_size -> num_heads * head_dim (global; split internally)
        gate_size = self.num_attention_heads * attn_head_dim
        self.linear_gate = build_module(
            submodules.linear_gate,
            self.hidden_size,
            gate_size,
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=getattr(config, "add_bias_linear", False),
            skip_bias_add=False,
            is_expert=False,
        )

        # Core attention (TP-aware g_gamma computation)
        self.core_attention = LightningCoreAttention(
            config=config,
            layer_number=layer_number,
            attn_head_dim=attn_head_dim,
        )

        # Output projection (global input size; RowParallelLinear splits internally)
        self.linear_proj = build_module(
            submodules.linear_proj,
            self.num_attention_heads * attn_head_dim,
            self.hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=getattr(config, "add_bias_linear", False),
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
        )

        # Q/K RMSNorm (per-head on head_dim, no TP split needed)
        self.q_layernorm = nn.RMSNorm(
            attn_head_dim,
            eps=config.layernorm_epsilon,
        )
        self.k_layernorm = nn.RMSNorm(
            attn_head_dim,
            eps=config.layernorm_epsilon,
        )

        # Gate norm (GroupRMSNorm with LOCAL head count for TP)
        # HF's group_norm_size = number of groups (NOT heads per group).
        # At TP>1, local_num_groups = total_num_groups / tp_size to maintain the same
        # number of elements per group (num_heads/num_groups * head_dim) as the full model.
        # E.g., group_norm_size=4, H=32, D=128: HF has 4 groups of 1024 elements.
        # At TP=2: local_num_groups=2, local groups have 8*128=1024 elements each. Exact match.
        tp_size = _get_tp_world_size()
        num_groups = max(linear_attn_norm_group_size // tp_size, 1)
        self.gate_norm = GroupRMSNorm(
            num_heads=self.num_heads_per_partition,
            head_dim=attn_head_dim,
            num_groups=num_groups,
            eps=config.layernorm_epsilon,
        )
        # Mark gate_norm weight as TP-sharded for correct checkpoint save/load
        self.gate_norm.weight.tensor_model_parallel = True
        self.gate_norm.weight.partition_dim = 0

        # Lightning-specific rotary embedding
        # MLA layers use qk_pos_emb_head_dim for RoPE dim, but Lightning layers
        # use attn_head_dim * partial_rotary_factor. We create our own RotaryEmbedding.
        try:
            from megatron.core.models.common.embeddings.rotary_pos_embedding import (
                RotaryEmbedding,
            )

            self.lightning_rotary_emb = RotaryEmbedding(
                kv_channels=self.rotary_dim,
                rotary_percent=1.0,
                rotary_base=getattr(config, "rotary_base", 600000.0),
            )
        except (ImportError, TypeError, RuntimeError):
            logger.warning(
                "Could not create Lightning RotaryEmbedding, will use passed rotary_pos_emb"
            )
            self.lightning_rotary_emb = None

    def _apply_rope(self, query, key, seq_len, rotary_pos_emb=None):
        """Apply RoPE to query and key tensors."""
        if self.lightning_rotary_emb is not None:
            # packed_seq=True prevents RotaryEmbedding from auto-slicing by CP rank.
            # We handle CP ourselves (all-to-all + undo zigzag) so positions are already
            # sequential [0, seq_len) and should not be further sliced.
            lightning_rotary = self.lightning_rotary_emb(seq_len, packed_seq=True)
            if not isinstance(lightning_rotary, tuple):
                lightning_rotary = (lightning_rotary,) * 2
            query = apply_rotary_pos_emb(query, lightning_rotary[0], self.rope_config)
            key = apply_rotary_pos_emb(key, lightning_rotary[1], self.rope_config)
        elif rotary_pos_emb is not None:
            if not isinstance(rotary_pos_emb, tuple):
                rotary_pos_emb = (rotary_pos_emb,) * 2
            query = apply_rotary_pos_emb(query, rotary_pos_emb[0], self.rope_config)
            key = apply_rotary_pos_emb(key, rotary_pos_emb[1], self.rope_config)
        return query, key

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        key_value_states=None,
        inference_params=None,
        rotary_pos_emb=None,
        packed_seq_params=None,
        **kwargs,
    ):
        """Forward pass for Lightning Self-Attention.

        Args:
            hidden_states: [S, B, H] input tensor (post-layernorm from TransformerLayer)
            rotary_pos_emb: RoPE embeddings (may be MLA-specific dim, we use our own)
            packed_seq_params: Packed sequence parameters with cu_seqlens.

        Returns:
            output: [S, B, H] attention output
            bias: output bias (None for this implementation)
        """
        # NOTE on Sequence Parallelism (SP):
        # When TP>1, SP is auto-enabled. hidden_states is [S/TP, B, H].
        # TEColumnParallelLinear internally all-gathers the sequence dim.
        # With CP, the "full" length from SP's perspective is S/CP.
        batch_size = hidden_states.shape[1]

        # Extract cu_seqlens for packed sequence support
        cu_seqlens = None
        if packed_seq_params is not None:
            cu_seqlens = getattr(packed_seq_params, "cu_seqlens_q", None)

        # Fused QKV projection
        qkv, _ = self.linear_qkv(hidden_states)
        seq_len = qkv.shape[0]  # S/CP (or S when CP=1)

        # Split into Q, K, V from interleaved layout [H_local, 3, D]
        qkv = qkv.view(
            seq_len, batch_size, self.num_heads_per_partition, 3, self.attn_head_dim
        )
        query = qkv[:, :, :, 0, :].contiguous()
        key = qkv[:, :, :, 1, :].contiguous()
        value = qkv[:, :, :, 2, :].contiguous()

        # Apply Q/K LayerNorm (before CP communication)
        query = self.q_layernorm(query)
        key = self.k_layernorm(key)

        # Gate projection (before CP communication — stays in [S/CP] dimension)
        gate, _ = self.linear_gate(hidden_states)
        gate = gate.view(
            seq_len, batch_size, self.num_heads_per_partition, self.attn_head_dim
        )

        cp_size = _get_cp_world_size()

        if cp_size > 1:
            cp_group = _get_cp_group()
            cp_rank = _get_cp_rank()

            # All-to-all: [S/CP, B, H_local, D] -> [S, B, H_local/CP, D]
            query = _all_to_all_cp2hp(query, cp_group)
            key = _all_to_all_cp2hp(key, cp_group)
            value = _all_to_all_cp2hp(value, cp_group)

            full_seq_len = query.shape[0]

            # Undo zigzag: restore sequential token order for linear attention
            undo_idx = _build_zigzag_undo_indices(
                full_seq_len, cp_size, cu_seqlens, query.device
            )
            query = query[undo_idx]
            key = key[undo_idx]
            value = value[undo_idx]

            # RoPE applied AFTER undo zigzag (tokens now in correct sequential order)
            query, key = self._apply_rope(query, key, full_seq_len, rotary_pos_emb)

            # Core attention with CP-sliced g_gamma and cu_seqlens
            attn_output = self.core_attention(
                query, key, value, cu_seqlens=cu_seqlens, cp_rank=cp_rank
            )

            # Redo zigzag: restore zigzag order for all-to-all
            redo_idx = _build_zigzag_redo_indices(undo_idx)
            attn_output = attn_output[redo_idx]

            # All-to-all: [S, B, H_local/CP, D] -> [S/CP, B, H_local, D]
            attn_output = _all_to_all_hp2cp(attn_output, cp_group)
        else:
            # Non-CP path
            query, key = self._apply_rope(query, key, seq_len, rotary_pos_emb)
            attn_output = self.core_attention(query, key, value, cu_seqlens=cu_seqlens)

        # Apply gate norm + gating
        attn_output = self.gate_norm(attn_output)
        output = attn_output * gate.sigmoid()
        output = output.reshape(
            seq_len, batch_size, self.num_heads_per_partition * self.attn_head_dim
        )

        # Output projection
        output, output_bias = self.linear_proj(output)

        return output, output_bias
