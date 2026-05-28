# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from areal.experimental.models.archon.base import BaseModelArgs
from areal.experimental.models.archon.moe import MoEArgs

if TYPE_CHECKING:
    from transformers import PretrainedConfig


@dataclass
class Qwen3_5ModelArgs(BaseModelArgs):
    """Model arguments for Qwen3.5 (dense + MoE, hybrid architecture).

    Qwen3.5 is a hybrid architecture with both full_attention and
    linear_attention (GatedDeltaNet) layers. The layer pattern is
    configured via ``layer_types``.

    Attributes:
        dim: Hidden size.
        n_layers: Number of transformer layers.
        n_heads: Number of attention heads (full attention layers).
        n_kv_heads: Number of key/value heads (full attention layers).
        head_dim: Per-head dimension. Explicit in config (default 256),
            NOT always ``hidden_size // num_attention_heads``.
        hidden_dim: Intermediate size for dense MLP / FFN.
        vocab_size: Vocabulary size.
        norm_eps: RMSNorm epsilon.
        rope_theta: RoPE base frequency.
        partial_rotary_factor: Fraction of head_dim for RoPE (default 0.25).
        layer_types: Per-layer type list — ``"full_attention"`` or
            ``"linear_attention"``.
        linear_conv_kernel_dim: Conv1d kernel size for GatedDeltaNet.
        linear_key_head_dim: Per-head key dim for linear attention.
        linear_value_head_dim: Per-head value dim for linear attention.
        linear_num_key_heads: Number of key heads for linear attention.
        linear_num_value_heads: Number of value heads for linear attention.
        attention_bias: Whether attention projections use bias.
        moe_enabled: Whether MoE is enabled for this model.
        moe_inter_dim: Intermediate dimension for MoE experts.
        shared_expert_intermediate_size: Intermediate dimension for shared experts.
        num_experts: Number of MoE experts.
        num_experts_per_tok: Number of experts per token.
        moe_args: MoE configuration.
    """

    dim: int = 3072
    n_layers: int = 36
    n_heads: int = 24
    n_kv_heads: int = 4
    vocab_size: int = 151936
    head_dim: int = 256
    hidden_dim: int = 9216

    norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    partial_rotary_factor: float = 0.25

    max_seq_len: int = 131072
    depth_init: bool = True

    eos_id: int = 151645
    enable_weight_tying: bool = False
    is_critic: bool = False
    num_labels: int = 1

    # Hybrid layer configuration
    layer_types: list[str] = field(default_factory=list)

    # Linear attention (GatedDeltaNet) configuration
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32

    attention_bias: bool = False

    # MoE configuration
    moe_enabled: bool = False
    moe_inter_dim: int = 768
    shared_expert_intermediate_size: int = 0
    num_experts: int = 0
    num_experts_per_tok: int = 0
    moe_args: MoEArgs | None = None

    def __post_init__(self):
        if self.layer_types and len(self.layer_types) != self.n_layers:
            raise ValueError(
                f"layer_types length ({len(self.layer_types)}) must equal "
                f"n_layers ({self.n_layers})"
            )

    @classmethod
    def from_hf_config(
        cls,
        hf_config: PretrainedConfig,
        is_critic: bool = False,
        **kwargs,
    ) -> Qwen3_5ModelArgs:
        # Handle composite VLM config (model_type="qwen3_5" or "qwen3_5_moe")
        if hasattr(hf_config, "text_config"):
            hf_config = hf_config.text_config

        # Extract partial_rotary_factor (may be top-level or inside rope_parameters)
        partial_rotary_factor = getattr(hf_config, "partial_rotary_factor", None)
        if partial_rotary_factor is None:
            rope_params = getattr(hf_config, "rope_parameters", {}) or {}
            partial_rotary_factor = rope_params.get("partial_rotary_factor", 0.25)

        # Extract rope_theta (may be top-level or inside rope_parameters)
        rope_theta = cls._get_rope_theta(hf_config, default=1000000.0)

        # Check if MoE is enabled
        num_experts = getattr(hf_config, "num_experts", None)
        if num_experts is None:
            num_experts = getattr(hf_config, "num_local_experts", None)
        moe_enabled = num_experts is not None and num_experts > 1

        # Build MoEArgs from HF config if MoE is enabled
        moe_args = None
        if moe_enabled:
            moe_args = MoEArgs.from_hf_config(hf_config)
            # Shared expert configuration (Qwen3.5 MoE uses explicit intermediate size + gate)
            shared_inter = getattr(hf_config, "shared_expert_intermediate_size", 0)
            if shared_inter > 0:
                moe_args.shared_expert_intermediate_size = shared_inter
                moe_args.use_shared_expert_gate = True
            # Fallback: older configs with num_shared_experts field
            if hasattr(hf_config, "num_shared_experts"):
                moe_args.num_shared_experts = hf_config.num_shared_experts

        return cls(
            dim=hf_config.hidden_size,
            n_layers=hf_config.num_hidden_layers,
            n_heads=hf_config.num_attention_heads,
            n_kv_heads=getattr(
                hf_config, "num_key_value_heads", hf_config.num_attention_heads
            ),
            vocab_size=hf_config.vocab_size,
            head_dim=getattr(
                hf_config,
                "head_dim",
                hf_config.hidden_size // hf_config.num_attention_heads,
            ),
            # MoE configs lack intermediate_size (only moe_intermediate_size).
            # hidden_dim is unused when moe_enabled since FeedForward is replaced by MoE.
            hidden_dim=getattr(hf_config, "intermediate_size", 0),
            norm_eps=hf_config.rms_norm_eps,
            rope_theta=rope_theta,
            partial_rotary_factor=partial_rotary_factor,
            max_seq_len=getattr(hf_config, "max_position_embeddings", 131072),
            eos_id=getattr(hf_config, "eos_token_id", 151645),
            enable_weight_tying=getattr(hf_config, "tie_word_embeddings", False),
            is_critic=is_critic,
            # Truncate layer_types to n_layers (partial configs keep the full list).
            layer_types=list(getattr(hf_config, "layer_types", []))[
                : hf_config.num_hidden_layers
            ],
            linear_conv_kernel_dim=getattr(hf_config, "linear_conv_kernel_dim", 4),
            linear_key_head_dim=getattr(hf_config, "linear_key_head_dim", 128),
            linear_value_head_dim=getattr(hf_config, "linear_value_head_dim", 128),
            linear_num_key_heads=getattr(hf_config, "linear_num_key_heads", 16),
            linear_num_value_heads=getattr(hf_config, "linear_num_value_heads", 32),
            attention_bias=getattr(hf_config, "attention_bias", False),
            moe_enabled=moe_enabled,
            moe_inter_dim=getattr(hf_config, "moe_intermediate_size", 768),
            shared_expert_intermediate_size=getattr(
                hf_config, "shared_expert_intermediate_size", 0
            ),
            num_experts=num_experts if num_experts is not None else 0,
            num_experts_per_tok=getattr(hf_config, "num_experts_per_tok", 0),
            moe_args=moe_args,
            attn_type=kwargs.get("attn_type", BaseModelArgs.attn_type),
        )


__all__ = ["Qwen3_5ModelArgs"]
