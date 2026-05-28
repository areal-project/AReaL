# SPDX-License-Identifier: Apache-2.0

# Adapted from torchtitan: torchtitan/models/qwen3/model/args.py

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from areal.experimental.models.archon.base import BaseModelArgs
from areal.experimental.models.archon.moe import MoEArgs

if TYPE_CHECKING:
    from transformers import PretrainedConfig


@dataclass
class Qwen3ModelArgs(BaseModelArgs):
    """Model arguments for Qwen3. Default values are for Qwen3-0.6B.

    Attributes:
        moe_enabled: Whether MoE is enabled for this model.
        moe_inter_dim: Intermediate dimension for MoE experts.
        moe_args: MoE configuration (num_experts, top_k, etc.).
        decoder_sparse_step: Interval for MoE layers. If 1, all layers use MoE.
            If 2, every other layer uses MoE (starting from layer 1).
            If 0 or negative, MoE is disabled.
    """

    dim: int = 1024
    n_layers: int = 28
    n_heads: int = 16
    n_kv_heads: int = 8
    vocab_size: int = 151936
    head_dim: int = 128
    hidden_dim: int = 3072

    norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    qk_norm: bool = True
    max_seq_len: int = 4096
    depth_init: bool = True

    eos_id: int = 151645
    enable_weight_tying: bool = False
    is_critic: bool = False
    num_labels: int = 1

    # MoE configuration
    moe_enabled: bool = False
    moe_inter_dim: int = 768
    moe_args: MoEArgs | None = None
    decoder_sparse_step: int = 1  # Every layer is MoE if moe_enabled

    @classmethod
    def from_hf_config(
        cls,
        hf_config: PretrainedConfig,
        is_critic: bool = False,
        **kwargs,
    ) -> Qwen3ModelArgs:
        # Check if MoE is enabled
        num_experts = getattr(hf_config, "num_experts", None)
        if num_experts is None:
            num_experts = getattr(hf_config, "num_local_experts", None)
        moe_enabled = num_experts is not None and num_experts > 1

        # Build MoEArgs from HF config if MoE is enabled
        moe_args = None
        if moe_enabled:
            moe_args = MoEArgs.from_hf_config(hf_config)
            # Override with additional fields from HF config
            if hasattr(hf_config, "num_shared_experts"):
                moe_args.num_shared_experts = hf_config.num_shared_experts
            router_dtype = kwargs.get("router_dtype", None)
            if router_dtype is not None:
                moe_args.router_dtype = router_dtype

        # Get decoder_sparse_step (default to 1 = all MoE layers)
        decoder_sparse_step = getattr(hf_config, "decoder_sparse_step", 1)

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
            hidden_dim=hf_config.intermediate_size,
            norm_eps=hf_config.rms_norm_eps,
            rope_theta=cls._get_rope_theta(hf_config, default=1000000.0),
            qk_norm=getattr(hf_config, "qk_norm", True),
            max_seq_len=getattr(hf_config, "max_position_embeddings", 4096),
            eos_id=getattr(hf_config, "eos_token_id", 151645),
            enable_weight_tying=getattr(hf_config, "tie_word_embeddings", False),
            is_critic=is_critic,
            moe_enabled=moe_enabled,
            moe_inter_dim=getattr(hf_config, "moe_intermediate_size", 768),
            moe_args=moe_args,
            decoder_sparse_step=decoder_sparse_step,
            attn_type=kwargs.get("attn_type", BaseModelArgs.attn_type),
        )


__all__ = ["Qwen3ModelArgs"]
