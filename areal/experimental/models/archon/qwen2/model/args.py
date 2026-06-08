# SPDX-License-Identifier: Apache-2.0

# Adapted from torchtitan: torchtitan/models/qwen3/model/args.py

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from areal.experimental.models.archon.base import BaseModelArgs

if TYPE_CHECKING:
    from transformers import PretrainedConfig


@dataclass
class Qwen2ModelArgs(BaseModelArgs):
    """Model arguments for Qwen2. Default values are for Qwen2-0.5B."""

    dim: int = 896
    n_layers: int = 24
    n_heads: int = 14
    n_kv_heads: int = 2
    vocab_size: int = 151936
    head_dim: int = 64
    hidden_dim: int = 4864

    norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    max_seq_len: int = 32768
    depth_init: bool = True
    attention_bias: bool = True
    eos_id: int = 151645
    enable_weight_tying: bool = False
    is_critic: bool = False
    num_labels: int = 1

    @classmethod
    def from_hf_config(
        cls,
        hf_config: PretrainedConfig,
        is_critic: bool = False,
        **kwargs,
    ) -> Qwen2ModelArgs:
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
            rope_theta=cls._get_rope_theta(hf_config, default=10000.0),
            max_seq_len=getattr(hf_config, "max_position_embeddings", 32768),
            attention_bias=getattr(hf_config, "attention_bias", True),
            eos_id=getattr(hf_config, "eos_token_id", 151645),
            enable_weight_tying=getattr(hf_config, "tie_word_embeddings", False),
            is_critic=is_critic,
            attn_type=kwargs.get("attn_type", BaseModelArgs.attn_type),
        )


__all__ = ["Qwen2ModelArgs"]
