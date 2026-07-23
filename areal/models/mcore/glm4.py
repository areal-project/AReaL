# SPDX-License-Identifier: Apache-2.0

import torch
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer import TransformerConfig
from transformers import PretrainedConfig

from areal.models.mcore.common import (
    check_and_construct_configs,
    hf_to_mcore_base_args,
)


def hf_to_mcore_config_glm4moe(
    hf_config: PretrainedConfig, dtype: torch.dtype
) -> TransformerConfig:
    """Convert GLM-4 MoE HuggingFace config to Megatron TransformerConfig."""
    args: dict = hf_to_mcore_base_args(
        hf_config=hf_config,
        dtype=dtype,
        use_cpu_initialization=False,
        add_bias_linear=getattr(hf_config, "add_bias_linear", False),
        add_qkv_bias=getattr(hf_config, "attention_bias", False),
        qk_layernorm=getattr(hf_config, "use_qk_norm", False),
    )

    # GLM-4 MoE specific settings
    if hasattr(hf_config, "num_experts"):
        args["num_moe_experts"] = hf_config.num_experts
    if hasattr(hf_config, "n_routed_experts"):  # GLM-4 uses n_routed_experts
        args["num_moe_experts"] = hf_config.n_routed_experts
    if hasattr(hf_config, "num_experts_per_tok"):
        args["moe_router_topk"] = hf_config.num_experts_per_tok
    if hasattr(hf_config, "moe_intermediate_size"):
        args["moe_ffn_hidden_size"] = hf_config.moe_intermediate_size
    if hasattr(hf_config, "router_aux_loss_coef"):
        args["moe_aux_loss_coeff"] = hf_config.router_aux_loss_coef

    # GLM-4 specific: shared expert settings
    if hasattr(hf_config, "shared_expert_intermediate_size"):
        args["moe_shared_expert_intermediate_size"] = (
            hf_config.shared_expert_intermediate_size
        )
    elif hasattr(hf_config, "n_shared_experts") and hf_config.n_shared_experts > 0:
        # If n_shared_experts exists but shared_expert_intermediate_size doesn't, use intermediate_size
        args["moe_shared_expert_intermediate_size"] = hf_config.intermediate_size

    # Post-attention and post-MLP layernorm (GLM-4 specific)
    if hasattr(hf_config, "post_layer_norm") and hf_config.post_layer_norm:
        args["apply_residual_connection_post_layernorm"] = True

    return check_and_construct_configs(args, TransformerConfig)


def make_mcore_layer_specs_glm4moe(tfconfig: TransformerConfig, use_te: bool = True):
    """Create layer specs for GLM-4 MoE architecture."""
    assert tfconfig.normalization == "RMSNorm", "only RMSNorm is supported for GLM-4"

    # Get base spec
    spec = get_gpt_decoder_block_spec(tfconfig, use_transformer_engine=use_te)

    # FORCE REMOVE QK layernorms if config says so
    if not tfconfig.qk_layernorm:
        for layer_spec in spec.layer_specs:
            if hasattr(layer_spec, "submodules") and hasattr(
                layer_spec.submodules, "self_attention"
            ):
                attn_submodules = layer_spec.submodules.self_attention.submodules
                # Set to None to remove these layers
                attn_submodules.q_layernorm = None
                attn_submodules.k_layernorm = None

    return spec
