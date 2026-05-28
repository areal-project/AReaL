# SPDX-License-Identifier: Apache-2.0

"""DeepseekV3ForCausalLM / GLM-5.1 / GLM-4.7-Flash support for megatron-core.

This module provides:
1. HF config -> MLATransformerConfig conversion
2. Homogeneous MLA layer spec construction
3. DSA (Dynamic Sparse Attention) support for GLM-5.1

DeepSeek V3 / GLM-5.1 / GLM-4.7-Flash uses:
- MLA (Multi-head Latent Attention) for all layers
- MoE: sigmoid routing, grouped TopK, shared experts
- Dense layers for first `first_k_dense_replace` layers, MoE for the rest
- YaRN RoPE scaling (optional, GLM-4.7-Flash uses plain RoPE)
- DSA indexer (GLM-5.1 only): per-layer sparse attention token selector

Note: The MLA RoPE patch for CP>1 is applied in bailing_moe.py at module level
and automatically benefits all MLA models including DeepSeek V3.
"""

import os

import torch
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.transformer.enums import LayerType
from megatron.core.transformer.multi_latent_attention import MLATransformerConfig
from megatron.core.transformer.transformer_block import (
    TransformerBlockSubmodules,
    get_num_layers_to_build,
)
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from transformers import PretrainedConfig

from areal.models.mcore.common import check_and_construct_configs, hf_to_mcore_base_args
from areal.utils import logging

logger = logging.getLogger("DeepSeekV3")


def _has_dsa(hf_config: PretrainedConfig) -> bool:
    # DSA enabled when the HF config exposes indexer topk + n_heads. Uses a
    # slime-style native DSA MLA module (see dsa_mla_attention.py) that
    # inherits Attention directly instead of going through mcore's DSAttention
    # container, so packed THD inputs work without modification.
    return (
        getattr(hf_config, "index_topk", None) is not None
        and getattr(hf_config, "index_n_heads", None) is not None
    )


def hf_to_mcore_config_deepseek_v3(
    hf_config: PretrainedConfig,
    dtype: torch.dtype,
) -> MLATransformerConfig:
    """Convert DeepSeek V3 / GLM-5.1 HuggingFace config to MLATransformerConfig.

    DeepSeek V3 architecture uses MLA for all layers (no Lightning Attention),
    which makes it simpler than BailingMoeV2_5.

    Args:
        hf_config: HuggingFace PretrainedConfig for DeepseekV3ForCausalLM
        dtype: Data type for the model parameters

    Returns:
        MLATransformerConfig with MLA + MoE parameters
    """
    # MTP layers are not used during RL training (only for SGLang EAGLE-style
    # inference). Setting AREAL_DISABLE_MTP=1 zeroes out
    # num_nextn_predict_layers so no MTP module is built, preventing rare bwd
    # NaN paths through the MTP block.
    if os.environ.get("AREAL_DISABLE_MTP", "0") == "1":
        if getattr(hf_config, "num_nextn_predict_layers", 0):
            logger.warning(
                f"AREAL_DISABLE_MTP=1: overriding "
                f"hf_config.num_nextn_predict_layers "
                f"from {hf_config.num_nextn_predict_layers} to 0"
            )
            hf_config.num_nextn_predict_layers = 0

    # Build moe_layer_freq: 0 for dense, 1 for MoE
    num_layers = hf_config.num_hidden_layers
    first_k_dense_replace = getattr(hf_config, "first_k_dense_replace", 3)
    moe_layer_freq = [0 if i < first_k_dense_replace else 1 for i in range(num_layers)]

    # Shared expert intermediate size
    n_shared_experts = getattr(hf_config, "n_shared_experts", 0)
    moe_intermediate_size = getattr(
        hf_config, "moe_intermediate_size", hf_config.intermediate_size
    )
    shared_expert_intermediate_size = (
        n_shared_experts * moe_intermediate_size if n_shared_experts > 0 else None
    )

    # Get base args common to all models
    base_args = hf_to_mcore_base_args(
        hf_config=hf_config,
        dtype=dtype,
        use_cpu_initialization=False,
        add_bias_linear=False,
        add_qkv_bias=False,
        qk_layernorm=True,
    )

    # MLA-specific parameters
    #
    # DeepSeek V3 uses YaRN RoPE scaling. The rotary_scaling_factor, mscale,
    # and mscale_all_dim must be set correctly from the HF config's rope_scaling.
    #
    # DeepSeek V3 HF config rope_scaling example:
    #   {"type": "yarn", "factor": 4.0, "mscale": 0.707, "mscale_all_dim": 0.707, ...}
    rope_scaling = getattr(hf_config, "rope_scaling", None) or {}
    rotary_scaling_factor = rope_scaling.get("factor", 1.0)

    # rope_theta: top-level field or inside rope_parameters (GLM-5.1)
    rope_theta = getattr(hf_config, "rope_theta", None)
    if rope_theta is None:
        rope_params = getattr(hf_config, "rope_parameters", None) or {}
        rope_theta = rope_params.get("rope_theta", 10000.0)

    mla_args = {
        "multi_latent_attention": True,
        "q_lora_rank": getattr(hf_config, "q_lora_rank", None),
        "kv_lora_rank": getattr(hf_config, "kv_lora_rank", 512),
        "qk_head_dim": getattr(hf_config, "qk_nope_head_dim", 128),
        "qk_pos_emb_head_dim": getattr(hf_config, "qk_rope_head_dim", 64),
        "v_head_dim": getattr(hf_config, "v_head_dim", 128),
        # RoPE
        "rope_type": "rope",
        "rotary_base": rope_theta,
        "rotary_percent": getattr(hf_config, "partial_rotary_factor", 1.0),
        "rotary_scaling_factor": rotary_scaling_factor,
        "apply_rope_fusion": False,
    }
    if rope_scaling.get("type") == "yarn" or rope_scaling.get("rope_type") == "yarn":
        mla_args["mscale"] = rope_scaling.get("mscale", 0.707)
        mla_args["mscale_all_dim"] = rope_scaling.get("mscale_all_dim", 0.707)

    # MoE-specific parameters
    n_routed_experts = getattr(hf_config, "n_routed_experts", None)
    if n_routed_experts is None:
        n_routed_experts = getattr(hf_config, "num_local_experts", None)

    moe_args = {
        "num_moe_experts": n_routed_experts,
        "moe_router_topk": getattr(hf_config, "num_experts_per_tok", 8),
        "moe_router_score_function": getattr(hf_config, "scoring_func", "sigmoid"),
        "moe_router_num_groups": getattr(hf_config, "n_group", 8),
        "moe_router_group_topk": getattr(hf_config, "topk_group", 4),
        "moe_router_topk_scaling_factor": getattr(
            hf_config, "routed_scaling_factor", None
        ),
        "moe_ffn_hidden_size": moe_intermediate_size,
        "moe_shared_expert_intermediate_size": shared_expert_intermediate_size,
        "moe_layer_freq": moe_layer_freq,
        "moe_router_enable_expert_bias": True,
        "moe_router_load_balancing_type": "none",
        "moe_grouped_gemm": True,
        "moe_router_dtype": "fp32",
        "moe_router_bias_update_rate": 0.0,
        "moe_z_loss_coeff": 3.5e-6,
        "moe_enable_routing_replay": bool(os.environ.get("AREAL_DUMP_ROUTING", "")),
    }
    if moe_args["moe_enable_routing_replay"]:
        logger.info("AREAL_DUMP_ROUTING is set; moe_enable_routing_replay=True")

    # Numerical stability flags:
    # bf16 attention softmax + MoE forward output can amplify numeric range,
    # breaking bf16 linear/RMSNorm backward on long-context training.
    # attention_softmax_in_fp32 is the most impactful stability knob.
    # check_and_construct_configs (common.py) silently drops keys not on
    # MLATransformerConfig, so older mcore versions still work.
    stability_args = {
        "attention_softmax_in_fp32": True,
        "cross_entropy_loss_fusion": False,  # use fp32 unfused cross-entropy
        "disable_bf16_reduced_precision_matmul": True,
    }

    # Merge all args
    all_args = {**base_args, **mla_args, **moe_args, **stability_args}

    # DSA (Dynamic Sparse Attention) parameters for GLM-5.1
    if _has_dsa(hf_config):
        dsa_indexer_loss_coeff = getattr(hf_config, "dsa_indexer_loss_coeff", 0.0)
        # NOTE: do NOT set experimental_attention_variant="dsa" — that triggers
        # mcore's own DSA code paths inside multi_latent_attention.py, which we
        # bypass by providing a slime-style custom self_attention module spec.
        dsa_args = {
            "dsa_indexer_n_heads": hf_config.index_n_heads,
            "dsa_indexer_head_dim": hf_config.index_head_dim,
            "dsa_indexer_topk": hf_config.index_topk,
            "dsa_indexer_loss_coeff": dsa_indexer_loss_coeff,
            "dsa_indexer_use_sparse_loss": getattr(
                hf_config, "dsa_indexer_use_sparse_loss", False
            ),
        }
        all_args.update(dsa_args)
        logger.info(
            f"DSA enabled: index_n_heads={hf_config.index_n_heads}, "
            f"index_head_dim={hf_config.index_head_dim}, "
            f"index_topk={hf_config.index_topk}, "
            f"indexer_loss_coeff={dsa_indexer_loss_coeff}"
        )

    return check_and_construct_configs(all_args, MLATransformerConfig)


def make_mcore_layer_specs_deepseek_v3(
    tf_config: MLATransformerConfig,
    hf_config: PretrainedConfig,
    use_te: bool = True,
    vp_stage: int | None = None,
) -> TransformerBlockSubmodules:
    """Build homogeneous MLA layer specs for DeepSeek V3 / GLM-5.1.

    All layers use MLA attention. The only variation is Dense MLP vs MoE MLP,
    determined by `first_k_dense_replace`.

    Args:
        tf_config: MLATransformerConfig with all model parameters
        hf_config: HF config for first_k_dense_replace
        use_te: Whether to use Transformer Engine modules
        vp_stage: Virtual pipeline stage (for VPP support)

    Returns:
        TransformerBlockSubmodules with MLA layer specs (PP-sliced if PP>1)
    """
    assert tf_config.normalization == "RMSNorm", "only RMSNorm is supported"

    num_layers = tf_config.num_layers
    first_k_dense_replace = getattr(hf_config, "first_k_dense_replace", 3)
    use_dsa = _has_dsa(hf_config)

    if use_dsa:
        # Build a slime-style DSA self-attention spec that wraps our custom
        # DSAMLASelfAttention (inherits Attention directly, not DSAttention).
        from megatron.core.extensions.transformer_engine import (
            TEDotProductAttention,
            TELayerNormColumnParallelLinear,
            TELinear,
            TENorm,
            TERowParallelLinear,
        )
        from megatron.core.transformer.enums import AttnMaskType
        from megatron.core.transformer.identity_op import IdentityOp
        from megatron.core.transformer.spec_utils import ModuleSpec

        from areal.models.mcore.dsa_mla_attention import (
            DSAMLASelfAttention,
            DSASelfAttentionSubmodules,
        )

        dsa_attention_spec = ModuleSpec(
            module=DSAMLASelfAttention,
            params={"attn_mask_type": AttnMaskType.causal},
            submodules=DSASelfAttentionSubmodules(
                linear_q_down_proj=TELinear,
                linear_q_up_proj=TELayerNormColumnParallelLinear,
                linear_kv_down_proj=TELinear,
                linear_kv_up_proj=TELayerNormColumnParallelLinear,
                linear_v_up_proj=IdentityOp,
                core_attention=TEDotProductAttention,
                linear_proj=TERowParallelLinear,
                q_layernorm=IdentityOp,
                kv_layernorm=IdentityOp,
                wq_b=TELinear,
                wk=TELinear,
                k_norm=TENorm,
                weights_proj=TELinear,
            ),
        )

    # Build MLA layer specs (all layers use MLA, optionally with DSA)
    def _make_layer_spec(num_experts, moe_grouped_gemm):
        base_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=num_experts,
            moe_grouped_gemm=moe_grouped_gemm,
            qk_layernorm=tf_config.qk_layernorm,
            multi_latent_attention=True,
        )
        if use_dsa:
            base_spec.submodules.self_attention = dsa_attention_spec
        return base_spec

    mla_dense_spec = _make_layer_spec(num_experts=None, moe_grouped_gemm=False)
    mla_moe_spec = _make_layer_spec(
        num_experts=tf_config.num_moe_experts,
        moe_grouped_gemm=tf_config.moe_grouped_gemm,
    )

    # Build per-layer specs
    layer_specs = []
    for layer_idx in range(num_layers):
        is_moe = layer_idx >= first_k_dense_replace
        spec = mla_moe_spec if is_moe else mla_dense_spec
        layer_specs.append(spec)

    n_moe = sum(1 for i in range(num_layers) if i >= first_k_dense_replace)
    n_dense = num_layers - n_moe
    attn_type = "MLA+DSA" if use_dsa else "MLA"
    logger.info(
        f"Built DeepSeek V3 layer specs: {num_layers} layers (all {attn_type}), "
        f"first_k_dense={first_k_dense_replace}, "
        f"num_experts={tf_config.num_moe_experts}"
    )
    logger.info(f"Layer composition: {n_dense} Dense + {n_moe} MoE")

    # PP slicing: when PP>1, only include layers for the current pipeline stage.
    num_layers_to_build = get_num_layers_to_build(tf_config, vp_stage=vp_stage)

    if tf_config.pipeline_model_parallel_layout is not None:
        local_layer_specs = [
            layer_specs[layer_id]
            for layer_id in tf_config.pipeline_model_parallel_layout.get_layer_id_list(
                layer_type=LayerType.decoder, vp_stage=vp_stage
            )
        ]
    elif num_layers_to_build < num_layers:
        offset = get_transformer_layer_offset(tf_config, vp_stage=vp_stage)
        local_layer_specs = layer_specs[offset : offset + num_layers_to_build]
    else:
        local_layer_specs = layer_specs

    if len(local_layer_specs) != num_layers:
        logger.info(
            f"PP slicing: building {len(local_layer_specs)}/{num_layers} layers "
            f"for this pipeline stage"
        )

    # Get layer norm implementation
    if use_te:
        try:
            from megatron.core.extensions.transformer_engine import TENorm

            layer_norm_impl = TENorm
        except ImportError:
            from megatron.core.transformer.torch_norm import WrappedTorchNorm

            layer_norm_impl = WrappedTorchNorm
    else:
        try:
            from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

            layer_norm_impl = FusedLayerNorm
        except ImportError:
            from megatron.core.transformer.torch_norm import WrappedTorchNorm

            layer_norm_impl = WrappedTorchNorm

    return TransformerBlockSubmodules(
        layer_specs=local_layer_specs,
        layer_norm=layer_norm_impl,
    )
