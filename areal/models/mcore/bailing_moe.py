# SPDX-License-Identifier: Apache-2.0

"""BailingMoeV2_5ForCausalLM support for megatron-core.

This module provides:
1. HF config -> MLATransformerConfig conversion
2. Heterogeneous layer spec construction (Lightning Attention + MLA)

BailingMoeV2_5 uses:
- Mixed attention: Lightning Attention (most layers) + MLA (every layer_group_size-th layer)
- MoE: sigmoid routing, grouped TopK (n_group=8, topk_group=4), shared experts
- Dense layers for first `first_k_dense_replace` layers, MoE for the rest
"""

import copy

import torch
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.transformer.enums import LayerType
from megatron.core.transformer.multi_latent_attention import MLATransformerConfig
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import (
    TransformerBlockSubmodules,
    get_num_layers_to_build,
)
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from transformers import PretrainedConfig

from areal.models.mcore.common import check_and_construct_configs, hf_to_mcore_base_args
from areal.models.mcore.lightning_attention import (
    LightningAttentionSubmodules,
    LightningSelfAttention,
)
from areal.utils import logging

logger = logging.getLogger("BailingMoe")


def _patch_mla_thd_rope_for_cp():
    """Fix MLA RoPE for THD packed sequences with CP>1.

    Root cause: MLA's qkv_up_proj_and_rope_apply slices rotary_pos_emb to
    [0:q_len] where q_len = total_tokens/CP. But _get_thd_freqs_on_this_cp_rank
    needs the FULL frequency table (covering original sequence length) to select
    zigzag positions. The pre-sliced table is too short, causing index-out-of-bounds.

    Fix: In apply_rotary_pos_emb, when detecting unfused THD + CP>1 with a
    truncated freq table, reconstruct the full table using the linear relationship
    freqs[p] = p * freqs[1] (exact because freqs = outer(positions, inv_freq)).
    """
    import megatron.core.models.common.embeddings.rope_utils as rope_utils
    from megatron.core import parallel_state

    _original = rope_utils.apply_rotary_pos_emb

    def _patched(t, freqs, config, cu_seqlens=None, mscale=1.0, cp_group=None):
        if cp_group is None:
            cp_group = parallel_state.get_context_parallel_group()

        # Fix: extend pre-sliced freq table for unfused THD + CP>1
        if (
            not config.apply_rope_fusion
            and cu_seqlens is not None
            and cp_group is not None
            and cp_group.size() > 1
            and freqs.size(0) >= 2
        ):
            # cu_seqlens is original (pre-CP). _get_thd_freqs_on_this_cp_rank
            # accesses freqs up to max(S_i) where S_i are individual seq lengths.
            # cu_seqlens[-1] = sum(S_i) >= max(S_i), safe upper bound.
            max_needed = cu_seqlens[-1].item()
            if freqs.size(0) < max_needed:
                # Reconstruct: freqs[p] = p * freqs[1] (freqs[0] = 0, linear in position)
                unit = freqs[1:2]  # [1, 1, 1, dim]
                positions = torch.arange(
                    max_needed, device=freqs.device, dtype=freqs.dtype
                )
                # Reshape positions to broadcast with unit: [max_needed, 1, 1, 1] * [1, 1, 1, dim]
                positions = positions.view(-1, *([1] * (unit.dim() - 1)))
                freqs = positions * unit

        return _original(t, freqs, config, cu_seqlens, mscale, cp_group)

    rope_utils.apply_rotary_pos_emb = _patched
    # Also patch the reference imported in multi_latent_attention module
    try:
        import megatron.core.transformer.multi_latent_attention as mla_module

        mla_module.apply_rotary_pos_emb = _patched
    except (ImportError, AttributeError):
        pass
    # Also patch the reference imported in rotary_pos_embedding module
    try:
        import megatron.core.models.common.embeddings.rotary_pos_embedding as rope_module

        rope_module.apply_rotary_pos_emb = _patched
    except (ImportError, AttributeError):
        pass
    logger.info(
        "Patched apply_rotary_pos_emb: extend truncated freq table for MLA THD+CP>1"
    )


_patch_mla_thd_rope_for_cp()


def is_lightning_layer(layer_number: int, layer_group_size: int) -> bool:
    """Determine if a layer uses Lightning Attention (vs MLA).

    In BailingMoeV2_5, layers are grouped by `layer_group_size`. Within each group,
    the last layer uses MLA (standard softmax attention) and all others use
    Lightning Attention (linear attention with learned decay).

    For layer_group_size=4: layers 0,1,2 are Lightning, layer 3 is MLA, repeating.

    Args:
        layer_number: 0-indexed layer number
        layer_group_size: Number of layers per group

    Returns:
        True if the layer should use Lightning Attention
    """
    if layer_group_size <= 1:
        return False
    return (layer_number + 1) % layer_group_size != 0


def hf_to_mcore_config_bailing_moe(
    hf_config: PretrainedConfig,
    dtype: torch.dtype,
) -> MLATransformerConfig:
    """Convert BailingMoeV2_5 HuggingFace config to megatron-core MLATransformerConfig.

    Args:
        hf_config: HuggingFace PretrainedConfig for BailingMoeV2_5ForCausalLM
        dtype: Data type for the model parameters

    Returns:
        MLATransformerConfig with MLA + MoE parameters
    """
    # Build moe_layer_freq as a list: 0 for dense, 1 for MoE
    num_layers = hf_config.num_hidden_layers
    first_k_dense_replace = getattr(hf_config, "first_k_dense_replace", 0)
    moe_layer_freq = [0 if i < first_k_dense_replace else 1 for i in range(num_layers)]

    # Shared expert intermediate size: use direct config value if available
    shared_expert_intermediate_size = getattr(
        hf_config, "moe_shared_expert_intermediate_size", None
    )
    if shared_expert_intermediate_size is None:
        # Fallback: compute from num_shared_experts * moe_intermediate_size
        num_shared_experts = getattr(hf_config, "num_shared_experts", 0)
        intermediate_size = getattr(
            hf_config, "moe_intermediate_size", hf_config.intermediate_size
        )
        shared_expert_intermediate_size = (
            num_shared_experts * intermediate_size if num_shared_experts > 0 else None
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

    # MLA-specific parameters (for MLA layers)
    #
    # rotary_scaling_factor: Must be set to 1 for plain rope (no YaRN scaling).
    # MLATransformerConfig defaults rotary_scaling_factor=40 (for DeepSeek-V2 YaRN),
    # which triggers mscale computation even with rope_type="rope", causing incorrect
    # softmax_scale when the mcore default mscale != 0.
    #
    # CRITICAL FIX: HybridEngine changed MLATransformerConfig defaults:
    # - mscale: 0.707 → 1.0
    # - mscale_all_dim: 0.707 → 0.0
    # We explicitly set them to maintain compatibility with original rope/mscale behavior.
    rope_scaling = getattr(hf_config, "rope_scaling", None) or {}
    rotary_scaling_factor = rope_scaling.get("factor", 1.0)
    mla_args = {
        "multi_latent_attention": True,
        "q_lora_rank": getattr(hf_config, "q_lora_rank", None),
        "kv_lora_rank": getattr(hf_config, "kv_lora_rank", 512),
        "qk_head_dim": getattr(hf_config, "qk_nope_head_dim", 128),
        "qk_pos_emb_head_dim": getattr(hf_config, "qk_rope_head_dim", 64),
        "v_head_dim": getattr(hf_config, "v_head_dim", 128),
        # RoPE (for MLA layers; Lightning layers handle their own RoPE)
        "rope_type": "rope",
        "rotary_base": (getattr(hf_config, "rope_parameters", None) or {}).get(
            "rope_theta", getattr(hf_config, "rope_theta", 10000.0)
        ),
        "rotary_percent": 1.0,
        "rotary_scaling_factor": rotary_scaling_factor,
        "apply_rope_fusion": False,
        # YaRN RoPE scaling parameters (explicitly set to prevent HybridEngine default drift)
        "mscale": 0.707,
        "mscale_all_dim": 0.707,
    }

    # MoE-specific parameters
    # moe_grouped_gemm=True: use batched GEMM for all experts (matches HybridEngine).
    # moe_router_dtype="fp32": compute routing scores in FP32 for stable expert assignment.
    # Both settings improve determinism and match HybridEngine's configuration.
    moe_args = {
        "num_moe_experts": getattr(hf_config, "num_experts", None),
        "moe_router_topk": getattr(hf_config, "num_experts_per_tok", 8),
        "moe_router_score_function": getattr(hf_config, "scoring_func", "sigmoid"),
        "moe_router_num_groups": getattr(hf_config, "n_group", 8),
        "moe_router_group_topk": getattr(hf_config, "topk_group", 4),
        "moe_router_topk_scaling_factor": getattr(
            hf_config, "routed_scaling_factor", None
        ),
        "moe_ffn_hidden_size": getattr(hf_config, "moe_intermediate_size", None),
        "moe_shared_expert_intermediate_size": shared_expert_intermediate_size,
        "moe_layer_freq": moe_layer_freq,
        "moe_router_enable_expert_bias": True,
        "moe_router_load_balancing_type": "none",
        "moe_grouped_gemm": True,
        "moe_router_dtype": "fp32",
    }

    # Merge all args
    all_args = {**base_args, **mla_args, **moe_args}

    return check_and_construct_configs(all_args, MLATransformerConfig)


def _build_lightning_attn_spec(hf_config: PretrainedConfig) -> ModuleSpec:
    """Build a ModuleSpec for LightningSelfAttention with params from HF config.

    Returns:
        ModuleSpec that can replace self_attention in TransformerLayerSubmodules
    """
    try:
        from megatron.core.extensions.transformer_engine import (
            TEColumnParallelLinear,
            TERowParallelLinear,
        )
    except ImportError:
        from megatron.core.tensor_parallel import (
            ColumnParallelLinear as TEColumnParallelLinear,
        )
        from megatron.core.tensor_parallel import (
            RowParallelLinear as TERowParallelLinear,
        )

    attn_head_dim = getattr(
        hf_config, "attn_head_dim", getattr(hf_config, "head_dim", 256)
    )
    partial_rotary_factor = getattr(hf_config, "partial_rotary_factor", 0.5)
    linear_attn_norm_group_size = getattr(
        hf_config,
        "linear_attn_norm_group_size",
        getattr(hf_config, "group_norm_size", 1),
    )

    return ModuleSpec(
        module=LightningSelfAttention,
        submodules=LightningAttentionSubmodules(
            linear_qkv=TEColumnParallelLinear,
            linear_gate=TEColumnParallelLinear,
            linear_proj=TERowParallelLinear,
        ),
        params={
            "attn_head_dim": attn_head_dim,
            "partial_rotary_factor": partial_rotary_factor,
            "linear_attn_norm_group_size": linear_attn_norm_group_size,
        },
    )


def make_mcore_layer_specs_bailing_moe(
    tf_config: MLATransformerConfig,
    hf_config: PretrainedConfig,
    use_te: bool = True,
    vp_stage: int | None = None,
) -> TransformerBlockSubmodules:
    """Build heterogeneous layer specs for BailingMoeV2_5.

    Creates 4 types of layer specs based on attention type and MLP type:
    1. Lightning Attention + Dense MLP
    2. Lightning Attention + MoE MLP
    3. MLA Attention + Dense MLP
    4. MLA Attention + MoE MLP

    Lightning Attention layers use our custom LightningSelfAttention module with the fla
    kernel. MLA layers use megatron-core's built-in MLASelfAttention.

    When PP>1, the full list of layer specs is sliced to only include layers for the
    current pipeline stage, matching the slicing logic in get_gpt_decoder_block_spec().

    Args:
        tf_config: MLATransformerConfig with all model parameters
        hf_config: HF config for layer_group_size and Lightning attention params
        use_te: Whether to use Transformer Engine modules
        vp_stage: Virtual pipeline stage (for VPP support)

    Returns:
        TransformerBlockSubmodules with heterogeneous layer specs (PP-sliced if PP>1)
    """
    assert tf_config.normalization == "RMSNorm", "only RMSNorm is supported"

    layer_group_size = getattr(hf_config, "layer_group_size", 1)
    num_layers = tf_config.num_layers
    first_k_dense_replace = getattr(hf_config, "first_k_dense_replace", 0)

    # Build MLA layer specs (using megatron-core's built-in MLASelfAttention)
    mla_dense_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=None,
        moe_grouped_gemm=False,
        qk_layernorm=tf_config.qk_layernorm,
        multi_latent_attention=True,
    )
    mla_moe_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=tf_config.num_moe_experts,
        moe_grouped_gemm=tf_config.moe_grouped_gemm,
        qk_layernorm=tf_config.qk_layernorm,
        multi_latent_attention=True,
    )

    # Build Lightning Attention layer specs
    # Start from standard non-MLA specs (correct MLP, layernorm, etc.)
    # then replace self_attention with our custom LightningSelfAttention
    lightning_attn_spec = _build_lightning_attn_spec(hf_config)

    lightning_dense_base = get_gpt_layer_with_transformer_engine_spec(
        num_experts=None,
        moe_grouped_gemm=False,
        qk_layernorm=True,
        multi_latent_attention=False,
    )
    lightning_moe_base = get_gpt_layer_with_transformer_engine_spec(
        num_experts=tf_config.num_moe_experts,
        moe_grouped_gemm=tf_config.moe_grouped_gemm,
        qk_layernorm=True,
        multi_latent_attention=False,
    )

    # Replace self_attention in Lightning specs with our custom module.
    # CRITICAL: The base non-MLA specs use input_layernorm=IdentityOp with the layernorm
    # fused into TELayerNormColumnParallelLinear for QKV. Our custom LightningSelfAttention
    # uses plain TEColumnParallelLinear (no fused layernorm), so we must restore a real
    # input_layernorm to ensure hidden_states are normalized before QKV and gate projections.
    try:
        from megatron.core.extensions.transformer_engine import TENorm
    except ImportError:
        from megatron.core.transformer.torch_norm import WrappedTorchNorm as TENorm

    lightning_dense_spec = copy.deepcopy(lightning_dense_base)
    lightning_dense_spec.submodules.self_attention = lightning_attn_spec
    lightning_dense_spec.submodules.input_layernorm = TENorm

    lightning_moe_spec = copy.deepcopy(lightning_moe_base)
    lightning_moe_spec.submodules.self_attention = lightning_attn_spec
    lightning_moe_spec.submodules.input_layernorm = TENorm

    # Build per-layer specs
    layer_specs = []
    for layer_idx in range(num_layers):
        is_lightning = is_lightning_layer(layer_idx, layer_group_size)
        is_moe = layer_idx >= first_k_dense_replace

        if is_lightning:
            spec = lightning_moe_spec if is_moe else lightning_dense_spec
        else:
            spec = mla_moe_spec if is_moe else mla_dense_spec
        layer_specs.append(spec)

    # Log layer composition (before PP slicing)
    n_lightning = sum(
        1 for i in range(num_layers) if is_lightning_layer(i, layer_group_size)
    )
    n_mla = num_layers - n_lightning
    n_moe = sum(1 for i in range(num_layers) if i >= first_k_dense_replace)
    n_dense = num_layers - n_moe
    logger.info(
        f"Built BailingMoe layer specs: {num_layers} layers, "
        f"layer_group_size={layer_group_size}, first_k_dense={first_k_dense_replace}, "
        f"num_experts={tf_config.num_moe_experts}"
    )
    logger.info(
        f"Layer composition: {n_lightning} Lightning + {n_mla} MLA, "
        f"{n_dense} Dense + {n_moe} MoE"
    )

    # Validate: Lightning Attention + CP requires H/TP divisible by CP
    if tf_config.context_parallel_size > 1 and n_lightning > 0:
        tp_size = tf_config.tensor_model_parallel_size
        cp_size = tf_config.context_parallel_size
        heads_per_tp = tf_config.num_attention_heads // tp_size
        if heads_per_tp % cp_size != 0:
            raise ValueError(
                f"For Lightning Attention with CP, num_heads_per_tp_partition "
                f"({heads_per_tp}) must be divisible by context_parallel_size "
                f"({cp_size}). Got {tf_config.num_attention_heads} total heads, "
                f"TP={tp_size}, CP={cp_size}."
            )
        logger.info(
            f"Lightning Attention CP enabled: CP={cp_size}, "
            f"heads_per_tp={heads_per_tp}, heads_per_cp={heads_per_tp // cp_size}"
        )

    # PP slicing: when PP>1, only include layers for the current pipeline stage.
    # This replicates the slicing logic from get_gpt_decoder_block_spec().
    # TransformerBlock._build_layers() builds ALL specs in layer_specs without slicing,
    # so we must pre-slice here.
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
        local_lightning = sum(
            1
            for i, spec in enumerate(local_layer_specs)
            if hasattr(spec, "submodules")
            and hasattr(spec.submodules, "self_attention")
            and hasattr(spec.submodules.self_attention, "module")
            and spec.submodules.self_attention.module is LightningSelfAttention
        )
        logger.info(
            f"PP slicing: building {len(local_layer_specs)}/{num_layers} layers "
            f"({local_lightning} Lightning) for this pipeline stage"
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

    block_spec = TransformerBlockSubmodules(
        layer_specs=local_layer_specs,
        layer_norm=layer_norm_impl,
    )

    return block_spec
