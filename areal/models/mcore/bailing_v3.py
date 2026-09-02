# SPDX-License-Identifier: Apache-2.0

"""BailingMoeV3ForCausalLM (Ling V3) support for megatron-core.

This module provides:
1. HF config -> MLATransformerConfig conversion
2. Heterogeneous layer spec construction (KDA + gated MLA)

BailingMoeV3 uses:
- Mixed attention: KDA / Kimi Delta Attention (most layers) + gated MLA (every
  ``layer_group_size``-th layer). KDA replaces the Lightning Attention used in V2.5.
- MoE: sigmoid routing, grouped TopK (n_group=8, topk_group=4), shared experts.
- Dense MLP for the first ``first_k_dense_replace`` layers, MoE for the rest.

Layer pattern (layer_group_size=4): layers 0,1,2 = KDA, layer 3 = MLA, repeating.

NOTE (first bring-up scope): MTP and FP8 are intentionally NOT wired here; the tiny
config is brought up in bf16 with MTP disabled. See docs/bailing-moe-v3-adaptation-plan.md.
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

from areal.models.mcore.bailing_v3_mla import BailingV3MLASelfAttention
from areal.models.mcore.common import check_and_construct_configs, hf_to_mcore_base_args
from areal.models.mcore.kda_attention import (
    KimiDeltaAttention,
    KimiDeltaAttentionSubmodules,
)
from areal.utils import logging

logger = logging.getLogger("BailingV3")


def is_kda_layer(
    layer_number: int, layer_group_size: int, num_layers: int | None = None
) -> bool:
    """Determine if a layer uses KDA (linear attention) vs MLA.

    In BailingMoeV3, layers are grouped by ``layer_group_size``. Within each group, the
    last layer uses MLA (softmax attention) and the others use KDA (linear attention).

    For layer_group_size=4: layers 0,1,2 are KDA, layer 3 is MLA, repeating.

    Args:
        layer_number: 0-indexed layer number.
        layer_group_size: Number of layers per group.

    Returns:
        True if the layer should use KDA.
    """
    if layer_group_size <= 1:
        return False
    if num_layers is not None:
        # Treat any incomplete tail group as softmax attention. HybridEngine v3
        # normally asserts divisibility, but this keeps AReaL's mapping total.
        full_group_layers = (num_layers // layer_group_size) * layer_group_size
        if layer_number >= full_group_layers:
            return False
    return (layer_number + 1) % layer_group_size != 0


def _kda_head_dim(hf_config: PretrainedConfig) -> int:
    """KDA key/value head dim. In Ling V3 this equals ``kv_channels`` (128)."""
    return (
        getattr(hf_config, "kv_channels", None)
        or getattr(hf_config, "head_dim", None)
        or getattr(hf_config, "v_head_dim", None)
        or getattr(hf_config, "qk_nope_head_dim", 128)
    )


def hf_to_mcore_config_bailing_v3(
    hf_config: PretrainedConfig,
    dtype: torch.dtype,
) -> MLATransformerConfig:
    """Convert a BailingMoeV3 HuggingFace config to megatron-core MLATransformerConfig.

    KDA-specific knobs (conv kernel, no_kda_lora, head_dim) are NOT stored on the config;
    they are passed to the KDA module via ModuleSpec params in ``_build_kda_attn_spec``.
    """
    num_layers = hf_config.num_hidden_layers
    first_k_dense_replace = getattr(hf_config, "first_k_dense_replace", 0)
    moe_layer_freq = [0 if i < first_k_dense_replace else 1 for i in range(num_layers)]

    # Shared-expert intermediate size (direct value, else num_shared * moe_intermediate).
    shared_expert_intermediate_size = getattr(
        hf_config, "moe_shared_expert_intermediate_size", None
    )
    if shared_expert_intermediate_size is None:
        num_shared_experts = getattr(hf_config, "num_shared_experts", 0)
        intermediate_size = getattr(
            hf_config, "moe_intermediate_size", hf_config.intermediate_size
        )
        shared_expert_intermediate_size = (
            num_shared_experts * intermediate_size if num_shared_experts > 0 else None
        )

    base_args = hf_to_mcore_base_args(
        hf_config=hf_config,
        dtype=dtype,
        use_cpu_initialization=False,
        add_bias_linear=False,
        add_qkv_bias=False,
        qk_layernorm=True,
        attention_softmax_in_fp32=True,
        cross_entropy_loss_fusion=False,
    )

    # MLA-specific parameters (for MLA layers).
    #
    # CRITICAL (carried over from V2.5): AReaL's mcore MLA receives qk_pos_emb_head_dim
    # directly as the pure RoPE slice, so rotary_percent MUST be 1.0 (NOT the 0.5 that
    # ant-megatron uses, where it passes kv_channels=128 and takes half). Using 0.5 here
    # would halve the RoPE frequency table and badly degrade MLA accuracy.
    #
    # MLATransformerConfig defaults are tuned for DeepSeek-V2 YaRN; we pin them to plain
    # rope to avoid spurious mscale scaling (rotary_scaling_factor=40 by default!).
    rope_scaling = getattr(hf_config, "rope_scaling", None) or {}
    rotary_scaling_factor = rope_scaling.get("factor", 1.0)
    mla_args = {
        "multi_latent_attention": True,
        "q_lora_rank": getattr(hf_config, "q_lora_rank", None),
        "kv_lora_rank": getattr(hf_config, "kv_lora_rank", 512),
        "qk_head_dim": getattr(hf_config, "qk_nope_head_dim", 128),
        "qk_pos_emb_head_dim": getattr(hf_config, "qk_rope_head_dim", 64),
        "v_head_dim": getattr(hf_config, "v_head_dim", 128),
        "rope_type": "rope",
        "rotary_base": getattr(hf_config, "rope_theta", 10000.0),
        "rotary_percent": 1.0,
        "rotary_scaling_factor": rotary_scaling_factor,
        "apply_rope_fusion": False,
        "mscale": 0.707,
        "mscale_all_dim": 0.707,
        "original_max_position_embeddings": (
            rope_scaling.get("original_max_position_embeddings")
            or getattr(hf_config, "original_max_position_embeddings", None)
            or getattr(hf_config, "max_position_embeddings", 4096)
        ),
    }

    # MoE-specific parameters (same router family as V2.5: sigmoid + grouped TopK).
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
        # Bias update rate only affects expert-bias drift across steps (not the forward),
        # so it does not influence single-step SFT-loss alignment. Default frozen here for
        # determinism; production training may set the HF value (~1e-3).
        "moe_router_bias_update_rate": getattr(
            hf_config, "router_bias_update_speed", 0.0
        ),
        "moe_z_loss_coeff": 3.5e-6,
    }

    all_args = {**base_args, **mla_args, **moe_args}
    return check_and_construct_configs(all_args, MLATransformerConfig)


def _te_linear_and_norm():
    """Return (ColumnParallel, RowParallel, Norm) classes, preferring TE variants."""
    try:
        from megatron.core.extensions.transformer_engine import (
            TEColumnParallelLinear,
            TENorm,
            TERowParallelLinear,
        )

        return TEColumnParallelLinear, TERowParallelLinear, TENorm
    except ImportError:
        from megatron.core.tensor_parallel import (
            ColumnParallelLinear as TEColumnParallelLinear,
        )
        from megatron.core.tensor_parallel import (
            RowParallelLinear as TERowParallelLinear,
        )
        from megatron.core.transformer.torch_norm import WrappedTorchNorm as TENorm

        return TEColumnParallelLinear, TERowParallelLinear, TENorm


def _build_kda_attn_spec(hf_config: PretrainedConfig) -> ModuleSpec:
    """Build a ModuleSpec for KimiDeltaAttention with params from the HF config.

    KDA-specific knobs travel via ModuleSpec ``params`` (not the TransformerConfig) so we
    do not have to extend cli_args / TransformerConfig for the bring-up.
    """
    col, row, norm = _te_linear_and_norm()
    return ModuleSpec(
        module=KimiDeltaAttention,
        submodules=KimiDeltaAttentionSubmodules(
            in_proj=col,
            beta_proj=col,
            out_norm=norm,
            out_proj=row,
        ),
        params={
            "head_dim": _kda_head_dim(hf_config),
            # V3 uses ``short_conv_kernel_size``; fall back to the V2.5 name and then 4.
            "conv_kernel_dim": getattr(
                hf_config,
                "short_conv_kernel_size",
                getattr(hf_config, "linear_conv_kernel_dim", 4),
            ),
            "no_kda_lora": getattr(hf_config, "no_kda_lora", True),
            "use_qk_l2norm": getattr(hf_config, "use_qk_l2norm", True),
            # Clamped (safe) decay gate. BailingMoeV3 ckpts set kda_safe_gate=True /
            # kda_lower_bound=-5.0; these must reach chunk_kda or the decay gate
            # silently reverts to the unbounded softplus form (wrong loss).
            "safe_gate": getattr(hf_config, "kda_safe_gate", False),
            "lower_bound": getattr(hf_config, "kda_lower_bound", None),
        },
    )


def _build_gated_mla_spec(base_mla_spec: ModuleSpec, enable_gate: bool) -> ModuleSpec:
    """Swap the MLA module for the v3 gated variant, keeping the same submodules/params."""
    spec = copy.deepcopy(base_mla_spec)
    if enable_gate:
        spec.submodules.self_attention.module = BailingV3MLASelfAttention
    return spec


def make_mcore_layer_specs_bailing_v3(
    tf_config: MLATransformerConfig,
    hf_config: PretrainedConfig,
    use_te: bool = True,
    vp_stage: int | None = None,
) -> TransformerBlockSubmodules:
    """Build heterogeneous layer specs for BailingMoeV3 (KDA + gated MLA).

    Creates 4 layer-spec variants (KDA/MLA x Dense/MoE). KDA layers use the custom
    ``KimiDeltaAttention`` module; MLA layers use ``BailingV3MLASelfAttention`` (mcore MLA
    plus head-wise gate). When PP>1, the full spec list is sliced for the current stage.
    """
    assert tf_config.normalization == "RMSNorm", "only RMSNorm is supported"

    layer_group_size = getattr(hf_config, "layer_group_size", 1)
    num_layers = tf_config.num_layers
    first_k_dense_replace = getattr(hf_config, "first_k_dense_replace", 0)
    gate_granularity = getattr(hf_config, "gated_attention_proj_granularity_type", None)
    if gate_granularity is None:
        enable_gate = bool(getattr(hf_config, "enable_gated_attention", False))
    else:
        if gate_granularity != "head_wise":
            raise ValueError(
                "BailingMoeV3 currently supports only head-wise gated MLA; got "
                f"gated_attention_proj_granularity_type={gate_granularity!r}."
            )
        enable_gate = True

    # The KDA/MLA pattern is derived from layer_group_size (last layer of each group is
    # MLA). Incomplete tail groups are kept as MLA as a conservative fallback;
    # HybridEngine v3 normally asserts divisibility.

    _, _, te_norm = _te_linear_and_norm()

    # MLA layer specs (gated MLA via module swap on the standard MLA spec).
    mla_dense_base = get_gpt_layer_with_transformer_engine_spec(
        num_experts=None,
        moe_grouped_gemm=False,
        qk_layernorm=tf_config.qk_layernorm,
        multi_latent_attention=True,
    )
    mla_moe_base = get_gpt_layer_with_transformer_engine_spec(
        num_experts=tf_config.num_moe_experts,
        moe_grouped_gemm=tf_config.moe_grouped_gemm,
        qk_layernorm=tf_config.qk_layernorm,
        multi_latent_attention=True,
    )
    mla_dense_spec = _build_gated_mla_spec(mla_dense_base, enable_gate)
    mla_moe_spec = _build_gated_mla_spec(mla_moe_base, enable_gate)

    # KDA layer specs: start from standard (non-MLA) specs for correct MLP/layernorm, then
    # replace self_attention with KDA. CRITICAL: restore a real input_layernorm (TENorm)
    # because KDA's in_proj is a plain TEColumnParallelLinear (no fused layernorm), unlike
    # the standard fused TELayerNormColumnParallelLinear QKV that the base spec assumes.
    kda_attn_spec = _build_kda_attn_spec(hf_config)
    kda_dense_base = get_gpt_layer_with_transformer_engine_spec(
        num_experts=None,
        moe_grouped_gemm=False,
        qk_layernorm=True,
        multi_latent_attention=False,
    )
    kda_moe_base = get_gpt_layer_with_transformer_engine_spec(
        num_experts=tf_config.num_moe_experts,
        moe_grouped_gemm=tf_config.moe_grouped_gemm,
        qk_layernorm=True,
        multi_latent_attention=False,
    )
    kda_dense_spec = copy.deepcopy(kda_dense_base)
    kda_dense_spec.submodules.self_attention = kda_attn_spec
    kda_dense_spec.submodules.input_layernorm = te_norm
    kda_moe_spec = copy.deepcopy(kda_moe_base)
    kda_moe_spec.submodules.self_attention = kda_attn_spec
    kda_moe_spec.submodules.input_layernorm = te_norm

    # Per-layer assignment.
    layer_specs = []
    for layer_idx in range(num_layers):
        is_kda = is_kda_layer(layer_idx, layer_group_size, num_layers)
        is_moe = layer_idx >= first_k_dense_replace
        if is_kda:
            spec = kda_moe_spec if is_moe else kda_dense_spec
        else:
            spec = mla_moe_spec if is_moe else mla_dense_spec
        layer_specs.append(spec)

    n_kda = sum(
        1 for i in range(num_layers) if is_kda_layer(i, layer_group_size, num_layers)
    )
    n_mla = num_layers - n_kda
    n_moe = sum(1 for i in range(num_layers) if i >= first_k_dense_replace)
    n_dense = num_layers - n_moe
    logger.info(
        f"Built BailingV3 layer specs: {num_layers} layers, "
        f"layer_group_size={layer_group_size}, first_k_dense={first_k_dense_replace}, "
        f"num_experts={tf_config.num_moe_experts}, gated_mla={enable_gate}"
    )
    logger.info(
        f"Layer composition: {n_kda} KDA + {n_mla} MLA, {n_dense} Dense + {n_moe} MoE"
    )

    # KDA supports HybridEngine-style all2all CP. Heads are moved from CP sequence
    # shards into CP head shards before KDA, so each TP partition must have a head count
    # divisible by CP.
    if tf_config.context_parallel_size > 1 and n_kda > 0:
        tp_size = tf_config.tensor_model_parallel_size
        cp_size = tf_config.context_parallel_size
        heads_per_tp = tf_config.num_attention_heads // tp_size
        if heads_per_tp % cp_size != 0:
            raise ValueError(
                "For BailingMoeV3 KDA with CP, num_attention_heads / TP "
                f"({heads_per_tp}) must be divisible by CP ({cp_size})."
            )
        logger.info(
            f"KDA all2all CP enabled: CP={cp_size}, "
            f"heads_per_tp={heads_per_tp}, heads_per_cp={heads_per_tp // cp_size}"
        )

    # PP slicing: TransformerBlock._build_layers() builds ALL specs without slicing, so we
    # must pre-slice for the current pipeline stage (mirrors get_gpt_decoder_block_spec).
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

    if use_te:
        layer_norm_impl = te_norm
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
