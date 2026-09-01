# SPDX-License-Identifier: Apache-2.0

"""Runtime adaptations for Megatron-Bridge LoRA on NPU."""

from __future__ import annotations

from typing import Any

import areal.utils.logging as logging

logger = logging.getLogger("MegatronBridgePatches")


def patch_qwen35_hybrid_lora_specs() -> None:
    """Unfuse Qwen3.5 attention/MLP inputs without changing its GDN layout.

    Megatron-Bridge builds Qwen3.5 as a heterogeneous list containing both
    Gated DeltaNet and full-attention layers. Replacing that list with a Qwen3
    dense spec would drop GDN entirely. Instead, wrap the native block-spec
    builder and replace only the fused LayerNorm+ColumnParallelLinear modules
    that LoRA targets. GDN's fused ``in_proj`` is deliberately retained until
    it has matching adapter and export support.
    """
    try:
        from megatron.bridge.models.qwen_vl import qwen35_vl_provider
        from megatron.core.tensor_parallel.layers import ColumnParallelLinear
        from mindspeed.core.megatron_basic.megatron_basic import PTNorm
    except ImportError:
        return

    original = qwen35_vl_provider.get_transformer_block_with_experimental_attention_variant_spec
    if getattr(original, "_areal_qwen35_hybrid_lora_applied", False):
        return

    def _unfused_block_spec(*args, **kwargs):
        block_spec = original(*args, **kwargs)
        for layer_spec in block_spec.layer_specs:
            submodules = layer_spec.submodules

            attention = submodules.self_attention
            attention_submodules = getattr(attention, "submodules", None)
            # Full-attention layers expose linear_qkv; GDN layers expose
            # in_proj instead and must keep their native fused implementation.
            if attention_submodules is not None and hasattr(
                attention_submodules, "linear_qkv"
            ):
                submodules.input_layernorm = PTNorm
                attention_submodules.linear_qkv = ColumnParallelLinear
                if getattr(attention, "metainfo", None) is not None:
                    attention.metainfo["fuse_input_layernorm"] = False

            mlp = submodules.mlp
            mlp_submodules = getattr(mlp, "submodules", None)
            if mlp_submodules is not None and hasattr(mlp_submodules, "linear_fc1"):
                submodules.pre_mlp_layernorm = PTNorm
                mlp_submodules.linear_fc1 = ColumnParallelLinear
                if getattr(mlp, "metainfo", None) is not None:
                    mlp.metainfo["fuse_pre_mlp_layernorm"] = False
        return block_spec

    _unfused_block_spec._areal_qwen35_hybrid_lora_applied = True
    qwen35_vl_provider.get_transformer_block_with_experimental_attention_variant_spec = _unfused_block_spec
    logger.info(
        "Using unfused full-attention QKV and MLP FC1 specs for Qwen3.5 LoRA; "
        "the native hybrid GDN layer layout is preserved."
    )


def patch_qwen3_moe_lora_spec(provider: Any) -> None:
    """Unfuse Qwen3-MoE attention QKV without replacing its MoE layer spec.

    MindSpeed's fused LayerNorm+ColumnParallelLinear does not return the
    normalized activation required by Megatron-Bridge's LoRA wrapper.  Keep the
    provider's router and expert specifications intact, and replace only the
    attention input norm and QKV projection after the native spec is built.
    """
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear
    from mindspeed.core.megatron_basic.megatron_basic import PTNorm

    original = provider.transformer_layer_spec
    if getattr(original, "_areal_qwen3_moe_lora_applied", False):
        return

    def _unfused_layer_spec(*args, **kwargs):
        layer_spec = original(*args, **kwargs)
        layer_specs = getattr(layer_spec, "layer_specs", [layer_spec])
        for spec in layer_specs:
            submodules = spec.submodules
            attention = submodules.self_attention
            attention_submodules = attention.submodules

            submodules.input_layernorm = PTNorm
            attention_submodules.linear_qkv = ColumnParallelLinear
            if getattr(attention, "metainfo", None) is not None:
                attention.metainfo["fuse_input_layernorm"] = False
        return layer_spec

    _unfused_layer_spec._areal_qwen3_moe_lora_applied = True
    provider.transformer_layer_spec = _unfused_layer_spec
    logger.info(
        "Using unfused attention QKV spec for Qwen3-MoE LoRA; preserving the "
        "native router and expert specifications."
    )
