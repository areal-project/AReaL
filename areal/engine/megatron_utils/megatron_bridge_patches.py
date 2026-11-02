# SPDX-License-Identifier: Apache-2.0

"""Runtime adaptations for Megatron-Bridge on NPU."""

from __future__ import annotations

from functools import wraps
from typing import Any

import areal.utils.logging as logging

logger = logging.getLogger("MegatronBridgePatches")


def patch_qwen35_hybrid_lora_specs(unfuse_gdn: bool = True) -> None:
    """Unfuse Qwen3.5 attention and MLP inputs without dropping GDN layers.

    Megatron-Bridge builds Qwen3.5 as a heterogeneous list containing both
    Gated DeltaNet and full-attention layers. Replacing that list with a Qwen3
    dense spec would drop GDN entirely. Instead, wrap the native block-spec
    builder and replace fused LayerNorm+linear modules that LoRA targets.
    Full-attention ``linear_qkv`` and MLP ``linear_fc1`` are always unfused.
    When ``unfuse_gdn`` is true, Gated DeltaNet ``in_proj`` is split into
    ``PTNorm`` plus ``ColumnParallelLinear`` and ``out_proj`` becomes
    ``RowParallelLinear``. The hybrid layer layout itself is preserved.

    The resulting ``in_proj`` is still one LoRA adapter on the concatenated
    qkv/z/b/a matrix. See ``docs/qwen35-gdn-lora.md``.
    """
    try:
        from megatron.bridge.models.qwen_vl import qwen35_vl_provider
        from megatron.core.tensor_parallel.layers import (
            ColumnParallelLinear,
            RowParallelLinear,
        )
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
            # Full-attention layers expose linear_qkv. GDN layers expose
            # in_proj (fused with the input layernorm) and out_proj.
            if attention_submodules is not None and hasattr(
                attention_submodules, "linear_qkv"
            ):
                submodules.input_layernorm = PTNorm
                attention_submodules.linear_qkv = ColumnParallelLinear
                if getattr(attention, "metainfo", None) is not None:
                    attention.metainfo["fuse_input_layernorm"] = False
            elif (
                unfuse_gdn
                and attention_submodules is not None
                and hasattr(attention_submodules, "in_proj")
            ):
                # MindSpeed's fused LayerNorm+linear does not return the
                # normalized activation Megatron-Bridge LoRA needs. Split GDN
                # in_proj the same way as full-attention QKV, and use a plain
                # row-parallel out_proj so that projection can take an adapter.
                submodules.input_layernorm = PTNorm
                attention_submodules.in_proj = ColumnParallelLinear
                attention_submodules.out_proj = RowParallelLinear
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
    if unfuse_gdn:
        logger.info(
            "Using unfused full-attention QKV, GDN in_proj/out_proj, and MLP FC1 "
            "specs for Qwen3.5 LoRA; the native hybrid layer layout is preserved."
        )
    else:
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


def patch_qwen3_vl_vision_config() -> None:
    """Keep inherited MindSpeed MoE options out of dense Qwen vision configs."""
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_config import (
        Qwen3VLTransformerConfig,
    )

    original = Qwen3VLTransformerConfig.__post_init__
    if getattr(original, "_areal_dense_vision_config", False):
        return

    @wraps(original)
    def _post_init(config: Any) -> None:
        if not config.num_moe_experts:
            for name in (
                "gemm_gradient_accumulation_fusion",
                "moe_alltoall_overlap_comm",
                "moe_permute_fusion",
                "use_fused_moe_token_permute_and_unpermute",
            ):
                if hasattr(config, name):
                    setattr(config, name, False)
            if hasattr(config, "moe_zero_memory"):
                config.moe_zero_memory = "disable"
            if hasattr(config, "moe_zero_memory_num_layers"):
                config.moe_zero_memory_num_layers = None
        original(config)

    _post_init._areal_dense_vision_config = True
    Qwen3VLTransformerConfig.__post_init__ = _post_init
