# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointWrapper,
)
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

from areal.experimental.models.archon import moe as moe_module
from areal.experimental.models.archon.activation_checkpoint import apply_ac
from areal.experimental.models.archon.compile import Compilable
from areal.experimental.models.archon.moe import grouped_experts
from areal.utils import logging

if TYPE_CHECKING:
    from areal.experimental.models.archon import ArchonParallelDims
    from areal.experimental.models.archon.activation_checkpoint import (
        ActivationCheckpointConfig,
    )


@functools.cache
def _get_logger() -> logging.Logger:
    """Get rank-aware logger for this module."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    return logging.getLogger(f"[Archon Qwen3_5Parallelize Rank {rank}]")


def _get_op_sac_save_list() -> set[torch._ops.OpOverload]:
    # Import varlen to register torch.ops.areal._varlen_attn custom op
    from areal.experimental.models.archon.attention import varlen as _  # noqa: F401

    return {
        torch.ops.aten.mm.default,
        torch.ops.aten._scaled_dot_product_efficient_attention.default,
        torch.ops.aten._scaled_dot_product_flash_attention.default,
        torch.ops.aten._scaled_dot_product_cudnn_attention.default,
        torch.ops.aten._scaled_dot_product_attention_math.default,
        torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default,
        torch.ops._c10d_functional.reduce_scatter_tensor.default,
        torch.ops._c10d_functional.all_to_all_single.default,
        # for low precision training, save the result of max for quantization.
        torch.ops.aten.max.default,
        torch._higher_order_ops.flex_attention,
        torch.ops.areal._varlen_attn.default,
    }


def parallelize_qwen3_5(
    model: nn.Module,
    parallel_dims: ArchonParallelDims,
    param_dtype: torch.dtype = torch.bfloat16,
    reduce_dtype: torch.dtype = torch.float32,
    loss_parallel: bool = True,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
    ac_config: ActivationCheckpointConfig | None = None,
    enable_compile: bool = True,
) -> nn.Module:
    """Apply parallelization to Qwen3.5 hybrid model.

    Currently applies FSDP only. TP, CP, EP are not yet supported and will
    be logged as warnings if requested.

    Order of operations:
        1. TP (not yet supported)
        2. EP+TP for MoE (not yet supported)
        3. CP (not yet supported)
        4. AC (Activation Checkpointing) — applied if configured
        5. torch.compile — applied if enabled
        6. FSDP (Fully Sharded Data Parallelism) — always applied

    Args:
        model: The Qwen3.5 model to parallelize.
        parallel_dims: Parallel dimensions configuration.
        param_dtype: Data type for model parameters.
        reduce_dtype: Data type for gradient reduction.
        loss_parallel: Whether to keep output sharded for loss parallelism.
        cpu_offload: Whether to enable CPU offloading for FSDP.
        reshard_after_forward_policy: Policy for resharding after forward pass.
        ac_config: Activation checkpointing configuration.
        enable_compile: Whether to apply torch.compile to TransformerBlocks.

    Returns:
        The parallelized model.
    """
    # Log warnings if non-DP parallelism is requested
    if parallel_dims.tp_enabled:
        _get_logger().warning(
            "Qwen3.5 does not yet support Tensor Parallelism. "
            "TP will be ignored. Use FSDP-only for now."
        )

    if parallel_dims.cp_enabled:
        _get_logger().warning(
            "Qwen3.5 does not yet support Context Parallelism. "
            "CP will be ignored. Use FSDP-only for now."
        )

    if parallel_dims.ep_enabled:
        _get_logger().warning(
            "Qwen3.5 does not yet support Expert Parallelism. "
            "EP will be ignored. Use FSDP-only for now."
        )

    # AC must be after TP/CP (which are currently no-ops)
    if ac_config is not None and ac_config.mode != "none":
        apply_ac(
            model,
            ac_config,
            model_compile_enabled=enable_compile,
            op_sac_save_list=_get_op_sac_save_list(),
        )

    # torch.compile must be after AC, before FSDP
    if enable_compile:
        _apply_compile(model)

    # Apply FSDP
    dp_mesh = parallel_dims.get_mesh("dp_shard_cp")
    if dp_mesh is not None:
        apply_fsdp(
            model,
            dp_mesh,
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            pp_enabled=parallel_dims.pp_enabled,
            cpu_offload=cpu_offload,
            reshard_after_forward_policy=reshard_after_forward_policy,
        )

    if getattr(model.model_args, "enable_weight_tying", False):
        if model.output is not None and model.tok_embeddings is not None:
            model.output.weight = model.tok_embeddings.weight

    return model


def apply_fsdp(
    model: nn.Module,
    dp_mesh: torch.distributed.device_mesh.DeviceMesh,
    param_dtype: torch.dtype = torch.bfloat16,
    reduce_dtype: torch.dtype = torch.float32,
    pp_enabled: bool = False,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
) -> None:
    """Apply FSDP2 to Qwen3.5 model.

    Wraps each component with FSDP for memory-efficient training:
      - Token embeddings (separately wrapped)
      - Each TransformerBlock (separately wrapped, both layer types)
      - Final norm + output/score (wrapped together)
      - Root model (for any remaining params)

    Args:
        model: The model to apply FSDP to.
        dp_mesh: Device mesh for data parallelism (dp_shard_cp).
        param_dtype: Data type for model parameters.
        reduce_dtype: Data type for gradient reduction.
        pp_enabled: Whether pipeline parallelism is enabled.
        cpu_offload: Whether to enable CPU offloading.
        reshard_after_forward_policy: Policy for resharding after forward pass.
    """
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config: dict[str, Any] = {"mesh": dp_mesh, "mp_policy": mp_policy}

    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    match reshard_after_forward_policy:
        case "always":
            reshard_after_forward = True
        case "never":
            reshard_after_forward = False
        case "default":
            reshard_after_forward = not pp_enabled
        case _:
            raise ValueError(
                f"Invalid reshard_after_forward_policy: {reshard_after_forward_policy}."
            )

    if model.tok_embeddings is not None:
        fully_shard(
            model.tok_embeddings,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    for transformer_block in model.layers.values():
        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    # As an optimization, do not reshard_after_forward the last layers by default
    # since FSDP would prefetch them immediately after the forward pass
    final_layers = [model.norm] if model.norm is not None else []
    if model.output is not None:
        final_layers.append(model.output)
    if hasattr(model, "score") and model.score is not None:
        final_layers.append(model.score)

    if final_layers:
        fully_shard(
            final_layers,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward_policy == "always",
        )

    fully_shard(model, **fsdp_config)

    _get_logger().info("Applied FSDP to the model")
    if cpu_offload:
        _get_logger().info("Applied CPU Offloading to the model")


def _apply_compile(model: Compilable) -> None:
    """Apply torch.compile to Qwen3.5 model (hybrid-aware).

    Both full_attention and linear_attention TransformerBlocks are compiled.

    For MoE layers, compile submodules separately to avoid graph breaks
    from FSDP(GroupedExperts). For non-MoE layers, compile the whole block.

    Must be called AFTER AC, BEFORE FSDP.

    Args:
        model: The model to compile.
    """
    torch._dynamo.config.capture_scalar_outputs = True
    # Workaround for https://github.com/pytorch/pytorch/issues/166926
    if hasattr(torch._C._dynamo.eval_frame, "_set_lru_cache"):
        torch._C._dynamo.eval_frame._set_lru_cache(False)

    for name, block in model.layers.items():
        if getattr(block, "moe_enabled", False):
            # MoE layer: compile submodules separately to avoid graph breaks
            if isinstance(block, CheckpointWrapper):
                inner_block = block._checkpoint_wrapped_module
            else:
                inner_block = block

            for attr_name, submod in inner_block.named_children():
                assert getattr(block, attr_name) == getattr(inner_block, attr_name)

                if isinstance(submod, moe_module.MoE):
                    for moe_attr, moe_submod in submod.named_children():
                        if moe_attr == "experts":
                            continue
                        setattr(
                            submod,
                            moe_attr,
                            torch.compile(
                                moe_submod, backend="inductor", fullgraph=True
                            ),
                        )
                elif attr_name in ("attention_norm", "ffn_norm"):
                    continue
                else:
                    setattr(
                        inner_block,
                        attr_name,
                        torch.compile(submod, backend="inductor", fullgraph=True),
                    )
        else:
            model.layers[name] = torch.compile(
                block,
                backend="inductor",
                fullgraph=True,
            )

    # Also compile grouped_mm if MoE is present
    already_patched = (
        "_run_experts_grouped_mm_dynamic"
        in grouped_experts._run_experts_grouped_mm.__qualname__
    )
    if not already_patched:
        grouped_experts._run_experts_grouped_mm = torch.compile(
            grouped_experts._run_experts_grouped_mm,
            backend="inductor",
            fullgraph=True,
        )

    _get_logger().info(
        f"Compiled {len(model.layers)} TransformerBlocks with torch.compile (hybrid-aware)"
    )


__all__ = [
    "parallelize_qwen3_5",
    "apply_fsdp",
]
