# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import ProcessGroup
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)

from areal.experimental.models.archon.activation_checkpoint import apply_ac
from areal.experimental.models.archon.compile import apply_compile
from areal.experimental.models.archon.utils import (
    validate_cp_constraints,
    validate_tp_constraints,
)
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
    return logging.getLogger(f"[Archon Qwen2Parallelize Rank {rank}]")


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
        # for low precision training, it's useful to always save
        # the result of max, since the absolute maximum is
        # used to compute the scaling factor for quantization.
        torch.ops.aten.max.default,
        torch._higher_order_ops.flex_attention,
        torch.ops.areal._varlen_attn.default,
        # When torch.compile is used, inductor wraps compiled code in this HOP.
        # Saving its output avoids re-compilation during backward recompute and
        # ensures SAC correctly interacts with compiled regions.
        # NOTE: Upgrading PyTorch will enable this in the future.
        # torch._higher_order_ops.inductor_compiled_code,
    }


def parallelize_qwen2(
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
    """Apply parallelization to Qwen2 model.

    This is the main entry point for parallelizing a Qwen2 model.
    It applies parallelization strategies based on parallel_dims configuration.

    Order of operations:
    1. Apply TP (Tensor Parallelism)
    2. Apply CP (Context Parallelism / Ulysses SP)
    3. Apply AC (Activation Checkpointing) - must be after TP
    4. Apply torch.compile - must be after AC, before FSDP
    5. Apply FSDP (Fully Sharded Data Parallelism)

    Args:
        model: The Qwen2 model to parallelize.
        parallel_dims: Parallel dimensions configuration containing mesh and group info.
        param_dtype: Data type for model parameters.
        reduce_dtype: Data type for gradient reduction.
        loss_parallel: Whether to keep output sharded for loss parallelism.
        cpu_offload: Whether to enable CPU offloading for FSDP.
        reshard_after_forward_policy: Policy for resharding after forward pass.
            - "default": applies default resharding behavior (disabled for PP)
            - "always": enable reshard_after_forward for all forward passes
            - "never": disable reshard_after_forward for all forward passes
        ac_config: Activation checkpointing configuration. If None, AC is not applied.
        enable_compile: Whether to apply torch.compile to TransformerBlocks.

    Returns:
        The parallelized model.

    Note:
        Context Parallelism (CP) implements Ulysses Sequence Parallelism using
        All-to-All communication. It scatters attention heads and gathers sequences.
    """
    # Apply TP (Tensor Parallelism)
    tp_mesh = parallel_dims.get_mesh("tp") if parallel_dims.tp_enabled else None
    if tp_mesh is not None:
        apply_tp(model, tp_mesh, loss_parallel=loss_parallel)

    # Apply CP (Context Parallelism / Ulysses SP)
    if parallel_dims.cp_enabled:
        cp_group = parallel_dims.get_group("cp")
        apply_cp(model, cp_group, tp_size=parallel_dims.tp)

    # AC must be after TP/CP
    if ac_config is not None and ac_config.mode != "none":
        apply_ac(
            model,
            ac_config,
            model_compile_enabled=enable_compile,
            op_sac_save_list=_get_op_sac_save_list(),
        )

    # torch.compile must be after AC, before FSDP
    if enable_compile:
        apply_compile(model)

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


def apply_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    loss_parallel: bool = True,
) -> None:
    """Apply tensor parallelism to Qwen2 model.

    This applies TP with Sequence Parallelism to the model:
    - Embedding: RowwiseParallel (output sharded on sequence dim)
    - Attention: ColwiseParallel for q/k/v, RowwiseParallel for output
    - FFN: ColwiseParallel for w1/w3, RowwiseParallel for w2
    - Final norm: SequenceParallel
    - Output (lm_head): ColwiseParallel
    - Score (critic): Replicated (small layer, no need to shard)

    Note: Unlike Qwen3, Qwen2 does NOT have Q/K normalization layers.

    Args:
        model: The model to apply TP to.
        tp_mesh: Device mesh for tensor parallelism.
        loss_parallel: Whether to keep output sharded for loss parallelism.
    """
    validate_tp_constraints(model.model_args, tp_mesh.size())

    root_plan = {}

    if model.tok_embeddings is not None:
        root_plan["tok_embeddings"] = RowwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Shard(1),
        )

    if model.norm is not None:
        root_plan["norm"] = SequenceParallel()

    if model.output is not None:
        # use_local_output=True for vocab_parallel loss
        root_plan["output"] = ColwiseParallel(
            input_layouts=Shard(1),
            output_layouts=Shard(-1) if loss_parallel else Replicate(),
            use_local_output=True,
        )

    if model.score is not None:
        root_plan["score"] = PrepareModuleInput(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
        )

    if root_plan:
        parallelize_module(model, tp_mesh, root_plan)

    for transformer_block in model.layers.values():
        layer_plan = {
            "attention_norm": SequenceParallel(),
            "attention": PrepareModuleInput(
                input_layouts=(Shard(1), Replicate(), Replicate(), None, None),
                desired_input_layouts=(
                    Replicate(),
                    Replicate(),
                    Replicate(),
                    None,
                    None,
                ),
            ),
            # Qwen2 has no Q/K norms, so all output local tensors directly
            "attention.wq": ColwiseParallel(use_local_output=True),
            "attention.wk": ColwiseParallel(use_local_output=True),
            "attention.wv": ColwiseParallel(use_local_output=True),
            "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
            "ffn_norm": SequenceParallel(),
            "feed_forward": PrepareModuleInput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            "feed_forward.w1": ColwiseParallel(),
            "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
            "feed_forward.w3": ColwiseParallel(),
        }

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

    _get_logger().info("Applied Tensor Parallelism to the model")


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype = torch.bfloat16,
    reduce_dtype: torch.dtype = torch.float32,
    pp_enabled: bool = False,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
) -> None:
    """Apply FSDP2 to Qwen2 model.

    This wraps each component with FSDP for memory-efficient training:
    - Token embeddings (separately wrapped)
    - Each TransformerBlock (separately wrapped)
    - Final norm + output/score (wrapped together)
    - Root model (for any remaining params)

    Args:
        model: The model to apply FSDP to.
        dp_mesh: Device mesh for data parallelism.
        param_dtype: Data type for model parameters.
        reduce_dtype: Data type for gradient reduction.
        pp_enabled: Whether pipeline parallelism is enabled.
        cpu_offload: Whether to enable CPU offloading.
        reshard_after_forward_policy: Policy for resharding after forward pass.
            - "default": applies default resharding behavior (disabled for PP)
            - "always": enable reshard_after_forward for all forward passes
            - "never": disable reshard_after_forward for all forward passes
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
            # For PP, by default do not reshard after forward to avoid per-microbatch
            # all-gathers, which can be expensive and non-overlapped
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


def apply_cp(
    model: nn.Module,
    cp_group: ProcessGroup,
    tp_size: int = 1,
) -> None:
    """Apply context parallelism (Ulysses SP) to Qwen2 model.

    This configures each Attention layer to use Ulysses sequence parallelism
    with All-to-All communication for distributed attention computation.

    Args:
        model: The model to apply CP to.
        cp_group: Process group for Ulysses All-to-All communication.
        tp_size: Tensor parallelism size, used to compute local head counts.

    Raises:
        ValueError: If head counts don't satisfy CP constraints.
    """
    cp_size = dist.get_world_size(cp_group)
    validate_cp_constraints(model.model_args, cp_size, tp_size)

    for transformer_block in model.layers.values():
        transformer_block.attention.set_cp_group(cp_group)

    _get_logger().info(
        f"Applied Context Parallelism (Ulysses SP) to the model, cp_size={cp_size}"
    )


__all__ = [
    "parallelize_qwen2",
    "apply_tp",
    "apply_fsdp",
    "apply_cp",
]
