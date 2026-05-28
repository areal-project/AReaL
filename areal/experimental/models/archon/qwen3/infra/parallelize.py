# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import ProcessGroup
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointWrapper,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import Partial, Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    PrepareModuleInputOutput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)

from areal.experimental.models.archon import moe as moe_module
from areal.experimental.models.archon.activation_checkpoint import apply_ac
from areal.experimental.models.archon.compile import Compilable
from areal.experimental.models.archon.expert_parallel import (
    ExpertParallel,
    ExpertTensorParallel,
    ReordererSequenceParallel,
    TensorParallel,
)
from areal.experimental.models.archon.moe import grouped_experts
from areal.experimental.models.archon.utils import (
    validate_cp_constraints,
    validate_ep_constraints,
    validate_tp_constraints,
)
from areal.models.parallel_styles import ReplicateParallel
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
    return logging.getLogger(f"[Archon Qwen3Parallelize Rank {rank}]")


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


def parallelize_qwen3(
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
    """Apply parallelization to Qwen3 model.

    This is the main entry point for parallelizing a Qwen3 model.
    It applies parallelization strategies based on parallel_dims configuration.

    Order of operations:
    1. Apply non-MoE TP (Tensor Parallelism for dense layers)
    2. Apply MoE EP+TP (Expert Parallelism + MoE-specific TP)
    3. Apply CP (Context Parallelism / Ulysses SP)
    4. Apply AC (Activation Checkpointing) - must be after TP/EP
    5. Apply torch.compile - must be after AC, before FSDP
    6. Apply FSDP (Fully Sharded Data Parallelism)

    Args:
        model: The Qwen3 model to parallelize.
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

        Expert Parallelism (EP) distributes MoE experts across devices, using
        All-to-All communication to dispatch tokens to their assigned experts.
    """
    # Apply non-MoE TP first (attention, norms, dense FFN layers)
    tp_mesh = parallel_dims.get_mesh("tp") if parallel_dims.tp_enabled else None
    if tp_mesh is not None:
        apply_non_moe_tp(model, tp_mesh, loss_parallel=loss_parallel)

    # Apply MoE EP+TP (handles both MoE-specific TP and EP)
    # Only apply when tp > 1 or ep > 1
    ep_mesh = parallel_dims.get_mesh("ep") if parallel_dims.ep_enabled else None
    ep_tp_mesh = parallel_dims.get_mesh("ep_tp") if parallel_dims.etp_enabled else None
    if tp_mesh is not None or ep_mesh is not None:
        apply_moe_ep_tp(
            model,
            tp_mesh,
            ep_mesh,
            etp=parallel_dims.etp,
            ep_tp_mesh=ep_tp_mesh,
        )

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
        ep_enabled = parallel_dims.ep > 1
        _apply_compile(model, ep_enabled=ep_enabled)

    # Apply FSDP
    # dp_shard_cp mesh for FSDP sharding of dense params
    dp_mesh = parallel_dims.get_mesh("dp_shard_cp")
    if dp_mesh is not None:
        # dp_shard_mod_ep mesh for MoE experts FSDP sharding (only when EP enabled)
        dp_mod_ep_mesh = parallel_dims.get_mesh("dp_shard_mod_ep")

        apply_fsdp(
            model,
            dp_mesh,
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            pp_enabled=parallel_dims.pp_enabled,
            cpu_offload=cpu_offload,
            reshard_after_forward_policy=reshard_after_forward_policy,
            ep_degree=parallel_dims.ep,
            dp_mod_ep_mesh=dp_mod_ep_mesh,
            gradient_divide_factor=parallel_dims.fsdp_gradient_divide_factor,
        )

    if getattr(model.model_args, "enable_weight_tying", False):
        if model.output is not None and model.tok_embeddings is not None:
            model.output.weight = model.tok_embeddings.weight

    return model


def apply_non_moe_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    loss_parallel: bool = True,
) -> None:
    """Apply tensor parallelism to non-MoE components of Qwen3 model.

    This handles TP for all non-MoE components:
    - Embedding: RowwiseParallel (output sharded on sequence dim)
    - Attention: ColwiseParallel for q/k/v, RowwiseParallel for output
    - FFN (non-MoE layers only): ColwiseParallel for w1/w3, RowwiseParallel for w2
    - Q/K norm: SequenceParallel
    - Final norm: SequenceParallel
    - Output (lm_head): ColwiseParallel
    - Score (critic): Replicated (small layer, no need to shard)

    For MoE layers, this function only handles attention and norms.
    MoE-specific parallelism (input/output conversion, router, experts)
    is handled by apply_moe_ep_tp().

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
            # wq/wk output DTensor for q_norm/k_norm; wv outputs local tensor (no norm)
            "attention.wq": ColwiseParallel(use_local_output=False),
            "attention.wk": ColwiseParallel(use_local_output=False),
            "attention.wv": ColwiseParallel(use_local_output=True),
            # DTensor -> local conversion happens in model.py via maybe_to_local()
            "attention.q_norm": SequenceParallel(sequence_dim=2),
            "attention.k_norm": SequenceParallel(sequence_dim=2),
            "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
            "ffn_norm": SequenceParallel(),
        }

        # Only apply FFN TP for non-MoE layers
        # MoE layers have their FFN handled by apply_moe_ep_tp()
        is_moe_layer = getattr(transformer_block, "moe_enabled", False)
        if not is_moe_layer and transformer_block.feed_forward is not None:
            layer_plan.update(
                {
                    "feed_forward": PrepareModuleInput(
                        input_layouts=(Shard(1),),
                        desired_input_layouts=(Replicate(),),
                    ),
                    "feed_forward.w1": ColwiseParallel(),
                    "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
                    "feed_forward.w3": ColwiseParallel(),
                }
            )

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

    _get_logger().info("Applied Tensor Parallelism (non-MoE) to the model")


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype = torch.bfloat16,
    reduce_dtype: torch.dtype = torch.float32,
    pp_enabled: bool = False,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
    ep_degree: int = 1,
    dp_mod_ep_mesh: DeviceMesh | None = None,
    gradient_divide_factor: int | None = None,
) -> None:
    """Apply FSDP2 to Qwen3 model.

    This wraps each component with FSDP for memory-efficient training:
    - Token embeddings (separately wrapped)
    - Each TransformerBlock (separately wrapped)
    - MoE experts (separately wrapped with dp_mod_ep_mesh when EP is enabled)
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
            - "default": applies default resharding behavior (disabled for PP)
            - "always": enable reshard_after_forward for all forward passes
            - "never": disable reshard_after_forward for all forward passes
        ep_degree: Expert parallelism degree.
        dp_mod_ep_mesh: Device mesh for MoE experts FSDP sharding (dp_shard_mod_ep).
            Only used when ep_degree > 1.
        gradient_divide_factor: Gradient divide factor for FSDP.
            Used to ensure consistent gradient scaling for MoE experts.
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
        # When EP is enabled, MoE experts are sharded with dp_mod_ep_mesh
        # while the rest of the transformer block uses dp_mesh (dp_shard_cp)
        if (
            getattr(transformer_block, "moe_enabled", False)
            and ep_degree > 1
            and dp_mod_ep_mesh is not None
        ):
            fsdp_ep_config = fsdp_config.copy()
            fsdp_ep_config["mesh"] = dp_mod_ep_mesh

            # When dp_mod_ep * ep > num_experts, FSDP default dim-0 sharding
            # causes inefficiency, so we choose to do FSDP sharding on dim-1.
            _experts_shard_placement_fn = None
            if (
                dp_mod_ep_mesh.size() * ep_degree
                > transformer_block.moe.experts.num_experts
            ):
                _experts_shard_placement_fn = lambda param: Shard(1)  # noqa: E731

            # FSDP wrap the MoE experts with dp_mod_ep_mesh
            fully_shard(
                transformer_block.moe.experts,
                **fsdp_ep_config,
                reshard_after_forward=reshard_after_forward,
                shard_placement_fn=_experts_shard_placement_fn,
            )

            # Although the FSDP sharding of experts is done on a mesh of a different
            # size than other parameters, the gradient division factor should be
            # consistent with data parallelism.
            if gradient_divide_factor is not None:
                transformer_block.moe.experts.set_gradient_divide_factor(
                    gradient_divide_factor,
                )

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

    # Set up explicit prefetching when EP is enabled
    # D2H syncs in EP token dispatch can interfere with FSDP's implicit prefetching
    if ep_degree > 1:
        _setup_fsdp_prefetch(model)

    _get_logger().info("Applied FSDP to the model")
    if cpu_offload:
        _get_logger().info("Applied CPU Offloading to the model")


def _setup_fsdp_prefetch(model: nn.Module) -> None:
    """Set up explicit FSDP prefetching for EP.

    When EP is enabled, D2H syncs in token dispatch can interfere with
    FSDP's implicit prefetching. This function sets up explicit prefetch
    chains to ensure optimal overlap.

    Args:
        model: The FSDP-wrapped model.
    """
    transformer_blocks = list(model.layers.values())
    if not transformer_blocks:
        return

    # === Forward prefetch ===
    # tok_embeddings -> first block
    if model.tok_embeddings is not None:
        model.tok_embeddings.set_modules_to_forward_prefetch([transformer_blocks[0]])

    # block[i] -> block[i+1] (+ experts if MoE), or -> final layers for last block
    next_blocks = transformer_blocks[1:] + [None]
    for block, next_block in zip(transformer_blocks, next_blocks):
        if next_block is not None:
            if getattr(next_block, "moe_enabled", False):
                block.set_modules_to_forward_prefetch(
                    [next_block, next_block.moe.experts]
                )
            else:
                block.set_modules_to_forward_prefetch([next_block])
        else:
            # Last block -> final layers (norm, output/score)
            # These are wrapped together in apply_fsdp
            forward_final = [model.norm] if model.norm is not None else []
            if model.output is not None:
                forward_final.append(model.output)
            if hasattr(model, "score") and model.score is not None:
                forward_final.append(model.score)
            if forward_final:
                block.set_modules_to_forward_prefetch(forward_final)

    # === Backward prefetch ===
    reversed_blocks = list(reversed(transformer_blocks))
    prev_blocks = reversed_blocks[1:] + [None]

    # final layer (output or score) -> last block
    if model.output is not None:
        model.output.set_modules_to_backward_prefetch([reversed_blocks[0]])
    elif hasattr(model, "score") and model.score is not None:
        model.score.set_modules_to_backward_prefetch([reversed_blocks[0]])

    # block[i] -> block[i-1] (+ experts if MoE), or -> tok_embeddings for first block
    for block, prev_block in zip(reversed_blocks, prev_blocks):
        if prev_block is not None:
            if getattr(prev_block, "moe_enabled", False):
                block.set_modules_to_backward_prefetch(
                    [prev_block, prev_block.moe.experts]
                )
            else:
                block.set_modules_to_backward_prefetch([prev_block])
        elif model.tok_embeddings is not None:
            # First block -> tok_embeddings
            block.set_modules_to_backward_prefetch([model.tok_embeddings])

    _get_logger().info("Set up explicit FSDP prefetching for EP")


def apply_cp(
    model: nn.Module,
    cp_group: ProcessGroup,
    tp_size: int = 1,
) -> None:
    """Apply context parallelism (Ulysses SP) to Qwen3 model.

    This configures each Attention layer to use Ulysses sequence parallelism
    with All-to-All communication for distributed attention computation.

    Note: Pad+slice of inputs is handled by Engine layer, not Model layer.

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


def _apply_compile(model: Compilable, ep_enabled: bool = False) -> None:
    """Apply torch.compile to Qwen3 model (MoE-aware).

    For MoE layers, compile submodules separately to avoid graph breaks
    from FSDP(GroupedExperts). For non-MoE layers, compile the whole block.

    Must be called AFTER TP and AC, BEFORE FSDP.

    Args:
        model: The model to compile.
        ep_enabled: Whether Expert Parallelism is enabled. If True, marks
            dynamic shapes for varying token counts per expert.
    """
    # NOTE: This flag is needed for torch.compile to avoid graph breaking on dynamic shapes in token-choice MoE
    torch._dynamo.config.capture_scalar_outputs = True
    # Workaround for https://github.com/pytorch/pytorch/issues/166926
    # NOTE: Upgrading PyTorch will resolve this in the future.
    if hasattr(torch._C._dynamo.eval_frame, "_set_lru_cache"):
        torch._C._dynamo.eval_frame._set_lru_cache(False)

    for name, block in model.layers.items():
        if getattr(block, "moe_enabled", False):
            # MoE layer: compile submodules separately to avoid graph breaks
            # from FSDP(GroupedExperts) hooks which use torch._dynamo.disable
            if isinstance(block, CheckpointWrapper):
                inner_block = block._checkpoint_wrapped_module
            else:
                inner_block = block

            for attr_name, submod in inner_block.named_children():
                assert getattr(block, attr_name) == getattr(inner_block, attr_name)

                if isinstance(submod, moe_module.MoE):
                    # avoid graph breaking on the GroupedExperts' FSDP hooks
                    # by wrapping each submod's forward instead of their __call__
                    for moe_attr, moe_submod in submod.named_children():
                        if moe_attr == "experts":
                            # NOTE: We don't compile token dispatch and token combine due to an issue on B200:
                            # https://github.com/pytorch/torchtitan/issues/1940
                            continue
                        setattr(
                            submod,
                            moe_attr,
                            torch.compile(
                                moe_submod, backend="inductor", fullgraph=True
                            ),
                        )
                elif attr_name in ("attention_norm", "ffn_norm"):
                    # NOTE: attention_norm/ffn_norm may use SequenceParallel
                    # which has issues with torch.compile + Inductor
                    # SequenceParallel has async redistribute which breaks
                    # the graph by introducing async tensors in forward
                    # while the backward expects local tensors.
                    # NOTE: Upgrading PyTorch may resolve this in the future.
                    continue
                else:
                    setattr(
                        inner_block,
                        attr_name,
                        torch.compile(submod, backend="inductor", fullgraph=True),
                    )
        else:
            # If it's not a MoE layer, there is no FSDP(GroupedExperts)
            # So we can compile the whole block
            model.layers[name] = torch.compile(
                block,
                backend="inductor",
                fullgraph=True,
            )

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

        if ep_enabled:
            compiled_fn = grouped_experts._run_experts_grouped_mm

            def _run_experts_grouped_mm_dynamic(
                w1: torch.Tensor,
                w2: torch.Tensor,
                w3: torch.Tensor,
                x: torch.Tensor,
                num_tokens_per_expert: torch.Tensor,
            ) -> torch.Tensor:
                torch._dynamo.mark_dynamic(x, 0)
                return compiled_fn(w1, w2, w3, x, num_tokens_per_expert)

            grouped_experts._run_experts_grouped_mm = _run_experts_grouped_mm_dynamic

    _get_logger().info(
        f"Compiled {len(model.layers)} TransformerBlocks with torch.compile (MoE-aware)"
    )


def apply_moe_ep_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh | None,
    ep_mesh: DeviceMesh | None,
    etp: int = 1,
    ep_tp_mesh: DeviceMesh | None = None,
) -> None:
    """Apply MoE-specific parallelism (Expert Parallelism and MoE TP) to Qwen3 model.

    This handles all MoE-related parallelism:
    1. Input/output tensor conversion for MoE layers (when TP enabled)
    2. Router gate handling (when TP enabled)
    3. Expert parallelism via all-to-all dispatch/combine (when EP enabled)

    Strategy Selection:
        The expert parallelism strategy depends on EP and ETP configuration:

        | EP  | TP  | etp | Strategy              | Expert Weight Sharding         |
        |-----|-----|-----|-----------------------|--------------------------------|
        | 1   | 1   | -   | None                  | Replicate                      |
        | 1   | >1  | -   | TensorParallel        | [Shard(1/2)]                   |
        | >1  | 1   | -   | ExpertParallel        | [Shard(0)]                     |
        | >1  | >1  | 1   | ExpertParallel        | [Shard(0)] (TP borrowed by EP) |
        | >1  | >1  | tp  | ExpertTensorParallel  | [Shard(0), Shard(1/2)]         |

        When EP>1 and TP>1:
        - etp=1: TP dimension is borrowed by EP for token dispatch. Experts use
          only EP sharding [Shard(0)].
        - etp=tp: TP dimension remains independent. Experts use 2D sharding
          [Shard(0), Shard(1/2)] combining EP and TP.

    Args:
        model: The model to apply MoE parallelism to.
        tp_mesh: TP device mesh. If None, skip TP-related MoE handling.
        ep_mesh: EP device mesh. If None, skip expert parallelism.
        etp: Expert Tensor Parallel size (must be 1 or equal to tp).
        ep_tp_mesh: 2D mesh for ExpertTensorParallel (when etp=tp).

    Note:
        This function is a no-op for non-MoE models.
        For models with MoE, at least one of tp_mesh or ep_mesh should be provided.

    Raises:
        ValueError: If num_experts is not divisible by ep_size.
    """
    # Early exit if nothing to do
    if tp_mesh is None and ep_mesh is None:
        return

    # Validate expert count if EP is enabled
    if ep_mesh is not None:
        validate_ep_constraints(model.model_args, ep_mesh.size())

    for transformer_block in model.layers.values():
        if not getattr(transformer_block, "moe_enabled", False):
            continue

        moe = transformer_block.moe
        if moe is None:
            continue

        # Apply TP-related MoE handling (input/output conversion, router gate)
        # This handles DTensor/Tensor conversion for sequence parallelism
        if tp_mesh is not None:
            moe_tp_plan = {
                "moe": PrepareModuleInputOutput(
                    input_layouts=(Shard(1),),
                    desired_input_layouts=(Replicate(),),
                    use_local_input=True,
                    output_layouts=(Partial(),),
                    desired_output_layouts=(Shard(1),),
                ),
                "moe.router.gate": ReplicateParallel(),
            }

            # Apply ReordererSequenceParallel when etp=1 and EP is enabled
            # This ensures each TP rank processes different tokens so the
            # subsequent EP all-to-all doesn't send duplicate data
            if ep_mesh is not None and etp == 1:
                moe_tp_plan["moe.reorderer"] = ReordererSequenceParallel()

            parallelize_module(
                module=transformer_block,
                device_mesh=tp_mesh,
                parallelize_plan=moe_tp_plan,
            )

        experts_mesh, experts_plan = None, None
        if ep_mesh is None:
            experts_mesh = tp_mesh
            experts_plan = TensorParallel()
        elif tp_mesh is None or etp == 1:
            experts_mesh = ep_mesh
            experts_plan = ExpertParallel()
        else:
            experts_mesh = ep_tp_mesh
            experts_plan = ExpertTensorParallel()

        if experts_mesh is not None:
            parallelize_module(
                module=moe.experts,
                device_mesh=experts_mesh,
                parallelize_plan=experts_plan,
            )

    applied = []
    if tp_mesh is not None:
        applied.append(f"MoE TP (tp_size={tp_mesh.size()})")
    if ep_mesh is not None:
        if etp > 1:
            applied.append(f"EP+ETP (ep_size={ep_mesh.size()}, etp={etp})")
        else:
            applied.append(f"EP (ep_size={ep_mesh.size()})")
    if applied:
        _get_logger().info(f"Applied {', '.join(applied)} to MoE layers")


__all__ = [
    "parallelize_qwen3",
    "apply_non_moe_tp",
    "apply_moe_ep_tp",
    "apply_fsdp",
    "apply_cp",
]
