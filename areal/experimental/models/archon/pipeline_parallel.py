# SPDX-License-Identifier: Apache-2.0

# Adapted from torchtitan: torchtitan/distributed/pipeline_parallel.py

import copy
import functools
import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import (
    PipelineScheduleMulti,
    PipelineScheduleSingle,
    ScheduleDualPipeV,
    ScheduleZBVZeroBubble,
    get_schedule_class,
)

from areal.utils import logging

if TYPE_CHECKING:
    from torch.distributed.pipelining.schedules import _PipelineSchedule

    from areal.api.cli_args import ArchonEngineConfig
    from areal.experimental.models.archon import ArchonParallelDims


@functools.cache
def _get_logger() -> logging.Logger:
    """Get rank-aware logger for this module."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    return logging.getLogger(f"[Archon PipelineParallel Rank {rank}]")


__all__ = [
    "generate_llm_fqn_per_model_part",
    "pipeline_module_split",
    "pipeline_llm",
    "build_pipeline_schedule",
]


def build_pipeline_schedule(
    stages: list[PipelineStage],
    pp_schedule: str,
    n_microbatches: int,
    pp_degree: int = 1,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> "_PipelineSchedule":
    """Build pipeline schedule, passing single stage or list based on schedule type.

    Args:
        stages: Pipeline stages for this rank (list, even for single-stage schedules)
        pp_schedule: Schedule name (e.g., "1F1B", "Interleaved1F1B")
        n_microbatches: Number of microbatches
        pp_degree: Pipeline parallel degree (for bubble warning)
        loss_fn: Loss function (None for eval mode)

    Returns:
        Configured pipeline schedule instance
    """
    schedule_class = get_schedule_class(pp_schedule)
    looped_schedule = issubclass(schedule_class, PipelineScheduleMulti)

    num_total_stages = len(stages) * pp_degree
    if n_microbatches < num_total_stages:
        _get_logger().warning(
            f"n_microbatches ({n_microbatches}) < num_total_stages ({num_total_stages}), "
            "may result in pipeline bubble"
        )

    return schedule_class(
        stages if looped_schedule else stages[0],
        n_microbatches=n_microbatches,
        loss_fn=loss_fn,
        scale_grads=False,
    )


def generate_llm_fqn_per_model_part(
    num_stages: int,
    num_layers: int,
    first_stage_less_layers: int = 1,
    last_stage_less_layers: int = 1,
    is_critic: bool = False,
) -> list[list[str]]:
    """Generate module FQN lists for each pipeline stage.

    This function distributes transformer layers across pipeline stages,
    accounting for the computational cost of embedding and output layers.

    Args:
        num_stages: Number of pipeline stages (must equal pp_degree for 1F1B)
        num_layers: Number of transformer layers in the model
        first_stage_less_layers: Weight for input modules (tok_embeddings), default 1
        last_stage_less_layers: Weight for output modules (norm + output/score), default 1
        is_critic: Whether the model is a critic (uses 'score' instead of 'output')

    Returns:
        List of module name lists, one per stage.

    Example:
        >>> generate_llm_fqn_per_model_part(2, 4)
        [['tok_embeddings', 'layers.0', 'layers.1'],
         ['layers.2', 'layers.3', 'norm', 'output']]

        >>> generate_llm_fqn_per_model_part(4, 8, is_critic=True)
        [['tok_embeddings', 'layers.0', 'layers.1'],
         ['layers.2', 'layers.3', 'layers.4'],
         ['layers.5', 'layers.6'],
         ['layers.7', 'norm', 'score']]
    """
    # Determine output module name based on model type
    output_module = "score" if is_critic else "output"

    # Validation
    if num_stages < 1:
        raise ValueError(f"num_stages must be >= 1, got {num_stages}")

    # Single stage: return everything
    if num_stages == 1:
        layer_names = [f"layers.{i}" for i in range(num_layers)]
        return [["tok_embeddings"] + layer_names + ["norm", output_module]]

    # Calculate effective layers including embedding/output overhead
    num_effective_layers = num_layers + first_stage_less_layers + last_stage_less_layers

    if num_stages > num_effective_layers:
        raise ValueError(
            f"num_stages ({num_stages}) cannot exceed effective layers "
            f"({num_effective_layers} = {num_layers} + {first_stage_less_layers} + {last_stage_less_layers})"
        )

    layers_per_stage = num_effective_layers // num_stages
    extra_layers = num_effective_layers % num_stages

    if layers_per_stage == 0:
        raise ValueError(
            f"layers_per_stage is 0 with {num_effective_layers} effective layers "
            f"and {num_stages} stages"
        )

    if first_stage_less_layers > layers_per_stage:
        raise ValueError(
            f"first_stage_less_layers ({first_stage_less_layers}) exceeds layers_per_stage ({layers_per_stage})"
        )
    if last_stage_less_layers > layers_per_stage:
        raise ValueError(
            f"last_stage_less_layers ({last_stage_less_layers}) exceeds layers_per_stage ({layers_per_stage})"
        )

    module_names_per_stage: list[list[str]] = []
    current_layer = 0

    for stage_idx in range(num_stages):
        stage_modules: list[str] = []

        # Calculate effective layers for this stage (extra layers go to earlier stages)
        effective_layers_for_stage = layers_per_stage
        if stage_idx < extra_layers:
            effective_layers_for_stage += 1

        if stage_idx == 0:
            # First stage: tok_embeddings + transformer layers
            stage_modules.append("tok_embeddings")
            num_transformer_layers = (
                effective_layers_for_stage - first_stage_less_layers
            )
            for _ in range(num_transformer_layers):
                if current_layer < num_layers:
                    stage_modules.append(f"layers.{current_layer}")
                    current_layer += 1

        elif stage_idx == num_stages - 1:
            # Last stage: transformer layers + norm + output/score
            num_transformer_layers = effective_layers_for_stage - last_stage_less_layers
            for _ in range(num_transformer_layers):
                if current_layer < num_layers:
                    stage_modules.append(f"layers.{current_layer}")
                    current_layer += 1
            stage_modules.extend(["norm", output_module])

        else:
            # Middle stages: only transformer layers
            for _ in range(effective_layers_for_stage):
                if current_layer < num_layers:
                    stage_modules.append(f"layers.{current_layer}")
                    current_layer += 1

        module_names_per_stage.append(stage_modules)

    return module_names_per_stage


def pipeline_module_split(
    whole_model: nn.Module,
    pp_mesh: DeviceMesh,
    pp_schedule: str,
    device: torch.device,
    module_names_per_stage: list[list[str]],
) -> tuple[list[PipelineStage], list[nn.Module]]:
    """Split model into pipeline stages based on module names.

    Key points:
    - Archon uses ModuleDict (keys are "0", "1", ...) for layers
    - Modules not in this stage are set to None
    - Model's forward() must handle None modules gracefully

    Args:
        whole_model: The complete model to split
        pp_mesh: Pipeline parallel device mesh
        pp_schedule: Schedule type ("1F1B", "Interleaved1F1B", "InterleavedZeroBubble", or "ZBVZeroBubble")
        device: Target device for stages
        module_names_per_stage: Module FQNs for each stage

    Returns:
        Tuple of (list of PipelineStage, list of model parts)
    """
    pp_rank = pp_mesh.get_local_rank()
    pp_degree = pp_mesh.size()
    num_stages = len(module_names_per_stage)

    def _build_stage_from_modules(
        stage_idx: int,
        module_names: list[str],
        num_stages: int,
    ) -> tuple[PipelineStage, nn.Module]:
        """Build a single pipeline stage from module names."""
        assert next(whole_model.parameters()).device.type == "meta", (
            "Model must be on meta device for pipeline splitting"
        )
        # Deep copy to create independent model part
        model = copy.deepcopy(whole_model)
        modules_to_keep = set(module_names)

        for module_name, module_value in list(model.named_children()):
            if isinstance(module_value, nn.ModuleDict):
                # Handle layers (ModuleDict in Archon, keys are "0", "1", ...)
                # Extract layer indices to keep: "layers.0" -> "0"
                layers_to_keep = {
                    name.split(".", 1)[1]
                    for name in modules_to_keep
                    if name.startswith(f"{module_name}.")
                }

                if layers_to_keep:
                    # Delete unwanted layers
                    for layer_key in list(module_value.keys()):
                        if layer_key not in layers_to_keep:
                            del module_value[layer_key]
                else:
                    # No layers to keep in this stage, set to empty ModuleDict
                    setattr(model, module_name, nn.ModuleDict())

            elif isinstance(module_value, nn.ModuleList):
                # Handle ModuleList (if used)
                layers_to_keep = {
                    name.split(".", 1)[1]
                    for name in modules_to_keep
                    if name.startswith(f"{module_name}.")
                }

                if layers_to_keep:
                    indices_to_keep = {
                        int(idx) for idx in layers_to_keep if idx.isdigit()
                    }
                    new_layers = nn.ModuleList(
                        [
                            layer
                            for i, layer in enumerate(module_value)
                            if i in indices_to_keep
                        ]
                    )
                    setattr(model, module_name, new_layers)
                else:
                    setattr(model, module_name, nn.ModuleList())

            elif module_name not in modules_to_keep:
                # Simple module not in this stage, set to None
                setattr(model, module_name, None)

        # Create PipelineStage
        stage = PipelineStage(
            model,
            stage_idx,
            num_stages,
            device,
            group=pp_mesh.get_group(),
        )

        return stage, model

    def _get_stage_indices() -> tuple[int, ...]:
        """Get stage indices for this rank based on schedule style.

        Examples (pp_degree=4, num_stages=8):
            1F1B:                  Rank 0->(0,), Rank 1->(1,), ...
            Interleaved1F1B:       Rank 0->(0,4), Rank 1->(1,5), Rank 2->(2,6), Rank 3->(3,7)
            InterleavedZeroBubble: (same loop-style assignment as Interleaved1F1B)
            ZBVZeroBubble:         Rank 0->(0,7), Rank 1->(1,6), Rank 2->(2,5), Rank 3->(3,4)
        """
        if num_stages % pp_degree != 0:
            raise ValueError(
                f"num_stages ({num_stages}) must be evenly divisible by "
                f"pp_degree ({pp_degree})"
            )
        stages_per_rank = num_stages // pp_degree

        schedule_class = get_schedule_class(pp_schedule)
        v_style_schedules = (ScheduleZBVZeroBubble, ScheduleDualPipeV)
        style = "v" if schedule_class in v_style_schedules else "loop"

        if style == "v":
            if stages_per_rank != 2:
                raise ValueError(
                    f"V-style schedules require exactly 2 stages per rank, "
                    f"got {stages_per_rank}"
                )
            stage_v_pairs = list(
                zip(range(pp_degree), range(num_stages - 1, pp_degree - 1, -1))
            )
            return stage_v_pairs[pp_rank]
        else:
            return tuple(pp_rank + s * pp_degree for s in range(stages_per_rank))

    stages: list[PipelineStage] = []
    model_parts: list[nn.Module] = []

    for stage_idx in _get_stage_indices():
        stage, model_part = _build_stage_from_modules(
            stage_idx, module_names_per_stage[stage_idx], num_stages
        )
        stages.append(stage)
        model_parts.append(model_part)

        _get_logger().info(
            f"Built stage {stage_idx} (pp_rank={pp_rank}) "
            f"with modules: {module_names_per_stage[stage_idx]}"
        )

    return stages, model_parts


def pipeline_llm(
    model: nn.Module,
    device: torch.device,
    parallel_dims: "ArchonParallelDims",
    archon_config: "ArchonEngineConfig",
    parallelize_fn: Callable,
    **parallelize_kwargs,
) -> tuple[list[PipelineStage], list[nn.Module], bool, bool]:
    """Main entry point for pipeline parallelism.

    Workflow:
    1. Generate module names for each virtual stage
    2. Split model into stages (multiple per rank for Interleaved1F1B/InterleavedZeroBubble/ZBVZeroBubble)
    3. Apply parallelization (TP, FSDP) to each model part

    Args:
        model: The complete model to pipeline
        device: Target device
        parallel_dims: ArchonParallelDims with PP configuration
        archon_config: ArchonEngineConfig containing PP settings
        parallelize_fn: Function to apply TP/FSDP to model parts
        **parallelize_kwargs: Additional arguments for parallelize_fn

    Returns:
        Tuple of:
        - stages: List of PipelineStage (1 for 1F1B, 2+ for Interleaved1F1B/InterleavedZeroBubble/ZBVZeroBubble)
        - model_parts: List of model parts
        - has_first_stage: Whether this rank has the first stage
        - has_last_stage: Whether this rank has the last stage
    """
    pp_mesh = parallel_dims.get_mesh("pp")
    if pp_mesh is None:
        raise RuntimeError("PP mesh not found. Ensure pp > 1 in parallel_dims")

    pp_schedule = archon_config.pp_schedule
    layers_per_stage = archon_config.pp_layers_per_stage
    first_stage_less_layers = archon_config.pp_first_stage_less_layers
    last_stage_less_layers = archon_config.pp_last_stage_less_layers

    # Get number of layers from model
    # Archon models may use n_layers or num_hidden_layers
    if hasattr(model, "model_args") and hasattr(model.model_args, "num_hidden_layers"):
        num_layers = model.model_args.num_hidden_layers
    elif hasattr(model, "model_args") and hasattr(model.model_args, "n_layers"):
        num_layers = model.model_args.n_layers
    elif hasattr(model, "config") and hasattr(model.config, "num_hidden_layers"):
        num_layers = model.config.num_hidden_layers
    elif hasattr(model, "config") and hasattr(model.config, "n_layers"):
        num_layers = model.config.n_layers
    else:
        raise RuntimeError(
            "Cannot determine num_layers from model. "
            "Model must have model_args.num_hidden_layers/n_layers or config.num_hidden_layers/n_layers"
        )

    pp_degree = parallel_dims.pp

    # Detect if model is a critic (uses 'score' instead of 'output')
    is_critic = getattr(getattr(model, "model_args", None), "is_critic", False)

    schedule_class = get_schedule_class(pp_schedule)
    is_single_stage_schedule = issubclass(schedule_class, PipelineScheduleSingle)

    if layers_per_stage is not None:
        # Calculate number of virtual stages from layers_per_stage
        num_virtual_stages = math.ceil(
            (num_layers + first_stage_less_layers + last_stage_less_layers)
            / layers_per_stage
        )
        # Validate divisibility
        if num_virtual_stages % pp_degree != 0:
            raise ValueError(
                f"num_virtual_stages ({num_virtual_stages}) must be divisible by "
                f"pp_degree ({pp_degree}). Calculated from: "
                f"ceil(({num_layers} + {first_stage_less_layers} + {last_stage_less_layers}) / {layers_per_stage})"
            )
        stages_per_rank = num_virtual_stages // pp_degree
        # Validate schedule compatibility
        if is_single_stage_schedule and stages_per_rank != 1:
            raise ValueError(
                f"{pp_schedule} schedule requires exactly 1 stage per rank, "
                f"but got {stages_per_rank} (from layers_per_stage={layers_per_stage}). "
                f"Use Interleaved1F1B for multiple stages per rank."
            )
        if not is_single_stage_schedule and stages_per_rank < 2:
            raise ValueError(
                f"{pp_schedule} schedule requires >= 2 stages per rank, "
                f"but got {stages_per_rank} (from layers_per_stage={layers_per_stage}). "
                f"Use 1F1B schedule for single stage per rank."
            )
        # V-style schedules require exactly 2 stages per rank
        v_style_schedules = (ScheduleZBVZeroBubble, ScheduleDualPipeV)
        if schedule_class in v_style_schedules and stages_per_rank != 2:
            raise ValueError(
                f"{pp_schedule} requires exactly 2 stages per rank, "
                f"but got {stages_per_rank}. "
                f"Set pp_layers_per_stage to achieve 2 stages per rank, "
                f"or let it default (None)."
            )
    else:
        # Default: 1 for single-stage schedules, 2 for multi-stage schedules
        stages_per_rank = 1 if is_single_stage_schedule else 2
        num_virtual_stages = pp_degree * stages_per_rank

    _get_logger().info(
        f"PP setup: schedule={pp_schedule}, stages_per_rank={stages_per_rank}, "
        f"num_virtual_stages={num_virtual_stages}"
    )

    # 1. Generate module names per stage for ALL virtual stages
    module_names_per_stage = generate_llm_fqn_per_model_part(
        num_stages=num_virtual_stages,
        num_layers=num_layers,
        first_stage_less_layers=first_stage_less_layers,
        last_stage_less_layers=last_stage_less_layers,
        is_critic=is_critic,
    )

    _get_logger().info(f"PP module distribution: {module_names_per_stage}")

    # 2. Split model into stages
    stages, model_parts = pipeline_module_split(
        whole_model=model,
        pp_mesh=pp_mesh,
        pp_schedule=pp_schedule,
        device=device,
        module_names_per_stage=module_names_per_stage,
    )

    # 3. Apply parallelization to each model part
    for i, m in enumerate(model_parts):
        m = parallelize_fn(m, parallel_dims, **parallelize_kwargs)
        model_parts[i] = m
        # Update stage's reference to the parallelized model
        stages[i].submod = m

    # Determine first/last stage status
    has_first_stage = any(s.is_first for s in stages)
    has_last_stage = any(s.is_last for s in stages)

    _get_logger().info(
        f"Pipeline setup complete: has_first_stage={has_first_stage}, "
        f"has_last_stage={has_last_stage}"
    )

    return stages, model_parts, has_first_stage, has_last_stage
