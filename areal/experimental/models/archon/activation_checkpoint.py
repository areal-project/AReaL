# SPDX-License-Identifier: Apache-2.0

# Adapted from torchtitan: torchtitan/distributed/activation_checkpoint.py

import functools
import os
from collections import defaultdict
from dataclasses import dataclass, field

import torch
import torch._functorch.config
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.utils.checkpoint import (
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)

from areal.utils import logging


@functools.cache
def _get_logger() -> logging.Logger:
    """Get rank-aware logger for this module."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    return logging.getLogger(f"[Archon ActivationCheckpoint Rank {rank}]")


# Global counter for layer-level selective AC
_layer_sac_count = 0


@dataclass
class ActivationCheckpointConfig:
    """Activation checkpointing configuration."""

    mode: str = "selective"
    """Type of activation checkpointing to use: 'selective', 'full', 'memory_budget', 'none'"""

    selective_ac_option: str = "op"
    """Selective AC options: 'op' for op-level, or integer string for every Nth layer."""

    per_op_sac_force_recompute_mm_shapes_by_fqns: list[str] = field(
        default_factory=lambda: ["moe.router.gate"]
    )
    """FQNs of nn.Linear modules whose mm shapes should be force recomputed."""

    early_stop: bool = False
    """Stop recomputing early when all activations are rematerialized."""

    memory_budget: float = 0.5
    """For 'memory_budget' mode: 0.0 = min memory, 1.0 = default behavior."""

    visualize_memory_budget_pareto: bool = False
    """Dump SVG visualization of runtime vs memory tradeoffs."""

    preserve_rng_state: bool = False
    """Preserve RNG state for deterministic output. May be slower."""

    determinism_check: str = "default"
    """Determinism check function: 'default', 'none', 'throw'."""

    debug: bool = False
    """Capture AC debug information. Will be slower."""

    def __post_init__(self):
        valid_modes = ("none", "full", "selective", "memory_budget")
        if self.mode not in valid_modes:
            raise ValueError(
                f"Invalid AC mode: {self.mode!r}. Valid modes: {valid_modes}"
            )

        if self.mode == "selective":
            if (
                self.selective_ac_option != "op"
                and not self.selective_ac_option.isdigit()
            ):
                raise ValueError(
                    f"Invalid selective_ac_option: {self.selective_ac_option!r}. "
                    "Must be 'op' or a positive integer string (e.g., '1', '2')."
                )


def _apply_layer_sac(
    module: nn.Module,
    ac_config: ActivationCheckpointConfig,
) -> nn.Module:
    """Apply layer selective activation checkpointing to the module.

    Args:
        module: The module to apply layer selective activation checkpointing to.
        ac_config: The activation checkpointing config.

    Returns:
        The module with layer selective activation checkpointing applied.
    """
    global _layer_sac_count
    _layer_sac_count += 1
    ac_freq = int(ac_config.selective_ac_option)
    if not ac_freq or _layer_sac_count % ac_freq == 0:
        return ptd_checkpoint_wrapper(
            module,
            preserve_rng_state=ac_config.preserve_rng_state,
            determinism_check=ac_config.determinism_check,
            early_stop=ac_config.early_stop,
            debug=ac_config.debug,
        )
    else:
        return module


def _apply_op_sac(
    module: nn.Module,
    ac_config: ActivationCheckpointConfig,
    *,
    base_fqn: str | None = None,
    op_sac_save_list: set[torch._ops.OpOverload],
) -> nn.Module:
    """Apply selective activation checkpointing to the module.

    Args:
        module: The module to apply selective activation checkpointing to.
        ac_config: The activation checkpointing config.
        base_fqn: The base fully qualified name of the module.
        op_sac_save_list: The list of ops to save instead of recomputing.

    Returns:
        The module with selective activation checkpointing applied.
    """
    mm_recompute_shapes = set()
    if len(ac_config.per_op_sac_force_recompute_mm_shapes_by_fqns) > 0:
        for module_fqn, submod in module.named_modules():
            fqn = module_fqn
            if base_fqn is not None:
                fqn = f"{base_fqn}.{module_fqn}"
            if not any(
                filter_fqn in fqn
                for filter_fqn in ac_config.per_op_sac_force_recompute_mm_shapes_by_fqns
            ):
                continue
            if not isinstance(submod, nn.Linear):
                raise ValueError(
                    "per_op_sac_force_recompute_mm_shapes_by_fqns expected to match "
                    f"a nn.Linear, but got: {submod}"
                )
            out_f, in_f = submod.weight.shape
            mm_recompute_shapes.add((in_f, out_f))
        if mm_recompute_shapes:
            _get_logger().debug(
                f"Selective op AC force recomputing mms with rhs shapes {mm_recompute_shapes}"
            )

    def _get_custom_policy(meta):
        def _custom_policy(ctx, func, *args, **kwargs):
            if (
                func == torch.ops.aten._to_copy.default
                and "cuda" in str(args[0].device)
                and "device" in kwargs
                and str(kwargs["device"]) == "cpu"
            ):
                return CheckpointPolicy.MUST_SAVE

            mode = "recompute" if ctx.is_recompute else "forward"
            mm_count_key = f"{mode}_mm_count"
            if func == torch.ops.aten.mm.default:
                if args[1].shape in mm_recompute_shapes:
                    return CheckpointPolicy.PREFER_RECOMPUTE
                meta[mm_count_key] += 1

            # Saves output of all compute ops, except every second mm
            to_save = func in op_sac_save_list and not (
                func == torch.ops.aten.mm.default and meta[mm_count_key] % 2 == 0
            )
            return (
                CheckpointPolicy.MUST_SAVE
                if to_save
                else CheckpointPolicy.PREFER_RECOMPUTE
            )

        return _custom_policy

    def selective_checkpointing_context_fn():
        meta = defaultdict(int)
        return create_selective_checkpoint_contexts(_get_custom_policy(meta))

    return ptd_checkpoint_wrapper(
        module,
        context_fn=selective_checkpointing_context_fn,
        preserve_rng_state=ac_config.preserve_rng_state,
        determinism_check=ac_config.determinism_check,
        early_stop=ac_config.early_stop,
        debug=ac_config.debug,
    )


def _apply_full_ac(
    module: nn.Module,
    ac_config: ActivationCheckpointConfig,
) -> nn.Module:
    """Apply full activation checkpointing to the module.

    Args:
        module: The module to apply full activation checkpointing to.
        ac_config: The activation checkpointing config.

    Returns:
        The module with full activation checkpointing applied.
    """
    return ptd_checkpoint_wrapper(
        module,
        preserve_rng_state=ac_config.preserve_rng_state,
        determinism_check=ac_config.determinism_check,
        early_stop=ac_config.early_stop,
        debug=ac_config.debug,
    )


def _apply_ac_to_transformer_block(
    module: nn.Module,
    ac_config: ActivationCheckpointConfig,
    *,
    base_fqn: str | None = None,
    model_compile_enabled: bool = False,
    op_sac_save_list: set[torch._ops.OpOverload] | None = None,
) -> nn.Module:
    """Apply AC to a single transformer block."""
    valid_ac_modes = ("full", "selective")
    if ac_config.mode not in valid_ac_modes:
        raise ValueError(
            f"Invalid AC mode: {ac_config.mode}. Valid modes: {valid_ac_modes}"
        )

    if ac_config.mode == "full":
        return _apply_full_ac(module, ac_config)

    # selective mode: op-level or layer-level (validated in __post_init__)
    if ac_config.selective_ac_option == "op":
        op_sac_save_list = op_sac_save_list or set()
        return _apply_op_sac(
            module, ac_config, base_fqn=base_fqn, op_sac_save_list=op_sac_save_list
        )

    return _apply_layer_sac(module, ac_config)


def apply_ac(
    model: nn.Module,
    ac_config: ActivationCheckpointConfig,
    *,
    model_compile_enabled: bool = False,
    op_sac_save_list: set[torch._ops.OpOverload] | None = None,
    base_folder: str = "",
) -> None:
    """Apply activation checkpointing to the model.

    Args:
        model: The model to apply activation checkpointing to.
        ac_config: The activation checkpointing config.
        model_compile_enabled: Whether torch.compile is enabled for the model.
        op_sac_save_list: The list of ops to save instead of recomputing.
            This is model-specific and should be passed from parallelize.py.
        base_folder: Base folder for memory_budget pareto visualization.

    Returns:
        None
    """
    if ac_config.mode == "memory_budget":
        assert model_compile_enabled, "Memory budget mode requires model to be compiled"
        if ac_config.visualize_memory_budget_pareto:
            pareto_dir = os.path.join(base_folder, "memory_budget_pareto")
            if not os.path.exists(pareto_dir):
                os.makedirs(pareto_dir, exist_ok=True)
            torch._functorch.config.memory_budget_pareto_dir = pareto_dir
            torch._functorch.config.visualize_memory_budget_pareto = True

        torch._functorch.config.activation_memory_budget = ac_config.memory_budget
        _get_logger().info(f"Selected {ac_config.memory_budget} budget option")
        return

    if ac_config.mode == "none":
        _get_logger().debug("Activation checkpointing is disabled")
        return

    if not hasattr(model, "layers"):
        raise ValueError(
            "Model must have a 'layers' attribute (ModuleDict) to apply AC"
        )

    global _layer_sac_count
    _layer_sac_count = 0

    for layer_id, transformer_block in model.layers.named_children():
        transformer_block = _apply_ac_to_transformer_block(
            transformer_block,
            ac_config,
            base_fqn=f"layers.{layer_id}",
            model_compile_enabled=model_compile_enabled,
            op_sac_save_list=op_sac_save_list,
        )
        model.layers.register_module(layer_id, transformer_block)

    _get_logger().info(
        f"Applied {ac_config.mode} activation checkpointing to the model"
    )


__all__ = [
    "ActivationCheckpointConfig",
    "apply_ac",
]
