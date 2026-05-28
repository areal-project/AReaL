# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ModelArgsProtocol(Protocol):
    """Protocol for model args that have head configuration."""

    n_heads: int
    n_kv_heads: int | None


def validate_tp_constraints(
    model_args: ModelArgsProtocol,
    tp_size: int,
) -> None:
    """Validate tensor parallelism constraints for attention heads.

    This validates that the model's attention head configuration is compatible
    with the requested tensor parallelism configuration.

    Args:
        model_args: Model arguments containing n_heads and n_kv_heads.
        tp_size: Tensor parallelism size.

    Raises:
        ValueError: If head counts don't satisfy TP constraints.
    """
    n_heads = model_args.n_heads
    n_kv_heads = model_args.n_kv_heads if model_args.n_kv_heads is not None else n_heads

    if n_heads % tp_size != 0:
        raise ValueError(
            f"n_heads ({n_heads}) must be divisible by tp_size ({tp_size})"
        )

    if n_kv_heads % tp_size != 0:
        raise ValueError(
            f"n_kv_heads ({n_kv_heads}) must be divisible by tp_size ({tp_size})"
        )


def validate_cp_constraints(
    model_args: ModelArgsProtocol,
    cp_size: int,
    tp_size: int = 1,
) -> None:
    """Validate context parallelism constraints for attention heads.

    This validates that the model's attention head configuration is compatible
    with the requested context parallelism (Ulysses SP) configuration.

    Args:
        model_args: Model arguments containing n_heads and n_kv_heads.
        cp_size: Context parallelism size (world size of CP group).
        tp_size: Tensor parallelism size, used to compute local head counts.

    Raises:
        ValueError: If head counts don't satisfy CP constraints.
    """
    n_heads = model_args.n_heads
    n_kv_heads = model_args.n_kv_heads if model_args.n_kv_heads is not None else n_heads
    n_rep = n_heads // n_kv_heads

    q_heads = n_heads // tp_size
    kv_heads = n_kv_heads // tp_size

    # Constraint 1: q_heads must be divisible by cp_size (no repeat allowed for q)
    if q_heads % cp_size != 0:
        raise ValueError(
            f"q_heads after TP ({q_heads}) must be divisible by cp_size ({cp_size})"
        )

    # Constraint 2: kv_heads must be divisible by or divide cp_size,
    # and if repeat is needed, repeat count must be a factor of n_rep
    if kv_heads >= cp_size:
        if kv_heads % cp_size != 0:
            raise ValueError(
                f"kv_heads after TP ({kv_heads}) must be divisible by cp_size ({cp_size})"
            )
    else:
        if cp_size % kv_heads != 0:
            raise ValueError(
                f"cp_size ({cp_size}) must be divisible by kv_heads after TP ({kv_heads})"
            )
        repeats = cp_size // kv_heads
        if n_rep % repeats != 0:
            raise ValueError(
                f"repeat count ({repeats}) must be a factor of n_rep ({n_rep})"
            )


@runtime_checkable
class MoEModelArgsProtocol(Protocol):
    """Protocol for model args that have MoE configuration."""

    moe_enabled: bool
    moe_args: object | None  # MoEArgs, but use object to avoid circular import


def validate_ep_constraints(
    model_args: MoEModelArgsProtocol,
    ep_size: int,
) -> None:
    """Validate expert parallelism constraints for MoE models.

    This validates that the model's MoE configuration is compatible
    with the requested expert parallelism configuration.

    Args:
        model_args: Model arguments containing moe_enabled and moe_args.
        ep_size: Expert parallelism size.

    Raises:
        ValueError: If MoE configuration doesn't satisfy EP constraints.
    """
    if ep_size <= 1:
        return

    if not model_args.moe_enabled:
        raise ValueError(
            f"Expert parallelism (ep_size={ep_size}) requires MoE to be enabled, "
            f"but moe_enabled={model_args.moe_enabled}"
        )

    if model_args.moe_args is None:
        raise ValueError(
            f"Expert parallelism (ep_size={ep_size}) requires moe_args to be set, "
            f"but moe_args is None"
        )

    num_experts = model_args.moe_args.num_experts
    if num_experts % ep_size != 0:
        raise ValueError(
            f"num_experts ({num_experts}) must be divisible by ep_size ({ep_size})"
        )


def is_moe_model_config(model_config: object) -> bool:
    """Check if a HuggingFace PretrainedConfig represents a Mixture-of-Experts model.

    Inspects common HF config attributes (num_experts, num_local_experts)
    to determine whether the model uses MoE layers.

    Args:
        model_config: A HuggingFace PretrainedConfig (or any object with
            num_experts / num_local_experts attributes).

    Returns:
        True if the config indicates an MoE model with more than one expert.
    """
    num_experts = getattr(model_config, "num_experts", None)
    if num_experts is None:
        num_experts = getattr(model_config, "num_local_experts", None)
    return num_experts is not None and num_experts > 1


__all__ = [
    "ModelArgsProtocol",
    "MoEModelArgsProtocol",
    "is_moe_model_config",
    "validate_cp_constraints",
    "validate_tp_constraints",
    "validate_ep_constraints",
]
