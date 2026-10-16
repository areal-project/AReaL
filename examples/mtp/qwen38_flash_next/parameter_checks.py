# SPDX-License-Identifier: Apache-2.0
"""Independent parameter ownership checks for native MTP-only acceptance."""

from __future__ import annotations

from typing import Any

import torch


def classify_parameters(
    models: list[torch.nn.Module],
) -> dict[int, tuple[str, torch.nn.Parameter, bool]]:
    """A shared parameter is MTP-only only if every registration belongs to MTP."""
    result = {}
    for chunk, model in enumerate(models):
        for name, parameter in model.named_parameters(remove_duplicate=False):
            exclusive_mtp = "mtp" in name.split(".")[:-1]
            previous = result.get(id(parameter))
            if previous is not None:
                exclusive_mtp = exclusive_mtp and previous[2]
                name = previous[0]
            else:
                name = f"chunk_{chunk}.{name}"
            result[id(parameter)] = (name, parameter, exclusive_mtp)
    return result


def optimizer_model_parameter_ids(optimizer: Any) -> set[int]:
    """Use MCore's original model references, never its FP32 master shards."""
    if hasattr(optimizer, "chained_optimizers"):
        return set().union(
            *(
                optimizer_model_parameter_ids(opt)
                for opt in optimizer.chained_optimizers
            )
        )
    # DistributedOptimizer's original full model references (the recipe is DP1).
    if hasattr(optimizer, "model_float16_groups"):
        groups = list(optimizer.model_float16_groups) + list(
            optimizer.model_fp32_groups
        )
    elif hasattr(optimizer, "float16_groups"):
        groups = list(optimizer.float16_groups) + list(optimizer.fp32_from_fp32_groups)
    else:
        groups = [group["params"] for group in optimizer.param_groups]
    return {id(parameter) for group in groups for parameter in group}


def verify_mtp_optimizer_membership(
    models: list[torch.nn.Module], optimizer: Any
) -> dict[str, int]:
    """Reject accidental backbone unfreezing, missing MTP, or optimizer extras."""
    classified = classify_parameters(models)
    for name, parameter, exclusive_mtp in classified.values():
        if parameter.requires_grad != exclusive_mtp:
            raise AssertionError(
                f"Parameter trainability disagrees with MTP ownership: {name}"
            )
    expected = {identity for identity, (_, _, is_mtp) in classified.items() if is_mtp}
    actual = optimizer_model_parameter_ids(optimizer)
    if actual != expected:
        raise AssertionError(
            f"Optimizer membership differs: missing={len(expected - actual)}, extra={len(actual - expected)}"
        )
    return {
        "frozen_parameters": len(classified) - len(expected),
        "mtp_parameters": len(expected),
        "optimizer_model_parameters": len(actual),
    }
