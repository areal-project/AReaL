# SPDX-License-Identifier: Apache-2.0
"""Preserve Qwen4Exp inference-only visual parameters across weight residency.

The language-only actor does not send these parameters. Native SGLang preserves
buffers but discards parameter contents when weights CPU backup is disabled.
Caller integration must validate the language-only actor contract before use.
"""

from types import ModuleType
from typing import Any

import torch
from torch import nn


def _visual_parameters(model: nn.Module) -> dict[str, nn.Parameter]:
    if type(model).__name__ != "Qwen4ExpForConditionalGeneration":
        raise TypeError("Frozen visual state requires a Qwen4Exp inference model")
    parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if name.startswith("visual.")
    }
    if not parameters:
        raise ValueError("Qwen4Exp inference model has no visual parameters")
    return parameters


@torch.no_grad()
def snapshot_visual_parameters(model: nn.Module) -> dict[str, torch.Tensor]:
    """Copy only visual parameters to independent CPU storage before release."""
    return {
        name: parameter.detach().to(device="cpu", copy=True)
        for name, parameter in _visual_parameters(model).items()
    }


@torch.no_grad()
def restore_visual_parameters(model: nn.Module, saved: dict[str, torch.Tensor]) -> None:
    """Restore into the same parameter objects after native weights resume."""
    targets = _visual_parameters(model)
    if targets.keys() != saved.keys():
        raise ValueError("Frozen visual parameter names changed across resume")
    # Validate the entire state before writing any destination.
    for name, target in targets.items():
        source = saved[name]
        if source.device.type != "cpu":
            raise ValueError(f"Frozen visual backup must be on CPU: {name}")
        if source.shape != target.shape or source.dtype != target.dtype:
            raise ValueError(f"Frozen visual parameter shape or dtype changed: {name}")
    for name, target in targets.items():
        target.copy_(saved[name])


def install_static_state_hooks(weight_updater: ModuleType) -> None:
    """Extend native SGLang static-state hooks for Qwen4Exp visual parameters.

    Call inside each inference worker before its first weights release. Native
    buffer handling stays in the original hooks. This does not authorize any
    parameter exclusion from the AWEX transfer contract.
    """
    if getattr(weight_updater, "_areal_qwen4_exp_static_hooks", False):
        return
    original_export = weight_updater._export_static_state
    original_import = weight_updater._import_static_state
    key = "_areal_qwen4_exp_visual_parameters"

    def export_state(model: nn.Module) -> dict[str, Any]:
        state = original_export(model)
        if type(model).__name__ == "Qwen4ExpForConditionalGeneration":
            if key in state:
                raise ValueError("Duplicate Qwen4Exp frozen visual state")
            state[key] = snapshot_visual_parameters(model)
        return state

    def import_state(model: nn.Module, state: dict[str, Any]) -> None:
        is_qwen4_exp = type(model).__name__ == "Qwen4ExpForConditionalGeneration"
        if is_qwen4_exp and key not in state:
            raise ValueError("Qwen4Exp visual state was not saved before release")
        original_import(model, state)
        if is_qwen4_exp:
            restore_visual_parameters(model, state[key])

    weight_updater._export_static_state = export_state
    weight_updater._import_static_state = import_state
    weight_updater._areal_qwen4_exp_static_hooks = True
