# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch


def _enable_mtp_input_grad(
    module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
    """Give reentrant MTP checkpoints a gradient-bearing input activation."""
    if not module.training or not torch.is_grad_enabled():
        return None
    # MCore MTP blocks take (input_ids, position_ids, hidden_states, ...).
    hidden_states = kwargs.get("hidden_states", args[2] if len(args) > 2 else None)
    if not isinstance(hidden_states, torch.Tensor) or hidden_states.requires_grad:
        return None
    # Use a separate leaf so the frozen backbone and its main-loss activation
    # remain untouched. Parameters inside the checkpoint still receive gradients.
    hidden_states = hidden_states.detach().requires_grad_(True)
    if "hidden_states" in kwargs:
        return args, {**kwargs, "hidden_states": hidden_states}
    return (*args[:2], hidden_states, *args[3:]), kwargs


def freeze_non_mtp_parameters(
    models: list[torch.nn.Module],
    *,
    allow_missing_mtp: bool = False,
) -> list[torch.nn.Module]:
    """Bridge pre-wrap hook: train only parameters exclusively owned by ``mtp``.

    Inspect aliases too: an embedding/output parameter registered both inside MTP
    and on the backbone must remain frozen. Run before DDP allocates gradient
    buffers and before the optimizer selects parameters. ``allow_missing_mtp``
    is for earlier pipeline stages after the provider validates the global MTP
    configuration; the stage owning MTP must still have trainable parameters.
    """
    parameters: dict[int, tuple[torch.nn.Parameter, bool]] = {}
    for model in models:
        for name, parameter in model.named_parameters(remove_duplicate=False):
            is_mtp = "mtp" in name.split(".")[:-1]
            previous = parameters.get(id(parameter))
            if previous is not None:
                is_mtp = is_mtp and previous[1]
            parameters[id(parameter)] = (parameter, is_mtp)

    if not allow_missing_mtp and not any(is_mtp for _, is_mtp in parameters.values()):
        raise ValueError("mtp_only found no MTP-specific parameters to train")
    for parameter, is_mtp in parameters.values():
        parameter.requires_grad_(is_mtp)
    for model in models:
        for name, module in model.named_modules():
            if name.split(".")[-1] == "mtp":
                if _enable_mtp_input_grad not in module._forward_pre_hooks.values():
                    module.register_forward_pre_hook(
                        _enable_mtp_input_grad, with_kwargs=True
                    )
    return models
