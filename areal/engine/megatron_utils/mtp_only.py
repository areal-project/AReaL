# SPDX-License-Identifier: Apache-2.0

import torch


def freeze_non_mtp_parameters(
    models: list[torch.nn.Module],
) -> list[torch.nn.Module]:
    """Bridge pre-wrap hook: train only parameters exclusively owned by ``mtp``.

    Inspect aliases too: an embedding/output parameter registered both inside MTP
    and on the backbone must remain frozen. Run before DDP allocates gradient
    buffers and before the optimizer selects parameters.
    """
    parameters: dict[int, tuple[torch.nn.Parameter, bool]] = {}
    for model in models:
        for name, parameter in model.named_parameters(remove_duplicate=False):
            is_mtp = "mtp" in name.split(".")[:-1]
            previous = parameters.get(id(parameter))
            if previous is not None:
                is_mtp = is_mtp and previous[1]
            parameters[id(parameter)] = (parameter, is_mtp)

    if not any(is_mtp for _, is_mtp in parameters.values()):
        raise ValueError("mtp_only found no MTP-specific parameters to train")
    for parameter, is_mtp in parameters.values():
        parameter.requires_grad_(is_mtp)
    return models
