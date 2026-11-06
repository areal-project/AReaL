# SPDX-License-Identifier: Apache-2.0
"""Lightweight device and timeout helpers for AWEX colocated transfer."""

from __future__ import annotations

import os

from areal.utils.environ import get_float_env_var


def resolve_physical_gpu_id(relative_gpu_id: int) -> int:
    """Map a CUDA-masked relative device index to its physical GPU id.

    CUDA IPC keys must be unique per node, so both sides of a colocated
    transfer have to agree on physical GPU ids. Inside a process that was
    given a device mask, ``torch.cuda.current_device()`` and SGLang's
    ``gpu_id`` are indices into that mask rather than physical ids, so the
    mask itself is the only ground truth. UUID masks and invalid indices are
    rejected because they cannot produce the node-local ordinal AWEX keys use.
    """
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not cuda_visible:
        return relative_gpu_id
    visible_devices = [item.strip() for item in cuda_visible.split(",") if item.strip()]
    if not all(item.isdigit() for item in visible_devices):
        raise ValueError(
            "AWEX colocate requires numeric CUDA_VISIBLE_DEVICES entries; "
            f"got {visible_devices!r}"
        )
    if relative_gpu_id >= len(visible_devices):
        raise ValueError(
            f"CUDA device {relative_gpu_id} is outside "
            f"CUDA_VISIBLE_DEVICES={visible_devices!r}"
        )
    return int(visible_devices[relative_gpu_id])


def awex_colocate_timeout_s(default: float = 1800.0) -> float:
    return get_float_env_var("AWEX_COLOCATE_TIMEOUT_S", default)
