# SPDX-License-Identifier: Apache-2.0
"""Best-effort GPU-memory snapshots for AWEX reconnect diagnostics."""

from __future__ import annotations

import os
import socket
from typing import Any

import torch


def _physical_device_id(logical_device: int) -> str | None:
    """Resolve a CUDA logical device through CUDA_VISIBLE_DEVICES when possible."""

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible_devices:
        return str(logical_device)
    devices = [item.strip() for item in visible_devices.split(",") if item.strip()]
    if logical_device >= len(devices):
        return None
    return devices[logical_device]


def collect_awex_memory_probe(
    *,
    role: str,
    pair_names: list[str],
    rank: int | str | None = None,
) -> dict[str, Any]:
    """Return a JSON-serializable local memory snapshot without adding a dependency.

    NVML is imported only at probe time. Missing NVML support is reported in the
    snapshot rather than making an opt-in diagnostic interfere with recovery.
    """

    result: dict[str, Any] = {
        "role": role,
        "rank": rank,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "pair_names": sorted(set(pair_names)),
        "cuda_available": torch.cuda.is_available(),
    }
    if not torch.cuda.is_available():
        return result

    logical_device = torch.cuda.current_device()
    physical_device = _physical_device_id(logical_device)
    result.update(
        {
            "cuda_logical_device": logical_device,
            "cuda_physical_device": physical_device,
            "torch_allocated_bytes": torch.cuda.memory_allocated(logical_device),
            "torch_reserved_bytes": torch.cuda.memory_reserved(logical_device),
        }
    )

    nvml_initialized = False
    try:
        import pynvml

        pynvml.nvmlInit()
        nvml_initialized = True
        if physical_device is None or not physical_device.isdecimal():
            raise RuntimeError(
                "CUDA_VISIBLE_DEVICES does not expose a numeric physical device ID"
            )
        handle = pynvml.nvmlDeviceGetHandleByIndex(int(physical_device))
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        result["nvml_total_bytes"] = memory.total
        result["nvml_used_bytes"] = memory.used
    except Exception as exc:
        result["nvml_error"] = str(exc)
    finally:
        if nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception as exc:
                result["nvml_shutdown_error"] = str(exc)
    return result
