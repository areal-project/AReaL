# SPDX-License-Identifier: Apache-2.0

import shlex
from collections.abc import Sequence
from dataclasses import dataclass

from areal.api.cli_args import SchedulingStrategy, is_colocation_strategy


def is_v2_training_guard_colocation(
    role: str, strategy: SchedulingStrategy | None, command: str | None
) -> bool:
    """Identify the v2 training guard path without matching other services."""
    if not is_colocation_strategy(strategy) or not role.endswith("-guard"):
        return False
    try:
        command_parts = shlex.split(command or "")
    except ValueError:
        return False
    if "-m" not in command_parts:
        return False
    module_index = command_parts.index("-m") + 1
    return (
        module_index < len(command_parts)
        and command_parts[module_index] == "areal.v2.training_service.guard"
    )


def colocated_gpu_rank_key(
    host: str, device_id: int, local_rank: int
) -> tuple[int, str, int]:
    """Return the canonical rank key shared by schedulers and the gateway."""
    return device_id - local_rank, host, local_rank


@dataclass(frozen=True)
class ColocatedGpuSlot:
    """A physical GPU slot in the canonical colocated rank order."""

    group_index: int
    host: str
    device_id: int
    local_rank: int


def canonical_colocated_gpu_slots(
    groups: Sequence[tuple[str, Sequence[str]]],
) -> list[ColocatedGpuSlot]:
    """Order physical GPU slots like the weight-update gateway.

    Each input group represents one inference server and must list contiguous
    node-local devices in local-rank order. The returned order uses server base
    device first, then host and local rank, so it is independent of scheduler
    worker creation order.
    """
    slots: list[ColocatedGpuSlot] = []
    seen_devices: set[tuple[str, int]] = set()
    for group_index, (host, raw_devices) in enumerate(groups):
        if not host:
            raise ValueError(f"GPU group {group_index} has no host")
        try:
            devices = [int(device) for device in raw_devices]
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"GPU group {group_index} has non-numeric devices: {raw_devices}"
            ) from e
        if not devices:
            raise ValueError(f"GPU group {group_index} has no devices")

        expected = list(range(devices[0], devices[0] + len(devices)))
        if devices != expected:
            raise ValueError(
                f"GPU group {group_index} devices must be contiguous in local-rank "
                f"order, got {devices}"
            )

        for local_rank, device_id in enumerate(devices):
            device = (host, device_id)
            if device in seen_devices:
                raise ValueError(
                    f"Physical GPU {host}:{device_id} belongs to multiple groups"
                )
            seen_devices.add(device)
            slots.append(
                ColocatedGpuSlot(
                    group_index=group_index,
                    host=host,
                    device_id=device_id,
                    local_rank=local_rank,
                )
            )

    return sorted(
        slots,
        key=lambda slot: colocated_gpu_rank_key(
            slot.host, slot.device_id, slot.local_rank
        ),
    )


_TMS_ENV_OFF = {
    "LD_PRELOAD": "",
    "TMS_INIT_ENABLE": "0",
    "TMS_INIT_ENABLE_CPU_BACKUP": "0",
}


def colocated_train_guard_fork_env(
    base_env: dict[str, str], gpu_slot: int
) -> dict[str, str]:
    """Build the environment for a train guard forked from a rollout guard.

    Rollout guards use the torch-memory-saver preload hook, while Megatron
    workers manage their own offload state. The actor fork must disable the
    inherited hook before creating CUDA-IPC-exported transfer tensors.
    """
    return {**base_env, **_TMS_ENV_OFF, "CUDA_VISIBLE_DEVICES": str(gpu_slot)}
