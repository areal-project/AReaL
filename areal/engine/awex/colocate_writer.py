# SPDX-License-Identifier: Apache-2.0
"""Compatibility facade for the shared Megatron AWEX adapter."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from areal.utils.environ import get_float_env_var

if TYPE_CHECKING:
    from areal.engine.awex.megatron_adapter import AwexMegatronAdapter
    from areal.engine.megatron_engine import MegatronEngine
    from areal.engine.megatron_utils.weight_residency import MegatronWeightResidency


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


class AwexWeightPublisher:
    """Preserve the v1 publisher API while delegating to the shared adapter."""

    def __init__(
        self,
        engine: MegatronEngine,
        residency: MegatronWeightResidency | None = None,
    ) -> None:
        from areal.engine.awex.megatron_adapter import AwexMegatronAdapter

        self._adapter = AwexMegatronAdapter(engine, residency)

    @property
    def residency(self) -> MegatronWeightResidency:
        return self._adapter.residency

    def eager_publish_train_info(self, meta_server_addr: str | None) -> None:
        self._adapter.eager_publish_train_info(meta_server_addr)

    def _prepare_residency_for_publish(self) -> None:
        self._adapter._prepare_residency_for_publish()

    def init_colocate_weight_update(
        self,
        meta_server_addr: str | None = None,
        pair_name: str = "default",
        transfer_rank: int = 0,
        timeout_s: float | None = None,
    ) -> None:
        self._adapter.init_legacy_colocate_weight_update(
            meta_server_addr=meta_server_addr,
            pair_name=pair_name,
            transfer_rank=transfer_rank,
            timeout_s=timeout_s,
        )

    def execute_colocate_weight_update(self, version: int) -> None:
        self._adapter.execute_legacy_colocate_weight_update(version)

    def finish_colocate_weight_update(self, training_world_size: int) -> None:
        self._adapter.finish_legacy_colocate_weight_update(training_world_size)

    def release_memory(self, tags: list[str] | None = None) -> None:
        self._adapter.release_memory(tags)

    def resume_memory(self, tags: list[str] | None = None) -> None:
        self._adapter.resume_memory(tags)

    @property
    def _released_tags(self) -> set[str]:
        return set(self.residency.released_tags)

    def _release_grad_memory(self) -> None:
        self.residency.release_grad_memory()

    def ensure_grad_buffers(self) -> None:
        self.residency.ensure_grad_buffers()


def __getattr__(name: str) -> Any:
    if name == "AwexMegatronAdapter":
        from areal.engine.awex.megatron_adapter import AwexMegatronAdapter

        return AwexMegatronAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AwexMegatronAdapter",
    "AwexWeightPublisher",
    "awex_colocate_timeout_s",
    "resolve_physical_gpu_id",
]
