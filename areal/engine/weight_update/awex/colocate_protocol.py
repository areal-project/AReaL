# SPDX-License-Identifier: Apache-2.0

"""Controller-independent protocol values for AWEX colocation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class ColocateTopology:
    """Resolved rank layout for one colocated inference process.

    Controller-specific code resolves the exact transfer rank. This class only
    validates and exposes the controller-independent engine decomposition.
    """

    transfer_rank: int
    infer_world_size: int
    train_world_size: int
    instance_world_size: int

    def __post_init__(self) -> None:
        if self.infer_world_size != self.train_world_size:
            raise ValueError(
                "Colocate mode requires equal inference and training rank counts; "
                f"got infer={self.infer_world_size}, train={self.train_world_size}"
            )
        if self.instance_world_size < 1:
            raise ValueError(
                f"instance_world_size must be positive, got {self.instance_world_size}"
            )
        if self.infer_world_size % self.instance_world_size != 0:
            raise ValueError(
                f"infer_world_size ({self.infer_world_size}) must be divisible by "
                f"instance_world_size ({self.instance_world_size})"
            )
        if not 0 <= self.transfer_rank < self.infer_world_size:
            raise ValueError(
                f"transfer_rank must be in [0, {self.infer_world_size}), "
                f"got {self.transfer_rank}"
            )

    @property
    def num_infer_engines(self) -> int:
        return self.infer_world_size // self.instance_world_size

    @property
    def engine_rank(self) -> int:
        return self.transfer_rank // self.instance_world_size

    @property
    def instance_local_rank(self) -> int:
        return self.transfer_rank % self.instance_world_size


@dataclass(frozen=True)
class ColocateKeyspace:
    """MetaServer keys for one physical training/inference device pair."""

    ip_address: str
    physical_gpu_id: int

    AWEX_TRAIN_INFO: ClassVar[str] = "awex_train_info"
    INFER_CONF: ClassVar[str] = "infer_conf"
    INFER_PARAMS_META: ClassVar[str] = "infer_params_meta"
    NUM_INFER_ENGINES: ClassVar[str] = "num_infer_engines"
    TRAINING_PARAMS_META: ClassVar[str] = "training_params_meta"
    TRAINING_DEVICE_RANK_ENTRIES: ClassVar[str] = "training_device_rank_entries"
    ALL_TRAINING_OFFLOADED_WEIGHTS: ClassVar[str] = "all_training_offloaded_weights"
    FINISHED_WEIGHT_UPDATE_ENGINES: ClassVar[str] = "finished_weights_update_engines"

    def __post_init__(self) -> None:
        if not self.ip_address:
            raise ValueError("ip_address must not be empty")
        if self.physical_gpu_id < 0:
            raise ValueError(
                f"physical_gpu_id must be non-negative, got {self.physical_gpu_id}"
            )

    @property
    def writer_version(self) -> str:
        return f"awex_writer_version_{self.ip_address}_{self.physical_gpu_id}"

    def serialized_weights(self, version: int) -> str:
        return self._versioned("training_serialized_weights", version)

    def update_finished(self, version: int) -> str:
        return self._versioned("weights_update_finished", version)

    def write_finished(self, version: int) -> str:
        return self._versioned("write_finished", version)

    def _versioned(self, prefix: str, version: int) -> str:
        if version < 0:
            raise ValueError(f"version must be non-negative, got {version}")
        return f"{prefix}_{self.ip_address}_{self.physical_gpu_id}_{version}"
