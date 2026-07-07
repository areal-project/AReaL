# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class AwexTrainingAdapter(Protocol):
    """Protocol for training-side weight update adapters."""

    @property
    def parallelism_strategy(self) -> dict:
        """Report parallelism strategy.

        Returns dict with world_size, tp_size, pp_size, dp_size, ep_size.
        """
        ...

    def get_weight_metadata(self) -> list:
        """Extract this worker's parameter shard metadata in awex format.

        Returns list[ParameterMeta].
        """
        ...

    def get_local_shard_parameters(
        self, required_names: list[str] | None = None
    ) -> dict[str, torch.Tensor]:
        """Return local shard tensors in canonical HF naming."""
        ...

    def init_weight_update_group(
        self,
        pair_name: str,
        master_addr: str,
        master_port: int,
        transfer_rank: int,
        world_size: int,
        kv_store_url: str,
        infer_world_size: int,
        train_world_size: int,
        num_engines: int,
    ) -> None:
        """Pull peer meta from KV store, build local send plan, join NCCL group."""
        ...

    def execute_weight_update(self, version: int) -> None:
        """Execute cached local P2P send plan."""
        ...

    def batch_isend_irecv(self, **kwargs) -> None:
        """Execute awex batch P2P send/recv operations."""
        ...

    def teardown_weight_update_group(self) -> None:
        """Destroy NCCL group and clear cached state."""
        ...

    def init_colocate_weight_update(
        self,
        pair_name: str,
        kv_store_url: str,
        transfer_rank: int,
        infer_world_size: int,
        train_world_size: int,
        num_engines: int,
        master_port: int,
        admin_api_key: str = "areal-admin-key",
        timeout_s: float = 120.0,
    ) -> None:
        """Register device info in KV store for colocated weight transfer."""
        ...

    def execute_colocate_weight_update(self, version: int) -> None:
        """Serialize weights via IPC and put to KV store."""
        ...

    def release_memory(self, tags: list[str] | None = None) -> None:
        """Release GPU memory (optimizer/weights) for colocated mode."""
        ...

    def resume_memory(self, tags: list[str] | None = None) -> None:
        """Resume GPU memory occupation."""
        ...
