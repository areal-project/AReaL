# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class AwexInferenceAdapter(Protocol):
    """Protocol for inference-side weight update adapters."""

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
        """Pull peer meta from KV store, build local recv plan, join NCCL group."""
        ...

    def execute_weight_update(self, version: int) -> None:
        """Execute cached local P2P recv plan."""
        ...

    def batch_isend_irecv(self, **kwargs) -> None:
        """Execute awex batch P2P send/recv operations."""
        ...

    def teardown_weight_update_group(self) -> None:
        """Destroy NCCL group and clear cached state."""
        ...


@runtime_checkable
class AwexColocateInferenceAdapter(Protocol):
    """Inference-side lifecycle required by AWEX colocation."""

    def init_colocate_weight_update(
        self,
        *,
        pair_name: str,
        meta_server_addr: str,
        transfer_rank: int,
        infer_world_size: int,
        train_world_size: int,
        num_engines: int,
        timeout_s: float,
    ) -> None: ...

    def execute_colocate_weight_update(self, version: int) -> None: ...

    def release_memory(self, tags: list[str] | None = None) -> None: ...

    def resume_memory(self, tags: list[str] | None = None) -> None: ...
