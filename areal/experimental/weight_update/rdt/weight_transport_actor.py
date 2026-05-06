# SPDX-License-Identifier: Apache-2.0
"""WeightTransportActor for TW tensor transport via RDT."""

from __future__ import annotations

import ray
import torch

from areal.utils import logging

logger = logging.getLogger("WeightTransportActor")


@ray.remote
class WeightTransportActor:
    """Dedicated actor for weight tensor transport via RDT.

    Created inside TW subprocess, directly holds engine reference.
    Actor and engine are in the same process, actor accesses engine's weight tensors directly.

    Uses adapter (RDTFSDPAdapter/RDTMegatronAdapter) to encapsulate engine-specific logic:
    - parallelism_strategy: Extract TP/DP/PP info from engine
    - get_weight_metadata: Extract parameter metadata, handle DTensor
    - get_weights_for_infer_rank: Return shard tensors for specific infer_rank (TransferPlan slicing)
    """

    def __init__(self, engine):
        self._engine = engine
        self._adapter = None

    def _create_adapter(self):
        """Lazily create adapter based on engine type."""
        if self._adapter is not None:
            return self._adapter

        from areal.engine.fsdp_engine import FSDPEngine
        from areal.engine.megatron_engine import MegatronEngine
        from areal.experimental.weight_update.rdt.fsdp_adapter import RDTFSDPAdapter
        from areal.experimental.weight_update.rdt.megatron_adapter import (
            RDTMegatronAdapter,
        )

        if isinstance(self._engine, FSDPEngine):
            self._adapter = RDTFSDPAdapter(self._engine)
        elif isinstance(self._engine, MegatronEngine):
            self._adapter = RDTMegatronAdapter(self._engine)
        else:
            raise TypeError(
                f"Unsupported engine type for RDT: {type(self._engine).__name__}"
            )

        return self._adapter

    def get_parallelism_strategy(self) -> dict:
        """Return parallelism strategy for TransferPlan building."""
        return self._create_adapter().parallelism_strategy

    def get_weight_metadata(self) -> list:
        """Return parameter metadata for TransferPlan building."""
        return self._create_adapter().get_weight_metadata()

    def init_weight_update_group(
        self,
        pair_name: str,
        kv_store_url: str,
        infer_world_size: int,
        train_world_size: int,
        num_engines: int,
        transfer_rank: int,
    ) -> None:
        """Initialize weight update group, store TransferPlan."""
        self._create_adapter().init_weight_update_group(
            pair_name=pair_name,
            kv_store_url=kv_store_url,
            infer_world_size=infer_world_size,
            train_world_size=train_world_size,
            num_engines=num_engines,
            transfer_rank=transfer_rank,
        )

    @ray.method(tensor_transport="YR")
    def get_weights_tensor_yr(
        self, pair_name: str, infer_rank: int, version: int
    ) -> dict[str, torch.Tensor]:
        """Return sliced tensors for NPU (YR backend)."""
        return self._create_adapter().get_weights_for_infer_rank(
            pair_name, infer_rank, version
        )

    @ray.method(tensor_transport="NIXL")
    def get_weights_tensor_nixl(
        self, pair_name: str, infer_rank: int, version: int
    ) -> dict[str, torch.Tensor]:
        """Return sliced tensors for GPU (NIXL backend)."""
        return self._create_adapter().get_weights_for_infer_rank(
            pair_name, infer_rank, version
        )
