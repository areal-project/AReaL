# SPDX-License-Identifier: Apache-2.0
"""RDT FSDP Adapter for training-side weight update."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
from awex.meta.weight_meta import (
    ParameterMeta,
    ParameterReplicaMeta,
    ParameterShardMeta,
)
from awex.sharding.param_sharding import ShardingType
from awex.sharding.rank_info import RankInfo
from awex.transfer.transfer_plan import TransferPlan, TransferPlanBuilder
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Shard

from areal.engine.core.model import is_qwen_vl_model
from areal.experimental.weight_update.rdt import fetch_kv_metadata
from areal.utils import logging

if TYPE_CHECKING:
    from areal.engine.fsdp_engine import FSDPEngine

logger = logging.getLogger("RDTFSDPAdapter")


class RDTFSDPAdapter:
    """RDT training adapter for FSDPEngine."""

    def __init__(self, engine: FSDPEngine):
        """Initialize adapter with FSDPEngine reference.

        Args:
            engine: FSDPEngine instance holding the model
        """
        self._engine = engine
        self._transfer_plans: dict[str, TransferPlan] = {}
        self._transfer_ranks: dict[str, int] = {}

    @property
    def parallelism_strategy(self) -> dict:
        """Parallelism strategy for TransferPlan building.

        Returns:
            dict: Contains world_size, tp_size, pp_size, dp_size, ep_size
        """
        mesh = self._engine.world_mesh
        dim_names = tuple(mesh.mesh_dim_names or ())
        tp_size = mesh.size(dim_names.index("sp_tp")) if "sp_tp" in dim_names else 1

        return {
            "world_size": self._engine.world_size,
            "tp_size": tp_size,
            "pp_size": 1,
            "dp_size": self._engine.data_parallel_world_size,
            "ep_size": 1,
            "dp_replicated": False,
        }

    def get_weight_metadata(self) -> list[ParameterMeta]:
        """Extract parameter shard metadata for TransferPlan building.

        Returns:
            list[ParameterMeta]: Parameter metadata for all model parameters
        """
        rank_info = self._build_rank_info()
        metadata: list[ParameterMeta] = []

        for raw_name, param in self._engine.model.named_parameters():
            name = self._to_hf_name(raw_name)
            tensor = param.data
            if isinstance(tensor, DTensor):
                shard_meta = self._extract_dtensor_shard_meta(name, tensor, rank_info)
                global_shape = tuple(tensor.shape)
                global_numel = int(tensor.numel())
                dtype = tensor.dtype
            else:
                shard_meta = self._extract_plain_shard_meta(name, tensor, rank_info)
                global_shape = tuple(tensor.shape)
                global_numel = int(tensor.numel())
                dtype = tensor.dtype

            replica = ParameterReplicaMeta(shards=[shard_meta])
            metadata.append(
                ParameterMeta(
                    name=name,
                    global_numel=global_numel,
                    global_shape=global_shape,
                    dtype=dtype,
                    shards=[shard_meta],
                    replicas=[replica],
                )
            )

        return metadata

    def get_local_shard_parameters(
        self, required_names: list[str] | None = None
    ) -> dict[str, torch.Tensor]:
        """Return local shard tensors in HF naming.

        Args:
            required_names: Optional filter for specific parameters

        Returns:
            dict[str, torch.Tensor]: Local shard tensors by HF name
        """
        required = set(required_names) if required_names else None
        local_params: dict[str, torch.Tensor] = {}

        for raw_name, param in self._engine.model.named_parameters():
            name = self._to_hf_name(raw_name)
            if required is not None and name not in required:
                continue

            tensor = param.data
            if isinstance(tensor, DTensor):
                local_params[name] = tensor._local_tensor
            else:
                local_params[name] = tensor

        return local_params

    def save_parameters(self, save_path: str, names: list[str] | None = None) -> None:
        """Save local shard parameters to file for debugging.

        Args:
            save_path: File path to save parameters
            names: Optional filter for specific parameters
        """
        params = self.get_local_shard_parameters(names)
        cpu_params = {k: v.detach().cpu().clone() for k, v in params.items()}
        torch.save(cpu_params, save_path)

    def init_weight_update_group(
        self,
        pair_name: str,
        kv_store_url: str,
        infer_world_size: int,
        train_world_size: int,
        num_engines: int,
        transfer_rank: int,
    ) -> None:
        """Initialize RDT weight update group for TW.

        Args:
            pair_name: TW-IW pair identifier
            kv_store_url: Gateway KV store URL
            infer_world_size: Total IW world size
            train_world_size: Total TW world size
            num_engines: Number of IW engines
            transfer_rank: TW's transfer rank
        """
        self._transfer_ranks[pair_name] = transfer_rank

        infer_meta, train_meta = fetch_kv_metadata(kv_store_url, pair_name)

        builder = TransferPlanBuilder(
            infer_world_size=infer_world_size,
            train_world_size=train_world_size,
            num_infer_engines=num_engines,
        )
        self._transfer_plans[pair_name] = builder.build_local_transfer_plan(
            infer_meta, train_meta, global_transfer_rank=transfer_rank
        )
        logger.info(
            f"RDT TW init: Built TransferPlan for pair '{pair_name}' transfer_rank={transfer_rank}"
        )

    def get_weights_for_infer_rank(
        self, pair_name: str, infer_rank: int, version: int
    ) -> dict[str, torch.Tensor]:
        """Return sliced parameters for a specific infer_rank.

        TW uses its TransferPlan to determine what to send to each IW.

        Args:
            pair_name: TW-IW pair identifier
            infer_rank: IW's global rank (determines which tensors to return)
            version: Weight version number

        Returns:
            dict[str, torch.Tensor]: Sliced tensors for this infer_rank
        """
        if pair_name not in self._transfer_plans:
            raise RuntimeError(f"TransferPlan not initialized for pair '{pair_name}'")

        plan = self._transfer_plans[pair_name]
        local_params = self.get_local_shard_parameters()

        sliced_tensors: dict[str, torch.Tensor] = {}

        for send_rank, operations in plan.inter_operations.items():
            for op in operations:
                if op.recv_shard_meta.global_rank == infer_rank:
                    full_tensor = local_params.get(op.send_shard_meta.name)
                    if full_tensor is None:
                        logger.warning(
                            f"TW tensor not found: {op.send_shard_meta.name}"
                        )
                        continue
                    sliced_tensors[op.recv_shard_meta.name] = full_tensor[
                        op.train_slices
                    ]

        return sliced_tensors

    def teardown_weight_update_group(self, pair_name: str | None = None) -> None:
        """Clear stored TransferPlans.

        Args:
            pair_name: Optional specific pair to teardown; clears all if None
        """
        if pair_name:
            self._transfer_plans.pop(pair_name, None)
            self._transfer_ranks.pop(pair_name, None)
        else:
            self._transfer_plans.clear()
            self._transfer_ranks.clear()

    def _to_hf_name(self, name: str) -> str:
        """Convert to HuggingFace canonical format for Qwen-VL.

        Args:
            name: Internal parameter name

        Returns:
            str: HF canonical name
        """
        if self._engine.is_vision_model and is_qwen_vl_model(
            self._engine.model_config.model_type
        ):
            new_name = name
            if new_name.startswith("model.model."):
                new_name = new_name.replace("model.model.", "model.", 1)
            if new_name.startswith("model.visual."):
                new_name = new_name.replace("model.", "", 1)
            return new_name
        return name

    def _build_rank_info(self) -> RankInfo:
        """Build RankInfo for shard metadata extraction.

        Returns:
            RankInfo: Rank information for current worker
        """
        mesh = self._engine.world_mesh
        dim_names = tuple(mesh.mesh_dim_names or ())

        tp_size = mesh.size(dim_names.index("sp_tp")) if "sp_tp" in dim_names else 1
        tp_rank = (
            mesh.get_local_rank(dim_names.index("sp_tp")) if "sp_tp" in dim_names else 0
        )
        cp_size = mesh.size(dim_names.index("sp")) if "sp" in dim_names else 1
        cp_rank = mesh.get_local_rank(dim_names.index("sp")) if "sp" in dim_names else 0
        local_rank = int(os.environ.get("LOCAL_RANK", self._engine.rank))

        return RankInfo(
            tp_rank=tp_rank,
            tp_size=tp_size,
            pp_rank=0,
            pp_size=1,
            dp_size=self._engine.data_parallel_world_size,
            dp_rank=self._engine.dp_rank,
            ep_rank=0,
            ep_size=1,
            ep_tp_rank=0,
            ep_tp_size=1,
            attn_tp_rank=tp_rank,
            attn_tp_size=tp_size,
            attn_dp_rank=self._engine.dp_rank,
            world_size=self._engine.world_size,
            global_rank=self._engine.rank,
            local_rank=local_rank,
            engine_rank=0,
            is_infer=False,
            cp_rank=cp_rank,
            cp_size=cp_size,
            cp_mode="none",
        )

    @staticmethod
    def _compute_dtensor_offset(dtensor: DTensor) -> tuple[int, ...]:
        """Compute global offset for DTensor shard.

        Args:
            dtensor: Distributed tensor

        Returns:
            tuple[int, ...]: Global offset for local shard
        """
        global_shape = tuple(dtensor.shape)
        placements = dtensor.placements
        mesh = dtensor.device_mesh

        offset = [0] * len(global_shape)
        remaining_shape = list(global_shape)

        for mesh_dim, placement in enumerate(placements):
            if isinstance(placement, Shard):
                shard_dim = placement.dim
                mesh_size = mesh.size(mesh_dim)
                chunk_size = remaining_shape[shard_dim] // mesh_size
                coord = mesh.get_local_rank(mesh_dim)
                offset[shard_dim] += coord * chunk_size
                remaining_shape[shard_dim] = chunk_size

        return tuple(offset)

    @staticmethod
    def _extract_dtensor_sharding(dtensor: DTensor) -> tuple[int, int]:
        """Extract sharding dimension and num_shards from DTensor.

        Args:
            dtensor: Distributed tensor

        Returns:
            tuple[int, int]: (sharding_dim, num_shards)
        """
        shard_info: dict[int, int] = {}
        for mesh_dim, placement in enumerate(dtensor.placements):
            if isinstance(placement, Shard):
                dim = placement.dim
                mesh_size = dtensor.device_mesh.size(mesh_dim)
                shard_info[dim] = shard_info.get(dim, 1) * mesh_size

        if not shard_info:
            return 0, 1

        primary_dim = max(shard_info.items(), key=lambda item: item[1])[0]
        return primary_dim, shard_info[primary_dim]

    def _extract_dtensor_shard_meta(
        self,
        name: str,
        dtensor: DTensor,
        rank_info: RankInfo,
    ) -> ParameterShardMeta:
        """Extract ParameterShardMeta for DTensor.

        Args:
            name: Parameter name
            dtensor: Distributed tensor
            rank_info: Rank information

        Returns:
            ParameterShardMeta: Shard metadata
        """
        local_tensor = dtensor._local_tensor
        sharding_dim, num_shards = self._extract_dtensor_sharding(dtensor)
        sharding_type = (
            ShardingType.TP_SHARDING if num_shards > 1 else ShardingType.NO_SHARDING
        )

        return ParameterShardMeta(
            tp_rank=rank_info.tp_rank,
            attn_tp_rank=rank_info.attn_tp_rank,
            pp_rank=rank_info.pp_rank,
            ep_rank=rank_info.ep_rank,
            ep_tp_rank=rank_info.ep_tp_rank,
            global_rank=rank_info.global_rank,
            world_size=rank_info.world_size,
            engine_rank=rank_info.engine_rank,
            cp_rank=rank_info.cp_rank,
            cp_size=rank_info.cp_size,
            cp_mode=rank_info.cp_mode,
            name=name,
            shape=tuple(local_tensor.shape),
            numel=int(local_tensor.numel()),
            dtype=local_tensor.dtype,
            global_offset=self._compute_dtensor_offset(dtensor),
            sharding_type=sharding_type,
            num_shards=num_shards,
            sharding_dim=sharding_dim,
        )

    def _extract_plain_shard_meta(
        self,
        name: str,
        tensor: torch.Tensor,
        rank_info: RankInfo,
    ) -> ParameterShardMeta:
        """Extract ParameterShardMeta for plain tensor.

        Args:
            name: Parameter name
            tensor: Plain tensor (non-DTensor)
            rank_info: Rank information

        Returns:
            ParameterShardMeta: Shard metadata
        """
        return ParameterShardMeta(
            tp_rank=rank_info.tp_rank,
            attn_tp_rank=rank_info.attn_tp_rank,
            pp_rank=rank_info.pp_rank,
            ep_rank=rank_info.ep_rank,
            ep_tp_rank=rank_info.ep_tp_rank,
            global_rank=rank_info.global_rank,
            world_size=rank_info.world_size,
            engine_rank=rank_info.engine_rank,
            cp_rank=rank_info.cp_rank,
            cp_size=rank_info.cp_size,
            cp_mode=rank_info.cp_mode,
            name=name,
            shape=tuple(tensor.shape),
            numel=int(tensor.numel()),
            dtype=tensor.dtype,
            global_offset=tuple([0] * len(tuple(tensor.shape))),
            sharding_type=ShardingType.NO_SHARDING,
            num_shards=1,
            sharding_dim=0,
        )
