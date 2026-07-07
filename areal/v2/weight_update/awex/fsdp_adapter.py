# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

# pyright: reportMissingImports=false
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
from awex.transfer.nccl_comm import batch_send_recv, nccl_build_send_ops
from awex.transfer.transfer_plan import TransferPlan, TransferPlanBuilder
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Shard

from areal.engine.core.model import is_qwen_vl_model
from areal.utils import logging
from areal.v2.weight_update.awex import (
    awex_wu_use_group,
    fetch_kv_metadata,
)
from areal.v2.weight_update.nccl_group import (
    init_weights_update_group,
    setup_batch_isend_irecv,
)
from areal.v2.weight_update.training_adapter import (
    AwexTrainingAdapter,
)

if TYPE_CHECKING:
    from areal.engine.fsdp_engine import FSDPEngine

logger = logging.getLogger("AwexFSDPAdapter")


class AwexFSDPAdapter(AwexTrainingAdapter):
    """Awex training adapter wrapping FSDPEngine for shard-direct NCCL P2P updates."""

    def __init__(self, engine: FSDPEngine):
        self._engine = engine
        self._transfer_plan: TransferPlan | None = None
        self._weights_update_group = None
        self._transfer_rank: int | None = None

    @property
    def parallelism_strategy(self) -> dict:
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

    @property
    def _tie_word_embeddings(self) -> bool:
        return getattr(self._engine.model_config, "tie_word_embeddings", False)

    def get_weight_metadata(self) -> list[ParameterMeta]:
        rank_info = self._build_rank_info()
        metadata: list[ParameterMeta] = []

        for raw_name, param in self._engine.model.named_parameters():
            name = self._to_hf_name(raw_name)
            if self._tie_word_embeddings and name == "lm_head.weight":
                continue
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
        required = set(required_names) if required_names else None
        local_params: dict[str, torch.Tensor] = {}

        for raw_name, param in self._engine.model.named_parameters():
            name = self._to_hf_name(raw_name)
            if self._tie_word_embeddings and name == "lm_head.weight":
                continue
            if required is not None and name not in required:
                continue

            tensor = param.data
            if isinstance(tensor, DTensor):
                local_params[name] = tensor._local_tensor
            else:
                local_params[name] = tensor

        return local_params

    def save_parameters(self, save_path: str, names: list[str] | None = None) -> None:
        params = self.get_local_shard_parameters(names)
        cpu_params = {k: v.detach().cpu().clone() for k, v in params.items()}
        torch.save(cpu_params, save_path)

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
        self._transfer_rank = transfer_rank

        infer_meta, train_meta = fetch_kv_metadata(kv_store_url, pair_name)

        builder = TransferPlanBuilder(
            infer_world_size=infer_world_size,
            train_world_size=train_world_size,
            num_infer_engines=num_engines,
        )
        self._transfer_plan = builder.build_local_transfer_plan(
            infer_meta, train_meta, global_transfer_rank=transfer_rank
        )

        os.environ["TORCHELASTIC_USE_AGENT_STORE"] = str(False)
        self._weights_update_group = init_weights_update_group(
            master_address=master_addr,
            master_port=master_port,
            rank=transfer_rank,
            world_size=world_size,
            group_name=f"awex_{pair_name}",
            role="training",
        )

    def execute_weight_update(self, version: int) -> None:
        del version
        if self._transfer_plan is None:
            raise RuntimeError("Transfer plan is not initialized")
        if self._weights_update_group is None:
            raise RuntimeError("Weight update group is not initialized")
        if self._transfer_rank is None:
            raise RuntimeError("Transfer rank is not initialized")

        params = self.get_local_shard_parameters()
        send_ops, _, _ = nccl_build_send_ops(
            params,
            self._transfer_plan,
            self._weights_update_group,
            copy_rank=self._transfer_rank,
        )
        batch_send_recv(
            send_ops=send_ops,
            recv_ops=[],
            blocking=True,
            use_group=awex_wu_use_group(),
        )
        torch.distributed.barrier(group=self._weights_update_group)

    def batch_isend_irecv(self, **kwargs) -> None:
        setup_kwargs = {k: v for k, v in kwargs.items() if k != "world_size"}
        setup_batch_isend_irecv(
            self._weights_update_group,
            self._transfer_rank,
            kwargs.get("world_size", 0),
            **setup_kwargs,
        )

    def teardown_weight_update_group(self) -> None:
        if (
            self._weights_update_group is not None
            and torch.distributed.is_initialized()
        ):
            torch.distributed.destroy_process_group(self._weights_update_group)
        self._weights_update_group = None
        self._transfer_plan = None
        self._transfer_rank = None

    def _to_hf_name(self, name: str) -> str:
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
