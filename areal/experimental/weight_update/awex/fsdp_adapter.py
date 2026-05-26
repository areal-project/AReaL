# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

# pyright: reportMissingImports=false
import gc
import os
import threading
import time
from typing import TYPE_CHECKING

import httpx
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
from awex.util.tensor_util import (
    cuda_ipc_serialize,
    group_tensors_by_shape_and_dtype,
)
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Shard

from areal.engine.core.model import is_qwen_vl_model
from areal.experimental.weight_update.awex import fetch_kv_metadata
from areal.experimental.weight_update.nccl_group import (
    init_weights_update_group,
    setup_batch_isend_irecv,
)
from areal.experimental.weight_update.training_adapter import (
    AwexTrainingAdapter,
)
from areal.utils import logging

if TYPE_CHECKING:
    from areal.engine.fsdp_engine import FSDPEngine

logger = logging.getLogger("AwexFSDPAdapter")


class AwexFSDPAdapter(AwexTrainingAdapter):
    """Awex training adapter wrapping FSDPEngine for shard-direct NCCL P2P
    updates and colocated CUDA-IPC updates (DP-only)."""

    def __init__(self, engine: FSDPEngine):
        self._engine = engine
        self._transfer_plan: TransferPlan | None = None
        self._weights_update_group = None
        self._transfer_rank: int | None = None

        # ── Colocate state ────────────────────────────────────────
        self._colocate_lock = threading.Lock()
        self._colocate_pair_name: str | None = None
        self._colocate_kv_store_url: str | None = None
        self._colocate_transfer_rank: int | None = None
        self._colocate_infer_world_size: int | None = None
        self._colocate_admin_api_key: str = "areal-admin-key"
        self._colocate_http_client: httpx.Client | None = None
        self._colocate_timeout_s: float = 120.0

        # ── release_memory / resume_memory state ──────────────────
        self._released_tags: set[str] = set()
        self._offloaded_weights: dict[str, torch.Tensor] = {}

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
                # Inference engines (SGLang/vLLM) collapse the tied head into
                # `model.embed_tokens.weight`; mirror that to keep param key
                # sets aligned across train and infer.
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
        weights_offloaded = "weights" in self._released_tags
        if weights_offloaded:
            self.resume_memory(tags=["weights"])
        try:
            params = self.get_local_shard_parameters(names)
            cpu_params = {k: v.detach().cpu().clone() for k, v in params.items()}
            torch.save(cpu_params, save_path)
        finally:
            if weights_offloaded:
                self.release_memory(tags=["weights"])

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
        batch_send_recv(send_ops=send_ops, recv_ops=[], blocking=True)
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
        if self._colocate_http_client is not None:
            self._colocate_http_client.close()
            self._colocate_http_client = None

    # ── Colocated weight transfer methods ─────────────────────────────────

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
        del train_world_size, num_engines, master_port  # not needed on training side
        self._colocate_pair_name = pair_name
        self._colocate_kv_store_url = kv_store_url
        self._colocate_transfer_rank = transfer_rank
        self._colocate_infer_world_size = infer_world_size
        self._colocate_admin_api_key = admin_api_key
        self._colocate_timeout_s = timeout_s
        if self._colocate_http_client is None:
            self._colocate_http_client = httpx.Client()
        logger.info(
            "Initialized colocate weight update for pair '%s', transfer_rank=%d",
            pair_name,
            transfer_rank,
        )

    def _iter_hf_params_local(self):
        """Yield (hf_name, local_shard_tensor) for every parameter on this rank.

        Each rank yields its own DTensor ``_local_tensor`` (the actual Shard(0)
        chunk owned by this rank), or the plain tensor if the parameter is not
        a DTensor.  This matches the ``Shard(0)`` metadata reported by
        ``_extract_dtensor_shard_meta`` so that awex's ``slice_tensor``
        contract holds: ``send_parameters[name].shape == shard_meta.shape``.

        Cross-engine reassembly (i.e. infer rank 0 needs train rank 1's slice
        for the second half of every param) is handled by the awex transfer
        plan via ``_recv_transfer_plan`` + the colocate transport's P2P phase
        — we just publish our local chunk via CUDA IPC and let awex route
        slices to whichever infer rank needs them.

        When ``tie_word_embeddings`` is set, ``lm_head.weight`` is skipped so
        the train-side key set matches inference engines (e.g. SGLang) which
        collapse the tied head into ``model.embed_tokens.weight``.
        """
        device = self._engine.device
        for raw_name, param in self._engine.model.named_parameters():
            name = self._to_hf_name(raw_name)
            if self._tie_word_embeddings and name == "lm_head.weight":
                continue
            tensor = param.data
            if isinstance(tensor, DTensor):
                local = tensor._local_tensor
            else:
                local = tensor
            if local.device.type == "cpu":
                local = local.to(device, non_blocking=True)
            yield name, local.detach()

    def execute_colocate_weight_update(self, version: int) -> None:
        with self._colocate_lock:
            self._execute_colocate_weight_update_locked(version)

    def _execute_colocate_weight_update_locked(self, version: int) -> None:
        kv_store_url = self._colocate_kv_store_url
        pair_name = self._colocate_pair_name
        transfer_rank = self._colocate_transfer_rank
        assert self._colocate_http_client is not None, (
            "init_colocate_weight_update must be called first"
        )
        client = self._colocate_http_client
        auth_headers = {"Authorization": f"Bearer {self._colocate_admin_api_key}"}
        timeout_s = self._colocate_timeout_s

        weights_offloaded = "weights" in self._released_tags
        if weights_offloaded:
            self.resume_memory(tags=["weights"])
        try:
            # Publish each rank's local DTensor shard (no all-gather).  The awex
            # transfer plan routes the right slice from each rank's IPC payload
            # to whichever infer rank needs it.
            params: dict[str, torch.Tensor] = {}
            for hf_name, tensor in self._iter_hf_params_local():
                params[hf_name] = tensor
            tensors = list(params.values())
            names = list(params.keys())

            group_tensors, metadata = group_tensors_by_shape_and_dtype(tensors)
            torch.cuda.synchronize()

            del tensors

            group_shared = [t.share_memory_() for t in group_tensors]
            serialized_weights = cuda_ipc_serialize((group_shared, metadata, names))
            torch.cuda.synchronize()

            kv_key = f"colocate_weights_rank{transfer_rank}_{version}"
            client.put(
                f"{kv_store_url}/weight_meta/{pair_name}/{kv_key}",
                json={"value": serialized_weights.hex()},
                headers=auth_headers,
                timeout=timeout_s,
            )

            logger.info(
                "Serialized %d params (%d groups) for colocate transfer v%d, rank %d",
                len(names),
                len(group_shared),
                version,
                transfer_rank,
            )

            done_key = f"colocate_done_rank{transfer_rank}_{version}"
            deadline = time.monotonic() + timeout_s
            poll_count = 0
            last_status = -1
            while time.monotonic() < deadline:
                resp = client.get(
                    f"{kv_store_url}/weight_meta/{pair_name}/{done_key}",
                    timeout=5.0,
                )
                last_status = resp.status_code
                if resp.status_code == 200:
                    break
                poll_count += 1
                time.sleep(0.1)
            else:
                raise TimeoutError(
                    f"Inference did not signal completion within {timeout_s}s "
                    f"(waiting_key={done_key}, put_key={kv_key}, "
                    f"polls={poll_count}, last_status={last_status})"
                )

            del group_shared, group_tensors, serialized_weights
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
        finally:
            if weights_offloaded:
                self.release_memory(tags=["weights"])

    # ── Memory release / resume (colocate only) ───────────────────────────

    _SUPPORTED_RELEASE_TAGS = {"weights"}

    def release_memory(self, tags: list[str] | None = None) -> None:
        """Release GPU memory for ``weights`` tag by offloading to CPU.

        v1 supports only ``weights``. ``optimizer`` is intentionally not
        supported on FSDP2 because per-parameter optimizer state lives in
        sharded DTensor form and replacing it requires coordination with
        FSDP2's reshard hooks (see ``release_memory`` design notes).

        Unsupported tags are logged as warnings and ignored.
        """
        tags = tags or list(self._SUPPORTED_RELEASE_TAGS)

        unsupported = [t for t in tags if t not in self._SUPPORTED_RELEASE_TAGS]
        if unsupported:
            logger.warning(
                "release_memory: tags %s not supported by FSDP adapter "
                "(supported: %s), ignoring",
                unsupported,
                sorted(self._SUPPORTED_RELEASE_TAGS),
            )

        tags_to_release = [
            t
            for t in tags
            if t in self._SUPPORTED_RELEASE_TAGS and t not in self._released_tags
        ]
        if not tags_to_release:
            logger.info("release_memory: tags=%s already released, skipping", tags)
            return

        if "weights" in tags_to_release:
            self._offload_model_weights()
            self._released_tags.add("weights")

        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("release_memory: done for tags=%s", tags_to_release)

    def _offload_model_weights(self) -> None:
        if self._engine.model is None:
            return
        for name, param in self._engine.model.named_parameters():
            tensor = param.data
            if isinstance(tensor, DTensor):
                local = tensor._local_tensor
                if local.is_cuda:
                    self._offloaded_weights[name] = local.detach().to(
                        "cpu", non_blocking=True
                    )
                    tensor._local_tensor = torch.empty(
                        0, dtype=local.dtype, device="cpu"
                    )
            else:
                if tensor.is_cuda:
                    self._offloaded_weights[name] = tensor.detach().to(
                        "cpu", non_blocking=True
                    )
                    param.data = torch.empty(0, dtype=tensor.dtype, device="cpu")
        logger.info(
            "Offloaded %d FSDP weight tensors to CPU",
            len(self._offloaded_weights),
        )

    def resume_memory(self, tags: list[str] | None = None) -> None:
        tags = tags or list(self._SUPPORTED_RELEASE_TAGS)

        tags_to_resume = [
            t
            for t in tags
            if t in self._SUPPORTED_RELEASE_TAGS and t in self._released_tags
        ]
        if not tags_to_resume:
            logger.info("resume_memory: tags=%s not released, skipping", tags)
            return

        if "weights" in tags_to_resume:
            self._reload_model_weights()
            self._released_tags.discard("weights")

        torch.cuda.synchronize()
        logger.info("resume_memory: done for tags=%s", tags_to_resume)

    def _reload_model_weights(self) -> None:
        if not self._offloaded_weights:
            return
        if self._engine.model is None:
            return

        device = self._engine.device
        for name, param in self._engine.model.named_parameters():
            if name not in self._offloaded_weights:
                continue
            saved = self._offloaded_weights[name]
            tensor = param.data
            if isinstance(tensor, DTensor):
                tensor._local_tensor = saved.to(device, non_blocking=True)
            else:
                param.data = saved.to(device, non_blocking=True)
        self._offloaded_weights.clear()
        logger.info("Reloaded FSDP weights to GPU")

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
        # Report this rank's actual local Shard(0) chunk: shape = local shape,
        # global_offset = where this chunk starts in the global tensor.  The
        # colocate IPC payload (`_iter_hf_params_local`) publishes that same
        # local chunk, so the awex transfer plan's `train_slices` (which are
        # relative to shard.start_offset) correctly index into it, and any
        # cross-engine P2P slice that reassembles the full tensor on the infer
        # side is computed against truthful per-rank ownership.
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
