# SPDX-License-Identifier: Apache-2.0
# pyright: reportMissingImports=false
from __future__ import annotations

import gc
import math
import os
import time
from typing import Any

import httpx
import torch
import torch.distributed as dist
from awex.meta.weight_meta import (
    ParameterMeta,
    ParameterReplicaMeta,
    ParameterShardMeta,
)
from awex.sharding.param_sharding import ShardingType
from awex.sharding.rank_info import RankInfo
from awex.sharding.sglang_sharding import (
    get_sglang_rank_info,
    get_sglang_sharding_strategy,
)
from awex.transfer.nccl_comm import batch_send_recv, nccl_build_recv_ops
from awex.transfer.nccl_stream_batch import NcclColocateStreamBatchTransport
from awex.transfer.transfer_plan import TransferPlan, TransferPlanBuilder, slice_tensor
from awex.util.tensor_util import (
    cuda_ipc_deserialize,
    reconstruct_tensors_from_groups,
)

from areal.infra.platforms import current_platform
from areal.utils import logging
from areal.v2.weight_update.awex import (
    awex_wu_use_group,
    fetch_kv_metadata,
)
from areal.v2.weight_update.awex.delta_config import (
    DTERuntimeConfig,
    synchronize_wire_dtypes,
    validate_dte_world_size,
)
from areal.v2.weight_update.inference_adapter import (
    AwexInferenceAdapter,
)
from areal.v2.weight_update.nccl_group import (
    init_weights_update_group,
    setup_batch_isend_irecv,
)

logger = logging.getLogger("AwexSGLangAdapter")


class AwexSGLangAdapter(AwexInferenceAdapter):
    """Awex inference adapter for in-process SGLang schedulers."""

    def __init__(self, scheduler: Any):
        self._scheduler = scheduler
        self._transfer_plan: TransferPlan | None = None
        self._weights_update_group = None
        self._weights_update_group_gloo = None
        self._world_size: int | None = None
        self._separation_delta_transport: NcclColocateStreamBatchTransport | None = None
        self._separation_wire_dtypes: tuple[torch.dtype, ...] | None = None
        self._transfer_rank: int | None = None
        self._rank_info: RankInfo | None = None
        self._weight_converter = None
        self._parameters: dict[str, torch.Tensor] | None = None
        self._released_tags: set[str] = set()
        self._colocate_admin_api_key: str = "areal-admin-key"
        self._colocate_http_client: httpx.Client | None = None
        self._colocate_timeout_s: float = 120.0
        self._colocate_transport = None
        self._train_to_infer_device_mapping: dict | None = None
        self._infer_to_train_device_mapping: dict | None = None
        self._dte_config = DTERuntimeConfig.from_env()

    def _get_model(self) -> torch.nn.Module:
        return self._scheduler.tp_worker.model_runner.model

    def _get_model_context(self) -> dict[str, Any]:
        server_args = self._scheduler.server_args
        tp_size = int(getattr(server_args, "tp_size", 1))
        pp_size = int(getattr(server_args, "pp_size", 1))
        dp_size = int(getattr(server_args, "dp_size", 1))

        if dist.is_available() and dist.is_initialized():
            world_size = int(dist.get_world_size())
            global_rank = int(dist.get_rank())
        else:
            world_size = int(tp_size * pp_size)
            global_rank = int(getattr(self._scheduler, "tp_rank", 0))

        local_rank = int(
            getattr(
                self._scheduler,
                "local_rank",
                os.environ.get("LOCAL_RANK", getattr(self._scheduler, "gpu_id", 0)),
            )
        )

        return {
            "scheduler": self._scheduler,
            "tp_rank": int(getattr(self._scheduler, "tp_rank", 0)),
            "tp_size": tp_size,
            "pp_rank": int(getattr(self._scheduler, "pp_rank", 0)),
            "pp_size": pp_size,
            "dp_size": dp_size,
            "world_size": world_size,
            "global_rank": global_rank,
            "local_rank": local_rank,
            "attn_tp_rank": int(
                getattr(
                    self._scheduler,
                    "attn_tp_rank",
                    getattr(self._scheduler, "tp_rank", 0),
                )
            ),
            "attn_tp_size": int(getattr(self._scheduler, "attn_tp_size", tp_size)),
            "attn_dp_rank": int(getattr(self._scheduler, "attn_dp_rank", 0)),
        }

    @property
    def parallelism_strategy(self) -> dict:
        model_context = self._get_model_context()
        server_args = self._scheduler.server_args
        tp_size = int(getattr(server_args, "tp_size", model_context["tp_size"]))
        pp_size = int(getattr(server_args, "pp_size", model_context["pp_size"]))
        dp_size = int(getattr(server_args, "dp_size", model_context["dp_size"]))
        ep_size = int(getattr(server_args, "ep_size", 1))

        return {
            "world_size": int(model_context["world_size"]),
            "tp_size": tp_size,
            "pp_size": pp_size,
            "dp_size": dp_size,
            "ep_size": ep_size,
            "num_engines": 1,
        }

    def _get_weight_converter(self, rank_info: RankInfo):
        """Build the same AWEX SGLang-to-HF converter used by the v1 reader."""
        if self._weight_converter is None:
            from awex.models.registry import get_infer_weights_converter

            model = self._get_model()
            self._weight_converter = get_infer_weights_converter(
                "sglang",
                type(model).__name__,
                hf_config=model.config,
                rank_info=rank_info,
                infer_engine_config=self._scheduler.server_args,
            )
        return self._weight_converter

    def _build_rank_info(self) -> RankInfo:
        model_context = self._get_model_context()
        return get_sglang_rank_info(model_context, engine_rank=0)

    def _build_sharding_strategy(self, rank_info: RankInfo):
        model = self._get_model()
        infer_engine_config = self._scheduler.server_args
        return get_sglang_sharding_strategy(
            type(model).__name__, infer_engine_config, rank_info
        )

    def get_weight_metadata(self) -> list[ParameterMeta]:
        rank_info = self._build_rank_info()
        strategy = self._build_sharding_strategy(rank_info)
        weight_converter = self._get_weight_converter(rank_info)
        self._rank_info = rank_info

        metadata: list[ParameterMeta] = []

        for name, param in self._get_model().named_parameters():
            for hf_name, local_tensor in weight_converter.convert_param(
                name, param.data
            ):
                local_shape = tuple(local_tensor.shape)
                sharding_type, sharding_dim, num_shards = (
                    strategy.get_sharding_strategy(hf_name)
                )

                global_offset = [0] * len(local_shape)
                if sharding_type == ShardingType.TP_SHARDING:
                    rank_pos = rank_info.tp_rank
                elif sharding_type == ShardingType.DP_TP_SHARDING:
                    rank_pos = rank_info.attn_tp_rank
                elif sharding_type == ShardingType.EP_SHARDING:
                    rank_pos = rank_info.ep_rank
                elif sharding_type == ShardingType.EP_TP_SHARDING:
                    rank_pos = rank_info.ep_tp_rank
                else:
                    rank_pos = 0

                if (
                    sharding_type != ShardingType.NO_SHARDING
                    and 0 <= sharding_dim < len(local_shape)
                ):
                    global_offset[sharding_dim] = int(rank_pos) * int(
                        local_shape[sharding_dim]
                    )

                global_shape = list(local_shape)
                if (
                    sharding_type != ShardingType.NO_SHARDING
                    and 0 <= sharding_dim < len(global_shape)
                ):
                    global_shape[sharding_dim] = int(local_shape[sharding_dim]) * int(
                        num_shards
                    )

                shard_meta = ParameterShardMeta(
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
                    name=hf_name,
                    shape=local_shape,
                    numel=int(local_tensor.numel()),
                    dtype=local_tensor.dtype,
                    global_offset=tuple(global_offset),
                    sharding_type=sharding_type,
                    num_shards=int(num_shards),
                    sharding_dim=int(sharding_dim),
                )

                replica = ParameterReplicaMeta(shards=[shard_meta])
                metadata.append(
                    ParameterMeta(
                        name=hf_name,
                        global_numel=math.prod(global_shape) if global_shape else 1,
                        global_shape=tuple(global_shape),
                        dtype=local_tensor.dtype,
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
        rank_info = self._rank_info or self._build_rank_info()
        weight_converter = self._get_weight_converter(rank_info)

        for name, param in self._get_model().named_parameters():
            for hf_name, hf_tensor in weight_converter.convert_param(name, param.data):
                if required is None or hf_name in required:
                    local_params[hf_name] = hf_tensor

        self._parameters = local_params
        return local_params

    def save_parameters(self, save_path: str, names: list[str] | None = None) -> None:
        params = self.get_local_shard_parameters(names)
        cpu_params = {k: v.detach().cpu().clone() for k, v in params.items()}
        torch.save(cpu_params, save_path)

    def randomize_parameters(self) -> None:
        for _, param in self._get_model().named_parameters():
            param.data.normal_()

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
        if self._dte_config.enabled:
            validate_dte_world_size(world_size, infer_world_size, train_world_size)

        per_engine_world = infer_world_size // num_engines
        ctx = self._get_model_context()
        tp_size = int(ctx["tp_size"])
        tp_rank = int(ctx["tp_rank"])
        pp_size = int(ctx["pp_size"])
        pp_rank = int(ctx["pp_rank"])
        if per_engine_world != tp_size * pp_size:
            raise RuntimeError(
                "awex per-engine world mismatch: gateway reports "
                f"infer_world_size={infer_world_size} / num_engines={num_engines} "
                f"= {per_engine_world}, but local engine has "
                f"tp_size*pp_size={tp_size * pp_size}"
            )

        engine_local_rank = pp_rank * tp_size + tp_rank
        global_rank = transfer_rank * per_engine_world + engine_local_rank
        self._transfer_rank = global_rank
        self._world_size = world_size

        infer_meta, train_meta = fetch_kv_metadata(kv_store_url, pair_name)

        builder = TransferPlanBuilder(
            infer_world_size=infer_world_size,
            train_world_size=train_world_size,
            num_infer_engines=num_engines,
        )
        self._transfer_plan = builder.build_local_transfer_plan(
            infer_meta, train_meta, global_transfer_rank=global_rank
        )

        os.environ["TORCHELASTIC_USE_AGENT_STORE"] = str(False)
        self._weights_update_group = init_weights_update_group(
            master_address=master_addr,
            master_port=master_port,
            rank=global_rank,
            world_size=world_size,
            group_name=f"awex_{pair_name}",
            role="inference",
        )
        self._weights_update_group_gloo = init_weights_update_group(
            master_address=master_addr,
            master_port=master_port,
            rank=global_rank,
            world_size=world_size,
            group_name=f"awex_{pair_name}_gloo",
            backend="gloo",
            role="inference",
        )
        if self._dte_config.enabled:
            self._separation_wire_dtypes = synchronize_wire_dtypes(
                self._transfer_plan,
                self._weights_update_group_gloo,
            )
        logger.info(
            "Initialized AWEX weight update groups for pair=%s role=inference "
            "rank=%s world_size=%s nccl=awex_%s gloo=awex_%s_gloo",
            pair_name,
            global_rank,
            world_size,
            pair_name,
            pair_name,
        )

    def execute_weight_update(self, version: int) -> None:
        if self._dte_config.enabled:
            self._execute_separation_weight_update(version)
            return

        if self._transfer_plan is None:
            raise RuntimeError("Transfer plan is not initialized")
        if self._weights_update_group is None:
            raise RuntimeError("Weight update group is not initialized")
        if self._weights_update_group_gloo is None:
            raise RuntimeError("Gloo weight update group is not initialized")

        params = self.get_local_shard_parameters()
        recv_ops, non_contiguous_pairs, _ = nccl_build_recv_ops(
            params,
            self._transfer_plan,
            self._weights_update_group,
        )
        batch_send_recv(
            send_ops=[],
            recv_ops=recv_ops,
            blocking=True,
            use_group=awex_wu_use_group(),
        )

        for original, contiguous in non_contiguous_pairs:
            original.copy_(contiguous)

        current_platform.synchronize()
        dist.barrier(group=self._weights_update_group_gloo)

    def _execute_separation_weight_update(self, version: int) -> None:
        """Receive either a sparse AdamW update or its dense fallback."""
        if self._transfer_plan is None:
            raise RuntimeError("Transfer plan is not initialized")
        if self._weights_update_group is None:
            raise RuntimeError("Weight update group is not initialized")
        if self._weights_update_group_gloo is None:
            raise RuntimeError("Gloo weight update group is not initialized")

        decision = torch.tensor([1], dtype=torch.int64)
        dist.all_reduce(
            decision, op=dist.ReduceOp.MIN, group=self._weights_update_group_gloo
        )
        use_delta = bool(decision.item())
        params = self.get_local_shard_parameters()

        if use_delta:
            self._execute_separation_delta_recv(params, version)
        else:
            recv_ops, non_contiguous_pairs, _ = nccl_build_recv_ops(
                params,
                self._transfer_plan,
                self._weights_update_group,
            )
            batch_send_recv(
                send_ops=[],
                recv_ops=recv_ops,
                blocking=True,
                use_group=awex_wu_use_group(),
            )
            for original, contiguous in non_contiguous_pairs:
                original.copy_(contiguous)

        current_platform.synchronize()
        dist.barrier(group=self._weights_update_group_gloo)

    def _execute_separation_delta_recv(
        self,
        recv_params: dict[str, torch.Tensor],
        version: int,
    ) -> None:
        from dte.core.colocate_protocol import (
            _filter_plan_by_dtype,
            _ops_by_recv_dtype,
            _PlanView,
            two_round_delta_exchange,
        )

        if self._transfer_plan is None:
            raise RuntimeError("Transfer plan is not initialized")
        if self._weights_update_group is None:
            raise RuntimeError("Weight update group is not initialized")
        if self._transfer_rank is None or self._world_size is None:
            raise RuntimeError("Transfer rank/world size is not initialized")
        if self._separation_wire_dtypes is None:
            raise RuntimeError("Separation DTE wire dtypes are not initialized")

        operations = [
            op for ops in self._transfer_plan.operations.values() for op in ops
        ]
        operations_by_dtype = _ops_by_recv_dtype(operations)
        identity_mapping = {rank: rank for rank in range(self._world_size)}
        empty_plan = _PlanView({})
        device = torch.device(f"cuda:{torch.cuda.current_device()}")

        if self._separation_delta_transport is None:
            self._separation_delta_transport = NcclColocateStreamBatchTransport(
                self._transfer_rank, self._world_size
            )
        schedule_fn = (
            self._separation_delta_transport.execute_recursive_partition_stream_transfer
        )

        operation_count = 0
        for dtype in self._separation_wire_dtypes:
            ops = operations_by_dtype.get(dtype, [])
            recv_plan = _filter_plan_by_dtype(self._transfer_plan, dtype, is_send=False)
            two_round_delta_exchange(
                transfer_rank=self._transfer_rank,
                world_size=self._world_size,
                send_plan=empty_plan,
                recv_plan=recv_plan,
                train_to_infer_device_mapping=identity_mapping,
                weights_update_group=self._weights_update_group,
                send_payloads_by_op={},
                recv_params=recv_params,
                value_dtype=dtype,
                device=device,
                schedule_fn=schedule_fn,
                slice_fn=slice_tensor,
                rank_coordinate=f"infer-{self._transfer_rank}",
                step_id=version,
            )
            operation_count += len(ops)

        logger.info(
            "separation delta v%d received %d ops across %d dtypes",
            version,
            operation_count,
            len(self._separation_wire_dtypes),
        )

    def batch_isend_irecv(self, **kwargs) -> None:
        if self._weights_update_group_gloo is None:
            raise RuntimeError("Gloo weight update group is not initialized")
        setup_kwargs = {
            k: v for k, v in kwargs.items() if k not in ("world_size", "barrier_group")
        }
        setup_batch_isend_irecv(
            self._weights_update_group,
            self._transfer_rank,
            kwargs.get("world_size", 0),
            barrier_group=self._weights_update_group_gloo,
            **setup_kwargs,
        )

    def teardown_weight_update_group(self) -> None:
        if self._weights_update_group is not None and dist.is_initialized():
            dist.destroy_process_group(self._weights_update_group)
        if self._weights_update_group_gloo is not None and dist.is_initialized():
            dist.destroy_process_group(self._weights_update_group_gloo)
        self._weights_update_group = None
        self._weights_update_group_gloo = None
        self._transfer_plan = None
        self._transfer_rank = None
        self._world_size = None
        self._separation_delta_transport = None
        self._separation_wire_dtypes = None
        self._rank_info = None
        self._weight_converter = None
        self._parameters = None
        if self._colocate_http_client is not None:
            self._colocate_http_client.close()
            self._colocate_http_client = None
        self._colocate_transport = None
        self._train_to_infer_device_mapping = None
        self._infer_to_train_device_mapping = None

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
        if infer_world_size != train_world_size:
            raise ValueError(
                f"Colocate mode requires infer_world_size == train_world_size. "
                f"Got infer_world_size={infer_world_size}, "
                f"train_world_size={train_world_size}"
            )
        self._colocate_pair_name = pair_name
        self._colocate_kv_store_url = kv_store_url
        self._transfer_rank = transfer_rank
        self._colocate_infer_world_size = infer_world_size
        self._colocate_train_world_size = train_world_size
        self._colocate_admin_api_key = admin_api_key
        self._colocate_timeout_s = timeout_s
        if self._colocate_http_client is None:
            self._colocate_http_client = httpx.Client()

        infer_meta, train_meta = fetch_kv_metadata(kv_store_url, pair_name)

        builder = TransferPlanBuilder(
            infer_world_size=infer_world_size,
            train_world_size=train_world_size,
            num_infer_engines=num_engines,
        )

        train_to_infer = {}
        infer_to_train = {}
        for i in range(min(infer_world_size, train_world_size)):
            train_rank = infer_world_size + i
            train_to_infer[train_rank] = i
            infer_to_train[i] = train_rank
        self._train_to_infer_device_mapping = train_to_infer
        self._infer_to_train_device_mapping = infer_to_train

        self._send_transfer_plan = builder.build_local_transfer_plan(
            infer_meta,
            train_meta,
            global_transfer_rank=infer_to_train[transfer_rank],
        )
        self._recv_transfer_plan = builder.build_local_transfer_plan(
            infer_meta,
            train_meta,
            global_transfer_rank=transfer_rank,
        )

        os.environ["TORCHELASTIC_USE_AGENT_STORE"] = str(False)
        self._weights_update_group = init_weights_update_group(
            master_address="127.0.0.1",
            master_port=master_port,
            rank=transfer_rank,
            world_size=infer_world_size,
            group_name=f"awex_colocate_{pair_name}",
            role="inference",
        )

        self._colocate_transport = NcclColocateStreamBatchTransport(
            transfer_rank, infer_world_size
        )

        logger.info(
            "Initialized colocate weight update for pair '%s', "
            "transfer_rank=%d, infer_world_size=%d",
            pair_name,
            transfer_rank,
            infer_world_size,
        )

    def execute_colocate_weight_update(self, version: int) -> None:
        kv_store_url = self._colocate_kv_store_url
        pair_name = self._colocate_pair_name
        transfer_rank = self._transfer_rank
        assert self._colocate_http_client is not None, (
            "init_colocate_weight_update must be called first"
        )
        assert self._infer_to_train_device_mapping is not None
        client = self._colocate_http_client
        auth_headers = {"Authorization": f"Bearer {self._colocate_admin_api_key}"}
        timeout_s = self._colocate_timeout_s

        paired_train_rank = self._infer_to_train_device_mapping[transfer_rank]
        kv_key = f"colocate_weights_rank{paired_train_rank}_{version}"

        deadline = time.monotonic() + timeout_s
        serialized_hex = None
        poll_count = 0
        last_status = -1
        while time.monotonic() < deadline:
            resp = client.get(
                f"{kv_store_url}/weight_meta/{pair_name}/{kv_key}",
                timeout=5.0,
            )
            last_status = resp.status_code
            if resp.status_code == 200:
                serialized_hex = resp.json()["value"]
                break
            poll_count += 1
            time.sleep(0.1)
        if serialized_hex is None:
            raise TimeoutError(
                f"Training did not put colocate weights within {timeout_s}s "
                f"(waiting_key={kv_key}, polls={poll_count}, "
                f"last_status={last_status})"
            )

        serialized_weights = bytes.fromhex(serialized_hex)
        group_shared, metadata, names = cuda_ipc_deserialize(serialized_weights)
        torch.cuda.synchronize()
        tensors = reconstruct_tensors_from_groups(group_shared, metadata)
        torch.cuda.synchronize()
        deserialized_weights = dict(zip(names, tensors))

        recv_parameters = self.get_local_shard_parameters()

        rank_info = self._build_rank_info()
        rank_coordinate = f"infer_{rank_info.global_rank}"

        assert self._colocate_transport is not None
        self._colocate_transport.update_weights_in_colocate_mode(
            self._train_to_infer_device_mapping,
            self._infer_to_train_device_mapping,
            transfer_rank,
            rank_coordinate,
            self._colocate_infer_world_size,
            self._send_transfer_plan,
            self._recv_transfer_plan,
            self._weights_update_group,
            deserialized_weights,
            recv_parameters,
            step_id=version,
        )

        done_key = f"colocate_done_rank{paired_train_rank}_{version}"
        client.put(
            f"{kv_store_url}/weight_meta/{pair_name}/{done_key}",
            json={"value": True},
            headers=auth_headers,
            timeout=10.0,
        )

        del deserialized_weights, group_shared, tensors, serialized_weights
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        logger.info(
            "Colocate weight update completed for v%d, rank %d",
            version,
            transfer_rank,
        )

    # Tags understood by SGLang's native release/resume_memory_occupation.
    _SGLANG_MEMORY_TAGS = {"kv_cache"}

    def release_memory(self, tags: list[str] | None = None) -> None:
        from sglang.srt.managers.io_struct import ReleaseMemoryOccupationReqInput

        native_tags = (
            [t for t in tags if t in self._SGLANG_MEMORY_TAGS] if tags else None
        )
        unsupported = (
            [t for t in tags if t not in self._SGLANG_MEMORY_TAGS] if tags else []
        )
        if unsupported:
            logger.warning(
                "release_memory: tags %s not supported by SGLang adapter "
                "(supported: %s), ignoring",
                unsupported,
                self._SGLANG_MEMORY_TAGS,
            )
        if native_tags:
            req = ReleaseMemoryOccupationReqInput(tags=native_tags)
            self._scheduler.release_memory_occupation(req)
            self._released_tags.update(native_tags)
        logger.info("release_memory completed with tags=%s", tags)

    def resume_memory(self, tags: list[str] | None = None) -> None:
        from sglang.srt.managers.io_struct import ResumeMemoryOccupationReqInput

        native_tags = (
            [
                t
                for t in tags
                if t in self._SGLANG_MEMORY_TAGS and t in self._released_tags
            ]
            if tags
            else None
        )
        unsupported = (
            [t for t in tags if t not in self._SGLANG_MEMORY_TAGS] if tags else []
        )
        if unsupported:
            logger.warning(
                "resume_memory: tags %s not supported by SGLang adapter "
                "(supported: %s), ignoring",
                unsupported,
                self._SGLANG_MEMORY_TAGS,
            )
        if native_tags:
            req = ResumeMemoryOccupationReqInput(tags=native_tags)
            self._scheduler.resume_memory_occupation(req)
            self._released_tags.difference_update(native_tags)
        logger.info("resume_memory completed with tags=%s", tags)
