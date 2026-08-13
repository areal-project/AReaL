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
from awex.transfer.transfer_plan import TransferPlanBuilder
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
from areal.v2.weight_update.awex.state import (
    AwexPairState,
    SGLangColocatePairState,
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
        self._pair_states: dict[str, AwexPairState] = {}
        self._colocate_pair_states: dict[str, SGLangColocatePairState] = {}
        self._rank_info: RankInfo | None = None
        self._parameters: dict[str, torch.Tensor] | None = None
        self._released_tags: set[str] = set()

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

    def _unfuse_params(
        self, name: str, tensor: torch.Tensor
    ) -> list[tuple[str, torch.Tensor]]:
        """Split SGLang fused parameters into HuggingFace-style unfused pairs.

        SGLang fuses Q/K/V into ``qkv_proj`` and gate/up into ``gate_up_proj``
        for efficiency.  For MoE models, SGLang also fuses all routed experts
        into ``experts.w13_weight`` (gate+up) and ``experts.w2_weight`` (down).
        The training side keeps per-expert HF names, so we unfuse here to match.
        """
        if "qkv_proj" in name:
            cfg = self._get_model().config
            num_heads = cfg.num_attention_heads
            num_kv_heads = getattr(cfg, "num_key_value_heads", num_heads)
            total_head_units = num_heads + 2 * num_kv_heads
            dim0 = tensor.shape[0]
            q_size = dim0 * num_heads // total_head_units
            kv_size = dim0 * num_kv_heads // total_head_units
            return [
                (name.replace("qkv_proj", "q_proj"), tensor.narrow(0, 0, q_size)),
                (
                    name.replace("qkv_proj", "k_proj"),
                    tensor.narrow(0, q_size, kv_size),
                ),
                (
                    name.replace("qkv_proj", "v_proj"),
                    tensor.narrow(0, q_size + kv_size, kv_size),
                ),
            ]
        if "gate_up_proj" in name:
            half = tensor.shape[0] // 2
            return [
                (name.replace("gate_up_proj", "gate_proj"), tensor.narrow(0, 0, half)),
                (name.replace("gate_up_proj", "up_proj"), tensor.narrow(0, half, half)),
            ]
        if "shared_experts" in name and "gate_up_weight" in name:
            half = tensor.shape[0] // 2
            return [
                (
                    name.replace("gate_up_weight", "gate_proj.weight"),
                    tensor.narrow(0, 0, half),
                ),
                (
                    name.replace("gate_up_weight", "up_proj.weight"),
                    tensor.narrow(0, half, half),
                ),
            ]
        if "shared_experts" in name and name.endswith("down_weight"):
            return [(name.replace("down_weight", "down_proj.weight"), tensor)]
        if ".experts.w13_weight" in name:
            # w13_weight shape: [num_total_experts, 2*ffn_hidden, hidden]
            # num_total_experts may include shared experts appended after
            # routed experts (e.g. 128 routed + 1 shared = 129 total).
            cfg = self._get_model().config
            num_routed = getattr(cfg, "num_experts", None) or cfg.n_routed_experts
            prefix = name.replace(".w13_weight", "")
            result = []
            ffn_hidden = tensor.shape[1] // 2
            for i in range(tensor.shape[0]):
                expert_tensor = tensor[i]
                if i < num_routed:
                    expert_prefix = f"{prefix}.{i}"
                else:
                    shared_idx = i - num_routed
                    num_shared = tensor.shape[0] - num_routed
                    if num_shared > 1:
                        expert_prefix = prefix.replace(
                            "experts", f"shared_experts.{shared_idx}"
                        )
                    else:
                        expert_prefix = prefix.replace("experts", "shared_experts")
                result.append(
                    (f"{expert_prefix}.gate_proj.weight", expert_tensor[:ffn_hidden])
                )
                result.append(
                    (f"{expert_prefix}.up_proj.weight", expert_tensor[ffn_hidden:])
                )
            return result
        if ".experts.w2_weight" in name:
            # w2_weight shape: [num_total_experts, hidden, ffn_hidden]
            cfg = self._get_model().config
            num_routed = getattr(cfg, "num_experts", None) or cfg.n_routed_experts
            prefix = name.replace(".w2_weight", "")
            result = []
            for i in range(tensor.shape[0]):
                if i < num_routed:
                    expert_prefix = f"{prefix}.{i}"
                else:
                    shared_idx = i - num_routed
                    num_shared = tensor.shape[0] - num_routed
                    if num_shared > 1:
                        expert_prefix = prefix.replace(
                            "experts", f"shared_experts.{shared_idx}"
                        )
                    else:
                        expert_prefix = prefix.replace("experts", "shared_experts")
                result.append((f"{expert_prefix}.down_proj.weight", tensor[i]))
            return result
        return [(name, tensor)]

    def _build_rank_info(self) -> RankInfo:
        model_context = self._get_model_context()
        return get_sglang_rank_info(model_context, engine_rank=0)

    def _build_sharding_strategy(self, rank_info: RankInfo):
        model = self._get_model()
        model_name = None
        model_config = getattr(model, "config", None)
        if model_config is not None:
            architectures = getattr(model_config, "architectures", None)
            if architectures and len(architectures) > 0:
                model_name = architectures[0]

        if model_name is None:
            model_name = type(model).__name__

        infer_engine_config = self._scheduler.server_args
        return get_sglang_sharding_strategy(model_name, infer_engine_config, rank_info)

    def get_weight_metadata(self) -> list[ParameterMeta]:
        rank_info = self._build_rank_info()
        strategy = self._build_sharding_strategy(rank_info)
        self._rank_info = rank_info

        metadata: list[ParameterMeta] = []

        for name, param in self._get_model().named_parameters():
            for hf_name, local_tensor in self._unfuse_params(name, param.data):
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

        for name, param in self._get_model().named_parameters():
            for hf_name, hf_tensor in self._unfuse_params(name, param.data):
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
        existing_state = self._pair_states.get(pair_name)
        if existing_state is not None:
            if (
                existing_state.weights_update_group is not None
                and existing_state.control_group is not None
            ):
                logger.info("AWEX pair '%s' is already initialized", pair_name)
                return
            raise RuntimeError(
                f"AWEX pair {pair_name!r} is only partially initialized; "
                "teardown must complete before retrying initialization"
            )
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
        infer_meta, train_meta = fetch_kv_metadata(kv_store_url, pair_name)

        builder = TransferPlanBuilder(
            infer_world_size=infer_world_size,
            train_world_size=train_world_size,
            num_infer_engines=num_engines,
        )
        transfer_plan = builder.build_local_transfer_plan(
            infer_meta, train_meta, global_transfer_rank=global_rank
        )

        os.environ["TORCHELASTIC_USE_AGENT_STORE"] = str(False)
        weights_update_group = None
        control_group = None
        try:
            weights_update_group = init_weights_update_group(
                master_address=master_addr,
                master_port=master_port,
                rank=global_rank,
                world_size=world_size,
                group_name=f"awex_{pair_name}",
                role="inference",
            )
            control_group = init_weights_update_group(
                master_address=master_addr,
                master_port=master_port,
                rank=global_rank,
                world_size=world_size,
                group_name=f"awex_{pair_name}_gloo",
                backend="gloo",
                role="inference",
            )
        except BaseException:
            if control_group is not None and dist.is_initialized():
                try:
                    dist.destroy_process_group(control_group)
                except Exception:
                    logger.warning(
                        "Failed to rollback control group for AWEX pair '%s'",
                        pair_name,
                        exc_info=True,
                    )
                else:
                    control_group = None
            if weights_update_group is not None and dist.is_initialized():
                try:
                    dist.destroy_process_group(weights_update_group)
                except Exception:
                    logger.warning(
                        "Failed to rollback payload group for AWEX pair '%s'",
                        pair_name,
                        exc_info=True,
                    )
                else:
                    weights_update_group = None
            if weights_update_group is not None or control_group is not None:
                self._pair_states[pair_name] = AwexPairState(
                    weights_update_group=weights_update_group,
                    control_group=control_group,
                    transfer_plan=transfer_plan,
                    transfer_rank=global_rank,
                )
            raise

        self._pair_states[pair_name] = AwexPairState(
            weights_update_group=weights_update_group,
            control_group=control_group,
            transfer_plan=transfer_plan,
            transfer_rank=global_rank,
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

    def execute_weight_update(self, pair_name: str, version: int) -> None:
        del version
        state = self._require_pair_state(pair_name)

        params = self.get_local_shard_parameters()
        recv_ops, non_contiguous_pairs, _ = nccl_build_recv_ops(
            params,
            state.transfer_plan,
            state.weights_update_group,
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
        dist.barrier(group=state.control_group)

    def batch_isend_irecv(self, pair_name: str, **kwargs) -> None:
        state = self._require_pair_state(pair_name)
        setup_kwargs = {
            k: v for k, v in kwargs.items() if k not in ("world_size", "barrier_group")
        }
        setup_batch_isend_irecv(
            state.weights_update_group,
            state.transfer_rank,
            kwargs.get("world_size", 0),
            barrier_group=state.control_group,
            **setup_kwargs,
        )

    def teardown_weight_update_group(self, pair_name: str) -> None:
        state = self._pair_states.get(pair_name)
        errors: list[Exception] = []
        if state is not None:
            if dist.is_initialized():
                for attr in ("weights_update_group", "control_group"):
                    group = getattr(state, attr)
                    if group is None:
                        continue
                    try:
                        dist.destroy_process_group(group)
                    except Exception as exc:
                        logger.warning(
                            "Failed to teardown %s for AWEX pair '%s'",
                            attr,
                            pair_name,
                            exc_info=True,
                        )
                        errors.append(exc)
                    else:
                        setattr(state, attr, None)
            else:
                state.weights_update_group = None
                state.control_group = None
            if state.weights_update_group is None and state.control_group is None:
                self._pair_states.pop(pair_name, None)

        colocate_state = self._colocate_pair_states.get(pair_name)
        if colocate_state is not None:
            if colocate_state.weights_update_group is not None:
                if dist.is_initialized():
                    try:
                        dist.destroy_process_group(colocate_state.weights_update_group)
                    except Exception as exc:
                        logger.warning(
                            "Failed to teardown colocate group for AWEX pair '%s'",
                            pair_name,
                            exc_info=True,
                        )
                        errors.append(exc)
                    else:
                        colocate_state.weights_update_group = None
                else:
                    colocate_state.weights_update_group = None
            if colocate_state.http_client is not None:
                try:
                    colocate_state.http_client.close()
                except Exception as exc:
                    logger.warning(
                        "Failed to close colocate client for AWEX pair '%s'",
                        pair_name,
                        exc_info=True,
                    )
                    errors.append(exc)
                else:
                    colocate_state.http_client = None
            if (
                colocate_state.weights_update_group is None
                and colocate_state.http_client is None
            ):
                self._colocate_pair_states.pop(pair_name, None)

        self._rank_info = None
        self._parameters = None

        if errors:
            raise RuntimeError(
                f"Failed to teardown AWEX pair {pair_name!r} on this rank"
            ) from errors[0]

    def _require_pair_state(self, pair_name: str) -> AwexPairState:
        state = self._pair_states.get(pair_name)
        if state is None:
            raise RuntimeError(f"AWEX pair {pair_name!r} is not initialized")
        if state.weights_update_group is None or state.control_group is None:
            raise RuntimeError(f"AWEX pair {pair_name!r} is partially torn down")
        return state

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
        if pair_name in self._colocate_pair_states:
            logger.info("AWEX colocate pair '%s' is already initialized", pair_name)
            return
        if infer_world_size != train_world_size:
            raise ValueError(
                f"Colocate mode requires infer_world_size == train_world_size. "
                f"Got infer_world_size={infer_world_size}, "
                f"train_world_size={train_world_size}"
            )
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
        send_transfer_plan = builder.build_local_transfer_plan(
            infer_meta,
            train_meta,
            global_transfer_rank=infer_to_train[transfer_rank],
        )
        recv_transfer_plan = builder.build_local_transfer_plan(
            infer_meta,
            train_meta,
            global_transfer_rank=transfer_rank,
        )

        os.environ["TORCHELASTIC_USE_AGENT_STORE"] = str(False)
        weights_update_group = None
        http_client = httpx.Client()
        try:
            weights_update_group = init_weights_update_group(
                master_address="127.0.0.1",
                master_port=master_port,
                rank=transfer_rank,
                world_size=infer_world_size,
                group_name=f"awex_colocate_{pair_name}",
                role="inference",
            )

            transport = NcclColocateStreamBatchTransport(
                transfer_rank, infer_world_size
            )
            self._colocate_pair_states[pair_name] = SGLangColocatePairState(
                weights_update_group=weights_update_group,
                transfer_rank=transfer_rank,
                kv_store_url=kv_store_url,
                infer_world_size=infer_world_size,
                train_world_size=train_world_size,
                admin_api_key=admin_api_key,
                timeout_s=timeout_s,
                http_client=http_client,
                transport=transport,
                train_to_infer_device_mapping=train_to_infer,
                infer_to_train_device_mapping=infer_to_train,
                send_transfer_plan=send_transfer_plan,
                recv_transfer_plan=recv_transfer_plan,
            )
        except BaseException:
            if weights_update_group is not None and dist.is_initialized():
                dist.destroy_process_group(weights_update_group)
            http_client.close()
            raise

        logger.info(
            "Initialized colocate weight update for pair '%s', "
            "transfer_rank=%d, infer_world_size=%d",
            pair_name,
            transfer_rank,
            infer_world_size,
        )

    def execute_colocate_weight_update(self, pair_name: str, version: int) -> None:
        state = self._colocate_pair_states.get(pair_name)
        if state is None:
            raise RuntimeError(f"AWEX colocate pair {pair_name!r} is not initialized")
        kv_store_url = state.kv_store_url
        transfer_rank = state.transfer_rank
        client = state.http_client
        auth_headers = {"Authorization": f"Bearer {state.admin_api_key}"}
        timeout_s = state.timeout_s

        paired_train_rank = state.infer_to_train_device_mapping[transfer_rank]
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

        state.transport.update_weights_in_colocate_mode(
            state.train_to_infer_device_mapping,
            state.infer_to_train_device_mapping,
            transfer_rank,
            rank_coordinate,
            state.infer_world_size,
            state.send_transfer_plan,
            state.recv_transfer_plan,
            state.weights_update_group,
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
