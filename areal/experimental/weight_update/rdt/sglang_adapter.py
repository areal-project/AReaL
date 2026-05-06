# SPDX-License-Identifier: Apache-2.0
"""RDT SGLang Adapter for IW weight update."""

from __future__ import annotations

import math
import os
from typing import Any

import ray
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
from awex.transfer.transfer_plan import TransferPlan, TransferPlanBuilder

from areal.experimental.weight_update.rdt import (
    deserialize_actor_handle_bytes,
    fetch_kv_metadata,
    get_tensor_transport,
)
from areal.utils import logging

logger = logging.getLogger("RDTSGLangAdapter")


class RDTSGLangAdapter:
    """RDT inference adapter for in-process SGLang schedulers.

    Handles one-sided RDMA weight pull via Ray RPC from TW actors.
    """

    def __init__(self, scheduler: Any) -> None:
        self._scheduler = scheduler
        self._tw_handles: dict[str, list] = {}  # pair_name -> TW actor handles
        self._transfer_plans: dict[str, TransferPlan] = {}  # pair_name -> plan
        self._infer_world_sizes: dict[str, int] = {}  # pair_name -> infer_world_size
        self._tensor_transport: str | None = None
        self._ray_initialized: bool = False
        self._rank_info: RankInfo | None = None

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

    def _get_expert_prefix(
        self, prefix: str, expert_idx: int, num_routed: int, total_experts: int
    ) -> str:
        if expert_idx < num_routed:
            return f"{prefix}.{expert_idx}"

        shared_idx = expert_idx - num_routed
        num_shared = total_experts - num_routed
        if num_shared > 1:
            return prefix.replace("experts", f"shared_experts.{shared_idx}")
        return prefix.replace("experts", "shared_experts")

    def _unfuse_params(
        self, name: str, tensor: torch.Tensor
    ) -> list[tuple[str, torch.Tensor]]:
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
            cfg = self._get_model().config
            num_routed = getattr(cfg, "num_experts", None) or cfg.n_routed_experts
            prefix = name.replace(".w13_weight", "")
            result = []
            ffn_hidden = tensor.shape[1] // 2
            for i in range(tensor.shape[0]):
                expert_tensor = tensor[i]
                expert_prefix = self._get_expert_prefix(
                    prefix, i, num_routed, tensor.shape[0]
                )
                result.append(
                    (f"{expert_prefix}.gate_proj.weight", expert_tensor[:ffn_hidden])
                )
                result.append(
                    (f"{expert_prefix}.up_proj.weight", expert_tensor[ffn_hidden:])
                )
            return result
        if ".experts.w2_weight" in name:
            cfg = self._get_model().config
            num_routed = getattr(cfg, "num_experts", None) or cfg.n_routed_experts
            prefix = name.replace(".w2_weight", "")
            result = []
            for i in range(tensor.shape[0]):
                expert_prefix = self._get_expert_prefix(
                    prefix, i, num_routed, tensor.shape[0]
                )
                result.append((f"{expert_prefix}.down_proj.weight", tensor[i]))
            return result
        return [(name, tensor)]

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

        return local_params

    def save_parameters(self, save_path: str, names: list[str] | None = None) -> None:
        params = self.get_local_shard_parameters(names)
        cpu_params = {k: v.detach().cpu().clone() for k, v in params.items()}
        torch.save(cpu_params, save_path)

    def randomize_parameters(self) -> None:
        for _, param in self._get_model().named_parameters():
            param.data.normal_()

    # ---------------------------------------------------------------------------
    # RDT-specific methods: Ray init, TW handle storage, weight pull
    # ---------------------------------------------------------------------------

    def _ensure_ray_init(self) -> None:
        if self._ray_initialized:
            return

        if not ray.is_initialized():
            ray.init(address="auto")

        self._ray_initialized = True

        transport = get_tensor_transport()
        if transport == "YR":
            try:
                from ray_ascend import register_yr_tensor_transport

                register_yr_tensor_transport(["npu", "cpu"])
                logger.info("Registered YR tensor transport for NPU")
            except ImportError:
                logger.warning("ray_ascend not available, YR transport may not work")

        self._tensor_transport = transport

    def rdt_init_weight_update_group(
        self,
        pair_name: str,
        kv_store_url: str,
        tw_actor_bytes_b64_list: list[str],
        infer_world_size: int,
        train_world_size: int,
        num_engines: int,
        transfer_rank: int,
    ) -> None:
        import time

        self._ensure_ray_init()

        tw_handles = [
            deserialize_actor_handle_bytes(b64_bytes)
            for b64_bytes in tw_actor_bytes_b64_list
        ]
        self._tw_handles[pair_name] = tw_handles
        logger.info(
            f"RDT init: Stored {len(tw_handles)} TW handles for pair '{pair_name}'"
        )

        infer_meta, train_meta = fetch_kv_metadata(kv_store_url, pair_name)

        builder = TransferPlanBuilder(
            infer_world_size=infer_world_size,
            train_world_size=train_world_size,
            num_infer_engines=num_engines,
        )
        self._transfer_plans[pair_name] = builder.build_local_transfer_plan(
            infer_meta, train_meta, global_transfer_rank=transfer_rank
        )
        self._infer_world_sizes[pair_name] = infer_world_size
        logger.info(
            f"RDT init: Built TransferPlan for pair '{pair_name}' transfer_rank={transfer_rank} "
            f"infer_world_size={infer_world_size}"
        )

        # Warmup NIXL agents: call warmup on each TW handle
        # This triggers NIXL agent initialization on IW (driver) and TW Actor sides
        # Moving ~9s overhead from update_weights to connect phase
        transport = self._tensor_transport or get_tensor_transport()
        if transport == "NIXL":
            warmup_refs = []
            for handle in tw_handles:
                warmup_refs.append(handle.warmup_nixl.remote())
            t0 = time.monotonic()
            ray.get(warmup_refs)
            t1 = time.monotonic()
            logger.info(
                f"[RDT-Warmup] NIXL agent warmup completed in {1000 * (t1 - t0):.1f}ms "
                f"for {len(tw_handles)} TW handles"
            )

    def rdt_execute_weight_update(self, version: int = 0) -> None:
        import time

        t0 = time.monotonic()
        pair_name = self._get_current_pair_name()
        if pair_name not in self._tw_handles:
            raise RuntimeError(f"TW handles not initialized for pair '{pair_name}'")

        tw_handles = self._tw_handles[pair_name]
        plan = self._transfer_plans[pair_name]

        infer_world_size = self._infer_world_sizes.get(pair_name, 1)

        required_indices = self._get_required_tw_indices(plan, infer_world_size)
        required_handles = [tw_handles[i] for i in required_indices]

        # IW's own global rank (infer_rank)
        infer_rank = self._rank_info.global_rank if self._rank_info else 0

        t1 = time.monotonic()
        logger.info(
            f"[RDT-IW] Pulling from TW shards {required_indices} for pair '{pair_name}' v{version}"
        )

        transport = self._tensor_transport or get_tensor_transport()
        if transport == "YR":
            raise RuntimeError("YR backend not implemented yet")
        elif transport == "NIXL":
            method_name = "get_weights_tensor_nixl"
        else:
            raise RuntimeError(f"Unsupported tensor transport: {transport}")

        # IW only passes infer_rank; TW uses TransferPlan to determine what to send
        refs = []
        for handle in required_handles:
            refs.append(
                handle.__getattr__(method_name).remote(pair_name, infer_rank, version)
            )

        t2 = time.monotonic()
        logger.info(f"[RDT-IW] Submitted {len(refs)} RPCs, calling ray.get...")
        raw_results = ray.get(refs)
        t3 = time.monotonic()

        # Unpack merged buffers back to tensor dicts
        shard_tensor_by_rank: dict[int, dict[str, torch.Tensor]] = {}
        total_bytes = 0
        for idx, (result, send_rank) in enumerate(
            zip(raw_results, list(plan.inter_operations.keys()))
        ):
            buffer = result["buffer"]
            metadata = result["metadata"]
            names = metadata["names"]
            offsets = metadata["offsets"]
            shapes = metadata["shapes"]
            dtypes = metadata["dtypes"]

            # Split buffer back into individual tensors
            tensor_dict = {}
            for i, name in enumerate(names):
                offset = offsets[i]
                shape = shapes[i]
                dtype_str = dtypes[i]

                # Calculate numel from shape
                numel = 1
                for s in shape:
                    numel *= s

                # Extract tensor slice from buffer
                if i + 1 < len(offsets):
                    next_offset = offsets[i + 1]
                else:
                    next_offset = buffer.numel()

                flat_slice = buffer[offset:next_offset]
                tensor = flat_slice.reshape(shape)

                # Cast to correct dtype if needed
                dtype_map = {
                    "torch.bfloat16": torch.bfloat16,
                    "torch.float16": torch.float16,
                    "torch.float32": torch.float32,
                    "torch.int32": torch.int32,
                    "torch.int64": torch.int64,
                }
                target_dtype = dtype_map.get(dtype_str)
                if target_dtype and tensor.dtype != target_dtype:
                    tensor = tensor.to(target_dtype)

                tensor_dict[name] = tensor
                total_bytes += numel * tensor.element_size()

            shard_tensor_by_rank[send_rank] = tensor_dict

        t4 = time.monotonic()
        logger.info(
            f"[RDT-IW] Unpacked {len(raw_results)} buffers, "
            f"total tensors={sum(len(d) for d in shard_tensor_by_rank.values())}, "
            f"total_bytes={total_bytes / 1024 / 1024:.1f}MB, "
            f"unpack_time={1000 * (t4 - t3):.1f}ms"
        )

        self._apply_transfer_plan_to_model(plan, shard_tensor_by_rank, infer_world_size)
        t5 = time.monotonic()

        # Cleanup IPC handles after transfer
        for handle in required_handles:
            ray.get(handle.clear_ipc_handles.remote(pair_name, infer_rank, version))
        t6 = time.monotonic()

        logger.info(
            f"[RDT-IW-Timing] prep={1000 * (t1 - t0):.1f}ms | "
            f"rpc_submit={1000 * (t2 - t1):.1f}ms | ray_get={1000 * (t3 - t2):.1f}ms | "
            f"unpack={1000 * (t4 - t3):.1f}ms | apply_model={1000 * (t5 - t4):.1f}ms | "
            f"cleanup={1000 * (t6 - t5):.1f}ms | total={1000 * (t6 - t0):.1f}ms"
        )

    def _get_current_pair_name(self) -> str:
        if len(self._tw_handles) == 0:
            raise RuntimeError("No TW handles initialized")
        if len(self._tw_handles) == 1:
            return next(iter(self._tw_handles.keys()))
        raise RuntimeError("Multi-pair scenario requires pair_name specification")

    def _get_required_tw_indices(
        self, plan: TransferPlan, infer_world_size: int
    ) -> list[int]:
        send_ranks = list(plan.inter_operations.keys())
        tw_indices = [r - infer_world_size for r in send_ranks]

        logger.info(
            f"TransferPlan: send_ranks={send_ranks}, "
            f"tw_indices={tw_indices}, infer_world_size={infer_world_size}"
        )

        return tw_indices

    def _apply_transfer_plan_to_model(
        self,
        plan: TransferPlan,
        shard_tensors_by_rank: dict[int, dict[str, torch.Tensor]],
        infer_world_size: int,
    ) -> None:
        local_params = self.get_local_shard_parameters()

        non_contiguous_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []

        for send_rank, operations in plan.inter_operations.items():
            tw_tensors = shard_tensors_by_rank.get(send_rank, {})

            for op in operations:
                tw_sliced = tw_tensors.get(op.send_shard_meta.name)
                if tw_sliced is None:
                    logger.warning(
                        f"TW tensor not found: {op.send_shard_meta.name} from rank {send_rank}"
                    )
                    continue

                iw_tensor = local_params.get(op.recv_shard_meta.name)
                if iw_tensor is None:
                    logger.warning(f"IW tensor not found: {op.recv_shard_meta.name}")
                    continue

                iw_sliced = iw_tensor[op.inf_slices]

                if not iw_sliced.is_contiguous():
                    contiguous = iw_sliced.contiguous()
                    non_contiguous_pairs.append((iw_sliced, contiguous))
                    iw_sliced = contiguous

                iw_sliced.copy_(tw_sliced)

        for original, contiguous in non_contiguous_pairs:
            original.copy_(contiguous)

        logger.info(
            f"Applied TransferPlan: {len(non_contiguous_pairs)} non-contiguous pairs handled"
        )

    def teardown_weight_update_group(self) -> None:
        self._tw_handles.clear()
        self._transfer_plans.clear()
        self._infer_world_sizes.clear()
        self._rank_info = None
