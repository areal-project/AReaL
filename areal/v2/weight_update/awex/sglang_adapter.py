# SPDX-License-Identifier: Apache-2.0
# pyright: reportMissingImports=false
# ruff: noqa: E402, I001
from __future__ import annotations

import math
import os
from typing import Any

import torch
import torch.distributed as dist

# Compatibility must run before importing AWEX model modules.
from areal.engine.weight_update.awex.protocol import (  # noqa: E402
    ColocateKeyspace,
    ColocateTopology,
)
from areal.engine.weight_update.awex.sglang import (  # noqa: E402
    SGLangColocateBackend,
)

from awex.meta.weight_meta import (  # noqa: E402
    ParameterMeta,
    ParameterReplicaMeta,
    ParameterShardMeta,
)
from awex.sharding.param_sharding import ShardingType  # noqa: E402
from awex.sharding.rank_info import RankInfo  # noqa: E402
from awex.sharding.sglang_sharding import (  # noqa: E402
    get_sglang_rank_info,
    get_sglang_sharding_strategy,
)
from awex.util.common import simple_hf_config  # noqa: E402
from awex.transfer.nccl_comm import batch_send_recv, nccl_build_recv_ops  # noqa: E402
from awex.transfer.transfer_plan import TransferPlan, TransferPlanBuilder  # noqa: E402
from areal.infra.platforms import current_platform  # noqa: E402
from areal.utils import logging  # noqa: E402
from areal.v2.weight_update.awex import (  # noqa: E402
    awex_wu_use_group,
    fetch_kv_metadata,
)
from areal.v2.weight_update.inference_adapter import (  # noqa: E402
    AwexInferenceAdapter,
)
from areal.v2.weight_update.nccl_group import (  # noqa: E402
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
        self._transfer_rank: int | None = None
        self._separation_init_fingerprint: tuple | None = None
        self._rank_info: RankInfo | None = None
        self._parameters: dict[str, torch.Tensor] | None = None
        self._colocate_backend = SGLangColocateBackend(scheduler)
        self._colocate_timeout_s = 300.0
        self._colocate_initialized = False
        self._colocate_init_fingerprint: tuple | None = None

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
        fingerprint = (
            pair_name,
            master_addr,
            master_port,
            transfer_rank,
            world_size,
            kv_store_url,
            infer_world_size,
            train_world_size,
            num_engines,
        )
        if self._colocate_init_fingerprint is not None:
            raise RuntimeError("AWEX adapter is already initialized for colocation")
        if self._separation_init_fingerprint is not None:
            if self._separation_init_fingerprint == fingerprint:
                return
            raise RuntimeError(
                "AWEX separation group is already initialized with different settings"
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
        try:
            self._transfer_rank = global_rank
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
        except Exception:
            self.teardown_weight_update_group()
            raise
        self._separation_init_fingerprint = fingerprint
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
        del version
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
        self._rank_info = None
        self._parameters = None
        self._separation_init_fingerprint = None
        self._colocate_backend.teardown()
        self._colocate_initialized = False
        self._colocate_init_fingerprint = None

    # ── Colocated weight transfer methods ─────────────────────────────────

    def init_colocate_weight_update(
        self,
        *,
        pair_name: str,
        meta_server_addr: str,
        transfer_rank: int,
        infer_world_size: int,
        train_world_size: int,
        num_engines: int,
        timeout_s: float = 300.0,
    ) -> None:
        """Publish inference metadata without waiting for training-side data.

        The native reader is intentionally constructed on the first update,
        after training has published ``training_params_meta``.
        """
        fingerprint = (
            pair_name,
            meta_server_addr,
            transfer_rank,
            infer_world_size,
            train_world_size,
            num_engines,
            timeout_s,
        )
        if self._separation_init_fingerprint is not None:
            raise RuntimeError("AWEX adapter is already initialized for separation")
        if self._colocate_init_fingerprint is not None:
            if self._colocate_init_fingerprint == fingerprint:
                return
            raise RuntimeError(
                "AWEX colocation is already initialized with different settings"
            )
        server_args = self._scheduler.server_args
        tp_size = int(getattr(server_args, "tp_size", 1))
        pp_size = int(getattr(server_args, "pp_size", 1))
        instance_world = max(1, tp_size * pp_size)
        # The gateway sends ONE init per server carrying the engine-base rank;
        # the RPC is broadcast to every scheduler process of the server, so
        # each derives its own global rank from its tp/pp coordinates.
        if transfer_rank % instance_world != 0:
            raise ValueError(
                f"expected an engine-base transfer_rank (multiple of "
                f"{instance_world}), got {transfer_rank}"
            )
        local_rank = int(getattr(self._scheduler, "pp_rank", 0) or 0) * tp_size + int(
            getattr(self._scheduler, "tp_rank", 0) or 0
        )
        topology = ColocateTopology(
            transfer_rank=transfer_rank + local_rank,
            infer_world_size=infer_world_size,
            train_world_size=train_world_size,
            instance_world_size=instance_world,
        )
        model_config = self._get_model().config
        try:
            self._colocate_timeout_s = timeout_s
            self._colocate_backend.initialize(
                meta_server_addr=meta_server_addr,
                topology=topology,
                infer_hf_config=simple_hf_config(model_config),
                router_dtype=getattr(model_config, "router_dtype", "bf16"),
                expected_num_infer_engines=num_engines,
                publish_infer_params_meta=True,
            )
        except Exception:
            self.teardown_weight_update_group()
            raise
        self._colocate_initialized = True
        self._colocate_init_fingerprint = fingerprint

        logger.info(
            "Initialized colocate reader metadata for pair=%r, transfer_rank=%d, "
            "engine_rank=%d, instance_local_rank=%d",
            pair_name,
            topology.transfer_rank,
            topology.engine_rank,
            topology.instance_local_rank,
        )

    def execute_colocate_weight_update(self, version: int) -> None:
        self.wait_for_training_offloaded()
        self.resume_memory(["weights"])
        self._quiesce_scheduler_streams()
        self._colocate_backend.update_weights(version)
        topology = self._colocate_backend.topology
        if topology.instance_local_rank == 0:
            self._colocate_backend.meta_server_client.add_object_to_set(
                ColocateKeyspace.FINISHED_WEIGHT_UPDATE_ENGINES,
                topology.engine_rank,
            )
        logger.info("Colocate weight update completed for version %d", version)

    def _quiesce_scheduler_streams(self) -> None:
        """Drain in-flight device work before the NCCL transfer starts.

        The preceding resume_memory remaps the weight pages through the memory
        saver, and those remaps are asynchronous device work. NCCL P2P must not
        race them, so synchronize the device before the peers write through
        those pointers.
        """
        torch.cuda.synchronize()

    def wait_for_training_offloaded(self) -> None:
        """Wait until every training rank has offloaded model weights."""
        if not self._colocate_initialized:
            raise RuntimeError("Colocate weight update is not initialized")
        topology = self._colocate_backend.topology
        self._colocate_backend.meta_server_client.wait_set_until_size(
            ColocateKeyspace.ALL_TRAINING_OFFLOADED_WEIGHTS,
            topology.train_world_size,
            timeout=self._colocate_timeout_s,
        )

    # Tags understood by SGLang's native release/resume_memory_occupation.
    _SGLANG_MEMORY_TAGS = {"kv_cache", "weights", "cuda_graph"}

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
        logger.info("release_memory completed with tags=%s", tags)

    def resume_memory(self, tags: list[str] | None = None) -> None:
        from sglang.srt.managers.io_struct import ResumeMemoryOccupationReqInput

        # Not gated on locally-tracked released tags: the colocate handover
        # releases inference memory through the rollout controller, so this
        # adapter never observes that release. Resume is not idempotent on the
        # SGLang side, so it must be issued exactly once per released tag.
        native_tags = (
            [t for t in tags if t in self._SGLANG_MEMORY_TAGS] if tags else None
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
        if not native_tags:
            logger.warning("resume_memory: nothing to resume for tags=%s", tags)
            return
        req = ResumeMemoryOccupationReqInput(tags=native_tags)
        self._scheduler.resume_memory_occupation(req)
        logger.info("resume_memory resumed tags=%s", native_tags)
