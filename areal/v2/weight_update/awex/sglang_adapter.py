# SPDX-License-Identifier: Apache-2.0
# pyright: reportMissingImports=false
from __future__ import annotations

import gc
import math
import os
from typing import Any

import torch
import torch.distributed as dist


def _patch_tms_hook_mode() -> None:
    """Ignore late torch-memory-saver hook changes after initialization.

    Importing the Megatron converter can assign ``hook_mode`` after SGLang has
    initialized the memory-saver singleton and removed its constructor state.
    That assignment is unsupported and would prevent AWEX model converters from
    registering, so make only the late assignment a no-op.
    """
    try:
        import torch_memory_saver as _tms
    except Exception:
        return
    instance = getattr(_tms, "torch_memory_saver", None)
    if instance is None:
        return
    cls = type(instance)
    prop = cls.hook_mode
    if getattr(prop.fset, "_awex_safe", False):
        return

    def _safe_setter(self, value):
        if not hasattr(self, "_impl_ctor_kwargs"):
            return
        prop.fset(self, value)

    _safe_setter._awex_safe = True
    cls.hook_mode = property(prop.fget, _safe_setter)


# This must run before AWEX imports trigger model-registry auto-discovery.
_patch_tms_hook_mode()

from awex.meta.infer_meta_resolver import InferParamMetaResolver  # noqa: E402
from awex.meta.meta_resolver import ParamMetaResolver  # noqa: E402
from awex.meta.weight_meta import (  # noqa: E402
    ParameterMeta,
    ParameterReplicaMeta,
    ParameterShardMeta,
)
from awex.reader.nccl_reader import NCCLWorkerWeightsReader  # noqa: E402
from awex.sharding import get_sharding_strategy_builder  # noqa: E402
from awex.sharding.param_sharding import ShardingType  # noqa: E402
from awex.sharding.rank_info import RankInfo  # noqa: E402
from awex.sharding.sglang_sharding import (  # noqa: E402
    get_sglang_rank_info,
    get_sglang_sharding_strategy,
)
from awex.transfer.nccl_comm import batch_send_recv, nccl_build_recv_ops  # noqa: E402
from awex.transfer.nccl_stream_batch import (  # noqa: E402
    NcclColocateStreamBatchTransport,
)
from awex.transfer.transfer_plan import (  # noqa: E402
    TransferPlan,
    TransferPlanBuilder,
    slice_tensor,
)
from awex.util.common import simple_hf_config  # noqa: E402

from areal.utils import logging  # noqa: E402
from areal.v2.weight_update.awex import (  # noqa: E402
    awex_wu_use_group,
    fetch_kv_metadata,
)
from areal.v2.weight_update.awex.delta_config import (  # noqa: E402
    delta_transfer_enabled,
    make_delta_engine,
    payload_carries_delta,
    separation_delta_transfer_enabled,
)
from areal.v2.weight_update.awex.separation_verify import (  # noqa: E402
    separation_post_apply_verify_enabled,
    verify_separation_post_apply,
)
from areal.v2.weight_update.awex.weight_digest import log_tensor_digest  # noqa: E402
from areal.v2.weight_update.inference_adapter import (  # noqa: E402
    AwexInferenceAdapter,
)
from areal.v2.weight_update.nccl_group import (  # noqa: E402
    init_weights_update_group,
    setup_batch_isend_irecv,
)

logger = logging.getLogger("AwexSGLangAdapter")


def _ensure_awex_models_registered() -> None:
    """Rebuild the AWEX registry if an earlier auto-import was incomplete."""
    try:
        from awex.models import registry as registry

        registry.import_model_configs.cache_clear()
        registry.ModelRegistry.models = registry.import_model_configs()
        missing = [
            model_name
            for model_name in ("BailingMoeV2_5ForCausalLM", "BailingMoeV2ForCausalLM")
            if model_name not in registry.ModelRegistry.models
        ]
        if missing:
            logger.warning("AWEX model registry is missing converters: %s", missing)
    except Exception as exc:  # pragma: no cover - diagnostic guard
        logger.warning("Failed to rebuild AWEX model registry: %s", exc)


_ensure_awex_models_registered()


class _SingleInstanceMetaResolver(ParamMetaResolver):
    """Aggregate raw metadata for one inference engine instance."""

    def __init__(self, hf_config, engine_name, infer_engine_config, raw_meta_list):
        super().__init__(hf_config)
        self._raw_meta_list = raw_meta_list
        rank0 = self._select_rank0(raw_meta_list)
        self._model_arch_name = rank0["model_arch_name"]
        self._sharding_strategy = get_sharding_strategy_builder(engine_name)(
            self._model_arch_name,
            infer_engine_config,
            rank0["rank_info"],
        )

    @staticmethod
    def _select_rank0(raw_meta_list):
        for info in raw_meta_list:
            if info["rank_info"].global_rank == 0:
                return info
        return raw_meta_list[0]

    def get_model_arch_name(self) -> str:
        return self._model_arch_name

    def get_parameters_meta(self):
        return self._build_params_meta()

    def _get_params_raw_meta(self):
        return self._raw_meta_list

    def _get_sharding_info(self, name, rank_info, param_meta):
        return self._sharding_strategy.get_sharding_strategy(
            name, rank_info=rank_info, param_meta=param_meta
        )


class AwexSGLangAdapter(AwexInferenceAdapter):
    """Awex inference adapter for in-process SGLang schedulers."""

    def __init__(self, scheduler: Any):
        self._scheduler = scheduler
        self._transfer_plan: TransferPlan | None = None
        self._weights_update_group = None
        self._weights_update_group_gloo = None
        self._world_size: int | None = None
        self._separation_delta_transport: NcclColocateStreamBatchTransport | None = None
        self._transfer_rank: int | None = None
        self._rank_info: RankInfo | None = None
        self._parameters: dict[str, torch.Tensor] | None = None
        self._released_tags: set[str] = set()
        self._meta_server_client = None
        self._reader: NCCLWorkerWeightsReader | None = None
        self._meta_server_addr: str | None = None
        self._infer_world_size: int | None = None
        self._train_world_size: int | None = None
        self._infer_instance_world_size: int | None = None
        self._num_infer_engines: int | None = None
        self._engine_rank: int | None = None
        self._instance_local_rank: int | None = None
        self._infer_params_meta = None
        self._infer_conf: dict[str, Any] | None = None
        self._colocate_timeout_s = 300.0
        self._colocate_initialized = False
        self._delta_engine = None

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

    def _get_colocate_model_context(self) -> dict[str, Any]:
        """Build the AWEX context for one inference engine instance."""
        server_args = self._scheduler.server_args
        tp_size = int(getattr(server_args, "tp_size", 1))
        pp_size = int(getattr(server_args, "pp_size", 1))
        tp_rank = int(getattr(self._scheduler, "tp_rank", 0))
        instance_world = self._infer_instance_world_size or tp_size * pp_size
        instance_local_rank = (
            self._instance_local_rank
            if self._instance_local_rank is not None
            else tp_rank
        )
        return {
            "scheduler": self._scheduler,
            "infer_engine_config": server_args,
            "tp_rank": tp_rank,
            "tp_size": tp_size,
            "pp_rank": int(getattr(self._scheduler, "pp_rank", 0)),
            "pp_size": pp_size,
            "dp_size": int(getattr(server_args, "dp_size", 1)),
            "world_size": instance_world,
            "global_rank": instance_local_rank,
            "local_rank": tp_rank,
            "attn_tp_rank": int(getattr(self._scheduler, "attn_tp_rank", tp_rank)),
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

    def execute_weight_update(self, version: int) -> None:
        if separation_delta_transfer_enabled():
            self._execute_separation_weight_update(version)
            return

        if self._transfer_plan is None:
            raise RuntimeError("Transfer plan is not initialized")
        if self._weights_update_group is None:
            raise RuntimeError("Weight update group is not initialized")
        if self._weights_update_group_gloo is None:
            raise RuntimeError("Weight update control group is not initialized")

        params = self.get_local_shard_parameters()
        log_tensor_digest(
            params.items(),
            role="infer",
            phase="pre_apply",
            version=version,
            extra={
                "transfer_path": "separation_full",
                "transfer_rank": self._transfer_rank,
                "payload_manifest": "receiver_params",
            },
        )
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

        log_tensor_digest(
            params.items(),
            role="infer",
            phase="post_apply",
            version=version,
            extra={
                "transfer_path": "separation_full",
                "transfer_rank": self._transfer_rank,
                "payload_manifest": "receiver_params",
            },
        )
        dist.barrier(group=self._weights_update_group_gloo)
        if separation_post_apply_verify_enabled():
            verify_separation_post_apply(
                params,
                self._transfer_plan,
                self._weights_update_group_gloo,
                role="infer",
                version=version,
                mode="full",
            )

    def _execute_separation_weight_update(self, version: int) -> None:
        """Receive a sparse separated-card update, with dense fallback."""
        if self._transfer_plan is None:
            raise RuntimeError("Transfer plan is not initialized")
        if self._weights_update_group is None:
            raise RuntimeError("Weight update group is not initialized")
        if self._weights_update_group_gloo is None:
            raise RuntimeError("Separation control group is not initialized")

        decision = torch.tensor([1], dtype=torch.int64)
        dist.all_reduce(
            decision, op=dist.ReduceOp.MIN, group=self._weights_update_group_gloo
        )
        use_delta = bool(decision.item())
        params = self.get_local_shard_parameters()
        log_tensor_digest(
            params.items(),
            role="infer",
            phase="pre_apply",
            version=version,
            extra={
                "transfer_path": "separation_delta" if use_delta else "separation_full",
                "transfer_rank": self._transfer_rank,
                "payload_manifest": "receiver_params",
            },
        )

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

        log_tensor_digest(
            params.items(),
            role="infer",
            phase="post_apply",
            version=version,
            extra={
                "transfer_path": "separation_delta" if use_delta else "separation_full",
                "transfer_rank": self._transfer_rank,
                "payload_manifest": "receiver_params",
            },
        )
        dist.barrier(group=self._weights_update_group_gloo)
        if separation_post_apply_verify_enabled():
            verify_separation_post_apply(
                params,
                self._transfer_plan,
                self._weights_update_group_gloo,
                role="infer",
                version=version,
                mode="delta" if use_delta else "full",
            )

    def _execute_separation_delta_recv(
        self,
        recv_params: dict[str, torch.Tensor],
        version: int,
    ) -> None:
        """Apply sparse parameter shards through DTE's two-round P2P protocol."""
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
        for dtype, ops in operations_by_dtype.items():
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
            len(operations_by_dtype),
        )

    def batch_isend_irecv(self, **kwargs) -> None:
        setup_kwargs = {k: v for k, v in kwargs.items() if k != "world_size"}
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
        self._rank_info = None
        self._parameters = None
        self._reader = None
        self._meta_server_client = None
        self._colocate_initialized = False

    # ── Colocated weight transfer methods ─────────────────────────────────

    def _compute_local_raw_meta(self) -> dict:
        """Compute this rank's metadata with AWEX's SGLang converter."""
        return InferParamMetaResolver._get_model_param_info(
            "sglang",
            self._scheduler.server_args,
            convert_params=True,
            engine_rank=self._engine_rank or 0,
            model=self._get_model(),
            model_context=self._get_colocate_model_context(),
        )

    def _build_instance_params_meta(self):
        """Exchange one instance's raw metadata through the MetaServer.

        Do not gather over SGLang's ``tp_cpu_group``. The scheduler MainThread
        uses that group to broadcast requests, while this method can run on a
        different thread; concurrent collectives on the shared, non-thread-safe
        group can race and deadlock. MetaServer exchange avoids process-group
        collectives and isolates keys by inference-engine rank.
        """
        local_raw = self._compute_local_raw_meta()
        instance_world = self._infer_instance_world_size or 1
        if instance_world > 1:
            client = self._meta_server_client
            prefix = f"infer_instance_raw_meta_{self._engine_rank}"
            client.put_object(f"{prefix}_{self._instance_local_rank}", local_raw)
            raw_meta_list = [
                client.get_object(f"{prefix}_{rank}", timeout=300.0)
                for rank in range(instance_world)
            ]
        else:
            raw_meta_list = [local_raw]

        for info in raw_meta_list:
            rank_info = info.get("rank_info")
            if isinstance(rank_info, dict):
                info["rank_info"] = RankInfo(**rank_info)

        resolver = _SingleInstanceMetaResolver(
            self._get_model().config,
            "sglang",
            self._scheduler.server_args,
            raw_meta_list,
        )
        return resolver.get_parameters_meta()

    def _ensure_reader(self) -> NCCLWorkerWeightsReader:
        if self._reader is not None:
            return self._reader
        if not self._colocate_initialized:
            raise RuntimeError("Colocate weight update is not initialized")

        training_params_meta = self._meta_server_client.get_object(
            "training_params_meta", timeout=10000.0
        )
        reader = NCCLWorkerWeightsReader(
            engine_name="sglang",
            model=self._get_model(),
            model_context=self._get_colocate_model_context(),
            infer_conf=self._infer_conf,
            engine_rank=self._engine_rank,
            num_engines=self._num_infer_engines,
            meta_server_addr=self._meta_server_addr,
            parameters_meta=self._infer_params_meta,
            training_params_meta=training_params_meta,
            enable_colocate_mode=True,
            ipc_backend="cuda",
            enable_debug_mode=False,
        )
        reader.initialize()
        self._reader = reader
        logger.info(
            "Constructed NCCLWorkerWeightsReader for transfer rank %d",
            reader.transfer_rank,
        )
        return reader

    def init_colocate_weight_update(
        self,
        meta_server_addr: str,
        transfer_rank: int,
        infer_world_size: int,
        train_world_size: int,
        num_engines: int,
        master_port: int,
        timeout_s: float = 300.0,
        **kwargs: Any,
    ) -> None:
        """Publish inference metadata without waiting for training-side data.

        The native reader is intentionally constructed on the first update,
        after training has published ``training_params_meta``.
        """
        expected_delta_enabled = kwargs.pop("expected_delta_enabled", None)
        del master_port, num_engines, kwargs
        if infer_world_size != train_world_size:
            raise ValueError(
                "Colocate mode requires equal inference and training rank counts; "
                f"got infer={infer_world_size}, train={train_world_size}"
            )

        from awex.meta.meta_server import MetaServerClient

        self._infer_world_size = infer_world_size
        self._train_world_size = train_world_size
        self._meta_server_addr = meta_server_addr
        self._colocate_timeout_s = timeout_s

        server_args = self._scheduler.server_args
        tp_size = int(getattr(server_args, "tp_size", 1))
        pp_size = int(getattr(server_args, "pp_size", 1))
        instance_world = max(1, tp_size * pp_size)
        if infer_world_size % instance_world != 0:
            raise ValueError(
                f"infer_world_size ({infer_world_size}) must be divisible by "
                f"tp_size * pp_size ({instance_world})"
            )
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
        self._transfer_rank = transfer_rank + local_rank
        self._infer_instance_world_size = instance_world
        self._num_infer_engines = infer_world_size // instance_world
        self._engine_rank = transfer_rank // instance_world
        self._instance_local_rank = local_rank

        host, port = meta_server_addr.rsplit(":", 1)
        self._meta_server_client = MetaServerClient(host, int(port))
        self._infer_params_meta = self._build_instance_params_meta()

        infer_conf = {
            "engine_name": "sglang",
            "infer_atten_tp_size": tp_size,
            "infer_world_size": infer_world_size,
            "hf_config": simple_hf_config(self._get_model().config),
            # Preserve the inference router dtype. Falling back to bf16 for an
            # fp32 gate changes message sizes and can wedge the transfer.
            "router_dtype": getattr(self._get_model().config, "router_dtype", "bf16"),
        }
        self._infer_conf = infer_conf

        if self._transfer_rank == 0:
            self._meta_server_client.put_object("infer_conf", infer_conf)
            self._meta_server_client.put_object(
                "num_infer_engines", self._num_infer_engines
            )
            self._meta_server_client.put_object(
                "infer_params_meta", self._infer_params_meta
            )

        self._colocate_initialized = True

        logger.info(
            "Initialized colocate reader metadata for transfer_rank=%d, "
            "engine_rank=%d, instance_local_rank=%d",
            transfer_rank,
            self._engine_rank,
            self._instance_local_rank,
        )
        local_delta = delta_transfer_enabled()
        if (
            expected_delta_enabled is not None
            and bool(expected_delta_enabled) != local_delta
        ):
            raise ValueError(
                "Colocate delta config mismatch on inference rank "
                f"{self._transfer_rank}: expected={expected_delta_enabled}, "
                f"local={local_delta}. Check DTE_DELTA_TRANSFER propagation."
            )
        if local_delta and self._delta_engine is None:
            self._delta_engine = make_delta_engine(
                f"cuda:{torch.cuda.current_device()}"
            )
            logger.info("colocate delta enabled (receiver); DTE DeltaEngine ready")

    def execute_colocate_weight_update(self, version: int) -> None:
        if delta_transfer_enabled():
            self._execute_colocate_delta_weight_update(version)
            return

        self.wait_for_training_offloaded(version)
        self.resume_memory(["weights"])
        self._quiesce_scheduler_streams()
        reader = self._ensure_reader()
        reader.update_weights(step_id=version)
        self._rebuild_derived_weights()
        logger.info("Colocate weight update completed for version %d", version)

    def _execute_colocate_delta_weight_update(self, version: int) -> None:
        self.wait_for_training_offloaded(version)
        reader = self._ensure_reader()
        reader.collect_training_weights(step_id=version)
        deserialized_weights = reader.deserialized_weights
        names = list(deserialized_weights.keys())
        carries_delta = payload_carries_delta(names)
        decoded_delta = None
        if carries_delta:
            decoded_delta = self._delta_decode_for_live_apply(
                deserialized_weights, version
            )
        else:
            self._delta_mark_full_sync(deserialized_weights, version)

        resumed_weights = False
        update_succeeded = False
        try:
            if self._decoded_delta_is_empty(decoded_delta):
                self._delta_engine.commit_live_apply(decoded_delta)
                self._finish_reader_colocate_update(reader, version)
                update_succeeded = True
                return

            self.resume_memory(["weights"])
            resumed_weights = True
            self._quiesce_scheduler_streams()
            recv_parameters = self.get_local_shard_parameters()
            transfer_path = (
                "colocate_delta" if decoded_delta is not None else "colocate_full"
            )
            log_tensor_digest(
                recv_parameters.items(),
                role="infer",
                phase="pre_apply",
                version=version,
                extra={
                    "transfer_path": transfer_path,
                    "transfer_rank": reader.transfer_rank,
                    "payload_manifest": "receiver_params",
                },
            )

            if decoded_delta is not None:
                from dte.core.colocate_protocol import apply_decoded_delta_colocate

                apply_decoded_delta_colocate(
                    transfer_rank=reader.transfer_rank,
                    world_size=reader.infer_world_size,
                    send_plan=reader.send_transfer_plan,
                    recv_plan=reader.transfer_plan,
                    train_to_infer_device_mapping=reader.train_to_infer_device_mapping,
                    infer_to_train_device_mapping=reader.infer_to_train_device_mapping,
                    weights_update_group=reader.weights_update_group,
                    decoded=decoded_delta,
                    recv_params=recv_parameters,
                    device=torch.device(f"cuda:{torch.cuda.current_device()}"),
                    schedule_fn=reader.colocate_transport.execute_recursive_partition_stream_transfer,
                    slice_fn=slice_tensor,
                    rank_coordinate=reader.rank_coordinate,
                    step_id=version,
                )
            else:
                reader.colocate_transport.update_weights_in_colocate_mode(
                    reader.train_to_infer_device_mapping,
                    reader.infer_to_train_device_mapping,
                    reader.transfer_rank,
                    reader.rank_coordinate,
                    reader.infer_world_size,
                    reader.send_transfer_plan,
                    reader.transfer_plan,
                    reader.weights_update_group,
                    deserialized_weights,
                    recv_parameters,
                    step_id=version,
                )

            self._rebuild_derived_weights()
            if decoded_delta is not None:
                self._delta_engine.commit_live_apply(decoded_delta)
            log_tensor_digest(
                recv_parameters.items(),
                role="infer",
                phase="post_apply",
                version=version,
                extra={
                    "transfer_path": transfer_path,
                    "transfer_rank": reader.transfer_rank,
                    "payload_manifest": "receiver_params",
                },
            )
            self._finish_reader_colocate_update(reader, version)
            update_succeeded = True
        finally:
            reader.deserialized_weights = None
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            if resumed_weights and not update_succeeded:
                self.release_memory(["weights"])

        logger.info("Colocate DTE weight update completed for version %d", version)

    @staticmethod
    def _decoded_delta_is_empty(decoded_delta) -> bool:
        if decoded_delta is None:
            return False
        sparse = getattr(decoded_delta, "sparse", {}) or {}
        dense = getattr(decoded_delta, "dense", {}) or {}
        sparse_nnz = sum(int(indices.numel()) for indices, _values in sparse.values())
        dense_numel = sum(int(tensor.numel()) for tensor in dense.values())
        return sparse_nnz == 0 and dense_numel == 0

    def _finish_reader_colocate_update(
        self,
        reader: NCCLWorkerWeightsReader,
        version: int,
    ) -> None:
        from awex.util import device as device_util
        from awex.util.common import get_ip_address

        ip_address = get_ip_address()
        device_id = device_util.current_device()
        key_suffix = f"_{ip_address}_{device_id}_{version}"
        update_finished_key = f"weights_update_finished{key_suffix}"
        reader.meta_server_client.put_object(update_finished_key, True)
        dist.barrier(
            group=reader.weights_update_group,
            device_ids=[device_util.current_device()],
        )
        write_finished_key = f"write_finished{key_suffix}"
        reader.meta_server_client.get_object_then_delete(write_finished_key)

    def seed_delta_base(self, version: int = 0) -> None:
        """Skip virtual delta seeding; the first real update is the full base."""
        if not delta_transfer_enabled():
            logger.info("seed_delta_base skipped: delta transfer disabled")
            return
        logger.info(
            "seed_delta_base skipped for DTE delta transfer at v%d; "
            "the first real payload will be a full sync base",
            version,
        )

    def _delta_mark_full_sync(
        self,
        named_tensors: dict[str, torch.Tensor],
        version: int,
    ) -> None:
        if self._delta_engine is None:
            self._delta_engine = make_delta_engine(
                f"cuda:{torch.cuda.current_device()}"
            )
        decoded = self._delta_engine.decode_for_live_apply(named_tensors, version)
        if decoded is not None:
            raise RuntimeError("Expected a full-sync payload, got a delta payload")
        logger.info("DTE receiver full base advanced to v%d", version)

    def _delta_decode_for_live_apply(
        self,
        named_tensors: dict[str, torch.Tensor],
        version: int,
    ):
        if self._delta_engine is None:
            self._delta_engine = make_delta_engine(
                f"cuda:{torch.cuda.current_device()}"
            )
        decoded = self._delta_engine.decode_for_live_apply(named_tensors, version)
        if decoded is None:
            raise RuntimeError("Expected a delta payload, got a full-sync payload")
        sparse_nnz = sum(int(indices.numel()) for indices, _ in decoded.sparse.values())
        dense_numel = sum(int(tensor.numel()) for tensor in decoded.dense.values())
        logger.info(
            "[dte-perf][infer-delta] v%d sparse_params=%d sparse_nnz=%d "
            "dense_params=%d dense_numel=%d payload_tensors=%d",
            version,
            len(decoded.sparse),
            sparse_nnz,
            len(decoded.dense),
            dense_numel,
            len(named_tensors),
        )
        return decoded

    def _quiesce_scheduler_streams(self) -> None:
        """Drain in-flight device work before the NCCL transfer starts.

        The preceding resume_memory remaps the weight pages through the memory
        saver, and those remaps are asynchronous device work. NCCL P2P must not
        race them, so synchronize the device before the peers write through
        those pointers.
        """
        torch.cuda.synchronize()

    def wait_for_training_offloaded(self, version: int) -> None:
        """Wait until every training rank has offloaded model weights."""
        del version
        if not self._colocate_initialized:
            raise RuntimeError("Colocate weight update is not initialized")
        self._meta_server_client.wait_set_until_size(
            "all_training_offloaded_weights",
            self._train_world_size,
            timeout=self._colocate_timeout_s,
        )

    def _rebuild_derived_weights(self) -> None:
        """Rebuild non-parameter MLA tensors after every in-place transfer.

        SGLang derives absorbed-path ``w_kc`` and ``w_vc`` tensors from model
        parameters during ``post_load_weights`` and stores them outside
        ``named_parameters``. A memory-saver release/resume remaps their pages,
        while AWEX writes named parameters with in-place ``copy_`` and bypasses
        normal model loading. Rebuild on every transfer because source weights
        change each version.
        """
        post_load_weights = getattr(self._get_model(), "post_load_weights", None)
        if post_load_weights is None:
            return
        post_load_weights()
        torch.cuda.synchronize()
        logger.info("Rebuilt derived model weights with post_load_weights()")

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
            self._released_tags.update(native_tags)
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
        self._released_tags.difference_update(native_tags)
        logger.info("resume_memory resumed tags=%s", native_tags)
