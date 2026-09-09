# SPDX-License-Identifier: Apache-2.0
# pyright: reportMissingImports=false
"""Shared AWEX inference adapter for in-process SGLang schedulers."""

from __future__ import annotations

import gc
import math
import os
import time
from typing import Any

import httpx
import torch
import torch.distributed as dist

from areal.engine.awex.memory_saver import patch_tms_hook_mode

# AWEX imports model converters eagerly; preserve the main-branch memory-saver
# hook before importing the registry.
patch_tms_hook_mode()

from awex.meta.infer_meta_resolver import InferParamMetaResolver  # noqa: E402
from awex.meta.meta_resolver import ParamMetaResolver  # noqa: E402
from awex.meta.weight_meta import (  # noqa: E402
    ParameterMeta,
    ParameterReplicaMeta,
    ParameterShardMeta,
)
from awex.models.registry import get_infer_weights_converter  # noqa: E402
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
from awex.util.tensor_util import (  # noqa: E402
    cuda_ipc_deserialize,
    reconstruct_tensors_from_groups,
)

from areal.engine.awex.adapters.inference_adapter import (  # noqa: E402
    AwexInferenceAdapter,
)
from areal.engine.awex.delta_config import (  # noqa: E402
    DTERuntimeConfig,
    synchronize_wire_dtypes,
    validate_dte_world_size,
)
from areal.engine.awex.transport.metadata import (  # noqa: E402
    awex_wu_use_group,
    fetch_kv_metadata,
)
from areal.engine.awex.transport.nccl_group import (  # noqa: E402
    batch_send_recv_by_peer,
    init_weights_update_group,
    setup_batch_isend_irecv,
)
from areal.infra.platforms import current_platform  # noqa: E402
from areal.utils import logging  # noqa: E402

logger = logging.getLogger("AwexSGLangAdapter")


class _PhysicalDeviceMetaServerClient:
    """Use physical GPU ids in AWEX colocate metadata and handshake keys."""

    _DEVICE_KEY_PREFIXES = (
        "training_serialized_weights_",
        "weights_update_finished_",
        "write_finished_",
    )

    def __init__(self, client: Any, physical_gpu_id: int):
        self._client = client
        self._physical_gpu_id = physical_gpu_id

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def _rewrite_device_key(self, key: str) -> str:
        if not key.startswith(self._DEVICE_KEY_PREFIXES):
            return key
        prefix_and_ip, step = key.rsplit("_", 1)
        prefix_and_ip, _logical_gpu_id = prefix_and_ip.rsplit("_", 1)
        return f"{prefix_and_ip}_{self._physical_gpu_id}_{step}"

    def add_object_to_set(self, key: str, value: Any) -> Any:
        if key == "inference_device_rank_entries":
            ip_address, _logical_gpu_id, transfer_rank = value
            value = (ip_address, self._physical_gpu_id, transfer_rank)
        return self._client.add_object_to_set(key, value)

    def get_object(self, key: str, *args: Any, **kwargs: Any) -> Any:
        return self._client.get_object(self._rewrite_device_key(key), *args, **kwargs)

    def put_object(self, key: str, *args: Any, **kwargs: Any) -> Any:
        return self._client.put_object(self._rewrite_device_key(key), *args, **kwargs)

    def get_object_then_delete(self, key: str, *args: Any, **kwargs: Any) -> Any:
        return self._client.get_object_then_delete(
            self._rewrite_device_key(key), *args, **kwargs
        )


def _get_router_dtype(config: Any) -> Any:
    """Read router dtype from a flat or multimodal Hugging Face config."""
    router_dtype = getattr(config, "router_dtype", None)
    if router_dtype is not None:
        return router_dtype
    text_config = getattr(config, "text_config", config)
    return getattr(text_config, "router_dtype", "bf16")


def _normalize_router_dtype(dtype: Any) -> str:
    normalized = str(dtype).lower().replace("torch.", "")
    aliases = {
        "bf16": "bf16",
        "bfloat16": "bf16",
        "fp16": "fp16",
        "float16": "fp16",
        "half": "fp16",
        "fp32": "fp32",
        "float32": "fp32",
        "float": "fp32",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported AWEX router dtype: {dtype}")
    return aliases[normalized]


def _get_legacy_awex_hf_config(model, model_runner=None):
    """Serialize the complete SGLang runtime config for legacy AWEX."""
    model_config = getattr(model_runner, "model_config", None)
    config = getattr(model_config, "hf_config", None)
    if config is None:
        config = model.config
    serialized_config = simple_hf_config(config)
    if not getattr(serialized_config, "architectures", None):
        serialized_config.architectures = [type(model).__name__]
    return serialized_config


def _ensure_awex_models_registered() -> None:
    """Rebuild AWEX's registry if an earlier eager model import failed."""
    try:
        from awex.models import registry

        registry.import_model_configs.cache_clear()
        registry.ModelRegistry.models = registry.import_model_configs()
    except Exception as exc:  # pragma: no cover - diagnostics only
        logger.warning("Failed to rebuild AWEX model registry: %s", exc)


_ensure_awex_models_registered()


class _SingleInstanceMetaResolver(ParamMetaResolver):
    """Aggregate one SGLang instance's raw metadata with AWEX primitives."""

    def __init__(self, hf_config, engine_name, infer_engine_config, raw_meta_list):
        super().__init__(hf_config)
        self._raw_meta_list = raw_meta_list
        rank0 = next(
            (info for info in raw_meta_list if info["rank_info"].global_rank == 0),
            raw_meta_list[0],
        )
        self._model_arch_name = rank0["model_arch_name"]
        self._sharding_strategy = get_sharding_strategy_builder(engine_name)(
            self._model_arch_name,
            infer_engine_config,
            rank0["rank_info"],
        )

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
    """Shared SGLang adapter for v1 colocate and v2 separated transfer."""

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
        self._parameter_layout: str | None = None
        self._parameters: dict[str, torch.Tensor] | None = None
        self._released_tags: set[str] = set()
        self._colocate_admin_api_key: str = "areal-admin-key"
        self._colocate_http_client: httpx.Client | None = None
        self._colocate_timeout_s: float = 120.0
        self._colocate_transport = None
        self._train_to_infer_device_mapping: dict | None = None
        self._infer_to_train_device_mapping: dict | None = None
        self._dte_config = DTERuntimeConfig.from_env()
        self._legacy_meta_server_client = None
        self._legacy_reader: NCCLWorkerWeightsReader | None = None
        self._legacy_transfer_rank: int | None = None
        self._legacy_local_gpu_id: int | None = None
        self._legacy_infer_world_size: int | None = None
        self._legacy_train_world_size: int | None = None
        self._legacy_meta_server_addr: str | None = None
        self._legacy_instance_world_size: int | None = None
        self._legacy_num_infer_engines: int | None = None
        self._legacy_engine_rank: int | None = None
        self._legacy_instance_local_rank: int | None = None
        self._legacy_infer_params_meta = None
        self._legacy_infer_conf: dict | None = None
        self._initialized = False

    def _get_model(self) -> torch.nn.Module:
        return self._scheduler.tp_worker.model_runner.model

    def _get_awex_hf_config(self):
        """Return the complete runtime HF config retained by SGLang."""
        model_runner = self._scheduler.tp_worker.model_runner
        model_config = getattr(model_runner, "model_config", None)
        hf_config = getattr(model_config, "hf_config", None)
        if hf_config is None:
            hf_config = self._get_model().config
        return hf_config

    def _serialize_awex_hf_config(self) -> dict[str, Any]:
        config = self._get_awex_hf_config().to_dict()
        if not config.get("architectures"):
            config["architectures"] = [self._get_model_arch_name()]
        return config

    def _get_model_arch_name(self) -> str:
        return type(self._get_model()).__name__

    def _get_model_context(self) -> dict[str, Any]:
        server_args = self._scheduler.server_args
        tp_size = int(getattr(server_args, "tp_size", 1))
        pp_size = int(getattr(server_args, "pp_size", 1))
        dp_size = int(getattr(server_args, "dp_size", 1))

        if self._legacy_instance_world_size is not None:
            world_size = self._legacy_instance_world_size
            global_rank = self._legacy_instance_local_rank
        elif dist.is_available() and dist.is_initialized():
            world_size = int(dist.get_world_size())
            global_rank = int(dist.get_rank())
        else:
            world_size = int(tp_size * pp_size)
            global_rank = int(getattr(self._scheduler, "tp_rank", 0))

        if self._legacy_instance_world_size is not None:
            local_rank = int(getattr(self._scheduler, "tp_rank", 0))
        else:
            local_rank = int(
                getattr(
                    self._scheduler,
                    "local_rank",
                    os.environ.get("LOCAL_RANK", getattr(self._scheduler, "gpu_id", 0)),
                )
            )

        return {
            "scheduler": self._scheduler,
            "infer_engine_config": server_args,
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
            "num_engines": self._legacy_num_infer_engines or 1,
            "converter_context": {
                "engine_name": "sglang",
                "infer_atten_tp_size": int(model_context["attn_tp_size"]),
                "hf_config": self._serialize_awex_hf_config(),
                "router_dtype": _normalize_router_dtype(
                    _get_router_dtype(self._get_model().config)
                ),
                "device_backend": current_platform.device_type,
            },
        }

    def get_parallelism(self) -> dict:
        """Compatibility method used by the v1 SGLang scheduler plugin."""
        model_context = self._get_model_context()
        server_args = self._scheduler.server_args
        return {
            "world_size": model_context["world_size"],
            "tp_size": int(getattr(server_args, "tp_size", model_context["tp_size"])),
            "pp_size": int(getattr(server_args, "pp_size", model_context["pp_size"])),
            "dp_size": int(getattr(server_args, "dp_size", model_context["dp_size"])),
            "ep_size": int(getattr(server_args, "ep_size", 1)),
            "num_engines": self._legacy_num_infer_engines or 1,
        }

    def _build_rank_info(self) -> RankInfo:
        model_context = self._get_model_context()
        return get_sglang_rank_info(model_context, engine_rank=0)

    def _unfuse_params(
        self, name: str, tensor: torch.Tensor
    ) -> list[tuple[str, torch.Tensor]]:
        """Expose fused SGLang tensors as the HF layout used by FSDP."""
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

    def _build_sharding_strategy(self, rank_info: RankInfo):
        model_name = self._get_model_arch_name()
        if self._parameter_layout == "hf":
            architectures = getattr(self._get_model().config, "architectures", None)
            if architectures:
                model_name = architectures[0]
        infer_engine_config = self._scheduler.server_args
        return get_sglang_sharding_strategy(model_name, infer_engine_config, rank_info)

    def _get_weight_converter(self, rank_info: RankInfo):
        if self._weight_converter is None:
            self._weight_converter = get_infer_weights_converter(
                "sglang",
                self._get_model_arch_name(),
                self._get_model().config,
                rank_info,
                self._scheduler.server_args,
            )
        return self._weight_converter

    def _iter_hf_params(self, rank_info: RankInfo):
        """Yield parameters in the canonical layout selected by training."""
        if self._parameter_layout == "hf":
            for name, param in self._get_model().named_parameters():
                yield from self._unfuse_params(name, param.data)
            return

        converter = self._get_weight_converter(rank_info)
        converted_names: set[str] = set()
        embed_tensor: torch.Tensor | None = None

        for name, param in self._get_model().named_parameters():
            for hf_name, hf_tensor in converter.convert_param(name, param.data):
                converted_names.add(hf_name)
                if hf_name == "model.embed_tokens.weight":
                    embed_tensor = hf_tensor
                yield hf_name, hf_tensor

        model_context = self._get_model_context()
        if (
            getattr(self._get_model().config, "tie_word_embeddings", False)
            and model_context["pp_rank"] == model_context["pp_size"] - 1
            and "lm_head.weight" not in converted_names
            and embed_tensor is not None
        ):
            yield "lm_head.weight", embed_tensor

    def _compute_legacy_local_raw_meta(self) -> dict:
        return InferParamMetaResolver._get_model_param_info(
            "sglang",
            self._scheduler.server_args,
            convert_params=True,
            engine_rank=self._legacy_engine_rank or 0,
            model=self._get_model(),
            model_context=self._get_model_context(),
        )

    def _build_legacy_instance_params_meta(self):
        local_raw = self._compute_legacy_local_raw_meta()
        instance_world = self._legacy_instance_world_size or 1
        if instance_world > 1:
            client = self._legacy_meta_server_client
            prefix = f"infer_instance_raw_meta_{self._legacy_engine_rank}"
            client.put_object(f"{prefix}_{self._legacy_instance_local_rank}", local_raw)
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

    def _get_legacy_weight_metadata(self):
        if self._legacy_engine_rank is None:
            raise RuntimeError("Legacy SGLang AWEX adapter is not initialized")
        if self._legacy_infer_params_meta is None:
            self._legacy_infer_params_meta = self._build_legacy_instance_params_meta()
        return self._legacy_infer_params_meta

    def get_weight_metadata(self, parameter_layout: str = "hf") -> list[ParameterMeta]:
        if self._legacy_engine_rank is not None:
            return self._get_legacy_weight_metadata()
        if parameter_layout not in {"awex", "hf"}:
            raise ValueError(f"Unsupported parameter layout: {parameter_layout}")
        if self._parameter_layout not in (None, parameter_layout):
            raise RuntimeError(
                "AWEX SGLang adapter is already configured for parameter layout "
                f"{self._parameter_layout}, got {parameter_layout}"
            )
        self._parameter_layout = parameter_layout
        rank_info = self._build_rank_info()
        strategy = self._build_sharding_strategy(rank_info)
        self._rank_info = rank_info

        metadata: list[ParameterMeta] = []

        for hf_name, local_tensor in self._iter_hf_params(rank_info):
            local_shape = tuple(local_tensor.shape)
            sharding_type, sharding_dim, num_shards = strategy.get_sharding_strategy(
                hf_name
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

            if sharding_type != ShardingType.NO_SHARDING and 0 <= sharding_dim < len(
                local_shape
            ):
                global_offset[sharding_dim] = int(rank_pos) * int(
                    local_shape[sharding_dim]
                )

            global_shape = list(local_shape)
            if sharding_type != ShardingType.NO_SHARDING and 0 <= sharding_dim < len(
                global_shape
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
        for hf_name, hf_tensor in self._iter_hf_params(rank_info):
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
        timeout_s: float = 300.0,
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

        infer_meta, train_meta = fetch_kv_metadata(kv_store_url, pair_name, timeout_s)

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
        batch_send_recv_by_peer(
            send_ops=[],
            recv_ops=recv_ops,
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
        self._parameter_layout = None
        self._parameters = None
        if self._colocate_http_client is not None:
            self._colocate_http_client.close()
            self._colocate_http_client = None
        self._colocate_transport = None
        self._train_to_infer_device_mapping = None
        self._infer_to_train_device_mapping = None

    def initialize(
        self,
        meta_server_addr: str,
        transfer_rank: int,
        infer_world_size: int,
        train_world_size: int,
        local_gpu_id: int,
        timeout_s: float = 300.0,
    ) -> None:
        """Initialize the v1 MetaServer-based SGLang reader."""
        from awex.meta.meta_server import MetaServerClient

        if infer_world_size != train_world_size:
            raise ValueError(
                "Colocate mode requires equal total rank counts, got "
                f"infer={infer_world_size} vs train={train_world_size}"
            )

        self._legacy_transfer_rank = transfer_rank
        self._legacy_local_gpu_id = local_gpu_id
        self._legacy_infer_world_size = infer_world_size
        self._legacy_train_world_size = train_world_size
        self._legacy_meta_server_addr = meta_server_addr

        server_args = self._scheduler.server_args
        tp_size = int(getattr(server_args, "tp_size", 1))
        pp_size = int(getattr(server_args, "pp_size", 1))
        instance_world = max(1, tp_size * pp_size)
        if infer_world_size % instance_world != 0:
            raise ValueError(
                f"infer_world_size ({infer_world_size}) must be divisible by "
                f"tp_size * pp_size ({instance_world})"
            )
        self._legacy_instance_world_size = instance_world
        self._legacy_num_infer_engines = infer_world_size // instance_world
        self._legacy_engine_rank = transfer_rank // instance_world
        self._legacy_instance_local_rank = transfer_rank % instance_world

        host, port = meta_server_addr.rsplit(":", 1)
        self._legacy_meta_server_client = MetaServerClient(host, int(port))
        self._get_legacy_weight_metadata()

        model_runner = self._scheduler.tp_worker.model_runner
        parallelism = self.get_parallelism()
        infer_conf = {
            "engine_name": "sglang",
            "infer_atten_tp_size": parallelism["tp_size"],
            "infer_world_size": infer_world_size,
            "hf_config": _get_legacy_awex_hf_config(self._get_model(), model_runner),
            "router_dtype": _get_router_dtype(self._get_model().config),
        }
        self._legacy_infer_conf = infer_conf
        if transfer_rank == 0:
            self._legacy_meta_server_client.put_object("infer_conf", infer_conf)
            self._legacy_meta_server_client.put_object(
                "num_infer_engines", self._legacy_num_infer_engines
            )

        self._initialized = True
        logger.info(
            "Legacy SGLang AWEX initialized: transfer_rank=%d, "
            "engine_rank=%d, instance_world=%d, num_engines=%d, timeout=%s",
            transfer_rank,
            self._legacy_engine_rank,
            instance_world,
            self._legacy_num_infer_engines,
            timeout_s,
        )

    def _ensure_reader(self) -> NCCLWorkerWeightsReader:
        if self._legacy_reader is not None:
            return self._legacy_reader
        if self._legacy_meta_server_client is None:
            raise RuntimeError("Legacy SGLang AWEX adapter is not initialized")

        training_params_meta = self._legacy_meta_server_client.get_object(
            "training_params_meta", timeout=10000.0
        )
        reader = NCCLWorkerWeightsReader(
            engine_name="sglang",
            model=self._get_model(),
            model_context=self._get_model_context(),
            infer_conf=self._legacy_infer_conf,
            engine_rank=self._legacy_engine_rank,
            num_engines=self._legacy_num_infer_engines,
            meta_server_addr=self._legacy_meta_server_addr,
            parameters_meta=self._legacy_infer_params_meta,
            training_params_meta=training_params_meta,
            enable_colocate_mode=True,
            ipc_backend="cuda",
            enable_debug_mode=False,
        )
        reader.meta_server_client = _PhysicalDeviceMetaServerClient(
            reader.meta_server_client, self._legacy_local_gpu_id
        )
        reader.initialize()
        self._legacy_reader = reader
        return reader

    @torch.no_grad()
    def update_weights(self, version: int) -> None:
        """Execute one v1 native AWEX colocate reader update."""
        if not self._initialized:
            raise RuntimeError("Legacy SGLang AWEX adapter is not initialized")
        self._ensure_reader().update_weights(step_id=version)
        self._rebuild_derived_weights()
        logger.info("Legacy SGLang weight update completed: version=%d", version)

    def _rebuild_derived_weights(self) -> None:
        model = self._get_model()
        post_load_weights = getattr(model, "post_load_weights", None)
        if post_load_weights is None:
            return
        post_load_weights()
        torch.cuda.synchronize()
        logger.info("post_load_weights() re-derived SGLang weights")

    def wait_for_training_offloaded(self, version: int) -> None:
        del version
        from areal.engine.awex.utils import awex_colocate_timeout_s

        if self._legacy_meta_server_client is None:
            raise RuntimeError("Legacy SGLang AWEX adapter is not initialized")
        self._legacy_meta_server_client.wait_set_until_size(
            "all_training_offloaded_weights",
            self._legacy_train_world_size,
            timeout=awex_colocate_timeout_s(),
        )

    def wait_for_weights_ready(
        self, version: int, timeout_s: float | None = None
    ) -> None:
        from awex.util.common import get_ip_address

        from areal.engine.awex.utils import awex_colocate_timeout_s

        if self._legacy_meta_server_client is None:
            raise RuntimeError("Legacy SGLang AWEX adapter is not initialized")
        key = (
            f"training_serialized_weights_{get_ip_address()}_"
            f"{self._legacy_local_gpu_id}_{version}"
        )
        self._legacy_meta_server_client.wait_key(
            key,
            timeout=awex_colocate_timeout_s() if timeout_s is None else timeout_s,
        )

    def signal_finished_weights_update(self) -> None:
        if self._legacy_instance_local_rank != 0:
            return
        if self._legacy_meta_server_client is None:
            raise RuntimeError("Legacy SGLang AWEX adapter is not initialized")
        self._legacy_meta_server_client.add_object_to_set(
            "finished_weights_update_engines", self._legacy_engine_rank
        )

    def teardown(self) -> None:
        self._legacy_reader = None

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

    def release_memory(self, tags: list[str] | None = None) -> None:
        from sglang.srt.managers.io_struct import ReleaseMemoryOccupationReqInput

        tags = tags or ["kv_cache"]
        tags_to_release = [tag for tag in tags if tag not in self._released_tags]
        if tags_to_release:
            req = ReleaseMemoryOccupationReqInput(tags=tags_to_release)
            self._scheduler.release_memory_occupation(req)
            self._released_tags.update(tags_to_release)
        logger.info("release_memory: tags=%s", tags)

    def resume_memory(self, tags: list[str] | None = None) -> None:
        from sglang.srt.managers.io_struct import ResumeMemoryOccupationReqInput

        tags = tags or ["kv_cache"]
        tags_to_resume = [tag for tag in tags if tag in self._released_tags]
        if tags_to_resume:
            req = ResumeMemoryOccupationReqInput(tags=tags_to_resume)
            self._scheduler.resume_memory_occupation(req)
            self._released_tags.difference_update(tags_to_resume)
        logger.info("resume_memory: tags=%s", tags)


__all__ = ["AwexSGLangAdapter"]
