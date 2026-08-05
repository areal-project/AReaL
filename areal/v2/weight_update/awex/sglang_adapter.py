# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402, I001
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


def _patch_tms_hook_mode_for_awex_registry() -> None:
    """Avoid a torch_memory_saver late-set crash during AWEX model registration."""
    try:
        import torch_memory_saver as _tms
    except Exception:
        return

    inst = getattr(_tms, "torch_memory_saver", None)
    if inst is None:
        return

    cls = type(inst)
    prop = getattr(cls, "hook_mode", None)
    if not isinstance(prop, property) or prop.fget is None or prop.fset is None:
        return
    if getattr(prop.fset, "_areal_awex_safe", False):
        return

    orig_fset = prop.fset

    def _safe_setter(self, value):
        if not hasattr(self, "_impl_ctor_kwargs"):
            return
        return orig_fset(self, value)

    _safe_setter._areal_awex_safe = True
    cls.hook_mode = property(prop.fget, _safe_setter, prop.fdel, prop.__doc__)


# Must run before importing awex modules.  AWEX auto-registers model converters
# during import; without this guard, BailingMoe registration can fail silently in
# the already-initialized SGLang scheduler process.
_patch_tms_hook_mode_for_awex_registry()

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

from areal.utils import logging
from areal.v2.weight_update.awex import (
    awex_wu_use_group,
    fetch_kv_metadata,
    load_kv_metadata_file,
)
from areal.v2.weight_update.awex.delta_config import (
    cuda_mem_stats_mb,
    delta_transfer_enabled,
    make_delta_engine,
    payload_carries_delta,
)
from areal.v2.weight_update.awex.colocate_device import (
    device_mapping_key,
    get_colocate_ip_address,
    get_physical_cuda_device_id,
)
from areal.v2.weight_update.inference_adapter import (
    AwexInferenceAdapter,
)
from areal.v2.weight_update.nccl_group import (
    init_weights_update_group,
    setup_batch_isend_irecv,
)

logger = logging.getLogger("AwexSGLangAdapter")


def _ensure_awex_bailing_models_registered() -> None:
    """Rebuild AWEX model registry if an earlier import cached a failed state."""
    required = ("BailingMoeV2_5ForCausalLM", "BailingMoeV2ForCausalLM")
    try:
        from awex.models import registry as _reg

        cache_clear = getattr(_reg.import_model_configs, "cache_clear", None)
        if cache_clear is not None:
            cache_clear()
        _reg.ModelRegistry.models = _reg.import_model_configs()
        models = getattr(_reg.ModelRegistry, "models", {}) or {}
        missing = [name for name in required if name not in models]
        if missing:
            logger.warning(
                "AWEX model registry missing Bailing converters after rebuild: %s",
                missing,
            )
        else:
            logger.info("AWEX model registry has BailingMoe converters registered")
    except Exception:
        logger.warning("Failed to rebuild AWEX model registry", exc_info=True)


_ensure_awex_bailing_models_registered()


class AwexSGLangAdapter(AwexInferenceAdapter):
    """Awex inference adapter for in-process SGLang schedulers."""

    def __init__(self, scheduler: Any):
        self._scheduler = scheduler
        self._transfer_plan: TransferPlan | None = None
        self._weights_update_group = None
        self._transfer_rank: int | None = None
        self._rank_info: RankInfo | None = None
        self._parameters: dict[str, torch.Tensor] | None = None
        self._released_tags: set[str] = set()
        self._colocate_admin_api_key: str = "areal-admin-key"
        self._colocate_http_client: httpx.Client | None = None
        self._colocate_timeout_s: float = 120.0
        self._colocate_transport = None
        self._train_to_infer_device_mapping: dict | None = None
        self._infer_to_train_device_mapping: dict | None = None
        self._colocate_device_ip: str = ""
        self._colocate_device_id: str = ""
        # Lazy dte DeltaEngine (receiver side); persists across versions to hold
        # only the version chain for live delta apply. Created on first
        # delta-enabled transfer.
        self._delta_engine = None
        self._rebuild_derived_weights("adapter init")

    def _get_model(self) -> torch.nn.Module:
        return self._scheduler.tp_worker.model_runner.model

    def _derived_weight_abs_sums(self) -> dict[str, float]:
        sums: dict[str, float] = {}
        inner = getattr(self._get_model(), "model", None)
        for i, layer in enumerate(getattr(inner, "layers", []) or []):
            attn = getattr(layer, "attention", None)
            for attr in ("w_kc", "w_vc"):
                t = getattr(attn, attr, None)
                if isinstance(t, torch.Tensor):
                    sums[f"layers.{i}.attention.{attr}"] = float(
                        t.detach().abs().sum().item()
                    )
        return sums

    def _check_derived_weight_sanity(self, reason: str, sums: dict[str, float]) -> None:
        enabled = os.environ.get("DTE_DERIVED_WEIGHT_SANITY", "1").lower()
        if enabled in {"0", "false", "no", "off"} or not sums:
            return

        bad = {
            name: value
            for name, value in sums.items()
            if not math.isfinite(value) or value <= 0.0
        }
        if not bad:
            logger.info(
                "derived weight sanity OK (%s): count=%d min_abs_sum=%.6e max_abs_sum=%.6e",
                reason,
                len(sums),
                min(sums.values()),
                max(sums.values()),
            )
            return

        logger.error(
            "derived weight sanity FAILED (%s): %d/%d tensors have non-positive "
            "or non-finite abs_sum; examples=%s",
            reason,
            len(bad),
            len(sums),
            list(bad.items())[:8],
        )
        fail = os.environ.get("DTE_DERIVED_WEIGHT_SANITY_FAIL", "0").lower()
        if fail in {"1", "true", "yes", "on"}:
            raise RuntimeError(
                f"derived weight sanity failed after {reason}: "
                f"{len(bad)}/{len(sums)} tensors invalid"
            )

    def _rebuild_derived_weights(self, reason: str) -> None:
        """Refresh plain tensor attributes derived from model parameters."""
        model = self._get_model()
        fn = getattr(model, "post_load_weights", None)
        if fn is None:
            return
        torch.cuda.synchronize()
        before = self._derived_weight_abs_sums()
        fn()
        torch.cuda.synchronize()
        after = self._derived_weight_abs_sums()
        logger.info(
            "post_load_weights rebuilt derived weights (%s): "
            "derived abs_sum before=%s after=%s",
            reason,
            before,
            after,
        )
        self._check_derived_weight_sanity(reason, after)

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

    def _build_awex_infer_conf(self, model_context: dict[str, Any]) -> dict[str, Any]:
        from awex.util.common import simple_hf_config

        def json_safe(value: Any) -> Any:
            if value is None or isinstance(value, str | int | float | bool):
                return value
            if isinstance(value, dict):
                return {str(k): json_safe(v) for k, v in value.items()}
            if isinstance(value, list | tuple):
                return [json_safe(v) for v in value]
            return str(value)

        model = self._get_model()
        hf_config = getattr(model, "config", None)
        server_args = self._scheduler.server_args
        infer_engine_config: dict[str, Any] = {}
        for key in (
            "device",
            "device_backend",
            "device_type",
            "comm_backend",
            "tp_size",
            "pp_size",
            "dp_size",
            "ep_size",
        ):
            value = getattr(server_args, key, None)
            if value is None or isinstance(value, str | int | float | bool):
                infer_engine_config[key] = value

        device_backend = (
            infer_engine_config.get("device_backend")
            or infer_engine_config.get("device_type")
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        include_tied_lm_head = any(
            hf_name == "lm_head.weight"
            for name, param in model.named_parameters()
            for hf_name, _ in self._unfuse_params(name, param.data)
        )
        return {
            "engine_name": "sglang",
            "infer_atten_tp_size": int(model_context["attn_tp_size"]),
            "router_dtype": str(getattr(hf_config, "router_dtype", "bf16")),
            "infer_engine_config": json_safe(infer_engine_config),
            "hf_config": (
                json_safe(dict(simple_hf_config(hf_config))) if hf_config else {}
            ),
            "device_backend": device_backend,
            "include_tied_lm_head": include_tied_lm_head,
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
            "awex_infer_conf": self._build_awex_infer_conf(model_context),
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
        if name == "model.word_embeddings.weight":
            return [("model.embed_tokens.weight", tensor)]
        if ".attention.fused_qkv_a_proj_with_mqa." in name:
            return [(name, tensor)]
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
        if ".attention.o_proj." in name:
            return [(name.replace(".attention.o_proj.", ".attention.dense."), tensor)]
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

    def execute_weight_update(self, version: int) -> None:
        del version
        if self._transfer_plan is None:
            raise RuntimeError("Transfer plan is not initialized")
        if self._weights_update_group is None:
            raise RuntimeError("Weight update group is not initialized")

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

        dist.barrier(group=self._weights_update_group)

    def batch_isend_irecv(self, **kwargs) -> None:
        setup_kwargs = {k: v for k, v in kwargs.items() if k != "world_size"}
        setup_batch_isend_irecv(
            self._weights_update_group,
            self._transfer_rank,
            kwargs.get("world_size", 0),
            **setup_kwargs,
        )

    def teardown_weight_update_group(self) -> None:
        if self._weights_update_group is not None and dist.is_initialized():
            dist.destroy_process_group(self._weights_update_group)
        self._weights_update_group = None
        self._transfer_plan = None
        self._transfer_rank = None
        self._rank_info = None
        self._parameters = None
        if self._colocate_http_client is not None:
            self._colocate_http_client.close()
            self._colocate_http_client = None
        self._colocate_transport = None
        self._train_to_infer_device_mapping = None
        self._infer_to_train_device_mapping = None

    # ── Colocated weight transfer methods ─────────────────────────────────

    def _fetch_colocate_metadata(
        self,
        kv_store_url: str,
        pair_name: str,
        timeout_s: float,
        metadata_path: str = "",
    ) -> tuple[Any, Any]:
        """Fetch colocate metadata once per SGLang engine, then fan out locally."""
        scheduler = self._scheduler
        tp_rank = int(getattr(scheduler, "tp_rank", 0))
        tp_size = int(getattr(scheduler, "tp_size", 1))

        payload: list[Any] = [None]
        if tp_rank == 0:
            try:
                if metadata_path:
                    try:
                        logger.info(
                            "Loading colocate metadata for pair '%s' from %s",
                            pair_name,
                            metadata_path,
                        )
                        infer_meta, train_meta = load_kv_metadata_file(metadata_path)
                    except Exception:
                        logger.warning(
                            "Failed to load colocate metadata file for pair '%s'; "
                            "falling back to %s",
                            pair_name,
                            kv_store_url,
                            exc_info=True,
                        )
                        infer_meta, train_meta = fetch_kv_metadata(
                            kv_store_url, pair_name, timeout_s=timeout_s
                        )
                else:
                    logger.info(
                        "Fetching colocate metadata for pair '%s' from %s",
                        pair_name,
                        kv_store_url,
                    )
                    infer_meta, train_meta = fetch_kv_metadata(
                        kv_store_url, pair_name, timeout_s=timeout_s
                    )
                payload[0] = ("ok", infer_meta, train_meta)
                logger.info(
                    "Fetched colocate metadata for pair '%s': infer_params=%d, "
                    "train_params=%d",
                    pair_name,
                    len(infer_meta) if hasattr(infer_meta, "__len__") else -1,
                    len(train_meta) if hasattr(train_meta, "__len__") else -1,
                )
            except Exception as exc:
                payload[0] = ("error", repr(exc))

        if dist.is_available() and dist.is_initialized() and tp_size > 1:
            tp_cpu_group = getattr(scheduler, "tp_cpu_group", None)
            if tp_cpu_group is None:
                raise RuntimeError(
                    "SGLang tp_cpu_group is required for TP metadata broadcast"
                )
            try:
                src_rank = dist.get_global_rank(tp_cpu_group, 0)
            except Exception:
                pp_rank = int(getattr(scheduler, "pp_rank", 0))
                src_rank = pp_rank * tp_size
            dist.broadcast_object_list(payload, src=src_rank, group=tp_cpu_group)

        status = payload[0]
        if not status:
            raise RuntimeError(
                f"Failed to fetch colocate metadata for pair '{pair_name}'"
            )
        if status[0] == "error":
            raise RuntimeError(
                f"Failed to fetch colocate metadata for pair '{pair_name}': {status[1]}"
            )
        return status[1], status[2]

    def init_colocate_weight_update(
        self,
        pair_name: str,
        kv_store_url: str,
        transfer_rank: int,
        infer_world_size: int,
        train_world_size: int,
        num_engines: int,
        master_addr: str,
        master_port: int,
        admin_api_key: str = "areal-admin-key",
        timeout_s: float = 120.0,
        expected_delta_enabled: bool | None = None,
        metadata_path: str = "",
    ) -> None:
        if infer_world_size != train_world_size:
            raise ValueError(
                f"Colocate mode requires infer_world_size == train_world_size. "
                f"Got infer_world_size={infer_world_size}, "
                f"train_world_size={train_world_size}"
            )
        self._colocate_pair_name = pair_name
        self._colocate_kv_store_url = kv_store_url
        self._colocate_infer_world_size = infer_world_size
        self._colocate_train_world_size = train_world_size
        self._colocate_admin_api_key = admin_api_key
        self._colocate_timeout_s = timeout_s
        if self._colocate_http_client is None:
            self._colocate_http_client = httpx.Client()

        per_engine_world = infer_world_size // num_engines
        ctx = self._get_model_context()
        tp_size = int(ctx["tp_size"])
        tp_rank = int(ctx["tp_rank"])
        pp_size = int(ctx["pp_size"])
        pp_rank = int(ctx["pp_rank"])
        if per_engine_world != tp_size * pp_size:
            raise RuntimeError(
                "awex colocate per-engine world mismatch: gateway reports "
                f"infer_world_size={infer_world_size} / num_engines={num_engines} "
                f"= {per_engine_world}, but local engine has "
                f"tp_size*pp_size={tp_size * pp_size}"
            )

        engine_local_rank = pp_rank * tp_size + tp_rank
        global_rank = transfer_rank * per_engine_world + engine_local_rank
        self._transfer_rank = global_rank

        train_to_infer, infer_to_train = self._register_and_resolve_device_mapping(
            global_rank=global_rank,
            infer_world_size=infer_world_size,
            train_world_size=train_world_size,
        )

        infer_meta, train_meta = self._fetch_colocate_metadata(
            kv_store_url, pair_name, timeout_s, metadata_path
        )
        self._rebuild_derived_weights("colocate init")

        builder = TransferPlanBuilder(
            infer_world_size=infer_world_size,
            train_world_size=train_world_size,
            num_infer_engines=num_engines,
        )

        self._train_to_infer_device_mapping = train_to_infer
        self._infer_to_train_device_mapping = infer_to_train

        self._send_transfer_plan = builder.build_local_transfer_plan(
            infer_meta,
            train_meta,
            global_transfer_rank=infer_to_train[global_rank],
        )
        self._recv_transfer_plan = builder.build_local_transfer_plan(
            infer_meta,
            train_meta,
            global_transfer_rank=global_rank,
        )

        os.environ["TORCHELASTIC_USE_AGENT_STORE"] = str(False)
        self._weights_update_group = init_weights_update_group(
            master_address=master_addr,
            master_port=master_port,
            rank=global_rank,
            world_size=infer_world_size,
            group_name=f"awex_colocate_{pair_name}",
            role="inference",
        )

        self._colocate_transport = NcclColocateStreamBatchTransport(
            global_rank, infer_world_size
        )

        logger.info(
            "Initialized colocate weight update for pair '%s', "
            "engine_rank=%d, local_rank=%d, global_rank=%d, infer_world_size=%d, "
            "paired_train_rank=%d, device=%s/%s",
            pair_name,
            transfer_rank,
            engine_local_rank,
            global_rank,
            infer_world_size,
            infer_to_train[global_rank],
            self._colocate_device_ip,
            self._colocate_device_id,
        )
        local_delta = delta_transfer_enabled()
        if (
            expected_delta_enabled is not None
            and bool(expected_delta_enabled) != local_delta
        ):
            raise ValueError(
                "Colocate delta config mismatch on inference rank "
                f"{global_rank}: expected={expected_delta_enabled}, "
                f"local={local_delta}. Check DTE_DELTA_TRANSFER propagation."
            )
        if local_delta and self._delta_engine is None:
            # Fail fast: surface a missing dte / bad config at init, not mid-run.
            self._delta_engine = make_delta_engine(
                f"cuda:{torch.cuda.current_device()}"
            )
            logger.info("colocate delta enabled (receiver); dte DeltaEngine ready")

    def _put_colocate_kv(self, key: str, value: Any) -> None:
        assert self._colocate_http_client is not None
        resp = self._colocate_http_client.put(
            f"{self._colocate_kv_store_url}/weight_meta/"
            f"{self._colocate_pair_name}/{key}",
            json={"value": value},
            headers={"Authorization": f"Bearer {self._colocate_admin_api_key}"},
            timeout=self._colocate_timeout_s,
        )
        resp.raise_for_status()

    def _get_colocate_kv(self, key: str, timeout_s: float, label: str) -> Any:
        assert self._colocate_http_client is not None
        deadline = time.monotonic() + timeout_s
        last_status = -1
        polls = 0
        while time.monotonic() < deadline:
            resp = self._colocate_http_client.get(
                f"{self._colocate_kv_store_url}/weight_meta/"
                f"{self._colocate_pair_name}/{key}",
                timeout=5.0,
            )
            last_status = resp.status_code
            if resp.status_code == 200:
                return resp.json()["value"]
            polls += 1
            time.sleep(0.1)
        raise TimeoutError(
            f"Timed out waiting for {label} "
            f"(key={key}, polls={polls}, last_status={last_status})"
        )

    def _register_and_resolve_device_mapping(
        self,
        *,
        global_rank: int,
        infer_world_size: int,
        train_world_size: int,
    ) -> tuple[dict[int, int], dict[int, int]]:
        if infer_world_size != train_world_size:
            raise ValueError(
                "Colocate mode requires equal total rank counts "
                f"(infer={infer_world_size}, train={train_world_size})"
            )

        ip_address = get_colocate_ip_address()
        device_id = get_physical_cuda_device_id(int(torch.cuda.current_device()))
        self._colocate_device_ip = ip_address
        self._colocate_device_id = device_id
        device_info = {
            "ip": ip_address,
            "device_id": device_id,
            "infer_rank": global_rank,
        }
        self._put_colocate_kv(
            f"colocate_infer_device_by_rank_{global_rank}", device_info
        )
        self._put_colocate_kv(
            f"colocate_infer_rank_by_device_"
            f"{device_mapping_key(ip_address, device_id)}",
            global_rank,
        )

        infer_to_train: dict[int, int] = {}
        train_to_infer: dict[int, int] = {}
        for infer_rank in range(infer_world_size):
            info = self._get_colocate_kv(
                f"colocate_infer_device_by_rank_{infer_rank}",
                self._colocate_timeout_s,
                f"inference device mapping for rank {infer_rank}",
            )
            mapped_device_key = device_mapping_key(
                str(info["ip"]), str(info["device_id"])
            )
            train_rank = int(
                self._get_colocate_kv(
                    f"colocate_train_rank_by_device_{mapped_device_key}",
                    self._colocate_timeout_s,
                    f"training rank for inference rank {infer_rank}",
                )
            )
            infer_to_train[infer_rank] = train_rank
            train_to_infer[train_rank] = infer_rank

        mapping_complete = (
            len(infer_to_train) == infer_world_size
            and len(train_to_infer) == train_world_size
        )
        if not mapping_complete:
            raise RuntimeError(
                "Incomplete colocate device mapping: "
                f"infer_to_train={len(infer_to_train)}/{infer_world_size}, "
                f"train_to_infer={len(train_to_infer)}/{train_world_size}"
            )
        logger.info(
            "Resolved colocate device mapping for rank %d: paired_train_rank=%d",
            global_rank,
            infer_to_train[global_rank],
        )
        return train_to_infer, infer_to_train

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
        offloaded_key = (
            f"colocate_train_weights_offloaded_rank{paired_train_rank}_{version}"
        )
        perf_start = time.monotonic()
        perf_prev = perf_start
        # Reset the peak counter so the first stage's peak_mb is attributable.
        cuda_mem_stats_mb()

        def perf_mark(stage: str) -> None:
            nonlocal perf_prev
            now = time.monotonic()
            alloc_mb, peak_mb = cuda_mem_stats_mb()
            logger.info(
                "[dte-perf][infer] v%d rank %d paired_train_rank=%d "
                "%s_ms=%.1f total_ms=%.1f alloc_mb=%.0f peak_mb=%.0f",
                version,
                transfer_rank,
                paired_train_rank,
                stage,
                (now - perf_prev) * 1000,
                (now - perf_start) * 1000,
                alloc_mb,
                peak_mb,
            )
            perf_prev = now

        deadline = time.monotonic() + timeout_s
        offloaded = False
        poll_count = 0
        last_status = -1
        while time.monotonic() < deadline:
            resp = client.get(
                f"{kv_store_url}/weight_meta/{pair_name}/{offloaded_key}",
                timeout=5.0,
            )
            last_status = resp.status_code
            if resp.status_code == 200:
                offloaded = True
                break
            poll_count += 1
            time.sleep(0.1)
        if not offloaded:
            raise TimeoutError(
                f"Training did not offload colocate weights within {timeout_s}s "
                f"(waiting_key={offloaded_key}, polls={poll_count}, "
                f"last_status={last_status})"
            )

        logger.info(
            "Observed colocate train weights offloaded for v%d, paired rank %d",
            version,
            paired_train_rank,
        )
        perf_mark("wait_train_offloaded")

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
        perf_mark("wait_payload_kv")

        serialized_weights = bytes.fromhex(serialized_hex)
        logger.info(
            "[dte-perf][infer] v%d rank %d serialized_payload_mb=%.1f",
            version,
            transfer_rank,
            len(serialized_weights) / 1e6,
        )
        perf_mark("hex_to_bytes")
        group_shared, metadata, names = cuda_ipc_deserialize(serialized_weights)
        torch.cuda.synchronize()
        perf_mark("cuda_ipc_deserialize")
        tensors = reconstruct_tensors_from_groups(group_shared, metadata)
        torch.cuda.synchronize()
        perf_mark("reconstruct_group_tensors")
        deserialized_weights = dict(zip(names, tensors))
        perf_mark("build_deserialized_dict")
        # Reconstruct when delta is enabled locally OR the payload itself carries
        # a delta header. The latter defends against the trainer enabling delta
        # while this worker's DTE_DELTA_TRANSFER did not propagate: without it,
        # the sparse header/@delta_idx names would flow straight into the apply.
        carries_delta = payload_carries_delta(names)
        decoded_delta = None
        if delta_transfer_enabled() or carries_delta:
            if carries_delta:
                decoded_delta = self._delta_decode_for_live_apply(
                    deserialized_weights, version
                )
                perf_mark("delta_decode_for_live_apply")
            else:
                self._delta_mark_full_sync(deserialized_weights, version)
                perf_mark("delta_mark_full_sync")

        resumed_weights = False
        update_succeeded = False
        done_key = f"colocate_done_rank{paired_train_rank}_{version}"

        def put_done() -> None:
            client.put(
                f"{kv_store_url}/weight_meta/{pair_name}/{done_key}",
                json={"value": True},
                headers=auth_headers,
                timeout=10.0,
            )

        try:
            if self._decoded_delta_is_empty(decoded_delta):
                self._delta_engine.commit_live_apply(decoded_delta)
                perf_mark("commit_empty_delta")
                put_done()
                perf_mark("put_done")
                update_succeeded = True
            else:
                # During colocate update, weights are only made addressable here;
                # the train payload is applied immediately after.  Rebuilding Flash
                # derived tensors before the payload sees SGLang's offloaded zero
                # buffers and can fail sanity checks prematurely.
                self.resume_memory(tags=["weights"], rebuild_derived=False)
                resumed_weights = True
                perf_mark("resume_weights")
                recv_parameters = self.get_local_shard_parameters()
                perf_mark("get_local_shard_params")

                rank_info = self._build_rank_info()
                rank_coordinate = f"infer_{rank_info.global_rank}"
                perf_mark("build_rank_info")

                assert self._colocate_transport is not None
                if decoded_delta is not None:
                    from dte.core.colocate_protocol import apply_decoded_delta_colocate

                    schedule_fn = self._colocate_transport.execute_recursive_partition_stream_transfer
                    apply_decoded_delta_colocate(
                        transfer_rank=transfer_rank,
                        world_size=self._colocate_infer_world_size,
                        send_plan=self._send_transfer_plan,
                        recv_plan=self._recv_transfer_plan,
                        train_to_infer_device_mapping=self._train_to_infer_device_mapping,
                        infer_to_train_device_mapping=self._infer_to_train_device_mapping,
                        weights_update_group=self._weights_update_group,
                        decoded=decoded_delta,
                        recv_params=recv_parameters,
                        device=torch.device(f"cuda:{torch.cuda.current_device()}"),
                        schedule_fn=schedule_fn,
                        slice_fn=slice_tensor,
                        rank_coordinate=rank_coordinate,
                        step_id=version,
                    )
                    perf_mark("apply_decoded_delta_colocate")
                else:
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
                    perf_mark("apply_full_colocate")
                self._rebuild_derived_weights(f"colocate update v{version}")
                perf_mark("rebuild_derived_weights")
                if decoded_delta is not None:
                    self._delta_engine.commit_live_apply(decoded_delta)
                    perf_mark("commit_live_delta")

                put_done()
                perf_mark("put_done")
                update_succeeded = True
        finally:
            del deserialized_weights, group_shared, tensors, serialized_weights
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            if resumed_weights and not update_succeeded:
                self.release_memory(tags=["weights"])
            perf_mark("cleanup")

        logger.info(
            "Colocate weight update completed for v%d, rank %d",
            version,
            transfer_rank,
        )

    @staticmethod
    def _decoded_delta_is_empty(decoded_delta) -> bool:
        if decoded_delta is None:
            return False
        sparse = getattr(decoded_delta, "sparse", {}) or {}
        dense = getattr(decoded_delta, "dense", {}) or {}
        sparse_nnz = 0
        for indices, _values in sparse.values():
            sparse_nnz += int(indices.numel())
        dense_numel = sum(int(tensor.numel()) for tensor in dense.values())
        return sparse_nnz == 0 and dense_numel == 0

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

    def _delta_reconstruct(
        self, named_tensors: dict[str, torch.Tensor], version: int
    ) -> dict[str, torch.Tensor]:
        """Reconstruct full weights from a dte delta/full payload (receiver side).

        The bytes already arrived via cuda-IPC, so we use dte's transport-free
        ``DeltaEngine.reconstruct``: it verifies the version chain, decodes the
        sparse patch against the CPU base, refreshes that base, and returns the
        full ``{name: tensor}``. Full weights then flow into
        ``update_weights_in_colocate_mode`` unchanged, so the cross-rank NCCL
        reshard never sees a delta payload. The per-param change masks dte also
        returns are unused for now (first cut applies full weights).
        """
        if self._delta_engine is None:
            self._delta_engine = make_delta_engine(
                f"cuda:{torch.cuda.current_device()}"
            )
        full_params, _masks = self._delta_engine.reconstruct(named_tensors, version)
        return full_params

    def _delta_mark_full_sync(
        self, named_tensors: dict[str, torch.Tensor], version: int
    ) -> None:
        """Record a dense full-sync version without saving a CPU model copy."""
        if self._delta_engine is None:
            self._delta_engine = make_delta_engine(
                f"cuda:{torch.cuda.current_device()}"
            )
        decoded = self._delta_engine.decode_for_live_apply(named_tensors, version)
        if decoded is not None:
            raise RuntimeError("Expected a full-sync payload, got a delta payload")
        logger.info(
            "dte receiver full base advanced to v%d without CPU snapshot", version
        )

    def _delta_decode_for_live_apply(
        self, named_tensors: dict[str, torch.Tensor], version: int
    ):
        """Decode delta and verify version chain without reconstructing full params."""
        if self._delta_engine is None:
            self._delta_engine = make_delta_engine(
                f"cuda:{torch.cuda.current_device()}"
            )
        decoded = self._delta_engine.decode_for_live_apply(named_tensors, version)
        if decoded is None:
            raise RuntimeError("Expected a delta payload, got a full-sync payload")
        sparse_nnz = sum(int(indices.numel()) for indices, _ in decoded.sparse.values())
        dense_numel = sum(int(t.numel()) for t in decoded.dense.values())
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

    # Tags understood by SGLang's native release/resume_memory_occupation.
    _SGLANG_MEMORY_TAGS = {"weights", "kv_cache"}

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
        requested_tags = list(self._SGLANG_MEMORY_TAGS) if tags is None else native_tags
        tags_to_release = [t for t in requested_tags if t not in self._released_tags]
        if tags_to_release:
            req = ReleaseMemoryOccupationReqInput(tags=tags_to_release)
            self._scheduler.release_memory_occupation(req)
            self._released_tags.update(tags_to_release)
        logger.info(
            "release_memory completed with tags=%s released_now=%s",
            tags,
            tags_to_release,
        )

    def resume_memory(
        self, tags: list[str] | None = None, *, rebuild_derived: bool = False
    ) -> None:
        from sglang.srt.managers.io_struct import ResumeMemoryOccupationReqInput

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
        requested_tags = list(self._SGLANG_MEMORY_TAGS) if tags is None else native_tags
        tags_to_resume = [t for t in requested_tags if t in self._released_tags]
        if tags_to_resume:
            req = ResumeMemoryOccupationReqInput(tags=tags_to_resume)
            self._scheduler.resume_memory_occupation(req)
            self._released_tags.difference_update(tags_to_resume)
            if "weights" in tags_to_resume and rebuild_derived:
                self._rebuild_derived_weights("resume weights")
        logger.info(
            "resume_memory completed with tags=%s resumed_now=%s rebuild_derived=%s",
            tags,
            tags_to_resume,
            rebuild_derived,
        )
