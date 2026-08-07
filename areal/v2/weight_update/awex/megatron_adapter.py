# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import gc
import os
import threading
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

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
from awex.transfer.nccl_comm import batch_send_recv, nccl_build_send_ops
from awex.transfer.transfer_plan import TransferPlan, TransferPlanBuilder
from awex.util.tensor_util import (
    cuda_ipc_serialize,
    group_tensors_by_shape_and_dtype,
    release_tensors,
)

from areal.utils import logging
from areal.v2.weight_update.awex import (
    awex_wu_use_group,
    fetch_kv_metadata,
)
from areal.v2.weight_update.awex.colocate_device import (
    device_mapping_key,
    get_colocate_ip_address,
    get_physical_cuda_device_id,
)
from areal.v2.weight_update.awex.delta_config import (
    delta_transfer_enabled,
    make_delta_tracker,
)
from areal.v2.weight_update.awex.delta_detect import (
    build_detector,
    delta_detector_mode,
)
from areal.v2.weight_update.nccl_group import (
    init_weights_update_group,
    setup_batch_isend_irecv,
)
from areal.v2.weight_update.training_adapter import (
    AwexTrainingAdapter,
)

if TYPE_CHECKING:
    from areal.engine.megatron_engine import MegatronEngine

logger = logging.getLogger("AwexMegatronAdapter")


class AwexMegatronAdapter(AwexTrainingAdapter):
    """Awex training adapter for MegatronEngine supporting DP, TP, and PP.

    PP: get_named_parameters already yields only the current stage's layers
    (with globally-correct HF layer indices via get_transformer_layer_offset),
    so each rank naturally reports and sends only its own subset of parameters.
    The gateway merges metadata by name to union disjoint PP stage params
    by name, so the full model is covered across all PP ranks.

    TP: all_gather_param gathers the full tensor on every TP rank before
    convert_to_hf. dp_replicated=True tells awex that TP ranks within a DP
    group hold identical full tensors and only one needs to send.
    """

    def __init__(self, engine: MegatronEngine):
        self._engine = engine
        setattr(engine, "_awex_weight_update_adapter", self)
        self._transfer_plan: TransferPlan | None = None
        self._weights_update_group = None
        self._transfer_rank: int | None = None
        self._offloaded_optimizer_states: dict = {}
        self._offloaded_weights: dict[str, torch.Tensor] = {}
        self._released_tags: set[str] = set()
        self._colocate_lock = threading.Lock()
        self._colocate_admin_api_key: str = "areal-admin-key"
        self._colocate_http_client: httpx.Client | None = None
        self._colocate_timeout_s: float = 120.0
        self._colocate_device_ip: str = ""
        self._colocate_device_id: str = ""
        # Lazy dte DeltaTracker (sender side); persists across versions to hold
        # the CPU snapshot baseline. Created on first delta-enabled transfer.
        self._delta_tracker = None
        # Lazy change detector: snapshot (default) | inversion (DTE_DELTA_DETECTOR).
        self._delta_detector = None
        # AWEX native mcore converter/resolver, used for colocated transfer.
        # This matches the proven 1.0 colocate path and avoids the hand-written
        # TP all-gather + HF conversion peak during the first full sync.
        self._awex_weight_converter = None
        self._awex_train_meta: list[ParameterMeta] | None = None
        self._awex_infer_conf: dict[str, Any] | None = None
        self._awex_rank_info = None

    @property
    def parallelism_strategy(self) -> dict:
        from megatron.core import parallel_state as mpu

        tp_size = mpu.get_tensor_model_parallel_world_size()
        cp_size = mpu.get_context_parallel_world_size()
        return {
            "world_size": self._engine.world_size,
            "tp_size": tp_size,
            "pp_size": mpu.get_pipeline_model_parallel_world_size(),
            "dp_size": self._engine.data_parallel_world_size,
            "ep_size": mpu.get_expert_model_parallel_world_size(),
            "dp_replicated": tp_size > 1 or cp_size > 1,
        }

    def get_weight_metadata(
        self, infer_conf: dict[str, Any] | None = None
    ) -> list[ParameterMeta]:
        # Meta/converter init reads live param tensors. After the pre-rollout
        # release_memory, Megatron DDP param_data storages are resize_(0)'d,
        # so every param is a dangling view into freed device memory — any
        # kernel/collective over them is cudaErrorIllegalAddress. Resume
        # weights first (the transfer path below does the same before reading
        # payloads); the post-update release re-offloads them afterwards.
        if "weights" in self._released_tags:
            self.resume_memory(tags=["weights"])
        if infer_conf is not None:
            return self._ensure_awex_converter(infer_conf)[0]
        if self._awex_train_meta is not None:
            return self._awex_train_meta

        rank_info = self._build_rank_info()
        metadata: list[ParameterMeta] = []

        for hf_name, tensor in self._iter_hf_params():
            shape = tuple(tensor.shape)
            numel = int(tensor.numel())
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
                shape=shape,
                numel=numel,
                dtype=tensor.dtype,
                global_offset=tuple([0] * len(shape)),
                sharding_type=ShardingType.NO_SHARDING,
                num_shards=1,
                sharding_dim=0,
            )
            replica = ParameterReplicaMeta(shards=[shard_meta])
            metadata.append(
                ParameterMeta(
                    name=hf_name,
                    global_numel=numel,
                    global_shape=shape,
                    dtype=tensor.dtype,
                    shards=[shard_meta],
                    replicas=[replica],
                )
            )

        return metadata

    def get_local_shard_parameters(
        self, required_names: list[str] | None = None
    ) -> dict[str, torch.Tensor]:
        if self._awex_weight_converter is not None:
            return self._convert_parameters(required_names=required_names)

        required = set(required_names) if required_names else None
        result: dict[str, torch.Tensor] = {}
        for hf_name, tensor in self._iter_hf_params():
            if required is not None and hf_name not in required:
                continue
            result[hf_name] = tensor
        return result

    def _iter_model_params_for_delta(self):
        """Yield model tensors in the same order used by the payload converter."""
        if self._awex_weight_converter is not None:
            from awex.converter.mcore_converter import get_mcore_model_parameters

            seen: set[int] = set()
            for model in self._iter_model_chunks():
                for param in get_mcore_model_parameters(model).values():
                    pid = id(param)
                    if pid in seen:
                        continue
                    seen.add(pid)
                    yield param
            return

        from areal.engine.megatron_utils.megatron import get_named_parameters

        num_moe_experts = getattr(self._engine.tf_config, "num_moe_experts", None)
        seen: set[int] = set()
        for _mcore_name, param in get_named_parameters(
            self._engine.model, num_moe_experts
        ):
            pid = id(param)
            if pid in seen:
                continue
            seen.add(pid)
            yield param

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
        batch_send_recv(
            send_ops=send_ops,
            recv_ops=[],
            blocking=True,
            use_group=awex_wu_use_group(),
        )
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
        if self._colocate_http_client is not None:
            self._colocate_http_client.close()
            self._colocate_http_client = None

    def _build_rank_info(self) -> RankInfo:
        from megatron.core import parallel_state as mpu

        tp_size = mpu.get_tensor_model_parallel_world_size()
        tp_rank = mpu.get_tensor_model_parallel_rank()
        pp_size = mpu.get_pipeline_model_parallel_world_size()
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        ep_size = mpu.get_expert_model_parallel_world_size()
        ep_rank = mpu.get_expert_model_parallel_rank()
        etp_size = mpu.get_expert_tensor_parallel_world_size()
        etp_rank = mpu.get_expert_tensor_parallel_rank()
        cp_size = mpu.get_context_parallel_world_size()
        cp_rank = mpu.get_context_parallel_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", self._engine.rank))

        return RankInfo(
            tp_rank=tp_rank,
            tp_size=tp_size,
            pp_rank=pp_rank,
            pp_size=pp_size,
            dp_size=self._engine.data_parallel_world_size,
            dp_rank=self._engine.data_parallel_rank,
            ep_rank=ep_rank,
            ep_size=ep_size,
            ep_tp_rank=etp_rank,
            ep_tp_size=etp_size,
            attn_tp_rank=tp_rank,
            attn_tp_size=tp_size,
            attn_dp_rank=self._engine.data_parallel_rank,
            world_size=self._engine.world_size,
            global_rank=self._engine.rank,
            local_rank=local_rank,
            engine_rank=0,
            is_infer=False,
            cp_rank=cp_rank,
            cp_size=cp_size,
            cp_mode="ring" if cp_size > 1 else "none",
        )

    def _normalize_awex_infer_conf(
        self, infer_conf: dict[str, Any] | None
    ) -> dict[str, Any]:
        conf = dict(infer_conf or {})
        hf_conf = conf.get("hf_config")
        if isinstance(hf_conf, dict):
            conf["hf_config"] = SimpleNamespace(**hf_conf)
        elif hf_conf is None:
            conf["hf_config"] = self._engine.hf_config
        if "infer_atten_tp_size" not in conf:
            conf["infer_atten_tp_size"] = int(
                conf.get("tp_size", self.parallelism_strategy.get("tp_size", 1))
            )
        if "router_dtype" not in conf:
            conf["router_dtype"] = getattr(
                self._engine.hf_config, "router_dtype", "bf16"
            )
        if "device_backend" not in conf:
            conf["device_backend"] = os.environ.get("AWEX_DEVICE_TYPE", "cuda")
        return conf

    def _get_tf_config(self):
        for model in self._iter_model_chunks():
            for attr in ("transformer_config", "config"):
                cfg = getattr(model, attr, None)
                if cfg is not None:
                    return cfg
        return None

    def _make_awex_engine_shim(self):
        class _EngineShim:
            def __init__(self, engine):
                self.model = engine.model
                if not isinstance(self.model, (list, tuple)):
                    self.model = [self.model]
                self.hf_config = engine.hf_config
                self.enable_debug_mode = False
                self.enable_colocate_mode = True
                self.engine_name = "mcore"
                self.config = {}
                self.meta_server_addr = ""

            def release_memory_occupation(self, tags=None):
                pass

            def resume_memory_occupation(self, tags=None):
                pass

        return _EngineShim(self._engine)

    def _ensure_awex_converter(
        self, infer_conf: dict[str, Any] | None = None
    ) -> tuple[list[ParameterMeta], Any]:
        if (
            self._awex_weight_converter is not None
            and self._awex_train_meta is not None
        ):
            return self._awex_train_meta, self._awex_weight_converter

        from awex.meta.train_meta_resolver import McoreParamMetaResolver
        from awex.models.registry import get_train_weights_converter

        conf = self._normalize_awex_infer_conf(infer_conf or self._awex_infer_conf)
        shim = self._make_awex_engine_shim()
        meta_resolver = McoreParamMetaResolver(shim, self._engine.hf_config, conf)
        train_meta = meta_resolver.get_parameters_meta()

        from awex.sharding.param_sharding import get_rank_info_extractor

        rank_info = get_rank_info_extractor("mcore")()
        converter = get_train_weights_converter(
            "mcore",
            self._engine.hf_config.architectures[0],
            self._engine.hf_config,
            rank_info,
            {
                **conf,
                "train_pp_stage_layer_id_map": (
                    meta_resolver.get_pp_stage_layer_id_map()
                ),
            },
            tf_config=self._get_tf_config(),
        )

        self._awex_infer_conf = conf
        self._awex_rank_info = rank_info
        self._awex_train_meta = train_meta
        self._awex_weight_converter = converter
        logger.info(
            "Initialized AWEX mcore converter for colocate payload (params_meta=%d)",
            len(train_meta),
        )
        return train_meta, converter

    @torch.no_grad()
    def _convert_parameters(
        self,
        required_names: list[str] | None = None,
        theta_by_id: dict[int, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Convert Megatron tensors through AWEX's native mcore converter."""
        from awex.converter.mcore_converter import get_mcore_model_parameters

        if self._awex_weight_converter is None:
            self._ensure_awex_converter()
        converter = self._awex_weight_converter
        assert converter is not None

        required = set(required_names) if required_names else None
        overrides = theta_by_id or {}
        converted: dict[str, torch.Tensor] = {}
        for vp_stage, model in enumerate(self._iter_model_chunks()):
            for name, param in get_mcore_model_parameters(model).items():
                src = overrides.get(id(param), param)
                for hf_name, hf_param in converter.convert_param(
                    name, src.detach(), vp_stage=vp_stage
                ):
                    if required is None or hf_name in required:
                        converted[hf_name] = hf_param.detach()

        hf_config = self._engine.hf_config
        if (
            getattr(hf_config, "tie_word_embeddings", False)
            and self._awex_rank_info is not None
            and self._awex_rank_info.pp_rank == self._awex_rank_info.pp_size - 1
            and "lm_head.weight" not in converted
            and "model.embed_tokens.weight" in converted
            and (required is None or "lm_head.weight" in required)
        ):
            converted["lm_head.weight"] = converted["model.embed_tokens.weight"]

        return converted

    def _live_module_storage_ptrs(self) -> set[int]:
        live_storages: set[int] = set()
        for chunk in self._iter_model_chunks():
            for _, param in chunk.named_parameters():
                live_storages.add(param.untyped_storage().data_ptr())
            for _, buf in chunk.named_buffers():
                live_storages.add(buf.untyped_storage().data_ptr())
        return live_storages

    def _release_owned_payload_tensors(self, tensors: list[torch.Tensor]) -> None:
        if not tensors:
            return
        owned = self._owned_payload_tensors(tensors)
        if not owned:
            logger.info(
                "No owned converted payload tensors to release "
                "(all %d alias live module storage)",
                len(tensors),
            )
            return
        release_tensors(owned)
        logger.info(
            "Released %d/%d converted payload tensors before weights offload",
            len(owned),
            len(tensors),
        )

    def _owned_payload_tensors(self, tensors: list[torch.Tensor]) -> list[torch.Tensor]:
        live_storages = self._live_module_storage_ptrs()
        return [
            tensor
            for tensor in tensors
            if tensor.untyped_storage().data_ptr() not in live_storages
        ]

    def _iter_hf_params(self, theta_by_id: dict[int, torch.Tensor] | None = None):
        """Yield (hf_name, tensor) for every parameter on this rank.

        Uses get_named_parameters + all_gather_param + convert_to_hf to produce
        HF-style per-expert names (e.g. experts.0.gate_proj.weight). The SGLang
        adapter's _unfuse_params converts SGLang's fused w13/w2 format to the
        same per-expert names, so both sides match for the transfer plan.

        ``theta_by_id`` maps ``id(model_param) -> replacement tensor`` (same
        shape/dtype as the mcore param). The AdamW-inversion detector uses it to
        push reconstructed pre-step weights through the EXACT same all_gather +
        convert path as the live payload; a param without an override is
        converted as-is.
        """
        from areal.engine.megatron_utils.megatron import (
            all_gather_param,
            convert_to_hf,
            get_named_parameters,
        )

        overrides = theta_by_id or {}
        num_moe_experts = getattr(self._engine.tf_config, "num_moe_experts", None)
        model_name = self._engine.hf_config.model_type
        tie_word_embeddings = getattr(
            self._engine.hf_config, "tie_word_embeddings", False
        )

        for mcore_name, param in get_named_parameters(
            self._engine.model, num_moe_experts
        ):
            src = overrides.get(id(param), param)
            gathered = all_gather_param(
                mcore_name,
                src,
                fp8_direct_convert=False,
                quantization_config=None,
                duplicated_param_names=self._engine._duplicated_param_names,
            )
            if not isinstance(gathered, torch.Tensor):
                gathered = gathered.data

            for hf_name, tensor in convert_to_hf(
                self._engine.tf_config,
                model_name,
                mcore_name,
                gathered,
            ):
                if tie_word_embeddings and hf_name == "lm_head.weight":
                    continue
                yield hf_name, tensor.detach()

    def _get_inner_optimizers(self):
        """Inner per-chunk optimizers (unwrap Megatron ChainedOptimizer).

        Used by the AdamW-inversion detector to read resident AdamW moments.
        """
        optimizer = self._engine.optimizer
        if optimizer is None:
            return []

        def flatten(opt):
            chained = getattr(opt, "chained_optimizers", None)
            if chained is not None:
                result = []
                for child in chained:
                    result.extend(flatten(child))
                return result
            optimizers = getattr(opt, "optimizers", None)
            if optimizers is not None:
                result = []
                for child in optimizers:
                    result.extend(flatten(child))
                return result
            return [opt]

        return flatten(optimizer)

    @staticmethod
    def _get_optimizer_state_owner(opt):
        """Return the torch optimizer object that owns ``state``.

        Megatron's ChainedOptimizer exposes an ``optimizer`` property inherited
        from MegatronOptimizer, but accessing it asserts when there is more than
        one child optimizer. Callers should flatten chains first, then use this
        helper for wrappers such as DistributedOptimizer.
        """
        state = getattr(opt, "state", None)
        if state is not None:
            return opt
        try:
            base_opt = opt.optimizer
        except AttributeError:
            return opt
        except AssertionError as exc:
            logger.warning(
                "Skipping optimizer state owner lookup for %s: %s",
                type(opt).__name__,
                exc,
            )
            return None
        return base_opt

    @torch.no_grad()
    def _sync_model_params_from_optimizer(self) -> None:
        """Make the HF payload read see the latest optimizer main params.

        Megatron's distributed optimizer updates fp32 main-param shards, copies
        them into DDP param buffers, then all-gathers into the model-visible
        params. Colocate weight exchange can run after offload/resume, so do a
        conservative copy + synchronous param sync before reading HF tensors.
        """
        copied = 0
        gathered = 0
        for opt in self._get_inner_optimizers():
            copy_fn = getattr(opt, "_copy_main_params_to_model_params", None)
            if copy_fn is None:
                copy_fn = getattr(opt, "_copy_main_params_to_param_buffer", None)
            if copy_fn is None:
                continue
            copy_fn()
            copied += 1

            gather_fn = getattr(
                opt, "_reset_metadata_and_sync_gather_all_model_params", None
            )
            if gather_fn is not None:
                gather_fn(force_sync=True)
                gathered += 1

        model = self._engine.model
        if model is None:
            return
        model_chunks = model if isinstance(model, (list, tuple)) else [model]
        synced = 0
        if gathered == 0:
            for model_chunk in model_chunks:
                start_param_sync = getattr(model_chunk, "start_param_sync", None)
                if start_param_sync is None:
                    continue
                start_param_sync(force_sync=True)
                synced += 1

        if copied or gathered or synced:
            torch.cuda.synchronize()
            logger.info(
                "Synced Megatron optimizer main params before AWEX payload read "
                "(copied=%d, optimizer_gather=%d, param_sync=%d)",
                copied,
                gathered,
                synced,
            )

    @torch.no_grad()
    def _convert_hf_with_overrides(
        self, theta_by_id: dict[int, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Convert mcore params to HF, substituting reconstructed pre-step
        weights by ``id(model_param)``.

        The inversion detector builds the previous-version HF payload through
        this (the same all_gather + convert path as the live payload), then
        bitwise-compares it against the current payload to get change masks.
        """
        if self._awex_weight_converter is not None:
            return self._convert_parameters(theta_by_id=theta_by_id)
        return dict(self._iter_hf_params(theta_by_id))

    # ── Colocated weight transfer methods ─────────────────────────────────

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
        infer_conf: dict[str, Any] | None = None,
    ) -> None:
        self._colocate_pair_name = pair_name
        self._colocate_kv_store_url = kv_store_url
        self._colocate_transfer_rank = transfer_rank
        self._colocate_infer_world_size = infer_world_size
        self._colocate_admin_api_key = admin_api_key
        self._colocate_timeout_s = timeout_s
        if self._colocate_http_client is None:
            self._colocate_http_client = httpx.Client()
        if infer_conf is not None:
            self._ensure_awex_converter(infer_conf)
        self._register_colocate_training_device()
        logger.info(
            "Initialized colocate weight update for pair '%s', transfer_rank=%d",
            pair_name,
            transfer_rank,
        )
        local_delta = delta_transfer_enabled()
        if (
            expected_delta_enabled is not None
            and bool(expected_delta_enabled) != local_delta
        ):
            raise ValueError(
                "Colocate delta config mismatch on training rank "
                f"{transfer_rank}: expected={expected_delta_enabled}, "
                f"local={local_delta}. Check DTE_DELTA_TRANSFER propagation."
            )
        if local_delta and self._delta_tracker is None:
            # Fail fast: surface a missing dte / bad config at init, not mid-run.
            self._delta_tracker = make_delta_tracker()
            logger.info("colocate delta enabled (sender); dte DeltaTracker ready")

    def _register_colocate_training_device(self) -> None:
        assert self._colocate_http_client is not None
        ip_address = get_colocate_ip_address()
        device_id = get_physical_cuda_device_id()
        self._colocate_device_ip = ip_address
        self._colocate_device_id = device_id

        pair_name = self._colocate_pair_name
        transfer_rank = self._colocate_transfer_rank
        kv_store_url = self._colocate_kv_store_url
        auth_headers = {"Authorization": f"Bearer {self._colocate_admin_api_key}"}
        device_key = device_mapping_key(ip_address, device_id)
        value = {
            "ip": ip_address,
            "device_id": device_id,
            "train_rank": transfer_rank,
        }
        resp = self._colocate_http_client.put(
            f"{kv_store_url}/weight_meta/{pair_name}/"
            f"colocate_train_rank_by_device_{device_key}",
            json={"value": transfer_rank},
            headers=auth_headers,
            timeout=self._colocate_timeout_s,
        )
        resp.raise_for_status()
        resp = self._colocate_http_client.put(
            f"{kv_store_url}/weight_meta/{pair_name}/"
            f"colocate_train_device_by_rank_{transfer_rank}",
            json={"value": value},
            headers=auth_headers,
            timeout=self._colocate_timeout_s,
        )
        resp.raise_for_status()
        logger.info(
            "Registered colocate training device mapping: ip=%s, device=%s, "
            "train_rank=%d",
            ip_address,
            device_id,
            transfer_rank,
        )

    def execute_colocate_weight_update(self, version: int) -> None:
        with self._colocate_lock:
            self._execute_colocate_weight_update_locked(version)

    def _ensure_delta_components(self) -> None:
        if self._delta_tracker is None:
            self._delta_tracker = make_delta_tracker()
        if self._delta_detector is None:
            self._delta_detector = build_detector(delta_detector_mode(), self)

    def _needs_inversion_sync_before_payload(self, version: int) -> bool:
        """Whether this step will need optimizer-derived inversion masks.

        Initial/anchor full syncs should follow the dense AWEX path and avoid
        the extra Megatron param-sync peak. The sync is only required once we
        are about to compute AdamW-inversion masks for a real delta payload.
        """
        if not delta_transfer_enabled() or delta_detector_mode() != "inversion":
            return False
        self._ensure_delta_components()
        reason = self._delta_tracker.full_sync_reason(version)
        if reason is not None:
            logger.info(
                "Skipping Megatron optimizer param sync before payload v%d: "
                "delta full sync is required (%s)",
                version,
                reason,
            )
            return False
        has_watermark = getattr(
            self._delta_detector, "has_synced_watermark", lambda: False
        )
        if not has_watermark():
            logger.info(
                "Skipping Megatron optimizer param sync before payload v%d: "
                "inversion watermark missing; first payload will be full sync",
                version,
            )
            return False
        return True

    @staticmethod
    def _colocate_full_group_max_bytes() -> int:
        raw = os.environ.get("DTE_COLOCATE_FULL_GROUP_MAX_BYTES")
        if raw is None or raw.strip() == "":
            raw = os.environ.get("AWEX_COLOCATE_FULL_GROUP_MAX_BYTES")
        if raw is None or raw.strip() == "":
            return 512 * 1024 * 1024
        try:
            return max(int(raw), 1)
        except ValueError as exc:
            raise ValueError(
                "DTE_COLOCATE_FULL_GROUP_MAX_BYTES must be an integer byte count"
            ) from exc

    def _full_tensors_for_ipc(
        self,
        tensors: list[torch.Tensor],
        names: list[str] | None = None,
    ) -> tuple[list[torch.Tensor], list[dict]]:
        """Build bounded AWEX groups for colocated full-sync IPC.

        The first delta-transfer frame is a dense full sync. It must use
        exporter-owned storage because training weights are offloaded before the
        inference process maps the CUDA IPC handles. One handle per parameter is
        fragile for large colocated Megatron shards, while AWEX's default 5 GiB
        groups can create too high a temporary cat+clone peak. This path keeps
        AWEX's cat+clone ownership invariant but caps each same-shape group.
        """
        if names is not None and len(names) != len(tensors):
            raise ValueError(
                "names must match tensors when building colocate IPC payload"
            )

        max_group_bytes = self._colocate_full_group_max_bytes()
        live_storages = self._live_module_storage_ptrs()
        group_tensors: list[torch.Tensor] = []
        metadata: list[dict] = []
        packed_live = 0
        packed_owned = 0
        made_contiguous = 0
        zero_sized = 0
        oversized_groups = 0
        # Bucket by dtype only. Reconstruction slices each entry by
        # offset/size and reshapes from per-entry metadata, so same-shape
        # packing is not required for correctness — and keying by shape
        # degenerates to one group per parameter on sparse delta payloads
        # (every param's values/indices have unique lengths), recreating the
        # per-parameter handle flood this grouping exists to avoid (observed:
        # 4636 groups for 4888 params, receiver cudaIpcOpenMemHandle driver
        # OOM with >50GB free).
        buckets: dict[torch.dtype, list[int]] = {}

        def append_zero_group(original_index: int, source: torch.Tensor) -> None:
            nonlocal zero_sized
            group_index = len(group_tensors)
            group_tensors.append(
                torch.empty((1,), dtype=source.dtype, device=source.device)
            )
            metadata.append(
                {
                    "original_index": original_index,
                    "shape": source.shape,
                    "dtype": source.dtype,
                    "group_index": group_index,
                    "offset": 0,
                    "size": 0,
                }
            )
            zero_sized += 1

        for original_index, tensor in enumerate(tensors):
            source = tensor.detach()
            if source.numel() == 0:
                append_zero_group(original_index, source)
                continue
            buckets.setdefault(source.dtype, []).append(original_index)

        def finalize_group(
            current: list[tuple[int, torch.Tensor, torch.Size, torch.dtype, int]],
            current_bytes: int,
        ) -> None:
            nonlocal oversized_groups
            if not current:
                return
            if current_bytes > max_group_bytes:
                oversized_groups += 1
            group_index = len(group_tensors)
            flat_tensors = [entry[1].reshape(-1) for entry in current]
            group_tensor = torch.cat(flat_tensors, dim=0).clone()
            group_tensors.append(group_tensor)
            offset = 0
            for original_index, _flat, shape, dtype, size in current:
                metadata.append(
                    {
                        "original_index": original_index,
                        "shape": shape,
                        "dtype": dtype,
                        "group_index": group_index,
                        "offset": offset,
                        "size": size,
                    }
                )
                offset += size

        for indices in buckets.values():
            current: list[tuple[int, torch.Tensor, torch.Size, torch.dtype, int]] = []
            current_bytes = 0
            for original_index in indices:
                source = tensors[original_index].detach()
                tensor_bytes = source.numel() * source.element_size()
                if current and current_bytes + tensor_bytes > max_group_bytes:
                    finalize_group(current, current_bytes)
                    current = []
                    current_bytes = 0

                aliases_live_storage = (
                    source.untyped_storage().data_ptr() in live_storages
                )
                compact = source.contiguous()
                if (
                    compact.untyped_storage().data_ptr()
                    != source.untyped_storage().data_ptr()
                ):
                    made_contiguous += 1
                if aliases_live_storage:
                    packed_live += 1
                else:
                    packed_owned += 1
                current.append(
                    (
                        original_index,
                        compact,
                        source.shape,
                        source.dtype,
                        source.numel(),
                    )
                )
                current_bytes += tensor_bytes
            finalize_group(current, current_bytes)

        logger.info(
            "Built bounded colocate full IPC payload: params=%d, groups=%d, "
            "max_group_bytes=%d, packed_live_sources=%d, packed_owned_sources=%d, "
            "made_contiguous=%d, zero_sized=%d, oversized_groups=%d",
            len(tensors),
            len(group_tensors),
            max_group_bytes,
            packed_live,
            packed_owned,
            made_contiguous,
            zero_sized,
            oversized_groups,
        )
        return group_tensors, metadata

    def _ungrouped_tensors_for_ipc(
        self,
        tensors: list[torch.Tensor],
        names: list[str] | None = None,
    ) -> tuple[list[torch.Tensor], list[dict]]:
        """Build one tensor per IPC group.

        Kept for targeted diagnostics; full sync uses ``_full_tensors_for_ipc``
        so large colocated shards do not export hundreds of CUDA IPC handles.
        """
        live_storages = self._live_module_storage_ptrs()
        group_tensors: list[torch.Tensor] = []
        metadata: list[dict] = []
        cloned_live = 0
        cloned_owned = 0
        made_contiguous = 0
        zero_sized = 0
        for original_index, tensor in enumerate(tensors):
            source = tensor.detach()
            aliases_live_storage = source.untyped_storage().data_ptr() in live_storages
            if names is not None and original_index >= len(names):
                raise ValueError(
                    "names must match tensors when building colocate IPC payload"
                )
            if source.numel() == 0:
                group_tensor = torch.empty(
                    (1,),
                    dtype=source.dtype,
                    device=source.device,
                )
                zero_sized += 1
            else:
                compact = source.contiguous()
                if (
                    compact.untyped_storage().data_ptr()
                    == source.untyped_storage().data_ptr()
                ):
                    group_tensor = compact.clone(memory_format=torch.contiguous_format)
                    if aliases_live_storage:
                        cloned_live += 1
                    else:
                        cloned_owned += 1
                else:
                    group_tensor = compact
                    made_contiguous += 1

            if aliases_live_storage and source.numel() == 0:
                cloned_live += 1
            group_tensors.append(group_tensor)
            metadata.append(
                {
                    "original_index": original_index,
                    "shape": source.shape,
                    "dtype": source.dtype,
                    "group_index": original_index,
                    "offset": 0,
                    "size": source.numel(),
                }
            )
        logger.info(
            "Built ungrouped colocate IPC payload: params=%d, "
            "cloned_live_aliases=%d, cloned_owned=%d, "
            "made_contiguous=%d, zero_sized=%d",
            len(group_tensors),
            cloned_live,
            cloned_owned,
            made_contiguous,
            zero_sized,
        )
        return group_tensors, metadata

    @staticmethod
    def _ipc_tensor_debug(name: str, tensor: torch.Tensor) -> str:
        try:
            storage_size = tensor.untyped_storage().size()
            storage_ptr = tensor.untyped_storage().data_ptr()
        except Exception:
            storage_size = "?"
            storage_ptr = "?"
        return (
            f"name={name} shape={tuple(tensor.shape)} dtype={tensor.dtype} "
            f"device={tensor.device} numel={tensor.numel()} "
            f"contiguous={tensor.is_contiguous()} "
            f"storage_offset={tensor.storage_offset()} "
            f"storage_size={storage_size} data_ptr={storage_ptr}"
        )

    def _share_tensors_for_cuda_ipc(
        self,
        tensors: list[torch.Tensor],
        names: list[str],
        metadata: list[dict],
    ) -> list[torch.Tensor]:
        shared: list[torch.Tensor] = []
        for group_index, tensor in enumerate(tensors):
            meta_name = names[metadata[group_index]["original_index"]]
            try:
                shared.append(tensor.share_memory_())
            except Exception as exc:
                raise RuntimeError(
                    "Failed to prepare colocate CUDA IPC tensor: "
                    + self._ipc_tensor_debug(meta_name, tensor)
                ) from exc
        return shared

    def _validate_colocate_cuda_ipc_payload(
        self,
        tensors: list[torch.Tensor],
        names: list[str],
        metadata: list[dict],
    ) -> None:
        env = os.environ.get("DTE_VALIDATE_COLOCATE_IPC", "")
        if env.strip().lower() not in {"1", "true", "yes", "on"}:
            return
        for group_index, tensor in enumerate(tensors):
            meta = dict(metadata[group_index])
            name = names[meta["original_index"]]
            meta["original_index"] = 0
            meta["group_index"] = 0
            try:
                cuda_ipc_serialize(([tensor], [meta], [name]))
            except Exception as exc:
                raise RuntimeError(
                    "Colocate CUDA IPC validation failed: "
                    + self._ipc_tensor_debug(name, tensor)
                ) from exc
        logger.info("Validated %d colocate CUDA IPC tensor groups", len(tensors))

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
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()

        sync_env = os.environ.get("DTE_SYNC_MODEL_PARAMS_BEFORE_PAYLOAD")
        if sync_env is None or sync_env.strip() == "":
            sync_env = os.environ.get("AWEX_SYNC_MODEL_PARAMS_BEFORE_PAYLOAD")
        if sync_env is None or sync_env.strip() == "":
            # AdamW inversion compares and sends the BF16 HF payload, so the
            # model-visible Megatron buffers must reflect the latest fp32 main
            # params before we read them. This is a GPU-side refresh, not a CPU
            # snapshot.
            sync_before_payload = (
                delta_transfer_enabled() and delta_detector_mode() == "inversion"
            )
            sync_uses_inversion_gate = True
        else:
            sync_before_payload = sync_env.strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            sync_uses_inversion_gate = False
        did_resume_weights_for_sync = False
        if sync_before_payload:
            should_sync = (
                self._needs_inversion_sync_before_payload(version)
                if sync_uses_inversion_gate
                else True
            )
            if should_sync and weights_offloaded:
                self.resume_memory(tags=["weights"])
                did_resume_weights_for_sync = True
            if should_sync:
                self._sync_model_params_from_optimizer()
        if weights_offloaded and not did_resume_weights_for_sync:
            self.resume_memory(tags=["weights"])
        params = self.get_local_shard_parameters()
        zero_copy_full_payload = False
        if delta_transfer_enabled():
            names, tensors, zero_copy_full_payload = self._delta_encode(params, version)
        else:
            tensors = list(params.values())
            names = list(params.keys())

        delta_synced_state = self._delta_capture_synced_state(params)
        self.release_memory(tags=["optimizer"])
        self._release_grad_memory()

        if zero_copy_full_payload:
            group_tensors, metadata = self._full_tensors_for_ipc(tensors, names)
            logger.info(
                "Using bounded colocate full payload for v%d (params=%d, groups=%d)",
                version,
                len(tensors),
                len(group_tensors),
            )
        else:
            group_tensors, metadata = group_tensors_by_shape_and_dtype(tensors)
        torch.cuda.synchronize()

        if not zero_copy_full_payload:
            self._release_owned_payload_tensors(tensors)
        del tensors, params

        if zero_copy_full_payload:
            live_storages = self._live_module_storage_ptrs()
            live_aliases = sum(
                t.untyped_storage().data_ptr() in live_storages for t in group_tensors
            )
            if live_aliases:
                raise RuntimeError(
                    "Ungrouped colocate full payload still aliases live model "
                    f"storage for {live_aliases} tensors; refusing to offload "
                    "weights before IPC transfer"
                )

        self.release_memory(tags=["weights"])

        offloaded_key = (
            f"colocate_train_weights_offloaded_rank{transfer_rank}_{version}"
        )
        client.put(
            f"{kv_store_url}/weight_meta/{pair_name}/{offloaded_key}",
            json={"value": True},
            headers=auth_headers,
            timeout=timeout_s,
        )
        logger.info(
            "Signaled colocate train weights offloaded for v%d, rank %d",
            version,
            transfer_rank,
        )

        kv_key = f"colocate_weights_rank{transfer_rank}_{version}"
        done_key = f"colocate_done_rank{transfer_rank}_{version}"
        group_shared: list[torch.Tensor] = []
        serialized_weights: bytes | None = None
        try:
            group_shared = self._share_tensors_for_cuda_ipc(
                group_tensors,
                names,
                metadata,
            )
            self._validate_colocate_cuda_ipc_payload(group_shared, names, metadata)
            try:
                serialized_weights = cuda_ipc_serialize((group_shared, metadata, names))
            except Exception as exc:
                examples = "; ".join(
                    self._ipc_tensor_debug(
                        names[metadata[i]["original_index"]],
                        group_shared[i],
                    )
                    for i in range(min(3, len(group_shared)))
                )
                raise RuntimeError(
                    "Failed to serialize colocate CUDA IPC payload "
                    f"(groups={len(group_shared)}, examples={examples})"
                ) from exc
            torch.cuda.synchronize()

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

            self._delta_mark_synced(version, delta_synced_state)
        finally:
            release_tensors(group_tensors)
            if group_shared:
                release_tensors(group_shared)
            del group_shared, group_tensors, serialized_weights
            torch.cuda.synchronize()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

    def _delta_capture_synced_state(
        self, payload_params: dict[str, torch.Tensor] | None = None
    ):
        """Capture detector watermarks before colocate weight offload.

        The AdamW inversion detector needs to fingerprint model-visible params.
        Those tensors have empty storage after ``release_memory(tags=["weights"])``,
        so capture before offload and commit only after inference confirms the
        payload was applied.
        """
        if not delta_transfer_enabled() or self._delta_detector is None:
            return None
        capture = getattr(self._delta_detector, "capture_synced_state", None)
        if capture is None:
            return None
        return capture(payload_params)

    def _delta_mark_synced(self, version: int, captured_state=None) -> None:
        """Let an external delta detector record post-sync watermarks."""
        if not delta_transfer_enabled() or self._delta_detector is None:
            return
        mark_synced = getattr(self._delta_detector, "mark_synced", None)
        if mark_synced is not None:
            mark_synced(version, captured_state)

    def _delta_encode(
        self, params: dict[str, torch.Tensor], version: int
    ) -> tuple[list[str], list[torch.Tensor], bool]:
        """Encode ``params`` as a dte delta/full payload (sender side).

        Mirrors ``dte.engine.DeltaEngine.push`` but ships the payload through the
        existing cuda-IPC colocate channel instead of a dte transport. A full
        sync ships plain tensors (header-less, shapes unchanged); a delta ships
        dte's sparse payload (header + ``w@delta_idx``/``w@delta_val`` + dense
        fallback). The sglang receiver reconstructs full weights before applying,
        so delta stays transparent to the cross-rank NCCL reshard.

        The change mask comes from the configured detector (DTE_DELTA_DETECTOR):
        ``snapshot`` (default) returns None so dte diffs its CPU baseline;
        ``inversion`` reconstructs pre-step weights from the optimizer's resident
        moments (zero snapshot memory). Inversion infeasible this step -> dense.
        """
        self._ensure_delta_components()
        inversion = self._delta_detector.name == "inversion"
        names = list(params.keys())
        tensors = list(params.values())
        params_list = list(params.items())

        # Full sync on the first frame (not yet seeded) or a forced anchor.
        reason = self._delta_tracker.full_sync_reason(version)
        # Inversion: compute masks only when this rank would otherwise ship a
        # delta; if infeasible this step (precision-aware / step<1 / recover)
        # compute_masks returns None and we fall back to a dense full sync.
        masks = None
        if inversion and reason is None:
            has_watermark = getattr(
                self._delta_detector, "has_synced_watermark", lambda: False
            )
            if not has_watermark():
                reason = "initial_full"
        if inversion and reason is None:
            masks = self._delta_detector.compute_masks(names, tensors, version)
            if masks is None:
                reason = "inversion_infeasible"

        if reason is not None:
            # Full sync: ship plain tensors. Snapshot mode re-seeds its baseline;
            # inversion keeps no resident model snapshot.
            self._delta_tracker.seed(
                params_list,
                version,
                store_snapshot=not inversion,
            )
            logger.info("colocate delta v%d: FULL sync (%s)", version, reason)
            return names, tensors, True

        encoded = self._delta_tracker.encode(params_list, version, masks=masks)
        logger.info(
            "colocate delta v%d [%s]: changed %d/%d (%.2f%%) "
            "sparse=%d dense_fallback=%d unchanged=%d "
            "payload=%.1fMB vs dense=%.1fMB",
            version,
            self._delta_detector.name,
            encoded.changed_elements,
            encoded.total_elements,
            100.0 * encoded.changed_elements / max(encoded.total_elements, 1),
            encoded.num_sparse,
            encoded.num_dense_fallback,
            encoded.num_unchanged,
            encoded.payload_bytes / 1e6,
            encoded.dense_bytes / 1e6,
        )
        return encoded.names, encoded.tensors, False

    def release_memory(self, tags: list[str] | None = None) -> None:
        """Release GPU memory for specified tags by offloading to CPU.

        Supported tags:
            - "optimizer": Offload optimizer state tensors (exp_avg, exp_avg_sq, etc.)
            - "weights": Offload model parameters
        """
        tags = tags or ["optimizer", "weights"]
        tags_to_release = [t for t in tags if t not in self._released_tags]
        if not tags_to_release:
            logger.info("release_memory: tags=%s already released, skipping", tags)
            return

        logger.info("release_memory: offloading tags=%s", tags_to_release)

        if "optimizer" in tags_to_release:
            self._offload_optimizer_states()
            self._released_tags.add("optimizer")

        if "weights" in tags_to_release:
            self._offload_model_weights()
            self._released_tags.add("weights")

        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("release_memory: done for tags=%s", tags_to_release)

    def resume_memory(self, tags: list[str] | None = None) -> None:
        """Resume GPU memory for specified tags by reloading from CPU.

        Supported tags:
            - "optimizer": Reload optimizer state tensors to GPU
            - "weights": Reload model parameters to GPU
        """
        tags = tags or ["optimizer", "weights"]
        tags_to_resume = [t for t in tags if t in self._released_tags]
        if not tags_to_resume:
            logger.info("resume_memory: tags=%s not released, skipping", tags)
            return

        logger.info("resume_memory: reloading tags=%s", tags_to_resume)

        if "weights" in tags_to_resume:
            self._reload_model_weights()
            self._released_tags.discard("weights")

        if "optimizer" in tags_to_resume:
            self._reload_optimizer_states()
            self._released_tags.discard("optimizer")

        torch.cuda.synchronize()
        logger.info("resume_memory: done for tags=%s", tags_to_resume)

    def _offload_optimizer_states(self) -> None:
        """Move optimizer state tensors to CPU, keeping references for reload."""
        optimizer = self._engine.optimizer
        if optimizer is None:
            logger.warning("No optimizer found, skipping optimizer offload")
            return
        if os.environ.get("AWEX_OPT_OFFLOAD_VIA_HDO", "").strip() == "1" and hasattr(
            optimizer, "offload_to_cpu"
        ):
            optimizer.offload_to_cpu()
            logger.info("Offloaded optimizer via offload_to_cpu()")
            return

        count = 0
        for opt in self._get_inner_optimizers():
            fp32_groups = getattr(opt, "shard_fp32_from_float16_groups", None)
            if fp32_groups is not None:
                for group in fp32_groups:
                    if isinstance(group, list):
                        for tensor in group:
                            if tensor is not None and tensor.data.is_cuda:
                                tensor.data = tensor.data.to("cpu", non_blocking=True)
                                count += 1
                    elif group is not None and group.data.is_cuda:
                        group.data = group.data.to("cpu", non_blocking=True)
                        count += 1

            base_opt = self._get_optimizer_state_owner(opt)
            if base_opt is None:
                continue
            state_dict = getattr(base_opt, "state", None)
            if state_dict is None:
                logger.warning(
                    "Optimizer %s has no state dict; skipping optimizer offload",
                    type(base_opt).__name__,
                )
                continue
            for state in state_dict.values():
                for key in ("exp_avg", "exp_avg_sq"):
                    val = state.get(key)
                    if isinstance(val, torch.Tensor) and val.is_cuda:
                        state[key] = val.to("cpu", non_blocking=True)
                        count += 1

        try:
            from transformer_engine.pytorch.module.base import _dummy_wgrads

            purged = len(_dummy_wgrads)
            for key in list(_dummy_wgrads):
                del _dummy_wgrads[key]
            if purged:
                logger.info("Purged %d TE _dummy_wgrads cache entries", purged)
        except ImportError:
            pass

        torch.cuda.synchronize()
        logger.info("Offloaded %d optimizer state tensors to CPU", count)

    def _reload_optimizer_states(self) -> None:
        """Restore optimizer state tensors from CPU back to GPU."""
        optimizer = self._engine.optimizer
        if optimizer is None:
            return
        if os.environ.get("AWEX_OPT_OFFLOAD_VIA_HDO", "").strip() == "1" and hasattr(
            optimizer, "restore_from_cpu"
        ):
            optimizer.restore_from_cpu()
            logger.info("Reloaded optimizer via restore_from_cpu()")
            return

        device = self._engine.device
        count = 0
        for opt in self._get_inner_optimizers():
            fp32_groups = getattr(opt, "shard_fp32_from_float16_groups", None)
            if fp32_groups is not None:
                for group in fp32_groups:
                    if isinstance(group, list):
                        for tensor in group:
                            if tensor is not None and not tensor.data.is_cuda:
                                tensor.data = tensor.data.to(device, non_blocking=True)
                                count += 1
                    elif group is not None and not group.data.is_cuda:
                        group.data = group.data.to(device, non_blocking=True)
                        count += 1

            base_opt = self._get_optimizer_state_owner(opt)
            if base_opt is None:
                continue
            state_dict = getattr(base_opt, "state", None)
            if state_dict is None:
                continue
            for param, state in state_dict.items():
                legacy_state = self._offloaded_optimizer_states.get(param)
                for key in ("exp_avg", "exp_avg_sq"):
                    if legacy_state is not None and key in legacy_state:
                        state[key] = legacy_state[key].to(device, non_blocking=True)
                        count += 1
                        continue
                    val = state.get(key)
                    if isinstance(val, torch.Tensor) and not val.is_cuda:
                        state[key] = val.to(device, non_blocking=True)
                        count += 1
        self._offloaded_optimizer_states.clear()
        torch.cuda.synchronize()
        logger.info("Reloaded %d optimizer state tensors to GPU", count)

    def _iter_model_chunks(self):
        model = self._engine.model
        if model is None:
            return []
        return model if isinstance(model, (list, tuple)) else [model]

    @staticmethod
    def _iter_buffer_list(value):
        if value is None or callable(value):
            return []
        if isinstance(value, dict):
            return value.values()
        return value

    def _iter_ddp_buffers(self, chunk):
        """Yield Megatron DDP buffers without depending on a concrete DDP class."""
        seen: set[int] = set()
        for attr in ("buffers", "expert_parallel_buffers"):
            for buf in self._iter_buffer_list(getattr(chunk, attr, None)):
                if buf is None:
                    continue
                buf_id = id(buf)
                if buf_id in seen:
                    continue
                seen.add(buf_id)
                if hasattr(buf, "param_data") or hasattr(buf, "grad_data"):
                    yield buf

    def _offload_model_weights(self) -> None:
        """Move model parameters to CPU, preserving Megatron DDP buffer views."""
        if self._engine.model is None:
            return

        count = 0
        for chunk_idx, chunk in enumerate(self._iter_model_chunks()):
            ddp_buffers = list(self._iter_ddp_buffers(chunk))
            if ddp_buffers:
                for buf in ddp_buffers:
                    offload_to_cpu = getattr(buf, "offload_to_cpu", None)
                    if offload_to_cpu is not None:
                        offload_to_cpu()
                        count += 1
                        continue

                    param_data = getattr(buf, "param_data", None)
                    if param_data is not None:
                        param_storage = param_data.storage()
                        if param_storage.size() > 0:
                            if not hasattr(param_data, "cpu_data"):
                                try:
                                    param_data.cpu_data = torch.empty(
                                        param_data.data.shape,
                                        dtype=param_data.data.dtype,
                                        pin_memory=torch.cuda.is_available(),
                                        device="cpu",
                                    )
                                except RuntimeError:
                                    param_data.cpu_data = torch.empty(
                                        param_data.data.shape,
                                        dtype=param_data.data.dtype,
                                        device="cpu",
                                    )
                            param_data.cpu_data.copy_(
                                param_data.data, non_blocking=True
                            )
                            buf.param_data_size = param_storage.size()
                            param_storage.resize_(0)
                            count += 1

                    grad_data = getattr(buf, "grad_data", None)
                    if grad_data is not None and grad_data.storage().size() > 0:
                        buf.grad_data_size = grad_data.storage().size()
                        grad_data.storage().resize_(0)
                continue

            prefix = f"chunk{chunk_idx}."
            for name, param in chunk.named_parameters():
                if param.is_cuda:
                    self._offloaded_weights[prefix + name] = param.data.detach().to(
                        "cpu", non_blocking=True
                    )
                    param.data = torch.empty(0, device="cpu")
                    count += 1

        logger.info(
            "Offloaded %d model weight buffers/tensors to CPU",
            count,
        )

    def _reload_model_weights(self) -> None:
        """Restore model parameters without breaking Megatron DDP buffer views."""
        if not self._offloaded_weights and self._engine.model is None:
            return
        if self._engine.model is None:
            return

        device = self._engine.device
        count = 0
        for chunk_idx, chunk in enumerate(self._iter_model_chunks()):
            ddp_buffers = list(self._iter_ddp_buffers(chunk))
            if ddp_buffers:
                for buf in ddp_buffers:
                    reload_from_cpu = getattr(buf, "reload_from_cpu", None)
                    if reload_from_cpu is not None:
                        reload_from_cpu(move_grads=False)
                        count += 1
                        continue

                    param_data = getattr(buf, "param_data", None)
                    if param_data is None:
                        continue
                    if (
                        hasattr(buf, "param_data_size")
                        and param_data.storage().size() == 0
                    ):
                        param_data.storage().resize_(buf.param_data_size)
                    cpu_data = getattr(param_data, "cpu_data", None)
                    if cpu_data is not None:
                        param_data.copy_(cpu_data, non_blocking=True)
                        count += 1
                continue

            prefix = f"chunk{chunk_idx}."
            for name, param in chunk.named_parameters():
                key = prefix + name
                if key in self._offloaded_weights:
                    param.data = self._offloaded_weights[key].to(
                        device, non_blocking=True
                    )
                    count += 1

        self._offloaded_weights.clear()
        logger.info("Reloaded %d model weight buffers/tensors to GPU", count)

    def _release_grad_memory(self) -> None:
        """Release Megatron DDP grad buffers until the next training batch."""
        if self._engine.model is None:
            return

        count = 0
        for chunk in self._iter_model_chunks():
            for buf in self._iter_ddp_buffers(chunk):
                grad_data = getattr(buf, "grad_data", None)
                if grad_data is None or grad_data.storage().size() == 0:
                    continue
                buf.grad_data_size = grad_data.storage().size()
                grad_data.storage().resize_(0)
                count += 1
        if count:
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("Released %d grad buffers", count)
        else:
            logger.info("No Megatron grad buffers released before weight payload read")

    def ensure_grad_buffers(self) -> None:
        """Reallocate Megatron DDP grad buffers released with train weights."""
        if self._engine.model is None:
            return

        count = 0
        for chunk in self._iter_model_chunks():
            for buf in self._iter_ddp_buffers(chunk):
                grad_data = getattr(buf, "grad_data", None)
                if (
                    grad_data is not None
                    and hasattr(buf, "grad_data_size")
                    and grad_data.storage().size() == 0
                ):
                    grad_data.storage().resize_(buf.grad_data_size)
                    grad_data.zero_()
                    count += 1
        if count:
            torch.cuda.synchronize()
            logger.info("Allocated %d grad buffers for training", count)
