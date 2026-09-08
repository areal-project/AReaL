# SPDX-License-Identifier: Apache-2.0
"""Shared AWEX training adapter for AReaL Megatron engines."""

from __future__ import annotations

import gc
import os
import threading
import time
from typing import TYPE_CHECKING

import httpx
import torch
import torch.distributed as dist
from awex.meta.weight_meta import ParameterMeta
from awex.models.registry import get_train_weights_converter
from awex.sharding.param_sharding import get_rank_info_extractor
from awex.transfer.nccl_comm import batch_send_recv, nccl_build_send_ops
from awex.transfer.nccl_stream_batch import NcclColocateStreamBatchTransport
from awex.transfer.transfer_plan import TransferPlan, TransferPlanBuilder, slice_tensor
from awex.util.tensor_util import (
    cuda_ipc_serialize,
    group_tensors_by_shape_and_dtype,
)

from areal.engine.awex.delta_config import (
    DTERuntimeConfig,
    synchronize_wire_dtypes,
    validate_dte_world_size,
)
from areal.engine.awex.delta_detect import AdamWInversionDetector
from areal.engine.awex.metadata import (
    awex_wu_use_group,
    fetch_kv_metadata,
)
from areal.engine.awex.nccl_group import (
    init_weights_update_group,
    setup_batch_isend_irecv,
)
from areal.engine.awex.training_adapter import (
    AwexTrainingAdapter,
)
from areal.engine.awex.utils import (
    awex_colocate_timeout_s,
    resolve_physical_gpu_id,
)
from areal.engine.megatron_utils.weight_residency import MegatronWeightResidency
from areal.utils import logging

if TYPE_CHECKING:
    from areal.engine.megatron_engine import MegatronEngine

logger = logging.getLogger("AwexMegatronAdapter")


def _get_tf_config(models):
    if not isinstance(models, (list, tuple)):
        models = [models]
    for model in models:
        for attr in ("transformer_config", "config"):
            config = getattr(model, attr, None)
            if config is not None:
                return config
    return None


class AwexMegatronAdapter(AwexTrainingAdapter):
    """Shared Megatron adapter for v1 colocate and v2 separated transfer."""

    def __init__(
        self,
        engine: MegatronEngine,
        residency: MegatronWeightResidency | None = None,
    ):
        self._engine = engine
        self._residency = residency or MegatronWeightResidency(engine)
        self._transfer_plan: TransferPlan | None = None
        self._weights_update_group = None
        self._weights_update_group_gloo = None
        self._world_size: int | None = None
        self._separation_delta_transport: NcclColocateStreamBatchTransport | None = None
        self._separation_wire_dtypes: tuple[torch.dtype, ...] | None = None
        self._transfer_rank: int | None = None
        self._colocate_lock = threading.Lock()
        self._colocate_admin_api_key: str = "areal-admin-key"
        self._colocate_http_client: httpx.Client | None = None
        self._colocate_timeout_s: float = 120.0
        self._dte_config = DTERuntimeConfig.from_env()
        self._delta_tracker = None
        self._delta_detector = None
        self._weight_converter = None
        self._parameters_meta: list[ParameterMeta] | None = None
        self._rank_info = None
        self._legacy_meta_server_addr: str | None = None
        self._legacy_meta_server_client = None
        self._legacy_transfer_rank: int | None = None
        self._legacy_timeout_s: float = awex_colocate_timeout_s()
        self._legacy_initialized = False
        self._legacy_ip_address: str | None = None
        self._legacy_physical_gpu_id: int | None = None
        self._legacy_infer_world_size: int | None = None
        self._legacy_num_infer_engines: int | None = None
        self._legacy_logical_train_rank: int | None = None

    @property
    def residency(self) -> MegatronWeightResidency:
        """Return the shared residency manager used by this adapter."""
        return self._residency

    @property
    def _released_tags(self) -> set[str]:
        return set(self._residency.released_tags)

    def eager_publish_train_info(self, meta_server_addr: str | None) -> None:
        """Publish train world metadata before the colocated reader starts."""
        addr = meta_server_addr or os.environ.get("AWEX_META_SERVER_ADDR", "")
        if not addr or (dist.is_initialized() and dist.get_rank() != 0):
            return
        try:
            from awex.meta.meta_server import MetaServerClient

            host, port = addr.rsplit(":", 1)
            client = MetaServerClient(host, int(port))
            world = dist.get_world_size() if dist.is_initialized() else 1
            client.put_object("awex_train_info", {"train_world_size": world})
            logger.info(
                "Eager-published awex_train_info (train_world_size=%d) to %s",
                world,
                addr,
            )
        except Exception as exc:
            logger.warning("Eager publish awex_train_info failed: %s", exc)

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
            "parameter_layout": "awex",
        }

    def configure_model_converter(self, infer_conf: dict) -> None:
        """Initialize AWEX metadata and converters collectively on train ranks."""
        if self._weight_converter is not None:
            return

        from awex.meta.train_meta_resolver import McoreParamMetaResolver

        class _EngineShim:
            def __init__(self, engine: MegatronEngine):
                self.model = engine.model
                if not isinstance(self.model, (list, tuple)):
                    self.model = [self.model]
                self.hf_config = engine.hf_config
                self.enable_debug_mode = False
                self.enable_colocate_mode = False
                self.engine_name = "mcore"
                self.config = {}
                self.meta_server_addr = ""

            def release_memory_occupation(self, tags=None):
                del tags

            def resume_memory_occupation(self, tags=None):
                del tags

        resolver = McoreParamMetaResolver(
            _EngineShim(self._engine), self._engine.hf_config, infer_conf
        )
        self._parameters_meta = resolver.get_parameters_meta()
        self._rank_info = get_rank_info_extractor("mcore")()
        self._weight_converter = get_train_weights_converter(
            "mcore",
            self._engine.hf_config.architectures[0],
            self._engine.hf_config,
            self._rank_info,
            {
                **infer_conf,
                "train_pp_stage_layer_id_map": (resolver.get_pp_stage_layer_id_map()),
            },
            tf_config=_get_tf_config(self._engine.model),
        )

    def init_legacy_colocate_weight_update(
        self,
        meta_server_addr: str | None = None,
        pair_name: str = "default",
        transfer_rank: int = 0,
        timeout_s: float | None = None,
    ) -> None:
        """Initialize the v1 MetaServer-based colocate control plane."""
        from awex.meta.meta_server import MetaServerClient, start_meta_server

        if not meta_server_addr:
            meta_server_addr = os.environ.get("AWEX_META_SERVER_ADDR", "")
        if not meta_server_addr:
            host, port = start_meta_server()
            meta_server_addr = f"{host}:{port}"
            os.environ["AWEX_META_SERVER_ADDR"] = meta_server_addr
            logger.info("Started MetaServer at %s", meta_server_addr)

        host, port = meta_server_addr.rsplit(":", 1)
        self._legacy_meta_server_client = MetaServerClient(host, int(port))
        self._legacy_meta_server_addr = meta_server_addr
        self._legacy_transfer_rank = transfer_rank
        self._legacy_timeout_s = (
            awex_colocate_timeout_s() if timeout_s is None else timeout_s
        )
        if dist.get_rank() == 0:
            self._legacy_meta_server_client.put_object(
                "awex_train_info", {"train_world_size": dist.get_world_size()}
            )
            logger.info(
                "Registered awex_train_info (train_world_size=%d) with MetaServer",
                dist.get_world_size(),
            )

        logger.info(
            "Legacy AWEX colocate initialized: meta_server=%s, "
            "pair_name=%s, transfer_rank=%d",
            meta_server_addr,
            pair_name,
            transfer_rank,
        )

    def _lazy_initialize_legacy_colocate(self) -> None:
        """Finish v1 colocate initialization once live weights are available."""
        if self._legacy_initialized:
            return
        if self._legacy_meta_server_client is None:
            raise RuntimeError("Legacy AWEX colocate adapter is not initialized")

        from awex.util.common import get_ip_address

        rank = dist.get_rank()
        self._legacy_ip_address = get_ip_address()
        self._legacy_physical_gpu_id = resolve_physical_gpu_id(
            torch.cuda.current_device()
        )

        infer_conf = self._legacy_meta_server_client.get_object(
            "infer_conf", timeout=self._legacy_timeout_s
        )
        logger.info("Got infer_conf from MetaServer: %s", infer_conf)
        self.configure_model_converter(infer_conf)
        assert self._parameters_meta is not None
        assert self._rank_info is not None

        if rank == 0:
            self._legacy_meta_server_client.put_object(
                "training_params_meta", self._parameters_meta
            )
            logger.info("Registered training_params_meta with MetaServer")

        self._legacy_infer_world_size = infer_conf["infer_world_size"]
        self._legacy_logical_train_rank = (
            self._legacy_infer_world_size + self._rank_info.global_rank
        )
        self._legacy_meta_server_client.add_object_to_set(
            "training_device_rank_entries",
            (
                self._legacy_ip_address,
                self._legacy_physical_gpu_id,
                self._legacy_logical_train_rank,
            ),
        )
        self._legacy_num_infer_engines = self._legacy_meta_server_client.get_object(
            "num_infer_engines", timeout=self._legacy_timeout_s
        )
        self._legacy_initialized = True
        logger.info(
            "Legacy colocate train side initialized: logical_train_rank=%d, "
            "infer_world_size=%d, train_world_size=%d",
            self._legacy_logical_train_rank,
            self._legacy_infer_world_size,
            self._rank_info.world_size,
        )

    def get_weight_metadata(self) -> list[ParameterMeta]:
        if self._parameters_meta is None:
            raise RuntimeError("AWEX Megatron converter is not configured")
        # The native resolver gathers every rank into one global metadata list.
        # Publish it once so the gateway does not merge duplicate replicas.
        if dist.get_rank() != 0:
            return []
        return self._parameters_meta

    def get_local_shard_parameters(
        self, required_names: list[str] | None = None
    ) -> dict[str, torch.Tensor]:
        required = set(required_names) if required_names else None
        result: dict[str, torch.Tensor] = {}
        for hf_name, tensor in self._iter_hf_params():
            if required is not None and hf_name not in required:
                continue
            result[hf_name] = tensor
        return result

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
        timeout_s: float = 300.0,
    ) -> None:
        if self._dte_config.enabled:
            validate_dte_world_size(world_size, infer_world_size, train_world_size)

        self._transfer_rank = transfer_rank
        self._world_size = world_size

        infer_meta, train_meta = fetch_kv_metadata(kv_store_url, pair_name, timeout_s)

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
        self._weights_update_group_gloo = init_weights_update_group(
            master_address=master_addr,
            master_port=master_port,
            rank=transfer_rank,
            world_size=world_size,
            group_name=f"awex_{pair_name}_gloo",
            backend="gloo",
            role="training",
        )
        if self._dte_config.enabled:
            self._separation_wire_dtypes = synchronize_wire_dtypes(
                self._transfer_plan,
                self._weights_update_group_gloo,
            )
        logger.info(
            "Initialized AWEX weight update groups for pair=%s role=training "
            "rank=%s world_size=%s nccl=awex_%s gloo=awex_%s_gloo",
            pair_name,
            transfer_rank,
            world_size,
            pair_name,
            pair_name,
        )

    def execute_weight_update(self, version: int) -> None:
        if self._dte_config.enabled:
            self._release_grad_buffers_for_separation_sync()
            try:
                self._execute_separation_weight_update(version)
            finally:
                self._restore_grad_buffers_after_separation_sync()
            return

        if self._transfer_plan is None:
            raise RuntimeError("Transfer plan is not initialized")
        if self._weights_update_group is None:
            raise RuntimeError("Weight update group is not initialized")
        if self._weights_update_group_gloo is None:
            raise RuntimeError("Gloo weight update group is not initialized")
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
        dist.barrier(group=self._weights_update_group_gloo)

    def _release_grad_buffers_for_separation_sync(self) -> None:
        """Temporarily release Megatron DDP grad buffers during transfer."""
        model = getattr(self._engine, "model", None)
        if model is None:
            return
        modules = model if isinstance(model, (list, tuple)) else [model]
        for module in modules:
            release = getattr(module, "offload_grad_buffers", None)
            if release is not None:
                release(synchronize=False, empty_cache=False)

    def _restore_grad_buffers_after_separation_sync(self) -> None:
        """Restore Megatron DDP grad buffers even when transfer fails."""
        model = getattr(self._engine, "model", None)
        if model is None:
            return
        modules = model if isinstance(model, (list, tuple)) else [model]
        for module in modules:
            restore = getattr(module, "restore_grad_buffers", None)
            if restore is not None:
                restore(synchronize=False)

    def _execute_separation_weight_update(self, version: int) -> None:
        """Send an AdamW-derived sparse update, with a dense fallback."""
        if self._transfer_plan is None:
            raise RuntimeError("Transfer plan is not initialized")
        if self._weights_update_group is None:
            raise RuntimeError("Weight update group is not initialized")
        if self._weights_update_group_gloo is None:
            raise RuntimeError("Gloo weight update group is not initialized")
        if self._transfer_rank is None or self._world_size is None:
            raise RuntimeError("Transfer rank/world size is not initialized")

        params = self.get_local_shard_parameters()
        self._ensure_delta_components()
        synced_state = self._delta_detector.capture_synced_state(params)
        masks, local_is_delta = self._delta_prepare_masks(params, version)

        decision = torch.tensor([int(local_is_delta)], dtype=torch.int64)
        dist.all_reduce(
            decision, op=dist.ReduceOp.MIN, group=self._weights_update_group_gloo
        )
        use_delta = bool(decision.item()) and masks is not None

        if use_delta:
            self._execute_separation_delta_send(params, masks, version)
        else:
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

        # The receiver joins this barrier after applying the payload. Only then
        # may the sender advance its version/watermark state.
        dist.barrier(group=self._weights_update_group_gloo)
        if use_delta:
            self._delta_tracker.mark_delta_committed(version)
        self._delta_detector.mark_synced(version, synced_state)

    def _delta_prepare_masks(
        self,
        params: dict[str, torch.Tensor],
        version: int,
    ) -> tuple[dict[str, torch.Tensor] | None, bool]:
        reason = self._delta_tracker.full_sync_reason(version)
        if reason is None and not self._delta_detector.has_synced_watermark():
            reason = "initial_full"
        reason = self._sync_full_reason(reason, version)

        masks = None
        if reason is None:
            try:
                masks = self._delta_detector.compute_masks(
                    list(params), list(params.values()), version
                )
            except Exception:
                logger.exception(
                    "separation delta v%d: AdamW inversion failed; using full sync",
                    version,
                )
                reason = "adamw_inversion_error"
            if masks is None and reason is None:
                reason = "adamw_inversion_infeasible"

        reason = self._sync_full_reason(reason, version)
        if reason is not None:
            self._delta_tracker.seed(params.items(), version, store_snapshot=False)
            logger.info(
                "separation delta v%d: FULL sync fallback (%s)", version, reason
            )
            return None, False

        logger.info(
            "separation delta v%d: sparse AdamW path (%d params)",
            version,
            len(params),
        )
        return masks, True

    def _sync_full_reason(self, reason: str | None, version: int) -> str | None:
        """Promote a rank-local dense fallback to every training rank."""
        if not dist.is_available() or not dist.is_initialized():
            return reason
        try:
            world_size = dist.get_world_size()
        except RuntimeError:
            return reason
        if world_size <= 1:
            return reason

        device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        needs_full = torch.tensor(
            [1 if reason is not None else 0], dtype=torch.int32, device=device
        )
        dist.all_reduce(needs_full, op=dist.ReduceOp.MAX)
        if int(needs_full.item()) == 0:
            return None
        if reason is not None:
            return reason
        logger.warning("separation delta v%d: peer rank requires a full sync", version)
        return "peer_rank_fallback"

    def _execute_separation_delta_send(
        self,
        params: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
        version: int,
    ) -> None:
        from dte.core.colocate_protocol import (
            _filter_plan_by_dtype,
            _ops_by_recv_dtype,
            _PlanView,
            two_round_delta_exchange,
        )
        from dte.core.delta_p2p import build_send_payloads_by_op

        assert self._transfer_plan is not None
        assert self._weights_update_group is not None
        assert self._transfer_rank is not None
        assert self._world_size is not None
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

        payload_count = 0
        for dtype in self._separation_wire_dtypes:
            ops = operations_by_dtype.get(dtype, [])
            payloads = build_send_payloads_by_op(ops, masks, params)
            send_plan = _filter_plan_by_dtype(self._transfer_plan, dtype, is_send=True)
            two_round_delta_exchange(
                transfer_rank=self._transfer_rank,
                world_size=self._world_size,
                send_plan=send_plan,
                recv_plan=empty_plan,
                train_to_infer_device_mapping=identity_mapping,
                weights_update_group=self._weights_update_group,
                send_payloads_by_op=payloads,
                recv_params={},
                value_dtype=dtype,
                device=device,
                schedule_fn=schedule_fn,
                slice_fn=slice_tensor,
                rank_coordinate=f"train-{self._transfer_rank}",
                step_id=version,
            )
            payload_count += len(payloads)

        logger.info(
            "separation delta v%d sent %d payload ops across %d dtypes",
            version,
            payload_count,
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
        self._delta_tracker = None
        self._delta_detector = None
        if self._colocate_http_client is not None:
            self._colocate_http_client.close()
            self._colocate_http_client = None

    def _iter_hf_params(
        self,
        theta_by_id: dict[int, torch.Tensor] | None = None,
        consume_overrides: bool = False,
    ):
        """Yield canonical parameters through AWEX's model registry."""
        from awex.converter.mcore_converter import get_mcore_model_parameters

        if self._weight_converter is None or self._rank_info is None:
            raise RuntimeError("AWEX Megatron converter is not configured")

        overrides = theta_by_id if theta_by_id is not None else {}
        converted_names: set[str] = set()
        embed_tensor: torch.Tensor | None = None
        models = self._engine.model
        if not isinstance(models, (list, tuple)):
            models = [models]

        for vp_stage, model in enumerate(models):
            for mcore_name, param in get_mcore_model_parameters(model).items():
                source = overrides.get(id(param), param)
                for hf_name, tensor in self._weight_converter.convert_param(
                    mcore_name, source.detach(), vp_stage=vp_stage
                ):
                    converted_names.add(hf_name)
                    if hf_name == "model.embed_tokens.weight":
                        embed_tensor = tensor
                    yield hf_name, tensor.detach()
                if consume_overrides:
                    overrides.pop(id(param), None)

        if (
            getattr(self._engine.hf_config, "tie_word_embeddings", False)
            and self._rank_info.pp_rank == self._rank_info.pp_size - 1
            and "lm_head.weight" not in converted_names
            and embed_tensor is not None
        ):
            yield "lm_head.weight", embed_tensor.detach()

    def _iter_model_params_for_delta(self):
        """Yield model tensors in the same order used by the HF converter."""
        from awex.converter.mcore_converter import get_mcore_model_parameters

        seen: set[int] = set()
        models = self._engine.model
        if not isinstance(models, (list, tuple)):
            models = [models]
        for model in models:
            for param in get_mcore_model_parameters(model).values():
                if not isinstance(param, torch.nn.Parameter) or id(param) in seen:
                    continue
                seen.add(id(param))
                yield param

    def _convert_hf_with_overrides(
        self, theta_by_id: dict[int, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return dict(self._iter_hf_params(theta_by_id))

    @torch.no_grad()
    def _iter_hf_with_overrides(self, theta_by_id: dict[int, torch.Tensor]):
        yield from self._iter_hf_params(theta_by_id, consume_overrides=True)

    def _ensure_delta_components(self) -> None:
        if self._delta_tracker is None:
            self._delta_tracker = self._dte_config.create_delta_tracker()
        if self._delta_detector is None:
            self._delta_detector = AdamWInversionDetector(self)

    def _get_inner_optimizers(self):
        optimizer = self._engine.optimizer
        if optimizer is None:
            return []
        if hasattr(optimizer, "chained_optimizers"):
            return optimizer.chained_optimizers
        if hasattr(optimizer, "optimizers"):
            return optimizer.optimizers
        return [optimizer]

    @torch.no_grad()
    def execute_legacy_colocate_weight_update(self, version: int) -> None:
        """Execute the v1 MetaServer/CUDA-IPC colocate update unchanged."""
        from awex.util.tensor_util import (
            release_tensors,
        )

        if self._legacy_meta_server_client is None:
            raise RuntimeError("Legacy AWEX colocate adapter is not initialized")

        torch.cuda.ipc_collect()
        self._prepare_residency_for_publish()

        self._lazy_initialize_legacy_colocate()
        parameters = self.get_local_shard_parameters()
        tensors = list(parameters.values())
        names = list(parameters.keys())
        logger.info(
            "Converted %d params for legacy colocate IPC transfer (version=%d)",
            len(tensors),
            version,
        )

        group_tensors, metadata = group_tensors_by_shape_and_dtype(tensors)
        torch.cuda.synchronize()

        live_storages = set()
        model = self._engine.model
        for chunk in model if isinstance(model, (list, tuple)) else [model]:
            for _, param in chunk.named_parameters():
                live_storages.add(param.untyped_storage().data_ptr())
            for _, buffer in chunk.named_buffers():
                live_storages.add(buffer.untyped_storage().data_ptr())
        owned = [
            tensor
            for tensor in tensors
            if tensor.untyped_storage().data_ptr() not in live_storages
        ]
        release_tensors(owned)
        del tensors, owned
        parameters.clear()

        self.release_memory(tags=["weights"])

        assert self._legacy_ip_address is not None
        assert self._legacy_physical_gpu_id is not None
        assert self._legacy_logical_train_rank is not None
        assert self._rank_info is not None
        key_suffix = (
            f"_{self._legacy_ip_address}_{self._legacy_physical_gpu_id}_{version}"
        )

        self._legacy_meta_server_client.add_object_to_set(
            "all_training_offloaded_weights", self._legacy_logical_train_rank
        )

        group_shared = [tensor.share_memory_() for tensor in group_tensors]
        serialized_weights = cuda_ipc_serialize((group_shared, metadata, names))
        torch.cuda.synchronize()

        serialized_weights_key = f"training_serialized_weights{key_suffix}"
        writer_version_key = (
            "awex_writer_version_"
            f"{self._legacy_ip_address}_{self._legacy_physical_gpu_id}"
        )
        self._legacy_meta_server_client.put_object(writer_version_key, version)
        self._legacy_meta_server_client.put_object(
            serialized_weights_key,
            (self._legacy_logical_train_rank, self._rank_info, serialized_weights),
        )

        update_finished_key = f"weights_update_finished{key_suffix}"
        try:
            try:
                completion = self._legacy_meta_server_client.get_object(
                    update_finished_key, timeout=self._legacy_timeout_s
                )
            except Exception:
                logger.error(
                    "Timed out or failed after %ss waiting for inference to "
                    "consume legacy colocate weights (key=%s)",
                    self._legacy_timeout_s,
                    update_finished_key,
                )
                raise
            if isinstance(completion, dict) and not completion.get("ok", True):
                error = completion.get("error", "unknown inference-side error")
                raise RuntimeError(
                    "Inference rejected AWEX weights before IPC release: "
                    f"version={version}, device={self._legacy_physical_gpu_id}, "
                    f"error={error}"
                )
            self._legacy_meta_server_client.delete_if_exists(update_finished_key)
            self._legacy_meta_server_client.delete_if_exists(serialized_weights_key)
        finally:
            release_tensors(group_tensors)
            release_tensors(group_shared)
            del group_tensors, group_shared
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

        write_finished_key = f"write_finished{key_suffix}"
        self._legacy_meta_server_client.put_object(write_finished_key, True)
        logger.info("Legacy colocate weight update completed: version=%d", version)

    def finish_legacy_colocate_weight_update(self, training_world_size: int) -> None:
        """Finish the v1 MetaServer handshake and clean its coordination keys."""
        del training_world_size
        if self._legacy_meta_server_client is None:
            raise RuntimeError("Legacy AWEX colocate adapter is not initialized")
        if self._legacy_num_infer_engines is None:
            raise RuntimeError("Legacy AWEX colocate adapter is not ready")

        self._legacy_meta_server_client.wait_set_until_size(
            "finished_weights_update_engines",
            self._legacy_num_infer_engines,
            timeout=self._legacy_timeout_s,
        )
        dist.barrier(group=self._engine.cpu_group)
        if dist.get_rank() == 0:
            self._legacy_meta_server_client.delete_if_exists(
                "finished_weights_update_engines"
            )
            self._legacy_meta_server_client.delete_if_exists(
                "all_training_offloaded_weights"
            )

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

        params = self.get_local_shard_parameters()
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

        if weights_offloaded:
            self.release_memory(tags=["weights"])

    def _prepare_residency_for_publish(self) -> None:
        """Free optimizer/grad memory before making weights resident."""
        weights_were_offloaded = self._residency.is_released("weights")
        self._residency.release_memory(tags=["optimizer"])
        self._residency.release_grad_memory()
        if weights_were_offloaded:
            self._residency.resume_memory(tags=["weights"])

    def _release_grad_memory(self) -> None:
        self._residency.release_grad_memory()

    def ensure_grad_buffers(self) -> None:
        self._residency.ensure_grad_buffers()

    def release_memory(self, tags: list[str] | None = None) -> None:
        self._residency.release_memory(tags)

    def resume_memory(self, tags: list[str] | None = None) -> None:
        self._residency.resume_memory(tags)


__all__ = [
    "AwexMegatronAdapter",
    "awex_colocate_timeout_s",
    "resolve_physical_gpu_id",
]
