# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import gc
import os
import threading
import time
from typing import TYPE_CHECKING

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
from awex.transfer.nccl_stream_batch import NcclColocateStreamBatchTransport
from awex.transfer.transfer_plan import TransferPlan, TransferPlanBuilder, slice_tensor
from awex.util.tensor_util import (
    cuda_ipc_serialize,
    group_tensors_by_shape_and_dtype,
)

from areal.utils import logging
from areal.utils.dte import dte_verification_snapshot_commit_action
from areal.v2.weight_update.awex import (
    awex_wu_use_group,
    fetch_kv_metadata,
    resolve_physical_gpu_id,
)
from areal.v2.weight_update.awex.delta_config import (
    delta_transfer_enabled,
    make_delta_tracker,
    separation_delta_transfer_enabled,
)
from areal.v2.weight_update.awex.delta_detect import (
    build_detector,
    delta_detector_mode,
    external_delta_detector_enabled,
)
from areal.v2.weight_update.awex.separation_verify import (
    separation_post_apply_verify_enabled,
    verify_separation_post_apply,
)
from areal.v2.weight_update.awex.weight_digest import log_tensor_digest
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
_CAPTURE_MISS = object()


def awex_colocate_timeout_s(default: float = 1800.0) -> float:
    value = os.environ.get("AWEX_COLOCATE_TIMEOUT_S", "").strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning(
            "Invalid AWEX_COLOCATE_TIMEOUT_S=%r; using default %.1fs",
            value,
            default,
        )
        return default


class AwexMegatronAdapter(AwexTrainingAdapter):
    """Awex training adapter for MegatronEngine supporting DP, TP, and PP.

    PP: get_named_parameters already yields only the current stage's layers
    (with globally-correct HF layer indices via get_transformer_layer_offset),
    so each rank naturally reports and sends only its own subset of parameters.
    The gateway's _merge_training_meta_by_name unions disjoint PP stage params
    by name, so the full model is covered across all PP ranks.

    TP: all_gather_param gathers the full tensor on every TP rank before
    convert_to_hf. dp_replicated=True tells awex that TP ranks within a DP
    group hold identical full tensors and only one needs to send.
    """

    def __init__(self, engine: MegatronEngine):
        self._engine = engine
        engine._awex_adapter = self
        self._transfer_plan: TransferPlan | None = None
        self._weights_update_group = None
        self._weights_update_group_gloo = None
        self._world_size: int | None = None
        self._separation_delta_transport: NcclColocateStreamBatchTransport | None = None
        self._transfer_rank: int | None = None
        self._offloaded_weights: dict[str, torch.Tensor] = {}
        self._released_tags: set[str] = set()
        self._meta_server_addr: str | None = None
        self._meta_server_client = None
        self._weight_converter = None
        self._initialized = False
        self._rank_info: RankInfo | None = None
        self._ip_address: str | None = None
        self._physical_gpu_id: int | None = None
        self._infer_world_size: int | None = None
        self._num_infer_engines: int | None = None
        self._logical_train_rank: int | None = None
        self._weight_metadata_cache: list[ParameterMeta] | None = None
        self._timeout_s = awex_colocate_timeout_s()
        self._delta_tracker = None
        self._delta_detector = None
        self._colocate_lock = threading.Lock()
        self._precompute_param_synced_version: int | None = None
        self._precomputed_synced_state: tuple[int, object] | None = None

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

    def get_weight_metadata(self) -> list[ParameterMeta]:
        if self._weight_metadata_cache is not None:
            return self._weight_metadata_cache
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

        self._weight_metadata_cache = metadata
        return metadata

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
    ) -> None:
        self._transfer_rank = transfer_rank
        self._world_size = world_size

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
        self._weights_update_group_gloo = init_weights_update_group(
            master_address=master_addr,
            master_port=master_port,
            rank=transfer_rank,
            world_size=world_size,
            group_name=f"awex_{pair_name}_gloo",
            backend="gloo",
            role="training",
        )

    def execute_weight_update(self, version: int) -> None:
        if separation_delta_transfer_enabled():
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
            raise RuntimeError("Weight update control group is not initialized")
        if self._transfer_rank is None:
            raise RuntimeError("Transfer rank is not initialized")

        params = self.get_local_shard_parameters()
        log_tensor_digest(
            params.items(),
            role="train",
            phase="pre_send",
            version=version,
            extra={
                "transfer_path": "separation_full",
                "transfer_rank": self._transfer_rank,
                "transfer_world_size": self._world_size,
                "payload_manifest": "source_params",
            },
        )
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
        if separation_post_apply_verify_enabled():
            verify_separation_post_apply(
                params,
                self._transfer_plan,
                self._weights_update_group_gloo,
                role="train",
                version=version,
                mode="full",
            )

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
        """Send a sparse separated-card update, with dense fallback."""
        if self._transfer_plan is None:
            raise RuntimeError("Transfer plan is not initialized")
        if self._weights_update_group is None:
            raise RuntimeError("Weight update group is not initialized")
        if self._weights_update_group_gloo is None:
            raise RuntimeError("Separation control group is not initialized")
        if self._transfer_rank is None or self._world_size is None:
            raise RuntimeError("Transfer rank/world size is not initialized")

        params = self.get_local_shard_parameters()
        log_tensor_digest(
            params.items(),
            role="train",
            phase="pre_send",
            version=version,
            extra={
                "transfer_path": "separation_delta_or_full",
                "transfer_rank": self._transfer_rank,
                "transfer_world_size": self._world_size,
                "payload_manifest": "source_params",
            },
        )
        self._ensure_delta_components()
        synced_state = self._delta_capture_synced_state(params)
        masks: dict[str, torch.Tensor] | None = None
        prepare_failed = False
        try:
            masks, local_is_delta = self._delta_prepare_masks_for_separation(
                params, version
            )
        except Exception:
            logger.exception(
                "separation delta v%d: mask preparation failed; using dense",
                version,
            )
            local_is_delta = False
            prepare_failed = True

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

        dist.barrier(group=self._weights_update_group_gloo)
        if separation_post_apply_verify_enabled():
            verify_separation_post_apply(
                params,
                self._transfer_plan,
                self._weights_update_group_gloo,
                role="train",
                version=version,
                mode="delta" if use_delta else "full",
            )
        if use_delta:
            self._delta_tracker.mark_delta_committed(version)
        self._delta_commit_separation_verification_snapshot(
            params,
            masks,
            version,
            prepare_failed=prepare_failed,
            use_delta=use_delta,
        )
        self._delta_mark_synced(version, synced_state)

    def _delta_prepare_masks_for_separation(
        self,
        params: dict[str, torch.Tensor],
        version: int,
    ) -> tuple[dict[str, torch.Tensor] | None, bool]:
        """Choose sparse separation only when a detector provides safe masks."""
        self._ensure_delta_components()
        detector_name = self._delta_detector.name
        external_detector = detector_name != "snapshot"
        verify_snapshot = self._delta_verify_snapshot_enabled()
        names = list(params)
        tensors = list(params.values())
        params_list = list(params.items())

        reason = self._delta_tracker.full_sync_reason(version)
        if external_detector and reason is None:
            has_watermark = getattr(
                self._delta_detector, "has_synced_watermark", lambda: False
            )
            if not has_watermark():
                reason = "initial_full"

        reason = self._delta_sync_full_reason(reason, version)
        masks = None
        if external_detector and reason is None:
            masks = self._delta_detector.compute_masks(names, tensors, version)
            if masks is None:
                reason = f"{detector_name}_infeasible"
            elif verify_snapshot and not self._delta_verify_masks_against_snapshot(
                params_list, masks, version
            ):
                reason = f"{detector_name}_snapshot_mismatch"
        elif not external_detector and reason is None:
            masks, reason = self._delta_snapshot_masks_for_separation(
                params_list, version
            )
            if masks is not None and verify_snapshot:
                if not self._delta_verify_masks_against_snapshot(
                    params_list, masks, version
                ):
                    reason = f"{detector_name}_snapshot_mismatch"

        reason = self._delta_sync_full_reason(reason, version)
        if reason is not None:
            self._delta_tracker.seed(
                params_list,
                version,
                store_snapshot=(not external_detector or verify_snapshot),
            )
            if verify_snapshot:
                self._delta_mark_snapshot_names(params_list)
            logger.info(
                "separation delta v%d: FULL sync fallback (%s)", version, reason
            )
            return None, False

        logger.info(
            "separation delta v%d: sparse path (%s detector, %d params)",
            version,
            detector_name,
            len(params),
        )
        return masks, True

    def _execute_separation_delta_send(
        self,
        params: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
        version: int,
    ) -> None:
        """Send sparse parameter shards through DTE's two-round P2P protocol."""
        from dte.core.colocate_protocol import (
            _filter_plan_by_dtype,
            _ops_by_recv_dtype,
            _PlanView,
            two_round_delta_exchange,
        )
        from dte.core.delta_p2p import build_send_payloads_by_op

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

        payload_count = 0
        for dtype, ops in operations_by_dtype.items():
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

    def _iter_hf_params(
        self,
        theta_by_id: dict[int, torch.Tensor] | None = None,
        consume_overrides: bool = False,
    ):
        """Yield (hf_name, tensor) for every parameter on this rank.

        Uses get_named_parameters + all_gather_param + convert_to_hf to produce
        HF-style per-expert names (e.g. experts.0.gate_proj.weight). The SGLang
        adapter's _unfuse_params converts SGLang's fused w13/w2 format to the
        same per-expert names, so both sides match for the transfer plan.

        ``theta_by_id`` maps ``id(model_param)`` to a replacement tensor. The
        AdamW inversion detector uses it to convert reconstructed pre-step
        mcore weights through the exact live all-gather + HF conversion path.
        """
        from areal.engine.megatron_utils.megatron import (
            all_gather_param,
            convert_to_hf,
            get_named_parameters,
        )

        num_moe_experts = getattr(self._engine.tf_config, "num_moe_experts", None)
        model_name = self._engine.hf_config.model_type
        tie_word_embeddings = getattr(
            self._engine.hf_config, "tie_word_embeddings", False
        )
        overrides = theta_by_id if theta_by_id is not None else {}

        for mcore_name, param in get_named_parameters(
            self._engine.model, num_moe_experts
        ):
            src = overrides.get(id(param), param)
            if src is not param:
                for attr in (
                    "tensor_model_parallel",
                    "partition_dim",
                    "partition_stride",
                ):
                    if hasattr(param, attr):
                        setattr(src, attr, getattr(param, attr))
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
            if consume_overrides:
                overrides.pop(id(param), None)

    def _iter_model_params_for_delta(self):
        """Yield model tensors in the same order used by the payload converter."""
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

    def _convert_hf_with_overrides(
        self, theta_by_id: dict[int, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Convert mcore params to HF with reconstructed pre-step overrides."""
        return dict(self._iter_hf_params(theta_by_id))

    @torch.no_grad()
    def _iter_hf_with_overrides(self, theta_by_id: dict[int, torch.Tensor]):
        """Yield HF params with reconstructed overrides without materializing all."""
        yield from self._iter_hf_params(theta_by_id, consume_overrides=True)

    # ── Colocated weight transfer methods ─────────────────────────────────

    def init_colocate_weight_update(
        self,
        meta_server_addr: str | None = None,
        pair_name: str = "default",
        transfer_rank: int = 0,
        timeout_s: float | None = None,
        **kwargs,
    ) -> None:
        """Connect to the colocate MetaServer.

        Extra gateway fields are accepted for wire compatibility; colocate
        metadata and runtime coordination are resolved through MetaServer.
        """
        from awex.meta.meta_server import MetaServerClient

        expected_delta_enabled = kwargs.pop("expected_delta_enabled", None)
        del kwargs
        if not meta_server_addr:
            meta_server_addr = os.environ.get("AWEX_META_SERVER_ADDR", "")
        if not meta_server_addr:
            raise ValueError("meta_server_addr is required for colocate weight update")

        host, port = meta_server_addr.rsplit(":", 1)
        self._meta_server_client = MetaServerClient(host, int(port))
        self._meta_server_addr = meta_server_addr
        self._colocate_pair_name = pair_name
        self._transfer_rank = transfer_rank
        self._timeout_s = awex_colocate_timeout_s() if timeout_s is None else timeout_s

        if dist.get_rank() == 0:
            self._meta_server_client.put_object(
                "awex_train_info", {"train_world_size": dist.get_world_size()}
            )
        logger.info(
            "Initialized colocate weight update for pair %r at %s, transfer_rank=%d",
            pair_name,
            meta_server_addr,
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
        if local_delta:
            self._ensure_delta_components()
            logger.info("colocate delta enabled (sender); DTE components ready")

    def _lazy_initialize(self) -> None:
        """Initialize metadata and conversion after live weights are available."""
        if self._initialized:
            return
        if self._meta_server_client is None:
            raise RuntimeError("init_colocate_weight_update must be called first")

        from awex.meta.train_meta_resolver import McoreParamMetaResolver
        from awex.models.registry import get_train_weights_converter
        from awex.sharding.param_sharding import get_rank_info_extractor
        from awex.util.common import get_ip_address

        class _EngineShim:
            def __init__(self, engine):
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
                pass

            def resume_memory_occupation(self, tags=None):
                pass

        self._rank_info = get_rank_info_extractor("mcore")()
        self._ip_address = get_ip_address()
        self._physical_gpu_id = resolve_physical_gpu_id()

        infer_conf = self._meta_server_client.get_object(
            "infer_conf", timeout=self._timeout_s
        )
        logger.info("Got infer_conf from MetaServer: %s", infer_conf)
        if isinstance(infer_conf.get("hf_config"), dict):
            from types import SimpleNamespace

            infer_conf["hf_config"] = SimpleNamespace(**infer_conf["hf_config"])

        shim = _EngineShim(self._engine)
        meta_resolver = McoreParamMetaResolver(shim, self._engine.hf_config, infer_conf)
        parameters_meta = meta_resolver.get_parameters_meta()
        if dist.get_rank() == 0:
            self._meta_server_client.put_object("training_params_meta", parameters_meta)

        self._infer_world_size = infer_conf["infer_world_size"]
        # The transfer plan annotates shard ownership with Megatron's own
        # global rank, so the wire identity must be derived from it. The
        # gateway's (ip,gpu)-sorted transfer_rank may order nodes differently
        # and would pair readers with the wrong pp/dp stage.
        self._logical_train_rank = self._infer_world_size + self._rank_info.global_rank
        self._meta_server_client.add_object_to_set(
            "training_device_rank_entries",
            (self._ip_address, self._physical_gpu_id, self._logical_train_rank),
        )
        self._num_infer_engines = self._meta_server_client.get_object(
            "num_infer_engines", timeout=self._timeout_s
        )

        # Passing infer_conf through preserves the reader's router dtype. A
        # float32 gate payload paired with lower-precision metadata can wedge
        # the transfer because sender and receiver disagree on byte counts.
        self._weight_converter = get_train_weights_converter(
            "mcore",
            self._engine.hf_config.architectures[0],
            self._engine.hf_config,
            self._rank_info,
            {
                **infer_conf,
                "train_pp_stage_layer_id_map": (
                    meta_resolver.get_pp_stage_layer_id_map()
                ),
            },
            tf_config=_get_tf_config(self._engine.model),
        )
        self._initialized = True
        logger.info(
            "Colocate train side initialized: logical_train_rank=%d, "
            "infer_world_size=%d, train_world_size=%d",
            self._logical_train_rank,
            self._infer_world_size,
            self._rank_info.world_size,
        )

    def _release_grad_memory(self) -> None:
        """Release DDP gradient buffers while preserving sizes for reload."""
        from megatron.core.distributed import DistributedDataParallel as DDP

        model = self._engine.model
        if model is None:
            return
        if not isinstance(model, (list, tuple)):
            model = [model]
        count = 0
        for chunk in model:
            if isinstance(chunk, DDP):
                for buffers in [chunk.buffers, chunk.expert_parallel_buffers]:
                    for buf in buffers:
                        if buf.grad_data.storage().size() > 0:
                            buf.grad_data_size = buf.grad_data.storage().size()
                            buf.grad_data.storage().resize_(0)
                            count += 1
        if count:
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
        logger.info("Released %d grad buffers", count)

    @torch.no_grad()
    def execute_colocate_weight_update(self, version: int) -> None:
        """Publish local parameter shards to colocated inference via CUDA IPC."""
        with self._colocate_lock:
            self._execute_colocate_weight_update_locked(version)

    def _execute_colocate_weight_update_locked(self, version: int) -> None:
        from awex.util.tensor_util import release_tensors

        if self._meta_server_client is None:
            raise RuntimeError("init_colocate_weight_update must be called first")

        weights_were_offloaded = "weights" in self._released_tags
        torch.cuda.ipc_collect()
        try:
            if delta_transfer_enabled():
                self._execute_colocate_delta_weight_update(
                    version,
                    weights_were_offloaded=weights_were_offloaded,
                )
                return

            # Optimizer and gradient buffers must leave the GPU before weights
            # are restored because inference is already resident on the same GPU.
            self.release_memory(tags=["optimizer"])
            self._release_grad_memory()
            if weights_were_offloaded:
                self.resume_memory(tags=["weights"])

            dist.barrier()
            if dist.get_rank() == 0:
                self._meta_server_client.add_object_to_set(
                    "all_training_offloaded_optimizers", dist.get_rank()
                )

            # The resolver converts live parameters. It must run after resume;
            # resize_(0)-ed parameter storages are not valid conversion inputs.
            self._lazy_initialize()
            parameters = self._convert_parameters()
            tensors = list(parameters.values())
            names = list(parameters.keys())
            group_tensors, metadata = group_tensors_by_shape_and_dtype(tensors)
            torch.cuda.synchronize()

            live_storages = set()
            model = self._engine.model
            for chunk in model if isinstance(model, (list, tuple)) else [model]:
                for _, parameter in chunk.named_parameters():
                    live_storages.add(parameter.untyped_storage().data_ptr())
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
            if (
                self._ip_address is None
                or self._physical_gpu_id is None
                or self._logical_train_rank is None
                or self._rank_info is None
            ):
                raise RuntimeError("Colocate metadata is not initialized")

            key_suffix = f"_{self._ip_address}_{self._physical_gpu_id}_{version}"
            group_shared = [tensor.share_memory_() for tensor in group_tensors]
            serialized_weights = cuda_ipc_serialize((group_shared, metadata, names))
            torch.cuda.synchronize()

            writer_version_key = (
                f"awex_writer_version_{self._ip_address}_{self._physical_gpu_id}"
            )
            self._meta_server_client.put_object(writer_version_key, version)
            serialized_weights_key = f"training_serialized_weights{key_suffix}"
            self._meta_server_client.put_object(
                serialized_weights_key,
                (self._logical_train_rank, self._rank_info, serialized_weights),
            )
            self._meta_server_client.add_object_to_set(
                "all_training_offloaded_weights", self._logical_train_rank
            )

            update_finished_key = f"weights_update_finished{key_suffix}"
            try:
                try:
                    self._meta_server_client.get_object(
                        update_finished_key, timeout=self._timeout_s
                    )
                except Exception as exc:
                    raise RuntimeError(
                        "Inference did not finish the colocate weight update "
                        f"within {self._timeout_s}s; missing key "
                        f"{update_finished_key!r}"
                    ) from exc
                self._meta_server_client.delete_if_exists(update_finished_key)
                self._meta_server_client.delete_if_exists(serialized_weights_key)
            finally:
                release_tensors(group_tensors)
                release_tensors(group_shared)
                del group_tensors, group_shared
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()

            write_finished_key = f"write_finished{key_suffix}"
            self._meta_server_client.put_object(write_finished_key, True)
            logger.info("Colocate weight update completed: version=%d", version)
        finally:
            torch.cuda.ipc_collect()
            if weights_were_offloaded and "weights" not in self._released_tags:
                self.release_memory(tags=["weights"])

    @torch.no_grad()
    def _execute_colocate_delta_weight_update(
        self,
        version: int,
        *,
        weights_were_offloaded: bool,
    ) -> None:
        """Publish a DTE delta/full payload through the MetaServer IPC path."""
        from awex.util.tensor_util import release_tensors

        if self._meta_server_client is None:
            raise RuntimeError("init_colocate_weight_update must be called first")

        if weights_were_offloaded:
            self.resume_memory(tags=["weights"])

        # The resolver/converter needs live model weights and stores the
        # MetaServer device-rank entries used by the SGLang reader.
        self._lazy_initialize()

        sync_before_payload = self._needs_external_detector_sync_before_payload(version)
        if sync_before_payload and not self._precompute_param_sync_covers(version):
            self._sync_model_params_from_optimizer()

        params = self.get_local_shard_parameters()
        log_tensor_digest(
            params.items(),
            role="train",
            phase="pre_send",
            version=version,
            extra={
                "transfer_path": "colocate_delta_or_full",
                "transfer_rank": self._transfer_rank,
                "payload_manifest": "source_params",
            },
        )
        names, tensors, zero_copy_full_payload = self._delta_encode(params, version)
        delta_synced_state = self._pop_precomputed_synced_state(version)
        if delta_synced_state is _CAPTURE_MISS:
            delta_synced_state = self._delta_capture_synced_state(params)

        self.release_memory(tags=["optimizer"])
        self._release_grad_memory()

        if zero_copy_full_payload:
            group_tensors, metadata = self._full_tensors_for_ipc(tensors, names)
        else:
            group_tensors, metadata = group_tensors_by_shape_and_dtype(tensors)
            self._release_owned_payload_tensors(tensors)
        del tensors, params

        self.release_memory(tags=["weights"])
        if (
            self._ip_address is None
            or self._physical_gpu_id is None
            or self._logical_train_rank is None
            or self._rank_info is None
        ):
            raise RuntimeError("Colocate metadata is not initialized")

        key_suffix = f"_{self._ip_address}_{self._physical_gpu_id}_{version}"
        group_shared: list[torch.Tensor] = []
        try:
            group_shared = [tensor.share_memory_() for tensor in group_tensors]
            serialized_weights = cuda_ipc_serialize((group_shared, metadata, names))
            torch.cuda.synchronize()

            writer_version_key = (
                f"awex_writer_version_{self._ip_address}_{self._physical_gpu_id}"
            )
            self._meta_server_client.put_object(writer_version_key, version)
            serialized_weights_key = f"training_serialized_weights{key_suffix}"
            self._meta_server_client.put_object(
                serialized_weights_key,
                (self._logical_train_rank, self._rank_info, serialized_weights),
            )
            self._meta_server_client.add_object_to_set(
                "all_training_offloaded_weights", self._logical_train_rank
            )

            update_finished_key = f"weights_update_finished{key_suffix}"
            try:
                self._meta_server_client.get_object(
                    update_finished_key, timeout=self._timeout_s
                )
            except Exception as exc:
                raise RuntimeError(
                    "Inference did not finish the colocate DTE weight update "
                    f"within {self._timeout_s}s; missing key "
                    f"{update_finished_key!r}"
                ) from exc
            self._meta_server_client.delete_if_exists(update_finished_key)
            self._meta_server_client.delete_if_exists(serialized_weights_key)

            self._delta_mark_synced(version, delta_synced_state)
            write_finished_key = f"write_finished{key_suffix}"
            self._meta_server_client.put_object(write_finished_key, True)
            logger.info("Colocate DTE weight update completed: version=%d", version)
        finally:
            release_tensors(group_tensors)
            if group_shared:
                release_tensors(group_shared)
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

    def finish_colocate_weight_update(self, training_world_size: int) -> None:
        """Wait for all inference engines, then clear handshake state."""
        del training_world_size
        if self._meta_server_client is None or self._num_infer_engines is None:
            raise RuntimeError("Colocate weight update is not initialized")

        self._meta_server_client.wait_set_until_size(
            "finished_weights_update_engines",
            self._num_infer_engines,
            timeout=self._timeout_s,
        )
        dist.barrier(group=self._engine.cpu_group)
        if dist.get_rank() == 0:
            for key in (
                "finished_weights_update_engines",
                "all_training_offloaded_optimizers",
                "all_training_offloaded_weights",
            ):
                self._meta_server_client.delete_if_exists(key)

    def precompute_delta_masks(self, version: int) -> bool:
        """Precompute external detector masks before optimizer offload."""
        if not delta_transfer_enabled():
            return False
        with self._colocate_lock:
            return self._precompute_delta_masks_locked(version)

    def _precompute_delta_masks_locked(self, version: int) -> bool:
        self._ensure_delta_components()
        detector = self._delta_detector
        precompute = getattr(detector, "precompute_masks", None)
        if precompute is None:
            return False

        reason: str | None = None
        if "weights" in self._released_tags:
            reason = "weights_offloaded"
        if reason is None:
            reason = self._delta_tracker.full_sync_reason(version)
        if reason is None:
            has_watermark = getattr(detector, "has_synced_watermark", lambda: False)
            if not has_watermark():
                reason = "initial_full"
        reason = self._delta_sync_full_reason(reason, version)
        if reason is not None:
            logger.info(
                "precompute_delta_masks v%d: full sync pending (%s), skipping",
                version,
                reason,
            )
            return False

        self._sync_model_params_from_optimizer()
        self._precompute_param_synced_version = int(version)
        params = self.get_local_shard_parameters()
        t0 = time.monotonic()
        feasible = precompute(list(params.keys()), list(params.values()), version)
        captured = self._delta_capture_synced_state(params)
        self._precomputed_synced_state = (int(version), captured)
        logger.info(
            "precompute_delta_masks v%d: feasible=%s elapsed_ms=%.1f",
            version,
            feasible,
            (time.monotonic() - t0) * 1000,
        )
        return bool(feasible)

    def _needs_external_detector_sync_before_payload(self, version: int) -> bool:
        if not delta_transfer_enabled():
            return False
        if not external_delta_detector_enabled(delta_detector_mode()):
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
                "detector watermark missing; first payload will be full sync",
                version,
            )
            return False
        return True

    def _precompute_param_sync_covers(self, version: int) -> bool:
        local = self._precompute_param_synced_version == int(version)
        self._precompute_param_synced_version = None
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            device = (
                torch.device("cuda", torch.cuda.current_device())
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            flag = torch.tensor([1 if local else 0], dtype=torch.int32, device=device)
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
            if int(flag.item()) == 0:
                if local:
                    logger.warning(
                        "v%d: redoing param sync on all ranks; another rank "
                        "missed its precompute param-sync marker",
                        version,
                    )
                return False
        return local

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

    def _live_module_storage_ptrs(self) -> set[int]:
        live_storages: set[int] = set()
        model = self._engine.model
        chunks = model if isinstance(model, (list, tuple)) else [model]
        for chunk in chunks:
            for _, param in chunk.named_parameters():
                live_storages.add(param.untyped_storage().data_ptr())
            for _, buf in chunk.named_buffers():
                live_storages.add(buf.untyped_storage().data_ptr())
        return live_storages

    def _release_owned_payload_tensors(self, tensors: list[torch.Tensor]) -> None:
        from awex.util.tensor_util import release_tensors

        live_storages = self._live_module_storage_ptrs()
        owned = [
            tensor
            for tensor in tensors
            if tensor.untyped_storage().data_ptr() not in live_storages
        ]
        if owned:
            release_tensors(owned)

    def _full_tensors_for_ipc(
        self,
        tensors: list[torch.Tensor],
        names: list[str] | None = None,
    ) -> tuple[list[torch.Tensor], list[dict]]:
        """Build bounded exporter-owned groups for full-sync CUDA IPC."""
        if names is not None and len(names) != len(tensors):
            raise ValueError(
                "names must match tensors when building colocate IPC payload"
            )

        max_group_bytes = self._colocate_full_group_max_bytes()
        live_storages = self._live_module_storage_ptrs()
        group_tensors: list[torch.Tensor] = []
        metadata: list[dict] = []
        buckets: dict[torch.dtype, list[int]] = {}

        def append_zero_group(original_index: int, source: torch.Tensor) -> None:
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

        for original_index, tensor in enumerate(tensors):
            source = tensor.detach()
            if source.numel() == 0:
                append_zero_group(original_index, source)
                continue
            buckets.setdefault(source.dtype, []).append(original_index)

        def finalize_group(
            current: list[tuple[int, torch.Tensor, torch.Size, torch.dtype, int]],
        ) -> None:
            if not current:
                return
            group_index = len(group_tensors)
            group_tensor = torch.cat(
                [entry[1].reshape(-1) for entry in current]
            ).clone()
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
                    finalize_group(current)
                    current = []
                    current_bytes = 0
                compact = source.contiguous()
                if compact.untyped_storage().data_ptr() in live_storages:
                    compact = compact.clone(memory_format=torch.contiguous_format)
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
            finalize_group(current)

        logger.info(
            "Built bounded colocate full IPC payload: params=%d, groups=%d, "
            "max_group_bytes=%d",
            len(tensors),
            len(group_tensors),
            max_group_bytes,
        )
        return group_tensors, metadata

    @torch.no_grad()
    def _sync_model_params_from_optimizer(self) -> None:
        copied = 0
        gathered = 0
        for opt in self._get_inner_optimizers():
            copy_fn = getattr(opt, "_copy_main_params_to_model_params", None)
            if copy_fn is None:
                copy_fn = getattr(opt, "_copy_main_params_to_param_buffer", None)
            if copy_fn is not None:
                copy_fn()
                copied += 1

            gather_fn = getattr(
                opt, "_reset_metadata_and_sync_gather_all_model_params", None
            )
            if gather_fn is not None:
                gather_fn(force_sync=True)
                gathered += 1

        synced = 0
        if gathered == 0:
            model = self._engine.model
            chunks = model if isinstance(model, (list, tuple)) else [model]
            for chunk in chunks:
                start_param_sync = getattr(chunk, "start_param_sync", None)
                if start_param_sync is not None:
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

    def _pop_precomputed_synced_state(self, version: int):
        cached = self._precomputed_synced_state
        self._precomputed_synced_state = None
        if cached is not None and cached[0] == int(version):
            return cached[1]
        return _CAPTURE_MISS

    def _delta_encode(
        self,
        params: dict[str, torch.Tensor],
        version: int,
    ) -> tuple[list[str], list[torch.Tensor], bool]:
        """Encode colocate payload as DTE sparse delta or dense full sync."""
        self._ensure_delta_components()
        detector_name = self._delta_detector.name
        external_detector = detector_name != "snapshot"
        verify_snapshot = external_detector and self._delta_verify_snapshot_enabled()
        names = list(params.keys())
        tensors = list(params.values())
        params_list = list(params.items())

        reason = self._delta_tracker.full_sync_reason(version)
        masks = None
        if external_detector and reason is None:
            has_watermark = getattr(
                self._delta_detector, "has_synced_watermark", lambda: False
            )
            if not has_watermark():
                reason = "initial_full"
        reason = self._delta_sync_full_reason(reason, version)
        if external_detector and reason is None:
            masks = self._delta_detector.compute_masks(names, tensors, version)
            if masks is None:
                reason = f"{detector_name}_infeasible"
            elif verify_snapshot and not self._delta_verify_masks_against_snapshot(
                params_list, masks, version
            ):
                reason = f"{detector_name}_snapshot_mismatch"

        reason = self._delta_sync_full_reason(reason, version)
        if reason is not None:
            self._delta_tracker.seed(
                params_list,
                version,
                store_snapshot=(not external_detector or verify_snapshot),
            )
            if verify_snapshot:
                self._delta_mark_snapshot_names(params_list)
            logger.info("colocate delta v%d: FULL sync (%s)", version, reason)
            return names, tensors, True

        encoded = self._delta_tracker.encode(params_list, version, masks=masks)
        if verify_snapshot:
            self._delta_refresh_verification_snapshot(params_list)
        logger.info(
            "colocate delta v%d [%s]: changed %d/%d (%.2f%%) "
            "sparse=%d dense_fallback=%d unchanged=%d payload=%.1fMB vs dense=%.1fMB",
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

    def _ensure_delta_components(self) -> None:
        if self._delta_tracker is None:
            self._delta_tracker = make_delta_tracker()
        if self._delta_detector is None:
            self._delta_detector = build_detector(delta_detector_mode(), self)

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

    def _delta_capture_synced_state(
        self, payload_params: dict[str, torch.Tensor] | None = None
    ):
        """Capture detector watermarks after a payload-compatible state."""
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

    def _delta_sync_full_reason(self, reason: str | None, version: int) -> str | None:
        """Promote rank-local full-sync decisions to all training ranks."""
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

        logger.warning(
            "separation delta v%d: FULL sync (global_full_sync from peer rank)",
            version,
        )
        return "global_full_sync"

    @staticmethod
    def _delta_verify_snapshot_enabled() -> bool:
        env = os.environ.get("DTE_DELTA_VERIFY_SNAPSHOT")
        if env is None or env.strip() == "":
            env = os.environ.get("AWEX_DELTA_VERIFY_SNAPSHOT", "")
        return env.strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def _delta_mark_snapshot_names(
        self, params_list: list[tuple[str, torch.Tensor]]
    ) -> None:
        names = getattr(self._delta_tracker, "_snapshot_names", None)
        if names is not None:
            names.update(name for name, _ in params_list)

    @torch.no_grad()
    def _delta_snapshot_masks_for_separation(
        self,
        params_list: list[tuple[str, torch.Tensor]],
        version: int,
    ) -> tuple[dict[str, torch.Tensor] | None, str | None]:
        """Build sparse-index masks by comparing current params to snapshot."""
        from dte.core import bitwise_changed_mask

        snapshot = getattr(self._delta_tracker, "_snapshot", None)
        if not snapshot:
            return None, "snapshot_baseline_missing"

        masks: dict[str, torch.Tensor] = {}
        total = 0
        changed_total = 0
        for name, current in params_list:
            cur = current.detach().contiguous()
            baseline = snapshot.get(name)
            if (
                baseline is None
                or baseline.dtype != cur.dtype
                or baseline.shape != cur.shape
            ):
                return None, f"snapshot_baseline_mismatch:{name}"

            mask = bitwise_changed_mask(
                cur, baseline.to(cur.device, non_blocking=False)
            ).reshape(-1)
            indices = mask.nonzero(as_tuple=False).squeeze(1).to(torch.int32)
            masks[name] = indices
            total += cur.numel()
            changed_total += int(indices.numel())

        logger.info(
            "separation delta v%d: snapshot masks changed %d/%d values",
            version,
            changed_total,
            total,
        )
        return masks, None

    @torch.no_grad()
    def _delta_apply_snapshot_mask_updates(
        self,
        params: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
        version: int,
        *,
        log_label: str,
    ) -> None:
        """Advance a tracker snapshot after a committed separation update."""
        snapshot = getattr(self._delta_tracker, "_snapshot", None)
        names = getattr(self._delta_tracker, "_snapshot_names", None)
        if snapshot is None or names is None:
            raise RuntimeError("Snapshot storage is unavailable")

        updated_values = 0
        for name, current in params.items():
            baseline = snapshot.get(name)
            mask = masks.get(name)
            numel = current.numel()
            if (
                baseline is None
                or baseline.dtype != current.dtype
                or baseline.numel() != numel
                or mask is None
            ):
                raise RuntimeError(f"Verification snapshot mismatch for {name}")

            if mask.dtype == torch.bool:
                if mask.numel() != numel:
                    raise RuntimeError(f"Verification mask size mismatch for {name}")
                indices = (
                    mask.to(current.device, non_blocking=False)
                    .reshape(-1)
                    .nonzero(as_tuple=False)
                    .squeeze(1)
                )
            elif mask.dtype in {torch.int32, torch.int64}:
                indices = mask.to(
                    current.device, dtype=torch.long, non_blocking=False
                ).reshape(-1)
                if indices.numel() > numel or (
                    indices.numel()
                    and bool(((indices < 0) | (indices >= numel)).any().item())
                ):
                    raise RuntimeError(
                        f"Verification mask index out of range for {name}"
                    )
            else:
                raise RuntimeError(
                    f"Unsupported verification mask dtype for {name}: {mask.dtype}"
                )

            names.add(name)
            if indices.numel() == 0:
                continue
            flat_current = current.detach().contiguous().reshape(-1)
            cpu_indices = indices.to("cpu")
            baseline.reshape(-1)[cpu_indices] = flat_current[indices].to(
                "cpu", non_blocking=False
            )
            updated_values += indices.numel()

        logger.info(
            "separation delta v%d %s committed: %d values",
            version,
            log_label,
            updated_values,
        )

    @torch.no_grad()
    def _delta_apply_separation_verification_updates(
        self,
        params: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
        version: int,
    ) -> None:
        """Advance the debug snapshot after a committed separation update."""
        self._delta_apply_snapshot_mask_updates(
            params,
            masks,
            version,
            log_label="verification snapshot",
        )

    def _delta_commit_separation_verification_snapshot(
        self,
        params: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor] | None,
        version: int,
        *,
        prepare_failed: bool,
        use_delta: bool = False,
    ) -> None:
        """Commit verification state only after transfer completion."""
        detector = getattr(self, "_delta_detector", None)
        detector_name = getattr(detector, "name", None)
        if detector_name == "snapshot":
            if use_delta:
                if masks is None:
                    raise RuntimeError(
                        "Snapshot delta committed without separation masks"
                    )
                self._delta_apply_snapshot_mask_updates(
                    params,
                    masks,
                    version,
                    log_label="snapshot baseline",
                )
            return

        action = dte_verification_snapshot_commit_action(
            detector_name,
            self._delta_verify_snapshot_enabled(),
            has_masks=masks is not None,
            prepare_failed=prepare_failed,
        )
        if action == "apply_masks":
            assert masks is not None
            self._delta_apply_separation_verification_updates(params, masks, version)
        elif action == "refresh":
            logger.warning(
                "separation delta v%d: refreshing verification snapshot after "
                "mask preparation failure",
                version,
            )
            self._delta_refresh_verification_snapshot(list(params.items()))

    @torch.no_grad()
    def _delta_refresh_verification_snapshot(
        self, params_list: list[tuple[str, torch.Tensor]]
    ) -> None:
        """Refresh the debug snapshot without resetting delta version counters."""
        snapshot = getattr(self._delta_tracker, "_snapshot", None)
        names = getattr(self._delta_tracker, "_snapshot_names", None)
        if snapshot is None or names is None:
            return

        snapshot.clear()
        names.clear()
        by_storage: dict[tuple[int, int], torch.Tensor] = {}
        pin = torch.cuda.is_available()
        for name, param in params_list:
            data = param.detach().contiguous()
            key = (data.data_ptr(), data.numel())
            cpu_tensor = by_storage.get(key)
            if cpu_tensor is None:
                cpu_tensor = data.cpu().clone()
                if pin:
                    cpu_tensor = cpu_tensor.pin_memory()
                by_storage[key] = cpu_tensor
            snapshot[name] = cpu_tensor
            names.add(name)

    @torch.no_grad()
    def _delta_verify_masks_against_snapshot(
        self,
        params_list: list[tuple[str, torch.Tensor]],
        masks: dict[str, torch.Tensor],
        version: int,
    ) -> bool:
        """Compare external detector masks against a resident snapshot baseline."""
        from dte.core import bitwise_changed_mask

        snapshot = getattr(self._delta_tracker, "_snapshot", None)
        if not snapshot:
            logger.error(
                "separation delta v%d verify snapshot-vs-%s FAILED: "
                "snapshot baseline is missing",
                version,
                self._delta_detector.name,
            )
            return False

        total = 0
        snapshot_changed = 0
        detector_changed = 0
        false_negative = 0
        false_positive = 0
        unverifiable = 0
        fn_examples: list[str] = []
        fp_examples: list[str] = []
        bad_examples: list[str] = []

        for name, cur in params_list:
            cur = cur.detach().contiguous()
            numel = cur.numel()
            total += numel
            snap = snapshot.get(name)
            det_mask = masks.get(name)
            bad_mask = snap is None or snap.dtype != cur.dtype or snap.numel() != numel
            det_indices = None
            if det_mask is not None and not bad_mask:
                if det_mask.dtype == torch.bool:
                    bad_mask = det_mask.numel() != numel
                elif det_mask.dtype in {torch.int32, torch.int64}:
                    det_indices = det_mask.to(cur.device).reshape(-1).long()
                    if det_indices.numel() > 0:
                        bad_mask = bool(
                            det_indices.min().item() < 0
                            or det_indices.max().item() >= numel
                        )
                else:
                    bad_mask = True

            if bad_mask:
                unverifiable += 1
                if len(bad_examples) < 5:
                    bad_examples.append(f"{name}:missing_or_bad_mask")
                continue

            snap_mask = bitwise_changed_mask(
                cur, snap.to(cur.device, non_blocking=False)
            ).reshape(-1)
            if det_mask is None:
                det_count = numel
                fn_count = 0
                fp_count = numel - int(snap_mask.sum().item())
            elif det_indices is None:
                det_mask = (
                    det_mask.to(cur.device, non_blocking=False).reshape(-1).bool()
                )
                det_count = int(det_mask.sum().item())
                fn_count = int((snap_mask & ~det_mask).sum().item())
                fp_count = int((det_mask & ~snap_mask).sum().item())
            else:
                det_mask = torch.zeros(numel, dtype=torch.bool, device=cur.device)
                if det_indices.numel() > 0:
                    det_mask[det_indices] = True
                det_count = int(det_mask.sum().item())
                fn_count = int((snap_mask & ~det_mask).sum().item())
                fp_count = int((det_mask & ~snap_mask).sum().item())

            snap_count = int(snap_mask.sum().item())
            snapshot_changed += snap_count
            detector_changed += det_count
            false_negative += fn_count
            false_positive += fp_count
            if fn_count and len(fn_examples) < 5:
                fn_examples.append(
                    f"{name}:snapshot={snap_count},detector={det_count},"
                    f"fn={fn_count},fp={fp_count}"
                )
            elif fp_count and len(fp_examples) < 5:
                fp_examples.append(
                    f"{name}:snapshot={snap_count},detector={det_count},"
                    f"fn={fn_count},fp={fp_count}"
                )

        examples = (fn_examples + bad_examples + fp_examples)[:5]
        if false_negative or unverifiable:
            logger.error(
                "separation delta v%d verify snapshot-vs-%s MISMATCH: "
                "snapshot_changed=%d detector_changed=%d total=%d "
                "false_negative=%d false_positive=%d unverifiable_params=%d "
                "examples=%s",
                version,
                self._delta_detector.name,
                snapshot_changed,
                detector_changed,
                total,
                false_negative,
                false_positive,
                unverifiable,
                examples,
            )
            return False

        if false_positive:
            logger.warning(
                "separation delta v%d verify snapshot-vs-%s OK conservative: "
                "snapshot_changed=%d detector_changed=%d total=%d "
                "false_positive=%d examples=%s",
                version,
                self._delta_detector.name,
                snapshot_changed,
                detector_changed,
                total,
                false_positive,
                examples,
            )
            return True

        logger.info(
            "separation delta v%d verify snapshot-vs-%s OK: changed=%d/%d",
            version,
            self._delta_detector.name,
            detector_changed,
            total,
        )
        return True

    def release_memory(self, tags: list[str] | None = None) -> None:
        tags = tags or ["optimizer", "weights"]
        tags_to_release = [t for t in tags if t not in self._released_tags]
        if not tags_to_release:
            return

        if "weights" in tags_to_release and self._weight_metadata_cache is None:
            # Metadata extraction all-gathers parameters over the TP group,
            # which is illegal once the DDP bucket storages are resized to 0.
            # Snapshot it while weights are still resident so /connect can be
            # served after the initial colocate release.
            self.get_weight_metadata()

        if "optimizer" in tags_to_release:
            self._offload_optimizer_states()
            self._released_tags.add("optimizer")

        if "weights" in tags_to_release:
            self._offload_model_weights()
            self._released_tags.add("weights")

        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("release_memory done: tags=%s", tags_to_release)

    def resume_memory(self, tags: list[str] | None = None) -> None:
        tags = tags or ["optimizer", "weights"]
        tags_to_resume = [t for t in tags if t in self._released_tags]
        if not tags_to_resume:
            return

        if "weights" in tags_to_resume:
            self._reload_model_weights(load_grad=False)
            self._released_tags.discard("weights")

        if "optimizer" in tags_to_resume:
            self._reload_optimizer_states()
            self._released_tags.discard("optimizer")

        torch.cuda.synchronize()
        logger.info("resume_memory done: tags=%s", tags_to_resume)

    @torch.no_grad()
    def _convert_parameters(self) -> dict[str, torch.Tensor]:
        """Convert every virtual-pipeline stage to Hugging Face names."""
        from awex.converter.mcore_converter import get_mcore_model_parameters

        model = self._engine.model
        if not isinstance(model, (list, tuple)):
            model = [model]

        converted = {}
        for vp_stage, chunk in enumerate(model):
            for name, param in get_mcore_model_parameters(chunk).items():
                for hf_name, hf_param in self._weight_converter.convert_param(
                    name, param.detach(), vp_stage=vp_stage
                ):
                    converted[hf_name] = hf_param

        hf_config = self._engine.hf_config
        if (
            getattr(hf_config, "tie_word_embeddings", False)
            and self._rank_info.pp_rank == self._rank_info.pp_size - 1
            and "lm_head.weight" not in converted
            and "model.embed_tokens.weight" in converted
        ):
            converted["lm_head.weight"] = converted["model.embed_tokens.weight"]
        return converted

    def _offload_model_weights(self) -> None:
        """Offload complete Megatron DDP buffers rather than parameter views."""
        from megatron.core.distributed import DistributedDataParallel as DDP

        model = self._engine.model
        if model is None:
            return
        if not isinstance(model, (list, tuple)):
            model = [model]
        count = 0
        for chunk in model:
            if isinstance(chunk, DDP):
                for buffers in [chunk.buffers, chunk.expert_parallel_buffers]:
                    for buf in buffers:
                        if hasattr(buf, "offload_to_cpu"):
                            buf.offload_to_cpu()
                            count += 1
                            continue
                        if buf.param_data.storage().size() > 0:
                            if not hasattr(buf.param_data, "cpu_data"):
                                buf.param_data.cpu_data = torch.zeros(
                                    buf.param_data.data.shape,
                                    dtype=buf.param_data.data.dtype,
                                    pin_memory=True,
                                    device="cpu",
                                )
                            buf.param_data.cpu_data.copy_(buf.param_data.data)
                            buf.param_data_size = buf.param_data.storage().size()
                            buf.param_data.storage().resize_(0)
                            count += 1
                        if buf.grad_data.storage().size() > 0:
                            buf.grad_data_size = buf.grad_data.storage().size()
                            buf.grad_data.storage().resize_(0)
            else:
                for name, param in chunk.named_parameters():
                    if param.data.is_cuda:
                        self._offloaded_weights[name] = param.data.detach().to(
                            "cpu", non_blocking=True
                        )
                        param.data = torch.empty(0, device="cpu")
                        count += 1
        torch.cuda.synchronize()
        logger.info("Offloaded %d weight buffers to CPU", count)

    def _reload_model_weights(self, load_grad: bool = False) -> None:
        from megatron.core.distributed import DistributedDataParallel as DDP

        model = self._engine.model
        if model is None:
            return
        if not isinstance(model, (list, tuple)):
            model = [model]
        device = self._engine.device
        for chunk in model:
            if isinstance(chunk, DDP):
                for buffers in [chunk.buffers, chunk.expert_parallel_buffers]:
                    for buf in buffers:
                        if hasattr(buf, "reload_from_cpu"):
                            buf.reload_from_cpu(move_grads=load_grad)
                            continue
                        if buf.param_data.storage().size() == 0:
                            buf.param_data.storage().resize_(buf.param_data_size)
                        buf.param_data.copy_(buf.param_data.cpu_data, non_blocking=True)
                        if (
                            load_grad
                            and hasattr(buf, "grad_data_size")
                            and buf.grad_data.storage().size() == 0
                        ):
                            buf.grad_data.storage().resize_(buf.grad_data_size)
                            buf.grad_data.zero_()
            else:
                for name, param in chunk.named_parameters():
                    if name in self._offloaded_weights:
                        param.data = self._offloaded_weights[name].to(
                            device, non_blocking=True
                        )
        self._offloaded_weights.clear()
        torch.cuda.synchronize()
        logger.info("Reloaded model weights to GPU (load_grad=%s)", load_grad)

    def ensure_grad_buffers(self) -> None:
        """Restore gradient storage before the next backward pass."""
        from megatron.core.distributed import DistributedDataParallel as DDP

        model = self._engine.model
        if model is None:
            return
        if not isinstance(model, (list, tuple)):
            model = [model]
        count = 0
        for chunk in model:
            if isinstance(chunk, DDP):
                for buffers in [chunk.buffers, chunk.expert_parallel_buffers]:
                    for buf in buffers:
                        if (
                            hasattr(buf, "grad_data_size")
                            and buf.grad_data.storage().size() == 0
                        ):
                            buf.grad_data.storage().resize_(buf.grad_data_size)
                            buf.grad_data.zero_()
                            count += 1
        if count:
            torch.cuda.synchronize()
            logger.info("Allocated %d grad buffers for training", count)

    def _get_inner_optimizers(self):
        optimizer = self._engine.optimizer
        if optimizer is None:
            return []
        if hasattr(optimizer, "chained_optimizers"):
            return optimizer.chained_optimizers
        if hasattr(optimizer, "optimizers"):
            return optimizer.optimizers
        return [optimizer]

    def _offload_optimizer_states(self) -> None:
        optimizer = self._engine.optimizer
        if optimizer is None:
            return

        if os.environ.get("AWEX_OPT_OFFLOAD_VIA_HDO", "").strip() == "1" and hasattr(
            optimizer, "offload_to_cpu"
        ):
            optimizer.offload_to_cpu()
            logger.info("Offloaded optimizer via offload_to_cpu()")
            return

        count = 0
        for opt in self._get_inner_optimizers():
            if hasattr(opt, "shard_fp32_from_float16_groups"):
                for group in opt.shard_fp32_from_float16_groups:
                    if isinstance(group, list):
                        for tensor in group:
                            if tensor is not None and tensor.data.is_cuda:
                                tensor.data = tensor.data.to("cpu", non_blocking=True)
                                count += 1
                    elif group is not None and group.data.is_cuda:
                        group.data = group.data.to("cpu", non_blocking=True)
                        count += 1

            base_opt = getattr(opt, "optimizer", opt)
            if not hasattr(base_opt, "state") or base_opt.state is None:
                continue
            for state in base_opt.state.values():
                for key in ("exp_avg", "exp_avg_sq"):
                    if (
                        key in state
                        and isinstance(state[key], torch.Tensor)
                        and state[key].is_cuda
                    ):
                        state[key] = state[key].to("cpu", non_blocking=True)
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
            if hasattr(opt, "shard_fp32_from_float16_groups"):
                for group in opt.shard_fp32_from_float16_groups:
                    if isinstance(group, list):
                        for tensor in group:
                            if tensor is not None and not tensor.data.is_cuda:
                                tensor.data = tensor.data.to(device, non_blocking=True)
                                count += 1
                    elif group is not None and not group.data.is_cuda:
                        group.data = group.data.to(device, non_blocking=True)
                        count += 1

            base_opt = getattr(opt, "optimizer", opt)
            if not hasattr(base_opt, "state") or base_opt.state is None:
                continue
            for state in base_opt.state.values():
                for key in ("exp_avg", "exp_avg_sq"):
                    if (
                        key in state
                        and isinstance(state[key], torch.Tensor)
                        and not state[key].is_cuda
                    ):
                        state[key] = state[key].to(device, non_blocking=True)
                        count += 1
        torch.cuda.synchronize()
        logger.info("Reloaded %d optimizer state tensors to GPU", count)


def _get_tf_config(models):
    if not isinstance(models, (list, tuple)):
        models = [models]
    for model in models:
        for attr in ("transformer_config", "config"):
            config = getattr(model, attr, None)
            if config is not None:
                return config
    return None
