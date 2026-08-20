# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import gc
import os
import threading
import time
from typing import TYPE_CHECKING

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
from awex.transfer.nccl_stream_batch import NcclColocateStreamBatchTransport
from awex.transfer.transfer_plan import TransferPlan, TransferPlanBuilder, slice_tensor
from awex.util.tensor_util import (
    cuda_ipc_serialize,
    group_tensors_by_shape_and_dtype,
)

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
from areal.v2.weight_update.awex.delta_detect import AdamWInversionDetector
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
    The gateway's _merge_training_meta_by_name unions disjoint PP stage params
    by name, so the full model is covered across all PP ranks.

    TP: all_gather_param gathers the full tensor on every TP rank before
    convert_to_hf. dp_replicated=True tells awex that TP ranks within a DP
    group hold identical full tensors and only one needs to send.
    """

    def __init__(self, engine: MegatronEngine):
        self._engine = engine
        self._transfer_plan: TransferPlan | None = None
        self._weights_update_group = None
        self._weights_update_group_gloo = None
        self._world_size: int | None = None
        self._separation_delta_transport: NcclColocateStreamBatchTransport | None = None
        self._separation_wire_dtypes: tuple[torch.dtype, ...] | None = None
        self._transfer_rank: int | None = None
        self._offloaded_optimizer_states: dict = {}
        self._offloaded_weights: dict[str, torch.Tensor] = {}
        self._released_tags: set[str] = set()
        self._colocate_lock = threading.Lock()
        self._colocate_admin_api_key: str = "areal-admin-key"
        self._colocate_http_client: httpx.Client | None = None
        self._colocate_timeout_s: float = 120.0
        self._dte_config = DTERuntimeConfig.from_env()
        self._delta_tracker = None
        self._delta_detector = None

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
        if self._dte_config.enabled:
            validate_dte_world_size(world_size, infer_world_size, train_world_size)

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
        """Yield model tensors in the same order used by the HF converter."""
        from areal.engine.megatron_utils.megatron import get_named_parameters

        num_moe_experts = getattr(self._engine.tf_config, "num_moe_experts", None)
        seen: set[int] = set()
        for _mcore_name, param in get_named_parameters(
            self._engine.model, num_moe_experts
        ):
            if id(param) in seen:
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

        # Megatron's ChainedOptimizer wraps per-model-chunk optimizers;
        # each in turn wraps a base torch optimizer holding the state dict.
        if hasattr(optimizer, "optimizers"):
            inner_optimizers = optimizer.optimizers
        else:
            inner_optimizers = [optimizer]
            logger.warning(
                "Optimizer does not have 'optimizers' attribute. "
                "Treating it as a single optimizer; offload may be incomplete "
                "for non-standard Megatron optimizer structures."
            )
        for opt in inner_optimizers:
            base_opt = getattr(opt, "optimizer", opt)
            for param, state in base_opt.state.items():
                cpu_state: dict[str, torch.Tensor] = {}
                for key, val in state.items():
                    if isinstance(val, torch.Tensor) and val.is_cuda:
                        cpu_state[key] = val.detach().to("cpu", non_blocking=True)
                        state[key] = torch.empty(0, device="cpu")
                if cpu_state:
                    self._offloaded_optimizer_states[param] = cpu_state

        logger.info(
            "Offloaded optimizer states for %d params",
            len(self._offloaded_optimizer_states),
        )

    def _reload_optimizer_states(self) -> None:
        """Restore optimizer state tensors from CPU back to GPU."""
        if not self._offloaded_optimizer_states:
            return

        optimizer = self._engine.optimizer
        if optimizer is None:
            return

        inner_optimizers = getattr(optimizer, "optimizers", [optimizer])
        for opt in inner_optimizers:
            base_opt = getattr(opt, "optimizer", opt)
            for param, state in base_opt.state.items():
                if param in self._offloaded_optimizer_states:
                    cpu_state = self._offloaded_optimizer_states[param]
                    for key, val in cpu_state.items():
                        state[key] = val.to(param.device, non_blocking=True)

        self._offloaded_optimizer_states.clear()
        logger.info("Reloaded optimizer states to GPU")

    def _offload_model_weights(self) -> None:
        """Move model parameters to CPU, keeping references for reload."""
        if self._engine.model is None:
            return

        for name, param in self._engine.model.named_parameters():
            if param.is_cuda:
                self._offloaded_weights[name] = param.data.detach().to(
                    "cpu", non_blocking=True
                )
                param.data = torch.empty(0, device="cpu")

        logger.info(
            "Offloaded %d model weight tensors to CPU",
            len(self._offloaded_weights),
        )

    def _reload_model_weights(self) -> None:
        """Restore model parameters from CPU back to GPU."""
        if not self._offloaded_weights:
            return
        if self._engine.model is None:
            return

        device = self._engine.device
        for name, param in self._engine.model.named_parameters():
            if name in self._offloaded_weights:
                param.data = self._offloaded_weights[name].to(device, non_blocking=True)

        self._offloaded_weights.clear()
        logger.info("Reloaded model weights to GPU")
