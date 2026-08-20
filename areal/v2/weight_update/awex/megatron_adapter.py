# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Literal

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

from areal.engine.weight_update.awex.colocate_megatron import MegatronColocateBackend
from areal.engine.weight_update.awex.colocate_protocol import ColocateKeyspace
from areal.utils import logging
from areal.v2.weight_update.awex import (
    awex_wu_use_group,
    fetch_kv_metadata,
    resolve_physical_gpu_id,
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
        self._transfer_rank: int | None = None
        self._colocate_backend = MegatronColocateBackend(
            engine,
            physical_gpu_id_resolver=lambda: resolve_physical_gpu_id(strict=True),
            normalize_infer_hf_config=True,
            allow_hdo_optimizer_offload=False,
        )
        self._offloaded_weights = self._colocate_backend.offloaded_weights
        self._released_tags = self._colocate_backend.released_tags
        self._meta_server_client = None
        self._weight_converter = None
        self._initialized = False
        self._rank_info: RankInfo | None = None
        self._ip_address: str | None = None
        self._physical_gpu_id: int | None = None
        self._infer_world_size: int | None = None
        self._num_infer_engines: int | None = None
        self._logical_train_rank: int | None = None
        self._active_mode: Literal["separation", "colocate"] | None = None
        self._init_fingerprint: tuple | None = None
        self._timeout_s = 300.0

    def _is_init_retry(
        self, mode: Literal["separation", "colocate"], fingerprint: tuple
    ) -> bool:
        if self._active_mode is None:
            return False
        if self._active_mode == mode and self._init_fingerprint == fingerprint:
            return True
        else:
            raise RuntimeError(
                f"AWEX adapter is already initialized for {self._active_mode} mode"
            )

    def enable_colocate_memory_management(self) -> None:
        """Route Megatron memory lifecycle hooks through this adapter."""
        current = getattr(self._engine, "_awex_adapter", None)
        if current not in (None, self):
            raise RuntimeError(
                "MegatronEngine already has a different AWEX colocate adapter"
            )
        self._engine._awex_adapter = self

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
        if self._is_init_retry("separation", fingerprint):
            return
        try:
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
            self._weights_update_group_gloo = init_weights_update_group(
                master_address=master_addr,
                master_port=master_port,
                rank=transfer_rank,
                world_size=world_size,
                group_name=f"awex_{pair_name}_gloo",
                backend="gloo",
                role="training",
            )
        except Exception:
            self.teardown_weight_update_group()
            raise
        self._active_mode = "separation"
        self._init_fingerprint = fingerprint
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
        del version
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
        if self._active_mode == "separation":
            self._active_mode = None
            self._init_fingerprint = None

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

    def _iter_hf_params(self):
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

        for mcore_name, param in get_named_parameters(
            self._engine.model, num_moe_experts
        ):
            gathered = all_gather_param(
                mcore_name,
                param,
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
        timeout_s: float,
    ) -> None:
        """Connect to the colocate MetaServer."""
        from awex.meta.meta_server import MetaServerClient

        actual_train_world_size = dist.get_world_size(group=self._engine.cpu_group)
        if train_world_size != actual_train_world_size:
            raise ValueError(
                "Colocate training world size mismatch: "
                f"gateway={train_world_size}, local={actual_train_world_size}"
            )
        if infer_world_size != train_world_size:
            raise ValueError(
                "Colocate mode requires equal inference and training rank counts; "
                f"got infer={infer_world_size}, train={train_world_size}"
            )
        if num_engines <= 0:
            raise ValueError(f"num_engines must be positive, got {num_engines}")

        fingerprint = (
            pair_name,
            meta_server_addr,
            transfer_rank,
            infer_world_size,
            train_world_size,
            num_engines,
            timeout_s,
        )
        if self._is_init_retry("colocate", fingerprint):
            return
        try:
            host, port = meta_server_addr.rsplit(":", 1)
            self._meta_server_client = MetaServerClient(host, int(port))
            self._timeout_s = timeout_s
            self._colocate_backend.configure(
                meta_server_client=self._meta_server_client,
                timeout_s=timeout_s,
            )
            self.enable_colocate_memory_management()

            if dist.get_rank(group=self._engine.cpu_group) == 0:
                self._meta_server_client.put_object(
                    ColocateKeyspace.AWEX_TRAIN_INFO,
                    {"train_world_size": dist.get_world_size(self._engine.cpu_group)},
                )
        except Exception:
            self._rollback_colocate_init()
            raise
        self._active_mode = "colocate"
        self._init_fingerprint = fingerprint
        logger.info(
            "Initialized colocate weight update for pair %r at %s, transfer_rank=%d",
            pair_name,
            meta_server_addr,
            transfer_rank,
        )

    def _rollback_colocate_init(self) -> None:
        """Reset session state while preserving already-offloaded train memory."""
        if getattr(self._engine, "_awex_adapter", None) is self:
            self._engine._awex_adapter = None
        self._colocate_backend.reset_session()
        self._meta_server_client = None
        self._weight_converter = None
        self._initialized = False
        self._rank_info = None
        self._ip_address = None
        self._physical_gpu_id = None
        self._infer_world_size = None
        self._num_infer_engines = None
        self._logical_train_rank = None
        self._active_mode = None
        self._init_fingerprint = None
        self._timeout_s = 300.0

    def teardown_colocate_weight_update(self) -> None:
        """Clear colocate state before the training engine is destroyed."""
        self._rollback_colocate_init()
        self._colocate_backend.clear_memory_state()

    def _lazy_initialize(self) -> None:
        self._colocate_backend.lazy_initialize()
        self._weight_converter = self._colocate_backend.weight_converter
        self._initialized = self._colocate_backend.initialized
        self._rank_info = self._colocate_backend.rank_info
        self._ip_address = self._colocate_backend.ip_address
        self._physical_gpu_id = self._colocate_backend.physical_gpu_id
        self._infer_world_size = self._colocate_backend.infer_world_size
        self._num_infer_engines = self._colocate_backend.num_infer_engines
        self._logical_train_rank = self._colocate_backend.logical_train_rank

    def _release_grad_memory(self) -> None:
        self._colocate_backend.release_grad_memory()

    @torch.no_grad()
    def execute_colocate_weight_update(self, version: int) -> None:
        self._colocate_backend.execute_weight_update(
            version,
            publish_offloaded_before_payload=False,
            restore_initial_weight_state=True,
            collect_ipc_after_update=True,
            wrap_reader_timeout=True,
        )
        self._lazy_initialize()

    def finish_colocate_weight_update(self, training_world_size: int) -> None:
        """Wait for all inference engines, then clear handshake state."""
        if self._meta_server_client is None or self._num_infer_engines is None:
            raise RuntimeError("Colocate weight update is not initialized")

        cpu_group = self._engine.cpu_group
        actual_world_size = dist.get_world_size(group=cpu_group)
        if training_world_size != actual_world_size:
            raise RuntimeError(
                "Training world size changed during colocate update: "
                f"expected={training_world_size}, actual={actual_world_size}"
            )

        # Every writer reaches this barrier only after its colocated reader has
        # acknowledged the per-device, per-version payload. Rank 0 can then
        # wait for one completion marker from each inference engine and clear
        # the reusable global handover keys before the next version begins.
        dist.barrier(group=cpu_group)
        if dist.get_rank(group=cpu_group) == 0:
            self._meta_server_client.wait_set_until_size(
                ColocateKeyspace.FINISHED_WEIGHT_UPDATE_ENGINES,
                self._num_infer_engines,
                timeout=self._timeout_s,
            )
            for key in (
                ColocateKeyspace.FINISHED_WEIGHT_UPDATE_ENGINES,
                ColocateKeyspace.ALL_TRAINING_OFFLOADED_WEIGHTS,
            ):
                self._meta_server_client.delete_if_exists(key)
        dist.barrier(group=cpu_group)

    def release_memory(self, tags: list[str] | None = None) -> None:
        self._colocate_backend.release_memory(tags)

    def resume_memory(self, tags: list[str] | None = None) -> None:
        self._colocate_backend.resume_memory(tags)

    @torch.no_grad()
    def _convert_parameters(self) -> dict[str, torch.Tensor]:
        return self._colocate_backend.convert_parameters()

    def _offload_model_weights(self) -> None:
        self._colocate_backend._offload_model_weights()

    def _reload_model_weights(self, load_grad: bool = False) -> None:
        self._colocate_backend._reload_model_weights(load_grad)

    def ensure_grad_buffers(self) -> None:
        self._colocate_backend.ensure_grad_buffers()

    def _get_inner_optimizers(self):
        return self._colocate_backend._get_inner_optimizers()

    def _offload_optimizer_states(self) -> None:
        self._colocate_backend._offload_optimizer_states()

    def _reload_optimizer_states(self) -> None:
        self._colocate_backend._reload_optimizer_states()
