# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import gc
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
from awex.util.tensor_util import (
    cuda_ipc_serialize,
    group_tensors_by_shape_and_dtype,
)

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
        self._offloaded_weights: dict[str, torch.Tensor] = {}
        self._released_tags: set[str] = set()
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
            self.enable_colocate_memory_management()

            if dist.get_rank(group=self._engine.cpu_group) == 0:
                self._meta_server_client.put_object(
                    "awex_train_info",
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
        self._offloaded_weights.clear()
        self._released_tags.clear()

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
        self._physical_gpu_id = resolve_physical_gpu_id(strict=True)

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
        from awex.util.tensor_util import release_tensors

        if self._meta_server_client is None:
            raise RuntimeError("init_colocate_weight_update must be called first")

        weights_were_offloaded = "weights" in self._released_tags
        torch.cuda.ipc_collect()
        try:
            # Optimizer and gradient buffers must leave the GPU before weights
            # are restored because inference is already resident on the same GPU.
            self.release_memory(tags=["optimizer"])
            self._release_grad_memory()
            if weights_were_offloaded:
                self.resume_memory(tags=["weights"])

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
                "finished_weights_update_engines",
                self._num_infer_engines,
                timeout=self._timeout_s,
            )
            for key in (
                "finished_weights_update_engines",
                "all_training_offloaded_weights",
            ):
                self._meta_server_client.delete_if_exists(key)
        dist.barrier(group=cpu_group)

    def release_memory(self, tags: list[str] | None = None) -> None:
        tags = tags or ["optimizer", "weights"]
        tags_to_release = [t for t in tags if t not in self._released_tags]
        if not tags_to_release:
            return

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
