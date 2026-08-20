# SPDX-License-Identifier: Apache-2.0

"""Controller-independent Megatron backend for AWEX colocation."""

from __future__ import annotations

import gc
import os
from collections.abc import Callable
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

from areal.engine.weight_update.awex.colocate_protocol import ColocateKeyspace
from areal.utils.logging import getLogger

if TYPE_CHECKING:
    from areal.engine.megatron_engine import MegatronEngine

logger = getLogger("AwexMegatronColocate")


class MegatronColocateBackend:
    """Megatron memory and CUDA-IPC data plane shared by v1 and v2.

    The facades retain controller-specific connection setup, retry handling,
    and finish barriers. This backend owns only the live Megatron memory state,
    lazy AWEX metadata/converter state, and per-device transfer handshake.
    """

    def __init__(
        self,
        engine: MegatronEngine,
        *,
        physical_gpu_id_resolver: Callable[[], int],
        normalize_infer_hf_config: bool = True,
        allow_hdo_optimizer_offload: bool = False,
    ) -> None:
        self._engine = engine
        self._physical_gpu_id_resolver = physical_gpu_id_resolver
        self._normalize_infer_hf_config = normalize_infer_hf_config
        self._allow_hdo_optimizer_offload = allow_hdo_optimizer_offload

        self.offloaded_weights: dict[str, torch.Tensor] = {}
        self.released_tags: set[str] = set()
        self.meta_server_client: Any | None = None
        self.timeout_s = 300.0
        self.weight_converter = None
        self.initialized = False
        self.rank_info = None
        self.ip_address: str | None = None
        self.physical_gpu_id: int | None = None
        self.infer_world_size: int | None = None
        self.num_infer_engines: int | None = None
        self.logical_train_rank: int | None = None

    def configure(self, *, meta_server_client: Any, timeout_s: float) -> None:
        """Attach one MetaServer session without changing memory ownership."""
        self.meta_server_client = meta_server_client
        self.timeout_s = timeout_s

    def reset_session(self) -> None:
        """Discard connection/converter state while preserving offloaded memory."""
        self.meta_server_client = None
        self.weight_converter = None
        self.initialized = False
        self.rank_info = None
        self.ip_address = None
        self.physical_gpu_id = None
        self.infer_world_size = None
        self.num_infer_engines = None
        self.logical_train_rank = None
        self.timeout_s = 300.0

    def clear_memory_state(self) -> None:
        self.offloaded_weights.clear()
        self.released_tags.clear()

    def lazy_initialize(self) -> None:
        """Resolve AWEX metadata and converter after live weights are resident."""
        if self.initialized:
            return
        if self.meta_server_client is None:
            raise RuntimeError("init_colocate_weight_update must be called first")

        from awex.meta.train_meta_resolver import McoreParamMetaResolver
        from awex.models.registry import get_train_weights_converter
        from awex.sharding.param_sharding import get_rank_info_extractor
        from awex.util.common import get_ip_address

        self.rank_info = get_rank_info_extractor("mcore")()
        self.ip_address = get_ip_address()
        self.physical_gpu_id = self._physical_gpu_id_resolver()

        infer_conf = self.meta_server_client.get_object(
            ColocateKeyspace.INFER_CONF, timeout=self.timeout_s
        )
        logger.info("Got infer_conf from MetaServer: %s", infer_conf)
        if self._normalize_infer_hf_config and isinstance(
            infer_conf.get("hf_config"), dict
        ):
            infer_conf["hf_config"] = SimpleNamespace(**infer_conf["hf_config"])

        shim = _EngineShim(self._engine)
        meta_resolver = McoreParamMetaResolver(shim, self._engine.hf_config, infer_conf)
        parameters_meta = meta_resolver.get_parameters_meta()
        if dist.get_rank() == 0:
            self.meta_server_client.put_object(
                ColocateKeyspace.TRAINING_PARAMS_META, parameters_meta
            )

        self.infer_world_size = infer_conf["infer_world_size"]
        # Shard ownership is expressed in Megatron global ranks. Controller
        # transfer ranks may use a different node order and are not wire ids.
        self.logical_train_rank = self.infer_world_size + self.rank_info.global_rank
        self.meta_server_client.add_object_to_set(
            ColocateKeyspace.TRAINING_DEVICE_RANK_ENTRIES,
            (self.ip_address, self.physical_gpu_id, self.logical_train_rank),
        )
        self.num_infer_engines = self.meta_server_client.get_object(
            ColocateKeyspace.NUM_INFER_ENGINES, timeout=self.timeout_s
        )

        self.weight_converter = get_train_weights_converter(
            "mcore",
            self._engine.hf_config.architectures[0],
            self._engine.hf_config,
            self.rank_info,
            {
                **infer_conf,
                "train_pp_stage_layer_id_map": (
                    meta_resolver.get_pp_stage_layer_id_map()
                ),
            },
            tf_config=_get_tf_config(self._engine.model),
        )
        self.initialized = True
        logger.info(
            "Colocate train side initialized: logical_train_rank=%d, "
            "infer_world_size=%d, train_world_size=%d",
            self.logical_train_rank,
            self.infer_world_size,
            self.rank_info.world_size,
        )

    def release_grad_memory(self) -> None:
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
    def execute_weight_update(
        self,
        version: int,
        *,
        publish_offloaded_before_payload: bool,
        restore_initial_weight_state: bool,
        collect_ipc_after_update: bool,
        wrap_reader_timeout: bool,
    ) -> None:
        """Publish converted parameters through CUDA IPC.

        The keyword options describe protocol/lifecycle behavior rather than a
        facade version. They preserve the existing operation order of both
        callers, including their exact CUDA synchronization and IPC collection
        counts.
        """
        from awex.util.tensor_util import (
            cuda_ipc_serialize,
            group_tensors_by_shape_and_dtype,
            release_tensors,
        )

        if self.meta_server_client is None:
            raise RuntimeError("init_colocate_weight_update must be called first")

        weights_were_offloaded = "weights" in self.released_tags
        torch.cuda.ipc_collect()
        try:
            self.release_memory(tags=["optimizer"])
            self.release_grad_memory()
            if weights_were_offloaded:
                self.resume_memory(tags=["weights"])

            self.lazy_initialize()
            parameters = self.convert_parameters()
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
            keyspace = self._keyspace()
            if publish_offloaded_before_payload:
                self._publish_training_offloaded()

            group_shared = [tensor.share_memory_() for tensor in group_tensors]
            serialized_weights = cuda_ipc_serialize((group_shared, metadata, names))
            torch.cuda.synchronize()

            self.meta_server_client.put_object(keyspace.writer_version, version)
            serialized_weights_key = keyspace.serialized_weights(version)
            self.meta_server_client.put_object(
                serialized_weights_key,
                (self.logical_train_rank, self.rank_info, serialized_weights),
            )
            if not publish_offloaded_before_payload:
                self._publish_training_offloaded()

            update_finished_key = keyspace.update_finished(version)
            try:
                try:
                    self.meta_server_client.get_object(
                        update_finished_key, timeout=self.timeout_s
                    )
                except Exception as exc:
                    if wrap_reader_timeout:
                        raise RuntimeError(
                            "Inference did not finish the colocate weight update "
                            f"within {self.timeout_s}s; missing key "
                            f"{update_finished_key!r}"
                        ) from exc
                    logger.error(
                        "Timed out or failed after %ss waiting for the inference "
                        "side to consume published weights (key=%s). The reader "
                        "likely died or never entered the colocate update; the "
                        "transfer cannot complete.",
                        self.timeout_s,
                        update_finished_key,
                    )
                    raise
                self.meta_server_client.delete_if_exists(update_finished_key)
                self.meta_server_client.delete_if_exists(serialized_weights_key)
            finally:
                release_tensors(group_tensors)
                release_tensors(group_shared)
                del group_tensors, group_shared
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()

            self.meta_server_client.put_object(keyspace.write_finished(version), True)
            logger.info("Colocate weight update completed: version=%d", version)
        finally:
            if collect_ipc_after_update:
                torch.cuda.ipc_collect()
            if (
                restore_initial_weight_state
                and weights_were_offloaded
                and "weights" not in self.released_tags
            ):
                self.release_memory(tags=["weights"])

    def _keyspace(self) -> ColocateKeyspace:
        if (
            self.ip_address is None
            or self.physical_gpu_id is None
            or self.logical_train_rank is None
            or self.rank_info is None
        ):
            raise RuntimeError("Colocate metadata is not initialized")
        return ColocateKeyspace(self.ip_address, self.physical_gpu_id)

    def _publish_training_offloaded(self) -> None:
        if self.meta_server_client is None or self.logical_train_rank is None:
            raise RuntimeError("Colocate metadata is not initialized")
        self.meta_server_client.add_object_to_set(
            ColocateKeyspace.ALL_TRAINING_OFFLOADED_WEIGHTS,
            self.logical_train_rank,
        )

    @torch.no_grad()
    def convert_parameters(self) -> dict[str, torch.Tensor]:
        """Convert every virtual-pipeline stage to Hugging Face names."""
        from awex.converter.mcore_converter import get_mcore_model_parameters

        if self.weight_converter is None or self.rank_info is None:
            raise RuntimeError("Colocate parameter converter is not initialized")

        model = self._engine.model
        if not isinstance(model, (list, tuple)):
            model = [model]

        converted = {}
        for vp_stage, chunk in enumerate(model):
            for name, param in get_mcore_model_parameters(chunk).items():
                for hf_name, hf_param in self.weight_converter.convert_param(
                    name, param.detach(), vp_stage=vp_stage
                ):
                    converted[hf_name] = hf_param

        hf_config = self._engine.hf_config
        if (
            getattr(hf_config, "tie_word_embeddings", False)
            and self.rank_info.pp_rank == self.rank_info.pp_size - 1
            and "lm_head.weight" not in converted
            and "model.embed_tokens.weight" in converted
        ):
            converted["lm_head.weight"] = converted["model.embed_tokens.weight"]
        return converted

    def release_memory(self, tags: list[str] | None = None) -> None:
        tags = tags or ["optimizer", "weights"]
        tags_to_release = [tag for tag in tags if tag not in self.released_tags]
        if not tags_to_release:
            return

        if "optimizer" in tags_to_release:
            self._offload_optimizer_states()
            self.released_tags.add("optimizer")

        if "weights" in tags_to_release:
            self._offload_model_weights()
            self.released_tags.add("weights")

        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("release_memory done: tags=%s", tags_to_release)

    def resume_memory(self, tags: list[str] | None = None) -> None:
        tags = tags or ["optimizer", "weights"]
        tags_to_resume = [tag for tag in tags if tag in self.released_tags]
        if not tags_to_resume:
            return

        if "weights" in tags_to_resume:
            self._reload_model_weights(load_grad=False)
            self.released_tags.discard("weights")

        if "optimizer" in tags_to_resume:
            self._reload_optimizer_states()
            self.released_tags.discard("optimizer")

        torch.cuda.synchronize()
        logger.info("resume_memory done: tags=%s", tags_to_resume)

    def _offload_model_weights(self) -> None:
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
                        self.offloaded_weights[name] = param.data.detach().to(
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
                    if name in self.offloaded_weights:
                        param.data = self.offloaded_weights[name].to(
                            device, non_blocking=True
                        )
        self.offloaded_weights.clear()
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
        if (
            self._allow_hdo_optimizer_offload
            and os.environ.get("AWEX_OPT_OFFLOAD_VIA_HDO", "").strip() == "1"
            and hasattr(optimizer, "offload_to_cpu")
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
        if (
            self._allow_hdo_optimizer_offload
            and os.environ.get("AWEX_OPT_OFFLOAD_VIA_HDO", "").strip() == "1"
            and hasattr(optimizer, "restore_from_cpu")
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


class _EngineShim:
    def __init__(self, engine: MegatronEngine) -> None:
        self.model = engine.model
        if not isinstance(self.model, (list, tuple)):
            self.model = [self.model]
        self.hf_config = engine.hf_config
        self.enable_debug_mode = False
        self.enable_colocate_mode = False
        self.engine_name = "mcore"
        self.config = {}
        self.meta_server_addr = ""

    def release_memory_occupation(self, tags=None) -> None:
        pass

    def resume_memory_occupation(self, tags=None) -> None:
        pass


def _get_tf_config(models):
    if not isinstance(models, (list, tuple)):
        models = [models]
    for model in models:
        for attr in ("transformer_config", "config"):
            config = getattr(model, attr, None)
            if config is not None:
                return config
    return None


__all__ = ["MegatronColocateBackend"]
