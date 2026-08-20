# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0
"""AWEX colocate adapter for MegatronEngine (training side).

Provides:
- Manual GPU→CPU offload for model weights and optimizer states
- CUDA IPC weight transfer to colocated SGLang (same GPU, via MetaServer)
- Coordinates with SGLang inference via MetaServer signals

Weight transfer flow (mirrors the AWEX reference nccl_writer colocate mode):
  1. Convert Megatron params → HF format
  2. Group tensors by shape/dtype → share_memory_() → cuda_ipc_serialize
  3. Put serialized IPC handles to MetaServer
  4. Infer side (same GPU) deserializes via CUDA IPC (zero-copy)
  5. Infer-only NCCL group handles redistribution among infer ranks
  6. Infer signals done → train cleans up shared tensors

This adapter is used when weight_update_type == "awex" in colocate mode.
"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from areal.engine.megatron_engine import MegatronEngine

from areal.engine.megatron_utils.optimizer_chain import (
    OptimizerLifecyclePlan,
    OptimizerUndoAction,
    checkpoint_awex_residency,
    classify_optimizer_leaves,
    release_optimizer_lifecycle,
    resume_optimizer_lifecycle,
    retry_optimizer_recovery,
    rollback_optimizer_lifecycle,
)
from areal.engine.weight_finite import (
    check_named_tensors_finite,
    iter_module_named_tensors,
)
from areal.utils.logging import getLogger

logger = getLogger("AwexColocate")


@dataclass
class _TECachePurgeJournal:
    """Release-time snapshot and pending restores for TE's private dict cache."""

    cache: dict
    snapshot: tuple[tuple[object, object], ...]
    pending: list[tuple[object, object]]


def resolve_physical_gpu_id(relative_gpu_id: int) -> int:
    """Map a CUDA-masked relative device index to its physical GPU id.

    CUDA IPC keys must be unique per node, so both sides of a colocated
    transfer have to agree on physical GPU ids. Inside a process that was
    given a device mask, ``torch.cuda.current_device()`` and SGLang's
    ``gpu_id`` are indices into that mask rather than physical ids, so the
    mask itself is the only ground truth. Falls back to the relative index
    when the mask is absent or holds GPU UUIDs.
    """
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not cuda_visible:
        return relative_gpu_id
    try:
        gpu_ids = [int(x) for x in cuda_visible.split(",") if x.strip()]
        return gpu_ids[relative_gpu_id]
    except (ValueError, IndexError):
        return relative_gpu_id


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


class AwexMegatronAdapter:
    """Training-side adapter for AWEX colocated weight transfer.

    Uses CUDA IPC (share_memory + ForkingPickler serialization) for zero-copy
    weight transfer to the colocated SGLang process on the same GPU. The infer
    side handles redistribution among infer ranks via its own NCCL group.
    """

    def __init__(self, engine: MegatronEngine):
        self._engine = engine
        self._offloaded_weights: dict[str, torch.Tensor] = {}
        self._released_tags: set[str] = set()
        self._optimizer_lifecycle_cycle: OptimizerLifecyclePlan | None = None
        self._optimizer_rollback_recovery: OptimizerLifecyclePlan | None = None
        self._te_cache_purge_undo: _TECachePurgeJournal | None = None
        self._meta_server_addr: str | None = None
        self._meta_server_client = None
        self._transfer_rank: int | None = None
        self._weight_converter = None
        self._initialized = False
        self._rank_info = None
        self._ip_address: str | None = None
        self._infer_world_size: int | None = None
        self._num_infer_engines: int | None = None
        self._logical_train_rank: int | None = None

    def init_colocate_weight_update(
        self,
        meta_server_addr: str | None = None,
        pair_name: str = "default",
        transfer_rank: int = 0,
        timeout_s: float | None = None,
    ) -> None:
        """Initialize MetaServer connection. NCCL group creation is deferred
        to the first weight update (lazy init) to allow SGLang to start first.
        """
        from awex.meta.meta_server import MetaServerClient, start_meta_server

        if not meta_server_addr:
            meta_server_addr = os.environ.get("AWEX_META_SERVER_ADDR", "")
        if not meta_server_addr:
            host, port = start_meta_server()
            meta_server_addr = f"{host}:{port}"
            os.environ["AWEX_META_SERVER_ADDR"] = meta_server_addr
            logger.info("Started MetaServer at %s", meta_server_addr)

        host, port = meta_server_addr.rsplit(":", 1)
        self._meta_server_client = MetaServerClient(host, int(port))
        self._meta_server_addr = meta_server_addr
        self._transfer_rank = transfer_rank
        # Train-side wait budget for infer's weights_update_finished signal.
        # Keep the writer/reader/plugin on one env-controlled timeout to avoid
        # split-brain diagnostics.
        self._timeout_s = awex_colocate_timeout_s() if timeout_s is None else timeout_s
        if dist.get_rank() == 0:
            self._meta_server_client.put_object(
                "awex_train_info", {"train_world_size": dist.get_world_size()}
            )
            logger.info(
                "Registered awex_train_info (train_world_size=%d) with MetaServer",
                dist.get_world_size(),
            )

        logger.info(
            "AwexMegatronAdapter initialized: meta_server=%s, transfer_rank=%d",
            meta_server_addr,
            transfer_rank,
        )

    def _lazy_initialize(self) -> None:
        """Perform deferred initialization: metadata exchange and weight converter setup.

        In colocate mode, train side does NOT join any NCCL group.
        Weight transfer uses CUDA IPC (share_memory + serialize via MetaServer).
        The infer side creates its own infer-only NCCL group for redistribution.
        """
        if self._initialized:
            return

        from awex.models.registry import get_train_weights_converter
        from awex.sharding.param_sharding import get_rank_info_extractor
        from awex.util.common import get_ip_address

        rank = dist.get_rank()

        self._rank_info = get_rank_info_extractor("mcore")()
        training_world_size = self._rank_info.world_size
        self._ip_address = get_ip_address()
        self._physical_gpu_id = resolve_physical_gpu_id(torch.cuda.current_device())

        from awex.meta.train_meta_resolver import McoreParamMetaResolver

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

        shim = _EngineShim(self._engine)

        infer_conf = self._meta_server_client.get_object(
            "infer_conf", timeout=self._timeout_s
        )
        logger.info("Got infer_conf from MetaServer: %s", infer_conf)

        if isinstance(infer_conf.get("hf_config"), dict):
            from types import SimpleNamespace

            infer_conf["hf_config"] = SimpleNamespace(**infer_conf["hf_config"])

        meta_resolver = McoreParamMetaResolver(shim, self._engine.hf_config, infer_conf)
        parameters_meta = meta_resolver.get_parameters_meta()
        logger.info(
            "Collected training parameters metadata: %d params", len(parameters_meta)
        )

        if rank == 0:
            self._meta_server_client.put_object("training_params_meta", parameters_meta)
            logger.info("Registered training_params_meta with MetaServer")

        self._infer_world_size = infer_conf["infer_world_size"]
        self._logical_train_rank = self._infer_world_size + self._rank_info.global_rank

        # Register physical device entry for (ip, node_local_gpu_id) -> rank
        # pairing on the infer side (AWEX reader._init_reader_in_colocate_mode).
        # device_id must be the node-local physical GPU id (matching the infer
        # side and the CUDA IPC key), NOT a global rank. CUDA_VISIBLE_DEVICES is
        # the ground truth since torch.cuda.current_device() is always 0 here.
        self._meta_server_client.add_object_to_set(
            "training_device_rank_entries",
            (self._ip_address, self._physical_gpu_id, self._logical_train_rank),
        )
        logger.info(
            "Registered training_device_rank_entries: (ip=%s, gpu=%d, rank=%d)",
            self._ip_address,
            self._physical_gpu_id,
            self._logical_train_rank,
        )
        self._num_infer_engines = self._meta_server_client.get_object(
            "num_infer_engines", timeout=self._timeout_s
        )
        logger.info("Got num_infer_engines=%d from MetaServer", self._num_infer_engines)

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
            training_world_size,
        )

    def _release_grad_memory(self) -> None:
        """Release gradient buffers to free GPU memory before weight conversion.

        Mirrors the AWEX reference release_grad_memory().
        Saves original sizes to buffer.grad_data_size for later restoration.
        """
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
        if count > 0:
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
        logger.info("Released %d grad buffers", count)

    @torch.no_grad()
    def execute_colocate_weight_update(self, version: int) -> None:
        """Send weights to colocated inference via CUDA IPC through MetaServer.

        Flow (mirrors AWEX nccl_writer._write_weights_in_colocate_mode):
          1. Release optimizer states + grad memory to free GPU space
          2. Convert Megatron params to HF format
          3. Group tensors by shape/dtype → release originals → offload weights
          4. Signal all_training_offloaded_weights (reader waits for this)
          5. share_memory_() + cuda_ipc_serialize → put to MetaServer
          6. Wait for weights_update_finished (reader done copying)
          7. Clean up shared tensors → signal write_finished
          8. Wait for all infer engines to finish (finished_weights_update_engines)
        """
        from awex.util.tensor_util import (
            cuda_ipc_serialize,
            group_tensors_by_shape_and_dtype,
            release_tensors,
        )

        weights_were_offloaded = "weights" in self._released_tags

        # Reclaim any IPC-exported blocks from the previous version whose
        # peer mappings closed after our last collect (belt-and-braces with the
        # collect in this method's finally block).
        torch.cuda.ipc_collect()

        # Free optimizer states + grad buffers BEFORE reloading the train
        # weights, not after. The colocated inference engine has already
        # resumed its full PP=1 weight set on this same physical GPU, so
        # reloading the train param buffers (resize_ allocates GiB-scale
        # storage per buffer) OOMs on the very first buffer unless optimizer +
        # grad memory is freed first. This matches the AWEX weights_writer
        # order (release_memory(optimizer) THEN resume_memory(weights), see
        # _release_memory_for_weights_exchange).
        # Optimizer/grad offload operate on independent Megatron buffers and do
        # not require the weights to be resumed, so reordering is safe.
        self.release_memory(tags=["optimizer"])

        self._release_grad_memory()

        if weights_were_offloaded:
            self.resume_memory(tags=["weights"])

        check_named_tensors_finite(
            iter_module_named_tensors(self._engine.model),
            stage="awex_writer_source",
            version=version,
            logger=logger,
            process_group=self._engine.cpu_group,
        )

        # _lazy_initialize AFTER the weights resume — its meta resolver
        # runs convert_param over live params, which dies with CUDA invalid
        # argument on resize_(0)-ed storages. The recover path is the only
        # one that reaches the first transfer with weights offloaded (see
        # re-releases them right after the recover load), which is why the
        # normal step-1 path never hit this.
        self._lazy_initialize()

        parameters = self._convert_parameters()
        check_named_tensors_finite(
            parameters.items(),
            stage="awex_writer_converted",
            version=version,
            logger=logger,
            process_group=self._engine.cpu_group,
        )
        tensors = list(parameters.values())
        names = list(parameters.keys())
        logger.info(
            "Converted %d params for colocate IPC transfer (version=%d)",
            len(tensors),
            version,
        )

        group_tensors, metadata = group_tensors_by_shape_and_dtype(tensors)
        torch.cuda.synchronize()
        logger.info(
            "Grouped into %d tensor groups for IPC serialization", len(group_tensors)
        )

        # convert_param returns the ORIGINAL tensor (shared storage)
        # whenever the conversion is an identity — direct_name_mapping
        # (embed_tokens / final_layernorm) returns `parameter` as-is, and
        # `.to(dtype)` (gate.weight, expert_bias) is a no-op when the dtype
        # already matches. release_tensors() does untyped_storage().resize_(0),
        # so releasing those entries frees the LIVE module storage. Params are
        # later rebuilt by the weights offload/reload round-trip, but buffers
        # (e.g. the fp32 router expert_bias, a register_buffer) are not part of
        # offload/reload and stay dangling forever -> the post-transfer IMA in
        # the router/EP path. Only release tensors we actually own (copies).
        live_storages = set()
        model = self._engine.model
        for chunk in model if isinstance(model, (list, tuple)) else [model]:
            # NB: Megatron DDP shadows nn.Module.buffers with a LIST attribute
            # (ParamAndGradBuffer); named_parameters/named_buffers stay methods.
            for _, p in chunk.named_parameters():
                live_storages.add(p.untyped_storage().data_ptr())
            for _, b in chunk.named_buffers():
                live_storages.add(b.untyped_storage().data_ptr())
        owned = [
            t for t in tensors if t.untyped_storage().data_ptr() not in live_storages
        ]
        logger.info(
            "Releasing %d/%d converted tensors (%d alias live module storage, "
            "left to the weights offload path)",
            len(owned),
            len(tensors),
            len(tensors) - len(owned),
        )
        release_tensors(owned)
        del tensors, owned
        parameters.clear()

        self.release_memory(tags=["weights"])

        ip_address = self._ip_address
        device_id = self._physical_gpu_id
        key_suffix = f"_{ip_address}_{device_id}_{version}"

        self._meta_server_client.add_object_to_set(
            "all_training_offloaded_weights", self._logical_train_rank
        )
        logger.info(
            "Signaled all_training_offloaded_weights (rank=%d)",
            self._logical_train_rank,
        )

        group_shared = [t.share_memory_() for t in group_tensors]
        serialized_weights = cuda_ipc_serialize((group_shared, metadata, names))
        torch.cuda.synchronize()
        logger.info("CUDA IPC serialization complete, putting to MetaServer")

        serialized_weights_key = f"training_serialized_weights{key_suffix}"
        # Tell the reader which version we are publishing. The plugin's
        # background worker used to assume the stream starts at v1, which
        # deadlocks recover runs (writer resumes at v=global_step, e.g. 9).
        writer_version_key = f"awex_writer_version_{ip_address}_{device_id}"
        self._meta_server_client.put_object(writer_version_key, version)
        logger.info(
            "Put writer version to MetaServer: key=%s version=%s",
            writer_version_key,
            version,
        )
        self._meta_server_client.put_object(
            serialized_weights_key,
            (self._logical_train_rank, self._rank_info, serialized_weights),
        )
        logger.info("Put IPC weights to MetaServer: key=%s", serialized_weights_key)

        update_finished_key = f"weights_update_finished{key_suffix}"
        try:
            try:
                completion = self._meta_server_client.get_object(
                    update_finished_key, timeout=self._timeout_s
                )
            except Exception:
                logger.error(
                    "Timed out or failed after %ss waiting for the inference "
                    "side to consume published weights (key=%s). The reader "
                    "likely died or never entered the colocate update; the "
                    "transfer cannot complete.",
                    self._timeout_s,
                    update_finished_key,
                )
                raise
            if isinstance(completion, dict) and not completion.get("ok", True):
                error = completion.get("error", "unknown inference-side error")
                raise RuntimeError(
                    "Inference rejected AWEX weights before IPC release: "
                    f"version={version}, device={device_id}, error={error}"
                )
            self._meta_server_client.delete_if_exists(update_finished_key)
            self._meta_server_client.delete_if_exists(serialized_weights_key)
            logger.info("Got done signal from infer side: %s", update_finished_key)
        finally:
            release_tensors(group_tensors)
            release_tensors(group_shared)
            del group_tensors, group_shared
            torch.cuda.synchronize()
            gc.collect()
            # Storages exported via cudaIpcGetMemHandle park in PyTorch's
            # CudaIPCSentDataLimbo when freed and are NOT returned to the
            # allocator until ipc_collect() confirms the peer closed its
            # mapping. GiB-scale group tensors are exported every version;
            # without collection the train-process residual grows by GBs per
            # version, eating the colocated rollout's prefill headroom until
            # its logits all-gather OOMs.
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

        write_finished_key = f"write_finished{key_suffix}"
        self._meta_server_client.put_object(write_finished_key, True)
        logger.info("Signaled write_finished: %s", write_finished_key)

        logger.info("Colocate weight update completed: version=%d", version)

    def finish_colocate_weight_update(self, training_world_size: int) -> None:
        """Wait for all inference engines to finish weight update, then clean up.

        Mirrors AWEX _finish_weights_update().
        Called from megatron_engine.update_weights() after barrier.
        """
        num_infer_engines = self._num_infer_engines
        logger.info(
            "Waiting for %d inference engine(s) to signal finished_weights_update_engines",
            num_infer_engines,
        )
        self._meta_server_client.wait_set_until_size(
            "finished_weights_update_engines",
            num_infer_engines,
            timeout=self._timeout_s,
        )
        logger.info("All inference engines finished weights update")

        dist.barrier(group=self._engine.cpu_group)

        if dist.get_rank() == 0:
            self._meta_server_client.delete_if_exists("finished_weights_update_engines")
            self._meta_server_client.delete_if_exists("all_training_offloaded_weights")
        logger.info("Cleaned up MetaServer coordination keys")

    @torch.no_grad()
    def _convert_parameters(self) -> dict[str, torch.Tensor]:
        """Convert Megatron parameters to HF format for IPC transfer."""
        from awex.converter.mcore_converter import get_mcore_model_parameters

        model = self._engine.model
        if not isinstance(model, (list, tuple)):
            model = [model]

        converted = {}
        for vp_stage, m in enumerate(model):
            for name, param in get_mcore_model_parameters(m).items():
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

        logger.info("Converted %d parameters for IPC transfer", len(converted))
        return converted

    # ── Memory management (manual offload) ────────────────────────────────

    def checkpoint_residency(self, *, with_model: bool, with_optimizer: bool):
        """Temporarily restore only resources required by a checkpoint."""
        return checkpoint_awex_residency(
            self,
            self._engine.optimizer,
            with_model=with_model,
            with_optimizer=with_optimizer,
        )

    def release_memory(self, tags: list[str] | None = None) -> None:
        tags = tags or ["optimizer", "weights"]
        tags_to_release = [t for t in tags if t not in self._released_tags]
        if not tags_to_release:
            return

        optimizer_released = False
        try:
            if "optimizer" in tags_to_release:
                self._offload_optimizer_states()
                optimizer_released = True

            if "weights" in tags_to_release:
                self._offload_model_weights()

            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
        except BaseException as original:
            if optimizer_released:
                self._rollback_te_cache_purge(original)
                self._rollback_optimizer_release(original)
            raise

        self._released_tags.update(tags_to_release)
        self._te_cache_purge_undo = None
        logger.info("release_memory done: tags=%s", tags_to_release)

    def resume_memory(self, tags: list[str] | None = None) -> None:
        tags = tags or ["optimizer", "weights"]
        tags_to_resume = [t for t in tags if t in self._released_tags]
        if not tags_to_resume:
            return

        if "weights" in tags_to_resume:
            self._reload_model_weights(load_grad=False)
        if "optimizer" in tags_to_resume:
            self._reload_optimizer_states()
        torch.cuda.synchronize()
        self._released_tags.difference_update(tags_to_resume)
        if "optimizer" in tags_to_resume:
            self._optimizer_lifecycle_cycle = None
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
                raise RuntimeError(
                    "AWEX Megatron colocation requires MCore DDP flat buffers; "
                    "per-parameter weight offload is forbidden"
                )
        torch.cuda.synchronize()
        logger.info("Offloaded %d weight buffers to CPU", count)

    def _reload_model_weights(self, load_grad: bool = False) -> None:
        from megatron.core.distributed import DistributedDataParallel as DDP

        model = self._engine.model
        if model is None:
            return
        if not isinstance(model, (list, tuple)):
            model = [model]
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
                raise RuntimeError(
                    "Cannot reload AWEX Megatron weights without MCore DDP flat buffers"
                )
        self._offloaded_weights.clear()
        torch.cuda.synchronize()
        logger.info("Reloaded model weights to GPU (load_grad=%s)", load_grad)

    def ensure_grad_buffers(self) -> None:
        """Allocate grad buffers if they were freed during offload.

        Called before forward_backward (training) to ensure grad storage
        is available for backward pass. Separate from _reload_model_weights
        because compute_logp (inference-only) should not allocate grad buffers.
        """
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
        if count > 0:
            torch.cuda.synchronize()
            logger.info("Allocated %d grad buffers for training", count)

    def _offload_optimizer_states(self) -> None:
        optimizer = self._engine.optimizer
        if (
            self._optimizer_rollback_recovery is not None
            or self._te_cache_purge_undo is not None
        ):
            raise RuntimeError("optimizer release has unresolved rollback state")
        # Default path mirrors the AWEX reference optimizer offload
        # (megatron_util.offload_megatron_optimizer): swap .data / state-dict
        # references to CPU, never resize_ storages, then purge TE's global
        # _dummy_wgrads cache and synchronize. Megatron HybridDeviceOptimizer's
        # offload_to_cpu/restore_from_cpu is kept only as an opt-in fallback —
        # its internal pointer bookkeeping is hard to validate and the AWEX
        # reference integration deliberately avoids it.
        use_hdo_lifecycle = (
            os.environ.get("AWEX_OPT_OFFLOAD_VIA_HDO", "").strip() == "1"
        )
        te_cache_journal = self._prepare_te_cache_purge()
        cycle = classify_optimizer_leaves(
            optimizer,
            use_hdo_lifecycle=use_hdo_lifecycle,
            logger=logger,
        )
        try:
            release_optimizer_lifecycle(
                cycle,
                self._release_ordinary_optimizer,
                logger=logger,
            )
        except BaseException:
            if cycle.has_pending_recovery():
                self._optimizer_rollback_recovery = cycle
            self._optimizer_lifecycle_cycle = None
            raise

        try:
            # Complete all non-blocking state copies before publishing the
            # cycle to the outer release transaction.
            torch.cuda.synchronize()
        except BaseException as original:
            rollback_optimizer_lifecycle(cycle, original, logger=logger)
            if cycle.has_pending_recovery():
                self._optimizer_rollback_recovery = cycle
            raise

        try:
            if te_cache_journal is not None:
                self._purge_te_cache_transactionally(te_cache_journal)
        except BaseException as original:
            self._rollback_te_cache_purge(original)
            rollback_optimizer_lifecycle(cycle, original, logger=logger)
            if cycle.has_pending_recovery() or self._te_cache_purge_undo is not None:
                self._optimizer_rollback_recovery = cycle
            self._optimizer_lifecycle_cycle = None
            raise

        self._optimizer_lifecycle_cycle = cycle
        logger.info("Offloaded optimizer state tensors to CPU")

    def _reload_optimizer_states(self) -> None:
        cycle = self._optimizer_lifecycle_cycle
        if cycle is None:
            return
        for node_type in cycle.cycle_node_types:
            logger.warning(
                "Detected optimizer chain cycle at %s; using release-time plan",
                node_type,
            )
        resume_optimizer_lifecycle(cycle, logger=logger)
        logger.info("Reloaded optimizer state tensors to GPU")

    def _rollback_optimizer_release(self, original: BaseException) -> None:
        cycle = self._optimizer_lifecycle_cycle
        if cycle is None:
            return
        rollback_optimizer_lifecycle(cycle, original, logger=logger)
        if cycle.has_pending_recovery() or self._te_cache_purge_undo is not None:
            self._optimizer_rollback_recovery = cycle
        self._optimizer_lifecycle_cycle = None

    def _retry_optimizer_rollback_recovery(self) -> None:
        cycle = self._optimizer_rollback_recovery
        if cycle is None and self._te_cache_purge_undo is None:
            return
        recovery_error: BaseException | None = None
        if cycle is not None:
            try:
                retry_optimizer_recovery(cycle, logger=logger)
            except BaseException as error:
                recovery_error = error
        if self._te_cache_purge_undo is not None:
            try:
                self._restore_te_cache_items()
            except BaseException as error:
                if recovery_error is None:
                    recovery_error = error
                else:
                    recovery_error.add_note(
                        f"Additional AWEX TE cache recovery failure: {error!r}"
                    )
        if (
            cycle is not None
            and not cycle.has_pending_recovery()
            and self._te_cache_purge_undo is None
        ):
            self._optimizer_rollback_recovery = None
        if recovery_error is not None:
            raise recovery_error

    def _prepare_te_cache_purge(self) -> _TECachePurgeJournal | None:
        # ImportError compatibility is deliberately limited to importing TE.
        # Descriptor access, validation, snapshot, and mapping operations are
        # part of the optimizer release transaction and must propagate.
        try:
            import transformer_engine.pytorch.module.base as te_base
        except ImportError:
            return None
        return self._snapshot_te_cache(te_base._dummy_wgrads)

    @staticmethod
    def _snapshot_te_cache(cache) -> _TECachePurgeJournal:
        # Transformer Engine 2.14.1 initializes private `_dummy_wgrads` with a
        # dict literal. This compatibility adapter supports that concrete
        # contract (and subclasses with trustworthy built-in dict storage),
        # not arbitrary MutableMapping implementations.
        if not isinstance(cache, dict):
            raise TypeError(
                "Transformer Engine 2.14.1 _dummy_wgrads must be a dict or "
                f"dict subclass, got {type(cache).__module__}.{type(cache).__qualname__}"
            )
        snapshot = tuple(dict.items(cache))
        return _TECachePurgeJournal(
            cache=cache,
            snapshot=snapshot,
            pending=list(snapshot),
        )

    def _purge_te_cache_transactionally(self, journal: _TECachePurgeJournal) -> None:
        # Publish the complete trusted snapshot before the first dynamic
        # deletion. A dict subclass may delete any original entry before
        # raising, so prefix-only undo registration is insufficient.
        self._te_cache_purge_undo = journal
        for key, _value in journal.snapshot:
            del journal.cache[key]
        if journal.snapshot:
            logger.info(
                "Purged %d TE _dummy_wgrads cache entries", len(journal.snapshot)
            )

    def _restore_te_cache_items(self) -> None:
        if self._te_cache_purge_undo is None:
            return
        journal = self._te_cache_purge_undo
        failed: list[tuple[object, object]] = []
        restore_errors: list[BaseException] = []
        for key, value in reversed(tuple(journal.pending)):
            # Assignment is deliberately unconditional: it restores the
            # original value whether deletion happened before or after an
            # exception, without a separate membership-check failure window.
            try:
                journal.cache[key] = value
            except BaseException as error:
                failed.append((key, value))
                restore_errors.append(error)
            journal.pending = list(reversed(failed))

        # The supported contract treats built-in dict storage as authoritative.
        # Rebuild it from the trusted pre-snapshot to remove deletion side
        # effects (including newly inserted keys) and restore value identity.
        # AWEX requires this private TE cache to be quiescent during lifecycle.
        dict.clear(journal.cache)
        for key, value in journal.snapshot:
            dict.__setitem__(journal.cache, key, value)
        if restore_errors:
            primary = restore_errors[0]
            for error in restore_errors[1:]:
                primary.add_note(f"Additional AWEX TE cache restore failure: {error!r}")
            raise primary
        self._te_cache_purge_undo = None

    def _rollback_te_cache_purge(self, original: BaseException) -> None:
        if self._te_cache_purge_undo is None:
            return
        try:
            self._restore_te_cache_items()
        except BaseException as error:
            original.add_note(f"AWEX TE cache rollback failed: {error!r}")
            try:
                logger.error(
                    "AWEX TE cache rollback failed",
                    exc_info=(type(error), error, error.__traceback__),
                )
            except BaseException as logging_error:
                original.add_note(
                    f"AWEX failed to log TE rollback error: {logging_error!r}"
                )

    def _release_ordinary_optimizer(self, index, lifecycle, journal) -> None:
        del index
        opt = lifecycle.leaf
        if hasattr(opt, "shard_fp32_from_float16_groups"):
            for group in opt.shard_fp32_from_float16_groups:
                tensors = group if isinstance(group, list) else [group]
                for tensor in tensors:
                    if tensor is not None and tensor.data.is_cuda:
                        device = tensor.device
                        tensor.data = tensor.data.to("cpu", non_blocking=True)
                        journal.actions.append(
                            OptimizerUndoAction(
                                restore=lambda tensor=tensor, device=device: setattr(
                                    tensor,
                                    "data",
                                    tensor.data.to(device, non_blocking=True),
                                ),
                                description="legacy FP32 main parameter",
                            )
                        )

        base_opt = lifecycle.base_optimizer
        if base_opt is None or not hasattr(base_opt, "state") or base_opt.state is None:
            return
        if getattr(base_opt, "capturable", False):
            raise RuntimeError(
                "AWEX optimizer-state migration does not support capturable optimizers"
            )
        for state in base_opt.state.values():
            # Transformer Engine's precision-aware Adam owns its main parameter
            # in optimizer state instead of
            # ``shard_fp32_from_float16_groups``.  Move it with the moments so
            # the optimizer tag releases every CUDA-resident owner before the
            # colocated inference engine resumes its memory mappings.
            for key in ("master_param", "exp_avg", "exp_avg_sq"):
                if (
                    key in state
                    and isinstance(state[key], torch.Tensor)
                    and state[key].is_cuda
                ):
                    tensor = state[key]
                    if type(tensor) is not torch.Tensor:
                        raise TypeError(
                            "AWEX optimizer-state migration only supports plain "
                            f"Tensor values, got {type(tensor).__module__}."
                            f"{type(tensor).__qualname__} for {key}"
                        )
                    device = tensor.device
                    # Preserve the optimizer-state Tensor identity while
                    # replacing its storage.  External holders of this exact
                    # object (for example sharded/checkpoint metadata) then
                    # observe the move; replacing only ``state[key]`` can
                    # leave such holders owning the old CUDA storage.
                    cpu_data = tensor.data.to("cpu", non_blocking=True)
                    journal.actions.append(
                        OptimizerUndoAction(
                            restore=lambda tensor=tensor, device=device: setattr(
                                tensor,
                                "data",
                                tensor.data.to(device, non_blocking=True),
                            ),
                            description=f"optimizer state {key}",
                        )
                    )
                    tensor.data = cpu_data


def _get_tf_config(models):
    if not isinstance(models, (list, tuple)):
        models = [models]
    for model in models:
        for attr in ("transformer_config", "config"):
            cfg = getattr(model, attr, None)
            if cfg is not None:
                return cfg
    return None
