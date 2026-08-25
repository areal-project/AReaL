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
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from areal.engine.megatron_engine import MegatronEngine

from areal.engine.megatron_utils.optimizer_chain import (
    OptimizerResidencyEntry,
    OptimizerResidencyPlan,
    build_optimizer_residency_plan,
    checkpoint_awex_residency,
)
from areal.engine.weight_finite import (
    check_named_tensors_finite,
    iter_module_named_tensors,
)
from areal.utils.logging import getLogger

logger = getLogger("AwexColocate")


@torch.no_grad()
def _pack_tensors_for_cuda_ipc(
    tensors: list[torch.Tensor],
    max_group_bytes: int = 5 * 1024**3,
) -> tuple[list[torch.Tensor], list[dict]]:
    """Pack tensors into bounded, single-allocation CUDA IPC groups.

    AWEX's packer uses ``torch.cat(...).clone()`` and only checks the size
    limit after adding the next tensor.  The redundant allocation doubles the
    transient pressure and makes it much more likely that the eventual export
    occupies a split training segment.  CUDA IPC exports the whole segment, so
    one live payload can pin tens of GiB of inactive splits.

    Plan groups first, allocate each flat destination exactly once, and copy
    shaped source views directly into it.  Grouping by dtype preserves AWEX's
    low-handle-count format and remains compatible with
    ``reconstruct_tensors_from_groups``.
    """
    if not tensors:
        return [], []

    by_dtype: dict[torch.dtype, list[tuple[int, torch.Tensor]]] = {}
    for index, tensor in enumerate(tensors):
        by_dtype.setdefault(tensor.dtype, []).append((index, tensor))

    planned_groups: list[list[tuple[int, torch.Tensor]]] = []
    for dtype_group in by_dtype.values():
        current: list[tuple[int, torch.Tensor]] = []
        current_bytes = 0
        for index, tensor in dtype_group:
            tensor_bytes = tensor.numel() * tensor.element_size()
            if current and current_bytes + tensor_bytes > max_group_bytes:
                planned_groups.append(current)
                current = []
                current_bytes = 0
            current.append((index, tensor))
            current_bytes += tensor_bytes
            if current_bytes >= max_group_bytes:
                planned_groups.append(current)
                current = []
                current_bytes = 0
        if current:
            planned_groups.append(current)

    packed_groups: list[torch.Tensor] = []
    metadata: list[dict] = []
    for group_index, group in enumerate(planned_groups):
        dtype = group[0][1].dtype
        device = group[0][1].device
        total_numel = sum(tensor.numel() for _, tensor in group)
        packed = torch.empty(total_numel, dtype=dtype, device=device)
        offset = 0
        for original_index, tensor in group:
            size = tensor.numel()
            packed[offset : offset + size].view(tensor.shape).copy_(tensor)
            metadata.append(
                {
                    "original_index": original_index,
                    "shape": tensor.shape,
                    "dtype": tensor.dtype,
                    "group_index": group_index,
                    "offset": offset,
                    "size": size,
                }
            )
            offset += size
        packed_groups.append(packed)

    return packed_groups, metadata


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
        self._optimizer_residency_plan: OptimizerResidencyPlan | None = None
        self._ordinary_optimizer_restores: dict[
            int, list[tuple[torch.Tensor, torch.device]]
        ] = {}
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
        self._pending_ipc_export: (
            tuple[list[torch.Tensor], list[torch.Tensor]] | None
        ) = None

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
          7. Signal write_finished while retaining the exported tensors
          8. Wait for all infer engines, then release the exported tensors
        """
        from awex.util.tensor_util import (
            cuda_ipc_serialize,
            release_tensors,
        )

        if self._pending_ipc_export is not None:
            raise RuntimeError(
                "Previous CUDA IPC export is still pending final reader completion"
            )

        weights_were_offloaded = "weights" in self._released_tags

        # Reclaim any legacy IPC-exported blocks from a previous version.
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

        group_tensors, metadata = _pack_tensors_for_cuda_ipc(tensors)
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
        gc.collect()
        torch.cuda.empty_cache()

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
        # Keep the producer storage alive until *all* readers complete their
        # full update lifecycle.  Releasing it after the early per-GPU ack puts
        # the storage into CudaIPCSentDataLimbo while readers can still hold a
        # mapping, which is exactly the actor residue this path must avoid.
        self._pending_ipc_export = (group_tensors, group_shared)
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
            torch.cuda.synchronize()

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

        # ``weights_update_finished`` is only an early acknowledgement.  The
        # final engine set is the first point at which every reader has returned
        # from its full lifecycle.  Drop the still-live producer tensors here,
        # after peer mappings are closed, so they bypass CudaIPCSentDataLimbo.
        allocated_before = torch.cuda.memory_allocated()
        reserved_before = torch.cuda.memory_reserved()
        pending_ipc_export = self._pending_ipc_export
        if pending_ipc_export is None:
            raise RuntimeError("CUDA IPC export disappeared before final completion")
        self._pending_ipc_export = None
        del pending_ipc_export
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
        allocated_after = torch.cuda.memory_allocated()
        reserved_after = torch.cuda.memory_reserved()
        logger.info(
            "Final CUDA IPC cleanup: rank=%d device=%d "
            "allocated=%.3f->%.3f GiB reserved=%.3f->%.3f GiB",
            dist.get_rank(self._engine.cpu_group),
            torch.cuda.current_device(),
            allocated_before / 1024**3,
            allocated_after / 1024**3,
            reserved_before / 1024**3,
            reserved_after / 1024**3,
        )

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

        if "optimizer" in tags_to_release:
            self._offload_optimizer_states()
        if "weights" in tags_to_release:
            self._offload_model_weights()

        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        self._released_tags.update(tags_to_release)
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
            self._optimizer_residency_plan = None
            self._ordinary_optimizer_restores.clear()
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
        plan = build_optimizer_residency_plan(optimizer, logger=logger)
        if self._ordinary_optimizer_restores:
            raise RuntimeError("stale ordinary optimizer state before AWEX release")
        ordinary_restores: dict[int, list[tuple[torch.Tensor, torch.device]]] = {}
        for index, entry in enumerate(plan.entries):
            if entry.managed_optimizer is not None:
                entry.managed_optimizer.offload_to_cpu()
            else:
                ordinary_restores[index] = self._release_ordinary_optimizer(entry)
        torch.cuda.synchronize()
        self._purge_te_cache()
        self._ordinary_optimizer_restores = ordinary_restores
        self._optimizer_residency_plan = plan
        logger.info(
            "Released optimizer state for %d managed and %d ordinary leaves",
            sum(entry.managed_optimizer is not None for entry in plan.entries),
            sum(entry.managed_optimizer is None for entry in plan.entries),
        )

    def _reload_optimizer_states(self) -> None:
        plan = self._optimizer_residency_plan
        if plan is None:
            return
        for index, entry in enumerate(plan.entries):
            if entry.managed_optimizer is not None:
                entry.managed_optimizer.restore_from_cpu()
                continue
            for tensor, device in self._ordinary_optimizer_restores.get(index, []):
                tensor.data = tensor.data.to(device, non_blocking=True)
        logger.info("Restored managed and ordinary optimizer state")

    def _release_ordinary_optimizer(
        self, entry: OptimizerResidencyEntry
    ) -> list[tuple[torch.Tensor, torch.device]]:
        """Mirror AWEX's original ordinary Megatron optimizer migration."""
        restores: list[tuple[torch.Tensor, torch.device]] = []
        seen: set[int] = set()

        def move_tensor(tensor: torch.Tensor, description: str) -> None:
            if id(tensor) in seen or not tensor.data.is_cuda:
                return
            if type(tensor) is not torch.Tensor:
                raise TypeError(
                    "AWEX ordinary optimizer migration supports only plain "
                    f"Tensor values, got {type(tensor).__module__}."
                    f"{type(tensor).__qualname__} for {description}"
                )
            seen.add(id(tensor))
            device = tensor.device
            tensor.data = tensor.data.to("cpu", non_blocking=True)
            restores.append((tensor, device))

        leaf = entry.leaf
        for group in getattr(leaf, "shard_fp32_from_float16_groups", ()):
            tensors = group if isinstance(group, list) else [group]
            for tensor in tensors:
                if tensor is not None:
                    move_tensor(tensor, "legacy FP32 main parameter")

        base_optimizer = entry.base_optimizer
        if base_optimizer is None:
            return restores
        state = getattr(base_optimizer, "state", None)
        if state is None:
            return restores
        if getattr(base_optimizer, "capturable", False):
            raise RuntimeError(
                "AWEX optimizer-state migration does not support capturable optimizers"
            )
        for param_state in state.values():
            for key in (
                "master_param",
                "exp_avg",
                "exp_avg_sq",
                "momentum_buffer",
            ):
                value = param_state.get(key)
                if isinstance(value, torch.Tensor):
                    move_tensor(value, f"optimizer state {key}")
        return restores

    def _purge_te_cache(self) -> None:
        """Release Transformer Engine's private cached gradient buffers."""
        try:
            import transformer_engine.pytorch.module.base as te_base
        except ImportError:
            return
        cache = te_base._dummy_wgrads
        if not isinstance(cache, dict):
            raise TypeError(
                "Transformer Engine 2.14.1 _dummy_wgrads must be a dict or "
                f"dict subclass, got {type(cache).__module__}.{type(cache).__qualname__}"
            )
        count = len(cache)
        cache.clear()
        if count:
            logger.info("Purged %d TE _dummy_wgrads cache entries", count)


def _get_tf_config(models):
    if not isinstance(models, (list, tuple)):
        models = [models]
    for model in models:
        for attr in ("transformer_config", "config"):
            cfg = getattr(model, attr, None)
            if cfg is not None:
                return cfg
    return None
