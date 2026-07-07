# SPDX-License-Identifier: Apache-2.0
"""AWEX colocate adapter for MegatronEngine (training side).

Provides:
- Manual GPU→CPU offload for model weights and optimizer states
- CUDA IPC weight transfer to colocated SGLang (same GPU, via MetaServer)
- Coordinates with SGLang inference via MetaServer signals

Weight transfer flow (colocate mode):
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

from areal.utils.logging import getLogger

logger = getLogger("AwexColocateWriter")


class AwexColocateWriter:
    """Training-side adapter for AWEX colocated weight transfer.

    Uses CUDA IPC (share_memory + ForkingPickler serialization) for zero-copy
    weight transfer to the colocated SGLang process on the same GPU. The infer
    side handles redistribution among infer ranks via its own NCCL group.
    """

    def __init__(self, engine: MegatronEngine):
        self._engine = engine
        self._offloaded_weights: dict[tuple[int, str], torch.Tensor] = {}
        self._released_tags: set[str] = set()
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
        self._physical_gpu_id: int | None = None
        self._timeout_s: float = 300.0

    def init_colocate_weight_update(
        self,
        meta_server_addr: str | None = None,
        transfer_rank: int = 0,
        timeout_s: float = 300.0,
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
        # Train-side wait budget for infer's weights_update_finished signal,
        # overridable for slow diagnostic runs.
        env_timeout = os.environ.get("AWEX_COLOCATE_TIMEOUT_S", "").strip()
        if env_timeout:
            timeout_s = float(env_timeout)
        self._timeout_s = timeout_s
        if dist.get_rank() == 0:
            self._meta_server_client.put_object(
                "awex_train_info", {"train_world_size": dist.get_world_size()}
            )
            logger.info(
                "Registered awex_train_info (train_world_size=%d) with MetaServer",
                dist.get_world_size(),
            )

        logger.info(
            "AwexColocateWriter initialized: meta_server=%s, transfer_rank=%d",
            meta_server_addr,
            transfer_rank,
        )

    def _require_meta_server(self):
        if self._meta_server_client is None:
            raise RuntimeError(
                "init_colocate_weight_update() must be called before "
                "colocate weight update"
            )
        return self._meta_server_client

    def _lazy_initialize(self) -> None:
        """Perform deferred initialization: metadata exchange and weight converter setup.

        In colocate mode, train side does NOT join any NCCL group.
        Weight transfer uses CUDA IPC (share_memory + serialize via MetaServer).
        The infer side creates its own infer-only NCCL group for redistribution.
        """
        if self._initialized:
            return
        self._require_meta_server()

        from awex.models.registry import get_train_weights_converter
        from awex.sharding.param_sharding import get_rank_info_extractor
        from awex.util.common import get_ip_address

        rank = dist.get_rank()

        self._rank_info = get_rank_info_extractor("mcore")()
        training_world_size = self._rank_info.world_size
        self._ip_address = get_ip_address()
        # CUDA_VISIBLE_DEVICES is the ground truth for physical GPU assignment
        # (set by SlurmScheduler to isolate each rank to one GPU).
        # Cannot use torch.cuda.current_device() which always returns 0.
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_visible and "," not in cuda_visible:
            self._physical_gpu_id = int(cuda_visible)
        else:
            self._physical_gpu_id = torch.cuda.current_device()

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
        # pairing on the infer side reader.
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

    def _model_chunks(self) -> list:
        model = self._engine.model
        if model is None:
            return []
        return list(model) if isinstance(model, (list, tuple)) else [model]

    def _iter_ddp_buffers(self):
        """Yield Megatron DDP param/grad buffers across all model chunks."""
        from megatron.core.distributed import DistributedDataParallel as DDP

        for chunk in self._model_chunks():
            if isinstance(chunk, DDP):
                yield from chunk.buffers
                yield from chunk.expert_parallel_buffers

    @property
    def released_tags(self) -> frozenset[str]:
        return frozenset(self._released_tags)

    def release_grad_memory(self) -> None:
        """Release gradient buffers to free GPU memory before weight conversion.

        Saves original sizes to buffer.grad_data_size for later restoration.
        """
        count = 0
        for buf in self._iter_ddp_buffers():
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

        Flow:
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

        self._require_meta_server()

        weights_were_offloaded = "weights" in self._released_tags

        # Reclaim IPC-exported blocks from the previous version whose peer
        # mappings have closed since the last collect.
        torch.cuda.ipc_collect()

        # Free optimizer states + grad buffers BEFORE reloading the
        # train weights, not after. The colocated inference engine has already
        # resumed its full weight set on this same physical GPU, so reloading
        # the train param buffers (resize_ allocates GiB-scale storage per
        # buffer) can OOM unless optimizer + grad memory is freed first.
        # Optimizer/grad offload operate on independent Megatron buffers and do
        # not require the weights to be resumed, so reordering is safe.
        self.release_memory(tags=["optimizer"])

        self.release_grad_memory()

        if weights_were_offloaded:
            self.resume_memory(tags=["weights"])

        # _lazy_initialize must run AFTER the weights resume: its meta resolver
        # runs convert_param over live params and fails on resize_(0)-ed
        # storages (the recover path reaches the first transfer with weights
        # still offloaded).
        self._lazy_initialize()

        parameters = self._convert_parameters()
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
        # offload/reload and stay dangling forever, causing illegal memory
        # access in the router/EP path after the transfer. Only release
        # tensors we actually own (copies).
        live_storages = set()
        for chunk in self._model_chunks():
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

        for t in group_tensors:
            t.share_memory_()
        serialized_weights = cuda_ipc_serialize((group_tensors, metadata, names))
        torch.cuda.synchronize()
        logger.info("CUDA IPC serialization complete, putting to MetaServer")

        serialized_weights_key = f"training_serialized_weights{key_suffix}"
        # Tell the reader which version we are publishing; recover runs resume
        # at v=global_step rather than v1.
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
            self._meta_server_client.get_object(
                update_finished_key, timeout=self._timeout_s
            )
            self._meta_server_client.delete_if_exists(update_finished_key)
            logger.info("Got done signal from infer side: %s", update_finished_key)
        finally:
            self._meta_server_client.delete_if_exists(serialized_weights_key)
            release_tensors(group_tensors)
            del group_tensors
            torch.cuda.synchronize()
            gc.collect()
            # Storages exported via cudaIpcGetMemHandle park in PyTorch's
            # CudaIPCSentDataLimbo when freed and only return to the allocator
            # after ipc_collect() confirms the peer closed its mapping; without
            # this the train process leaks GiB-scale group tensors per version.
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

        write_finished_key = f"write_finished{key_suffix}"
        self._meta_server_client.put_object(write_finished_key, True)
        logger.info("Signaled write_finished: %s", write_finished_key)

        logger.info("Colocate weight update completed: version=%d", version)

    def finish_colocate_weight_update(self) -> None:
        """Wait for all inference engines to finish weight update, then clean up.

        Called from megatron_engine.update_weights() after barrier.
        """
        self._require_meta_server()
        if self._num_infer_engines is None:
            raise RuntimeError(
                "colocate writer not initialized: run a weight update first"
            )
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

        converted = {}
        for vp_stage, m in enumerate(self._model_chunks()):
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

    def _offload_model_weights(self) -> None:
        from megatron.core.distributed import DistributedDataParallel as DDP

        count = 0
        for chunk_idx, chunk in enumerate(self._model_chunks()):
            if isinstance(chunk, DDP):
                for buffers in (chunk.buffers, chunk.expert_parallel_buffers):
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
                        self._offloaded_weights[(chunk_idx, name)] = (
                            param.data.detach().to("cpu", non_blocking=True)
                        )
                        param.data = torch.empty(0, device="cpu")
                        count += 1
        torch.cuda.synchronize()
        logger.info("Offloaded %d weight buffers to CPU", count)

    def _reload_model_weights(self, load_grad: bool = False) -> None:
        from megatron.core.distributed import DistributedDataParallel as DDP

        device = self._engine.device
        for chunk_idx, chunk in enumerate(self._model_chunks()):
            if isinstance(chunk, DDP):
                for buffers in (chunk.buffers, chunk.expert_parallel_buffers):
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
                    key = (chunk_idx, name)
                    if key in self._offloaded_weights:
                        param.data = self._offloaded_weights[key].to(
                            device, non_blocking=True
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
        count = 0
        for buf in self._iter_ddp_buffers():
            if hasattr(buf, "grad_data_size") and buf.grad_data.storage().size() == 0:
                buf.grad_data.storage().resize_(buf.grad_data_size)
                buf.grad_data.zero_()
                count += 1
        if count > 0:
            torch.cuda.synchronize()
            logger.info("Allocated %d grad buffers for training", count)

    def _get_inner_optimizers(self):
        optimizer = self._engine.optimizer
        if optimizer is None:
            return []
        if hasattr(optimizer, "chained_optimizers"):
            inner_optimizers = optimizer.chained_optimizers
        elif hasattr(optimizer, "optimizers"):
            inner_optimizers = optimizer.optimizers
        else:
            inner_optimizers = [optimizer]
        return inner_optimizers

    def _move_optimizer_states(self, device: str | torch.device) -> int:
        """Move fp32 main params and Adam states to `device`; returns move count."""
        target = torch.device(device)
        to_gpu = target.type == "cuda"
        count = 0
        for opt in self._get_inner_optimizers():
            if hasattr(opt, "shard_fp32_from_float16_groups"):
                for group in opt.shard_fp32_from_float16_groups:
                    for t in group if isinstance(group, list) else [group]:
                        if t is not None and t.data.is_cuda != to_gpu:
                            t.data = t.data.to(target, non_blocking=True)
                            count += 1
            base_opt = getattr(opt, "optimizer", opt)
            if not hasattr(base_opt, "state") or base_opt.state is None:
                continue
            for state in base_opt.state.values():
                for key in ("exp_avg", "exp_avg_sq"):
                    val = state.get(key)
                    if isinstance(val, torch.Tensor) and val.is_cuda != to_gpu:
                        state[key] = val.to(target, non_blocking=True)
                        count += 1
        return count

    def _offload_optimizer_states(self) -> None:
        optimizer = self._engine.optimizer
        if optimizer is None:
            return
        # Swap .data / state-dict references to CPU, never resize_ storages,
        # then purge TE's global _dummy_wgrads cache and synchronize.
        count = self._move_optimizer_states("cpu")

        # transformer_engine caches dummy wgrad
        # tensors in a module-global dict; without purging it the GPU memory
        # is never actually freed and stale references survive the offload.
        try:
            from transformer_engine.pytorch.module.base import _dummy_wgrads

            if _dummy_wgrads:
                logger.info(
                    "Purged %d TE _dummy_wgrads cache entries", len(_dummy_wgrads)
                )
                _dummy_wgrads.clear()
        except ImportError:
            pass
        torch.cuda.synchronize()
        logger.info("Offloaded %d optimizer state tensors to CPU", count)

    def _reload_optimizer_states(self) -> None:
        optimizer = self._engine.optimizer
        if optimizer is None:
            return
        count = self._move_optimizer_states(self._engine.device)
        torch.cuda.synchronize()
        logger.info("Reloaded %d optimizer state tensors to GPU", count)


def _get_tf_config(models):
    if not isinstance(models, (list, tuple)):
        models = [models]
    for model in models:
        for attr in ("transformer_config", "config"):
            cfg = getattr(model, attr, None)
            if cfg is not None:
                return cfg
    return None
