# SPDX-License-Identifier: Apache-2.0

"""Megatron flat-buffer GPU residency management.

This module deliberately has no AWEX transport or publication state.  It owns
the single source of truth for model-weight, optimizer-state, and gradient
buffer residency used by both persistent scoring workers and AWEX publishers.
"""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING

import torch

from areal.engine.megatron_utils.optimizer_chain import (
    OptimizerResidencyEntry,
    OptimizerResidencyPlan,
    build_optimizer_residency_plan,
    checkpoint_awex_residency,
)
from areal.utils.logging import getLogger

if TYPE_CHECKING:
    from areal.engine.megatron_engine import MegatronEngine


logger = getLogger("MegatronResidency")


class MegatronWeightResidency:
    """Own CPU/GPU residency for MCore DDP flat buffers and optimizer state."""

    def __init__(self, engine: MegatronEngine) -> None:
        self._engine = engine
        self._released_tags: set[str] = set()
        self._optimizer_residency_plan: OptimizerResidencyPlan | None = None
        self._ordinary_optimizer_restores: dict[
            int, list[tuple[torch.Tensor, torch.device]]
        ] = {}
        # Layer-wise Muon uses DDP gradient buffers without param_data. Keep
        # storage identity intact: optimizer owner schemas and tied views refer
        # to the original storage, even while its allocation is absent.
        self._unbuffered_weights: dict[int, tuple] = {}
        self._master_backed_weights: dict[int, tuple] = {}

    @property
    def released_tags(self) -> frozenset[str]:
        """Return an immutable snapshot of currently offloaded state tags."""
        return frozenset(self._released_tags)

    def is_released(self, tag: str) -> bool:
        """Return whether one residency tag is currently offloaded."""
        return tag in self._released_tags

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

        adapter = getattr(self._engine, "_weftmoe_adapter", None)
        if adapter is not None and (
            "weights" in tags_to_release
            or getattr(adapter, "master_weights", None) is None
        ):
            adapter.suspend()

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

    def release_grad_memory(self) -> None:
        """Release gradient buffers while retaining sizes for training restore."""
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

    def ensure_grad_buffers(self) -> None:
        """Reallocate discarded gradient buffers before training."""
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

    def _offload_model_weights(self) -> None:
        from megatron.core.distributed import DistributedDataParallel as DDP

        model = self._engine.model
        if model is None:
            return
        if not isinstance(model, (list, tuple)):
            model = [model]
        flat_storages = {
            buf.param_data.untyped_storage()._cdata
            for chunk in model
            for buf in (*chunk.buffers, *chunk.expert_parallel_buffers)
            if buf.param_data is not None
        }
        self._offload_unbuffered_weights(model, flat_storages)
        count = 0
        nbytes = 0
        for chunk in model:
            if isinstance(chunk, DDP):
                for buffers in [chunk.buffers, chunk.expert_parallel_buffers]:
                    for buf in buffers:
                        if buf.param_data is not None:
                            size = buf.param_data.untyped_storage().nbytes()
                            nbytes += size
                            count += int(size > 0)
                        if hasattr(buf, "offload_to_cpu"):
                            buf.offload_to_cpu()
                            continue
                        if (
                            buf.param_data is not None
                            and buf.param_data.storage().size() > 0
                        ):
                            if not hasattr(buf, "cpu_param_data"):
                                buf.cpu_param_data = torch.zeros(
                                    buf.param_data.data.shape,
                                    dtype=buf.param_data.data.dtype,
                                    pin_memory=True,
                                    device="cpu",
                                )
                            buf.cpu_param_data.copy_(buf.param_data.data)
                            buf.param_data_size = buf.param_data.storage().size()
                            buf.param_data.storage().resize_(0)
                        if buf.grad_data.storage().size() > 0:
                            buf.grad_data_size = buf.grad_data.storage().size()
                            buf.grad_data.storage().resize_(0)
            else:
                raise RuntimeError(
                    "Megatron flat-buffer residency requires MCore DDP; "
                    "per-parameter weight offload is forbidden"
                )
        torch.cuda.synchronize()
        nbytes += sum(host.numel() for _, host, _ in self._unbuffered_weights.values())
        logger.info(
            "Offloaded %d flat weight buffers and %d unbuffered storages (%d bytes) to CPU; %d expert storages restored from existing FP32 masters",
            count,
            len(self._unbuffered_weights),
            nbytes,
            len(self._master_backed_weights),
        )

    @staticmethod
    def _storage_bytes(storage, size, device):
        return torch.empty(0, dtype=torch.uint8, device=device).set_(
            storage, 0, (size,), (1,)
        )

    @torch.no_grad()
    def _offload_unbuffered_weights(self, model, flat_storages) -> None:
        if self._unbuffered_weights or self._master_backed_weights:
            raise RuntimeError("Unbuffered weights are already offloaded")
        storages = {}
        grad_storages = {
            grad.untyped_storage()._cdata
            for chunk in model
            for param in chunk.parameters()
            for grad in (getattr(param, "main_grad", None), param.grad)
            if grad is not None
        }
        for chunk in model:
            for param in chunk.parameters():
                storage = param.untyped_storage()
                if storage._cdata in flat_storages or storage.nbytes() == 0:
                    continue
                if not storage.resizable():
                    raise RuntimeError("Cannot offload non-resizable parameter storage")
                if storage._cdata in grad_storages:
                    raise RuntimeError("Parameter storage aliases a gradient buffer")
                storages[storage._cdata] = (storage, param.device)
        # Complete consumers before releasing any allocation. Copy entire
        # storages once, including shared views with nonzero offsets.
        for device in {device for _, device in storages.values()}:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        masters = self._expert_restore_sources(model, storages)
        for key, (storage, device) in storages.items():
            size = storage.nbytes()
            if key in masters:
                param, owner, master = masters[key]
                self._master_backed_weights[key] = (
                    storage,
                    size,
                    param,
                    owner,
                    master,
                    master._version,
                )
                continue
            host = torch.empty(
                size, dtype=torch.uint8, device="cpu", pin_memory=device.type == "cuda"
            )
            host.copy_(self._storage_bytes(storage, size, device), non_blocking=True)
            self._unbuffered_weights[key] = (storage, host, device)
        for device in {device for _, device in storages.values()}:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        for storage, _ in storages.values():
            storage.resize_(0)

    @torch.no_grad()
    def _expert_restore_sources(self, model, storages):
        """Release Muon BF16 expert copies using their authoritative FP32 masters.

        Drain pending optimizer writes before recording restore sources. Do not
        copy or compare master values during offload: Muon produces the BF16
        parameters from these masters. Tied/partially owned matrices retain byte
        snapshots; checkpoint leases restore weights before changing masters.
        """
        optimizer = getattr(self._engine, "optimizer", None)
        owners = {
            id(unit.param): leaf.optimizer
            for leaf in getattr(optimizer, "chained_optimizers", ())
            if getattr(leaf.optimizer, "optimizer_kind", None) == "muon"
            for unit in getattr(leaf.optimizer, "units", ())
        }
        if not owners:
            return {}
        optimizer.drain()
        aliases = {}
        for chunk in model:
            for _, tensor in list(
                chunk.named_parameters(remove_duplicate=False)
            ) + list(chunk.named_buffers(remove_duplicate=False)):
                key = tensor.untyped_storage()._cdata
                aliases[key] = aliases.get(key, 0) + 1
        sources = {}
        for chunk in model:
            for param in chunk.parameters():
                key = param.untyped_storage()._cdata
                owner = owners.get(id(param))
                if (
                    key not in storages
                    or aliases.get(key) != 1
                    or owner is None
                    or getattr(param, "allreduce", True)
                    or param.dtype != torch.bfloat16
                    or not param.is_contiguous()
                    or param.storage_offset()
                    or param.numel() * param.element_size() != storages[key][0].nbytes()
                ):
                    continue
                master = owner.state[param].get("master_param")
                if (
                    master is None
                    or master.device.type != "cpu"
                    or master.dtype != torch.float32
                    or master.numel() != param.numel()
                ):
                    continue
                sources[key] = (param, owner, master)
        return sources

    @torch.no_grad()
    def _reload_unbuffered_weights(self) -> None:
        # Validate every source before restoring any storage. A changed master
        # while weights are released violates the residency lease ordering.
        for _, _, param, owner, master, version in self._master_backed_weights.values():
            if (
                owner.state[param]["master_param"] is not master
                or master._version != version
            ):
                raise RuntimeError(
                    "Muon master changed while model weights were offloaded"
                )
        for storage, size, param, _, master, _ in self._master_backed_weights.values():
            storage.resize_(size)
            param.data.copy_(master.to(device=param.device).view(param.shape))
        for storage, host, device in self._unbuffered_weights.values():
            storage.resize_(host.numel())
            self._storage_bytes(storage, host.numel(), device).copy_(
                host, non_blocking=True
            )
        for device in {device for _, _, device in self._unbuffered_weights.values()}:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        for device in {
            p.device for _, _, p, _, _, _ in self._master_backed_weights.values()
        }:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        self._unbuffered_weights.clear()
        self._master_backed_weights.clear()

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
                        if buf.param_data is not None:
                            if buf.param_data.storage().size() == 0:
                                buf.param_data.storage().resize_(buf.param_data_size)
                            buf.param_data.copy_(buf.cpu_param_data, non_blocking=True)
                        if (
                            load_grad
                            and hasattr(buf, "grad_data_size")
                            and buf.grad_data.storage().size() == 0
                        ):
                            buf.grad_data.storage().resize_(buf.grad_data_size)
                            buf.grad_data.zero_()
            else:
                raise RuntimeError(
                    "Cannot reload Megatron weights without MCore DDP flat buffers"
                )
        self._reload_unbuffered_weights()
        torch.cuda.synchronize()
        logger.info("Reloaded model weights to GPU (load_grad=%s)", load_grad)

    def _offload_optimizer_states(self) -> None:
        optimizer = self._engine.optimizer
        plan = build_optimizer_residency_plan(optimizer)
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
        torch.cuda.synchronize()
        self._ordinary_optimizer_restores.clear()
        self._optimizer_residency_plan = None
        logger.info("Restored managed and ordinary optimizer state")

    def _release_ordinary_optimizer(
        self, entry: OptimizerResidencyEntry
    ) -> list[tuple[torch.Tensor, torch.device]]:
        """Mirror AWEX's original ordinary Megatron optimizer migration."""
        restores: list[tuple[torch.Tensor, torch.device]] = []

        def move_tensor(tensor: torch.Tensor) -> None:
            if not tensor.data.is_cuda:
                return
            device = tensor.device
            tensor.data = tensor.data.to("cpu", non_blocking=True)
            restores.append((tensor, device))

        leaf = entry.leaf
        for group in getattr(leaf, "shard_fp32_from_float16_groups", ()):
            tensors = group if isinstance(group, list) else [group]
            for tensor in tensors:
                if tensor is not None:
                    move_tensor(tensor)

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
            for value in param_state.values():
                if isinstance(value, torch.Tensor):
                    move_tensor(value)
        return restores

    def _purge_te_cache(self) -> None:
        """Release Transformer Engine's private cached gradient buffers."""
        try:
            import transformer_engine.pytorch.module.base as te_base
        except ImportError:
            return
        cache = te_base._dummy_wgrads
        count = len(cache)
        cache.clear()
        if count:
            logger.info("Purged %d TE _dummy_wgrads cache entries", count)


__all__ = ["MegatronWeightResidency"]
