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
                            count += 1
                        if buf.grad_data.storage().size() > 0:
                            buf.grad_data_size = buf.grad_data.storage().size()
                            buf.grad_data.storage().resize_(0)
            else:
                raise RuntimeError(
                    "Megatron flat-buffer residency requires MCore DDP; "
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
        torch.cuda.synchronize()
        logger.info("Reloaded model weights to GPU (load_grad=%s)", load_grad)

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


__all__ = ["MegatronWeightResidency"]
