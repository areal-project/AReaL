# SPDX-License-Identifier: Apache-2.0

"""Megatron flat-buffer GPU residency management.

This module deliberately has no AWEX transport or publication state.  It owns
the single source of truth for model-weight, optimizer-state, and gradient
buffer residency used by both persistent scoring workers and AWEX publishers.
"""

from __future__ import annotations

import gc
import os
from typing import TYPE_CHECKING, Any

import torch

from areal.utils.logging import getLogger

if TYPE_CHECKING:
    from areal.engine.megatron_engine import MegatronEngine


logger = getLogger("MegatronResidency")


class MegatronWeightResidency:
    """Own CPU/GPU residency for MCore DDP flat buffers and optimizer state."""

    def __init__(self, engine: MegatronEngine) -> None:
        self._engine = engine
        self._released_tags: set[str] = set()

    @property
    def released_tags(self) -> frozenset[str]:
        """Return an immutable snapshot of currently offloaded state tags."""
        return frozenset(self._released_tags)

    def is_released(self, tag: str) -> bool:
        """Return whether one residency tag is currently offloaded."""
        return tag in self._released_tags

    def release_memory(self, tags: list[str] | None = None) -> None:
        """Offload the requested state classes to CPU exactly once."""
        tags = tags or ["optimizer", "weights"]
        tags_to_release = [tag for tag in tags if tag not in self._released_tags]
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
        """Restore the requested state classes to GPU exactly once."""
        tags = tags or ["optimizer", "weights"]
        tags_to_resume = [tag for tag in tags if tag in self._released_tags]
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

    def _get_inner_optimizers(self) -> list[Any]:
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

        inner_optimizers = self._get_inner_optimizers()
        if not inner_optimizers:
            return

        count = 0
        for opt in inner_optimizers:
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

        inner_optimizers = self._get_inner_optimizers()
        if not inner_optimizers:
            return

        device = self._engine.device
        count = 0
        for opt in inner_optimizers:
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


__all__ = ["MegatronWeightResidency"]
