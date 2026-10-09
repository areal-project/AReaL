# SPDX-License-Identifier: Apache-2.0
"""Checkpoint compatibility for Megatron's CPU offload optimizer."""

from collections.abc import Iterator
from types import MethodType
from typing import Any

import torch


@torch.no_grad()
def _restore_inner_master_params(optimizer: Any) -> None:
    if not optimizer.param_update_in_fp32:
        return
    # Native FP32 shards have an inner CPU copy but no param_to_fp32_param
    # entry. Restore the parameter actually consumed by the sub-optimizer.
    for param, state in optimizer.state.items():
        optimizer.param_to_inner_param[param].copy_(state["master_param"])


def iter_hybrid_optimizers(optimizer: Any) -> Iterator[Any]:
    if hasattr(optimizer, "chained_optimizers"):
        for child in optimizer.chained_optimizers:
            yield from iter_hybrid_optimizers(child)
    elif hasattr(optimizer, "optimizer"):
        yield from iter_hybrid_optimizers(optimizer.optimizer)
    elif hasattr(optimizer, "param_to_inner_param"):
        # Ordinary optimizers must not require the optional CPU-offload module.
        from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import (
            HybridDeviceOptimizer,
        )

        if isinstance(optimizer, HybridDeviceOptimizer):
            yield optimizer


def install_hybrid_optimizer_checkpoint_compat(optimizer: Any) -> None:
    """Handle both native FP32 shards and mixed-precision master parameters."""
    for hybrid in iter_hybrid_optimizers(optimizer):
        hybrid._update_fp32_params_by_new_state = MethodType(
            _restore_inner_master_params, hybrid
        )


def sync_loaded_hybrid_optimizer_state(optimizer: Any) -> None:
    """Publish loaded distributed checkpoint state to CPU/GPU sub-optimizers."""
    for hybrid in iter_hybrid_optimizers(optimizer):
        # dp_reshardable carries a nonpersistent template `step`, whereas
        # the checkpoint's authoritative Adam counter lives in param_groups.
        for group in hybrid.param_groups:
            if "step" in group:
                for param in group["params"]:
                    state = hybrid.state[param]
                    if isinstance(state.get("step"), torch.Tensor):
                        state["step"].fill_(group["step"])
        hybrid._sync_hdo_state_to_sub_optimizers()
