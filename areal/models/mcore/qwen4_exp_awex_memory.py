# SPDX-License-Identifier: Apache-2.0
"""Keep Qwen4Exp cache resets outside unmapped KV residency windows."""

from dataclasses import dataclass
from types import ModuleType
from typing import Any

import torch


@dataclass
class _KVState:
    released: bool = False
    invalidated: bool = False


def _state(scheduler: Any) -> _KVState:
    state = getattr(scheduler, "_areal_qwen4_exp_kv_state", None)
    if state is None:
        state = _KVState()
        scheduler._areal_qwen4_exp_kv_state = state
    return state


def install_kv_residency_hooks(
    weight_updater: ModuleType, scheduler_type: type
) -> None:
    """Preserve native offload while resetting PLE caches only when resident.

    Native SGLang release calls flush_cache after unmapping KV. Qwen4Exp's PLE
    short-conv reset writes into that unmapped allocation. Clear and synchronize
    before release, acknowledge repeated invalidation while safely paused, then
    reset the newly mapped cache before returning from KV resume.
    """
    manager_type = weight_updater.SchedulerWeightUpdaterManager
    if getattr(manager_type, "_areal_qwen4_exp_kv_hooks", False):
        return
    kv_tag = weight_updater.GPU_MEMORY_TYPE_KV_CACHE
    original_release = manager_type.release_memory_occupation
    original_resume = manager_type.resume_memory_occupation
    original_flush = scheduler_type.flush_cache

    def applies(manager: Any, request: Any) -> bool:
        model = manager.tp_worker.model_runner.model
        return (
            type(model).__name__ == "Qwen4ExpForConditionalGeneration"
            and manager.memory_saver_adapter.enabled
            and (not request.tags or kv_tag in request.tags)
        )

    def flush_cache(scheduler: Any, *args: Any, **kwargs: Any) -> bool:
        state = getattr(scheduler, "_areal_qwen4_exp_kv_state", None)
        if state is not None and state.released:
            if not (
                state.invalidated
                and getattr(scheduler, "_engine_paused", False)
                and scheduler.running_batch.is_empty()
            ):
                raise RuntimeError(
                    "Cannot flush nonresident Qwen4Exp KV without a paused, invalidated cache"
                )
            # No request can create cache entries while retract-paused. AWEX's
            # post-transfer invalidation is already satisfied by pre-release
            # clearing; GPU reset is deferred until KV memory is mapped again.
            return True
        return original_flush(scheduler, *args, **kwargs)

    def release_memory_occupation(manager: Any, request: Any) -> Any:
        if not applies(manager, request):
            return original_release(manager, request)
        state = _state(manager.scheduler)
        if state.released:
            raise RuntimeError("Qwen4Exp KV memory is already released")
        # The existing retract-pause adapter supplies the idle override for
        # waiting requests. Never clear live, running requests here.
        if not manager.is_fully_idle():
            raise RuntimeError(
                "Qwen4Exp KV release requires an idle or retract-paused scheduler"
            )
        if not manager.flush_cache():
            raise RuntimeError("Qwen4Exp cache flush failed before KV release")
        torch.get_device_module().synchronize()
        state.invalidated = True
        saved_flush = manager.flush_cache
        # Native release still performs tag bookkeeping, static-state backup,
        # barriers, and memory-saver calls. Its late flush is already complete.
        manager.flush_cache = lambda *args, **kwargs: True
        try:
            result = original_release(manager, request)
        finally:
            manager.flush_cache = saved_flush
        state.released = True
        return result

    def resume_memory_occupation(manager: Any, request: Any) -> Any:
        if not applies(manager, request):
            return original_resume(manager, request)
        state = _state(manager.scheduler)
        result = original_resume(manager, request)
        state.released = False
        # KV has no CPU backup. Reinitialize PLE/request-pool bookkeeping in
        # newly allocated storage before generation is allowed to continue.
        if not manager.flush_cache():
            raise RuntimeError("Qwen4Exp cache flush failed after KV resume")
        torch.get_device_module().synchronize()
        state.invalidated = False
        return result

    scheduler_type.flush_cache = flush_cache
    manager_type.release_memory_occupation = release_memory_occupation
    manager_type.resume_memory_occupation = resume_memory_occupation
    manager_type._areal_qwen4_exp_kv_hooks = True
