# SPDX-License-Identifier: Apache-2.0

"""Compatibility helpers for SGLang's torch-memory-saver integration."""

from __future__ import annotations

import os


def patch_tms_hook_mode() -> None:
    """Keep pauseable CUDA graphs on torch-memory-saver's preload hook.

    ``megatron.core.inference.contexts.dynamic_context`` assigns
    ``torch_memory_saver.hook_mode = "torch"`` at module import time. SGLang's
    pauseable CUDA graphs require the default ``preload`` hook, so drop that
    assignment when graph saving is enabled. Also ignore unsafe attempts to
    reconfigure the singleton after its implementation has been initialized.

    This must run from the SGLang entry module, before the scheduler imports
    Megatron transitively. Calling it later from the AWEX weight reader is too
    late because CUDA graphs are captured before that reader is constructed.
    """
    try:
        import torch_memory_saver as tms
    except Exception:
        return

    instance = getattr(tms, "torch_memory_saver", None)
    if instance is None:
        return
    cls = type(instance)
    prop = cls.hook_mode
    if getattr(prop.fset, "_awex_safe", False):
        return

    def safe_setter(self, value):
        if value == "torch" and os.environ.get(
            "SGLANG_MEMORY_SAVER_CUDA_GRAPH", ""
        ).lower() in {"1", "true", "yes", "on"}:
            return
        if not hasattr(self, "_impl_ctor_kwargs"):
            return
        prop.fset(self, value)

    safe_setter._awex_safe = True
    cls.hook_mode = property(prop.fget, safe_setter)


__all__ = ["patch_tms_hook_mode"]
