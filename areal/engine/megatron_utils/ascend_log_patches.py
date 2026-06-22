# SPDX-License-Identifier: Apache-2.0

"""Quiet known-benign NPU log spam (NPU-only; no-op elsewhere).

Three unrelated, harmless sources flood the per-rank logs, each filtered/guarded
below without editing the installed packages:

1. triton-ascend's JIT ``Please DO NOT tune args`` print on every kernel compile.
2. MindSpeed's ``tp_group is None`` DeprecationWarning, repeated per TP layer.
3. torch_npu's MSTX ``range_end`` TypeError from no-id native calls, per op-range.
"""

from __future__ import annotations

import warnings

import areal.utils.logging as logging

logger = logging.getLogger("AscendLogPatches")

_TRITON_NOISE = "Please DO NOT tune args"


def _silence_triton_tune_args_print() -> None:
    import importlib

    patched = []
    # triton.runtime.jit is where the active JITFunction.run resolves print;
    # triton_patch is patched too in case a code path uses that namespace.
    for modname in ("triton.runtime.jit", "triton.triton_patch.runtime.jit"):
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        if getattr(mod, "_areal_tune_warning_silenced", False):
            continue
        _orig_print = getattr(mod, "print", print)

        def _filtered_print(*args, _orig=_orig_print, **kwargs):
            if args and isinstance(args[0], str) and _TRITON_NOISE in args[0]:
                return
            return _orig(*args, **kwargs)

        mod.print = _filtered_print
        mod._areal_tune_warning_silenced = True
        patched.append(modname)
    if patched:
        logger.info(
            "Silenced triton-ascend tune-args log spam in: %s", ", ".join(patched)
        )


def _silence_tp_group_deprecation_warning() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"Warning: tp_group is None",
        category=DeprecationWarning,
    )
    logger.info("Filtered megatron-core 'tp_group is None' DeprecationWarning.")


def _silence_mstx_range_end_error() -> None:
    import importlib

    try:
        mod = importlib.import_module("torch_npu.npu.mstx")
    except Exception:
        return
    cls = getattr(mod, "mstx", None)
    if cls is None or getattr(cls, "_areal_range_end_guarded", False):
        return
    _orig_range_end = cls.range_end

    def _guarded_range_end(range_id=None, domain="default"):
        # The native MSTX emitter closes ranges via range_end() with no id;
        # drop those quietly, forward valid ids to the original.
        if not isinstance(range_id, int):
            return
        return _orig_range_end(range_id, domain)

    cls.range_end = staticmethod(_guarded_range_end)
    cls._areal_range_end_guarded = True
    logger.info("Guarded torch_npu mstx.range_end against no-id native calls.")


def _apply() -> None:
    from areal.infra.platforms import is_npu_available

    if not is_npu_available:
        return
    _silence_triton_tune_args_print()
    _silence_tp_group_deprecation_warning()
    _silence_mstx_range_end_error()


_apply()
