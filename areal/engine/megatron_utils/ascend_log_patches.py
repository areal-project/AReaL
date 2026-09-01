# SPDX-License-Identifier: Apache-2.0

"""Quiet known-benign NPU log spam (NPU-only; no-op elsewhere).

Two unrelated, harmless sources flood the per-rank logs, each filtered/guarded
below without editing the installed packages:

1. Transformers image-processor alias warnings on non-``Fast`` symbols.
2. torch_npu's MSTX ``range_end`` TypeError from no-id native calls, per op-range.
"""

from __future__ import annotations

import logging as stdlib_logging

import areal.utils.logging as logging

logger = logging.getLogger("AscendLogPatches")


def _silence_transformers_image_processing_alias_noise() -> None:
    """Keep v5.5.4's intended ``*Fast`` warnings, drop non-class alias spam."""
    import sys

    import transformers

    noise_filter = _TransformersImageAliasNoiseFilter()

    loggers = [transformers.logging.get_logger("transformers")]
    for name, module in sys.modules.items():
        if not (
            name.startswith("transformers.models.")
            and ".image_processing_" in name
            and name.endswith("_fast")
        ):
            continue
        getter = getattr(module, "__getattr__", None)
        alias_logger = getattr(getter, "__globals__", {}).get("logger")
        if alias_logger is not None:
            loggers.append(alias_logger)
            break

    for transformers_logger in {id(item): item for item in loggers}.values():
        if getattr(transformers_logger, "_areal_image_alias_noise_silenced", False):
            continue
        original_warning = transformers_logger.warning

        def _filtered_warning(
            message,
            *args,
            _logger=transformers_logger,
            _original=original_warning,
            **kwargs,
        ):
            record = stdlib_logging.LogRecord(
                _logger.name,
                stdlib_logging.WARNING,
                "",
                0,
                message,
                args,
                None,
            )
            if not noise_filter.filter(record):
                return
            return _original(message, *args, **kwargs)

        # Logger filters are cleared whenever AReaL rebuilds its logging config.
        # Wrapping these instances survives those later reconfigurations.
        transformers_logger.warning = _filtered_warning
        transformers_logger._areal_image_alias_noise_silenced = True
    logger.info("Filtered Transformers image-processing alias noise.")


class _TransformersImageAliasNoiseFilter(stdlib_logging.Filter):
    def filter(self, record: stdlib_logging.LogRecord) -> bool:
        message = record.getMessage()
        prefix = "Accessing `"
        if not (
            record.name == "transformers"
            and message.startswith(prefix)
            and " from `.models." in message
            and ".image_processing_" in message
            and message.endswith(
                "Behavior may be different and this alias will be removed in future versions."
            )
        ):
            return True
        name = message[len(prefix) :].partition("`")[0]
        return name.endswith("Fast")


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
        # The native MSTX emitter closes ranges via range_end() with no id.
        # torch_npu's own non-int guard cannot catch that: the call fails at
        # argument binding (range_id is positional-required) before the guard
        # runs, and its @_no_exception_func wrapper logs the TypeError per
        # op-range. The default here absorbs the no-arg call instead.
        if not isinstance(range_id, int):
            return
        return _orig_range_end(range_id, domain)

    cls.range_end = staticmethod(_guarded_range_end)
    cls._areal_range_end_guarded = True
    logger.info("Guarded torch_npu mstx.range_end against no-id native calls.")


def _apply() -> None:
    # Install this before platform detection imports torch. On Ascend,
    # importing torch_npu can trigger the malformed alias lookups.
    _silence_transformers_image_processing_alias_noise()

    from areal.infra.platforms import is_npu_available

    if not is_npu_available:
        return
    _silence_mstx_range_end_error()


_apply()
