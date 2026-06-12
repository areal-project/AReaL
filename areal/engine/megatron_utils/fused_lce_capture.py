# SPDX-License-Identifier: Apache-2.0
"""
LM-head hidden-state capture for the fused linear-cross-entropy fast path.

The fused LCE kernel needs ``(hidden, weight)`` instead of materialised
``[seq, vocab]`` logits.  This module temporarily monkey-patches
``output_layer.forward`` to capture those tensors for one microbatch.

Compatibility: incompatible with MuP (``use_mup``), MTP
(``mtp_num_layers > 0``), critic heads, and hidden sizes that do not
satisfy the fused-kernel alignment requirement.  The engine falls back
to the materialised path automatically when any of these conditions hold.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from megatron.core import parallel_state as mpu
from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
)

from areal.utils import logging

logger = logging.getLogger("FusedLCECapture")

FUSED_LCE_HIDDEN_KEY = "_fused_lce_hidden"
FUSED_LCE_WEIGHT_KEY = "_fused_lce_weight"
_HIDDEN_SIZE_ALIGNMENT = 128
_WARNED_INCOMPATIBILITIES: set[str] = set()


@dataclass
class _CaptureSlot:
    hidden: torch.Tensor | None = None
    weight: torch.Tensor | None = None


def _unwrap_to_post_process_module(model: torch.nn.Module) -> torch.nn.Module | None:
    inner = model
    for _ in range(8):
        if hasattr(inner, "output_layer") and inner.output_layer is not None:
            return inner
        if not hasattr(inner, "module"):
            return None
        inner = inner.module
    return None


def _warn_incompatible_once(key: str, message: str, *args: object) -> None:
    if key in _WARNED_INCOMPATIBILITIES:
        return
    _WARNED_INCOMPATIBILITIES.add(key)
    logger.warning(message, *args)


def _get_lm_head_hidden_size(
    config: object,
    output_layer: torch.nn.Module,
) -> int | None:
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is not None:
        return int(hidden_size)

    weight = getattr(output_layer, "weight", None)
    if weight is not None and hasattr(weight, "shape") and len(weight.shape) > 0:
        return int(weight.shape[-1])

    return None


def _is_compatible(post_process_module: torch.nn.Module) -> bool:
    config = getattr(post_process_module, "config", None)
    if config is None:
        return False

    if getattr(config, "use_mup", False):
        _warn_incompatible_once(
            "use_mup",
            "Fused LCE disabled: MuP scaling is enabled (config.use_mup=True).",
        )
        return False
    if getattr(config, "mtp_num_layers", 0):
        _warn_incompatible_once(
            "mtp", "Fused LCE disabled: MTP is enabled (config.mtp_num_layers>0)."
        )
        return False

    output_layer = getattr(post_process_module, "output_layer", None)
    if output_layer is None:
        return False

    hidden_size = _get_lm_head_hidden_size(config, output_layer)
    if hidden_size is not None and hidden_size % _HIDDEN_SIZE_ALIGNMENT != 0:
        _warn_incompatible_once(
            f"hidden_size:{hidden_size}",
            "Fused LCE disabled: hidden_size=%s is not divisible by %s.",
            hidden_size,
            _HIDDEN_SIZE_ALIGNMENT,
        )
        return False

    parallel_output = getattr(post_process_module, "parallel_output", True)
    if not parallel_output:
        _warn_incompatible_once(
            "parallel_output",
            "Fused LCE disabled: model has parallel_output=False; "
            "would require an extra TP gather.",
        )
        return False

    # The Triton kernel hard-requires hidden_size to be a multiple of 128
    # (BLOCK_HD constant). Surface this constraint at the gating layer so
    # incompatible models fall back to the materialised path before the
    # autograd graph is built; an assert raised inside ``backward`` would
    # otherwise hard-kill the training loop.
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None or hidden_size % 128 != 0:
        logger.warning(
            "Fused LCE disabled: hidden_size=%s is not a multiple of 128 "
            "(Triton kernel BLOCK_HD constraint).",
            hidden_size,
        )
        return False

    return True


@contextmanager
def capture_lm_head_hidden(
    model: torch.nn.Module, *, enabled: bool
) -> Iterator[_CaptureSlot | None]:
    if not enabled:
        yield None
        return

    post_process = _unwrap_to_post_process_module(model)
    if post_process is None or not _is_compatible(post_process):
        yield None
        return

    output_layer = post_process.output_layer
    slot = _CaptureSlot()
    original_forward = output_layer.forward

    config = getattr(post_process, "config", None)
    sequence_parallel = bool(getattr(config, "sequence_parallel", False))
    tp_world_size = mpu.get_tensor_model_parallel_world_size()
    needs_sp_gather = sequence_parallel and tp_world_size > 1

    def _patched_forward(input_, weight=None, runtime_gather_output=None):
        actual_weight = weight if weight is not None else output_layer.weight

        hidden = input_
        if needs_sp_gather:
            hidden = gather_from_sequence_parallel_region(hidden)

        if hidden.dtype != actual_weight.dtype:
            hidden = hidden.to(actual_weight.dtype)

        slot.hidden = hidden
        slot.weight = actual_weight
        return hidden, None

    output_layer.forward = _patched_forward  # type: ignore[assignment]
    try:
        yield slot
    finally:
        try:
            del output_layer.forward
        except AttributeError:
            output_layer.forward = original_forward  # type: ignore[assignment]


__all__ = [
    "FUSED_LCE_HIDDEN_KEY",
    "FUSED_LCE_WEIGHT_KEY",
    "capture_lm_head_hidden",
]
