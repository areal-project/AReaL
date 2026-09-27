# SPDX-License-Identifier: Apache-2.0

"""Activation checkpointing for the Hugging Face Qwen3.5 vision tower."""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def _checkpoint_forward(
    forward: Callable[..., torch.Tensor],
) -> Callable[..., torch.Tensor]:
    @functools.wraps(forward)
    def wrapped(*args: Any, **kwargs: Any) -> torch.Tensor:
        if torch.is_grad_enabled():
            return checkpoint(forward, *args, use_reentrant=False, **kwargs)
        return forward(*args, **kwargs)

    return wrapped


def checkpoint_qwen3_5_vision_blocks(models: Iterable[nn.Module]) -> int:
    """Checkpoint vision blocks missed by Megatron's language-model recompute.

    Transformers' Qwen3_5VisionModel currently calls its blocks directly, even
    when its gradient-checkpointing flag is enabled. Apply checkpointing only to
    this vision tower; model parameters and DDP registration stay unchanged.
    """
    count = 0
    for model in models:
        for module in model.modules():
            if type(module).__name__ != "Qwen3_5VisionModel" or not type(
                module
            ).__module__.startswith("transformers.models.qwen3_5"):
                continue
            for block in module.blocks:
                if getattr(block, "_areal_activation_checkpointed", False):
                    continue
                block.forward = _checkpoint_forward(block.forward)
                block._areal_activation_checkpointed = True
                count += 1
    return count
