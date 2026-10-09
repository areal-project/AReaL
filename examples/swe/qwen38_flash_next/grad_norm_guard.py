# SPDX-License-Identifier: Apache-2.0
"""Reject nonfinite optimizer norms before clipping changes gradient buffers."""

import functools
import math
from collections.abc import Callable


def guard_grad_norm(base: Callable) -> Callable:
    """Wrap Core's norm boundary, before ChainedOptimizer clips or steps."""
    if getattr(base, "_qwen_finite_norm_guard", False):
        return base

    @functools.wraps(base)
    def checked(optimizer, *args, **kwargs):
        norm = base(optimizer, *args, **kwargs)
        if not math.isfinite(float(norm)):
            raise RuntimeError(
                f"Nonfinite optimizer gradient norm before clipping: {norm}"
            )
        return norm

    checked._qwen_finite_norm_guard = True
    return checked


def install() -> None:
    from megatron.core.optimizer.optimizer import ChainedOptimizer

    ChainedOptimizer.get_grad_norm = guard_grad_norm(ChainedOptimizer.get_grad_norm)
