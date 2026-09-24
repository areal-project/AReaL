# SPDX-License-Identifier: Apache-2.0
"""Reject nonfinite optimizer norms before clipping changes gradient buffers."""

import functools
import json
import math
from collections.abc import Callable

import torch


@torch.no_grad()
def local_gradient_diagnostics(optimizer, chunk_size: int = 1048576) -> dict:
    """Inspect only this rank's norm inputs, without adding collectives.

    This runs only on failure. Chunking bounds temporary FP64 storage; results
    are local evidence, not a replacement for Core's global norm reduction.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    nonfinite = 0
    numel = 0
    sumsq = 0.0
    absmax = 0.0
    first_bad = None
    children = getattr(optimizer, "chained_optimizers", [optimizer])
    for child_idx, child in enumerate(children):
        for grad_idx, grad in enumerate(child.get_main_grads_for_grad_norm()):
            # Core norm inputs are contiguous flattened optimizer gradients.
            flat = grad.detach().view(-1)
            for part in flat.split(chunk_size):
                values = part.double()
                finite = torch.isfinite(values)
                bad = int((~finite).sum().item())
                if bad and first_bad is None:
                    first_bad = {
                        "optimizer": child_idx,
                        "gradient": grad_idx,
                        "shape": list(grad.shape),
                    }
                nonfinite += bad
                numel += values.numel()
                values = torch.where(finite, values, 0.0)
                sumsq += float(values.square().sum().item())
                if values.numel():
                    absmax = max(absmax, float(values.abs().max().item()))
    return {
        "scope": "rank_local",
        "numel": numel,
        "nonfinite_elements": nonfinite,
        "finite_absmax": absmax,
        "finite_fp64_norm": math.sqrt(sumsq),
        "first_nonfinite_gradient": first_bad,
    }


def guard_grad_norm(base: Callable) -> Callable:
    """Wrap Core's norm boundary, before ChainedOptimizer clips or steps."""
    if getattr(base, "_qwen_finite_norm_guard", False):
        return base

    @functools.wraps(base)
    def checked(optimizer, *args, **kwargs):
        norm = base(optimizer, *args, **kwargs)
        if not math.isfinite(float(norm)):
            try:
                details = local_gradient_diagnostics(optimizer)
            except Exception as exc:
                # Diagnostic failure must never allow the optimizer to proceed.
                details = {"diagnostic_error": type(exc).__name__}
            raise RuntimeError(
                f"Nonfinite optimizer gradient norm before clipping: {norm}; "
                f"{json.dumps(details, sort_keys=True)}"
            )
        return norm

    checked._qwen_finite_norm_guard = True
    return checked


def install() -> None:
    from megatron.core.optimizer.optimizer import ChainedOptimizer

    ChainedOptimizer.get_grad_norm = guard_grad_norm(ChainedOptimizer.get_grad_norm)
