# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.optim import ZeroRedundancyOptimizer

from areal.api.cli_args import OptimizerConfig


def parallelize_fn_zero1(
    model: nn.Module,
) -> nn.Module:
    """Zero-1 path keeps full parameter replicas without model wrapper."""
    return model


def create_zero1_optimizer(
    params: list[nn.Parameter],
    optimizer_config: OptimizerConfig,
    dp_group: dist.ProcessGroup,
) -> ZeroRedundancyOptimizer:
    """Create ZeroRedundancyOptimizer from optimizer config."""
    common_kwargs: dict[str, object] = {
        "lr": optimizer_config.lr,
        "weight_decay": optimizer_config.weight_decay,
    }
    if optimizer_config.type == "adam":
        return ZeroRedundancyOptimizer(
            params,
            optimizer_class=torch.optim.AdamW,
            process_group=dp_group,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            eps=optimizer_config.eps,
            fused=True,
            **common_kwargs,
        )
    if optimizer_config.type == "sgd":
        return ZeroRedundancyOptimizer(
            params,
            optimizer_class=torch.optim.SGD,
            process_group=dp_group,
            **common_kwargs,
        )
    raise ValueError(f"Unsupported optimizer type for Zero1: {optimizer_config.type}")


def zero1_clip_grad_norm(
    parameters: list[nn.Parameter],
    max_norm: float,
    dp_group: dist.ProcessGroup,
    eps: float = 1e-6,
) -> float:
    """Clip gradients by global norm across DP ranks."""
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return 0.0

    device = grads[0].device
    total_sq = torch.zeros((), device=device, dtype=torch.float32)
    for grad in grads:
        total_sq += grad.detach().float().pow(2).sum()

    total_norm = total_sq.sqrt()
    total_norm_value = float(total_norm)
    if not math.isfinite(total_norm_value):
        return total_norm_value

    clip_coef = (max_norm / (total_norm + eps)).clamp(max=1.0)
    for grad in grads:
        grad.mul_(clip_coef.to(device=grad.device, dtype=grad.dtype))
    return total_norm_value


def all_reduce_zero1_gradients(
    parameters: list[nn.Parameter],
    dp_group: dist.ProcessGroup,
) -> None:
    """Synchronize gradients across DP ranks for Zero-1 training."""
    for parameter in parameters:
        if parameter.grad is None:
            continue
        dist.all_reduce(parameter.grad, group=dp_group)
