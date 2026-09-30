# SPDX-License-Identifier: Apache-2.0

"""Batch-boundary coordination for optional stale-token filtering."""

import torch
import torch.distributed as dist


def has_global_loss_tokens(
    loss_mask: torch.Tensor, cpu_group: dist.ProcessGroup
) -> bool:
    """Keep locally empty ranks participating if a peer has trainable actions.

    Rollout batches normally arrive on CPU. This control decision runs once at
    the optimizer minibatch boundary, before model forward/backward, and uses
    the explicit CPU group so all model/data parallel ranks take the same path.
    """
    present = loss_mask.detach().any().to(device="cpu", dtype=torch.int32)
    if dist.is_initialized():
        dist.all_reduce(present, op=dist.ReduceOp.MAX, group=cpu_group)
    return bool(present)
