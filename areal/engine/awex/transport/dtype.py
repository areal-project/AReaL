# SPDX-License-Identifier: Apache-2.0
"""Receiver-dtype alignment for AWEX's direct P2P send operations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from areal.infra.platforms import current_platform

if TYPE_CHECKING:
    from awex.transfer.transfer_plan import TransferPlan


@torch.no_grad()
def align_send_ops_to_recv_dtype(
    send_ops: Sequence[dist.P2POp], transfer_plan: TransferPlan
) -> None:
    """Apply the colocate transport's numeric cast to direct-send buffers.

    AWEX's send builder interleaves peers but preserves plan order within each
    peer. Copy-only operations are not in ``send_ops``, so match by peer rather
    than flattening the plan. Each P2POp retains its cast tensor; callers must
    keep these operations alive until communication completes.
    """
    peer_operations = {
        peer: iter(operations) for peer, operations in transfer_plan.operations.items()
    }
    converted = False
    for send_op in send_ops:
        plan_op = next(peer_operations[send_op.peer])
        recv_dtype = plan_op.recv_shard_meta.dtype
        if send_op.tensor.dtype != recv_dtype:
            send_op.tensor = send_op.tensor.to(dtype=recv_dtype)
            converted = True

    if converted:
        # AWEX 0.8.1's non-group path uses private peer streams without waiting
        # for the producer stream. Finish casts before it can read the buffers;
        # a synchronization after sending would be too late to prevent a race.
        current_platform.synchronize()
