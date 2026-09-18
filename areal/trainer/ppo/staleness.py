# SPDX-License-Identifier: Apache-2.0

"""Token-age filtering after advantage computation and before engine packing."""

from typing import Any

import torch
import torch.distributed as dist

from areal.infra.platforms import current_platform
from areal.utils import stats_tracker
from areal.utils.data import TRANSPORT_DUMMY_KEY


def apply_staleness_mask(
    data: dict[str, Any], *, current_version: int, max_staleness: int
) -> None:
    """Mask old targets without changing the rollout's reward/GAE structure.

    Called once on the transient optimizer batch. ``loss_mask`` already uses
    next-token prediction positions, while rollout ``versions`` still uses
    token positions. Align versions here, before packing or CP partitioning.
    The boundary is inclusive: age == max_staleness remains trainable.
    """
    if type(max_staleness) is not int or max_staleness < 0:
        raise ValueError("max_token_staleness must be a non-negative integer")
    versions = data.get("versions")
    if not isinstance(versions, torch.Tensor):
        raise ValueError("Token staleness masking requires rollout 'versions'")
    loss_mask = data["loss_mask"]
    if versions.shape != loss_mask.shape or versions.ndim != 2:
        raise ValueError(
            "versions and loss_mask must have the same [batch, sequence] shape"
        )
    if versions.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("Rollout versions must have an integer dtype")

    aligned_versions = torch.roll(
        versions.to(device=loss_mask.device, dtype=torch.int64), shifts=-1, dims=-1
    )
    aligned_versions[:, -1] = -1
    generated = loss_mask.bool()
    torch._assert_async(
        torch.all(
            ~generated
            | ((aligned_versions >= 0) & (aligned_versions <= current_version))
        ),
        "Trainable tokens must have known rollout versions no newer than the actor",
    )
    stale = generated & (current_version - aligned_versions > max_staleness)
    stats_tracker.denominator(staleness_candidate_tokens=generated)
    stats_tracker.stat(
        stale_token_fraction=stale.float(), denominator="staleness_candidate_tokens"
    )
    data["loss_mask"] = loss_mask.masked_fill(stale, 0)
    # Proximal-policy approximation and version metrics must use the same
    # prediction positions as the new mask.
    data["versions"] = aligned_versions


def has_global_trainable_tokens(
    data: dict[str, Any], group: dist.ProcessGroup | None
) -> bool:
    """Make a common optimizer-step decision, including ranks with zero targets.

    This scalar synchronization is intentional at the optimizer boundary:
    skipping on local emptiness would desynchronize distributed training, and
    running AdamW with zero loss can still change weights through decay/momentum.
    TP/PP/CP ranks call this before the engine partitions their replicated batch.
    """
    count = data["loss_mask"].count_nonzero()
    if data.get(TRANSPORT_DUMMY_KEY) is True:
        count = torch.zeros_like(count)
    if dist.is_initialized():
        if group is None:
            raise ValueError("Token staleness masking requires an explicit DP group")
        device = (
            "cpu"
            if dist.get_backend(group) == "gloo"
            else current_platform.current_device()
        )
        count = count.to(device=device)
        dist.all_reduce(count, op=dist.ReduceOp.SUM, group=group)
    return bool(count > 0)
