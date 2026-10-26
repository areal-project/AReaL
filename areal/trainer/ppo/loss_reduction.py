# SPDX-License-Identifier: Apache-2.0

"""Prepare full-unit coefficients for the policy-gradient reduction interface."""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.distributed as dist

from areal.utils.data import get_batch_size
from areal.utils.functional.loss_aggregation import (
    LossAggregationMode,
    PolicyGradientReduction,
)


@dataclass(frozen=True, slots=True)
class _PreparedLossBatch:
    reduction: PolicyGradientReduction
    local_active_groups: torch.Tensor | None = None

    def for_steps(
        self,
        microbatches: Sequence[dict[str, Any]],
        *,
        dp_group: dist.ProcessGroup | None = None,
        device: torch.device | str | None = None,
    ) -> list[PolicyGradientReduction]:
        """Fix K/G and each microbatch's physical-response share after scheduling.

        The scheduler owns identical step counts across ranks. Reduce exact group
        and response counts over DP, excluding CP replicas, then copy the small
        counts to the host once so binding needs no device synchronization.
        """
        if not microbatches:
            raise ValueError("Cannot prepare an empty optimizer schedule.")
        if self.local_active_groups is None:
            return [self.reduction] * len(microbatches)
        collective_device = (
            device if device is not None else self.local_active_groups.device
        )
        if dist.is_initialized() and dist.get_backend(dp_group) == "gloo":
            collective_device = "cpu"
        row_counts = torch.tensor(
            [get_batch_size(mb) for mb in microbatches],
            dtype=torch.int64,
            device=collective_device,
        )
        counts = torch.cat(
            [
                self.local_active_groups.reshape(1).to(device=collective_device),
                row_counts,
            ]
        )
        if dist.is_initialized():
            dist.all_reduce(counts, group=dp_group)
        active_groups, *global_rows = counts.cpu().tolist()
        if active_groups == 0:
            raise ValueError(
                f"{self.reduction.mode} requires active units in the update."
            )
        if any(rows <= 0 for rows in global_rows):
            raise ValueError("Every optimizer step must contain real responses.")
        step_scale = len(microbatches) / active_groups
        return [
            replace(self.reduction, _step_scale=step_scale, _global_rows=rows)
            for rows in global_rows
        ]


def prepare_policy_gradient_batch(
    data: dict[str, Any],
    *,
    mode: LossAggregationMode,
    divisor: float | None = None,
    group_sizes: Sequence[int] | None = None,
) -> _PreparedLossBatch:
    """Prepare coefficients before splitting; keep the existing reduction API.

    Trajectory mean uses the same full-unit protocol as prompt mean, with one
    logical trajectory per group. Both modes retain their coefficients when a
    unit spans optimizer steps or forward microbatches.
    """
    reduction = PolicyGradientReduction(mode=mode, divisor=divisor)
    if mode not in ("traj-mean", "prompt-mean"):
        return _PreparedLossBatch(reduction)
    mask = data["loss_mask"].bool()
    if mask.ndim != 2:
        raise ValueError(f"Preparing {mode} requires a 2D loss_mask.")
    if group_sizes is None:
        raise ValueError(f"group_sizes are required to prepare {mode}.")
    if torch.is_tensor(group_sizes):
        raise TypeError("group_sizes must be a sequence of ints, not a tensor.")
    sizes = [int(size) for size in group_sizes]
    if any(size <= 0 for size in sizes):
        raise ValueError(f"group_sizes must be positive, got {sizes}.")
    if sum(sizes) != mask.shape[0]:
        raise ValueError(
            f"group_sizes sum to {sum(sizes)} but sequence count is {mask.shape[0]}."
        )
    ids = torch.arange(len(sizes), device=mask.device).repeat_interleave(
        torch.tensor(sizes, dtype=torch.long, device=mask.device),
        output_size=mask.shape[0],
    )
    counts = torch.zeros(len(sizes), dtype=torch.int64, device=mask.device)
    counts.scatter_add_(0, ids, mask.sum(dim=-1, dtype=torch.int64))
    data["unit_weights"] = mask.float() / counts[ids].unsqueeze(-1).clamp_min(1)
    return _PreparedLossBatch(reduction, counts.count_nonzero())
