# SPDX-License-Identifier: Apache-2.0

"""Prepare batch metadata and bind pure policy-gradient reductions."""

import functools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch
import torch.distributed as dist

from areal.utils.data import TRANSPORT_DUMMY_KEY, get_batch_size
from areal.utils.functional.loss_aggregation import (
    ConstantLength,
    GroupSizes,
    LossAggregationMode,
    PolicyGradientReduction,
    PromptMean,
    SequenceMean,
    TokenMean,
)

PG_TOKEN_WEIGHTS = "_pg_token_weights"
_MODES = ("token_mean", "seq_mean", "prompt_mean", "constant")


@dataclass(frozen=True, slots=True)
class PreparedLossStep:
    """Bind loss-side batches to one optimizer step's reduction contract.

    The callable retains only configuration and small step constants. Original
    masks and prepared coefficients come from the actual microbatch, including
    its device and packed layout, when the engine invokes either callback.
    """

    bind: Callable[[dict[str, Any]], PolicyGradientReduction]

    def loss_weight(self, data: dict[str, Any]) -> torch.Tensor:
        return self.bind(data).normalizer()


class PreparedLossBatch(Protocol):
    def for_steps(
        self,
        microbatches: Sequence[dict[str, Any]],
        *,
        dp_group: dist.ProcessGroup | None = None,
        device: torch.device | str | None = None,
    ) -> list[PreparedLossStep]: ...


@dataclass(frozen=True, slots=True)
class _IndependentSteps:
    step: PreparedLossStep

    def for_steps(
        self,
        microbatches: Sequence[dict[str, Any]],
        *,
        dp_group: dist.ProcessGroup | None = None,
        device: torch.device | str | None = None,
    ) -> list[PreparedLossStep]:
        if not microbatches:
            raise ValueError("Cannot prepare an empty optimizer schedule.")
        return [self.step] * len(microbatches)


@dataclass(frozen=True, slots=True)
class _PromptMeanBatch:
    local_active_groups: torch.Tensor

    def for_steps(
        self,
        microbatches: Sequence[dict[str, Any]],
        *,
        dp_group: dist.ProcessGroup | None = None,
        device: torch.device | str | None = None,
    ) -> list[PreparedLossStep]:
        """Fix K/G and each microbatch's physical-response share after scheduling.

        The scheduler owns identical step counts across ranks. Reduce exact group
        and response counts over DP, excluding CP replicas, then copy the small
        counts to the host once so binding needs no device synchronization.
        """
        if not microbatches:
            raise ValueError("Cannot prepare an empty optimizer schedule.")
        collective_device = (
            device if device is not None else self.local_active_groups.device
        )
        if dist.is_initialized() and dist.get_backend(dp_group) == "gloo":
            collective_device = "cpu"
        row_counts = torch.tensor(
            [
                0 if mb.get(TRANSPORT_DUMMY_KEY) is True else get_batch_size(mb)
                for mb in microbatches
            ],
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
            raise ValueError("Prompt mean requires active prompt groups in the update.")
        if any(rows <= 0 for rows in global_rows):
            raise ValueError("Every optimizer step must contain real responses.")
        step_scale = len(microbatches) / active_groups
        return [
            PreparedLossStep(
                functools.partial(
                    _bind_prompt_mean, step_scale=step_scale, global_rows=rows
                )
            )
            for rows in global_rows
        ]


def prepare_policy_gradient_batch(
    data: dict[str, Any],
    *,
    mode: LossAggregationMode,
    divisor: float | None = None,
    group_sizes: GroupSizes | None = None,
) -> PreparedLossBatch:
    """Prepare loss metadata before response-level splitting.

    Token coefficients travel with the batch through splitting and packing. The
    returned context retains only configuration or the active-group scalar;
    optimizer-step constants are resolved once the schedule is known.
    """
    if mode not in _MODES:
        raise ValueError(f"loss_aggregation must be one of {_MODES}, got {mode!r}.")
    if mode == "constant":
        if divisor is None or not math.isfinite(divisor) or divisor <= 0:
            raise ValueError("divisor must be a positive finite value.")
        return _IndependentSteps(
            PreparedLossStep(functools.partial(_bind_constant_length, divisor=divisor))
        )
    if divisor is not None:
        raise ValueError("divisor is only valid for loss_aggregation='constant'.")
    if mode == "token_mean":
        return _IndependentSteps(PreparedLossStep(_bind_token_mean))
    if mode == "seq_mean":
        return _IndependentSteps(PreparedLossStep(_bind_sequence_mean))
    mask = data["loss_mask"].bool()
    if mask.ndim != 2:
        raise ValueError("Preparing prompt mean requires a 2D loss_mask.")
    if group_sizes is None:
        raise ValueError("group_sizes are required to prepare prompt mean.")
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
    data[PG_TOKEN_WEIGHTS] = mask.float() / counts[ids].unsqueeze(-1).clamp_min(1)
    return _PromptMeanBatch(counts.count_nonzero())


def _bind_token_mean(data: dict[str, Any]) -> TokenMean:
    return TokenMean(data["loss_mask"])


def _bind_sequence_mean(data: dict[str, Any]) -> SequenceMean:
    return SequenceMean(data["loss_mask"], data.get("cu_seqlens"))


def _bind_constant_length(data: dict[str, Any], *, divisor: float) -> ConstantLength:
    return ConstantLength(divisor, data["loss_mask"], data.get("cu_seqlens"))


def _bind_prompt_mean(
    data: dict[str, Any], *, step_scale: float, global_rows: int
) -> PromptMean:
    weights = data.get(PG_TOKEN_WEIGHTS)
    if weights is None:
        raise ValueError("Prompt mean requires prepared token weights.")
    mask = data["loss_mask"]
    if weights.shape != mask.shape or weights.device != mask.device:
        raise ValueError(
            "Prepared token weights must match the loss mask shape/device."
        )
    rows = 0 if data.get(TRANSPORT_DUMMY_KEY) is True else get_batch_size(data)
    return PromptMean(weights, step_scale, rows / global_rows)
