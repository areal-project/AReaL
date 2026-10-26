# SPDX-License-Identifier: Apache-2.0

"""Policy-gradient loss aggregation and distributed normalizer contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

LossAggregationMode = Literal[
    "token-mean", "seq-mean", "traj-mean", "prompt-mean", "constant"
]
_LOSS_AGGREGATIONS = (
    "token-mean",
    "seq-mean",
    "traj-mean",
    "prompt-mean",
    "constant",
)
# Reductions carrying their denominator as a precomputed per-token column.
_WEIGHTED_AGGREGATIONS = ("traj-mean", "prompt-mean")


@dataclass(frozen=True, slots=True)
class PolicyGradientReduction:
    """Policy-gradient microbatch mean and its matching engine weight.

    The training engine combines microbatches as
    ``sum(local_mean * local_weight) / sum(local_weight)``.

    ``unit_weights`` holds full-trajectory or full-prompt token coefficients.
    Preparation fixes the update's K/U scale before row-level optimizer and
    forward splitting. Private step constants use physical response shares as
    engine weights, so a logical unit may span any number of minibatches.
    Without step preparation, the same interface reduces one complete step.
    """

    mode: LossAggregationMode = "token-mean"
    divisor: float | None = None
    _step_scale: float = field(default=1.0, repr=False)
    _global_rows: int | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.mode not in _LOSS_AGGREGATIONS:
            raise ValueError(
                f"loss_aggregation must be one of {_LOSS_AGGREGATIONS}, "
                f"got {self.mode!r}."
            )
        if self.mode == "constant":
            if (
                self.divisor is None
                or not math.isfinite(self.divisor)
                or self.divisor <= 0
            ):
                raise ValueError(
                    "divisor must be a positive finite value for "
                    "loss_aggregation='constant'."
                )
        elif self.divisor is not None:
            raise ValueError("divisor is only valid for loss_aggregation='constant'.")

    def normalizer_fn(self, data: dict[str, Any]) -> torch.Tensor:
        """Return the original microbatch's matching engine weight."""
        loss_mask = data["loss_mask"].bool()
        if self.mode == "token-mean":
            return loss_mask.count_nonzero()
        if self.mode in _WEIGHTED_AGGREGATIONS:
            return self._engine_weight(
                loss_mask, data.get("cu_seqlens"), data.get("unit_weights")
            )
        return _active_sequences(loss_mask, data.get("cu_seqlens"))

    def aggregate(
        self,
        loss: torch.Tensor,
        loss_mask: torch.Tensor,
        *,
        denominator_mask: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        unit_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Aggregate token losses without changing the original denominator."""
        numerator_mask, denominator_mask = _resolve_masks(
            loss, loss_mask, loss_mask if denominator_mask is None else denominator_mask
        )
        if self.mode == "token-mean":
            numerator = torch.where(numerator_mask, loss, 0).sum()
            return numerator / denominator_mask.count_nonzero().clamp_min(1)
        if self.mode in _WEIGHTED_AGGREGATIONS:
            weights = self._require_unit_weights(loss, unit_weights)
            numerator = (torch.where(numerator_mask, loss, 0).float() * weights).sum()
            engine_weight = self._engine_weight(denominator_mask, cu_seqlens, weights)
            return (
                numerator
                * self._step_scale
                / engine_weight.clamp_min(torch.finfo(torch.float32).tiny)
            )
        if self.mode == "constant":
            numerator = torch.where(numerator_mask, loss, 0).to(torch.float32).sum()
            active_sequences = _active_sequences(denominator_mask, cu_seqlens)
            return numerator / (active_sequences.clamp_min(1) * self.divisor)
        numerators, denominators = _sequence_loss_sums(
            loss, numerator_mask, denominator_mask, cu_seqlens
        )
        return _reduce_unit_means(numerators, denominators)

    def _engine_weight(
        self,
        loss_mask: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
        unit_weights: torch.Tensor | None,
    ) -> torch.Tensor:
        if self._global_rows is None:
            return self._unit_weight_sum(unit_weights)
        self._require_sequence_boundaries(loss_mask, cu_seqlens)
        rows = cu_seqlens.numel() - 1 if cu_seqlens is not None else loss_mask.shape[0]
        return torch.scalar_tensor(
            rows / self._global_rows, dtype=torch.float32, device=loss_mask.device
        )

    @staticmethod
    def _unit_weight_sum(unit_weights: torch.Tensor | None) -> torch.Tensor:
        if unit_weights is None:
            raise ValueError(
                "unit_weights are required for loss_aggregation in "
                f"{_WEIGHTED_AGGREGATIONS}."
            )
        return unit_weights.to(torch.float32).sum()

    @staticmethod
    def _require_unit_weights(
        loss: torch.Tensor, unit_weights: torch.Tensor | None
    ) -> torch.Tensor:
        if unit_weights is None:
            raise ValueError(
                "unit_weights are required for loss_aggregation in "
                f"{_WEIGHTED_AGGREGATIONS}."
            )
        if unit_weights.shape != loss.shape:
            raise ValueError(
                f"unit_weights shape {tuple(unit_weights.shape)} must match "
                f"loss shape {tuple(loss.shape)}."
            )
        return unit_weights.to(torch.float32)

    def _require_sequence_boundaries(
        self, loss_mask: torch.Tensor, cu_seqlens: torch.Tensor | None
    ) -> None:
        if cu_seqlens is None and loss_mask.ndim == 1:
            raise ValueError(
                f"loss_aggregation='{self.mode}' requires cu_seqlens for packed "
                "inputs; tree-packed training currently supports only token-mean."
            )


def _resolve_masks(
    loss: torch.Tensor,
    loss_mask: torch.Tensor,
    denominator_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if loss.shape != loss_mask.shape:
        raise ValueError(
            f"loss_mask shape {tuple(loss_mask.shape)} must match "
            f"loss shape {tuple(loss.shape)}."
        )
    if loss.shape != denominator_mask.shape:
        raise ValueError(
            f"denom_mask shape {tuple(denominator_mask.shape)} must match "
            f"loss shape {tuple(loss.shape)}."
        )
    return loss_mask.bool(), denominator_mask.bool()


def _sequence_sums(
    values: torch.Tensor, cu_seqlens: torch.Tensor | None
) -> torch.Tensor:
    """Sum token values per sequence for padded or packed inputs."""
    if cu_seqlens is None:
        if values.ndim == 1:
            raise ValueError(
                "Sequence-based loss aggregation requires cu_seqlens for packed "
                "inputs; tree-packed training currently supports only token_mean."
            )
        if values.ndim != 2:
            raise ValueError(
                "padded policy-gradient inputs must be 2D, "
                f"got shape {tuple(values.shape)}."
            )
        return values.sum(dim=-1)

    if values.ndim != 1:
        raise ValueError(
            "packed policy-gradient inputs must be 1D, "
            f"got shape {tuple(values.shape)}."
        )
    if cu_seqlens.ndim != 1:
        raise ValueError(f"cu_seqlens must be 1D, got shape {tuple(cu_seqlens.shape)}.")

    n_sequences = cu_seqlens.numel() - 1
    sequence_lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(
        device=values.device, dtype=torch.long
    )
    sequence_ids = torch.arange(n_sequences, device=values.device).repeat_interleave(
        sequence_lengths, output_size=values.numel()
    )
    result = torch.zeros(n_sequences, dtype=values.dtype, device=values.device)
    return result.scatter_add_(0, sequence_ids, values)


def _active_sequences(
    loss_mask: torch.Tensor, cu_seqlens: torch.Tensor | None
) -> torch.Tensor:
    return (
        _sequence_sums(loss_mask.bool().to(torch.float32), cu_seqlens)
        .count_nonzero()
        .to(torch.float32)
    )


def _sequence_loss_sums(
    loss: torch.Tensor,
    loss_mask: torch.Tensor,
    denominator_mask: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    numerator_mask, denominator_mask = _resolve_masks(loss, loss_mask, denominator_mask)
    masked_loss = torch.where(numerator_mask, loss, 0).to(torch.float32)
    return (
        _sequence_sums(masked_loss, cu_seqlens),
        _sequence_sums(denominator_mask.to(torch.float32), cu_seqlens),
    )


def _reduce_unit_means(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
) -> torch.Tensor:
    active = denominator > 0
    unit_means = torch.where(
        active,
        numerator / denominator.clamp_min(1),
        torch.zeros_like(numerator),
    )
    return unit_means.sum() / active.count_nonzero().clamp_min(1)
