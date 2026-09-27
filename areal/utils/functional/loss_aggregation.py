# SPDX-License-Identifier: Apache-2.0

"""Bound policy-gradient reductions with matching engine weights."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import torch

LossAggregationMode = Literal["token_mean", "seq_mean", "prompt_mean", "constant"]
GroupSizes = Sequence[int]


class PolicyGradientReduction(Protocol):
    """A bound local loss paired with its engine weight.

    The engine combines microbatches as
    ``sum(local_loss * local_weight) / sum(local_weight)``. Original denominator
    data is captured before numerator filtering. Preparation owns batch metadata
    and optimizer-step normalization; reductions consume only tensors.
    """

    def normalizer(self) -> torch.Tensor:
        """Return this original microbatch's engine weight."""
        ...

    def aggregate(
        self, loss: torch.Tensor, numerator_mask: torch.Tensor
    ) -> torch.Tensor:
        """Reduce token losses after optional numerator filtering."""
        ...


@dataclass(frozen=True, slots=True)
class TokenMean:
    """Average over original valid tokens, retaining the input loss dtype."""

    denominator_mask: torch.Tensor

    def normalizer(self) -> torch.Tensor:
        return self.denominator_mask.count_nonzero()

    def aggregate(
        self, loss: torch.Tensor, numerator_mask: torch.Tensor
    ) -> torch.Tensor:
        numerator_mask, denominator_mask = _resolve_masks(
            loss, numerator_mask, self.denominator_mask
        )
        numerator = torch.where(numerator_mask, loss, 0).sum()
        return numerator / denominator_mask.count_nonzero().clamp_min(1)


@dataclass(frozen=True, slots=True)
class SequenceMean:
    """Average per-response token means; weight by active response count."""

    denominator_mask: torch.Tensor
    cu_seqlens: torch.Tensor | None = None

    def normalizer(self) -> torch.Tensor:
        return _active_sequences(self.denominator_mask, self.cu_seqlens)

    def aggregate(
        self, loss: torch.Tensor, numerator_mask: torch.Tensor
    ) -> torch.Tensor:
        numerators, denominators = _sequence_loss_sums(
            loss, numerator_mask, self.denominator_mask, self.cu_seqlens
        )
        return _reduce_unit_means(numerators, denominators)


@dataclass(frozen=True, slots=True)
class PromptMean:
    """A full-prompt contribution normalized for one response-level step.

    Preparation supplies original full-group token coefficients, ``step_scale``
    equal to realized steps / global active groups, and ``engine_weight`` equal
    to this microbatch's share of the step's real responses. The engine weight
    cancels the local division, preserving full-group coefficients even when
    a prompt spans optimizer steps. Real all-masked steps retain positive weight.
    """

    token_weights: torch.Tensor
    step_scale: float
    engine_weight: float

    def normalizer(self) -> torch.Tensor:
        return torch.scalar_tensor(
            self.engine_weight, dtype=torch.float32, device=self.token_weights.device
        )

    def aggregate(
        self, loss: torch.Tensor, numerator_mask: torch.Tensor
    ) -> torch.Tensor:
        if loss.shape != numerator_mask.shape or loss.shape != self.token_weights.shape:
            raise ValueError(
                "Loss, numerator mask and token weights must have matching shapes."
            )
        numerator = (
            torch.where(numerator_mask.bool(), loss, 0).float() * self.token_weights
        ).sum()
        return (
            numerator
            * self.step_scale
            / (self.engine_weight if self.engine_weight > 0 else 1.0)
        )


@dataclass(frozen=True, slots=True)
class ConstantLength:
    """Divide total token loss by active response count times a fixed length."""

    divisor: float
    denominator_mask: torch.Tensor
    cu_seqlens: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.divisor) or self.divisor <= 0:
            raise ValueError("divisor must be a positive finite value.")

    def normalizer(self) -> torch.Tensor:
        return _active_sequences(self.denominator_mask, self.cu_seqlens)

    def aggregate(
        self, loss: torch.Tensor, numerator_mask: torch.Tensor
    ) -> torch.Tensor:
        numerator_mask, denominator_mask = _resolve_masks(
            loss, numerator_mask, self.denominator_mask
        )
        numerator = torch.where(numerator_mask, loss, 0).to(torch.float32).sum()
        active_sequences = _sequence_sums(
            denominator_mask.to(torch.float32), self.cu_seqlens
        ).count_nonzero()
        return numerator / (active_sequences.clamp_min(1) * self.divisor)


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
