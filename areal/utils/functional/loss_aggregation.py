# SPDX-License-Identifier: Apache-2.0

"""Policy-gradient loss aggregation and distributed normalizer contracts."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import torch

LossAggregationMode = Literal["token_mean", "seq_mean", "prompt_mean", "constant"]
_LOSS_AGGREGATIONS = ("token_mean", "seq_mean", "prompt_mean", "constant")

GroupSizes = Sequence[int]


class PolicyGradientReduction(Protocol):
    """A microbatch mean paired with its engine weight.

    The engine combines microbatches as
    ``sum(local_mean * local_weight) / sum(local_weight)``. Both operations use
    the original denominator data, even when filtering narrows the numerator.
    Weights count averaging units or a fragment's fractional share of them;
    originally empty units contribute zero loss and zero weight.

    Sequences remain intact across microbatches. Prompt groups remain intact
    across optimizer steps; their precomputed token weights preserve prompt mean
    across microbatch splits. Pipeline setup owns compatibility with other
    objectives such as distillation.
    """

    def normalizer(
        self,
        loss_mask: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
        prompt_token_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the engine weight from the original mask or prepared weights."""
        ...

    def aggregate(
        self,
        loss: torch.Tensor,
        loss_mask: torch.Tensor,
        *,
        denominator_mask: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        prompt_token_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reduce token-shaped loss, retaining the supplied denominator mask."""
        ...


class TokenMean:
    """Average over valid tokens; weight microbatches by valid token count."""

    def normalizer(
        self,
        loss_mask: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
        prompt_token_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return loss_mask.count_nonzero()

    def aggregate(
        self,
        loss: torch.Tensor,
        loss_mask: torch.Tensor,
        *,
        denominator_mask: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        prompt_token_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        numerator_mask, denominator_mask = _resolve_masks(
            loss, loss_mask, denominator_mask
        )
        # Preserve the pre-feature token-mean dtype and reduction path.
        numerator = torch.where(numerator_mask, loss, 0).sum()
        return numerator / denominator_mask.count_nonzero().clamp_min(1)


class SequenceMean:
    """Average per-response token means; weight by active response count."""

    def normalizer(
        self,
        loss_mask: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
        prompt_token_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return _active_sequences(loss_mask, cu_seqlens)

    def aggregate(
        self,
        loss: torch.Tensor,
        loss_mask: torch.Tensor,
        *,
        denominator_mask: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        prompt_token_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        numerators, denominators = _sequence_loss_sums(
            loss, loss_mask, denominator_mask, cu_seqlens
        )
        return _reduce_unit_means(numerators, denominators)


class PromptMean:
    """Average full-prompt token means across freely split microbatches.

    ``prompt_token_weights`` comes from :func:`prepare_prompt_token_weights`
    before splitting. Its sum is this fragment's share of active prompt groups.
    Filtering narrows the numerator without changing these original weights.
    """

    def normalizer(
        self,
        loss_mask: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
        prompt_token_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return _require_prompt_token_weights(loss_mask, prompt_token_weights).sum()

    def aggregate(
        self,
        loss: torch.Tensor,
        loss_mask: torch.Tensor,
        *,
        denominator_mask: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        prompt_token_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        numerator_mask, _ = _resolve_masks(loss, loss_mask, denominator_mask)
        weights = _require_prompt_token_weights(loss_mask, prompt_token_weights)
        numerator = (torch.where(numerator_mask, loss, 0).float() * weights).sum()
        weight = weights.sum()
        return numerator / torch.where(weight > 0, weight, 1.0)


@dataclass(frozen=True, slots=True)
class ConstantLength:
    """Divide total token loss by active response count times a fixed length.

    The engine weight is the active response count; the fixed divisor belongs
    only to the local loss, so it is not cancelled by engine normalization.
    """

    divisor: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.divisor) or self.divisor <= 0:
            raise ValueError("divisor must be a positive finite value.")

    def normalizer(
        self,
        loss_mask: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
        prompt_token_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return _active_sequences(loss_mask, cu_seqlens)

    def aggregate(
        self,
        loss: torch.Tensor,
        loss_mask: torch.Tensor,
        *,
        denominator_mask: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        prompt_token_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        numerator_mask, denominator_mask = _resolve_masks(
            loss, loss_mask, denominator_mask
        )
        numerator = torch.where(numerator_mask, loss, 0).to(torch.float32).sum()
        active_sequences = _sequence_sums(
            denominator_mask.to(torch.float32), cu_seqlens
        ).count_nonzero()
        return numerator / (active_sequences.clamp_min(1) * self.divisor)


def make_policy_gradient_reduction(
    mode: LossAggregationMode = "token_mean", divisor: float | None = None
) -> PolicyGradientReduction:
    """Resolve actor configuration once, before constructing engine callbacks."""
    if mode not in _LOSS_AGGREGATIONS:
        raise ValueError(
            f"loss_aggregation must be one of {_LOSS_AGGREGATIONS}, got {mode!r}."
        )
    if mode == "constant":
        if divisor is None:
            raise ValueError("divisor is required for loss_aggregation='constant'.")
        return ConstantLength(divisor)
    if divisor is not None:
        raise ValueError("divisor is only valid for loss_aggregation='constant'.")
    if mode == "token_mean":
        return TokenMean()
    if mode == "seq_mean":
        return SequenceMean()
    return PromptMean()


def prepare_prompt_token_weights(
    loss_mask: torch.Tensor, group_sizes: GroupSizes
) -> torch.Tensor:
    """Derive original token weights from complete physical prompt groups.

    The padded mask must precede M2/rejection filtering. Each valid token gets
    the reciprocal of its full prompt group's valid-token count, so every
    active group has total weight one and empty groups have total weight zero.
    Carry this tensor through optimizer scheduling, packing and microbatching;
    never recompute it from a prompt fragment.
    """
    if loss_mask.ndim != 2:
        raise ValueError("Preparing prompt token weights requires a 2D loss_mask.")
    mask = loss_mask.bool().float()
    ids, n_groups = _prompt_ids(mask.shape[0], group_sizes, mask.device)
    denominators = _prompt_sums(mask.sum(dim=-1), ids, n_groups)
    return mask / denominators[ids].unsqueeze(-1).clamp_min(1)


def _require_prompt_token_weights(
    loss_mask: torch.Tensor, prompt_token_weights: torch.Tensor | None
) -> torch.Tensor:
    if prompt_token_weights is None:
        raise ValueError("prompt_token_weights are required for prompt_mean.")
    if prompt_token_weights.shape != loss_mask.shape:
        raise ValueError("prompt_token_weights shape must match loss_mask shape.")
    return prompt_token_weights.float()


def _resolve_masks(
    loss: torch.Tensor,
    loss_mask: torch.Tensor,
    denominator_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if loss.shape != loss_mask.shape:
        raise ValueError(
            f"loss_mask shape {tuple(loss_mask.shape)} must match "
            f"loss shape {tuple(loss.shape)}."
        )
    if denominator_mask is not None and loss.shape != denominator_mask.shape:
        raise ValueError(
            f"denom_mask shape {tuple(denominator_mask.shape)} must match "
            f"loss shape {tuple(loss.shape)}."
        )
    numerator_mask = loss_mask.bool()
    return numerator_mask, (
        numerator_mask if denominator_mask is None else denominator_mask.bool()
    )


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
    denominator_mask: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    numerator_mask, denominator_mask = _resolve_masks(loss, loss_mask, denominator_mask)
    masked_loss = torch.where(numerator_mask, loss, 0).to(torch.float32)
    return (
        _sequence_sums(masked_loss, cu_seqlens),
        _sequence_sums(denominator_mask.to(torch.float32), cu_seqlens),
    )


def _prompt_ids(
    n_sequences: int, group_sizes: GroupSizes | None, device: torch.device
) -> tuple[torch.Tensor, int]:
    if group_sizes is None:
        raise ValueError("group_sizes are required for loss_aggregation='prompt_mean'.")
    if torch.is_tensor(group_sizes):
        raise TypeError(
            "group_sizes must be a sequence of ints, not a tensor; "
            "passing a GPU tensor would synchronize every microbatch."
        )
    sizes = [int(size) for size in group_sizes]
    if any(size <= 0 for size in sizes):
        raise ValueError(f"group_sizes must be positive, got {sizes}.")
    if sum(sizes) != n_sequences:
        raise ValueError(
            f"group_sizes sum to {sum(sizes)} but sequence count is {n_sequences}."
        )
    return (
        torch.arange(len(sizes), device=device).repeat_interleave(
            torch.tensor(sizes, device=device), output_size=n_sequences
        ),
        len(sizes),
    )


def _prompt_sums(
    values: torch.Tensor, ids: torch.Tensor, n_groups: int
) -> torch.Tensor:
    result = torch.zeros(n_groups, dtype=values.dtype, device=values.device)
    return result.scatter_add_(0, ids, values)


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
