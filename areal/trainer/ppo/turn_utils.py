# SPDX-License-Identifier: Apache-2.0

"""Canonical turn operations shared by advantage and policy reduction code."""

from __future__ import annotations

import torch

_INTEGRAL_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def validate_turn_ids(loss_mask: torch.Tensor, turn_ids: torch.Tensor) -> None:
    """Validate nondecreasing token-aligned turn IDs."""
    if loss_mask.ndim != 2:
        raise ValueError("turn-aware algorithms require a 2-D loss_mask")
    if turn_ids.shape != loss_mask.shape:
        raise ValueError(
            f"turn_ids must have shape {loss_mask.shape}, got {turn_ids.shape}"
        )
    if turn_ids.device != loss_mask.device:
        raise ValueError("turn_ids and loss_mask must share a device")
    if turn_ids.dtype not in _INTEGRAL_DTYPES:
        raise TypeError(f"turn_ids must use an integer dtype, got {turn_ids.dtype}")

    valid = loss_mask.bool()
    torch._assert_async(
        torch.all(~valid | (turn_ids >= 0)),
        "active loss tokens must have non-negative turn_ids",
    )
    torch._assert_async(
        torch.all(~valid | (turn_ids < loss_mask.shape[1])),
        "active turn_ids must be smaller than the sequence length",
    )
    signed_ids = turn_ids.long()
    active_ids = torch.where(valid, signed_ids, torch.full_like(signed_ids, -1))
    running_max = torch.cummax(active_ids, dim=1).values
    torch._assert_async(
        torch.all(~valid | (signed_ids >= running_max)),
        "active turn_ids must be temporally nondecreasing",
    )


def count_active_turns(
    loss_mask: torch.Tensor,
    turn_ids: torch.Tensor,
    *,
    require_nonempty: bool = True,
) -> torch.Tensor:
    """Count distinct active turns without allocating a dense float table."""
    validate_turn_ids(loss_mask, turn_ids)
    valid = loss_mask.bool()
    signed_ids = turn_ids.long()
    active_ids = torch.where(valid, signed_ids, torch.full_like(signed_ids, -1))
    previous_id = torch.cummax(active_ids, dim=1).values
    previous_id = torch.roll(previous_id, shifts=1, dims=1)
    previous_id[:, 0] = -1
    starts = valid & (signed_ids != previous_id)
    counts = starts.sum(dim=1)
    if require_nonempty:
        torch._assert_async(
            torch.all(counts > 0),
            "every trajectory must contain at least one generated turn",
        )
    return counts


def inverse_turn_lengths(
    loss_mask: torch.Tensor,
    turn_ids: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return 1/L at each supported token, zero elsewhere; each turn sums to one.

    L counts this mask's supported tokens in the same (sequence, turn). Whole
    sequences must be preserved when splitting microbatches. Global turn counts
    belong to the reducer, not this per-turn operation.
    """
    if turn_ids.dtype not in _INTEGRAL_DTYPES:
        raise TypeError("turn_ids must use an integer dtype")
    if turn_ids.device != loss_mask.device:
        raise ValueError("turn_ids and loss_mask must share a device")
    if loss_mask.ndim == 1:
        if turn_ids.shape != loss_mask.shape:
            raise ValueError("packed turn_ids must match loss_mask shape")
        if cu_seqlens is None:
            raise ValueError("packed turn_mean reduction requires cu_seqlens")
        if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
            raise ValueError("cu_seqlens must contain sequence boundaries")
        if loss_mask.numel() == 0:
            return torch.zeros_like(loss_mask, dtype=torch.float32)
        sequence_lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(
            device=loss_mask.device, dtype=torch.long
        )
        n_sequences = sequence_lengths.numel()
        sequence_ids = torch.repeat_interleave(
            torch.arange(n_sequences, device=loss_mask.device),
            sequence_lengths,
            output_size=loss_mask.numel(),
        )
        valid = loss_mask.bool()
        torch._assert_async(
            torch.all(~valid | (turn_ids >= 0)),
            "active packed tokens must have non-negative turn_ids",
        )
        # Canonical IDs are ordered within each sequence. Compact their starts
        # with fixed-size scans, avoiding CUDA synchronization from unique or
        # boolean-indexed, dynamically sized outputs.
        turn_id_stride = turn_ids.long().clamp_min(0).amax() + 1
        composite_ids = sequence_ids * turn_id_stride + turn_ids.long()
        active_ids = torch.where(valid, composite_ids, -1)
        previous = torch.cummax(active_ids, dim=0).values
        torch._assert_async(
            torch.all(~valid | (composite_ids >= previous)),
            "active packed turn_ids must be temporally nondecreasing",
        )
        previous = torch.roll(previous, shifts=1)
        previous[0] = -1
        starts = valid & (composite_ids != previous)
        slots = (starts.long().cumsum(0) - 1).clamp_min(0)
        counts = torch.zeros_like(loss_mask, dtype=torch.float32)
        counts.scatter_add_(0, slots, valid.float())
        return valid.float() / counts[slots].clamp_min(1)

    validate_turn_ids(loss_mask, turn_ids)
    valid = loss_mask.bool()
    safe_ids = turn_ids.long().clamp(min=0, max=loss_mask.shape[1] - 1)
    tokens_per_turn = torch.zeros_like(loss_mask, dtype=torch.float32)
    tokens_per_turn.scatter_add_(1, safe_ids, valid.to(torch.float32))
    token_turn_sizes = tokens_per_turn.gather(1, safe_ids).clamp_min(1.0)
    return valid.to(torch.float32) / token_turn_sizes
