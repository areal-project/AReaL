# SPDX-License-Identifier: Apache-2.0

"""Generalized advantage estimation kernels and turn metadata helpers."""

import torch

from areal.trainer.ppo.lambda_fn import GAELambdaContext


def _compute_token_level_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    loss_mask: torch.Tensor,
    seq_no_eos_mask: torch.Tensor,
    discount: float,
    gae_lambda: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE with each generated token treated as one timestep."""
    bs, max_seqlen = rewards.shape
    advantages_reversed = [torch.zeros(bs, dtype=torch.float32, device=values.device)]
    lastgaelam = torch.zeros(bs, dtype=torch.float32, device=values.device)
    nextvalues = values[:, max_seqlen - 1] * seq_no_eos_mask
    discounted_lambda = discount * gae_lambda
    for t in reversed(range(max_seqlen - 1)):
        delta = rewards[:, t] + discount * nextvalues - values[:, t]
        newgaelam = delta + discounted_lambda * lastgaelam

        # Skip tokens that do not contribute to the loss.
        mask = loss_mask[:, t]
        inverse_mask = 1 - mask
        nextvalues = nextvalues * inverse_mask + values[:, t] * mask
        lastgaelam = lastgaelam * inverse_mask + newgaelam * mask
        advantages_reversed.append(lastgaelam)

    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    return advantages, advantages + values


def _validate_turn_ids(loss_mask: torch.Tensor, turn_ids: torch.Tensor) -> None:
    """Validate token-aligned turn IDs without synchronizing GPU hot paths."""
    if turn_ids.shape != loss_mask.shape:
        raise ValueError(
            f"turn_ids must have shape {loss_mask.shape}, got {turn_ids.shape}"
        )
    if turn_ids.device != loss_mask.device:
        raise ValueError(
            "turn_ids and loss_mask must be on the same device, got "
            f"{turn_ids.device} and {loss_mask.device}"
        )
    integral_dtypes = {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
    if turn_ids.dtype not in integral_dtypes:
        raise ValueError(f"turn_ids must use an integer dtype, got {turn_ids.dtype}")

    valid_token_mask = loss_mask.bool()
    torch._assert_async(
        torch.all(~valid_token_mask | (turn_ids >= 0)),
        "turn_ids must be non-negative at every active loss token",
    )
    torch._assert_async(
        torch.all(~valid_token_mask | (turn_ids < loss_mask.shape[1])),
        "turn_ids at active loss tokens must be smaller than the sequence length",
    )

    # Prompt/tool gaps use -1 and do not reset ordering. Comparing each active
    # ID with the cumulative maximum detects a turn that moves backwards.
    active_turn_ids = torch.where(
        valid_token_mask, turn_ids, torch.full_like(turn_ids, -1)
    )
    running_max = torch.cummax(active_turn_ids, dim=1).values
    torch._assert_async(
        torch.all(~valid_token_mask | (turn_ids >= running_max)),
        "turn_ids at active loss tokens must be temporally nondecreasing",
    )


def _compute_turn_level_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    loss_mask: torch.Tensor,
    turn_ids: torch.Tensor,
    seq_no_eos_mask: torch.Tensor,
    discount: float,
    gae_lambda: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE with each generated turn treated as one timestep."""
    _validate_turn_ids(loss_mask, turn_ids)
    bs, max_seqlen = rewards.shape
    valid_token_mask = loss_mask.bool()
    safe_turn_ids = torch.clamp(turn_ids.long(), min=0, max=max_seqlen - 1)

    # Aggregate all reward increments belonging to the same generated turn.
    turn_rewards = torch.zeros_like(rewards)
    turn_rewards.scatter_add_(
        dim=1,
        index=safe_turn_ids,
        src=rewards * valid_token_mask.to(rewards.dtype),
    )

    # Find the first valid action-token position for every turn without a
    # Python loop over the (potentially very long) token sequence.
    token_positions = torch.arange(max_seqlen, device=turn_ids.device).expand(bs, -1)
    first_turn_positions = torch.full_like(safe_turn_ids, max_seqlen)
    first_turn_positions.scatter_reduce_(
        dim=1,
        index=safe_turn_ids,
        src=torch.where(
            valid_token_mask,
            token_positions,
            torch.full_like(token_positions, max_seqlen),
        ),
        reduce="amin",
        include_self=True,
    )
    valid_turn_mask = first_turn_positions < max_seqlen

    # A turn's state value is the value at its first valid action token.
    turn_values = values.gather(
        dim=1, index=first_turn_positions.clamp(max=max_seqlen - 1)
    )
    turn_values = turn_values * valid_turn_mask.to(values.dtype)

    turn_advantages = torch.zeros(
        bs, max_seqlen, dtype=torch.float32, device=values.device
    )
    lastgaelam = torch.zeros(bs, dtype=torch.float32, device=values.device)
    zero_advantages = torch.zeros_like(lastgaelam)
    nextvalues = values[:, max_seqlen - 1] * seq_no_eos_mask
    discounted_lambda = discount * gae_lambda
    # Advantage calculation normally runs over CPU rollout tensors. Avoid
    # iterating over every token slot there when trajectories contain only a
    # handful of turns. On accelerator inputs, retain a static loop bound to
    # avoid a GPU-to-CPU synchronization in this hot path.
    num_turn_slots = max_seqlen
    if turn_ids.device.type == "cpu":
        max_active_turn = torch.where(
            valid_token_mask, safe_turn_ids, torch.full_like(safe_turn_ids, -1)
        ).amax()
        num_turn_slots = int(max_active_turn.item()) + 1

    for turn_idx in reversed(range(num_turn_slots)):
        delta = (
            turn_rewards[:, turn_idx] + discount * nextvalues - turn_values[:, turn_idx]
        )
        newgaelam = delta + discounted_lambda * lastgaelam

        mask = valid_turn_mask[:, turn_idx]
        turn_advantages[:, turn_idx] = torch.where(mask, newgaelam, zero_advantages)
        nextvalues = torch.where(mask, turn_values[:, turn_idx], nextvalues)
        lastgaelam = torch.where(mask, newgaelam, lastgaelam)

    turn_returns = turn_advantages + turn_values
    advantages = torch.gather(turn_advantages, dim=1, index=safe_turn_ids)
    returns = torch.gather(turn_returns, dim=1, index=safe_turn_ids)

    advantages = advantages * loss_mask
    returns = returns * loss_mask + values * (1 - loss_mask)
    return advantages, returns


def _build_gae_lambda_context(
    loss_mask: torch.Tensor,
    turn_ids: torch.Tensor | None,
    *,
    gae_timestep_unit: str,
) -> GAELambdaContext:
    """Build stable per-sample lengths from the canonical GAE eligibility mask.

    Token-level custom lambda functions can operate on legacy rollout data that does
    not carry turn metadata. In that case, ``turn_counts`` is zero-filled because it
    is not the selected timestep length. Turn-level GAE still requires explicit
    ``turn_ids``.
    """
    valid_token_mask = loss_mask.bool()
    effective_token_lengths = valid_token_mask.sum(dim=1)

    if turn_ids is not None:
        _validate_turn_ids(loss_mask, turn_ids)
        max_seqlen = loss_mask.shape[1]
        safe_turn_ids = torch.clamp(turn_ids.long(), min=0, max=max_seqlen - 1)
        tokens_per_turn = torch.zeros_like(safe_turn_ids)
        tokens_per_turn.scatter_add_(
            dim=1, index=safe_turn_ids, src=valid_token_mask.long()
        )
        turn_counts = (tokens_per_turn > 0).sum(dim=1)
    else:
        if gae_timestep_unit == "turn":
            raise ValueError(
                "Turn-level GAE requires rollout data to include 'turn_ids'."
            )
        turn_counts = torch.zeros_like(effective_token_lengths)

    timestep_lengths = (
        effective_token_lengths if gae_timestep_unit == "token" else turn_counts
    )
    return {
        "effective_token_lengths": effective_token_lengths,
        "turn_counts": turn_counts,
        "timestep_lengths": timestep_lengths,
    }
