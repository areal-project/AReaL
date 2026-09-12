# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch

from areal.utils import stats_tracker


def _get_rewards(sample: dict[str, Any]) -> torch.Tensor:
    rewards = sample.get("original_rewards", sample.get("rewards"))
    if rewards is None:
        raise ValueError("dynamic filtering requires rewards or original_rewards")
    if not isinstance(rewards, torch.Tensor):
        rewards = torch.as_tensor(rewards)
    rewards = rewards.reshape(-1)
    if rewards.numel() == 0:
        raise ValueError("dynamic filtering requires at least one reward")
    if not torch.isfinite(rewards).all().item():
        raise ValueError("dynamic filtering rewards must be finite")
    return rewards


def _has_negative_process_reward(
    sample: dict[str, Any], *, ignore_no_eos: bool
) -> bool:
    token_rewards = sample.get("token_rewards")
    loss_mask = sample.get("loss_mask")
    if token_rewards is None or loss_mask is None:
        return False
    if not isinstance(token_rewards, torch.Tensor) or not isinstance(
        loss_mask, torch.Tensor
    ):
        raise ValueError("token_rewards and loss_mask must be local tensors")
    if token_rewards.shape != loss_mask.shape or token_rewards.ndim != 2:
        raise ValueError(
            "token_rewards and loss_mask must have matching [group, seqlen] shapes"
        )
    if not torch.isfinite(token_rewards).all().item():
        raise ValueError("token_rewards must be finite")

    valid_mask = loss_mask.bool()
    if ignore_no_eos:
        attention_mask = sample.get("attention_mask")
        if attention_mask is None:
            raise ValueError(
                "mask-no-EOS filtering requires attention_mask to identify "
                "truncated trajectories"
            )
        if not isinstance(attention_mask, torch.Tensor):
            raise ValueError("attention_mask must be a local tensor")
        if attention_mask.shape != loss_mask.shape:
            raise ValueError("attention_mask and loss_mask must have matching shapes")
        # A sequence filling the padded width is truncated and has no EOS.
        has_eos = attention_mask.bool().sum(dim=-1) < attention_mask.shape[-1]
        valid_mask = valid_mask & has_eos.unsqueeze(-1)
    return bool(((token_rewards < 0) & valid_mask).any().item())


def _record_filter_result(*, accept: bool, all_correct: bool) -> None:
    tracker = stats_tracker.get("rollout")
    tracker.scalar(rejected_by_failed_or_perfect=int(not accept))
    tracker.scalar(rejected_by_all_correct=int(not accept and all_correct))
    tracker.scalar(rejected_by_all_wrong=int(not accept and not all_correct))


def filter_function(sample):
    # Use original_rewards if available for filtering and statistics
    rewards = _get_rewards(sample)
    accept = bool((rewards != rewards[0]).any().item())
    _record_filter_result(accept=accept, all_correct=bool((rewards > 0).all().item()))

    return accept


def filter_mixed_or_penalized_all_wrong(sample: dict[str, Any]) -> bool:
    """Keep mixed groups and all-wrong groups with a valid PRM penalty."""
    rewards = _get_rewards(sample)
    mixed = bool((rewards != rewards[0]).any().item())
    all_wrong = bool((rewards <= 0).all().item())
    preserved_all_wrong = (
        not mixed
        and all_wrong
        and _has_negative_process_reward(sample, ignore_no_eos=False)
    )
    accept = mixed or preserved_all_wrong
    _record_filter_result(accept=accept, all_correct=bool((rewards > 0).all().item()))
    stats_tracker.get("rollout").scalar(
        accepted_penalized_all_wrong=int(preserved_all_wrong)
    )

    return accept


def filter_mixed_or_penalized_all_wrong_mask_no_eos(
    sample: dict[str, Any],
) -> bool:
    """As above, but ignore penalties the actor will clear for truncation."""
    rewards = _get_rewards(sample)
    mixed = bool((rewards != rewards[0]).any().item())
    all_wrong = bool((rewards <= 0).all().item())
    preserved_all_wrong = (
        not mixed
        and all_wrong
        and _has_negative_process_reward(sample, ignore_no_eos=True)
    )
    accept = mixed or preserved_all_wrong
    _record_filter_result(accept=accept, all_correct=bool((rewards > 0).all().item()))
    stats_tracker.get("rollout").scalar(
        accepted_penalized_all_wrong=int(preserved_all_wrong)
    )

    return accept
