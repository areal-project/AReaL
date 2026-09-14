"""CPU-only tests for PRM-aware rollout group filtering."""

from __future__ import annotations

from typing import Any

import pytest
import torch

import examples.swe.filter_function as filter_module


class _RecordedStats:
    def __init__(self) -> None:
        self.values: dict[str, Any] = {}

    def scalar(self, **kwargs) -> None:
        self.values.update(kwargs)


@pytest.fixture
def recorded_stats(monkeypatch) -> _RecordedStats:
    stats = _RecordedStats()
    monkeypatch.setattr(filter_module.stats_tracker, "get", lambda _: stats)
    return stats


def _sample(
    rewards: list[float],
    *,
    token_rewards: list[list[float]] | None = None,
    loss_mask: list[list[float]] | None = None,
    attention_mask: list[list[float]] | None = None,
) -> dict[str, torch.Tensor]:
    sample = {
        "rewards": torch.tensor(rewards),
        "original_rewards": torch.tensor(rewards),
    }
    if token_rewards is not None:
        sample["token_rewards"] = torch.tensor(token_rewards)
    if loss_mask is not None:
        sample["loss_mask"] = torch.tensor(loss_mask)
    if attention_mask is not None:
        sample["attention_mask"] = torch.tensor(attention_mask)
    return sample


def test_prm_filter_accepts_mixed_group_without_process_penalty(recorded_stats):
    accepted = filter_module.filter_mixed_or_penalized_all_wrong(_sample([0.0, 1.0]))

    assert accepted is True
    assert recorded_stats.values["accepted_penalized_all_wrong"] == 0


def test_prm_filter_accepts_penalized_all_wrong_group(recorded_stats):
    sample = _sample(
        [0.0, 0.0],
        token_rewards=[[0.0, 0.0, -0.1], [0.0, 0.0, 0.0]],
        loss_mask=[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
    )

    assert filter_module.filter_mixed_or_penalized_all_wrong(sample) is True
    assert recorded_stats.values["accepted_penalized_all_wrong"] == 1


@pytest.mark.parametrize(
    ("token_rewards", "loss_mask"),
    [
        pytest.param(None, None, id="no-process-signal"),
        pytest.param(
            [[-0.1, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            id="penalty-outside-loss-mask",
        ),
    ],
)
def test_prm_filter_rejects_unpenalized_all_wrong_group(
    recorded_stats, token_rewards, loss_mask
):
    sample = _sample([0.0, 0.0], token_rewards=token_rewards, loss_mask=loss_mask)

    assert filter_module.filter_mixed_or_penalized_all_wrong(sample) is False
    assert recorded_stats.values["rejected_by_all_wrong"] == 1


def test_prm_filter_rejects_all_correct_even_with_penalty(recorded_stats):
    sample = _sample(
        [1.0, 1.0],
        token_rewards=[[0.0, -0.1], [0.0, 0.0]],
        loss_mask=[[0.0, 1.0], [0.0, 1.0]],
    )

    assert filter_module.filter_mixed_or_penalized_all_wrong(sample) is False
    assert recorded_stats.values["rejected_by_all_correct"] == 1


def test_mask_no_eos_filter_ignores_truncated_only_penalty(recorded_stats):
    truncated_only = _sample(
        [0.0, 0.0],
        token_rewards=[[0.0, -0.1, 0.0], [0.0, 0.0, 0.0]],
        loss_mask=[[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        attention_mask=[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
    )

    assert (
        filter_module.filter_mixed_or_penalized_all_wrong_mask_no_eos(truncated_only)
        is False
    )

    truncated_only["attention_mask"][0, -1] = 0
    assert (
        filter_module.filter_mixed_or_penalized_all_wrong_mask_no_eos(truncated_only)
        is True
    )


@pytest.mark.parametrize("truncated", [True, False])
def test_mask_no_eos_filter_uses_explicit_termination_metadata(
    recorded_stats, truncated
):
    sample = _sample(
        [0.0, 0.0],
        token_rewards=[[0.0, -0.1], [0.0, 0.0]],
        loss_mask=[[0.0, 1.0], [0.0, 1.0]],
        attention_mask=[[1.0, 1.0], [1.0, 1.0]],
    )
    sample["is_truncated"] = torch.tensor([truncated, True])
    assert (
        filter_module.filter_mixed_or_penalized_all_wrong_mask_no_eos(sample)
        is not truncated
    )


def test_mask_no_eos_filter_requires_attention_mask(recorded_stats):
    sample = _sample(
        [0.0, 0.0],
        token_rewards=[[0.0, -0.1], [0.0, 0.0]],
        loss_mask=[[0.0, 1.0], [0.0, 1.0]],
    )

    with pytest.raises(ValueError, match="requires attention_mask"):
        filter_module.filter_mixed_or_penalized_all_wrong_mask_no_eos(sample)
