# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from areal.infra.controller.train_controller import TrainController, _dispatch_tensors
from areal.infra.dp_allocation import AllocationInput, allocate_trajectories
from areal.trainer.ppo.actor import PPOActorController


def _make_rw_pair(
    pair_idx: int, chosen_len: int = 3, rejected_len: int = 2
) -> tuple[dict[str, object], dict[str, object]]:
    chosen: dict[str, object] = {
        "input_ids": torch.full((1, chosen_len), pair_idx * 2 + 1, dtype=torch.long),
        "attention_mask": torch.ones((1, chosen_len), dtype=torch.bool),
        "meta": {"pair": pair_idx, "role": "chosen"},
    }
    rejected: dict[str, object] = {
        "input_ids": torch.full((1, rejected_len), pair_idx * 2 + 2, dtype=torch.long),
        "attention_mask": torch.ones((1, rejected_len), dtype=torch.bool),
        "meta": {"pair": pair_idx, "role": "rejected"},
    }
    return chosen, rejected


def _build_rw_batch(n_pairs: int) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for pair_idx in range(n_pairs):
        chosen, rejected = _make_rw_pair(pair_idx)
        items.extend([chosen, rejected])
    return items


def test_dta_rejects_group_size_greater_than_one() -> None:
    items = _build_rw_batch(n_pairs=2)

    with pytest.raises(ValueError, match="DTA requires sequence-level independence"):
        _dispatch_tensors(
            items,
            dp_size=2,
            group_size=2,
            packing_algorithm="dta",
        )

    with pytest.raises(ValueError, match="DTA requires sequence-level independence"):
        allocate_trajectories(
            AllocationInput(
                items=items,
                n_groups=2,
                algorithm="dta",
                group_size=2,
            )
        )


def test_allocate_trajectories_dta_flattens_grouped_rollouts() -> None:
    items: list[dict[str, torch.Tensor]] = [
        {
            "input_ids": torch.tensor([[1, 2, 3], [1, 2, 4]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.bool),
        },
        {
            "input_ids": torch.tensor([[5, 6, 0]]),
            "attention_mask": torch.tensor([[1, 1, 0]], dtype=torch.bool),
        },
    ]

    allocation = allocate_trajectories(
        AllocationInput(items=items, n_groups=2, algorithm="dta")
    )

    assert len(allocation.items) == 3
    assert len(allocation.group_indices) == 2
    flat_indices = [idx for group in allocation.group_indices for idx in group]
    assert len(flat_indices) == len(allocation.items)
    assert len(set(flat_indices)) == len(allocation.items)
    assert sorted(flat_indices) == list(range(len(allocation.items)))
    assert allocation.metrics is not None
    stats = allocation.metrics.to_stats()
    assert stats["dta/n_tokens"] == 8.0
    assert stats["dta/n_tree_tokens_before_allocation"] == 6.0
    assert stats["dta/n_tree_tokens_after_allocation"] == 6.0
    assert "dta/n_tree_tokens_after_allocation" in stats


def test_ppo_actor_prepare_batch_dta_flattens_grouped_rollouts(monkeypatch) -> None:
    controller = object.__new__(PPOActorController)
    controller.config = type("Config", (), {"packing_algorithm": "dta"})()
    batch: list[dict[str, torch.Tensor]] = [
        {
            "input_ids": torch.tensor([[1, 2, 3], [1, 2, 4]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.bool),
            "loss_mask": torch.ones((2, 3), dtype=torch.float32),
        }
    ]
    monkeypatch.setattr(
        TrainController,
        "prepare_batch",
        lambda self, *args, **kwargs: batch,
    )

    prepared = controller.prepare_batch(object(), workflow=object(), workflow_kwargs={})

    assert len(prepared) == 2
    assert all(item["input_ids"].shape[0] == 1 for item in prepared)
    torch.testing.assert_close(
        prepared[0]["input_ids"],
        torch.tensor([[1, 2, 3]]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        prepared[1]["input_ids"],
        torch.tensor([[1, 2, 4]]),
        rtol=0,
        atol=0,
    )
