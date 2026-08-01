# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from areal.api import RolloutWorkflow
from areal.experimental.openai import InteractionWithTokenLogpReward
from areal.infra import dist_rollout
from areal.infra.dist_rollout import DistRolloutCoordinator
from areal.infra.remote_inf_engine import GroupedRolloutWorkflow


class _ListWorkflow(RolloutWorkflow):
    def __init__(self, results):
        self.results = list(results)
        self.index = 0

    async def arun_episode(self, engine, data):
        result = self.results[self.index]
        self.index += 1
        return result


class _Logger:
    def __init__(self):
        self.messages: list[str] = []

    def warning(self, message: str):
        self.messages.append(message)


class _TrainEngine:
    def is_data_parallel_head(self):
        return True


class _RolloutEngine:
    def __init__(self):
        self.prepare_kwargs = None
        self.rollout_kwargs = None

    def prepare_batch(self, *args, **kwargs):
        self.prepare_kwargs = kwargs
        return [{"trajectory": True}]

    def rollout_batch(self, *args, **kwargs):
        self.rollout_kwargs = kwargs
        return [{"trajectory": True}]


def _interaction(reward: float) -> InteractionWithTokenLogpReward:
    return InteractionWithTokenLogpReward(
        reward=reward,
        _cache={
            "input_ids": torch.tensor([[1, 2]]),
            "loss_mask": torch.tensor([[0, 1]]),
            "logprobs": torch.tensor([[0.0, -0.1]]),
            "versions": torch.tensor([[-1, 0]]),
            "attention_mask": torch.tensor([[True, True]]),
            "rewards": torch.tensor([reward]),
        },
    )


@pytest.mark.asyncio
async def test_grouped_rollout_workflow_normalizes_rewards_and_updates_cache():
    first = _interaction(1.0)
    second = _interaction(3.0)
    workflow = GroupedRolloutWorkflow(
        _ListWorkflow([{"a": first}, {"b": second}]),
        group_size=2,
        logger=_Logger(),
        reward_normalization=True,
    )

    result = await workflow.arun_episode(engine=None, data={})

    assert result == {"a": first, "b": second}
    assert first.reward == pytest.approx(-1.0)
    assert second.reward == pytest.approx(1.0)
    assert first.original_reward == pytest.approx(1.0)
    assert second.original_reward == pytest.approx(3.0)
    assert first._cache is not None
    assert second._cache is not None
    assert first._cache["rewards"].item() == pytest.approx(-1.0)
    assert second._cache["rewards"].item() == pytest.approx(1.0)
    assert first._cache["original_rewards"].item() == pytest.approx(1.0)
    assert second._cache["original_rewards"].item() == pytest.approx(3.0)


@pytest.mark.asyncio
async def test_grouped_rollout_workflow_drops_incomplete_group():
    logger = _Logger()
    workflow = GroupedRolloutWorkflow(
        _ListWorkflow([{"a": _interaction(1.0)}, None]),
        group_size=2,
        logger=logger,
        drop_incomplete_group=True,
    )

    result = await workflow.arun_episode(engine=None, data={})

    assert result is None
    assert "dropping entire group" in logger.messages[0]


@pytest.mark.asyncio
async def test_grouped_rollout_workflow_reward_normalization_requires_full_group():
    logger = _Logger()
    workflow = GroupedRolloutWorkflow(
        _ListWorkflow([{"a": _interaction(1.0)}, None]),
        group_size=2,
        logger=logger,
        reward_normalization=True,
    )

    result = await workflow.arun_episode(engine=None, data={})

    assert result is None
    assert any("reward_normalization: dropping group" in m for m in logger.messages)


def test_dist_rollout_coordinator_forwards_reward_group_flags(monkeypatch):
    rollout_engine = _RolloutEngine()
    coordinator = DistRolloutCoordinator(rollout_engine, _TrainEngine())
    monkeypatch.setattr(
        coordinator,
        "_broadcast_and_redistribute_trajectories",
        lambda trajectories: trajectories,
    )
    monkeypatch.setattr(dist_rollout.current_platform, "current_device", lambda: "cpu")
    monkeypatch.setattr(dist_rollout, "tensor_container_to", lambda data, device: data)

    coordinator.prepare_batch(
        dataloader=object(),
        workflow=object(),
        reward_normalization=True,
        drop_incomplete_group=True,
    )
    coordinator.rollout_batch(
        data=[{}],
        workflow=object(),
        reward_normalization=True,
        drop_incomplete_group=True,
    )

    assert rollout_engine.prepare_kwargs["reward_normalization"] is True
    assert rollout_engine.prepare_kwargs["drop_incomplete_group"] is True
    assert rollout_engine.rollout_kwargs["reward_normalization"] is True
    assert rollout_engine.rollout_kwargs["drop_incomplete_group"] is True
