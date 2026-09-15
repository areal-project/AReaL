# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio

import pytest
import torch

from areal.api import RolloutWorkflow
from areal.experimental.openai import InteractionWithTokenLogpReward
from areal.infra import dist_rollout, workflow_context
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


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["child_error", "child_cancel", "parent_cancel"])
async def test_group_failure_cancels_siblings_before_finalizing(failure):
    """Failed or cancelled groups must not wait forever on unfinished siblings."""
    sibling_started = asyncio.Event()
    sibling_cancelled = asyncio.Event()
    never_set = asyncio.Event()
    finalized = []
    original_error = ValueError("candidate failed")

    class FailingWorkflow(RolloutWorkflow):
        async def arun_episode(self, engine, data):
            if workflow_context.get().sample_idx == 0:
                await sibling_started.wait()
                if failure == "child_error":
                    raise original_error
                if failure == "child_cancel":
                    raise asyncio.CancelledError()
                await never_set.wait()
            else:
                sibling_started.set()
                try:
                    await never_set.wait()
                except asyncio.CancelledError:
                    # Include async teardown to verify it is drained as well.
                    await asyncio.sleep(0)
                    sibling_cancelled.set()
                    raise

        async def _afinalize_processor_cache_group(self, context):
            finalized.append(sibling_cancelled.is_set())

    workflow = GroupedRolloutWorkflow(FailingWorkflow(), group_size=2, logger=_Logger())
    task = asyncio.create_task(workflow.arun_episode(engine=None, data={}))
    try:
        await asyncio.wait_for(sibling_started.wait(), timeout=1)
        if failure == "parent_cancel":
            task.cancel()
        error_type = ValueError if failure == "child_error" else asyncio.CancelledError
        with pytest.raises(error_type) as exc_info:
            # Shield keeps the timeout from making the broken implementation
            # pass by cancelling its otherwise indefinitely waiting siblings.
            await asyncio.wait_for(asyncio.shield(task), timeout=1)
        if failure == "child_error":
            assert exc_info.value is original_error
        assert sibling_cancelled.is_set()
        assert finalized == [True]
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_group_metrics_include_failed_slots_before_propagating_error():
    """Completed rewards survive a sibling error in attempted-group metrics."""
    completed = asyncio.Event()
    metrics = []

    class Workflow(RolloutWorkflow):
        async def arun_episode(self, engine, data):
            if workflow_context.get().sample_idx == 0:
                completed.set()
                return {"completion": _interaction(0.75)}
            await completed.wait()
            raise ValueError("failed slot")

        def record_group_metrics(self, data, rewards, group_size):
            metrics.append((rewards, group_size))

    workflow = GroupedRolloutWorkflow(Workflow(), group_size=2, logger=_Logger())
    with pytest.raises(ValueError, match="failed slot"):
        await workflow.arun_episode(engine=None, data={})

    assert metrics == [([0.75, None], 2)]


@pytest.mark.asyncio
async def test_group_metrics_observe_rewards_before_normalization():
    """Per-Stream pass rates must use terminal rewards, not normalized values."""
    metrics = []
    agent = _ListWorkflow([{"first": _interaction(0.0)}, {"second": _interaction(1.0)}])
    agent.record_group_metrics = lambda data, rewards, size: metrics.append(rewards)
    workflow = GroupedRolloutWorkflow(
        agent, group_size=2, logger=_Logger(), reward_normalization=True
    )

    result = await workflow.arun_episode(engine=None, data={})

    assert metrics == [[0.0, 1.0]]
    assert result["first"].reward == pytest.approx(-1.0)
    assert result["second"].reward == pytest.approx(1.0)
