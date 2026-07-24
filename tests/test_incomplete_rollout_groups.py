from unittest.mock import MagicMock

import pytest
import torch

from areal.api.cli_args import NormConfig, PPOActorConfig
from areal.experimental.openai import InteractionWithTokenLogpReward
from areal.infra import workflow_context
from areal.infra.remote_inf_engine import GroupedRolloutWorkflow
from areal.infra.workflow_context import WorkflowContext
from areal.trainer.ppo.actor import _group_training_metrics
from areal.trainer.rl_trainer import (
    _collect_nonempty_rollout_batch,
    _minimum_usable_group_size,
)
from areal.utils.functional import ppo_actor_loss_fn


class _SequenceWorkflow:
    def __init__(self, results: list[dict | None]):
        self.results = iter(results)
        self.calls = 0

    async def arun_episode(self, engine, data):
        self.calls += 1
        return next(self.results)


def _trajectory(token: int) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[token]]),
        "attention_mask": torch.ones(1, 1, dtype=torch.bool),
    }


@pytest.fixture
def rollout_stats(monkeypatch):
    tracker = MagicMock()
    monkeypatch.setattr(
        "areal.infra.remote_inf_engine.stats_tracker.get", lambda _: tracker
    )
    workflow_context.set(WorkflowContext(is_eval=False))
    return tracker


@pytest.mark.asyncio
async def test_grouped_rollout_keeps_each_usable_member_once(rollout_stats):
    workflow = _SequenceWorkflow([_trajectory(11), None, _trajectory(22)])
    grouped = GroupedRolloutWorkflow(workflow, group_size=3, logger=MagicMock())

    result = await grouped.arun_episode(MagicMock(), {})

    assert result is not None
    assert workflow.calls == 3
    torch.testing.assert_close(result["input_ids"], torch.tensor([[11], [22]]))
    rollout_stats.scalar.assert_called_once_with(
        target_slot_count=3,
        usable_slot_count=2,
        trainable_slot_count=2,
        fully_masked_group=False,
        singleton_slot_group=False,
        pre_filter_usable_slot_yield=2 / 3,
        pre_filter_trainable_slot_yield=2 / 3,
    )


@pytest.mark.asyncio
async def test_grouped_rollout_keeps_complete_group_unchanged(rollout_stats):
    workflow = _SequenceWorkflow([_trajectory(11), _trajectory(22), _trajectory(33)])
    grouped = GroupedRolloutWorkflow(workflow, group_size=3, logger=MagicMock())

    result = await grouped.arun_episode(MagicMock(), {})

    assert result is not None
    torch.testing.assert_close(result["input_ids"], torch.tensor([[11], [22], [33]]))


@pytest.mark.asyncio
async def test_grouped_rollout_marks_empty_group_unusable(rollout_stats):
    workflow = _SequenceWorkflow([None, None, None])
    grouped = GroupedRolloutWorkflow(workflow, group_size=3, logger=MagicMock())

    result = await grouped.arun_episode(MagicMock(), {})

    assert result is None
    assert workflow.calls == 3
    rollout_stats.scalar.assert_called_once_with(
        target_slot_count=3,
        usable_slot_count=0,
        trainable_slot_count=0,
        fully_masked_group=True,
        singleton_slot_group=False,
        pre_filter_usable_slot_yield=0.0,
        pre_filter_trainable_slot_yield=0.0,
    )


@pytest.mark.asyncio
async def test_grouped_rollout_marks_singleton_untrainable(rollout_stats):
    workflow = _SequenceWorkflow([None, _trajectory(11), None])
    grouped = GroupedRolloutWorkflow(
        workflow,
        group_size=3,
        logger=MagicMock(),
        min_usable_group_size=2,
    )

    result = await grouped.arun_episode(MagicMock(), {})

    assert result is None
    assert workflow.calls == 3
    rollout_stats.scalar.assert_called_once_with(
        target_slot_count=3,
        usable_slot_count=1,
        trainable_slot_count=0,
        fully_masked_group=False,
        singleton_slot_group=True,
        pre_filter_usable_slot_yield=1 / 3,
        pre_filter_trainable_slot_yield=0.0,
    )


@pytest.mark.asyncio
async def test_grouped_rollout_keeps_singleton_by_default(rollout_stats):
    workflow = _SequenceWorkflow([None, _trajectory(11)])
    grouped = GroupedRolloutWorkflow(workflow, group_size=2, logger=MagicMock())

    result = await grouped.arun_episode(MagicMock(), {})

    assert result is not None
    torch.testing.assert_close(result["input_ids"], torch.tensor([[11]]))


@pytest.mark.parametrize(
    "strict_policy",
    [
        {"drop_incomplete_group": True},
        {"reward_normalization": True},
    ],
)
@pytest.mark.asyncio
async def test_strict_group_policy_reports_partial_group_untrainable(
    rollout_stats, strict_policy
):
    workflow = _SequenceWorkflow(
        [{"usable": InteractionWithTokenLogpReward(reward=1.0)}, None]
    )
    grouped = GroupedRolloutWorkflow(
        workflow,
        group_size=2,
        logger=MagicMock(),
        **strict_policy,
    )

    result = await grouped.arun_episode(MagicMock(), {})

    assert result is None
    rollout_stats.scalar.assert_called_once_with(
        target_slot_count=2,
        usable_slot_count=1,
        trainable_slot_count=0,
        fully_masked_group=False,
        singleton_slot_group=True,
        pre_filter_usable_slot_yield=0.5,
        pre_filter_trainable_slot_yield=0.0,
    )


def test_grouped_rollout_rejects_impossible_minimum():
    with pytest.raises(ValueError, match="between 1 and group_size"):
        GroupedRolloutWorkflow(
            _SequenceWorkflow([]),
            group_size=2,
            logger=MagicMock(),
            min_usable_group_size=3,
        )


@pytest.mark.asyncio
async def test_grouped_rollout_rejects_duplicate_interaction_ids(rollout_stats):
    interaction = InteractionWithTokenLogpReward()
    workflow = _SequenceWorkflow(
        [{"duplicate": interaction}, {"duplicate": InteractionWithTokenLogpReward()}]
    )
    grouped = GroupedRolloutWorkflow(workflow, group_size=2, logger=MagicMock())

    with pytest.raises(ValueError, match="duplicate interaction IDs"):
        await grouped.arun_episode(MagicMock(), {})


def test_group_training_metrics_report_actual_size_and_token_weight():
    loss_mask = torch.tensor(
        [
            [1, 1, 0],
            [1, 0, 0],
            [1, 1, 1],
            [1, 1, 0],
            [1, 0, 0],
        ],
        dtype=torch.bool,
    )

    group_starts, group_sizes, loss_weights = _group_training_metrics(loss_mask, [2, 3])

    torch.testing.assert_close(
        group_starts, torch.tensor([True, False, True, False, False])
    )
    torch.testing.assert_close(group_sizes, torch.tensor([2.0, 0.0, 3.0, 0.0, 0.0]))
    torch.testing.assert_close(loss_weights, torch.tensor([3.0, 0.0, 6.0, 0.0, 0.0]))


def test_actor_loss_keeps_existing_token_weighted_group_reduction():
    """A larger usable group keeps proportionally more weight by default."""
    advantages = torch.tensor([[1.0], [1.0], [3.0], [3.0], [3.0]])
    zeros = torch.zeros_like(advantages)

    loss, _ = ppo_actor_loss_fn(
        logprobs=zeros,
        proximal_logprobs=zeros,
        old_logprobs=zeros,
        advantages=advantages,
        eps_clip=0.2,
        loss_mask=torch.ones_like(advantages, dtype=torch.bool),
    )

    # Groups [2, 3] have means -1 and -3. The existing pooled token mean is
    # (-1 * 2 + -3 * 3) / 5 = -2.2, not the equal-group mean -2.0.
    torch.testing.assert_close(loss, torch.tensor(-2.2), rtol=0, atol=1e-6)


def test_minimum_usable_group_size_is_owned_by_estimator():
    batch_relative = PPOActorConfig(
        adv_norm=NormConfig(mean_level="batch", std_level="batch")
    )
    group_relative = PPOActorConfig(
        reward_norm=NormConfig(mean_level="group", std_level="group", group_size=4)
    )

    assert _minimum_usable_group_size(batch_relative, target_group_size=4) == 1
    assert _minimum_usable_group_size(group_relative, target_group_size=4) == 2
    assert _minimum_usable_group_size(group_relative, target_group_size=1) == 1


def test_dynamic_collection_retries_an_empty_ready_batch():
    prepare_batch = MagicMock(side_effect=[[], [{"group": "next-ready"}]])

    result = _collect_nonempty_rollout_batch(prepare_batch, dynamic_bs=True)

    assert result == [{"group": "next-ready"}]
    assert prepare_batch.call_count == 2


def test_fixed_collection_rejects_an_empty_batch():
    with pytest.raises(RuntimeError, match="empty fixed-size batch"):
        _collect_nonempty_rollout_batch(MagicMock(return_value=[]), dynamic_bs=False)
