from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import DictConfig

from areal.api.cli_args import InferenceEngineConfig, NormConfig, PPOActorConfig
from areal.experimental.openai import InteractionWithTokenLogpReward
from areal.infra import workflow_context
from areal.infra.remote_inf_engine import GroupedRolloutWorkflow
from areal.infra.workflow_context import WorkflowContext
from areal.infra.workflow_executor import (
    WorkflowContractError,
    WorkflowExecutor,
    validate_rollout_group_sizes,
)
from areal.trainer.ppo.actor import _group_training_metrics
from areal.trainer.rl_trainer import _collect_trainable_rollout_batch
from areal.utils.functional import ppo_actor_loss_fn


class _SequenceWorkflow:
    def __init__(self, results: list[dict | None]):
        self.results = iter(results)
        self.calls = 0

    async def arun_episode(self, engine, data):
        self.calls += 1
        return next(self.results)


class _CyclingDataLoader:
    batch_size = 4

    def __iter__(self):
        return iter([[{}]])


def _trajectory(token: int, batch_size: int = 1) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.full((batch_size, 1), token),
        "attention_mask": torch.ones(batch_size, 1, dtype=torch.bool),
    }


@pytest.mark.parametrize(
    ("group_size", "min_usable_group_size"),
    [(0, 1), (2, 0), (2, 3)],
)
def test_rollout_group_size_contract_rejects_invalid_threshold(
    group_size, min_usable_group_size
):
    with pytest.raises(ValueError):
        validate_rollout_group_sizes(group_size, min_usable_group_size)


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
async def test_grouped_rollout_marks_empty_group_unusable(rollout_stats):
    grouped = GroupedRolloutWorkflow(
        _SequenceWorkflow([None, None, None]),
        group_size=3,
        logger=MagicMock(),
    )

    result = await grouped.arun_episode(MagicMock(), {})

    assert result is None
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
async def test_grouped_rollout_applies_estimator_minimum(rollout_stats):
    grouped = GroupedRolloutWorkflow(
        _SequenceWorkflow([None, _trajectory(11), None]),
        group_size=3,
        logger=MagicMock(),
        min_usable_group_size=2,
    )

    result = await grouped.arun_episode(MagicMock(), {})

    assert result is None
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
    grouped = GroupedRolloutWorkflow(
        _SequenceWorkflow([None, _trajectory(11)]),
        group_size=2,
        logger=MagicMock(),
    )

    result = await grouped.arun_episode(MagicMock(), {})

    assert result is not None
    torch.testing.assert_close(result["input_ids"], torch.tensor([[11]]))


@pytest.mark.asyncio
async def test_group_relative_tensor_slot_must_produce_one_member(rollout_stats):
    grouped = GroupedRolloutWorkflow(
        _SequenceWorkflow([_trajectory(11, batch_size=2), _trajectory(22)]),
        group_size=2,
        logger=MagicMock(),
        min_usable_group_size=2,
    )

    with pytest.raises(WorkflowContractError, match="slot sizes \\[2, 1\\]"):
        await grouped.arun_episode(MagicMock(), {})


@pytest.mark.asyncio
async def test_group_relative_interaction_slot_must_produce_one_member(rollout_stats):
    grouped = GroupedRolloutWorkflow(
        _SequenceWorkflow(
            [
                {
                    "first": InteractionWithTokenLogpReward(),
                    "second": InteractionWithTokenLogpReward(),
                },
                {"third": InteractionWithTokenLogpReward()},
            ]
        ),
        group_size=2,
        logger=MagicMock(),
        min_usable_group_size=2,
    )

    with pytest.raises(WorkflowContractError, match="slot sizes \\[2, 1\\]"):
        await grouped.arun_episode(MagicMock(), {})


@pytest.mark.parametrize("dynamic_bs", [False, True])
def test_executor_propagates_group_contract_error_without_retry(dynamic_bs):
    workflow = _SequenceWorkflow(
        [_trajectory(11, batch_size=2), _trajectory(22, batch_size=2)]
    )
    grouped = GroupedRolloutWorkflow(
        workflow,
        group_size=2,
        logger=MagicMock(),
        min_usable_group_size=2,
    )
    inference_engine = MagicMock()
    inference_engine.get_version.return_value = 0
    executor = WorkflowExecutor(
        InferenceEngineConfig(
            backend="sglang:d1",
            consumer_batch_size=1,
            max_concurrent_rollouts=1,
        ),
        inference_engine,
    )
    executor.initialize()

    try:
        with pytest.raises(WorkflowContractError, match="slot sizes \\[2, 2\\]"):
            executor.prepare_batch(
                _CyclingDataLoader(),
                grouped,
                dynamic_bs=dynamic_bs,
            )
    finally:
        executor.destroy()

    assert workflow.calls == 2


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

    torch.testing.assert_close(loss, torch.tensor(-2.2), rtol=0.0, atol=1e-6)


@pytest.mark.parametrize(
    "normalization",
    [
        NormConfig(mean_level="group", std_level=None, group_size=4),
        NormConfig(mean_level=None, std_level="group", group_size=4),
        {"mean_level": "group", "std_level": None, "group_size": 4},
        DictConfig({"mean_level": None, "std_level": "group", "group_size": 4}),
    ],
)
def test_min_usable_group_size_defaults_to_estimator_minimum(normalization):
    group_relative = PPOActorConfig(adv_norm=normalization)

    assert group_relative.resolve_min_usable_group_size(target_group_size=4) == 2
    assert group_relative.resolve_min_usable_group_size(target_group_size=1) == 1


def test_batch_relative_estimator_keeps_usable_singleton():
    actor = PPOActorConfig(adv_norm=NormConfig(mean_level="batch", std_level="batch"))

    assert actor.resolve_min_usable_group_size(target_group_size=4) == 1


def test_explicit_min_usable_group_size_overrides_derivation():
    actor = PPOActorConfig(
        adv_norm=NormConfig(mean_level="group", std_level="group", group_size=4),
        min_usable_group_size=3,
    )

    assert actor.resolve_min_usable_group_size(target_group_size=4) == 3


def test_group_statistics_reject_explicit_singleton_minimum():
    with pytest.raises(ValueError, match="at least 2"):
        PPOActorConfig(
            adv_norm=NormConfig(mean_level="group", std_level="group", group_size=4),
            min_usable_group_size=1,
        )


def test_batch_relative_estimator_accepts_explicit_singleton_minimum():
    actor = PPOActorConfig(
        adv_norm=NormConfig(mean_level="batch", std_level="batch"),
        min_usable_group_size=1,
    )

    assert actor.resolve_min_usable_group_size(target_group_size=4) == 1


def test_dynamic_collection_backfills_from_ready_groups():
    prepare_batch = MagicMock(
        side_effect=[[{"group": "first"}], [], [{"group": "second"}]]
    )

    result = _collect_trainable_rollout_batch(
        prepare_batch,
        dynamic_bs=True,
        min_batch_size=2,
    )

    assert result == [{"group": "first"}, {"group": "second"}]
    assert prepare_batch.call_count == 3


def test_dynamic_collection_aborts_after_consecutive_empty_rounds():
    prepare_batch = MagicMock(return_value=[])

    with pytest.raises(RuntimeError, match="added no trainable group"):
        _collect_trainable_rollout_batch(
            prepare_batch,
            dynamic_bs=True,
            min_batch_size=1,
            stall_timeout=0.0,
        )

    assert prepare_batch.call_count == 8


def test_dynamic_collection_empty_streak_resets_on_progress():
    prepare_batch = MagicMock(
        side_effect=[[], [{"group": "first"}], [], [{"group": "second"}]]
    )

    result = _collect_trainable_rollout_batch(
        prepare_batch,
        dynamic_bs=True,
        min_batch_size=2,
        max_empty_rounds=2,
        stall_timeout=0.0,
    )

    assert result == [{"group": "first"}, {"group": "second"}]
    assert prepare_batch.call_count == 4


def test_dynamic_collection_tolerates_fast_all_reject_bursts():
    # A staleness flush drains many rejected rounds in near-zero time; the
    # round bound alone must not abort while the stall timeout has not run.
    prepare_batch = MagicMock(side_effect=[[]] * 20 + [[{"group": "fresh"}]])

    result = _collect_trainable_rollout_batch(
        prepare_batch,
        dynamic_bs=True,
        min_batch_size=1,
    )

    assert result == [{"group": "fresh"}]
    assert prepare_batch.call_count == 21


def test_fixed_collection_rejects_undersized_batch():
    with pytest.raises(RuntimeError, match="only 1 trainable groups"):
        _collect_trainable_rollout_batch(
            MagicMock(return_value=[{"group": "only"}]),
            dynamic_bs=False,
            min_batch_size=2,
        )
