# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch

from areal.api import ModelResponse
from areal.experimental.openai.proxy.server import (
    deserialize_interactions,
    serialize_interactions,
)
from areal.experimental.openai.proxy.workflow import OpenAIProxyWorkflow
from areal.experimental.openai.types import InteractionWithTokenLogpReward
from areal.infra.workflow_executor import WorkflowExecutor, _RolloutTaskInput
from areal.utils.data import concat_padded_tensors
from areal.utils.rollout_stats import (
    ROLLOUT_HARNESS_OUTCOME,
    ROLLOUT_TURN_COUNT,
    record_rollout_turn_stats,
    with_rollout_turn_stats,
)
from areal.utils.stats_tracker import DistributedStatsTracker


@pytest.fixture
def recorded_stats(monkeypatch) -> dict[str, DistributedStatsTracker]:
    """Capture rollout metrics without touching global trackers."""
    trackers: dict[str, DistributedStatsTracker] = {}

    def get_tracker(name: str = "") -> DistributedStatsTracker:
        return trackers.setdefault(name, DistributedStatsTracker(name))

    monkeypatch.setattr(
        "areal.experimental.openai.proxy.workflow.stats_tracker.get", get_tracker
    )
    from areal.infra import workflow_context
    from areal.infra.workflow_context import WorkflowContext

    workflow_context.set(WorkflowContext(is_eval=False))
    return trackers


def _result(reward: float, turns: int) -> dict[str, InteractionWithTokenLogpReward]:
    generated_turn_ids = torch.arange(turns, dtype=torch.int32).repeat_interleave(2)
    turn_ids = torch.cat(
        [torch.tensor([-1], dtype=torch.int32), generated_turn_ids]
    ).unsqueeze(0)
    interaction = InteractionWithTokenLogpReward(
        reward=reward,
        _cache={
            "turn_ids": turn_ids,
            "input_ids": torch.ones_like(turn_ids),
            "loss_mask": turn_ids >= 0,
            "attention_mask": torch.ones_like(turn_ids, dtype=torch.bool),
            "rewards": torch.tensor([reward], dtype=torch.float32),
        },
    )
    return {f"completion-{reward}-{turns}": interaction}


def _trajectory(reward: float, turns: int, metadata: dict | None = None) -> dict:
    interaction = next(iter(_result(reward, turns).values()))
    return with_rollout_turn_stats(interaction.to_tensor_dict(), metadata)


def test_selected_batch_uses_eval_rollout_scope(recorded_stats, monkeypatch):
    """Evaluation reward and turns stay isolated from training rollout metrics."""
    monkeypatch.setattr(
        "areal.experimental.openai.proxy.workflow.workflow_context.get",
        lambda: SimpleNamespace(is_eval=True),
    )

    OpenAIProxyWorkflow._record_interaction_stats(_result(1.0, 2))
    record_rollout_turn_stats([_trajectory(1.0, 2)], scope="eval-train")

    metrics = recorded_stats["eval-train"].export()

    assert recorded_stats["eval-rollout"].export()[
        "eval-rollout/reward"
    ] == pytest.approx(1.0)
    assert metrics["eval-train/num_turns/avg"] == pytest.approx(2.0)
    assert metrics["eval-train/num_turns/min"] == pytest.approx(2.0)
    assert metrics["eval-train/num_turns/max"] == pytest.approx(2.0)
    assert "rollout" not in recorded_stats


def test_completion_records_reward_without_turn_metrics(recorded_stats):
    """Finishing a rollout cannot pollute the selected training batch's turns."""
    OpenAIProxyWorkflow._record_interaction_stats(_result(1.0, 50))

    assert recorded_stats["rollout"].export() == {
        "rollout/reward": pytest.approx(1.0),
        "rollout/reward__count": 1,
    }


def test_final_metadata_overrides_cached_interaction_status(recorded_stats):
    interactions = _result(0.0, 4)
    interaction = next(iter(interactions.values()))
    cached = interaction.to_tensor_dict()
    OpenAIProxyWorkflow._stamp_interaction_metadata(
        interactions,
        {
            "arena_status": "HARNESS_FAILED",
            "harness_outcome_code": "LLM_OUTPUT_TOKEN_LIMIT_EXCEEDED",
        },
    )

    row = with_rollout_turn_stats(interaction.to_tensor_dict(), interaction.metadata)
    record_rollout_turn_stats([row])
    metrics = recorded_stats["train"].export()

    assert ROLLOUT_HARNESS_OUTCOME not in cached  # Do not mutate the token cache.
    assert metrics["train/num_turns_no_harness_err_count"] == 0
    metric = "train/num_turns_harness_err/LLM_OUTPUT_TOKEN_LIMIT_EXCEEDED"
    assert metrics[f"{metric}_count"] == 1
    assert metrics[f"{metric}/avg"] == 4


def _generated_interaction(input_len, output_len=2, parent=None):
    return InteractionWithTokenLogpReward(
        model_response=ModelResponse(
            input_tokens=[1] * input_len,
            output_tokens=[2] * output_len,
            output_logprobs=[-0.1] * output_len,
            output_versions=[0] * output_len,
        ),
        reward=0.0,
        parent=parent,
        chat_template_type="concat",
        metadata={"arena_status": "DONE"},
    )


@pytest.mark.parametrize("http_transport", [False, True])
def test_interaction_history_survives_tensor_fallback_and_ignores_empty_response(
    recorded_stats, http_transport
):
    """Raw branch turns survive reset IDs; zero-token requests add no generation."""
    child_input_len, empty_last_response = 1, True
    parent = _generated_interaction(1)
    child = _generated_interaction(child_input_len, parent=parent)
    leaf = (
        _generated_interaction(child_input_len + 3, output_len=0, parent=child)
        if empty_last_response
        else child
    )
    tensor = leaf.to_tensor_dict()
    tensor["loss_mask"].zero_()
    interactions = {"leaf": leaf}
    if http_transport:
        interactions = deserialize_interactions(serialize_interactions(interactions))
        assert interactions["leaf"].parent is None
        assert interactions["leaf"].model_response is None
    executor = _executor()
    task = _RolloutTaskInput(
        task_id=1,
        data={},
        workflow=SimpleNamespace(arun_episode=AsyncMock(return_value=interactions)),
    )

    result = asyncio.run(WorkflowExecutor._create_workflow_task(executor, task)())
    assert result is not None
    record_rollout_turn_stats([result.trajectory])
    metrics = recorded_stats["train"].export()

    assert metrics["train/num_turns_count"] == 1
    assert metrics["train/num_turns/avg"] == 2
    assert metrics["train/num_turns_no_harness_err/avg"] == 2
    assert metrics["train/num_turns_harness_err_count"] == 0


def test_serialized_history_preserves_turns_but_uses_final_harness_status(
    recorded_stats,
):
    """Early turn snapshots must not freeze incomplete Harness classification."""
    leaf = _generated_interaction(1, parent=_generated_interaction(1))
    restored = deserialize_interactions(serialize_interactions({"leaf": leaf}))
    OpenAIProxyWorkflow._stamp_interaction_metadata(
        restored,
        {
            "arena_status": "HARNESS_FAILED",
            "harness_outcome_code": "LLM_CONTEXT_WINDOW_EXCEEDED",
        },
    )
    restored_leaf = restored["leaf"]
    row = with_rollout_turn_stats(
        restored_leaf.to_tensor_dict(),
        restored_leaf.metadata,
        interaction=restored_leaf,
    )

    record_rollout_turn_stats([row])
    metrics = recorded_stats["train"].export()

    assert metrics["train/num_turns_no_harness_err_count"] == 0
    assert metrics["train/num_turns_harness_err/LLM_CONTEXT_WINDOW_EXCEEDED/avg"] == 2


def test_remote_turn_history_is_not_fetched_or_replaced_after_hook(
    recorded_stats,
    monkeypatch,
):
    """Training mask edits leave V2 count/storage handles untouched."""
    from areal.infra.rpc.rtensor import RTensor, TensorShardInfo
    from areal.infra.rpc.serialization import deserialize_value, serialize_value

    row = _trajectory(1.0, 4)
    shard = TensorShardInfo(shard_id="old-turn-count", node_addr="test-storage")
    old_count = RTensor(shard=shard, data=torch.empty(1, device="meta"))
    row[ROLLOUT_TURN_COUNT] = old_count
    monkeypatch.setattr(
        "areal.infra.rpc.rtensor.get_backend",
        Mock(side_effect=AssertionError("must not fetch rollout tensors")),
    )
    executor = _executor()

    def mask_hook(stage, trajectory, state):
        trajectory["loss_mask"].zero_()
        return trajectory

    task = _RolloutTaskInput(
        task_id=1,
        data={},
        workflow=SimpleNamespace(arun_episode=AsyncMock(return_value=row)),
        should_accept_fn=lambda traj: (mask_hook(None, traj, None) is traj),
    )

    result = asyncio.run(WorkflowExecutor._create_workflow_task(executor, task)())
    assert result is not None
    restored = deserialize_value(serialize_value(result.trajectory))

    assert result.trajectory[ROLLOUT_TURN_COUNT] is old_count
    assert restored[ROLLOUT_TURN_COUNT].data.is_meta
    assert restored[ROLLOUT_TURN_COUNT].shard == shard
    assert RTensor.collect_shards(restored) == {"test-storage": ["old-turn-count"]}


def _executor():
    return SimpleNamespace(
        config=SimpleNamespace(
            check_trajectory_format=False,
            dump_to_file=False,
            enable_rollout_tracing=False,
        ),
        inference_engine=None,
        logger=None,
        staleness_manager=SimpleNamespace(
            on_rollout_accepted=Mock(), on_rollout_rejected=Mock()
        ),
        _clear_rejected_trajectory=AsyncMock(),
    )


@pytest.mark.parametrize("is_eval", [False, True])
@pytest.mark.parametrize("accept", [False, True])
def test_executor_records_only_accepted_eval_rows(recorded_stats, is_eval, accept):
    executor = _executor()
    task = _RolloutTaskInput(
        task_id=1,
        data={},
        workflow=SimpleNamespace(arun_episode=AsyncMock(return_value=_result(0.0, 4))),
        should_accept_fn=lambda trajectory: accept,
        is_eval=is_eval,
    )

    result = asyncio.run(WorkflowExecutor._create_workflow_task(executor, task)())

    assert (result is not None and result.trajectory is not None) == accept
    if is_eval and accept:
        metrics = recorded_stats["eval-train"].export()
        assert metrics["eval-train/num_turns_count"] == 1
        assert metrics["eval-train/num_turns/avg"] == 4
    else:
        assert "eval-train" not in recorded_stats
    if not is_eval:
        assert "train" not in recorded_stats
    if result is not None and result.trajectory is not None:
        assert result.trajectory[ROLLOUT_TURN_COUNT].shape == (1,)


@pytest.mark.parametrize("edit", ["loss_mask", "attention_mask", "crop"])
@pytest.mark.parametrize("harness_failed", [False, True])
def test_executor_preserves_real_turns_after_training_edits(
    recorded_stats, edit, harness_failed
):
    """Selected cases retain their real turns even if training masks/crops them."""
    executor = _executor()

    def change(trajectory, retained_turns):
        if edit != "crop":
            trajectory[edit][trajectory["turn_ids"] >= retained_turns] = 0
        else:
            for key, value in list(trajectory.items()):
                if isinstance(value, torch.Tensor) and value.ndim == 2:
                    trajectory[key] = value[:, : 1 + 2 * retained_turns]
        return trajectory

    def accept(trajectory):
        change(trajectory, 1)
        return True

    interactions = _result(0.0, 4)
    next(iter(interactions.values())).metadata = {
        "arena_status": "HARNESS_FAILED" if harness_failed else "DONE"
    }
    task = _RolloutTaskInput(
        task_id=1,
        data={},
        workflow=SimpleNamespace(arun_episode=AsyncMock(return_value=interactions)),
        should_accept_fn=accept,
    )

    result = asyncio.run(WorkflowExecutor._create_workflow_task(executor, task)())
    assert result is not None
    record_rollout_turn_stats([result.trajectory])
    metrics = recorded_stats["train"].export()

    assert metrics["train/num_turns/avg"] == 4
    subgroup = (
        "num_turns_harness_err/OTHER" if harness_failed else "num_turns_no_harness_err"
    )
    assert metrics[f"train/{subgroup}/avg"] == 4
    assert metrics[f"train/{subgroup}_count"] == 1


def test_concurrent_eval_records_masks_and_values_on_event_loop(
    recorded_stats, monkeypatch
):
    from areal.infra import workflow_executor

    executor = _executor()
    record_threads = []
    original_record = workflow_executor.record_rollout_turn_stats

    def record(*args, **kwargs):
        record_threads.append(threading.get_ident())
        return original_record(*args, **kwargs)

    monkeypatch.setattr(workflow_executor, "record_rollout_turn_stats", record)
    tasks = []
    for index, rows in enumerate(
        [
            [_trajectory(1.0, 2)],
            [
                _trajectory(0.0, 6, {"arena_status": "HARNESS_FAILED"}),
                _trajectory(0.0, 4, {"arena_status": "HARNESS_FAILED"}),
            ],
        ]
    ):
        task = _RolloutTaskInput(
            task_id=index,
            data={},
            workflow=SimpleNamespace(
                arun_episode=AsyncMock(return_value=concat_padded_tensors(rows))
            ),
            is_eval=True,
        )
        tasks.append(WorkflowExecutor._create_workflow_task(executor, task))

    async def run():
        return await asyncio.gather(*(task() for task in tasks))

    results = asyncio.run(run())
    metrics = recorded_stats["eval-train"].export()

    assert all(result is not None for result in results)
    assert record_threads == [threading.get_ident()] * 2
    assert metrics["eval-train/num_turns_count"] == 3
    assert metrics["eval-train/num_turns/avg"] == 4
    assert metrics["eval-train/num_turns_no_harness_err/avg"] == 2
    assert metrics["eval-train/num_turns_harness_err/avg"] == 5


def test_eval_metadata_failure_does_not_accept_then_reject(recorded_stats, monkeypatch):
    executor = _executor()
    monkeypatch.setattr(
        "areal.infra.workflow_executor.localize_rollout_turn_stats",
        Mock(side_effect=ValueError("invalid turn metadata")),
    )
    task = _RolloutTaskInput(
        task_id=1,
        data={},
        workflow=SimpleNamespace(arun_episode=AsyncMock(return_value=_result(1.0, 2))),
        is_eval=True,
    )

    result = asyncio.run(WorkflowExecutor._create_workflow_task(executor, task)())

    assert result is None
    executor.staleness_manager.on_rollout_accepted.assert_not_called()
    executor.staleness_manager.on_rollout_rejected.assert_called_once()


def test_stamp_interaction_metadata_preserves_existing_values():
    """Episode join keys are attached to every exported branch."""
    interactions = _result(1.0, 2)
    interaction = next(iter(interactions.values()))
    interaction.metadata = {"thinking_option": "on"}

    OpenAIProxyWorkflow._stamp_interaction_metadata(
        interactions,
        {
            "session_id": "268-2",
            "arena_task_id": "arena-task",
            "arena_status": "DONE",
        },
    )

    assert interaction.metadata == {
        "thinking_option": "on",
        "session_id": "268-2",
        "arena_task_id": "arena-task",
        "arena_status": "DONE",
    }


def test_record_interaction_stats_tensorless_records_reward_only(recorded_stats):
    """String-only interactions must not fail while recording optional metrics."""
    interaction = InteractionWithTokenLogpReward(
        reward=0.5,
        messages=[{"role": "user", "content": "question"}],
        output_message_list=[{"role": "assistant", "content": "answer"}],
    )

    OpenAIProxyWorkflow._record_interaction_stats({"completion": interaction})

    assert recorded_stats["rollout"].export() == {
        "rollout/reward": pytest.approx(0.5),
        "rollout/reward__count": 1,
    }
