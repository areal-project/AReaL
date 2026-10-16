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
from areal.utils.data import concat_padded_tensors, split_and_unpad_tensor
from areal.utils.rollout_stats import (
    HARNESS_OUTCOME_METRIC_CODES,
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


def test_selected_batch_records_one_turn_count_per_row(recorded_stats):
    """Turn metrics retain the original average/minimum/maximum/count names."""
    record_rollout_turn_stats(
        [_trajectory(1.0, 2), _trajectory(0.0, 3), _trajectory(1.0, 1)]
    )

    metrics = recorded_stats["train"].export()

    assert metrics["train/num_turns_count"] == 3
    assert metrics["train/num_turns/avg"] == pytest.approx(2.0)
    assert metrics["train/num_turns/min"] == pytest.approx(1.0)
    assert metrics["train/num_turns/max"] == pytest.approx(3.0)
    assert metrics["train/num_turns_no_harness_err_count"] == 3
    assert metrics["train/num_turns_no_harness_err/avg"] == pytest.approx(2.0)
    assert metrics["train/num_turns_no_harness_err/min"] == pytest.approx(1.0)
    assert metrics["train/num_turns_no_harness_err/max"] == pytest.approx(3.0)


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


def test_selected_batch_excludes_harness_errors_from_filtered_turns(
    recorded_stats,
):
    """Filtered turn stats omit retained Harness failures only."""
    record_rollout_turn_stats(
        [
            _trajectory(
                0.0,
                4,
                {
                    "arena_status": "HARNESS_FAILED",
                    "harness_outcome_code": "LLM_CONTEXT_WINDOW_EXCEEDED",
                },
            ),
            _trajectory(1.0, 2),
        ]
    )

    metrics = recorded_stats["train"].export()

    assert metrics["train/num_turns/avg"] == pytest.approx(3.0)
    assert metrics["train/num_turns/min"] == pytest.approx(2.0)
    assert metrics["train/num_turns/max"] == pytest.approx(4.0)
    assert metrics["train/num_turns_no_harness_err_count"] == 1
    assert metrics["train/num_turns_no_harness_err/avg"] == pytest.approx(2.0)
    assert metrics["train/num_turns_no_harness_err/min"] == pytest.approx(2.0)
    assert metrics["train/num_turns_no_harness_err/max"] == pytest.approx(2.0)
    assert metrics["train/num_turns_harness_err_count"] == 1
    assert metrics["train/num_turns_harness_err/avg"] == pytest.approx(4.0)
    assert metrics["train/num_turns_harness_err/min"] == pytest.approx(4.0)
    assert metrics["train/num_turns_harness_err/max"] == pytest.approx(4.0)
    outcome_metric = "train/num_turns_harness_err/LLM_CONTEXT_WINDOW_EXCEEDED"
    assert metrics[f"{outcome_metric}_count"] == 1
    assert metrics[f"{outcome_metric}/avg"] == pytest.approx(4.0)
    assert metrics[f"{outcome_metric}/min"] == pytest.approx(4.0)
    assert metrics[f"{outcome_metric}/max"] == pytest.approx(4.0)


@pytest.mark.parametrize("code", ["FUTURE_UNKNOWN_CODE", None, ["invalid"]])
def test_selected_batch_bounds_unknown_harness_outcome(recorded_stats, code):
    """Unknown outcome values aggregate under OTHER instead of adding keys."""
    record_rollout_turn_stats(
        [
            _trajectory(
                0.0, 3, {"arena_status": "HARNESS_FAILED", "harness_outcome_code": code}
            )
        ]
    )

    metrics = recorded_stats["train"].export()

    assert "train/num_turns_harness_err/FUTURE_UNKNOWN_CODE/avg" not in metrics
    assert metrics["train/num_turns_harness_err/OTHER/avg"] == pytest.approx(3.0)


def test_completion_records_reward_without_turn_metrics(recorded_stats):
    """Finishing a rollout cannot pollute the selected training batch's turns."""
    OpenAIProxyWorkflow._record_interaction_stats(_result(1.0, 50))

    assert recorded_stats["rollout"].export() == {
        "rollout/reward": pytest.approx(1.0),
        "rollout/reward__count": 1,
    }


def test_buffered_rollouts_count_only_when_consumed(recorded_stats):
    """Old siblings plus two new completions count as the full selected group."""
    buffered = [_trajectory(1.0, 2) for _ in range(10)]
    future_batch = _trajectory(0.0, 50)
    OpenAIProxyWorkflow._record_interaction_stats(_result(1.0, 2))
    recorded_stats["rollout"].export()  # Previous logging window ends.
    new_rows = [_trajectory(1.0, 4), _trajectory(0.0, 6)]

    record_rollout_turn_stats([concat_padded_tensors(buffered + new_rows)])
    metrics = recorded_stats["train"].export()

    assert metrics["train/num_turns_count"] == 12
    assert metrics["train/num_turns/avg"] == pytest.approx(2.5)
    assert metrics["train/num_turns/max"] == 6
    record_rollout_turn_stats([future_batch])
    next_metrics = recorded_stats["train"].export()
    assert next_metrics["train/num_turns_count"] == 1
    assert next_metrics["train/num_turns/avg"] == 50


def test_turn_metadata_tracks_concat_reorder_and_row_filter(recorded_stats):
    rows = [
        _trajectory(1.0, 2),
        _trajectory(0.0, 7),
        _trajectory(0.0, 4, {"arena_status": "HARNESS_FAILED"}),
    ]
    joined = concat_padded_tensors(rows)
    selected = {key: value[[2, 0]] for key, value in joined.items()}
    split = split_and_unpad_tensor(selected, n_trajs=2, traj_group_sizes=[1, 1])

    record_rollout_turn_stats(split)
    metrics = recorded_stats["train"].export()

    assert metrics["train/num_turns_count"] == 2
    assert metrics["train/num_turns/avg"] == 3
    assert metrics["train/num_turns_no_harness_err/avg"] == 2
    assert metrics["train/num_turns_harness_err/OTHER/avg"] == 4


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


def test_count_turns_includes_loss_masked_turns_but_excludes_original_padding():
    """Loss masking does not undo a generation; prompt/padding are not turns."""
    trajectory = {
        "turn_ids": torch.tensor([[-1, 2, 2, -1, 8, 8, 2, 12, 0]]),
        "loss_mask": torch.tensor([[0, 1, 1, 0, 1, 1, 1, 0, 1]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0]]),
    }

    row = with_rollout_turn_stats(trajectory, None)

    torch.testing.assert_close(
        row[ROLLOUT_TURN_COUNT], torch.tensor([3.0]), rtol=0, atol=0
    )


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


@pytest.mark.parametrize("child_input_len", [1, 3, 5])
@pytest.mark.parametrize("empty_last_response", [False, True])
@pytest.mark.parametrize("http_transport", [False, True])
def test_interaction_history_survives_tensor_fallback_and_ignores_empty_response(
    recorded_stats, child_input_len, empty_last_response, http_transport
):
    """Raw branch turns survive reset IDs; zero-token requests add no generation."""
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


@pytest.mark.parametrize("status", [None, "", " ", 123])
def test_required_harness_status_missing_is_neither_success_nor_failure(
    recorded_stats, status
):
    """Unknown V2 metadata counts only toward the overall turn distribution."""
    row = with_rollout_turn_stats(
        _trajectory(0.0, 3), {"arena_status": status}, require_harness_status=True
    )

    record_rollout_turn_stats([row])
    metrics = recorded_stats["train"].export()

    assert metrics["train/num_turns_count"] == 1
    assert metrics["train/num_turns/avg"] == 3
    assert metrics["train/num_turns_no_harness_err_count"] == 0
    assert metrics["train/num_turns_harness_err_count"] == 0


def test_original_turns_follow_row_reorder_despite_mask_edits(recorded_stats):
    """Case history and Harness labels stay aligned after token/row edits."""
    joined = concat_padded_tensors(
        [
            _trajectory(1.0, 2),
            _trajectory(0.0, 4, {"arena_status": "HARNESS_FAILED"}),
        ]
    )
    edited = {key: value[[1, 0]] for key, value in joined.items()}
    edited["loss_mask"][edited["turn_ids"] >= 1] = 0

    record_rollout_turn_stats([edited])
    metrics = recorded_stats["train"].export()

    torch.testing.assert_close(
        edited[ROLLOUT_TURN_COUNT], torch.tensor([4.0, 2.0]), rtol=0, atol=0
    )
    assert metrics["train/num_turns_no_harness_err_count"] == 1
    assert metrics["train/num_turns_harness_err/OTHER_count"] == 1
    assert metrics["train/num_turns_harness_err/OTHER/avg"] == 4
    assert metrics["train/num_turns_no_harness_err/avg"] == 2


def test_record_rejects_hook_that_drops_rows_without_slicing_metadata(recorded_stats):
    """A malformed hook must not silently misclassify remaining rows."""
    row = concat_padded_tensors([_trajectory(1.0, 2), _trajectory(0.0, 3)])
    row = {key: value[:1] if value.ndim == 2 else value for key, value in row.items()}

    with pytest.raises(ValueError, match="match the trajectory batch size"):
        record_rollout_turn_stats([row])


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


def test_zero_turn_row_is_retained(recorded_stats):
    record_rollout_turn_stats([_trajectory(0.0, 0)])

    metrics = recorded_stats["train"].export()

    assert metrics["train/num_turns_count"] == 1
    assert metrics["train/num_turns/avg"] == 0


def test_empty_selected_batch_has_zero_count_not_stale_average(recorded_stats):
    record_rollout_turn_stats([_trajectory(1.0, 5)])
    recorded_stats["train"].export()

    record_rollout_turn_stats([])
    metrics = recorded_stats["train"].export()

    assert metrics["train/num_turns_count"] == 0
    assert "train/num_turns/avg" not in metrics


def test_turn_stats_fetch_only_compact_columns(recorded_stats, monkeypatch):
    from areal.infra.rpc.rtensor import RTensor

    count_handle, outcome_handle = object(), object()
    large_handle = SimpleNamespace(shape=(1, 128000))
    row = {
        ROLLOUT_TURN_COUNT: count_handle,
        ROLLOUT_HARNESS_OUTCOME: outcome_handle,
        "turn_ids": large_handle,
        "input_ids": large_handle,
    }
    fetched = []

    def localize(payload):
        fetched.append(payload)
        return [
            {
                ROLLOUT_TURN_COUNT: torch.tensor([3.0]),
                ROLLOUT_HARNESS_OUTCOME: torch.tensor([0], dtype=torch.int32),
            }
        ]

    monkeypatch.setattr(RTensor, "localize", localize)
    record_rollout_turn_stats([row])

    assert fetched == [
        [{ROLLOUT_TURN_COUNT: count_handle, ROLLOUT_HARNESS_OUTCOME: outcome_handle}]
    ]
    assert recorded_stats["train"].export()["train/num_turns/avg"] == 3


def test_workflow_without_turn_metadata_does_not_emit_turn_metrics(recorded_stats):
    row = {"rewards": torch.tensor([1.0])}

    assert with_rollout_turn_stats(row, None) is row
    record_rollout_turn_stats([row])

    assert not recorded_stats


@pytest.mark.parametrize("code", sorted(HARNESS_OUTCOME_METRIC_CODES))
def test_each_known_harness_code_keeps_original_metric_name(recorded_stats, code):
    row = _trajectory(
        0.0, 3, {"arena_status": "HARNESS_FAILED", "harness_outcome_code": code}
    )

    record_rollout_turn_stats([row])
    metrics = recorded_stats["train"].export()

    assert metrics[f"train/num_turns_harness_err/{code}_count"] == 1
    assert metrics[f"train/num_turns_harness_err/{code}/avg"] == 3
    assert metrics["train/num_turns_no_harness_err_count"] == 0


def test_task_failure_without_harness_error_stays_in_filtered_turns(recorded_stats):
    record_rollout_turn_stats(
        [_trajectory(0.0, 4, {"arena_status": "DONE", "harness_outcome_code": "OTHER"})]
    )

    metrics = recorded_stats["train"].export()

    assert metrics["train/num_turns_no_harness_err_count"] == 1
    assert metrics["train/num_turns_harness_err_count"] == 0


@pytest.mark.parametrize("misalignment", ["missing_outcome", "outcome_shape", "rows"])
def test_misaligned_turn_metadata_is_rejected(recorded_stats, misalignment):
    row = _trajectory(1.0, 2)
    if misalignment == "missing_outcome":
        row.pop(ROLLOUT_HARNESS_OUTCOME)
    elif misalignment == "outcome_shape":
        row[ROLLOUT_HARNESS_OUTCOME] = torch.zeros(2, dtype=torch.int32)
    else:
        row["input_ids"] = torch.ones((2, 5), dtype=torch.int32)

    with pytest.raises(ValueError, match="rollout turn"):
        record_rollout_turn_stats([row])

    assert not recorded_stats


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
