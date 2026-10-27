# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from areal.infra import workflow_context
from areal.infra.workflow_context import WorkflowContext
from areal.infra.workflow_executor import is_consumed_rollout_metric
from areal.utils import stats_tracker
from areal.utils.episode_turns import (
    count_generated_responses,
    observe_generated_response,
    record_episode_turns,
    set_episode_turns,
    track_episode_turns,
)
from areal.utils.rollout_metrics import ConsumedMetrics
from areal.utils.rollout_stats import record_rollout_turn_stats
from areal.utils.scalar_capture import capture_scalars


@pytest.fixture(autouse=True)
def isolated_metrics():
    stats_tracker.export_all()
    workflow_context.set(WorkflowContext(is_eval=False))
    yield
    stats_tracker.export_all()


def test_rollout_counts_rejected_attempts_while_train_counts_retained_rows():
    consumer = ConsumedMetrics()
    pending = []
    for turns, accepted in [(2, True), (8, False), (5, True)]:
        with capture_scalars(select=is_consumed_rollout_metric) as metrics:
            with track_episode_turns():
                set_episode_turns(turns)
        pending.append(
            SimpleNamespace(trajectory={} if accepted else None, metrics=metrics)
        )
    assert consumer.export() == {}  # All three are still prefetched.
    consumer.record(pending[0].metrics)
    consumer.record(pending[1].metrics)
    record_rollout_turn_stats(
        [
            {
                "rollout_num_turns": torch.tensor([2.0]),
                "rollout_harness_outcome": torch.tensor([0]),
            }
        ]
    )
    rollout, train = consumer.export(), stats_tracker.export_all()
    assert rollout["rollout/num_turns_count"] == 2
    assert rollout["rollout/num_turns/avg"] == 5
    assert rollout["rollout/num_turns/max"] == 8
    assert train["train/num_turns_count"] == 1
    assert train["train/num_turns/avg"] == 2
    assert consumer.export() == {}
    consumer.record(pending[2].metrics)
    assert consumer.export()["rollout/num_turns/avg"] == 5


def test_nested_group_records_each_episode_once_including_failure():
    with capture_scalars(select=is_consumed_rollout_metric) as metrics:
        with track_episode_turns():
            with track_episode_turns():
                observe_generated_response(3)
                observe_generated_response(0)
                observe_generated_response(2)
            with pytest.raises(RuntimeError):
                with track_episode_turns():
                    observe_generated_response(1)
                    raise RuntimeError("failed after one response")
            with track_episode_turns():
                pass  # Unknown, not a zero-turn episode.
    consumer = ConsumedMetrics()
    consumer.record(metrics)
    result = consumer.export()
    assert result["rollout/num_turns_count"] == 2
    assert result["rollout/num_turns/avg"] == 1.5
    assert result["rollout/num_turns_missing_count"] == 1


def test_v2_observations_replace_outer_task_and_keep_unknown_status_separate():
    with capture_scalars(select=is_consumed_rollout_metric) as metrics:
        with track_episode_turns():
            record_episode_turns(4, require_status=True)
            record_episode_turns(
                2,
                {
                    "arena_status": "HARNESS_FAILED",
                    "harness_outcome_code": "INVALID_TOOL_JSON",
                },
                require_status=True,
            )
            record_episode_turns(None, require_status=True)
    consumer = ConsumedMetrics()
    consumer.record(metrics)
    result = consumer.export()
    assert result["rollout/num_turns_count"] == 2
    assert result["rollout/num_turns/avg"] == 3
    assert result["rollout/num_turns_harness_err_count"] == 1
    assert result["rollout/num_turns_missing_count"] == 1
    assert "rollout/num_turns_no_harness_err_count" not in result


def test_raw_cache_counts_retry_and_sibling_responses_once_before_export():
    # Raw records include ancestors as well as sibling/retry branches. Their
    # parent chains must not be added again when individual export retains both.
    records = {
        str(i): SimpleNamespace(model_response=SimpleNamespace(output_len=n))
        for i, n in enumerate([3, 2, 0, 5])
    }
    assert count_generated_responses(records) == 3
    assert count_generated_responses({}) == 0
    assert (
        count_generated_responses({"external": SimpleNamespace(model_response=None)})
        is None
    )


def test_evaluation_does_not_enter_training_capture():
    workflow_context.set(WorkflowContext(is_eval=True))
    with capture_scalars(select=is_consumed_rollout_metric) as metrics:
        with track_episode_turns():
            set_episode_turns(3)
    assert metrics == {}
    stats = stats_tracker.export_all()
    assert stats["eval-rollout/num_turns/avg"] == 3
    assert stats["eval-rollout/num_turns_count"] == 1


@pytest.mark.asyncio
async def test_builtin_multiturn_emits_one_episode_observation(monkeypatch):
    from unittest.mock import AsyncMock

    from areal.api import ModelResponse
    from areal.api.cli_args import GenerationHyperparameters
    from areal.workflow.multi_turn import MultiTurnWorkflow

    workflow = MultiTurnWorkflow.__new__(MultiTurnWorkflow)
    workflow.gconfig = GenerationHyperparameters()
    workflow.tokenizer = SimpleNamespace(eos_token_id=0, decode=lambda tokens: "text")
    workflow.max_turns = 2
    workflow.turn_discount = 1.0
    workflow.multi_turn_prompt_ids = [3]
    workflow.async_reward_fn = AsyncMock(side_effect=[0.0, 1.0])
    monkeypatch.setattr(
        "areal.workflow.multi_turn.apply_chat_template", lambda *args, **kwargs: [1]
    )

    async def generate(req):
        observe_generated_response(2)
        return ModelResponse(
            input_tokens=req.input_ids,
            output_tokens=[2, 0],
            output_logprobs=[-0.1, -0.1],
            output_versions=[0, 0],
            stop_reason="stop",
        )

    with capture_scalars(select=is_consumed_rollout_metric) as metrics:
        with track_episode_turns():
            await workflow.arun_episode(
                SimpleNamespace(agenerate=generate), {"messages": []}
            )
    consumer = ConsumedMetrics()
    consumer.record(metrics)
    result = consumer.export()
    assert result["rollout/num_turns_count"] == 1
    assert result["rollout/num_turns/avg"] == 2
    assert result["rollout/reward_sample_count"] == 1


def test_worker_merge_preserves_missing_and_empty_turn_counts():
    from areal.infra.controller.rollout_controller import _merge_worker_stats

    result = _merge_worker_stats(
        [
            {
                "eval-rollout/num_turns_missing_count": 2,
                "eval-train/num_turns_harness_err_count": 0,
            },
            {
                "eval-rollout/num_turns_missing_count": 1,
                "eval-rollout/num_turns/avg": 4,
                "eval-rollout/num_turns/min": 4,
                "eval-rollout/num_turns/max": 4,
                "eval-rollout/num_turns_count": 1,
                "eval-train/num_turns_harness_err_count": 0,
            },
        ]
    )
    assert result["eval-rollout/num_turns_missing_count"] == 3
    assert result["eval-rollout/num_turns_count"] == 1
    assert result["eval-rollout/num_turns/avg"] == 4
    assert result["eval-train/num_turns_harness_err_count"] == 0
