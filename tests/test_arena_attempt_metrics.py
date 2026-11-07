# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from examples.swe.arena_agent import ArenaStreamAgentWorkflow
from examples.swe.arena_client import ArenaTaskFailedError, ArenaTaskResult

from areal.utils.rollout_metrics import ConsumedMetrics, capture_metrics


@pytest.fixture
def workflow(monkeypatch):
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "test-token")
    monkeypatch.setenv("ARENA_LLM_API_KEY", "test-key")
    agent = ArenaStreamAgentWorkflow(
        econfig={"arena_base_url": "https://arena.example"}
    )
    monkeypatch.setattr(
        agent.client,
        "register_llm_proxy_async",
        AsyncMock(return_value=("http://arena.example/proxy", "model-1")),
    )
    monkeypatch.setattr(agent.client, "delete_llm_proxy_async", AsyncMock())
    monkeypatch.setattr(agent, "_dump_task_result", AsyncMock())
    return agent


async def run_agent(workflow, data_id="data-1"):
    async with httpx.AsyncClient(trust_env=False) as client:
        return await workflow.run(
            {"stream_id": "stream-1", "data_id": data_id},
            base_url="http://rollout-proxy",
            api_key="session-key",
            arena_http_client=client,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["EVAL_FAILED", "HARNESS_FAILED", "SETUP_FAILED"])
@pytest.mark.parametrize("with_result", [False, True])
async def test_rejected_terminal_failure_is_counted_without_export(
    workflow, monkeypatch, status, with_result
):
    result = ArenaTaskResult(task_id="task-1", status=status, score=None)
    error = ArenaTaskFailedError(
        "task-1", status, result=result if with_result else None
    )
    monkeypatch.setattr(
        workflow.client, "launch_one_task_result", AsyncMock(side_effect=error)
    )
    with capture_metrics() as metrics:
        with pytest.raises(ArenaTaskFailedError) as raised:
            await run_agent(workflow)
    assert raised.value is error
    assert metrics["rollout/task_attempt/error"] == [1.0]
    assert metrics["rollout/task_error"] == [1.0]
    assert metrics[f"rollout/task_status/{status}"] == [1.0]
    assert metrics["rollout/harness_error"] == [float(status == "HARNESS_FAILED")]
    assert "rollout/training_score" not in metrics


@pytest.mark.asyncio
async def test_recovered_harness_failure_export_does_not_double_count(
    workflow, monkeypatch
):
    result = ArenaTaskResult(
        task_id="task-1",
        status="HARNESS_FAILED",
        score=None,
        raw={"outcome_code": "AGENT_MAX_TURNS_EXCEEDED"},
    )
    error = ArenaTaskFailedError("task-1", result.status, result=result)
    monkeypatch.setattr(
        workflow.client, "launch_one_task_result", AsyncMock(side_effect=error)
    )
    data = {"stream_id": "stream-1", "data_id": "data-1"}
    with capture_metrics() as metrics:
        with pytest.raises(ArenaTaskFailedError):
            await run_agent(workflow)
        await workflow.record_failure_disposition(data, error, "model_failure_zero")
        workflow.record_episode_metrics(data, 0.0)
    assert metrics["rollout/harness_error"] == [1.0]
    assert metrics["rollout/harness_error/AGENT_MAX_TURNS_EXCEEDED"] == [1.0]
    assert metrics["rollout/task_attempt/error"] == [1.0]
    assert metrics["rollout/training_score"] == [0.0]


@pytest.mark.asyncio
async def test_tool_call_budget_exceeded_harness_failure_records_rate(
    workflow, monkeypatch
):
    summary = "GAMEAGENT_OUTCOME_CODE=LLM_RESPONSE_FAILED GAMEAGENT_OUTCOME_CLASS=agent"
    outcome = {
        "code": "LLM_RESPONSE_FAILED",
        "class": "agent",
        "source": "core_tool_decoder",
        "details": {
            "code": "tool_call_budget_exceeded",
            "budget": "argument_bytes",
            "limit": 65536,
            "observed": 94917,
        },
    }
    message = "tool_call_budget_exceeded: argument_bytes (limit 65536, observed 94917)"
    result = ArenaTaskResult(
        task_id="task-1",
        status="HARNESS_FAILED",
        score=0.0,
        raw={
            "error": f"harness: running setup hook (timeout=10m0s)\n{summary}",
            "harness": {
                "phase": "agent",
                "result_status": "ERROR",
                "exit_code": 1,
                "summary": summary,
            },
            "outcome": outcome,
        },
    )
    error = ArenaTaskFailedError("task-1", result.status, result=result)
    error.harness_result = {
        "implementation": "rust-core-runtime",
        "status": "ERROR",
        "exit_code": 1,
        "turn_statuses": ["failed"],
        "delivery": None,
        "summary": summary,
        "error": [{"message": message, "outcome": outcome.copy()}],
        "raw": {"outcome": outcome.copy()},
    }
    monkeypatch.setattr(
        workflow.client, "launch_one_task_result", AsyncMock(side_effect=error)
    )
    monkeypatch.setattr(
        workflow.client,
        "get_harness_result_async",
        AsyncMock(return_value=error.harness_result),
    )

    with capture_metrics() as metrics:
        with pytest.raises(ArenaTaskFailedError):
            await run_agent(workflow)

    assert metrics["rollout/harness_error"] == [1.0]
    assert metrics["rollout/harness_error/tool_call_budget_exceeded_rate"] == [1.0]


@pytest.mark.asyncio
async def test_other_harness_failure_records_zero_tool_call_budget_rate(
    workflow, monkeypatch
):
    error = ArenaTaskFailedError(
        "task-1",
        "HARNESS_FAILED",
        result=ArenaTaskResult(
            task_id="task-1", status="HARNESS_FAILED", score=0.0, raw={}
        ),
    )
    monkeypatch.setattr(
        workflow.client, "launch_one_task_result", AsyncMock(side_effect=error)
    )

    with capture_metrics() as metrics:
        with pytest.raises(ArenaTaskFailedError):
            await run_agent(workflow)

    assert metrics["rollout/harness_error/tool_call_budget_exceeded_rate"] == [0.0]


@pytest.mark.asyncio
async def test_structured_budget_metric_requires_matching_receipt(
    workflow, monkeypatch
):
    outcome = {
        "code": "LLM_RESPONSE_FAILED",
        "class": "agent",
        "source": "core_tool_decoder",
        "details": {
            "code": "tool_call_budget_exceeded",
            "budget": "argument_bytes",
            "limit": 100,
            "observed": 101,
        },
    }
    summary = "GAMEAGENT_OUTCOME_CODE=LLM_RESPONSE_FAILED GAMEAGENT_OUTCOME_CLASS=agent"
    result = ArenaTaskResult(
        task_id="task-1",
        status="HARNESS_FAILED",
        score=0.0,
        raw={
            "error": f"harness: running setup hook (timeout=10m0s)\n{summary}",
            "harness": {
                "phase": "agent",
                "result_status": "ERROR",
                "exit_code": 1,
                "summary": summary,
            },
            "outcome": outcome,
        },
    )
    error = ArenaTaskFailedError("task-1", result.status, result=result)
    error.harness_result = {
        "implementation": "rust-core-runtime",
        "status": "ERROR",
        "exit_code": 1,
        "turn_statuses": ["failed"],
        "delivery": None,
        "summary": summary,
        "error": [
            {
                "message": "tool_call_budget_exceeded: argument_bytes (limit 100, observed 101)",
                "outcome": outcome.copy(),
            }
        ],
        "raw": {"outcome": outcome.copy()},
    }
    monkeypatch.setattr(
        workflow.client, "launch_one_task_result", AsyncMock(side_effect=error)
    )
    monkeypatch.setattr(
        workflow.client,
        "get_harness_result_async",
        AsyncMock(return_value=error.harness_result),
    )

    with capture_metrics() as metrics:
        with pytest.raises(ArenaTaskFailedError):
            await run_agent(workflow)

    assert metrics["rollout/harness_error/tool_call_budget_exceeded_rate"] == [1.0]

    error.harness_result["raw"]["outcome"]["details"]["observed"] = 102
    monkeypatch.setattr(
        workflow.client, "launch_one_task_result", AsyncMock(side_effect=error)
    )
    monkeypatch.setattr(
        workflow.client,
        "get_harness_result_async",
        AsyncMock(return_value=error.harness_result),
    )
    with capture_metrics() as metrics:
        with pytest.raises(ArenaTaskFailedError):
            await run_agent(workflow)
    assert metrics["rollout/harness_error/tool_call_budget_exceeded_rate"] == [0.0]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["DONE", "EVAL_FAILED"])
async def test_non_harness_terminal_task_records_zero_tool_call_budget_rate(
    workflow, monkeypatch, status
):
    result = ArenaTaskResult(
        task_id="task-1", status=status, score=0.5 if status == "DONE" else None
    )
    if status == "DONE":
        launch = AsyncMock(return_value=result)
        expected_status_metric = "rollout/task_status/DONE"
    else:
        launch = AsyncMock(
            side_effect=ArenaTaskFailedError("task-1", status, result=result)
        )
        expected_status_metric = "rollout/task_status/EVAL_FAILED"
    monkeypatch.setattr(workflow.client, "launch_one_task_result", launch)

    with capture_metrics() as metrics:
        if status == "DONE":
            await run_agent(workflow)
        else:
            with pytest.raises(ArenaTaskFailedError):
                await run_agent(workflow)

    assert metrics[expected_status_metric] == [1.0]
    assert metrics["rollout/harness_error/tool_call_budget_exceeded_rate"] == [0.0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [RuntimeError("launch failed"), asyncio.CancelledError()]
)
async def test_local_failure_does_not_invent_terminal_result(
    workflow, monkeypatch, error
):
    monkeypatch.setattr(
        workflow.client, "launch_one_task_result", AsyncMock(side_effect=error)
    )
    with capture_metrics() as metrics:
        with pytest.raises(type(error)):
            await run_agent(workflow)
    assert metrics["rollout/task_attempt/success"] == [0.0]
    assert metrics["rollout/task_attempt/cancelled"] == [
        float(isinstance(error, asyncio.CancelledError))
    ]
    assert "rollout/harness_error" not in metrics


@pytest.mark.asyncio
async def test_concurrent_rejected_group_preserves_terminal_denominator(
    workflow, monkeypatch
):
    async def launch(**kwargs):
        # Let calls overlap to exercise the shared workflow's task-local state.
        await asyncio.sleep(0)
        data_id = kwargs["data_id"]
        if data_id == "success":
            return ArenaTaskResult(task_id=data_id, status="DONE", score=0.5)
        if data_id == "cancelled":
            raise asyncio.CancelledError()
        raise ArenaTaskFailedError(data_id, data_id)

    monkeypatch.setattr(workflow.client, "launch_one_task_result", launch)
    with capture_metrics() as metrics:
        await asyncio.gather(
            *(
                run_agent(workflow, name)
                for name in ["success", "HARNESS_FAILED", "EVAL_FAILED", "cancelled"]
            ),
            return_exceptions=True,
        )
    consumed = ConsumedMetrics()
    assert consumed.consume(SimpleNamespace(trajectory=None, metrics=metrics)) is None
    stats = consumed.export()
    assert stats["rollout/task_attempt_count"] == 4
    assert stats["rollout/task_result_count"] == 3
    assert stats["rollout/task_attempt/cancelled"] == 0.25
    assert stats["rollout/harness_error"] == pytest.approx(1 / 3)
    assert stats["rollout/task_error"] == pytest.approx(2 / 3)
    assert "rollout/reward_sample_count" not in stats
    assert consumed.export() == {}
