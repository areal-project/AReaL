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
    assert metrics["rollout/stream/stream-1/task_error"] == [1.0]
    assert metrics["rollout/stream/stream-1/harness_error"] == [
        float(status == "HARNESS_FAILED")
    ]
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
    assert "rollout/stream/stream-1/harness_error" not in metrics


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


@pytest.mark.asyncio
async def test_stream_attempts_keep_independent_terminal_denominators(workflow):
    """Concurrent streams retain failures even when no trajectory is accepted."""

    async def attempt(stream, status):
        result = (
            ArenaTaskResult(task_id=stream, status=status, score=1.0)
            if status == "DONE"
            else None
        )
        error = (
            asyncio.CancelledError()
            if status == "cancelled"
            else ArenaTaskFailedError(stream, status)
            if result is None
            else None
        )
        workflow._task_result.set(result)
        await asyncio.sleep(0)
        workflow._record_attempt_metrics({"arena_stream_name": stream}, error)

    with capture_metrics() as metrics:
        await asyncio.gather(
            attempt("a", "DONE"),
            attempt("a", "HARNESS_FAILED"),
            attempt("b", "EVAL_FAILED"),
            attempt("b", "cancelled"),
        )
    consumed = ConsumedMetrics()
    consumed.consume(SimpleNamespace(trajectory=None, metrics=metrics))
    stats = consumed.export()
    assert stats["rollout/stream/a/harness_error"] == 0.5
    assert stats["rollout/stream/a/task_error"] == 0.5
    assert stats["rollout/stream/a/task_result_count"] == 2
    assert stats["rollout/stream/b/harness_error"] == 0
    assert stats["rollout/stream/b/task_error"] == 1
    assert stats["rollout/stream/b/task_result_count"] == 1
    assert stats["rollout/stream/b/task_attempt_count"] == 2
    assert stats["rollout/task_result_count"] == 3
