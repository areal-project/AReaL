# SPDX-License-Identifier: Apache-2.0

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from examples.swe.arena_agent import ArenaStreamAgentWorkflow
from examples.swe.arena_client import (
    SELFCHECK_BUDGET_KEY,
    SELFCHECK_CANDIDATE_CHANGED,
    ArenaTaskFailedError,
    ArenaTaskResult,
)
from tests.test_selfcheck_metrics import _read_checks
from tests.test_swe_arena_abbreviated_receipt import _error as _runner_failure_error
from tests.test_swe_arena_multi_stream import (
    _core_model_failure_error,
    _core_tool_budget_failure_error,
)

from areal.infra.controller.rollout_controller import _merge_worker_stats
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
@pytest.mark.parametrize("enabled", [False, True])
async def test_stream_selfcheck_controls_artifact_reads_and_metrics(
    workflow, monkeypatch, enabled
):
    name = workflow.default_stream_name
    assert workflow.stream_configs[name].self_check is False
    workflow.stream_configs[name] = replace(
        workflow.stream_configs[name], self_check=enabled
    )
    telemetry = {
        "calls": 1.0,
        "check_1/score": 1.0,
        SELFCHECK_BUDGET_KEY: 3.0,
        SELFCHECK_CANDIDATE_CHANGED: 1.0,
    }
    result = ArenaTaskResult(
        task_id="task-1",
        status="DONE",
        score=1.0,
        artifacts_uri="tasks/task-1/attempts/1/",
        selfcheck_metrics=telemetry,
    )
    reader = AsyncMock(return_value=telemetry)
    monkeypatch.setattr(workflow.client, "get_selfcheck_metrics", reader)
    monkeypatch.setattr(
        workflow.client, "launch_one_task_result", AsyncMock(return_value=result)
    )
    with capture_metrics() as metrics:
        assert await run_agent(workflow) == 1.0
        workflow.record_episode_metrics(
            {"stream_id": "stream-1", "data_id": "data-1"}, reward=1.0
        )
    assert reader.await_count == int(enabled)
    checker_keys = {key for key in metrics if "/selfcheck/" in key}
    if enabled:
        assert "rollout/selfcheck/calls" in checker_keys
        assert f"rollout/stream/{name}/selfcheck/calls" in checker_keys
    else:
        assert not checker_keys
    assert metrics["rollout/arena_score"] == [1.0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "used_calls,max_calls", [(0, 11), (1, 11), (11, 11), (1, 10**400), (11, None)]
)
async def test_selfcheck_large_budget_preserves_known_calls_and_candidate_change(
    workflow, used_calls, max_calls
):
    """Large budgets retain exact counts without creating unbounded score slots."""
    name = workflow.default_stream_name
    workflow.stream_configs[name] = replace(
        workflow.stream_configs[name], self_check=True
    )
    budget = {"used_calls": used_calls}
    if max_calls is not None:
        budget["max_calls"] = max_calls
    telemetry = await _read_checks(
        [
            {"sha256": "a" * 64, "response": {"status": "OK", "score": 1}}
            for _ in range(used_calls)
        ],
        budget,
        {"phase": "agent", "sha256": "b" * 64},
    )
    assert not any(key.startswith("check_") for key in telemetry)
    workflow._task_result.set(
        ArenaTaskResult(
            task_id="task", status="DONE", score=1.0, selfcheck_metrics=telemetry
        )
    )
    with capture_metrics() as metrics:
        workflow.record_episode_metrics(
            {"stream_id": "stream-1", "data_id": "data-1"}, reward=1.0
        )
    consumed = ConsumedMetrics()
    consumed.consume(SimpleNamespace(trajectory={}, metrics=metrics))
    stats = _merge_worker_stats([consumed.export()])

    for prefix in ("rollout/selfcheck", f"rollout/stream/{name}/selfcheck"):
        assert stats[f"{prefix}/unknown_calls_count"] == 0
        group = f"{prefix}/reward_1"
        assert stats[f"{group}/mean_calls"] == used_calls
        assert stats[f"{group}/trajectory_count"] == 1
        assert stats[f"{group}/exact_calls_{used_calls}/count"] == 1
        assert stats[f"{group}/exact_calls_{used_calls}/ratio"] == 1
        assert (
            sum(
                value
                for key, value in stats.items()
                if key.startswith(f"{group}/exact_calls_") and key.endswith("/count")
            )
            == 1
        )
        assert f"{group}/exact_calls_12/count" not in stats
        if used_calls:
            assert stats[f"{prefix}/final_candidate/changed_count"] == 1
        else:
            assert f"{prefix}/final_candidate/changed_count" not in stats


def test_selfcheck_distribution_merges_consumed_final_reward_groups(workflow):
    name = workflow.default_stream_name
    workflow.stream_configs[name] = replace(
        workflow.stream_configs[name], self_check=True
    )
    data = {"stream_id": "stream-1", "data_id": "data-1"}
    prefix = f"rollout/stream/{workflow.default_stream_name}/selfcheck"

    def record(
        consumer,
        score,
        calls=None,
        budget=None,
        status="DONE",
        episode_data=data,
        changed=None,
    ):
        selfcheck = (
            {} if calls is None else {"calls": calls, SELFCHECK_BUDGET_KEY: budget}
        )
        if changed is not None:
            selfcheck[SELFCHECK_CANDIDATE_CHANGED] = float(changed)
        workflow._task_result.set(
            ArenaTaskResult(
                task_id="task", status=status, score=score, selfcheck_metrics=selfcheck
            )
        )
        with capture_metrics() as metrics:
            workflow.record_episode_metrics(episode_data, reward=0.5)
        episode = SimpleNamespace(trajectory={}, metrics=metrics)
        if consumer is not None:
            consumer.consume(episode)

    left, right = ConsumedMetrics(), ConsumedMetrics()
    record(left, 1.0, 0.0, 0.0)
    record(left, 1.0, 1.0, 2.0)
    record(left, 1.0)
    record(left, 0.0, 2.0, 2.0)
    record(right, 1.0, 4.0, 4.0)
    record(right, 1.0, 1.0, 4.0)
    record(right, 0.0)
    record(right, 0.0, 4.0, 4.0)
    record(right, 0.0, status="HARNESS_FAILED")
    record(right, 0.75, 2.0, 4.0)
    record(None, 1.0, 3.0, 4.0)

    left_stats, right_stats = left.export(), right.export()
    assert left_stats[f"{prefix}/reward_1/exact_calls_0/ratio"] == 0.5
    stats = _merge_worker_stats([left_stats, right_stats])
    with_local_stats = _merge_worker_stats([left_stats], consumed_stats=right_stats)
    assert {
        key: value
        for key, value in with_local_stats.items()
        if key.startswith(prefix) and not key.endswith("__count")
    } == {key: value for key, value in stats.items() if key.startswith(prefix)}
    assert stats[f"{prefix}/unknown_calls_count"] == 2
    assert stats[f"{prefix}/reward_0/trajectory_count"] == 2
    assert stats[f"{prefix}/reward_0/mean_calls"] == 3.0
    assert stats[f"{prefix}/reward_0/exact_calls_4/ratio"] == 0.5
    assert stats[f"{prefix}/reward_1/trajectory_count"] == 4
    assert stats[f"{prefix}/reward_1/mean_calls"] == 1.5
    for count, expected in enumerate((1, 2, 0, 0, 1)):
        bucket = f"{prefix}/reward_1/exact_calls_{count}"
        assert stats[f"{bucket}/count"] == expected
        assert stats[f"{bucket}/ratio"] == expected / 4
    assert left.export() == right.export() == {}

    second_stream = "second"
    workflow.stream_configs[second_stream] = replace(
        workflow.stream_configs[workflow.default_stream_name],
        name=second_stream,
        stream_id="stream-2",
    )
    second = {**data, "stream_id": "stream-2", "arena_stream_name": second_stream}
    record(left, 1.0, 0.0, 2.0)
    record(right, 1.0, 3.0, 4.0, episode_data=second, changed=False)
    record(right, 1.0, 3.0, 4.0, episode_data=second)
    record(left, 0.0, 1.0, 2.0, changed=True)
    record(right, 0.0, episode_data=second)
    workflow.stream_configs["baseline"] = replace(
        workflow.stream_configs[second_stream],
        name="baseline",
        stream_id="baseline",
        self_check=False,
    )
    record(
        right,
        1.0,
        4.0,
        4.0,
        episode_data={**data, "stream_id": "baseline", "arena_stream_name": "baseline"},
        changed=True,
    )
    merged = _merge_worker_stats([left.export()], consumed_stats=right.export())
    global_prefix = "rollout/selfcheck"
    assert merged[f"{global_prefix}/reward_1/trajectory_count"] == 3
    assert merged[f"{global_prefix}/reward_1/mean_calls"] == 2.0
    assert merged[f"{global_prefix}/reward_1/exact_calls_0/count"] == 1
    assert merged[f"{global_prefix}/reward_1/exact_calls_0/ratio"] == pytest.approx(
        1 / 3
    )
    assert merged[f"{global_prefix}/reward_1/exact_calls_3/count"] == 2
    assert merged[f"{global_prefix}/reward_1/exact_calls_4/count"] == 0
    assert merged[f"{global_prefix}/unknown_calls_count"] == 1
    assert merged[f"{global_prefix}/final_candidate/changed_count"] == 1
    assert merged[f"{prefix}/final_candidate/changed_count"] == 1
    assert merged["rollout/stream/second/selfcheck/final_candidate/changed_count"] == 0
    assert not any(key.endswith("/final_candidate/comparable_count") for key in merged)
    assert merged[f"{prefix}/reward_1/trajectory_count"] == 1
    assert merged["rollout/stream/second/selfcheck/reward_1/trajectory_count"] == 2
    assert not any("/stream/baseline/selfcheck/" in key for key in merged)


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


@pytest.mark.parametrize("receipt_kind", ["core", "review", "runner", "adopted_runner"])
@pytest.mark.parametrize(
    "fault",
    [
        None,
        "missing_receipt",
        "failed_collection",
        "malformed_collection",
        "runner_timeout",
        "external_interruption",
        "receipt_failure",
        "wrong_message",
        "eval_failed",
    ],
)
def test_budget_failure_metrics_and_metadata_require_healthy_receipt(
    workflow, receipt_kind, fault
):
    """Both metric names accept the same receipts as model-failure recovery."""
    if receipt_kind in {"core", "review"}:
        error = _core_tool_budget_failure_error(
            continuations=2 if receipt_kind == "review" else 0
        )
    else:
        error = _runner_failure_error(
            adopted=receipt_kind == "adopted_runner",
            message="tool_call_budget_exceeded: argument_bytes (limit 65536, observed 69773)",
        )
    if fault == "missing_receipt":
        error.harness_result = None
    elif fault == "failed_collection":
        error.harness_result["collection"] = {"status": "ERROR"}
    elif fault == "malformed_collection":
        error.harness_result["collection"] = None
    elif fault == "runner_timeout":
        error.harness_result["observed_runner_timeout"] = True
    elif fault == "external_interruption":
        error.harness_result["observed_external_interruption"] = True
    elif fault == "receipt_failure":
        error.harness_result["failure"] = {"code": "EXPORT_FAILED"}
    elif fault == "wrong_message":
        error.harness_result["error"][-1]["message"] = "HTTP 503 service unavailable"
    elif fault == "eval_failed":
        error.status = "EVAL_FAILED"

    workflow._task_result.set(error.result)
    with capture_metrics() as metrics:
        workflow._record_attempt_metrics({}, error)

    for name in ("tool_call_budget_exceeded_rate", "tool_call_argument_bytes_exceeded"):
        assert metrics[f"rollout/harness_error/{name}"] == [float(fault is None)]
    assert metrics["rollout/harness_error/empty_completion"] == [0.0]
    assert workflow.get_episode_metadata().get("harness_failure_reason") == (
        "tool_call_argument_bytes_exceeded" if fault is None else None
    )
    assert workflow.classify_proxy_failure(
        error, context_overflow=False, interaction_count=2
    ) == (
        "model_failure_zero"
        if fault is None
        else "system_failure_reject"
        if fault == "eval_failed"
        else "unknown_failure_reject"
    )
    # Cause metrics count the terminal Harness failure even without exportable tokens.
    assert (
        workflow.classify_proxy_failure(
            error, context_overflow=False, interaction_count=0
        )
        != "model_failure_zero"
    )


@pytest.mark.asyncio
async def test_concurrent_failure_reasons_keep_metadata_and_terminal_denominator(
    workflow, monkeypatch
):
    """Rejected failures count once, without sharing their reason across sessions."""
    errors = {
        "budget": _core_tool_budget_failure_error(),
        "empty": _core_model_failure_error(
            "LLM_RESPONSE_FAILED",
            "empty_completion",
            "model stopped without visible output or tool calls",
        ),
    }
    for name, error in errors.items():
        error.task_id = name
        error.result = replace(error.result, task_id=name)

    async def launch(**kwargs):
        await asyncio.sleep(0)
        name = kwargs["data_id"]
        if name in errors:
            raise errors[name]
        if name == "cancelled":
            raise asyncio.CancelledError()
        return ArenaTaskResult(task_id=name, status="DONE", score=0.5)

    async def receipt(task_id, **kwargs):
        return errors[task_id].harness_result

    async def run_and_check_reason(name):
        try:
            await run_agent(workflow, name)
        except (ArenaTaskFailedError, asyncio.CancelledError):
            pass
        await asyncio.sleep(0)
        expected = {
            "budget": "tool_call_argument_bytes_exceeded",
            "empty": "empty_completion",
        }.get(name)
        assert workflow.get_episode_metadata().get("harness_failure_reason") == expected
        # A later success in the same session must clear the previous reason.
        await run_agent(workflow, "success")
        assert "harness_failure_reason" not in workflow.get_episode_metadata()

    monkeypatch.setattr(workflow.client, "launch_one_task_result", launch)
    monkeypatch.setattr(workflow.client, "get_harness_result_async", receipt)
    with capture_metrics() as metrics:
        await asyncio.gather(
            *(
                run_and_check_reason(name)
                for name in ("budget", "empty", "success", "cancelled")
            )
        )

    consumed = ConsumedMetrics()
    consumed.consume(SimpleNamespace(trajectory=None, metrics=metrics))
    stats = consumed.export()
    assert stats["rollout/task_attempt_count"] == 8
    assert stats["rollout/task_result_count"] == 7
    for reason in (
        "tool_call_budget_exceeded_rate",
        "tool_call_argument_bytes_exceeded",
        "empty_completion",
    ):
        assert stats[f"rollout/harness_error/{reason}"] == pytest.approx(1 / 7)


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
