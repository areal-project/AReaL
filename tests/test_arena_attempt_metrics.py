# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections import Counter
from dataclasses import asdict, replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from omegaconf import OmegaConf

from examples.swe.arena_agent import ArenaStreamAgentWorkflow, _error_reason_bucket
from examples.swe.arena_client import (
    ArenaAPIError,
    ArenaTaskFailedError,
    ArenaTaskResult,
)
from examples.swe.utils import SWEPPOConfig
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


def test_auxiliary_request_policy_is_explicit_and_copied(workflow):
    workflow.gen_args = {"temperature": 0.7}
    assert workflow.get_session_metadata({}) == {
        "generation_args": {"temperature": 0.7}
    }
    prompts = ["Generate a session title."]
    workflow.econfig["arena_auxiliary_system_prompts"] = prompts
    metadata = workflow.get_session_metadata({})
    assert metadata["auxiliary_system_prompts"] == prompts
    metadata["auxiliary_system_prompts"].clear()
    assert prompts == ["Generate a session title."]


@pytest.mark.parametrize("prompts", [None, ["Title"]])
def test_auxiliary_policy_loads_through_training_yaml(
    tmp_path, monkeypatch, workflow, prompts
):
    from areal.api import cli_args
    from areal.utils.stats_logger import StatsLogger

    cfg = OmegaConf.structured(SWEPPOConfig)
    for key in OmegaConf.missing_keys(cfg):
        OmegaConf.update(cfg, key, "test")
    data = OmegaConf.to_container(cfg, resolve=True)
    data["econfig"].pop("arena_auxiliary_system_prompts", None)
    if prompts is not None:
        data["econfig"]["arena_auxiliary_system_prompts"] = prompts
    path = tmp_path / "experiment.yaml"
    OmegaConf.save(OmegaConf.create(data), path)
    monkeypatch.setattr(cli_args.name_resolve, "reconfigure", lambda config: None)
    monkeypatch.setattr(cli_args, "save_config", lambda config, path: None)
    monkeypatch.setattr(
        StatsLogger, "get_log_path", staticmethod(lambda config: str(tmp_path))
    )

    loaded, _ = cli_args.load_expr_config(["--config", str(path)], SWEPPOConfig)
    assert loaded.econfig.arena_auxiliary_system_prompts == (prompts or [])
    workflow.econfig = asdict(loaded.econfig)
    metadata = workflow.get_session_metadata({})
    assert metadata.get("auxiliary_system_prompts", []) == (prompts or [])


@pytest.mark.parametrize("prompts", ["Title request", [""], [123]])
def test_invalid_auxiliary_request_policy_is_rejected(workflow, prompts):
    workflow.econfig["arena_auxiliary_system_prompts"] = prompts
    with pytest.raises(ValueError, match="nonempty strings"):
        workflow.get_session_metadata({})


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
    assert metrics["rollout/error_rate/model"] == [1.0]
    assert metrics["rollout/error_rate/model_reason/max_turns_exceeded"] == [1.0]
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


def _api_failure(reason):
    error = ArenaAPIError(reason)
    error.arena_failure_reason = reason
    return error


def _terminal_failure(status, raw=None):
    result = ArenaTaskResult(task_id="task", status=status, score=None, raw=raw)
    return ArenaTaskFailedError(result.task_id, status, result=result)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ({"arca": {"http_status": 503}}, "arca_http_503"),
        ({"error": "reward exec (exec_eof): EOF"}, "arca_exec_eof"),
        (
            {"error": "shared reward completion (exec_timeout): timeout"},
            "arca_exec_timeout",
        ),
        ({"error": "adopted reward (http_5xx): failed"}, "arca_http_5xx"),
        ({"error": "harness: model printed reward exec (exec_eof): EOF"}, None),
        ({"error": "api_error"}, None),
        ({"arca": {"http_status": 403}}, None),
        (
            {"arca": {"http_status": 403, "operation": "exec", "success": False}},
            "arca_http_403",
        ),
        ({"arca": {"http_status": 403, "operation": "exec", "success": True}}, None),
        ({"arca": {"http_status": 403, "operation": "", "success": False}}, None),
        ({"arca": None}, None),
        (None, None),
    ],
)
def test_arena_attribution_requires_gateway_evidence(raw, expected):
    error = _terminal_failure("FAILED", raw)
    assert ArenaStreamAgentWorkflow._arena_failure_evidence(error) == expected


@pytest.mark.parametrize(
    "status,expected_origin",
    [
        ("SETUP_FAILED", "arena"),
        ("COLLECT_FAILED", "arena"),
        ("EVAL_FAILED", "arena"),
        ("FAILED", "arena"),
        ("ERROR", "arena"),
        ("HARNESS_FAILED", "unknown"),
        ("TIMEOUT", "unknown"),
        ("NO_OUTPUT", "unknown"),
        ("CANCELLED", "unknown"),
    ],
)
def test_terminal_status_classifies_one_rollout(
    workflow, monkeypatch, status, expected_origin
):
    audit = Mock()
    monkeypatch.setattr("examples.swe.arena_agent.logger.warning", audit)
    error = _terminal_failure(status, {"harness": {"result_status": "ERROR"}})
    with capture_metrics() as metrics:
        workflow._record_attempt_metrics({}, error)
    for origin in ("arena", "model", "unknown"):
        assert metrics[f"rollout/error_rate/{origin}"] == [
            float(origin == expected_origin)
        ]
    if status == "ERROR":
        assert audit.call_args.args[-1] == "grader_reported_error"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["HARNESS_FAILED", "TIMEOUT", "NO_OUTPUT"])
@pytest.mark.parametrize(
    "reason", ["empty_completion", "tool_call_argument_bytes_exceeded"]
)
async def test_verified_model_receipt_classifies_ambiguous_terminal_status(
    workflow, monkeypatch, status, reason
):
    error = (
        _core_tool_budget_failure_error()
        if reason == "tool_call_argument_bytes_exceeded"
        else _core_model_failure_error(
            "LLM_RESPONSE_FAILED",
            "empty_completion",
            "model stopped without visible output or tool calls",
        )
    )
    result = replace(error.result, status=status)
    terminal = ArenaTaskFailedError(error.task_id, status, result=result)
    monkeypatch.setattr(
        workflow.client,
        "get_harness_result_async",
        AsyncMock(return_value=error.harness_result),
    )
    monkeypatch.setattr(
        workflow.client, "launch_one_task_result", AsyncMock(side_effect=terminal)
    )
    with capture_metrics() as metrics:
        with pytest.raises(ArenaTaskFailedError):
            await run_agent(workflow)
    assert metrics[f"rollout/error_rate/model_reason/{reason}"] == [1.0]
    assert metrics["rollout/error_rate/model_reason/other_model_error"] == [0.0]
    assert metrics["rollout/error_rate/model"] == [1.0]
    assert (
        metrics["rollout/error_rate/arena"]
        == metrics["rollout/error_rate/unknown"]
        == [0.0]
    )


def _assert_batch_error_rates(stats, scope, reasons):
    """Check the public rates against one expected failure reason per attempt."""
    prefix = f"{scope}/error_rate/"
    rates = {
        key.removeprefix(prefix): value
        for key, value in stats.items()
        if key.startswith(prefix) and not key.endswith("__count")
    }
    assert stats[f"{scope}/task_attempt_count"] == len(reasons)
    assert not any(key.endswith("_count") for key in rates)
    counts = Counter(reason for reason in reasons if reason is not None)
    observed = {key: rate for key, rate in rates.items() if "_reason/" in key and rate}
    assert observed == pytest.approx(
        {reason: count / len(reasons) for reason, count in counts.items()}
    )
    for origin in ("arena", "model", "unknown"):
        total = sum(
            rate for key, rate in rates.items() if key.startswith(f"{origin}_reason/")
        )
        assert rates[origin] == pytest.approx(total)
    assert sum(
        rates[origin] for origin in ("arena", "model", "unknown")
    ) == pytest.approx(stats[f"{scope}/task_attempt/error"])


@pytest.mark.asyncio
async def test_reason_rates_partition_consumed_origins(workflow, monkeypatch):
    audit = Mock()
    monkeypatch.setattr("examples.swe.arena_agent.logger.warning", audit)
    refused_exec = {"arca": {"operation": "exec", "success": False, "http_status": 403}}
    attempts = [
        ("a", None, None),
        ("a", asyncio.CancelledError(), None),
        ("a", ArenaTaskFailedError("t", "SETUP_FAILED"), "arena_reason/setup_failed"),
        ("b", _api_failure("spanner_apigwmoe_502"), "arena_reason/openapi_error_502"),
        ("a", _api_failure("arena_openapi_http_502"), "arena_reason/openapi_error_502"),
        (
            "b",
            _api_failure("arena_openapi_ReadTimeout"),
            "arena_reason/openapi_network_error",
        ),
        (
            "a",
            _terminal_failure("TIMEOUT", {"arca": {"http_status": 503}}),
            "arena_reason/sandbox_error",
        ),
        (
            "b",
            _terminal_failure("HARNESS_FAILED", refused_exec),
            "arena_reason/sandbox_error",
        ),
        ("b", ArenaAPIError("invalid response payload"), "unknown_reason/client_error"),
        (
            "a",
            _terminal_failure(
                "HARNESS_FAILED", {"outcome_code": "AGENT_MAX_TURNS_EXCEEDED"}
            ),
            "model_reason/max_turns_exceeded",
        ),
        (
            "b",
            _terminal_failure(
                "HARNESS_FAILED", {"outcome_code": "AUTONOMOUS_INCOMPLETE_NO_SHIP"}
            ),
            "model_reason/other_model_error",
        ),
        ("a", ArenaTaskFailedError("t", "NO_OUTPUT"), "unknown_reason/no_output"),
        (
            "a",
            _core_tool_budget_failure_error(),
            "model_reason/tool_call_argument_bytes_exceeded",
        ),
        ("b", ArenaTaskFailedError("t", "TIMEOUT"), "unknown_reason/timeout"),
        ("b", ValueError("unexpected-private-detail"), "unknown_reason/other"),
    ]
    consumed = ConsumedMetrics()
    for stream, error, selected in attempts:
        monkeypatch.setattr(
            workflow, "_run_task", AsyncMock(side_effect=error, return_value=0.0)
        )
        data = {"arena_stream_name": stream}
        with capture_metrics() as metrics:
            if error is None:
                assert await workflow.run(data) == 0.0
            else:
                with pytest.raises(type(error)):
                    await workflow.run(data)
        assert metrics["rollout/task_attempt/cancelled"] == [
            float(isinstance(error, asyncio.CancelledError))
        ]
        assert metrics["rollout/task_attempt/success"] == [float(error is None)]
        consumed.consume(SimpleNamespace(metrics=metrics, trajectory=None))

    logged_reasons = {call.args[-1] for call in audit.call_args_list}
    assert {
        "spanner_apigwmoe_502",
        "arena_openapi_http_502",
        "arena_openapi_ReadTimeout",
        "arca_http_503",
    } <= logged_reasons
    stats = consumed.export()
    for scope, observations in (
        ("rollout", attempts),
        ("rollout/stream/a", [a for a in attempts if a[0] == "a"]),
        ("rollout/stream/b", [a for a in attempts if a[0] == "b"]),
    ):
        _assert_batch_error_rates(
            stats, scope, [selected for _, _, selected in observations]
        )
    assert not any("private-detail" in key for key in stats)
    assert consumed.export() == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "responses,action,retries,expected_reason",
    [
        ([503], "launch_deadline", 3, "arena_reason/openapi_error_503"),
        ([403], "error", 0, "arena_reason/openapi_error_403"),
        ([401], "error", 0, "arena_reason/openapi_error_401"),
        ([418], "error", 0, "arena_reason/openapi_error_other"),
        ([429], "deadline", 0, "arena_reason/openapi_error_429"),
        (["read_timeout"], "deadline", 3, "arena_reason/openapi_network_error"),
        (["connect_error"], "deadline", 0, "arena_reason/openapi_network_error"),
        (["read_timeout", "DONE"], "finish", 0, None),
        (
            ["read_timeout", "RUNNING"],
            "deadline",
            0,
            "unknown_reason/local_wait_timeout",
        ),
        (["local_protocol"], "error", 0, "unknown_reason/client_error"),
        (["spanner_403"], "deadline", 3, "arena_reason/openapi_error_403"),
        (["spanner_403", "DONE"], "finish", 0, None),
        ([503, 503], "deadline", 0, "arena_reason/openapi_error_503"),
        ([503], "deadline", 3, "arena_reason/openapi_error_503"),
        ([503, "RUNNING"], "deadline", 0, "unknown_reason/local_wait_timeout"),
        ([503, "read_error"], "deadline", 0, "arena_reason/openapi_network_error"),
        ([503, "DONE"], "finish", 0, None),
        ([503], "cancel", 0, None),
    ],
)
async def test_poll_outage_attribution_at_attempt_boundary(
    workflow, monkeypatch, responses, action, retries, expected_reason
):
    # Expire the workflow's actual deadline on the selected transport observation,
    # including when the client is about to sleep inside its request retry loop.
    deadline = asyncio.timeout(None)
    real_timeout = asyncio.timeout
    monkeypatch.setattr(
        "examples.swe.arena_agent.asyncio.timeout",
        lambda delay: deadline if delay == workflow.timeout else real_timeout(delay),
    )
    workflow.client.request_retries = retries
    workflow.client.poll_interval = 0
    reads = 0
    launches = 0
    cancels = 0

    def handle(request):
        nonlocal reads, launches, cancels
        if request.url.path.endswith("/cancel"):
            cancels += 1
            return httpx.Response(200, json={})
        if request.method == "POST":
            launches += 1
            if action == "launch_deadline":
                deadline.reschedule(asyncio.get_running_loop().time())
                return httpx.Response(503)
            return httpx.Response(
                202, json={"job_id": "j", "task_id": "t", "status": "RUNNING"}
            )
        assert request.url.path.endswith("/tasks/t/result")
        item = responses[reads]
        reads += 1
        if reads == len(responses):
            if action == "deadline":
                deadline.reschedule(asyncio.get_running_loop().time())
            elif action == "cancel":
                asyncio.get_running_loop().call_soon(asyncio.current_task().cancel)
        if item == "spanner_403":
            return httpx.Response(403, text="spanner-http-ant-group-watch-all")
        if item == "read_timeout":
            raise httpx.ReadTimeout("read timeout", request=request)
        if item == "connect_error":
            raise httpx.ConnectError("connection refused", request=request)
        if item == "local_protocol":
            raise httpx.LocalProtocolError("invalid request", request=request)
        if item == "read_error":
            raise httpx.ReadError("connection lost", request=request)
        if isinstance(item, int):
            return httpx.Response(item)
        return httpx.Response(
            200,
            json={
                "task_id": "t",
                "status": item,
                "score": 0.5 if item == "DONE" else None,
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with capture_metrics() as metrics:
            attempt = workflow.run(
                {"stream_id": "stream", "data_id": "d"},
                base_url="http://proxy",
                api_key="test",
                arena_http_client=client,
            )
            if action == "finish":
                assert await attempt == 0.5
            elif action == "error":
                with pytest.raises(ArenaAPIError):
                    await attempt
            else:
                with pytest.raises(
                    asyncio.CancelledError if action == "cancel" else TimeoutError
                ):
                    await attempt
    if action == "launch_deadline":
        assert (launches, reads, cancels) == (1, 0, 0)
    else:
        assert (launches, reads, cancels) == (
            1,
            len(responses),
            int(action != "finish"),
        )
    consumed = ConsumedMetrics()
    consumed.consume(SimpleNamespace(metrics=metrics, trajectory=None))
    stats = consumed.export()
    assert stats["rollout/task_attempt_count"] == 1
    assert stats["rollout/task_attempt/cancelled"] == float(action == "cancel")
    assert stats["rollout/task_attempt/error"] == float(
        action in {"error", "deadline", "launch_deadline"}
    )
    for scope in ("rollout", "rollout/stream/stream"):
        _assert_batch_error_rates(stats, scope, [expected_reason])


def test_http_code_rates_merge_workers_with_different_batch_sizes(workflow):
    batches = [[None, 503, 503, 403, None, 502], [418, 599, 503]]
    workers = []
    for batch in batches:
        buffer = ConsumedMetrics()
        with capture_metrics() as metrics:
            for code in batch:
                error = None
                if code is not None:
                    error = _api_failure(f"arena_openapi_http_{code}")
                workflow._record_attempt_metrics({"stream_id": "s"}, error)
        buffer.consume(SimpleNamespace(trajectory=None, metrics=metrics))
        workers.append(buffer.export())

    merged = _merge_worker_stats(workers)
    for scope in ("rollout", "rollout/stream/s"):
        prefix = f"{scope}/error_rate/arena_reason/"
        rates = {
            key.removeprefix(prefix): value
            for key, value in merged.items()
            if key.startswith(prefix)
        }
        assert {key: rate for key, rate in rates.items() if rate} == pytest.approx(
            {
                "openapi_error_503": 3 / 9,
                "openapi_error_502": 1 / 9,
                "openapi_error_403": 1 / 9,
                "openapi_error_other": 2 / 9,
            }
        )
        assert rates["openapi_error_429"] == 0
        assert "openapi_error_418" not in rates and "openapi_error_599" not in rates
        assert sum(rates.values()) == pytest.approx(merged[f"{scope}/error_rate/arena"])


@pytest.mark.parametrize(
    "origin,detail,reason",
    [
        ("arena", "spanner_apigwmoe_502", "openapi_error_502"),
        ("arena", "spanner_arenaonlinegateway_504", "openapi_error_504"),
        ("arena", "arena_openapi_http_302", "openapi_error_302"),
        ("arena", "arena_openapi_http_599", "openapi_error_other"),
        ("arena", "arena_openapi_ConnectError", "openapi_network_error"),
        ("arena", "arena_openapi_ReadTimeout", "openapi_network_error"),
        ("arena", "arca_http_503", "sandbox_error"),
        ("arena", "arca_exec_eof", "sandbox_error"),
        ("model", "LLM_CONTEXT_WINDOW_EXCEEDED", "context_window_exceeded"),
        ("model", "LLM_OUTPUT_TOKEN_LIMIT_EXCEEDED", "output_token_limit"),
        ("model", "INVALID_TOOL_JSON", "invalid_tool_json"),
        ("model", "LLM_RESPONSE_FAILED", "other_model_error"),
        ("unknown", "arena_openapi_LocalProtocolError", "client_error"),
        ("unknown", "arena_openapi_PoolTimeout", "client_error"),
        ("unknown", "ArenaAPIError", "client_error"),
        ("unknown", "RuntimeError", "other"),
    ],
)
def test_public_reason_keeps_transport_and_model_labels_bounded(origin, detail, reason):
    assert _error_reason_bucket(origin, detail) == reason
