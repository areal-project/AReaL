"""Admission and receipt lookup for Arena's error-only runner results."""

import asyncio
from dataclasses import replace

import pytest

from examples.swe.arena_agent import ArenaStreamAgentWorkflow
from examples.swe.arena_client import ArenaTaskFailedError, ArenaTaskResult


def _error(
    *, adopted=False, message="model stopped without visible output or tool calls"
):
    prefix = "harness: adopted harness: " if adopted else "harness: "
    raw = {
        "error": prefix
        + "harness agent phase exited with code 1: startup log\n"
        + 'harness: running task lifecycle for "/envarena/env/query/Target.md" '
        + "(runner=/tmp/test-runtime/runner)\n"
        + "harness: running setup hook (timeout=3m0s)"
    }
    error = ArenaTaskFailedError(
        task_id="abbreviated-task",
        status="HARNESS_FAILED",
        result=ArenaTaskResult(
            task_id="abbreviated-task", status="HARNESS_FAILED", score=0.0, raw=raw
        ),
    )
    error.harness_result = {
        "implementation": "rust-core-runtime",
        "status": "ERROR",
        "exit_code": 1,
        "turn_statuses": ["failed"],
        "delivery": None,
        "error": [{"message": message}],
        "collection": {"status": "OK", "file_count": 4},
    }
    return error


@pytest.mark.parametrize("adopted", [False, True])
@pytest.mark.parametrize("file_count", [0, 4])
@pytest.mark.parametrize(
    "message",
    [
        "model stopped without visible output or tool calls",
        "model stopped with reason: length",
    ],
)
def test_abbreviated_model_failure_with_native_collection_keeps_zero(
    adopted, file_count, message
):
    error = _error(adopted=adopted, message=message)
    error.harness_result["collection"]["file_count"] = file_count

    assert (
        ArenaStreamAgentWorkflow._harness_result_format(error.result.raw)
        == "native-runner"
    )
    assert (
        ArenaStreamAgentWorkflow.classify_proxy_failure(
            error, context_overflow=False, interaction_count=2
        )
        == "model_failure_zero"
    )


@pytest.mark.parametrize("context_overflow", [False, True])
@pytest.mark.parametrize(
    "mutation",
    [
        "missing_receipt",
        "missing_collection",
        "failed_collection",
        "negative_file_count",
        "boolean_file_count",
        "delivery",
        "missing_delivery",
        "receipt_failure",
        "receipt_nested_outcome",
        "receipt_native_family",
        "receipt_error_outcome",
        "wrong_implementation",
        "wrong_status",
        "boolean_exit",
        "multiple_turns",
        "multiple_errors",
        "provider_failure",
        "budget_wrong_kind",
        "budget_not_exceeded",
        "budget_malformed",
        "budget_zero_limit",
        "runner_timeout",
        "external_interruption",
        "public_input_changed",
        "deliverable_conflict",
        "no_interactions",
        "nonzero_score",
        "missing_score",
        "wrong_agent_exit",
        "missing_runner",
        "system_log",
        "conflicting_family",
    ],
)
def test_abbreviated_failure_with_missing_or_conflicting_evidence_rejected(
    mutation, context_overflow
):
    error = _error()
    receipt = error.harness_result
    interaction_count = 2
    if mutation == "missing_receipt":
        error.harness_result = None
    elif mutation == "missing_collection":
        receipt.pop("collection")
    elif mutation == "failed_collection":
        receipt["collection"]["status"] = "ERROR"
    elif mutation == "negative_file_count":
        receipt["collection"]["file_count"] = -1
    elif mutation == "boolean_file_count":
        receipt["collection"]["file_count"] = True
    elif mutation == "delivery":
        receipt["delivery"] = {"error": "export failed"}
    elif mutation == "missing_delivery":
        receipt.pop("delivery")
    elif mutation == "receipt_failure":
        receipt["failure"] = {"code": "EXPORT_FAILED"}
    elif mutation == "receipt_nested_outcome":
        receipt["raw"] = {"outcome": {"class": "infrastructure"}}
    elif mutation == "receipt_native_family":
        receipt["nativeExportHealthy"] = False
    elif mutation == "receipt_error_outcome":
        receipt["error"][0]["outcome"] = {"class": "infrastructure"}
    elif mutation == "wrong_implementation":
        receipt["implementation"] = "other-runtime"
    elif mutation == "wrong_status":
        receipt["status"] = "OK"
    elif mutation == "boolean_exit":
        receipt["exit_code"] = True
    elif mutation == "multiple_turns":
        receipt["turn_statuses"] = ["failed", "failed"]
    elif mutation == "multiple_errors":
        receipt["error"].append({"message": "provider failed"})
    elif mutation == "runner_timeout":
        receipt["observed_runner_timeout"] = True
    elif mutation == "external_interruption":
        receipt["observed_external_interruption"] = True
    elif mutation in {
        "provider_failure",
        "budget_wrong_kind",
        "budget_not_exceeded",
        "budget_malformed",
        "budget_zero_limit",
        "public_input_changed",
        "deliverable_conflict",
    }:
        receipt["error"][0]["message"] = {
            "provider_failure": "model service unavailable",
            "budget_wrong_kind": "tool_call_budget_exceeded: tokens (limit 10, observed 11)",
            "budget_not_exceeded": "tool_call_budget_exceeded: argument_bytes (limit 10, observed 10)",
            "budget_malformed": "tool_call_budget_exceeded: argument_bytes (limit x, observed 69773)",
            "budget_zero_limit": "tool_call_budget_exceeded: argument_bytes (limit 0, observed 69773)",
            "public_input_changed": "Public inputs changed during agent execution",
            "deliverable_conflict": "Conflicting workspace/output deliverable: renders/view.png",
        }[mutation]
    elif mutation == "no_interactions":
        interaction_count = 0
    elif mutation in {"nonzero_score", "missing_score"}:
        error.result = replace(
            error.result, score=1.0 if mutation == "nonzero_score" else None
        )
    elif mutation == "wrong_agent_exit":
        error.result.raw["error"] = error.result.raw["error"].replace(
            "code 1:", "code 2:"
        )
    elif mutation == "missing_runner":
        error.result.raw["error"] = (
            "harness: harness agent phase exited with code 1: startup"
        )
    elif mutation == "system_log":
        error.result.raw["error"] += "\ncollect failed: no space left"
    else:
        error.result.raw["nativeExportHealthy"] = False

    assert (
        ArenaStreamAgentWorkflow.classify_proxy_failure(
            error,
            context_overflow=context_overflow,
            interaction_count=interaction_count,
        )
        == "unknown_failure_reject"
    )


@pytest.mark.parametrize("adopted", [False, True])
def test_abbreviated_tool_call_budget_failure_keeps_zero(adopted):
    error = _error(
        adopted=adopted,
        message="tool_call_budget_exceeded: argument_bytes (limit 65536, observed 69773)",
    )

    assert (
        ArenaStreamAgentWorkflow.classify_proxy_failure(
            error, context_overflow=False, interaction_count=2
        )
        == "model_failure_zero"
    )


@pytest.mark.parametrize("context_overflow", [False, True])
@pytest.mark.parametrize(
    "fault",
    [
        None,
        "runner_timeout",
        "external_interruption",
        "collection",
        "receipt_failure",
        "provider_failure",
    ],
)
def test_runner_budget_receipt_overrides_earlier_proxy_overflow(
    context_overflow, fault
):
    """A full runner envelope cannot rescue a conflicting terminal receipt."""
    error = _error(
        message="tool_call_budget_exceeded: argument_bytes (limit 65536, observed 69773)"
    )
    error.result = replace(
        error.result,
        raw={"harness": {"phase": "agent", "result_status": "ERROR", "exit_code": 1}},
    )
    receipt = error.harness_result
    if fault == "runner_timeout":
        receipt["observed_runner_timeout"] = True
    elif fault == "external_interruption":
        receipt["observed_external_interruption"] = True
    elif fault == "collection":
        receipt["collection"]["status"] = "ERROR"
    elif fault == "receipt_failure":
        receipt["failure"] = {"code": "EXPORT_FAILED"}
    elif fault == "provider_failure":
        receipt["error"][0]["message"] = "HTTP 503 service unavailable"

    assert ArenaStreamAgentWorkflow.classify_proxy_failure(
        error, context_overflow=context_overflow, interaction_count=2
    ) == ("model_failure_zero" if fault is None else "unknown_failure_reject")
    assert ArenaStreamAgentWorkflow._is_tool_call_budget_failure(error) is (
        fault is None
    )


@pytest.mark.parametrize("adopted", [False, True])
@pytest.mark.parametrize("fetch_fails", [False, True])
def test_abbreviated_task_fetches_receipt_before_classification(
    monkeypatch, adopted, fetch_fails
):
    for proxy_var in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        monkeypatch.delenv(proxy_var, raising=False)
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "arena-token")
    monkeypatch.setenv("ARENA_LLM_API_KEY", "llm-key")
    workflow = ArenaStreamAgentWorkflow(
        econfig={
            "arena_base_url": "https://arena.example",
            "arena_streams": [{"name": "a", "stream_id": "stream-a"}],
        }
    )
    error = _error(adopted=adopted)
    receipt, error.harness_result = error.harness_result, None
    fetched = []

    async def register(**_kwargs):
        return "https://arena.example/api", "model-a"

    async def launch(**kwargs):
        kwargs["on_launch_success"](error.task_id)
        raise error

    async def get_receipt(task_id, **_kwargs):
        fetched.append(task_id)
        if fetch_fails:
            raise RuntimeError("receipt unavailable")
        return receipt

    async def delete(*_args, **_kwargs):
        return None

    monkeypatch.setattr(workflow.client, "register_llm_proxy_async", register)
    monkeypatch.setattr(workflow.client, "launch_one_task_result", launch)
    monkeypatch.setattr(workflow.client, "get_harness_result_async", get_receipt)
    monkeypatch.setattr(workflow.client, "delete_llm_proxy_async", delete)

    with pytest.raises(ArenaTaskFailedError) as caught:
        asyncio.run(
            workflow.run(
                {
                    "arena_stream_name": "a",
                    "stream_id": "stream-a",
                    "data_id": "data-a",
                },
                base_url="http://rollout-proxy",
                api_key="session-key",
            )
        )

    assert fetched == [error.task_id]
    assert workflow.classify_proxy_failure(
        caught.value, context_overflow=False, interaction_count=2
    ) == ("unknown_failure_reject" if fetch_fails else "model_failure_zero")
