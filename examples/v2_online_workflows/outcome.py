# SPDX-License-Identifier: Apache-2.0
"""Keep execution failures separate from benchmark quality scores."""

from __future__ import annotations

from typing import Any


class TaskExecutionError(RuntimeError):
    """A finished task failed and must not receive a training reward."""


def raise_for_inference_errors(calls: list[dict[str, Any]]) -> None:
    """Reject unfinished requests or an unresolved final request failure.

    Earlier errors remain diagnostic evidence after the agent retries successfully.
    The agent outcome and environment must still be checked separately.
    """
    if any(call.get("status") == "pending" for call in calls) or (
        calls and (calls[-1].get("status") == "error" or calls[-1].get("error"))
    ):
        raise RuntimeError("Model inference failed; this is not a zero-reward answer")


def require_successful_execution(
    outcome: dict[str, Any],
    environment: dict[str, Any] | None = None,
    inference_calls: list[dict[str, Any]] | None = None,
) -> None:
    """Reject explicit failures, including errors swallowed by an agent loop.

    An ordinary iteration budget stop is still a benchmark outcome. Never infer
    execution errors from answer text, or copy provider error bodies into logs.
    """
    # Hermes 0.19 flags a reasoning-only length stop as an error. This is a
    # completed model budget outcome, not an unavailable service or tool.
    reasoning_budget_stop = (
        outcome.get("partial") is True
        and outcome.get("completed") is False
        and outcome.get("error")
        == (
            "Model used all output tokens on reasoning with none left "
            "for the response. Try lowering reasoning effort or "
            "increasing max_tokens."
        )
        and bool(inference_calls)
        and inference_calls[-1].get("status") == "ok"
        and inference_calls[-1].get("finish_reason") == "length"
    )
    if (
        outcome.get("status") not in (None, "OK", "completed")
        or (outcome.get("error") and not reasoning_budget_stop)
        or outcome.get("failed")
        or outcome.get("interrupted")
    ):
        raise TaskExecutionError("agent_execution_error")
    if str(outcome.get("turn_exit_reason", "")).startswith(
        ("local_processing_error", "error_near_max_iterations")
    ):
        raise TaskExecutionError("agent_processing_error")
    environment = environment or {}
    if environment.get("infrastructure_error") or any(
        call.get("status") == "infrastructure_error"
        for call in environment.get("calls", [])
    ):
        raise TaskExecutionError("tool_infrastructure_error")
    try:
        raise_for_inference_errors(inference_calls or [])
    except RuntimeError as exc:
        raise TaskExecutionError("model_call_error") from exc
