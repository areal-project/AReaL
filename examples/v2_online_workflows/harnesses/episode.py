# SPDX-License-Identifier: Apache-2.0
"""Run one Hermes episode using an explicitly supplied task workspace."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from examples.v2_online_workflows.artifacts import write_json
from examples.v2_online_workflows.harnesses.hermes_runtime import (
    _bind_episode_requests,
    bind_environment,
    observe_inference,
    raise_for_inference_errors,
    register_environment_tools,
)
from examples.v2_online_workflows.harnesses.timing import mark as startup_mark
from examples.v2_online_workflows.harnesses.tools import (
    native_workspace_tools,
    require_native_tool_execution,
)
from examples.v2_online_workflows.harnesses.workspace import workspace_instructions
from examples.v2_online_workflows.outcome import require_successful_execution


def _format_unlimited_iterations(record: logging.LogRecord) -> bool:
    """Hermes 0.19's terminal diagnostic formats numeric caps with %d."""
    if isinstance(record.args, tuple) and any(
        isinstance(value, float) and value == float("inf") for value in record.args
    ):
        record.msg = record.msg.replace("%d", "%s")
        record.args = tuple(
            "unlimited" if isinstance(value, float) and value == float("inf") else value
            for value in record.args
        )
    return True


def run_episode(
    prompt: str,
    *,
    output: Path,
    base_url: str,
    api_key: str,
    model: str,
    sampling: dict[str, Any],
    episode: dict[str, Any] | None = None,
    session_id: str | None = None,
    mock: bool = False,
    runtime: Any,
    max_iterations: int | None,
    instructions: str | None = None,
    collect: Callable[[Any, Path], dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    """The caller owns session/reward lifecycle; the runner never sees golds."""
    if importlib.metadata.version("hermes-agent") != "0.19.0":
        raise RuntimeError("This runner requires hermes-agent==0.19.0")
    from run_agent import AIAgent

    startup_mark("hermes_import")

    if set(sampling) != {"max_tokens", "temperature", "top_p"}:
        raise ValueError("Explicit max_tokens, temperature and top_p are required")
    if sampling["max_tokens"] is not None and (
        type(sampling["max_tokens"]) is not int or sampling["max_tokens"] < 1
    ):
        raise ValueError(
            "max_tokens must be null (no output cap) or a positive integer"
        )
    if not 0 <= sampling["temperature"] <= 2 or not 0 < sampling["top_p"] <= 1:
        raise ValueError("Invalid temperature or top_p")
    if episode is not None and (
        set(episode) != {"incarnation_id", "trajectory_id"}
        or not isinstance(episode["incarnation_id"], str)
        or not episode["incarnation_id"]
        or type(episode["trajectory_id"]) is not int
        or episode["trajectory_id"] < 0
    ):
        raise ValueError("Invalid AReaL episode identity")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    task_id = "arena-task-" + uuid.uuid4().hex
    native = getattr(runtime, "native_tools", False)
    if native:
        task_id = runtime.task_id
    else:
        register_environment_tools(type(runtime))
    if max_iterations is not None and (
        type(max_iterations) is not int or max_iterations < 1
    ):
        raise ValueError("Hermes max_iterations must be null (unlimited) or positive")
    if max_iterations is None:
        logging.getLogger("agent.conversation_loop").addFilter(
            _format_unlimited_iterations
        )
    overrides = {"extra_body": {"areal_episode": episode}} if episode else {}
    agent = AIAgent(
        base_url=base_url,
        api_key=api_key,
        model=model,
        # Pinned Hermes 0.19 compares both the loop count and IterationBudget
        # against this numeric value, including after its per-turn reset.
        # Infinity disables both checks without substituting a large finite cap.
        # The public config and answer retain JSON null, never Infinity.
        max_iterations=float("inf") if max_iterations is None else max_iterations,
        enabled_toolsets=["terminal", "file"] if native else [runtime.toolset],
        save_trajectories=False,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        session_db=None,
        request_overrides=overrides,
    )
    startup_mark("agent_construct")
    if episode is not None:
        _bind_episode_requests(agent, episode)
    observe_inference(
        agent,
        sampling,
        request_snapshot=output / "model-input.json"
        if native or runtime.kind == "repository"
        else None,
    )
    if not native:
        agent._areal_call_trace.bind_tools(runtime)
    startup_mark("inference_observer_setup")
    artifacts = None
    try:
        tools = (
            native_workspace_tools(runtime, agent._areal_call_trace)
            if native
            else bind_environment(task_id, runtime)
        )
        with tools:
            if instructions is None and native:
                instructions = workspace_instructions(runtime)
            options = {"system_message": instructions} if instructions else {}
            startup_mark("tools_and_instructions")
            result = agent.run_conversation(prompt, task_id=task_id, **options)
        raise_for_inference_errors(agent._areal_inference_calls)
        if native:
            require_native_tool_execution(result, agent._areal_call_trace)
        environment = runtime.metadata()
        require_successful_execution(result, environment, agent._areal_inference_calls)
        artifacts = collect(runtime, output) if collect is not None else None
        answer = {
            "status": "OK",
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "max_iterations": max_iterations,
            "model_context_length": getattr(
                agent.context_compressor, "context_length", None
            ),
            "answer": result.get("final_response") or "",
            "completed": bool(result.get("completed")),
            "model_calls": result.get("api_calls", 0),
            "model_mode": "mock" if mock else "external",
            "training_eligible": not mock and bool(session_id or episode),
            "areal_episode": episode,
            "areal_session_id": session_id,
            "sampling": sampling,
            "turn_exit_reason": result.get("turn_exit_reason"),
            "tool_backend": "hermes_native" if native else runtime.toolset,
            "tool_use_enforcement": getattr(agent, "_tool_use_enforcement", None),
        }
        if result.get("partial") and result.get("error"):
            # require_successful_execution already accepted only the explicit
            # native reasoning-budget stop with successful model requests.
            answer["turn_exit_reason"] = "output_token_budget_exhausted"
        incomplete = answer["turn_exit_reason"] in {
            "empty_response_exhausted",
            "output_token_budget_exhausted",
        }
        answer["completed_normally"] = answer["completed"] and not incomplete
        answer["termination_class"] = (
            "model_response_incomplete" if incomplete else "normal"
        )
        if incomplete:
            answer["completed"] = False
            answer["training_eligible"] = False
        write_json(output / "answer.json", answer)
        return answer
    except Exception as exc:
        write_json(
            output / "answer.json",
            {
                "status": "ERROR",
                "completed": False,
                "training_eligible": False,
                "answer": "",
                "model_mode": "mock" if mock else "external",
                "areal_episode": episode,
                "areal_session_id": session_id,
                "sampling": sampling,
                "model_calls": len(agent._areal_inference_calls),
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc).strip() or type(exc).__name__,
                },
            },
        )
        raise
    finally:
        try:
            environment = runtime.metadata()
            if artifacts is not None:
                environment["artifacts"] = artifacts
            write_json(output / "environment.json", environment)
            write_json(output / "call_trace.json", agent._areal_call_trace.snapshot())
        finally:
            runtime.close()
            agent.client.close()


def configure_hermes_model(
    home: Path,
    headers: dict[str, str],
    context_length: int | None,
    *,
    repository_tools: bool = False,
    native_tools: bool = False,
) -> None:
    """Use native Hermes configuration rather than guessing from a model alias."""
    model: dict[str, Any] = {}
    if headers:
        model["default_headers"] = headers
    if context_length is not None:
        if type(context_length) is not int or context_length <= 0:
            raise ValueError("Model context length must be a positive integer")
        model["context_length"] = context_length
    config: dict[str, Any] = {"model": model} if model else {}
    if repository_tools or native_tools:
        config.update(
            agent={"tool_use_enforcement": True},
            terminal={"auto_source_bashrc": False},
            tool_output={"max_bytes": 65536},
        )
    if not config:
        return
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    # The task owns a fresh HERMES_HOME. Preserve fail-closed exclusive creation.
    with open(
        home / "config.yaml", "x", opener=lambda p, flags: os.open(p, flags, 0o600)
    ) as stream:
        json.dump(config, stream)
