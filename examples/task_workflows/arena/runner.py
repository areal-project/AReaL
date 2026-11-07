# SPDX-License-Identifier: Apache-2.0
"""Run one native Hermes conversation using Arena's file/environment contract."""

from __future__ import annotations

import importlib.metadata
import json
import os
import uuid
from pathlib import Path
from typing import Any

from examples.task_workflows.arena.python_tool import ArenaPython
from examples.task_workflows.environments.python import ToolSettings
from examples.task_workflows.harnesses.hermes_runtime import (
    _bind_episode_requests,
    bind_environment,
    observe_inference,
    register_environment_tools,
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


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
    runtime: Any = None,
) -> dict[str, Any]:
    """The caller owns session/reward lifecycle; the runner never sees golds."""
    if importlib.metadata.version("hermes-agent") != "0.19.0":
        raise RuntimeError("This runner requires hermes-agent==0.19.0")
    from run_agent import AIAgent

    if set(sampling) != {"max_tokens", "temperature", "top_p"}:
        raise ValueError("Explicit max_tokens, temperature and top_p are required")
    if type(sampling["max_tokens"]) is not int or sampling["max_tokens"] < 1:
        raise ValueError("max_tokens must be a positive integer")
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
    task_id = "arena-math-" + uuid.uuid4().hex
    runtime = runtime or ArenaPython(ToolSettings("arena", output / "tools"), task_id)
    register_environment_tools(type(runtime))
    overrides = {"extra_body": {"areal_episode": episode}} if episode else {}
    agent = AIAgent(
        base_url=base_url,
        api_key=api_key,
        model=model,
        enabled_toolsets=[runtime.toolset],
        save_trajectories=False,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        session_db=None,
        request_overrides=overrides,
    )
    if episode is not None:
        _bind_episode_requests(agent, episode)
    observe_inference(agent, sampling)
    agent._areal_call_trace.bind_tools(runtime)
    try:
        with bind_environment(task_id, runtime):
            result = agent.run_conversation(prompt, task_id=task_id)
        environment = runtime.metadata()
        if any(
            row.get("status") == "infrastructure_error" for row in environment["calls"]
        ):
            raise RuntimeError("The Python tool failed to execute in its sandbox")
        answer = {
            "answer": result.get("final_response") or "",
            "completed": bool(result.get("completed")),
            "model_calls": result.get("api_calls", 0),
            "model_mode": "mock" if mock else "external",
            "training_eligible": not mock and bool(session_id or episode),
            "areal_episode": episode,
            "areal_session_id": session_id,
            "sampling": sampling,
        }
        write_json(output / "answer.json", answer)
        return answer
    finally:
        try:
            write_json(output / "environment.json", runtime.metadata())
            write_json(output / "call_trace.json", agent._areal_call_trace.snapshot())
        finally:
            runtime.close()
            agent.client.close()


def main() -> None:
    output = Path(os.environ["ARENA_AGENT_OUTPUT_DIR"])
    prompt = Path(os.environ["ARENA_QUERY_PATH"]).read_text()
    if "AUTOPILOT_SAMPLING_JSON" not in os.environ:
        raise ValueError("Supply the frozen sampling configuration")
    episode = json.loads(os.environ.get("AUTOPILOT_AREAL_EPISODE_JSON", "null"))
    answer = run_episode(
        prompt,
        output=output,
        base_url=os.environ["OPENAI_BASE_URL"],
        api_key=os.environ["OPENAI_API_KEY"],
        model=os.environ["MODEL_NAME"],
        sampling=json.loads(os.environ["AUTOPILOT_SAMPLING_JSON"]),
        episode=episode,
        session_id=os.environ.get("AUTOPILOT_AREAL_SESSION_ID"),
    )
    write_json(
        Path(os.environ["ARENA_OUTPUT_DIR"]) / "harness_result.json",
        {
            "status": "OK",
            "raw": {"completed": answer["completed"], "model_mode": "external"},
        },
    )


if __name__ == "__main__":
    main()
