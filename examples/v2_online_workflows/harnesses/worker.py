# SPDX-License-Identifier: Apache-2.0
"""Execute one task at a time in each persistent Hermes worker process."""

from __future__ import annotations

import importlib
import os
import tempfile
from pathlib import Path


def initialize(root, context_length, harness_env):
    os.umask(0o077)
    if "AUTOPILOT_MAX_ITERATIONS" not in harness_env:
        raise ValueError("Supply explicitly approved AUTOPILOT_MAX_ITERATIONS")
    home = Path(tempfile.mkdtemp(prefix="local-hermes-worker-"))
    os.environ.update(harness_env)
    os.environ["HERMES_HOME"] = str(home)
    os.environ["AUTOPILOT_STARTUP_TIMING"] = "1"
    from examples.v2_online_workflows.harnesses.episode import configure_hermes_model

    configure_hermes_model(home, {}, context_length, native_tools=True)
    # Import once at worker startup, outside task latency.
    importlib.import_module("run_agent")


def execute(request, *, root, image, executable):
    import tools.terminal_tool as terminal

    from examples.v2_online_workflows.environments.hermes import LocalTaskEnvironment
    from examples.v2_online_workflows.harnesses.episode import run_episode
    from examples.v2_online_workflows.harnesses.workspace import TaskWorkspace

    allowed = {
        "request_id",
        "task_id",
        "prompt",
        "base_url",
        "api_key",
        "model",
        "sampling",
        "episode",
        "session_id",
        "mock",
    }
    if set(request) - allowed:
        raise ValueError("Unexpected worker input")
    request_id = request["request_id"]
    if not request_id or any(
        char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for char in request_id
    ):
        raise ValueError("Invalid request identity")
    output = Path(root) / "episodes" / request_id
    output.mkdir(parents=True, exist_ok=False)
    os.environ["ARENA_AGENT_OUTPUT_DIR"] = str(output)
    tool_path = os.environ.get("AUTOPILOT_TOOL_PATH", os.defpath)

    class LocalWorkspace(TaskWorkspace):
        def metadata(self):
            return {
                **super().metadata(),
                "backend": "local-singularity-native-tools",
                "tool_image": image,
            }

    workspace = output / "workspace-placeholder"
    workspace.mkdir()
    runtime = LocalWorkspace(
        output, request_id, workspace=workspace, tool_path=tool_path
    )
    runtime.workdir = "/work"
    runtime.terminal_env = "singularity"
    runtime.terminal_overrides = {
        "env_type": "singularity",
        "cwd": "/work",
        "singularity_image": image,
    }
    original = terminal._create_environment
    environments = []

    def factory(env_type, image, cwd, timeout, **kwargs):
        if env_type != "singularity":
            raise RuntimeError("Local Task cannot use a host tool backend")
        backend = LocalTaskEnvironment(
            image=image,
            directory=output / "sandbox",
            executable=executable,
            timeout=timeout,
        )
        environments.append(backend)
        backend.init_session()
        return backend

    terminal._create_environment = factory
    try:
        iteration_setting = os.environ["AUTOPILOT_MAX_ITERATIONS"]
        answer = run_episode(
            request["prompt"],
            output=output,
            base_url=request["base_url"],
            api_key=request["api_key"],
            model=request["model"],
            sampling=request["sampling"],
            episode=request.get("episode"),
            session_id=request.get("session_id"),
            mock=request.get("mock", False),
            runtime=runtime,
            max_iterations=None
            if iteration_setting == "null"
            else int(iteration_setting),
        )
        return {
            "output": str(output),
            "completed": answer["completed"],
            "worker_pid": os.getpid(),
        }
    except Exception as exc:
        # A returned failure is terminal evidence. A lost response remains unknown.
        return {
            "output": str(output),
            "completed": False,
            "worker_pid": os.getpid(),
            "execution_failed": True,
            "error_type": type(exc).__name__,
        }
    finally:
        terminal._create_environment = original
        for backend in environments:
            backend.cleanup()
