# SPDX-License-Identifier: Apache-2.0

"""Bind task environments to an existing v2 agent without owning its agent loop."""

from __future__ import annotations

import asyncio
import importlib.metadata
import os
import uuid
from contextvars import ContextVar
from dataclasses import replace
from typing import Any

from examples.v2_online_workflows.environments.python import (
    PythonSandbox,
    ToolSettings,
    resolve_image,
)
from examples.v2_online_workflows.environments.repository import (
    RepositoryEnvironment,
    RepositorySettings,
)
from examples.v2_online_workflows.environments.singularity import create_environment
from examples.v2_online_workflows.harnesses.hermes import (
    HermesAgent,
    bind_environment,
    observe_inference,
    register_environment_tools,
)
from examples.v2_online_workflows.harnesses.hermes_runtime import (
    raise_for_inference_errors,
)
from examples.v2_online_workflows.outcome import require_successful_execution
from examples.v2_online_workflows.types import EnvironmentSpec

_environment_spec: ContextVar[EnvironmentSpec | None] = ContextVar(
    "task_environment", default=None
)


def _read_spec(request: Any) -> EnvironmentSpec:
    if (request.metadata or {}).get("chat_request") is not None:
        raise ValueError("Task workflows use the structured /v1/responses endpoint")
    spec = EnvironmentSpec(
        **((request.metadata or {}).get("task_environment") or {"kind": "python"})
    )
    expected_options = set() if spec.kind == "python" else {"repo", "base_commit"}
    if set(spec.options) != expected_options:
        raise ValueError(
            "Task environment initialization fields do not match its capability"
        )
    return spec


def _settings_for(spec: EnvironmentSpec) -> Any:
    if spec.kind == "python":
        settings = ToolSettings.from_env()
        return replace(settings, image=spec.image) if spec.image else settings
    return RepositorySettings.from_env(spec.image)


class TaskAgent(HermesAgent):
    """One independently scored task per session, with the native Hermes loop."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._sampling = {
            "max_tokens": int(
                os.environ.get(
                    "AREAL_TASK_MAX_TOKENS",
                    os.environ.get("HERMES_MATH_MAX_TOKENS", "4096"),
                )
            ),
            "temperature": float(
                os.environ.get(
                    "AREAL_TASK_TEMPERATURE",
                    os.environ.get("HERMES_MATH_TEMPERATURE", "1.0"),
                )
            ),
            "top_p": float(
                os.environ.get(
                    "AREAL_TASK_TOP_P", os.environ.get("HERMES_MATH_TOP_P", "1.0")
                )
            ),
        }
        if self._sampling["max_tokens"] <= 0:
            raise ValueError("A positive per-call generation budget is required")
        self._images: dict[str, str] = {}

    @staticmethod
    def _settings(spec: EnvironmentSpec) -> Any:
        return _settings_for(spec)

    def _build_agent(self, upstream: Any) -> Any:
        if importlib.metadata.version("hermes-agent") != "0.19.0":
            raise RuntimeError(
                "This adapter targets the audited hermes-agent==0.19.0 API"
            )
        spec = _environment_spec.get() or EnvironmentSpec("python")
        settings = self._settings(spec)
        if settings.image not in self._images:
            self._images[settings.image] = resolve_image(settings)
        provider = PythonSandbox if spec.kind == "python" else RepositoryEnvironment
        register_environment_tools(provider)
        agent = super()._build_agent(upstream, enabled_toolsets=[provider.toolset])
        return observe_inference(agent, self._sampling)

    async def _run_conversation(self, state: Any, message: str, history: list) -> dict:
        if history:
            raise ValueError("Each task requires a fresh Agent Service session")
        spec = _environment_spec.get() or EnvironmentSpec("python")
        settings = self._settings(spec)
        lock = await self._session_lock(state.session_key)
        async with lock:
            if self._sessions.get(state.session_key) is not state:
                raise RuntimeError("Task episode was replaced before execution")

            def execute() -> dict:
                task_id = (
                    "math-" if spec.kind == "python" else "repository-"
                ) + uuid.uuid4().hex
                provider = (
                    PythonSandbox if spec.kind == "python" else RepositoryEnvironment
                )
                runtime = create_environment(
                    provider,
                    settings,
                    task_id,
                    image_id=self._images[settings.image],
                    **spec.options,
                )
                environment = {}
                trace = state.agent._areal_call_trace
                trace.bind_tools(runtime)
                try:
                    with bind_environment(task_id, runtime):
                        result = state.agent.run_conversation(message, task_id=task_id)
                    raise_for_inference_errors(state.agent._areal_inference_calls)
                    environment = runtime.metadata()
                    require_successful_execution(
                        result, environment, state.agent._areal_inference_calls
                    )
                    environment["artifacts"] = runtime.export_artifacts()
                    return result
                finally:
                    try:
                        runtime.close()
                    finally:
                        state.metadata["environment"] = (
                            environment or runtime.metadata()
                        )
                        state.metadata["environment"]["sampling"] = self._sampling
                        state.metadata["inference_calls"] = (
                            state.agent._areal_inference_calls
                        )
                        state.metadata["simulation_trace"] = trace.snapshot()

            return await asyncio.to_thread(execute)

    async def run(self, request: Any, *, emitter: Any) -> Any:
        spec = _read_spec(request)
        token = _environment_spec.set(spec)
        try:
            response = await super().run(request, emitter=emitter)
            response.metadata.update(self._sessions[request.session_key].metadata)
            return response
        finally:
            _environment_spec.reset(token)


class CCTaskAgent:
    """The same benchmark/environment contract with the native Claude Code SDK loop."""

    def __init__(self, **kwargs: Any) -> None:
        from examples.v2_online_workflows.harnesses.cc import ClaudeCodeHarness

        self._harness = ClaudeCodeHarness()
        self._active: dict[str, asyncio.Task] = {}
        self._images: dict[str, str] = {}

    async def run(self, request: Any, *, emitter: Any) -> Any:
        spec = _read_spec(request)
        if request.history or request.session_key in self._active:
            raise ValueError("Each task requires a fresh Agent Service session")
        self._active[request.session_key] = asyncio.current_task()
        runtime = None
        try:
            settings = _settings_for(spec)
            if settings.image not in self._images:
                self._images[settings.image] = await asyncio.to_thread(
                    resolve_image, settings
                )
            provider = PythonSandbox if spec.kind == "python" else RepositoryEnvironment
            runtime = await asyncio.to_thread(
                create_environment,
                provider,
                settings,
                "cc-" + uuid.uuid4().hex,
                image_id=self._images[settings.image],
                **spec.options,
            )
            response = await self._harness.run(
                request, runtime=runtime, emitter=emitter
            )
            environment = await asyncio.to_thread(runtime.metadata)
            environment["artifacts"] = await asyncio.to_thread(runtime.export_artifacts)
            response.metadata["environment"] = environment
            return response
        finally:
            try:
                if runtime is not None:
                    await asyncio.to_thread(runtime.close)
            finally:
                self._active.pop(request.session_key, None)

    async def close_session(self, session_key: str) -> None:
        task = self._active.get(session_key)
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def close_all_sessions(self) -> None:
        await asyncio.gather(*(self.close_session(key) for key in list(self._active)))
