# SPDX-License-Identifier: Apache-2.0
"""Hermes bindings shared by Agent Service and standalone task runners.

This module deliberately imports no training engines or Agent Service types.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from copy import deepcopy
from functools import partial, wraps
from typing import Any

from examples.task_workflows.trace import CallTrace, observe_openai


def _bind_episode_requests(agent: Any, episode: dict[str, Any]) -> None:
    """Bind all real SDK clients, including Hermes' direct summary requests.

    Hermes 0.19's iteration-limit summary bypasses request_overrides. Bind the
    instance's client factory as well as its existing client so recreation and
    per-request clients preserve the same identity. Responses are untouched.
    """
    identity = deepcopy(episode)

    def bind(client: Any) -> Any:
        create = client.chat.completions.create
        if getattr(create, "_areal_episode", None) == identity:
            return client

        @wraps(create)
        def create_for_episode(*args: Any, **kwargs: Any) -> Any:
            body = dict(kwargs.pop("extra_body", None) or {})
            if "areal_episode" in body and body["areal_episode"] != identity:
                raise ValueError("Hermes request conflicts with its bound episode")
            body["areal_episode"] = deepcopy(identity)
            return create(*args, **kwargs, extra_body=body)

        create_for_episode._areal_episode = identity
        client.chat.completions.create = create_for_episode
        return client

    create_client = agent._create_openai_client

    @wraps(create_client)
    def create_bound_client(*args: Any, **kwargs: Any) -> Any:
        return bind(create_client(*args, **kwargs))

    agent._create_openai_client = create_bound_client
    bind(agent.client)


class _TimedStream:
    """Preserve the SDK stream interface while timing consumption and cleanup."""

    def __init__(self, stream: Any, finish: Any, observe: Any = None) -> None:
        self._stream, self._finish = stream, finish
        self._observe = observe
        self._iterator = iter(stream)

    def __iter__(self) -> _TimedStream:
        return self

    def __next__(self) -> Any:
        try:
            chunk = next(self._iterator)
            if self._observe is not None:
                self._observe(chunk)
            return chunk
        except BaseException:
            self._finish()
            raise

    def close(self) -> Any:
        try:
            return self._stream.close()
        finally:
            self._finish()

    def __enter__(self) -> _TimedStream:
        self._stream.__enter__()
        return self

    def __exit__(self, *args: Any) -> Any:
        try:
            return self._stream.__exit__(*args)
        finally:
            self._finish()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def observe_inference(agent: Any, sampling: dict[str, Any]) -> Any:
    """Time actual SDK calls, including summary calls and stream consumption."""
    agent.request_overrides = {**agent.request_overrides, **sampling}
    agent._areal_call_trace = CallTrace()
    agent._areal_inference_calls = agent._areal_call_trace.requests

    def bind(client: Any) -> Any:
        create = client.chat.completions.create
        if getattr(create, "_areal_task_timing", None) is agent._areal_inference_calls:
            return client

        @wraps(create)
        def sampled(*args: Any, **kwargs: Any) -> Any:
            clock = time.perf_counter()
            record = {"started": time.time(), "finished": None, "seconds": None}
            agent._areal_call_trace.request(record, messages=kwargs.get("messages"))

            def finish() -> None:
                if record["finished"] is None:
                    record.update(
                        finished=time.time(), seconds=time.perf_counter() - clock
                    )

            try:
                if kwargs.get("stream"):
                    kwargs["stream_options"] = {
                        **(kwargs.get("stream_options") or {}),
                        "include_usage": True,
                    }
                response = create(*args, **{**kwargs, **sampling})
            except BaseException:
                finish()
                raise
            if kwargs.get("stream"):
                return _TimedStream(
                    response, finish, lambda chunk: observe_openai(record, chunk)
                )
            observe_openai(record, response)
            finish()
            return response

        sampled._areal_task_timing = agent._areal_inference_calls
        client.chat.completions.create = sampled
        return client

    factory = agent._create_openai_client

    @wraps(factory)
    def sampled_client(*args: Any, **kwargs: Any) -> Any:
        return bind(factory(*args, **kwargs))

    agent._create_openai_client = sampled_client
    bind(agent.client)
    return agent


_runtimes: dict[str, Any] = {}
_registry_lock = threading.RLock()


def _dispatch_tool(name: str, arguments: dict[str, Any], **kwargs: Any) -> str:
    with _registry_lock:
        runtime = _runtimes.get(kwargs.get("task_id"))
    if runtime is None:
        raise RuntimeError("Tool was invoked outside its bound task environment")
    return runtime.invoke(name, arguments)


def register_environment_tools(environment_type: type) -> None:
    from tools.registry import registry
    from toolsets import create_custom_toolset

    with _registry_lock:
        names = []
        for schema in environment_type.tool_schemas:
            name = schema["name"]
            names.append(name)
            entry = registry.get_entry(name)
            if entry is not None:
                if (
                    getattr(entry.handler, "_areal_environment_kind", None)
                    != environment_type.kind
                ):
                    raise RuntimeError(f"A different implementation owns tool {name}")
                continue
            handler = partial(_dispatch_tool, name)
            handler._areal_environment_kind = environment_type.kind
            registry.register(
                name=name,
                toolset=environment_type.toolset,
                schema=schema,
                handler=handler,
                check_fn=lambda: True,
            )
        create_custom_toolset(environment_type.toolset, "Task environment tools", names)


@contextmanager
def bind_environment(task_id: str, runtime: Any):
    with _registry_lock:
        _runtimes[task_id] = runtime
    try:
        yield
    finally:
        with _registry_lock:
            _runtimes.pop(task_id, None)
