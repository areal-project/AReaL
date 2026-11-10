# SPDX-License-Identifier: Apache-2.0
"""Hermes bindings shared by Agent Service and standalone task runners.

This module deliberately imports no training engines or Agent Service types.
"""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from copy import deepcopy
from functools import partial, wraps
from pathlib import Path
from typing import Any

from examples.v2_online_workflows.artifacts import write_json
from examples.v2_online_workflows.outcome import (
    raise_for_inference_errors as raise_for_inference_errors,
)
from examples.v2_online_workflows.trace import CallTrace, observe_openai


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

    def __init__(
        self, stream: Any, finish: Any, observe: Any = None, fail: Any = None
    ) -> None:
        self._stream, self._finish = stream, finish
        self._observe = observe
        self._fail = fail
        self._iterator = iter(stream)

    def __iter__(self) -> _TimedStream:
        return self

    def __next__(self) -> Any:
        try:
            chunk = next(self._iterator)
            if self._observe is not None:
                self._observe(chunk)
            return chunk
        except StopIteration:
            self._finish()
            raise
        except Exception as exc:
            if self._fail is not None:
                self._fail(exc)
            self._finish()
            raise

    def close(self) -> Any:
        try:
            return self._stream.close()
        except Exception as exc:
            if self._fail is not None:
                self._fail(exc)
            raise
        finally:
            self._finish()

    def __enter__(self) -> _TimedStream:
        try:
            self._stream.__enter__()
        except Exception as exc:
            if self._fail is not None:
                self._fail(exc)
            self._finish()
            raise
        return self

    def __exit__(self, *args: Any) -> Any:
        try:
            if args and isinstance(args[1], Exception) and self._fail is not None:
                self._fail(args[1])
            return self._stream.__exit__(*args)
        except Exception as exc:
            if self._fail is not None:
                self._fail(exc)
            raise
        finally:
            self._finish()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def observe_inference(
    agent: Any, sampling: dict[str, Any], *, request_snapshot: Path | None = None
) -> Any:
    """Time actual SDK calls, including summary calls and stream consumption."""
    agent.request_overrides = {**agent.request_overrides, **sampling}
    agent._areal_call_trace = CallTrace()
    agent._areal_inference_calls = agent._areal_call_trace.requests
    snapshot_lock = threading.Lock()

    def bind(client: Any) -> Any:
        create = client.chat.completions.create
        if getattr(create, "_areal_task_timing", None) is agent._areal_inference_calls:
            return client

        @wraps(create)
        def sampled(*args: Any, **kwargs: Any) -> Any:
            clock = time.perf_counter()
            record = {
                "started": time.time(),
                "finished": None,
                "seconds": None,
                "status": "pending",
            }
            agent._areal_call_trace.request(record, messages=kwargs.get("messages"))

            def finish() -> None:
                if record["finished"] is None:
                    record.update(
                        finished=time.time(), seconds=time.perf_counter() - clock
                    )
                    if record["status"] == "pending":
                        record["status"] = "ok"

            def fail(exc: Exception) -> None:
                record.update(
                    status="error",
                    error={
                        "type": type(exc).__name__,
                        "message": str(exc).strip() or type(exc).__name__,
                    },
                )

            try:
                if request_snapshot is not None:
                    # Retain the complete conversation for every SDK request.
                    # Headers and extra_body contain credentials/session keys.
                    payload = {
                        key: deepcopy(value)
                        for key, value in {**kwargs, **sampling}.items()
                        if key
                        in {
                            "model",
                            "messages",
                            "tools",
                            "tool_choice",
                            "max_tokens",
                            "max_completion_tokens",
                            "temperature",
                            "top_p",
                            "stream",
                        }
                    }
                    secrets = [
                        getattr(client, "api_key", None),
                        os.environ.get("OPENAI_API_KEY"),
                    ]
                    try:
                        secrets.append(
                            json.loads(
                                os.environ.get("AUTOPILOT_MODEL_HEADERS_JSON", "{}")
                            ).get("X-Session-Token")
                        )
                    except ValueError:
                        pass

                    def redact(value):
                        if isinstance(value, str):
                            for secret in secrets:
                                if isinstance(secret, str) and len(secret) >= 8:
                                    value = value.replace(secret, "[REDACTED]")
                            return value
                        if isinstance(value, list):
                            return [redact(item) for item in value]
                        if isinstance(value, dict):
                            return {key: redact(item) for key, item in value.items()}
                        return value

                    payload = redact(payload)
                    request_snapshot.parent.mkdir(parents=True, exist_ok=True)
                    with snapshot_lock:
                        with open(
                            request_snapshot.with_name("model-requests.jsonl"),
                            "a",
                            encoding="utf-8",
                            opener=lambda p, flags: os.open(p, flags, 0o600),
                        ) as stream:
                            stream.write(
                                json.dumps(
                                    {
                                        "request": record["name"],
                                        "started": record["started"],
                                        "payload": payload,
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                if request_snapshot is not None and not request_snapshot.exists():
                    # Explicit input allowlist: no headers, API keys or session tokens.
                    write_json(
                        request_snapshot,
                        {
                            "scope": "first SDK request input before HTTP serialization",
                            **{
                                key: payload[key]
                                for key in ("model", "messages", "tools", "tool_choice")
                                if key in kwargs
                            },
                            "sampling": deepcopy(sampling),
                        },
                    )
                if kwargs.get("stream"):
                    kwargs["stream_options"] = {
                        **(kwargs.get("stream_options") or {}),
                        "include_usage": True,
                    }
                response = create(*args, **{**kwargs, **sampling})
            except Exception as exc:
                fail(exc)
                finish()
                raise
            if kwargs.get("stream"):
                return _TimedStream(
                    response, finish, lambda chunk: observe_openai(record, chunk), fail
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
