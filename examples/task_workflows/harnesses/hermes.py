# SPDX-License-Identifier: Apache-2.0

"""Hermes Agent for AReaL Agent Service (in-process per-session ``AIAgent``).

Implements :class:`AgentRunnable` by instantiating **one Hermes ``AIAgent``
per RL session** directly inside the Worker process.  Unlike OpenClaw — which
exposes an OpenAI-compatible Gateway subprocess — Hermes
(``github.com/nousresearch/hermes-agent``) is a Python library whose
``AIAgent`` drives an OpenAI-compatible upstream itself.  Each per-session
agent is therefore bound to its own upstream LLM (base URL + API key + model)
so that, during training, a session's turns can be attributed to a distinct
per-session key (``sk-sess-*``).

Per turn the agent calls ``AIAgent.run_conversation`` with the conversation
history the DataProxy replays.  Hermes rebuilds its message list from that
history on every call (it does not retain messages across calls), so the
replayed history stays the single source of truth and one cached agent can
safely serve every turn of a session.

Requires the ``hermes-agent`` package importable as the top-level ``run_agent``
module (``pip install hermes-agent``).

Upstream selection
------------------
Per turn the agent prefers the inference upstream the DataProxy injects via
``AgentRequest.metadata['areal_inference']`` (``base_url`` / ``api_key`` /
``model``), so a session's LLM calls flow through AReaL's inference service
under a per-session ``sk-sess-*`` key and get captured for training.  Outside
training (e.g. the interactive demo) it falls back to the ``HERMES_UPSTREAM_*``
environment variables.

Environment variables
---------------------
    HERMES_UPSTREAM_BASE_URL  — upstream LLM base URL (fallback when no
        ``areal_inference`` metadata).
    HERMES_UPSTREAM_API_KEY   — upstream API key.
    HERMES_UPSTREAM_MODEL     — upstream model id (default ``default``).
    HERMES_MAX_TURNS          — optional override of Hermes' iteration budget.
    HERMES_ENABLED_TOOLSETS   — comma-separated toolsets to enable (optional).
    HERMES_DISABLED_TOOLSETS  — comma-separated toolsets to disable (optional).
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial, wraps
from typing import Any

from examples.task_workflows.trace import CallTrace, observe_openai

from areal.utils import logging
from areal.v2.agent_service.types import (
    AgentRequest,
    AgentResponse,
    EventEmitter,
    StreamResponse,
)

logger = logging.getLogger("HermesAgent")


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


def _split_csv(value: str) -> list[str] | None:
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


@dataclass
class _Upstream:
    """Upstream LLM that a Hermes ``AIAgent`` routes its calls to."""

    base_url: str
    api_key: str
    model: str
    episode: dict[str, Any] | None = None

    @classmethod
    def from_inference(cls, meta: dict[str, Any]) -> _Upstream | None:
        """Build an upstream from a turn's ``metadata['areal_inference']``.

        The DataProxy injects this when a turn opts into AReaL's inference
        service, so the session's LLM calls flow through the gateway under a
        per-session ``sk-sess-*`` key and get captured for training.
        """
        base_url = meta.get("base_url") or ""
        api_key = meta.get("api_key") or ""
        if not base_url or not api_key:
            return None
        model = meta.get("model") or "default"
        return cls(
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            model=model,
            episode=dict(meta["episode"]) if meta.get("episode") is not None else None,
        )

    @classmethod
    def from_env(cls) -> _Upstream | None:
        base_url = os.environ.get("HERMES_UPSTREAM_BASE_URL", "")
        api_key = os.environ.get("HERMES_UPSTREAM_API_KEY", "")
        if not base_url or not api_key:
            return None
        model = os.environ.get("HERMES_UPSTREAM_MODEL", "default")
        return cls(base_url=base_url.rstrip("/"), api_key=api_key, model=model)


@dataclass
class _SessionState:
    session_key: str
    agent: Any  # Hermes ``AIAgent`` instance.
    upstream: _Upstream | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class GSM8KAgent:
    """One math completion through Agent Service and the captured RL upstream.

    Supply the existing GSM8K dataset's formatted prompt as ``message`` and
    generation settings as ``metadata['generation']``. The caller keeps the
    gold answer, scores the returned text with ``gsm8k_reward_fn``, and submits
    that reward with this response's ``completion_id`` and episode identity.
    """

    async def run(
        self, request: AgentRequest, *, emitter: EventEmitter
    ) -> AgentResponse:
        from openai import AsyncOpenAI

        metadata = request.metadata or {}
        upstream = _Upstream.from_inference(metadata.get("areal_inference") or {})
        if upstream is None:
            raise ValueError("GSM8KAgent requires explicit AReaL inference routing")
        generation = metadata.get("generation")
        if not isinstance(generation, dict) or (
            set(generation) - {"max_tokens", "temperature", "top_p", "seed"}
        ):
            raise ValueError("generation requires explicit supported sampling settings")
        if (
            type(generation.get("max_tokens")) is not int
            or generation["max_tokens"] < 1
        ):
            raise ValueError("generation.max_tokens must be a positive integer")
        if not request.message or request.history:
            raise ValueError("Each GSM8K episode requires one independent prompt")
        messages = [{"role": "user", "content": request.message}]
        extra = (
            {"extra_body": {"areal_episode": upstream.episode}}
            if upstream.episode is not None
            else {}
        )
        # A retry after an uncertain completion can produce a second sample;
        # leave retry/abort decisions to the episode protocol instead.
        async with AsyncOpenAI(
            base_url=upstream.base_url,
            api_key=upstream.api_key,
            timeout=120.0,
            max_retries=0,
        ) as client:
            completion = await client.chat.completions.create(
                model=upstream.model,
                messages=messages,
                n=1,
                **generation,
                **extra,
            )
        if len(completion.choices) != 1:
            raise RuntimeError("A GSM8K episode requires exactly one completion")
        choice = completion.choices[0]
        text = choice.message.content
        if text is None:
            raise RuntimeError("The math completion contains no answer text")
        await emitter.emit_delta(text)
        return AgentResponse(
            summary=text,
            metadata={
                "completion_id": completion.id,
                "finish_reason": choice.finish_reason,
                "usage": completion.usage.model_dump() if completion.usage else None,
                "areal_episode": upstream.episode,
            },
        )


class HermesAgent:
    """AgentRunnable that runs one in-process Hermes ``AIAgent`` per session."""

    def __init__(self, **_: Any) -> None:
        max_turns = os.environ.get("HERMES_MAX_TURNS")
        self._max_turns = int(max_turns) if max_turns is not None else None
        self._enabled_toolsets = _split_csv(
            os.environ.get("HERMES_ENABLED_TOOLSETS", "")
        )
        self._disabled_toolsets = _split_csv(
            os.environ.get("HERMES_DISABLED_TOOLSETS", "")
        )
        self._env_upstream = _Upstream.from_env()

        self._sessions: dict[str, _SessionState] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()

        logger.info(
            "HermesAgent initialized (max_turns=%s, env_upstream=%s)",
            self._max_turns,
            self._env_upstream is not None,
        )

    # ------------------------------------------------------------------
    # AIAgent lifecycle
    # ------------------------------------------------------------------

    def _build_agent(
        self, upstream: _Upstream, *, enabled_toolsets: list[str] | None = None
    ) -> Any:
        """Construct a headless Hermes ``AIAgent`` bound to ``upstream``.

        Imported lazily so this module is importable (and unit-testable)
        without the ``hermes-agent`` package installed.  The persistence and
        context-injection flags are disabled so the agent stays stateless
        across turns — AReaL's DataProxy owns the conversation history.
        """
        from run_agent import AIAgent  # type: ignore[import-not-found]

        kwargs = (
            {"request_overrides": {"extra_body": {"areal_episode": upstream.episode}}}
            if upstream.episode is not None
            else {}
        )
        if self._max_turns is not None:
            kwargs["max_iterations"] = self._max_turns
        agent = AIAgent(
            base_url=upstream.base_url,
            api_key=upstream.api_key,
            model=upstream.model,
            enabled_toolsets=self._enabled_toolsets
            if enabled_toolsets is None
            else enabled_toolsets,
            disabled_toolsets=self._disabled_toolsets,
            save_trajectories=False,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            session_db=None,
            **kwargs,
        )
        if upstream.episode is not None:
            _bind_episode_requests(agent, upstream.episode)
        return agent

    async def _session_lock(self, session_key: str) -> asyncio.Lock:
        async with self._locks_guard:
            lock = self._locks.get(session_key)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[session_key] = lock
            return lock

    async def _teardown_state(self, state: _SessionState) -> None:
        # ``AIAgent`` cleanup is best-effort: not every build exposes a
        # ``close``, and a failing teardown must not mask the caller's intent.
        close = getattr(state.agent, "close", None)
        if callable(close):
            try:
                result = close()
                if asyncio.iscoroutine(result):
                    await result
            except Exception:  # noqa: BLE001 - teardown must not raise
                logger.debug("AIAgent.close() failed", exc_info=True)

    async def _ensure_session(
        self,
        session_key: str,
        upstream: _Upstream,
        *,
        rebuild: bool = False,
    ) -> _SessionState:
        lock = await self._session_lock(session_key)
        async with lock:
            existing = self._sessions.get(session_key)
            if existing is not None:
                if not rebuild:
                    return existing
                await self._teardown_state(self._sessions.pop(session_key))
            # Building an AIAgent loads toolsets and is comparatively heavy;
            # run it off the event loop so it does not block other sessions.
            agent = await asyncio.to_thread(self._build_agent, upstream)
            state = _SessionState(
                session_key=session_key, agent=agent, upstream=upstream
            )
            self._sessions[session_key] = state
            logger.info(
                "Built Hermes AIAgent (session=%s, model=%s)",
                session_key,
                upstream.model,
            )
            return state

    # ------------------------------------------------------------------
    # Lifecycle hooks (optional members of the AgentRunnable protocol)
    # ------------------------------------------------------------------

    async def close_session(self, session_key: str) -> None:
        state = self._sessions.pop(session_key, None)
        if state is not None:
            await self._teardown_state(state)
        # Drop the per-session lock so ``_locks`` does not grow unbounded as
        # sessions are created and destroyed over a long-running worker.
        async with self._locks_guard:
            self._locks.pop(session_key, None)

    async def close_all_sessions(self) -> None:
        for key in list(self._sessions.keys()):
            await self.close_session(key)

    # ------------------------------------------------------------------
    # Per-turn execution
    # ------------------------------------------------------------------

    async def _resolve_state(self, request: AgentRequest) -> _SessionState:
        """Pick (or build) the Hermes ``AIAgent`` backing this turn.

        Session-first: an existing per-session agent is reused as-is, so a turn
        need not re-supply its upstream once the session is live.  When no agent
        exists yet, an upstream must be resolvable — preferring the per-session
        inference routing the DataProxy injected (self-evolution, calls flow
        through AReaL's inference service under a ``sk-sess-*`` key), else the
        process-wide env upstream (e.g. the interactive demo).  If a live
        session's upstream changes (env → AReaL inference), rebuild so the agent
        routes to the new endpoint/key.
        """
        inf_meta = (request.metadata or {}).get("areal_inference")
        state = self._sessions.get(request.session_key)
        if (
            state is not None
            and state.upstream is not None
            and state.upstream.episode is not None
            and not (inf_meta or {}).get("episode")
        ):
            raise RuntimeError(
                "Every on-demand Hermes turn requires an episode identity"
            )
        upstream = (
            _Upstream.from_inference(inf_meta) if inf_meta else None
        ) or self._env_upstream

        if state is None:
            if upstream is None:
                raise RuntimeError(
                    f"No agent for session '{request.session_key}': send the turn "
                    "with inference-routing fields ('inf_base_url' + "
                    "'session_api_key', so the DataProxy injects an inference "
                    "upstream) or set the HERMES_UPSTREAM_* env."
                )
            return await self._ensure_session(request.session_key, upstream)
        if upstream is not None and state.upstream != upstream:
            return await self._ensure_session(
                request.session_key, upstream, rebuild=True
            )
        return state

    async def _run_conversation(
        self,
        state: _SessionState,
        message: str,
        history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Run one blocking Hermes turn off the event loop, serialized per session.

        ``run_conversation`` is synchronous and mutates agent state, so the
        per-session lock keeps a session's turns from overlapping.  The full
        replayed history is passed every turn (Hermes rebuilds its message list
        from it), keeping the DataProxy's history the single source of truth.
        """
        lock = await self._session_lock(state.session_key)
        async with lock:
            if self._sessions.get(state.session_key) is not state:
                raise RuntimeError(
                    "This turn's Hermes instance was replaced by a newer episode"
                )
            return await asyncio.to_thread(
                state.agent.run_conversation,
                message,
                conversation_history=history or None,
                task_id=f"areal-{uuid.uuid4().hex}",
            )

    @staticmethod
    def _collect_tool_calls(result: dict[str, Any]) -> list[dict[str, Any]]:
        """Surface tool calls from the result transcript for observability.

        Like the OpenClaw runtime, tool calls are reported in ``metadata`` only
        and deliberately NOT emitted as ``tool_call`` events: Hermes executes
        tools internally, so emitting a ``tool_call`` without the framework also
        feeding a paired ``tool_result`` would make the DataProxy build an
        invalid chat-completions history that the upstream rejects on replay.
        """
        tool_calls: list[dict[str, Any]] = []
        for msg in result.get("messages") or []:
            if msg.get("role") != "assistant":
                continue
            for call in msg.get("tool_calls") or []:
                fn = call.get("function") or {}
                tool_calls.append(
                    {
                        "name": fn.get("name", ""),
                        "input": fn.get("arguments", ""),
                    }
                )
        return tool_calls

    async def run(
        self,
        request: AgentRequest,
        *,
        emitter: EventEmitter,
    ) -> AgentResponse | StreamResponse:
        """Single entry point for both the structured and chat-completions channels.

        The ``/v1/chat/completions`` bridge stashes the original request body
        under ``metadata['chat_request']``; its presence selects the
        chat-completions channel (return a :class:`StreamResponse` carrying an
        OpenAI-compatible response synthesized from Hermes' final text).
        Otherwise the structured channel runs: report the final text through
        ``emitter`` and return an :class:`AgentResponse`.
        """
        state = await self._resolve_state(request)

        chat_request = (request.metadata or {}).get("chat_request")
        if chat_request is not None:
            return await self._chat_completions(state, dict(chat_request))

        result = await self._run_conversation(state, request.message, request.history)
        final = result.get("final_response") or ""
        if final:
            await emitter.emit_delta(final)

        return AgentResponse(
            summary=final,
            metadata={
                "tool_calls": self._collect_tool_calls(result),
                "completed": bool(result.get("completed")),
                "api_calls": result.get("api_calls", 0),
            },
        )

    # ------------------------------------------------------------------
    # Chat-completions channel
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_chat(chat_body: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
        """Split a chat-completions body into (last user message, prior history)."""
        messages = list(chat_body.get("messages") or [])
        message = ""
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].get("role") == "user":
                content = messages[idx].get("content", "")
                if isinstance(content, list):
                    content = "".join(
                        block.get("text", "")
                        for block in content
                        if isinstance(block, dict) and block.get("type") == "text"
                    )
                message = content or ""
                history = messages[:idx]
                return message, history
        return message, messages

    async def _chat_completions(
        self,
        state: _SessionState,
        chat_body: dict[str, Any],
    ) -> StreamResponse:
        """Chat-completions channel for the ``/v1/chat/completions`` bridge.

        Hermes has no OpenAI-compatible HTTP server to relay byte-for-byte, so
        this runs the structured turn and re-encodes Hermes' final text into the
        OpenAI chat-completions wire format (SSE when ``stream`` is set, a single
        JSON object otherwise), returned as a :class:`StreamResponse` the Worker
        and DataProxy relay verbatim.
        """
        message, history = self._extract_chat(chat_body)
        result = await self._run_conversation(state, message, history)
        final = result.get("final_response") or ""
        model = chat_body.get("model") or (
            state.upstream.model if state.upstream else "hermes-agent"
        )
        stream = bool(chat_body.get("stream"))

        if stream:
            return self._chat_sse_response(final, model)
        return self._chat_json_response(final, model)

    @staticmethod
    def _chat_json_response(content: str, model: str) -> StreamResponse:
        payload = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        }
        body = json.dumps(payload).encode("utf-8")

        async def _iter() -> Any:
            yield body

        return StreamResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            body=_iter(),
        )

    @staticmethod
    def _chat_sse_response(content: str, model: str) -> StreamResponse:
        cmpl_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        def _chunk(delta: dict[str, Any], finish: str | None) -> bytes:
            payload = {
                "id": cmpl_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            return f"data: {json.dumps(payload)}\n\n".encode()

        async def _iter() -> Any:
            yield _chunk({"role": "assistant"}, None)
            if content:
                yield _chunk({"content": content}, None)
            yield _chunk({}, "stop")
            yield b"data: [DONE]\n\n"

        return StreamResponse(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            body=_iter(),
        )


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
