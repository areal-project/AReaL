# SPDX-License-Identifier: Apache-2.0

"""Public types for the Agent Service protocol."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .protocol import QueueMode


@dataclass
class AgentRequest:
    """Structured request passed to the agent.

    Core fields are stable protocol-level attributes.  Framework-specific
    parameters should go in *metadata*.

    Reserved metadata keys:
        ``areal_inference``: present when the turn opts into AReaL's own
            inference service for self-evolution (the turn carries the
            inference-routing fields ``inf_base_url`` + ``session_api_key``).
            Value is ``{"base_url", "api_key", "model"}`` where ``api_key`` is
            the per-session ``sk-sess-*`` the **caller** obtained itself (e.g.
            via its own ``/rl/start_session``) and passed in on the request.  The
            Agent Service does not talk to the training side; it merely
            forwards these fields through.  Agents should route their internal
            LLM calls to this upstream so the trajectory's tokens/logprobs are
            captured for training.
        ``chat_request``: present on the ``/v1/chat/completions`` path; the
            full original request body, so an agent that fronts an
            OpenAI-compatible upstream can replay it verbatim and return a
            :class:`StreamResponse` for byte-for-byte relay.
    """

    message: str
    session_key: str
    run_id: str
    history: list[dict[str, Any]] = field(default_factory=list)
    queue_mode: QueueMode = QueueMode.COLLECT
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    """Structured result returned by the agent."""

    summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamResponse:
    """Raw streaming response from an agent, passed through verbatim.

    The structured channel (``run`` + :class:`EventEmitter` → ``AgentResponse``)
    parses an agent's output into deltas/tool-calls.  Some agents instead expose
    an OpenAI-compatible upstream whose response (often SSE) must reach the
    caller **byte-for-byte** — re-encoding it through the structured channel
    would drop fields (tool_calls, finish_reason, usage, ...) and break clients
    that expect the exact wire format.

    An agent opts into this behaviour by returning a :class:`StreamResponse`
    (instead of an :class:`AgentResponse`) from :meth:`AgentRunnable.run`.  The
    Worker and DataProxy relay ``status_code`` / ``headers`` / ``body`` through
    the single ``/run`` and ``/session/{key}/turn`` endpoints without inspecting
    the payload — the Worker tags the response with the ``x-areal-passthrough``
    marker header so they relay it verbatim (structured turns are parsed).  The
    marker, not ``Content-Type``, drives the decision, so a *non-streaming*
    passthrough whose body is ``application/json`` is still relayed byte-for-byte
    rather than mistaken for a structured turn.
    """

    status_code: int
    headers: dict[str, str]
    body: AsyncIterator[bytes]


class EventEmitter(Protocol):
    """Callback interface for streaming events from agent to caller."""

    async def emit_delta(self, text: str) -> None: ...
    async def emit_tool_call(self, name: str, args: str) -> None: ...
    async def emit_tool_result(self, name: str, result: str) -> None: ...


@runtime_checkable
class AgentRunnable(Protocol):
    """Minimal protocol for pluggable agent implementations.

    Agent classes are loaded via
    :func:`~areal.utils.dynamic_import.import_from_string` at worker startup.
    The framework handles its own tool execution, memory, and LLM
    interaction — the Agent Service only provides session lifecycle and
    event streaming.

    ``run`` is the single entry point and may return **either** shape,
    chosen per turn by the agent:

    - :class:`AgentResponse` — the structured channel.  The agent reports
      incremental output through the ``emitter`` and returns a final
      summary/metadata; the Worker serialises it to JSON and the DataProxy
      rebuilds conversation history from the emitted events.  This backs the
      ``/v1/responses`` (and WebSocket) protocol.
    - :class:`StreamResponse` — the raw-passthrough channel.  The agent fronts
      an OpenAI-compatible upstream whose response must reach the caller
      byte-for-byte (e.g. SSE chat completions); it returns the upstream's
      ``status_code`` / ``headers`` / ``body`` and the Worker / DataProxy relay
      them verbatim.  This backs the ``/v1/chat/completions`` protocol.

    The agent decides which to return from the request (e.g. the presence of
    ``metadata['chat_request']``, or a ``stream`` flag in the original body),
    so a single ``run`` implementation can serve every protocol and both
    streaming and non-streaming modes.

    The following methods are optional and discovered via ``getattr`` at
    runtime — implement them to participate in training-related lifecycle:

    - ``async close_session(session_key)`` — release per-session state
      when a session is closed by the DataProxy.
    - ``async close_all_sessions()`` — clean up everything on worker
      shutdown.
    """

    async def run(
        self,
        request: AgentRequest,
        *,
        emitter: EventEmitter,
    ) -> AgentResponse | StreamResponse: ...
