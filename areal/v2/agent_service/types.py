# SPDX-License-Identifier: Apache-2.0

"""Public types for the Agent Service protocol."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .protocol import QueueMode


@dataclass(frozen=True, slots=True)
class MemoryTurnResultV1:
    """Consumer output plus a ledger pointer for one actual Memory exposure.

    The capability deliberately returns neither an assignment nor a writable
    acknowledgement object.  ``exposure_id`` and its content hash let trusted
    evaluation code join the result back to the Memory ledger; they are not
    proof by themselves and grant no further Memory authority.
    """

    output: object = field(repr=False, compare=False)
    exposure_id: str
    exposure_content_sha256: str

    def __post_init__(self) -> None:
        if type(self.exposure_id) is not str or not self.exposure_id.strip():
            raise ValueError("exposure_id must be a non-blank str")
        digest = self.exposure_content_sha256
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(
                "exposure_content_sha256 must be a lowercase SHA-256 hex digest"
            )
        if self.exposure_id != f"mexp_{digest[:24]}":
            raise ValueError("exposure_id disagrees with exposure_content_sha256")


class MemoryTurnCapability(Protocol):
    """Turn-scoped, read-only access to the configured Memory consumer.

    The capability is already bound to a scope, assignment, session, and turn.
    An agent may choose only an idempotent operation key plus query/history
    bytes.  It cannot choose a scope or consumer and cannot create an
    acknowledgement or exposure.  The call returns only after the trusted
    runtime consumer has produced a canonical acknowledgement and the ledger
    has atomically committed the resulting exposure.
    """

    async def expose_memory(
        self,
        operation_key: str,
        *,
        query: bytes,
        history: tuple[bytes, ...] = (),
    ) -> MemoryTurnResultV1: ...


@dataclass
class AgentRequest:
    """Structured request passed to the agent.

    Core fields are stable protocol-level attributes.  Framework-specific
    parameters should go in *metadata*.

    Reserved metadata keys:
        ``areal_memory``: DataProxy-authored, closed-schema assignment-pin
            envelope requested through the trusted control-plane field.
            Callers cannot set this key through ordinary metadata.  This
            transport layer does not yet resolve it in the Worker; a future
            Worker-owned integration must do so before handing an agent a
            narrow Memory turn capability.  The raw envelope may therefore
            still be visible to an agent plugin.  It is neither caller
            authorization nor an exposure receipt and does not mean Memory
            reached a model.  Ingress must separately authorize the
            principal/session for its MemoryScope.
        ``areal_inference``: DataProxy-authored when the turn opts into AReaL's
            own inference service for self-evolution (the turn carries the
            top-level routing fields ``inf_base_url`` + ``session_api_key``).
            Value is ``{"base_url", "api_key", "model"}`` where ``api_key`` is
            the per-session ``sk-sess-*`` the **caller** obtained itself (e.g.
            via its own ``/rl/start_session``) and passed in on the request.  The
            Agent Service does not talk to the training side; it merely
            forwards these fields through.  Agents should route their internal
            LLM calls to this upstream so the trajectory's tokens/logprobs are
            captured for training.
        ``chat_request``: DataProxy-authored on the ``/v1/chat/completions``
            path; the original request body with Agent Service control fields
            removed, so an agent that fronts an OpenAI-compatible upstream can
            safely replay it and return a :class:`StreamResponse` for
            byte-for-byte relay.
    """

    message: str
    session_key: str
    run_id: str
    history: list[dict[str, Any]] = field(default_factory=list)
    queue_mode: QueueMode = QueueMode.COLLECT
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def memory(self) -> MemoryTurnCapability | None:
        """Return the host-bound runtime capability, if this turn has one.

        This is intentionally not a dataclass field: wire serializers such as
        :func:`dataclasses.asdict` must never traverse a live coordinator or
        include process-local authority in a request payload.
        """

        return getattr(self, "_areal_memory_turn_capability", None)


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
