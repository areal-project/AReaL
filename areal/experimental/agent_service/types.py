# SPDX-License-Identifier: Apache-2.0

"""Public types for the Agent Service protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .protocol import QueueMode


@dataclass
class AgentRequest:
    """Structured request passed to the agent.

    Core fields are stable protocol-level attributes.  Framework-specific
    parameters should go in *metadata*.
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
class TrainingContext:
    """Training-side handle injected into an agent at episode start.

    Carries everything an agent needs to make its internal LLM calls flow
    through AReaL's training pipeline (token + logprob capture).  Pass to
    an agent's optional ``on_episode_start`` hook (see ``AgentRunnable``).

    Attributes:
        session_id: Opaque episode identifier for trajectory binding.
        llm_base_url: OpenAI-compatible base URL of the proxy gateway
            that captures training data.  Agents should configure their
            internal LLM client to use this as the upstream endpoint.
        llm_api_key: Per-episode key bound to ``session_id``.  Typically a
            ``sk-sess-*`` value returned by ``/rl/start_session``.
        llm_model: Model identifier to use for inference.  Empty string
            means "agent decides".
        extras: Implementation-specific overrides (e.g. tool_call_parser).
    """

    session_id: str
    llm_base_url: str
    llm_api_key: str
    llm_model: str = ""
    extras: dict[str, Any] = field(default_factory=dict)


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

    Only ``run`` is required.  The following methods are optional and
    discovered via ``getattr`` at runtime — implement them to participate
    in training-related lifecycle:

    - ``async close_session(session_key)`` — release per-session state
      when a session is closed by the DataProxy.
    - ``async close_all_sessions()`` — clean up everything on worker
      shutdown.
    - ``async on_episode_start(session_key, training_ctx)`` — receive a
      :class:`TrainingContext` so the agent can route its internal LLM
      calls through the proxy gateway.  Called once per RL episode,
      before the first ``run`` of that episode.
    - ``async on_episode_end(session_key, reward)`` — called once when
      an episode terminates (with the final scalar reward, if any).
      Agents wiring trajectories to the training pipeline should flush
      / finalize here.
    """

    async def run(
        self,
        request: AgentRequest,
        *,
        emitter: EventEmitter,
    ) -> AgentResponse: ...
