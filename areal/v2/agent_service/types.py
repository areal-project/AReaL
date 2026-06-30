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

    Reward computation is **not** part of this interface.  Rewards are
    calculated externally by the training pipeline (via reward functions
    applied to exported trajectories), following AReaL's standard RLVR
    pattern.
    """

    async def run(
        self,
        request: AgentRequest,
        *,
        emitter: EventEmitter,
    ) -> AgentResponse: ...
