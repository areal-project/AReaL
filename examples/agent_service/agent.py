"""Claude Agent for AReaL Agent Service.

Implements :class:`AgentRunnable` using the Claude Agent SDK
(``claude-agent-sdk``).  Each Worker instance holds a pool of
:class:`ClaudeSDKClient` sessions keyed by ``session_key``, so multi-turn
conversations preserve full context without re-sending history.

Requires::

    pip install claude-agent-sdk

Environment variables:
    ANTHROPIC_API_KEY   — Anthropic API key (required)
    CLAUDE_MODEL        — model name (default: claude-sonnet-4-6)
    CLAUDE_SYSTEM_PROMPT — optional system prompt override
    CLAUDE_MAX_TURNS    — max agentic turns per query (default: 20)
"""

from __future__ import annotations

import os
from typing import Any, Literal

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
)

from areal.experimental.agent_service.types import (
    AgentRequest,
    AgentResponse,
    EventEmitter,
)
from areal.utils import logging

logger = logging.getLogger("ClaudeAgent")

PermissionMode = Literal["default", "acceptEdits", "plan", "bypassPermissions"]

_DEFAULT_PERMISSION_MODE: PermissionMode = "bypassPermissions"


class ClaudeAgent:
    """AgentRunnable backed by the Claude Agent SDK.

    Maintains a ``ClaudeSDKClient`` per session for true multi-turn
    continuity — the SDK's internal session keeps the full transcript,
    so ``request.history`` is only used for the very first turn of a
    new session (to seed context if provided by the caller).
    """

    def __init__(self, **kwargs: Any) -> None:
        self._model = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
        self._system_prompt = os.environ.get("CLAUDE_SYSTEM_PROMPT", "")
        self._max_turns = int(os.environ.get("CLAUDE_MAX_TURNS", "20"))
        self._permission_mode: PermissionMode = _DEFAULT_PERMISSION_MODE

        self._sessions: dict[str, ClaudeSDKClient] = {}

        logger.info(
            "ClaudeAgent initialized (model=%s, max_turns=%d)",
            self._model,
            self._max_turns,
        )

    def _make_options(self) -> ClaudeAgentOptions:
        opts = ClaudeAgentOptions(
            model=self._model,
            max_turns=self._max_turns,
            permission_mode=self._permission_mode,
        )
        if self._system_prompt:
            opts.system_prompt = self._system_prompt
        return opts

    async def _get_or_create_client(self, session_key: str) -> ClaudeSDKClient:
        if session_key not in self._sessions:
            client = ClaudeSDKClient(options=self._make_options())
            await client.__aenter__()
            self._sessions[session_key] = client
            logger.info("New session: %s", session_key)
        return self._sessions[session_key]

    async def close_session(self, session_key: str) -> None:
        client = self._sessions.pop(session_key, None)
        if client is not None:
            try:
                await client.__aexit__(None, None, None)
            except Exception:
                logger.warning("Error closing session %s", session_key, exc_info=True)

    async def close_all_sessions(self) -> None:
        keys = list(self._sessions.keys())
        for key in keys:
            await self.close_session(key)

    async def run(
        self,
        request: AgentRequest,
        *,
        emitter: EventEmitter,
    ) -> AgentResponse:
        client = await self._get_or_create_client(request.session_key)

        try:
            await client.query(request.message)

            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []

            async for msg in client.receive_response():
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            await emitter.emit_delta(block.text)
                            text_parts.append(block.text)
                        elif isinstance(block, ToolUseBlock):
                            await emitter.emit_tool_call(
                                name=block.name,
                                args=str(block.input),
                            )
                            tool_calls.append(
                                {"name": block.name, "input": block.input}
                            )
                elif isinstance(msg, ResultMessage):
                    break

            summary = "".join(text_parts)
            return AgentResponse(
                summary=summary[:200],
                metadata={"tool_calls": tool_calls},
            )
        except Exception:
            await self.close_session(request.session_key)
            raise
