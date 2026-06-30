"""Unit tests for the Hermes agent runtime.

These tests exercise :class:`HermesAgent` without the real ``hermes-agent``
package or a GPU: a stub ``AIAgent`` (whose ``run_conversation`` returns a
canned result) is injected per session, and the missing-binary path is driven
by leaving ``_build_agent`` unpatched while no session exists.
"""

from __future__ import annotations

import json

import pytest
from examples.agent_service.hermes.hermes import (
    HermesAgent,
    _SessionState,
    _Upstream,
)

from areal.v2.agent_service.types import AgentRequest


class _RecordingEmitter:
    def __init__(self) -> None:
        self.deltas: list[str] = []
        self.tool_calls: list[tuple[str, str]] = []
        self.tool_results: list[tuple[str, str]] = []

    async def emit_delta(self, text: str) -> None:
        self.deltas.append(text)

    async def emit_tool_call(self, name: str, args: str) -> None:
        self.tool_calls.append((name, args))

    async def emit_tool_result(self, name: str, result: str) -> None:
        self.tool_results.append((name, result))


class _StubAIAgent:
    """Stand-in for a Hermes ``AIAgent`` that records calls and returns a canned result."""

    def __init__(self, result: dict) -> None:
        self._result = result
        self.calls: list[dict] = []

    def run_conversation(
        self, user_message, conversation_history=None, task_id=None, **_
    ):
        self.calls.append(
            {
                "user_message": user_message,
                "conversation_history": conversation_history,
                "task_id": task_id,
            }
        )
        return self._result


def _attach_stub(agent: HermesAgent, session_key: str, result: dict) -> _StubAIAgent:
    stub = _StubAIAgent(result)
    agent._sessions[session_key] = _SessionState(
        session_key=session_key,
        agent=stub,
        upstream=_Upstream(base_url="http://u", api_key="k", model="m"),
    )
    return stub


@pytest.mark.asyncio
async def test_run_returns_final_response_and_tool_calls():
    """Structured run emits the final text once and surfaces tool calls in metadata.

    Tool calls are reported in ``metadata['tool_calls']`` for observability but
    are deliberately *not* emitted as ``tool_call`` events: Hermes executes
    tools internally and the framework does not feed a paired ``tool_result``,
    so emitting one would build an invalid replay history.
    """
    result = {
        "final_response": "Hello world",
        "completed": True,
        "api_calls": 3,
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "search", "arguments": '{"q":"hi"}'}}
                ],
            },
            {"role": "tool", "content": "ok"},
            {"role": "assistant", "content": "Hello world"},
        ],
    }
    agent = HermesAgent()
    stub = _attach_stub(agent, "s1", result)
    emitter = _RecordingEmitter()
    req = AgentRequest(
        message="hi",
        session_key="s1",
        run_id="r1",
        history=[{"role": "user", "content": "earlier"}],
    )

    resp = await agent.run(req, emitter=emitter)
    await agent.close_all_sessions()

    assert resp.summary == "Hello world"
    assert emitter.deltas == ["Hello world"]
    assert emitter.tool_calls == []
    assert resp.metadata["tool_calls"] == [{"name": "search", "input": '{"q":"hi"}'}]
    assert resp.metadata["completed"] is True
    assert resp.metadata["api_calls"] == 3
    # The replayed history is forwarded verbatim as the single source of truth.
    assert stub.calls[0]["conversation_history"] == [
        {"role": "user", "content": "earlier"}
    ]
    assert stub.calls[0]["user_message"] == "hi"


@pytest.mark.asyncio
async def test_chat_request_synthesizes_streaming_completion():
    """A streaming chat_request yields an OpenAI chat-completions SSE StreamResponse."""
    agent = HermesAgent()
    _attach_stub(agent, "s2", {"final_response": "Hi there", "messages": []})
    chat_body = {
        "model": "hermes-x",
        "stream": True,
        "messages": [
            {"role": "user", "content": "prev"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "now"},
        ],
    }
    req = AgentRequest(
        message="now",
        session_key="s2",
        run_id="r1",
        history=[],
        metadata={"chat_request": chat_body},
    )

    resp = await agent.run(req, emitter=_RecordingEmitter())
    chunks = b"".join([chunk async for chunk in resp.body]).decode("utf-8")
    await agent.close_all_sessions()

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "text/event-stream"
    assert "data: [DONE]" in chunks
    # The assistant content delta carries Hermes' final text.
    content_lines = [
        json.loads(line[len("data: ") :])
        for line in chunks.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    contents = [
        c["choices"][0]["delta"].get("content")
        for c in content_lines
        if c["choices"][0]["delta"].get("content")
    ]
    assert contents == ["Hi there"]
    assert content_lines[0]["object"] == "chat.completion.chunk"


@pytest.mark.asyncio
async def test_chat_request_non_streaming_json():
    """A non-streaming chat_request yields a single chat.completion JSON object."""
    agent = HermesAgent()
    _attach_stub(agent, "s3", {"final_response": "Done", "messages": []})
    chat_body = {
        "model": "hermes-x",
        "messages": [{"role": "user", "content": "go"}],
    }
    req = AgentRequest(
        message="go",
        session_key="s3",
        run_id="r1",
        history=[],
        metadata={"chat_request": chat_body},
    )

    resp = await agent.run(req, emitter=_RecordingEmitter())
    body = b"".join([chunk async for chunk in resp.body]).decode("utf-8")
    await agent.close_all_sessions()

    assert resp.headers["content-type"] == "application/json"
    payload = json.loads(body)
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["content"] == "Done"
    assert payload["choices"][0]["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_extract_chat_splits_last_user_message():
    """_extract_chat returns the last user text and the preceding history."""
    message, history = HermesAgent._extract_chat(
        {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "reply"},
                {"role": "user", "content": "second"},
            ]
        }
    )
    assert message == "second"
    assert history == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
    ]


@pytest.mark.asyncio
async def test_close_session_drops_lock():
    """close_session removes the per-session lock to bound _locks growth."""
    agent = HermesAgent()
    await agent._session_lock("sX")
    assert "sX" in agent._locks

    await agent.close_session("sX")
    assert "sX" not in agent._locks

    # idempotent: closing an already-closed session is safe
    await agent.close_session("sX")
    assert "sX" not in agent._locks


@pytest.mark.asyncio
async def test_run_without_session_or_env_raises():
    """run() with no live session and no env upstream raises clearly."""
    agent = HermesAgent()
    agent._env_upstream = None
    req = AgentRequest(message="hi", session_key="missing", run_id="r1", history=[])
    with pytest.raises(RuntimeError, match="No agent for session"):
        await agent.run(req, emitter=_RecordingEmitter())
