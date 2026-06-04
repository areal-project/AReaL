"""Unit tests for the OpenClaw agent runtime.

These tests exercise :class:`OpenClawAgent` without a real ``openclaw``
binary or GPU: the per-turn SSE parsing is driven by an
``httpx.MockTransport`` and subprocess-spawn failure is triggered by
pointing ``OPENCLAW_BIN`` at a non-existent executable.
"""

from __future__ import annotations

import glob
import os
import tempfile

import pytest

from areal.experimental.agent_service.runtimes.openclaw import (
    OpenClawAgent,
    _SessionState,
    _Upstream,
)
from areal.experimental.agent_service.types import AgentRequest

httpx = pytest.importorskip("httpx")


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


class _ExitedProc:
    """Stand-in for an already-terminated subprocess used by mock sessions.

    A non-None ``returncode`` makes ``_teardown_state`` skip the kill path.
    """

    returncode = 0

    async def wait(self) -> int:
        return 0


def _attach_mock_session(agent: OpenClawAgent, session_key: str, sse: str) -> None:
    """Inject a session whose subprocess HTTP client returns ``sse``."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, text=sse, headers={"content-type": "text/event-stream"}
        )

    client = httpx.AsyncClient(
        base_url="http://mock-openclaw",
        transport=httpx.MockTransport(handler),
    )
    agent._sessions[session_key] = _SessionState(
        port=1,
        gateway_token="t",
        config_dir=tempfile.mkdtemp(prefix="openclaw-mock-"),
        process=_ExitedProc(),  # type: ignore[arg-type]
        client=client,
        log_file=None,  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
async def test_run_accumulates_streamed_tool_calls():
    """Streamed tool-call name/args are buffered by index, emitted once each."""
    sse = (
        'data: {"choices":[{"delta":{"content":"Hel"}}]}\n'
        'data: {"choices":[{"delta":{"content":"lo"}}]}\n'
        # name arrives only in the first chunk; arguments stream across chunks
        'data: {"choices":[{"delta":{"tool_calls":'
        '[{"index":0,"function":{"name":"search"}}]}}]}\n'
        'data: {"choices":[{"delta":{"tool_calls":'
        '[{"index":0,"function":{"arguments":"{\\"q\\":"}}]}}]}\n'
        # no space after "data:" — valid per the SSE spec
        'data:{"choices":[{"delta":{"tool_calls":'
        '[{"index":0,"function":{"arguments":"\\"hi\\"}"}}]}}]}\n'
        # a second tool call at a different index
        'data: {"choices":[{"delta":{"tool_calls":'
        '[{"index":1,"function":{"name":"calc","arguments":"1+1"}}]}}]}\n'
        "data: [DONE]\n"
    )
    agent = OpenClawAgent()
    _attach_mock_session(agent, "s1", sse)
    emitter = _RecordingEmitter()
    req = AgentRequest(message="hi", session_key="s1", run_id="r1", history=[])

    resp = await agent.run(req, emitter=emitter)
    await agent.close_all_sessions()

    assert resp.summary == "Hello"
    assert emitter.deltas == ["Hel", "lo"]
    # each tool call emitted exactly once, fully accumulated
    assert emitter.tool_calls == [("search", '{"q":"hi"}'), ("calc", "1+1")]
    assert resp.metadata["tool_calls"] == [
        {"name": "search", "input": '{"q":"hi"}'},
        {"name": "calc", "input": "1+1"},
    ]


@pytest.mark.asyncio
async def test_run_skips_malformed_and_indexless_chunks():
    """Malformed JSON and tool-call chunks without an index are ignored."""
    sse = (
        "data: not-json\n"
        'data: {"choices":[]}\n'
        'data: {"choices":[{"delta":{"tool_calls":'
        '[{"function":{"name":"orphan"}}]}}]}\n'  # missing index
        'data: {"choices":[{"delta":{"content":"ok"}}]}\n'
        "data: [DONE]\n"
    )
    agent = OpenClawAgent()
    _attach_mock_session(agent, "s2", sse)
    emitter = _RecordingEmitter()
    req = AgentRequest(message="hi", session_key="s2", run_id="r1", history=[])

    resp = await agent.run(req, emitter=emitter)
    await agent.close_all_sessions()

    assert resp.summary == "ok"
    assert emitter.tool_calls == []
    assert resp.metadata["tool_calls"] == []


@pytest.mark.asyncio
async def test_spawn_cleans_up_on_failure(monkeypatch):
    """A spawn failure leaks neither the temp config dir nor a file handle."""
    monkeypatch.setenv("OPENCLAW_BIN", "areal-nonexistent-openclaw-binary")
    agent = OpenClawAgent()
    upstream = _Upstream(base_url="http://upstream", api_key="k", model="m")

    tmp_glob = os.path.join(tempfile.gettempdir(), "openclaw-*")
    before = set(glob.glob(tmp_glob))

    with pytest.raises(FileNotFoundError):
        await agent._spawn("s3", upstream)

    after = set(glob.glob(tmp_glob))
    assert after - before == set(), "spawn failure leaked a temp config dir"


@pytest.mark.asyncio
async def test_close_session_drops_lock():
    """close_session removes the per-session lock to bound _locks growth."""
    agent = OpenClawAgent()
    await agent._session_lock("sX")
    assert "sX" in agent._locks

    await agent.close_session("sX")
    assert "sX" not in agent._locks

    # idempotent: closing an already-closed session is safe
    await agent.close_session("sX")
    assert "sX" not in agent._locks


@pytest.mark.asyncio
async def test_run_without_session_or_env_raises():
    """run() with no open session and no env upstream raises clearly."""
    agent = OpenClawAgent()
    agent._env_upstream = None
    req = AgentRequest(message="hi", session_key="missing", run_id="r1", history=[])
    with pytest.raises(RuntimeError, match="No subprocess for session"):
        await agent.run(req, emitter=_RecordingEmitter())
