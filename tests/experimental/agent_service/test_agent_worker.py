"""Tests for Agent Worker HTTP server."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from areal.experimental.agent_service.types import (
    AgentRequest,
    AgentResponse,
    AgentRunnable,
    EventEmitter,
)
from areal.experimental.agent_service.worker.app import (
    _CollectingEmitter,
    create_worker_app,
)

httpx = pytest.importorskip("httpx")


class _EchoAgent:
    async def run(
        self, request: AgentRequest, *, emitter: EventEmitter
    ) -> AgentResponse:
        await emitter.emit_delta(f"echo: {request.message}")
        return AgentResponse(
            summary=f"echo: {request.message}",
            metadata={"history_len": len(request.history)},
        )


class _ToolAgent:
    async def run(
        self, request: AgentRequest, *, emitter: EventEmitter
    ) -> AgentResponse:
        await emitter.emit_tool_call("search", '{"q": "test"}')
        await emitter.emit_tool_result("search", "found it")
        await emitter.emit_delta("Done")
        return AgentResponse(summary="Done")


class _FailAgent:
    async def run(
        self, request: AgentRequest, *, emitter: EventEmitter
    ) -> AgentResponse:
        raise RuntimeError("boom")


def _make_client(agent_cls):
    with patch(
        "areal.experimental.agent_service.worker.app.import_from_string",
        return_value=agent_cls,
    ):
        app = create_worker_app("mock.path")
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://worker")


class TestWorkerHealth:
    @pytest.mark.asyncio
    async def test_health(self):
        async with _make_client(_EchoAgent) as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"


class TestWorkerRun:
    @pytest.mark.asyncio
    async def test_echo(self):
        async with _make_client(_EchoAgent) as client:
            resp = await client.post(
                "/run",
                json={"message": "hello", "session_key": "s1", "run_id": "r1"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["summary"] == "echo: hello"
            assert any(e["type"] == "delta" for e in data["events"])

    @pytest.mark.asyncio
    async def test_history_forwarded(self):
        async with _make_client(_EchoAgent) as client:
            resp = await client.post(
                "/run",
                json={
                    "message": "hi",
                    "session_key": "s1",
                    "run_id": "r1",
                    "history": [{"role": "user", "content": "prev"}],
                },
            )
            assert resp.json()["metadata"]["history_len"] == 1

    @pytest.mark.asyncio
    async def test_tool_events(self):
        async with _make_client(_ToolAgent) as client:
            resp = await client.post(
                "/run",
                json={"message": "go", "session_key": "s1", "run_id": "r1"},
            )
            types = [e["type"] for e in resp.json()["events"]]
            assert "tool_call" in types
            assert "tool_result" in types
            assert "delta" in types

    @pytest.mark.asyncio
    async def test_agent_failure(self):
        async with _make_client(_FailAgent) as client:
            resp = await client.post(
                "/run",
                json={"message": "x", "session_key": "s1", "run_id": "r1"},
            )
            assert resp.status_code == 500


class TestCollectingEmitter:
    @pytest.mark.asyncio
    async def test_collects_all_event_types(self):
        e = _CollectingEmitter()
        await e.emit_delta("hi")
        await e.emit_tool_call("fn", "{}")
        await e.emit_tool_result("fn", "ok")
        assert len(e.events) == 3


class TestAgentRunnableProtocol:
    def test_echo_satisfies(self):
        assert isinstance(_EchoAgent(), AgentRunnable)

    def test_plain_object_does_not(self):
        assert not isinstance(object(), AgentRunnable)
