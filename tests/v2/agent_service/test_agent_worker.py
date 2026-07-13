"""Tests for Agent Worker HTTP server."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from areal.v2.agent_service.auth import DEFAULT_ADMIN_API_KEY, admin_headers
from areal.v2.agent_service.session_keys import session_key_sha256
from areal.v2.agent_service.types import (
    AgentRequest,
    AgentResponse,
    AgentRunnable,
    EventEmitter,
)
from areal.v2.agent_service.worker.app import (
    _CollectingEmitter,
    create_worker_app,
    create_worker_app_with_hop_auth,
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


class _LifecycleAgent:
    runs = 0
    closed_sessions: list[str] = []

    async def run(
        self,
        request: AgentRequest,
        *,
        emitter: EventEmitter,
    ) -> AgentResponse:
        del request, emitter
        type(self).runs += 1
        return AgentResponse(summary="ok")

    async def close_session(self, session_key: str) -> None:
        type(self).closed_sessions.append(session_key)


class _AgentWithLegacyNamedKwarg:
    received: str | None = None

    def __init__(self, *, worker_hop_api_key: str) -> None:
        type(self).received = worker_hop_api_key

    async def run(
        self,
        request: AgentRequest,
        *,
        emitter: EventEmitter,
    ) -> AgentResponse:
        del request, emitter
        return AgentResponse(summary="ok")


def _make_client(agent_cls, *, worker_hop_api_key: str = ""):
    with patch(
        "areal.v2.agent_service.worker.app.import_from_string",
        return_value=agent_cls,
    ):
        app = (
            create_worker_app_with_hop_auth(
                "mock.path",
                worker_hop_api_key,
            )
            if worker_hop_api_key
            else create_worker_app("mock.path")
        )
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://worker")


class TestWorkerHealth:
    @pytest.mark.asyncio
    async def test_health(self):
        async with _make_client(_EchoAgent) as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_health_stays_public_when_worker_hop_auth_is_enabled(self):
        async with _make_client(
            _EchoAgent,
            worker_hop_api_key="test-worker-hop-key",
        ) as client:
            resp = await client.get("/health")
            assert resp.status_code == 200


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

    @pytest.mark.parametrize("session_key", (None, "s/b", "s%252Fb", "会话"))
    @pytest.mark.asyncio
    async def test_invalid_session_identity_never_reaches_agent(
        self,
        session_key: object,
    ):
        _LifecycleAgent.runs = 0
        async with _make_client(_LifecycleAgent) as client:
            response = await client.post(
                "/run",
                json={"message": "hello", "session_key": session_key, "run_id": "r1"},
            )

        assert response.status_code == 400
        assert _LifecycleAgent.runs == 0


class TestWorkerHopAuthentication:
    @pytest.mark.asyncio
    async def test_auth_check_proves_enforcement_instead_of_worker_health(self):
        key = "test-worker-hop-key"
        async with _make_client(_EchoAgent) as standalone_client:
            standalone = await standalone_client.get(
                "/internal/auth-check",
                headers=admin_headers(key),
            )
        assert standalone.status_code == 503

        async with _make_client(
            _EchoAgent,
            worker_hop_api_key=key,
        ) as authenticated_client:
            missing = await authenticated_client.get("/internal/auth-check")
            wrong = await authenticated_client.get(
                "/internal/auth-check",
                headers=admin_headers("wrong-key"),
            )
            accepted = await authenticated_client.get(
                "/internal/auth-check",
                headers=admin_headers(key),
            )

        assert missing.status_code == wrong.status_code == 401
        assert accepted.status_code == 200
        assert accepted.json() == {
            "status": "ok",
            "worker_hop_auth": True,
        }

    @pytest.mark.asyncio
    async def test_run_and_close_require_the_pair_credential(self):
        _LifecycleAgent.runs = 0
        _LifecycleAgent.closed_sessions.clear()
        key = "test-worker-hop-key"
        async with _make_client(
            _LifecycleAgent,
            worker_hop_api_key=key,
        ) as client:
            body = {"message": "hello", "session_key": "s1", "run_id": "r1"}
            assert (await client.post("/run", json=body)).status_code == 401
            assert (
                await client.post(
                    "/run",
                    json=body,
                    headers=admin_headers("wrong-key"),
                )
            ).status_code == 401
            assert _LifecycleAgent.runs == 0

            accepted = await client.post(
                "/run",
                json=body,
                headers=admin_headers(key),
            )
            assert accepted.status_code == 200
            assert _LifecycleAgent.runs == 1

            assert (
                await client.post(
                    "/sessions/close",
                    json={"session_key": "s1"},
                )
            ).status_code == 401
            assert _LifecycleAgent.closed_sessions == []
            invalid_close = await client.post(
                "/sessions/close",
                json={"session_key": "s%252Fb"},
                headers=admin_headers(key),
            )
            assert invalid_close.status_code == 400
            assert _LifecycleAgent.closed_sessions == []
            closed = await client.post(
                "/sessions/close",
                json={"session_key": "s1"},
                headers=admin_headers(key),
            )
            assert closed.status_code == 200
            assert closed.json() == {
                "status": "ok",
                "session_key_sha256": session_key_sha256("s1"),
            }
            assert _LifecycleAgent.closed_sessions == ["s1"]

    @pytest.mark.parametrize(
        "key",
        ("   ", DEFAULT_ADMIN_API_KEY, "areal-admin-key"),
    )
    def test_worker_rejects_unsafe_pair_credentials(self, key: str):
        with patch(
            "areal.v2.agent_service.worker.app.import_from_string",
            return_value=_EchoAgent,
        ):
            with pytest.raises(ValueError):
                create_worker_app_with_hop_auth("mock.path", key)

    @pytest.mark.asyncio
    async def test_standalone_factory_preserves_same_named_agent_kwarg(self):
        _AgentWithLegacyNamedKwarg.received = None
        with patch(
            "areal.v2.agent_service.worker.app.import_from_string",
            return_value=_AgentWithLegacyNamedKwarg,
        ):
            app = create_worker_app(
                "mock.path",
                worker_hop_api_key="agent-constructor-value",
            )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://worker",
        ) as client:
            response = await client.post(
                "/run",
                json={"message": "hello", "session_key": "s1", "run_id": "r1"},
            )
        assert response.status_code == 200
        assert _AgentWithLegacyNamedKwarg.received == "agent-constructor-value"


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
