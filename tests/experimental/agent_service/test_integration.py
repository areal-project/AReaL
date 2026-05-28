"""Integration tests for the Agent Service.

Tests the full HTTP microservice stack: Worker → DataProxy → Router,
plus utility functions from the Bridge and Gateway health endpoints.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from areal.experimental.agent_service.auth import DEFAULT_ADMIN_API_KEY, admin_headers
from areal.experimental.agent_service.data_proxy.app import create_data_proxy_app
from areal.experimental.agent_service.data_proxy.config import DataProxyConfig
from areal.experimental.agent_service.gateway.app import create_gateway_app
from areal.experimental.agent_service.gateway.bridge import OpenResponsesBridge
from areal.experimental.agent_service.gateway.config import GatewayConfig
from areal.experimental.agent_service.router.app import create_router_app
from areal.experimental.agent_service.router.config import RouterConfig
from areal.experimental.agent_service.types import (
    AgentRequest,
    AgentResponse,
    EventEmitter,
)
from areal.experimental.agent_service.worker.app import create_worker_app

httpx = pytest.importorskip("httpx")

_AUTH = admin_headers(DEFAULT_ADMIN_API_KEY)


class _EchoAgent:
    async def run(
        self, request: AgentRequest, *, emitter: EventEmitter
    ) -> AgentResponse:
        history_summary = f"history={len(request.history)}"
        await emitter.emit_delta(f"echo: {request.message} ({history_summary})")
        return AgentResponse(summary=f"echo: {request.message}")


class _ToolAgent:
    async def run(
        self, request: AgentRequest, *, emitter: EventEmitter
    ) -> AgentResponse:
        await emitter.emit_tool_call("lookup", '{"id": "123"}')
        await emitter.emit_tool_result("lookup", '{"status": "ok"}')
        await emitter.emit_delta("Lookup complete")
        return AgentResponse(
            summary="Lookup complete",
            metadata={"tool_calls": [{"name": "lookup", "arguments": {"id": "123"}}]},
        )


def _make_worker_app(agent_cls):
    with patch(
        "areal.experimental.agent_service.worker.app.import_from_string",
        return_value=agent_cls,
    ):
        return create_worker_app("mock.path")


class TestWorkerDataProxyIntegration:
    """Test DataProxy → Worker chain using ASGITransport for the Worker."""

    @pytest.mark.asyncio
    async def test_single_turn(self):
        worker_app = _make_worker_app(_EchoAgent)
        worker_transport = httpx.ASGITransport(app=worker_app)

        async with httpx.AsyncClient(
            transport=worker_transport, base_url="http://worker"
        ) as worker_client:
            # DataProxy forwards to worker — test worker directly first
            resp = await worker_client.post(
                "/run",
                json={
                    "message": "hello",
                    "session_key": "s1",
                    "run_id": "r1",
                    "history": [],
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "echo: hello" in data["summary"]

    @pytest.mark.asyncio
    async def test_data_proxy_manages_history(self):
        worker_app = _make_worker_app(_EchoAgent)
        worker_transport = httpx.ASGITransport(app=worker_app)

        # Create DataProxy pointing to worker
        proxy_app = create_data_proxy_app(DataProxyConfig(worker_addr="http://worker"))

        # Patch DataProxy's httpx client to use worker's ASGITransport
        original_post = httpx.AsyncClient.post

        async def patched_post(self, url, **kwargs):
            if "worker" in url:
                async with httpx.AsyncClient(
                    transport=worker_transport, base_url="http://worker"
                ) as wc:
                    path = url.split("http://worker")[-1]
                    return await wc.post(path, **kwargs)
            return await original_post(self, url, **kwargs)

        proxy_transport = httpx.ASGITransport(app=proxy_app)

        with patch.object(httpx.AsyncClient, "post", patched_post):
            async with httpx.AsyncClient(
                transport=proxy_transport, base_url="http://proxy"
            ) as proxy_client:
                # Turn 1
                r1 = await proxy_client.post(
                    "/session/s1/turn",
                    json={"message": "hello", "run_id": "r1"},
                )
                assert r1.status_code == 200
                assert "echo: hello" in r1.json()["summary"]

                # Turn 2 — history should have turn 1
                r2 = await proxy_client.post(
                    "/session/s1/turn",
                    json={"message": "world", "run_id": "r2"},
                )
                assert r2.status_code == 200

                # Check history grew
                h = await proxy_client.get("/session/s1/history")
                history = h.json()["history"]
                assert len(history) >= 2  # at least user+assistant from turn 1

    @pytest.mark.asyncio
    async def test_close_session_clears_history(self):
        worker_app = _make_worker_app(_EchoAgent)
        worker_transport = httpx.ASGITransport(app=worker_app)
        proxy_app = create_data_proxy_app(DataProxyConfig(worker_addr="http://worker"))

        original_post = httpx.AsyncClient.post

        async def patched_post(self, url, **kwargs):
            if "worker" in url:
                async with httpx.AsyncClient(
                    transport=worker_transport, base_url="http://worker"
                ) as wc:
                    path = url.split("http://worker")[-1]
                    return await wc.post(path, **kwargs)
            return await original_post(self, url, **kwargs)

        proxy_transport = httpx.ASGITransport(app=proxy_app)

        with patch.object(httpx.AsyncClient, "post", patched_post):
            async with httpx.AsyncClient(
                transport=proxy_transport, base_url="http://proxy"
            ) as proxy_client:
                await proxy_client.post(
                    "/session/s1/turn",
                    json={"message": "hi", "run_id": "r1"},
                )
                await proxy_client.post("/session/s1/close")
                h = await proxy_client.get("/session/s1/history")
                assert h.json()["history"] == []


class TestRouterIntegration:
    @pytest.mark.asyncio
    async def test_register_and_route(self):
        router_app = create_router_app(
            RouterConfig(admin_api_key=DEFAULT_ADMIN_API_KEY)
        )
        transport = httpx.ASGITransport(app=router_app)

        async with httpx.AsyncClient(
            transport=transport, base_url="http://router"
        ) as client:
            await client.post(
                "/register",
                json={"addr": "http://proxy1:9100"},
                headers=_AUTH,
            )
            resp = await client.post(
                "/route", json={"session_key": "s1"}, headers=_AUTH
            )
            assert resp.json()["data_proxy_addr"] == "http://proxy1:9100"

            resp2 = await client.post(
                "/route", json={"session_key": "s1"}, headers=_AUTH
            )
            assert resp2.json()["data_proxy_addr"] == "http://proxy1:9100"


class TestToolCallFlow:
    @pytest.mark.asyncio
    async def test_tool_events_through_proxy(self):
        worker_app = _make_worker_app(_ToolAgent)
        worker_transport = httpx.ASGITransport(app=worker_app)
        proxy_app = create_data_proxy_app(DataProxyConfig(worker_addr="http://worker"))

        original_post = httpx.AsyncClient.post

        async def patched_post(self, url, **kwargs):
            if "worker" in url:
                async with httpx.AsyncClient(
                    transport=worker_transport, base_url="http://worker"
                ) as wc:
                    path = url.split("http://worker")[-1]
                    return await wc.post(path, **kwargs)
            return await original_post(self, url, **kwargs)

        proxy_transport = httpx.ASGITransport(app=proxy_app)

        with patch.object(httpx.AsyncClient, "post", patched_post):
            async with httpx.AsyncClient(
                transport=proxy_transport, base_url="http://proxy"
            ) as proxy_client:
                resp = await proxy_client.post(
                    "/session/s1/turn",
                    json={"message": "lookup 123", "run_id": "r1"},
                )
                data = resp.json()
                assert data["summary"] == "Lookup complete"
                events = data["events"]
                types = {e["type"] for e in events}
                assert "tool_call" in types
                assert "tool_result" in types

                # History should include tool call records
                h = await proxy_client.get("/session/s1/history")
                history = h.json()["history"]
                tool_msgs = [m for m in history if m.get("role") == "tool"]
                assert len(tool_msgs) > 0
                assert "tool_call_id" in tool_msgs[0]


class TestGatewayHealth:
    @pytest.mark.asyncio
    async def test_health(self):
        app = create_gateway_app(GatewayConfig(router_addr="http://fake-router"))
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://gw"
        ) as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"


class TestBridgeExtractMessage:
    def test_text_message(self):
        items = [
            {
                "type": "message",
                "content": [{"type": "input_text", "text": "Hello"}],
            }
        ]
        assert OpenResponsesBridge._extract_message(items, "") == "Hello"

    def test_string_content(self):
        items = [{"type": "message", "content": "Simple"}]
        assert OpenResponsesBridge._extract_message(items, "") == "Simple"

    def test_instructions_prepended(self):
        items = [{"type": "message", "content": "Hi"}]
        result = OpenResponsesBridge._extract_message(items, "Be helpful")
        assert result.startswith("Be helpful")
        assert "Hi" in result

    def test_function_call_output(self):
        items = [{"type": "function_call_output", "output": "42"}]
        result = OpenResponsesBridge._extract_message(items, "")
        assert "[tool result] 42" in result


class TestBridgeDeriveSessionKey:
    def test_with_user(self):
        key = OpenResponsesBridge._derive_session_key("user1", "model1")
        assert key == "agent:model1:user1"

    def test_without_user_is_unique(self):
        k1 = OpenResponsesBridge._derive_session_key("", "m")
        k2 = OpenResponsesBridge._derive_session_key("", "m")
        assert k1 != k2
        assert k1.startswith("agent:m:")

    def test_default_model(self):
        key = OpenResponsesBridge._derive_session_key("u1", "")
        assert key == "agent:default:u1"
