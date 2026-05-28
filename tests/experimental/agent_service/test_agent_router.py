"""Tests for Agent Router HTTP server."""

from __future__ import annotations

import pytest

from areal.experimental.agent_service.auth import DEFAULT_ADMIN_API_KEY, admin_headers
from areal.experimental.agent_service.router.app import create_router_app
from areal.experimental.agent_service.router.config import RouterConfig

httpx = pytest.importorskip("httpx")

_AUTH = admin_headers(DEFAULT_ADMIN_API_KEY)


def _make_client():
    config = RouterConfig(admin_api_key=DEFAULT_ADMIN_API_KEY)
    app = create_router_app(config)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://router")


class TestRouterHealth:
    @pytest.mark.asyncio
    async def test_health(self):
        async with _make_client() as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert data["registered_proxies"] == 0


class TestRegistration:
    @pytest.mark.asyncio
    async def test_register_and_health(self):
        async with _make_client() as client:
            await client.post(
                "/register", json={"addr": "http://proxy1:9100"}, headers=_AUTH
            )
            resp = await client.get("/health")
            assert resp.json()["registered_proxies"] == 1

    @pytest.mark.asyncio
    async def test_unregister(self):
        async with _make_client() as client:
            await client.post(
                "/register", json={"addr": "http://proxy1:9100"}, headers=_AUTH
            )
            await client.post(
                "/unregister", json={"addr": "http://proxy1:9100"}, headers=_AUTH
            )
            resp = await client.get("/health")
            assert resp.json()["registered_proxies"] == 0


class TestRouting:
    @pytest.mark.asyncio
    async def test_route_new_session(self):
        async with _make_client() as client:
            await client.post(
                "/register", json={"addr": "http://proxy1:9100"}, headers=_AUTH
            )
            resp = await client.post(
                "/route", json={"session_key": "s1"}, headers=_AUTH
            )
            assert resp.json()["data_proxy_addr"] == "http://proxy1:9100"

    @pytest.mark.asyncio
    async def test_route_existing_session_returns_same(self):
        async with _make_client() as client:
            await client.post(
                "/register", json={"addr": "http://proxy1:9100"}, headers=_AUTH
            )
            await client.post(
                "/register", json={"addr": "http://proxy2:9101"}, headers=_AUTH
            )
            r1 = await client.post("/route", json={"session_key": "s1"}, headers=_AUTH)
            r2 = await client.post("/route", json={"session_key": "s1"}, headers=_AUTH)
            assert r1.json()["data_proxy_addr"] == r2.json()["data_proxy_addr"]

    @pytest.mark.asyncio
    async def test_round_robin(self):
        async with _make_client() as client:
            await client.post("/register", json={"addr": "http://p1"}, headers=_AUTH)
            await client.post("/register", json={"addr": "http://p2"}, headers=_AUTH)
            r1 = await client.post("/route", json={"session_key": "a"}, headers=_AUTH)
            r2 = await client.post("/route", json={"session_key": "b"}, headers=_AUTH)
            addrs = {r1.json()["data_proxy_addr"], r2.json()["data_proxy_addr"]}
            assert addrs == {"http://p1", "http://p2"}

    @pytest.mark.asyncio
    async def test_remove_session(self):
        async with _make_client() as client:
            await client.post("/register", json={"addr": "http://p1"}, headers=_AUTH)
            await client.post("/route", json={"session_key": "s1"}, headers=_AUTH)
            await client.post(
                "/remove_session", json={"session_key": "s1"}, headers=_AUTH
            )
            health = await client.get("/health")
            assert health.json()["active_sessions"] == 0
