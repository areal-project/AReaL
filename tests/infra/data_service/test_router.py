from __future__ import annotations

import asyncio

import httpx
import pytest
import pytest_asyncio

from areal.infra.data_service.router.app import create_router_app
from areal.infra.data_service.router.config import RouterConfig

ADMIN_KEY = "test-admin-key"
WORKER_1 = "http://worker-1:8000"
WORKER_2 = "http://worker-2:8000"
WORKER_3 = "http://worker-3:8000"


@pytest.fixture
def config():
    return RouterConfig(
        host="127.0.0.1",
        port=18091,
        admin_api_key=ADMIN_KEY,
        poll_interval=999,
    )


@pytest_asyncio.fixture
async def client(config):
    app = create_router_app(config)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def admin_headers():
    return {"Authorization": f"Bearer {ADMIN_KEY}"}


class TestHealth:
    @pytest.mark.asyncio
    async def test_health_returns_200_with_worker_count(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["status"] == "ok"
        assert payload["workers"] == 0

    @pytest.mark.asyncio
    async def test_health_shows_healthy_count(self, client):
        await client.post(
            "/register", json={"worker_addr": WORKER_1}, headers=admin_headers()
        )
        await client.post(
            "/register", json={"worker_addr": WORKER_2}, headers=admin_headers()
        )

        resp = await client.get("/health")
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["workers"] == 2
        assert payload["healthy"] == 2


class TestWorkerRegistration:
    @pytest.mark.asyncio
    async def test_register_worker_success(self, client):
        resp = await client.post(
            "/register", json={"worker_addr": WORKER_1}, headers=admin_headers()
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

        health = await client.get("/health")
        assert health.json()["workers"] == 1

    @pytest.mark.asyncio
    async def test_register_duplicate_noop(self, client):
        await client.post(
            "/register", json={"worker_addr": WORKER_1}, headers=admin_headers()
        )
        resp = await client.post(
            "/register", json={"worker_addr": WORKER_1}, headers=admin_headers()
        )
        assert resp.status_code == 200

        health = await client.get("/health")
        assert health.json()["workers"] == 1

    @pytest.mark.asyncio
    async def test_register_no_auth_401(self, client):
        resp = await client.post("/register", json={"worker_addr": WORKER_1})
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_register_wrong_key_403(self, client):
        resp = await client.post(
            "/register",
            json={"worker_addr": WORKER_1},
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_unregister_worker_removes(self, client):
        await client.post(
            "/register", json={"worker_addr": WORKER_1}, headers=admin_headers()
        )
        resp = await client.post(
            "/unregister", json={"worker_addr": WORKER_1}, headers=admin_headers()
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

        health = await client.get("/health")
        assert health.json()["workers"] == 0


class TestRouting:
    @pytest.mark.asyncio
    async def test_route_round_robin_cycles(self, client):
        await client.post(
            "/register", json={"worker_addr": WORKER_1}, headers=admin_headers()
        )
        await client.post(
            "/register", json={"worker_addr": WORKER_2}, headers=admin_headers()
        )

        picks = []
        for _ in range(4):
            resp = await client.post("/route", headers=admin_headers())
            assert resp.status_code == 200
            picks.append(resp.json()["worker_addr"])

        assert picks == [WORKER_1, WORKER_2, WORKER_1, WORKER_2]

    @pytest.mark.asyncio
    async def test_route_no_workers_503(self, client):
        resp = await client.post("/route", headers=admin_headers())
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_route_skips_unhealthy(self, client):
        await client.post(
            "/register",
            json={"worker_addr": WORKER_1},
            headers=admin_headers(),
        )
        await client.post(
            "/register",
            json={"worker_addr": WORKER_2},
            headers=admin_headers(),
        )

        app = client._transport.app
        app.state.worker_healthy[WORKER_2] = False

        for _ in range(3):
            resp = await client.post("/route", headers=admin_headers())
            assert resp.status_code == 200
            assert resp.json()["worker_addr"] == WORKER_1

    @pytest.mark.asyncio
    async def test_route_no_auth_401(self, client):
        resp = await client.post("/route")
        assert resp.status_code == 401


class TestWorkersList:
    @pytest.mark.asyncio
    async def test_workers_returns_all_registered(self, client):
        await client.post(
            "/register", json={"worker_addr": WORKER_1}, headers=admin_headers()
        )
        await client.post(
            "/register", json={"worker_addr": WORKER_2}, headers=admin_headers()
        )
        await client.post(
            "/register", json={"worker_addr": WORKER_3}, headers=admin_headers()
        )

        resp = await client.get("/workers", headers=admin_headers())
        assert resp.status_code == 200
        workers = resp.json()["workers"]
        assert {w["addr"] for w in workers} == {WORKER_1, WORKER_2, WORKER_3}

    @pytest.mark.asyncio
    async def test_workers_shows_health_status(self, client):
        await client.post(
            "/register", json={"worker_addr": WORKER_1}, headers=admin_headers()
        )

        resp = await client.get("/workers", headers=admin_headers())
        assert resp.status_code == 200
        workers = resp.json()["workers"]
        assert workers == [{"addr": WORKER_1, "healthy": True}]

    @pytest.mark.asyncio
    async def test_workers_no_auth_401(self, client):
        resp = await client.get("/workers")
        assert resp.status_code == 401


class TestConcurrentRouting:
    @pytest.mark.asyncio
    async def test_concurrent_routes_distribute_evenly(self, client):
        await client.post(
            "/register", json={"worker_addr": WORKER_1}, headers=admin_headers()
        )
        await client.post(
            "/register", json={"worker_addr": WORKER_2}, headers=admin_headers()
        )

        async def route_once() -> str:
            resp = await client.post("/route", headers=admin_headers())
            assert resp.status_code == 200
            return resp.json()["worker_addr"]

        routes = await asyncio.gather(*(route_once() for _ in range(10)))
        assert routes.count(WORKER_1) == 5
        assert routes.count(WORKER_2) == 5
