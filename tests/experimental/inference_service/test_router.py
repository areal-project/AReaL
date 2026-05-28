"""Unit tests for the Router service (Plan 2, Task 3).

Tests worker registry, session registry, routing strategies,
and all router endpoints.
"""

from __future__ import annotations

import httpx
import pytest
import pytest_asyncio

from areal.experimental.inference_service.router.app import create_app
from areal.experimental.inference_service.router.config import RouterConfig
from areal.experimental.inference_service.router.state import (
    SessionRegistry,
    WorkerInfo,
    WorkerRegistry,
)
from areal.experimental.inference_service.router.strategies import (
    RoundRobinStrategy,
    get_strategy,
)

# =============================================================================
# Constants
# =============================================================================

ADMIN_KEY = "test-admin-key"
WORKER_1 = "http://worker-1:18082"
WORKER_2 = "http://worker-2:18082"
WORKER_3 = "http://worker-3:18082"


# =============================================================================
# WorkerRegistry unit tests
# =============================================================================


class TestWorkerRegistry:
    @pytest.mark.asyncio
    async def test_register_worker(self):
        reg = WorkerRegistry()
        worker_id = await reg.register(WORKER_1)
        assert isinstance(worker_id, str)
        assert len(worker_id) > 0  # UUID string
        workers = await reg.get_all_workers()
        assert len(workers) == 1
        assert workers[0].worker_addr == WORKER_1
        assert workers[0].worker_id == worker_id
        assert workers[0].is_healthy is True

    @pytest.mark.asyncio
    async def test_register_duplicate_noop(self):
        reg = WorkerRegistry()
        worker_id_1 = await reg.register(WORKER_1)
        worker_id_2 = await reg.register(WORKER_1)
        assert worker_id_1 == worker_id_2
        workers = await reg.get_all_workers()
        assert len(workers) == 1

    @pytest.mark.asyncio
    async def test_deregister_worker(self):
        reg = WorkerRegistry()
        await reg.register(WORKER_1)
        await reg.deregister(WORKER_1)
        workers = await reg.get_all_workers()
        assert len(workers) == 0

    @pytest.mark.asyncio
    async def test_deregister_unknown_noop(self):
        reg = WorkerRegistry()
        await reg.deregister("http://unknown:9999")
        assert len(await reg.get_all_workers()) == 0

    @pytest.mark.asyncio
    async def test_health_update(self):
        reg = WorkerRegistry()
        await reg.register(WORKER_1)
        await reg.update_health(WORKER_1, False)
        workers = await reg.get_all_workers()
        assert workers[0].is_healthy is False

    @pytest.mark.asyncio
    async def test_get_healthy_workers(self):
        reg = WorkerRegistry()
        await reg.register(WORKER_1)
        await reg.register(WORKER_2)
        await reg.update_health(WORKER_1, False)
        healthy = await reg.get_healthy_workers()
        assert len(healthy) == 1
        assert healthy[0].worker_addr == WORKER_2

    @pytest.mark.asyncio
    async def test_get_all_workers(self):
        reg = WorkerRegistry()
        await reg.register(WORKER_1)
        await reg.register(WORKER_2)
        await reg.update_health(WORKER_1, False)
        all_w = await reg.get_all_workers()
        assert len(all_w) == 2

    @pytest.mark.asyncio
    async def test_list_worker_addrs(self):
        reg = WorkerRegistry()
        await reg.register(WORKER_1)
        await reg.register(WORKER_2)
        addrs = await reg.list_worker_addrs()
        assert set(addrs) == {WORKER_1, WORKER_2}

    @pytest.mark.asyncio
    async def test_deregister_by_id(self):
        reg = WorkerRegistry()
        worker_id = await reg.register(WORKER_1)
        addr = await reg.deregister_by_id(worker_id)
        assert addr == WORKER_1
        workers = await reg.get_all_workers()
        assert len(workers) == 0

    @pytest.mark.asyncio
    async def test_deregister_by_id_unknown_returns_none(self):
        reg = WorkerRegistry()
        result = await reg.deregister_by_id("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_id(self):
        reg = WorkerRegistry()
        worker_id = await reg.register(WORKER_1)
        info = await reg.get_by_id(worker_id)
        assert info is not None
        assert info.worker_addr == WORKER_1
        assert info.worker_id == worker_id

    @pytest.mark.asyncio
    async def test_get_by_id_unknown_returns_none(self):
        reg = WorkerRegistry()
        result = await reg.get_by_id("nonexistent-id")
        assert result is None


# =============================================================================
# SessionRegistry unit tests
# =============================================================================


class TestSessionRegistry:
    @pytest.mark.asyncio
    async def test_register_session(self):
        reg = SessionRegistry()
        await reg.register_session("key-1", "id-1", WORKER_1)
        assert await reg.lookup_by_key("key-1") == WORKER_1
        assert await reg.lookup_by_id("id-1") == WORKER_1

    @pytest.mark.asyncio
    async def test_lookup_by_key(self):
        reg = SessionRegistry()
        await reg.register_session("key-1", "id-1", WORKER_1)
        assert await reg.lookup_by_key("key-1") == WORKER_1

    @pytest.mark.asyncio
    async def test_lookup_by_id(self):
        reg = SessionRegistry()
        await reg.register_session("key-1", "id-1", WORKER_1)
        assert await reg.lookup_by_id("id-1") == WORKER_1

    @pytest.mark.asyncio
    async def test_lookup_unknown_key(self):
        reg = SessionRegistry()
        assert await reg.lookup_by_key("nonexistent") is None

    @pytest.mark.asyncio
    async def test_lookup_unknown_id(self):
        reg = SessionRegistry()
        assert await reg.lookup_by_id("nonexistent") is None

    @pytest.mark.asyncio
    async def test_revoke_by_worker(self):
        reg = SessionRegistry()
        await reg.register_session("key-1", "id-1", WORKER_1)
        await reg.register_session("key-2", "id-2", WORKER_1)
        await reg.register_session("key-3", "id-3", WORKER_2)  # different worker
        count = await reg.revoke_by_worker(WORKER_1)
        assert count == 2
        assert await reg.lookup_by_key("key-1") is None
        assert await reg.lookup_by_key("key-2") is None
        assert await reg.lookup_by_id("id-1") is None
        assert await reg.lookup_by_id("id-2") is None
        # Worker 2 sessions untouched
        assert await reg.lookup_by_key("key-3") == WORKER_2

    @pytest.mark.asyncio
    async def test_count(self):
        reg = SessionRegistry()
        assert await reg.count() == 0
        await reg.register_session("key-1", "id-1", WORKER_1)
        assert await reg.count() == 1
        await reg.register_session("key-2", "id-2", WORKER_2)
        assert await reg.count() == 2

    @pytest.mark.asyncio
    async def test_upsert_semantics(self):
        """Re-registering a session key updates the worker."""
        reg = SessionRegistry()
        await reg.register_session("key-1", "id-1", WORKER_1)
        await reg.register_session("key-1", "id-1", WORKER_2)
        assert await reg.lookup_by_key("key-1") == WORKER_2


# =============================================================================
# Routing strategies unit tests
# =============================================================================


class TestRoutingStrategies:
    def test_round_robin_cycling(self):
        s = RoundRobinStrategy()
        w1 = WorkerInfo(worker_id="w1", worker_addr=WORKER_1)
        w2 = WorkerInfo(worker_id="w2", worker_addr=WORKER_2)
        w3 = WorkerInfo(worker_id="w3", worker_addr=WORKER_3)
        workers = [w1, w2, w3]
        picks = [s.pick(workers).worker_addr for _ in range(6)]  # type: ignore[union-attr]
        assert picks == [
            WORKER_1,
            WORKER_2,
            WORKER_3,
            WORKER_1,
            WORKER_2,
            WORKER_3,
        ]

    def test_round_robin_empty(self):
        s = RoundRobinStrategy()
        assert s.pick([]) is None

    def test_get_strategy_round_robin(self):
        s = get_strategy("round_robin")
        assert isinstance(s, RoundRobinStrategy)

    def test_get_strategy_least_busy_not_implemented(self):
        with pytest.raises(NotImplementedError, match="least_busy"):
            get_strategy("least_busy")

    def test_get_strategy_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown routing strategy"):
            get_strategy("random")


# =============================================================================
# Router endpoint tests
# =============================================================================


@pytest.fixture
def config():
    return RouterConfig(
        host="127.0.0.1",
        port=18081,
        admin_api_key=ADMIN_KEY,
        poll_interval=999,  # effectively disable polling in tests
        routing_strategy="round_robin",
    )


@pytest_asyncio.fixture
async def client(config):
    """Create router app and yield an httpx async client.
    Bypasses lifespan (no background health poller) by setting state directly.
    """
    app = create_app(config)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def admin_headers():
    return {"Authorization": f"Bearer {ADMIN_KEY}"}


class TestRouterEndpoints:
    # ----- Health -----

    @pytest.mark.asyncio
    async def test_health_200(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["workers"] == 0
        assert data["sessions"] == 0
        assert data["strategy"] == "round_robin"

    @pytest.mark.asyncio
    async def test_hitl_route_binds_first_use(self, client):
        await client.post(
            "/register",
            json={"worker_addr": WORKER_1},
            headers=admin_headers(),
        )

        resp = await client.post(
            "/route",
            json={
                "api_key": ADMIN_KEY,
                "path": "/chat/completions",
            },
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["worker_addr"] == WORKER_1

        pinned = await client.post(
            "/route",
            json={
                "api_key": ADMIN_KEY,
                "path": "/chat/completions",
            },
            headers=admin_headers(),
        )
        assert pinned.status_code == 200
        assert pinned.json()["worker_addr"] == WORKER_1

    # ----- Worker registration -----

    @pytest.mark.asyncio
    async def test_register_worker_admin_key(self, client):
        resp = await client.post(
            "/register",
            json={"worker_addr": WORKER_1},
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert "worker_id" in resp.json()
        assert isinstance(resp.json()["worker_id"], str)
        assert len(resp.json()["worker_id"]) > 0

        # Verify via /health
        health = (await client.get("/health")).json()
        assert health["workers"] == 1

    @pytest.mark.asyncio
    async def test_register_worker_no_auth_401(self, client):
        resp = await client.post("/register", json={"worker_addr": WORKER_1})
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_register_worker_wrong_key_403(self, client):
        resp = await client.post(
            "/register",
            json={"worker_addr": WORKER_1},
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert resp.status_code == 403

    # ----- Worker deletion (with cascade) -----

    @pytest.mark.asyncio
    async def test_delete_worker_cascades(self, client):
        # Register worker + session pinned to it
        await client.post(
            "/register",
            json={"worker_addr": WORKER_1},
            headers=admin_headers(),
        )
        await client.post(
            "/register_session",
            json={
                "sessions": [
                    {"session_api_key": "sess-key-1", "session_id": "task-0-0"}
                ],
                "worker_addr": WORKER_1,
                "group_id": "grp-test-1",
            },
            headers=admin_headers(),
        )

        # Delete the worker
        resp = await client.post(
            "/unregister",
            json={"worker_addr": WORKER_1},
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["sessions_revoked"] == 1

        # Session key should no longer route
        resp = await client.post(
            "/route",
            json={"api_key": "sess-key-1", "path": "/chat/completions"},
            headers=admin_headers(),
        )
        assert resp.status_code == 404

    # ----- /route — admin key -----

    @pytest.mark.asyncio
    async def test_route_admin_key_sticky_hitl(self, client):
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

        resp1 = await client.post(
            "/route",
            json={"api_key": ADMIN_KEY, "path": "/generate"},
            headers=admin_headers(),
        )
        assert resp1.status_code == 200
        addr1 = resp1.json()["worker_addr"]

        resp2 = await client.post(
            "/route",
            json={"api_key": ADMIN_KEY, "path": "/generate"},
            headers=admin_headers(),
        )
        assert resp2.status_code == 200
        addr2 = resp2.json()["worker_addr"]

        assert addr1 == addr2

    # ----- /route — session key -----

    @pytest.mark.asyncio
    async def test_route_session_key_pinned(self, client):
        # Register worker + session
        await client.post(
            "/register",
            json={"worker_addr": WORKER_1},
            headers=admin_headers(),
        )
        await client.post(
            "/register_session",
            json={
                "sessions": [
                    {"session_api_key": "sess-key-1", "session_id": "task-0-0"}
                ],
                "worker_addr": WORKER_1,
                "group_id": "grp-test-2",
            },
            headers=admin_headers(),
        )

        # Route with session key → pinned to WORKER_1
        resp = await client.post(
            "/route",
            json={"api_key": "sess-key-1", "path": "/chat/completions"},
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["worker_addr"] == WORKER_1

    # ----- /route — unknown key -----

    @pytest.mark.asyncio
    async def test_route_unknown_key_404(self, client):
        resp = await client.post(
            "/route",
            json={"api_key": "unknown-key", "path": "/generate"},
            headers=admin_headers(),
        )
        assert resp.status_code == 404
        assert "Unknown API key" in resp.json()["detail"]

    # ----- /route — no healthy workers -----

    @pytest.mark.asyncio
    async def test_route_no_healthy_workers_503(self, client):
        """Admin key but no workers registered → 503."""
        resp = await client.post(
            "/route",
            json={"api_key": ADMIN_KEY, "path": "/generate"},
            headers=admin_headers(),
        )
        assert resp.status_code == 503
        assert "No registered workers" in resp.json()["detail"]

    # ----- /route — pinned worker unhealthy -----

    @pytest.mark.asyncio
    async def test_route_pinned_worker_unhealthy_still_routes(self, client):
        # Register worker + session
        await client.post(
            "/register",
            json={"worker_addr": WORKER_1},
            headers=admin_headers(),
        )
        await client.post(
            "/register_session",
            json={
                "sessions": [
                    {"session_api_key": "sess-key-1", "session_id": "task-0-0"}
                ],
                "worker_addr": WORKER_1,
                "group_id": "grp-test-4",
            },
            headers=admin_headers(),
        )

        # Mark worker unhealthy
        wr: WorkerRegistry = client._transport.app.state.worker_registry  # type: ignore[attr-defined]
        await wr.update_health(WORKER_1, False)

        # Routing still returns the pinned worker even if unhealthy
        resp = await client.post(
            "/route",
            json={"api_key": "sess-key-1", "path": "/chat/completions"},
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["worker_addr"] == WORKER_1

    # ----- /route — session_id lookup -----

    @pytest.mark.asyncio
    async def test_route_by_session_id(self, client):
        await client.post(
            "/register_session",
            json={
                "sessions": [
                    {"session_api_key": "sess-key-1", "session_id": "task-0-0"}
                ],
                "worker_addr": WORKER_1,
                "group_id": "grp-test-3",
            },
            headers=admin_headers(),
        )
        resp = await client.post(
            "/route",
            json={"session_id": "task-0-0"},
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["worker_addr"] == WORKER_1

    @pytest.mark.asyncio
    async def test_route_by_session_id_unknown_404(self, client):
        resp = await client.post(
            "/route",
            json={"session_id": "nonexistent"},
            headers=admin_headers(),
        )
        assert resp.status_code == 404

    # ----- /route — missing both api_key and session_id -----

    @pytest.mark.asyncio
    async def test_route_missing_both_keys_422(self, client):
        resp = await client.post(
            "/route",
            json={},
            headers=admin_headers(),
        )
        assert resp.status_code == 422

    # ----- /route — no auth -----

    @pytest.mark.asyncio
    async def test_route_no_auth_401(self, client):
        """Route without auth → 401."""
        resp = await client.post(
            "/route",
            json={"api_key": ADMIN_KEY, "path": "/generate"},
        )
        assert resp.status_code == 401

    # ----- /register_session — no auth -----

    @pytest.mark.asyncio
    async def test_register_session_no_auth_401(self, client):
        """Register session without auth → 401."""
        resp = await client.post(
            "/register_session",
            json={
                "sessions": [
                    {"session_api_key": "sess-key-1", "session_id": "task-0-0"}
                ],
                "worker_addr": WORKER_1,
                "group_id": "grp-test-5",
            },
        )
        assert resp.status_code == 401

    # ----- /register_session -----

    @pytest.mark.asyncio
    async def test_register_session(self, client):
        """Register a session, then verify pinned routing works."""
        await client.post(
            "/register",
            json={"worker_addr": WORKER_1},
            headers=admin_headers(),
        )

        resp = await client.post(
            "/register_session",
            json={
                "sessions": [
                    {"session_api_key": "sess-key-1", "session_id": "task-0-0"}
                ],
                "worker_addr": WORKER_1,
                "group_id": "grp-test-6",
            },
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

        # Verify: /route with session key returns pinned worker
        route_resp = await client.post(
            "/route",
            json={"api_key": "sess-key-1", "path": "/chat/completions"},
            headers=admin_headers(),
        )
        assert route_resp.status_code == 200
        assert route_resp.json()["worker_addr"] == WORKER_1

    # ----- /workers -----

    @pytest.mark.asyncio
    async def test_workers_list_admin_key(self, client):
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

        resp = await client.get("/workers", headers=admin_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["workers"]) == 2
        addrs = {w["addr"] for w in data["workers"]}
        assert addrs == {WORKER_1, WORKER_2}
        assert all(w["healthy"] is True for w in data["workers"])
        assert all("worker_id" in w for w in data["workers"])
        assert all(isinstance(w["worker_id"], str) for w in data["workers"])

    @pytest.mark.asyncio
    async def test_workers_list_no_auth_401(self, client):
        resp = await client.get("/workers")
        assert resp.status_code == 401

    # ----- /unregister by worker_id -----

    @pytest.mark.asyncio
    async def test_delete_worker_by_id(self, client):
        """Delete a worker by worker_id instead of worker_addr."""
        # Register
        reg_resp = await client.post(
            "/register",
            json={"worker_addr": WORKER_1},
            headers=admin_headers(),
        )
        worker_id = reg_resp.json()["worker_id"]

        # Delete by worker_id
        resp = await client.post(
            "/unregister",
            json={"worker_id": worker_id},
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

        # Verify worker is gone
        health = (await client.get("/health")).json()
        assert health["workers"] == 0

    @pytest.mark.asyncio
    async def test_delete_worker_by_id_not_found_404(self, client):
        """Delete by unknown worker_id → 404."""
        resp = await client.post(
            "/unregister",
            json={"worker_id": "nonexistent-id"},
            headers=admin_headers(),
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_worker_missing_both_422(self, client):
        """Delete without worker_id or worker_addr → 422."""
        resp = await client.post(
            "/unregister",
            json={},
            headers=admin_headers(),
        )
        assert resp.status_code == 422

    # ----- /resolve_worker/{worker_id} -----

    @pytest.mark.asyncio
    async def test_resolve_worker_200(self, client):
        """Resolve a registered worker by ID → 200 with worker_id and worker_addr."""
        reg_resp = await client.post(
            "/register",
            json={"worker_addr": WORKER_1},
            headers=admin_headers(),
        )
        worker_id = reg_resp.json()["worker_id"]

        resp = await client.get(
            f"/resolve_worker/{worker_id}",
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["worker_id"] == worker_id
        assert data["worker_addr"] == WORKER_1

    @pytest.mark.asyncio
    async def test_resolve_worker_not_found_404(self, client):
        """Resolve an unknown worker_id → 404."""
        resp = await client.get(
            "/resolve_worker/nonexistent-id",
            headers=admin_headers(),
        )
        assert resp.status_code == 404
