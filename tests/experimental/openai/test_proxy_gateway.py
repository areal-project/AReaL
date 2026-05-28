"""Unit tests for the proxy gateway (FastAPI gateway)."""

from __future__ import annotations

import asyncio
import json

import pytest

from areal.experimental.openai.proxy.proxy_gateway import (
    CompletedSessionInfo,
    _extract_bearer_token,
    _forwarding_headers,
    _ReadyWorkerEntry,
    _SessionRoute,
    create_proxy_gateway_app,
)
from areal.experimental.openai.proxy.server import DEFAULT_ADMIN_API_KEY

# ---------------------------------------------------------------------------
# We use httpx.AsyncClient with ASGITransport to test the FastAPI app
# in-process (no real HTTP server needed).
# ---------------------------------------------------------------------------

httpx = pytest.importorskip("httpx")


def _make_client(
    proxy_addrs: list[str] | None = None, admin_key: str = DEFAULT_ADMIN_API_KEY
):
    """Create an httpx.AsyncClient backed by the proxy gateway ASGI app."""
    addrs = proxy_addrs or ["http://worker-0:8000", "http://worker-1:8000"]
    app = create_proxy_gateway_app(proxy_addrs=addrs, admin_api_key=admin_key)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_ready_worker_entry(self):
        loop = asyncio.new_event_loop()
        future = loop.create_future()
        entry = _ReadyWorkerEntry(worker_addr="http://w:8000", future=future)
        assert entry.worker_addr == "http://w:8000"
        assert entry.future is future
        loop.close()

    def test_completed_session_info(self):
        info = CompletedSessionInfo(
            session_api_key="sk-123",
            session_id="sess-abc",
            worker_addr="http://w:8000",
        )
        assert info.session_api_key == "sk-123"
        assert info.session_id == "sess-abc"
        assert info.worker_addr == "http://w:8000"

    def test_session_route_defaults(self):
        route = _SessionRoute(worker_addr="http://w:8000", session_id="s1")
        assert route.pending_future is None

    def test_session_route_with_future(self):
        loop = asyncio.new_event_loop()
        future = loop.create_future()
        route = _SessionRoute(
            worker_addr="http://w:8000",
            session_id="s1",
            pending_future=future,
        )
        assert route.pending_future is future
        loop.close()


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_extract_bearer_token_valid(self):
        class FakeRequest:
            headers = {"authorization": "Bearer sk-test-token"}

        assert _extract_bearer_token(FakeRequest()) == "sk-test-token"

    def test_extract_bearer_token_missing(self):
        class FakeRequest:
            headers = {}

        assert _extract_bearer_token(FakeRequest()) is None

    def test_extract_bearer_token_not_bearer(self):
        class FakeRequest:
            headers = {"authorization": "Basic abc123"}

        assert _extract_bearer_token(FakeRequest()) is None

    def test_forwarding_headers_filters(self):
        raw = {
            "authorization": "Bearer sk-123",
            "content-type": "application/json",
            "x-custom": "should-be-dropped",
            "host": "localhost",
        }
        result = _forwarding_headers(raw)
        assert "authorization" in result
        assert "content-type" in result
        assert "x-custom" not in result
        assert "host" not in result

    def test_forwarding_headers_empty(self):
        assert _forwarding_headers({}) == {}


# ---------------------------------------------------------------------------
# App endpoint tests (using httpx ASGI transport)
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_ok(self):
        async with _make_client(["http://w1:8000", "http://w2:8000"]) as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert data["workers"] == 2


class TestStartSessionAuth:
    @pytest.mark.asyncio
    async def test_start_session_rejects_bad_token(self):
        async with _make_client() as client:
            resp = await client.post(
                "/rl/start_session",
                headers={"Authorization": "Bearer wrong-key"},
                json={},
            )
            assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_start_session_rejects_missing_auth(self):
        async with _make_client() as client:
            resp = await client.post("/rl/start_session", json={})
            assert resp.status_code == 401


class TestSessionForwardAuth:
    @pytest.mark.asyncio
    async def test_chat_completions_rejects_unknown_session(self):
        async with _make_client() as client:
            resp = await client.post(
                "/chat/completions",
                headers={"Authorization": "Bearer unknown-session-key"},
                json={"model": "test", "messages": []},
            )
            assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_set_reward_rejects_unknown_session(self):
        async with _make_client() as client:
            resp = await client.post(
                "/rl/set_reward",
                headers={"Authorization": "Bearer unknown-session-key"},
                json={},
            )
            assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_end_session_rejects_unknown_session(self):
        async with _make_client() as client:
            resp = await client.post(
                "/rl/end_session",
                headers={"Authorization": "Bearer unknown-session-key"},
                json={},
            )
            assert resp.status_code == 401


class TestWaitForSessionAuth:
    @pytest.mark.asyncio
    async def test_wait_for_session_rejects_bad_token(self):
        async with _make_client() as client:
            resp = await client.post(
                "/internal/wait_for_session",
                headers={"Authorization": "Bearer wrong-key"},
                json={"worker_addr": "http://w:8000"},
            )
            assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Mock aiohttp infrastructure for backend-forwarding tests
# ---------------------------------------------------------------------------


class _MockAiohttpResponse:
    """Simulates an aiohttp response for the gateway's ``_forward`` helper."""

    def __init__(self, status: int, body):
        self.status = status
        self._body = json.dumps(body).encode() if isinstance(body, dict) else body
        self.content_type = "application/json"

    async def read(self):
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        pass


class _MockAiohttpSession:
    """Controllable aiohttp session returning pre-enqueued responses.

    Responses are matched by the first enqueued entry whose *url_contains*
    substring appears in the request URL.  Matched entries are consumed (FIFO
    per pattern).
    """

    def __init__(self):
        self._queue: list[tuple[str, _MockAiohttpResponse]] = []
        self.calls: list[tuple[str, dict]] = []

    def enqueue(self, url_contains: str, status: int, body) -> None:
        self._queue.append((url_contains, _MockAiohttpResponse(status, body)))

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        for i, (pattern, resp) in enumerate(self._queue):
            if pattern in url:
                self._queue.pop(i)
                return resp
        return _MockAiohttpResponse(500, {"error": f"no mock for {url}"})

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        pass


def _make_mocked_client(
    proxy_addrs: list[str] | None = None,
    admin_key: str = DEFAULT_ADMIN_API_KEY,
    refresh_timeout: float = 120.0,
    key_pool_size: int = 4096,
):
    """Create gateway client + mock aiohttp session for forwarding tests.

    The httpx ASGI transport does not trigger the ASGI lifespan, so we
    inject the mock session directly into ``app.state.http_session``.
    """
    addrs = proxy_addrs or ["http://worker-0:8000"]
    mock_session = _MockAiohttpSession()

    app = create_proxy_gateway_app(
        proxy_addrs=addrs,
        admin_api_key=admin_key,
        refresh_timeout=refresh_timeout,
        key_pool_size=key_pool_size,
    )
    # Inject mock session directly (lifespan won't run under ASGI transport).
    app.state.http_session = mock_session

    transport = httpx.ASGITransport(app=app)
    client = httpx.AsyncClient(transport=transport, base_url="http://testserver")

    class _ClientWrapper:
        """Async context manager exposing both client and mock session."""

        def __init__(self):
            self.mock = mock_session

        async def __aenter__(self):
            await client.__aenter__()
            return self

        async def __aexit__(self, *a):
            await client.__aexit__(*a)

        # Delegate HTTP methods to the underlying client.
        def get(self, *a, **kw):
            return client.get(*a, **kw)

        def post(self, *a, **kw):
            return client.post(*a, **kw)

    return _ClientWrapper()


def _admin_headers(key: str = DEFAULT_ADMIN_API_KEY) -> dict:
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _session_headers(api_key: str) -> dict:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


# ---------------------------------------------------------------------------
# Refresh / key-reuse tests
# ---------------------------------------------------------------------------


class TestRefresh:
    """Tests for the session refresh (API key reuse) feature."""

    @pytest.mark.asyncio
    async def test_refresh_known_active_key(self):
        """Full refresh: end old session, wait for ready worker, start new."""
        async with _make_mocked_client() as ctx:
            # Enqueue backend responses:
            # 1) first start_session → success
            ctx.mock.enqueue(
                "start_session",
                200,
                {"api_key": "reuse-key", "session_id": "s1"},
            )
            # 2) end_session during refresh → success
            ctx.mock.enqueue(
                "end_session",
                200,
                {"message": "success", "interaction_count": 3},
            )
            # 3) second start_session (after refresh) → success
            ctx.mock.enqueue(
                "start_session",
                200,
                {"api_key": "reuse-key", "session_id": "s2"},
            )

            # Step 1: populate ready queue and start initial session.
            bg1 = asyncio.create_task(
                ctx.post(
                    "/internal/wait_for_session",
                    headers=_admin_headers(),
                    json={"worker_addr": "http://worker-0:8000"},
                )
            )
            await asyncio.sleep(0.01)

            resp1 = await ctx.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "t"},
            )
            assert resp1.status_code == 200
            assert resp1.json()["api_key"] == "reuse-key"
            assert resp1.json()["session_id"] == "s1"

            # Step 2: populate queue again for the post-refresh session.
            bg2 = asyncio.create_task(
                ctx.post(
                    "/internal/wait_for_session",
                    headers=_admin_headers(),
                    json={"worker_addr": "http://worker-0:8000"},
                )
            )
            await asyncio.sleep(0.01)

            # Step 3: refresh — same api_key triggers auto-end + new session.
            resp2 = await ctx.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "t", "api_key": "reuse-key"},
            )
            assert resp2.status_code == 200
            assert resp2.json()["api_key"] == "reuse-key"
            assert resp2.json()["session_id"] == "s2"

            # bg1 should have completed (future resolved during refresh).
            bg1_resp = await asyncio.wait_for(bg1, timeout=2.0)
            assert bg1_resp.status_code == 200

            # Clean up bg2 (still waiting on its future).
            bg2.cancel()
            try:
                await bg2
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_refresh_timeout_returns_429(self):
        """Refresh with no ready worker within timeout returns 429."""
        async with _make_mocked_client(refresh_timeout=0.1) as ctx:
            # Enqueue: first start_session + end_session, but NO second start.
            ctx.mock.enqueue(
                "start_session",
                200,
                {"api_key": "k1", "session_id": "s1"},
            )
            ctx.mock.enqueue(
                "end_session",
                200,
                {"message": "success", "interaction_count": 0},
            )

            # Start initial session via ready queue.
            bg1 = asyncio.create_task(
                ctx.post(
                    "/internal/wait_for_session",
                    headers=_admin_headers(),
                    json={"worker_addr": "http://worker-0:8000"},
                )
            )
            await asyncio.sleep(0.01)

            resp1 = await ctx.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "t"},
            )
            assert resp1.status_code == 200

            # Refresh — no ready worker, should timeout (0.1s).
            resp2 = await ctx.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "t", "api_key": "k1"},
            )
            assert resp2.status_code == 429
            assert "timed out" in resp2.json().get("detail", "").lower()

            bg1.cancel()
            try:
                await bg1
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_reuse_known_inactive_key_round_robin(self):
        """Known key with no active route goes through normal flow with key."""
        async with _make_mocked_client() as ctx:
            # First session — round-robin (no ready queue entries).
            ctx.mock.enqueue(
                "start_session",
                200,
                {"api_key": "k1", "session_id": "s1"},
            )
            resp1 = await ctx.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "t"},
            )
            assert resp1.status_code == 200
            assert resp1.json()["api_key"] == "k1"

            # End session (removes route, key stays in pool).
            ctx.mock.enqueue(
                "end_session",
                200,
                {"message": "success", "interaction_count": 1},
            )
            resp_end = await ctx.post(
                "/rl/end_session",
                headers=_session_headers("k1"),
                json={},
            )
            assert resp_end.status_code == 200

            # Reuse same key — known but no active route → normal flow.
            ctx.mock.enqueue(
                "start_session",
                200,
                {"api_key": "k1", "session_id": "s2"},
            )
            resp2 = await ctx.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "t", "api_key": "k1"},
            )
            assert resp2.status_code == 200
            assert resp2.json()["api_key"] == "k1"
            assert resp2.json()["session_id"] == "s2"

    @pytest.mark.asyncio
    async def test_unknown_key_stripped(self):
        """Unknown api_key is stripped — worker generates a fresh key."""
        async with _make_mocked_client() as ctx:
            ctx.mock.enqueue(
                "start_session",
                200,
                {"api_key": "worker-generated", "session_id": "s1"},
            )
            resp = await ctx.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "t", "api_key": "user-invented-key"},
            )
            assert resp.status_code == 200
            assert resp.json()["api_key"] == "worker-generated"

            # Verify the forwarded body had api_key stripped.
            _, call_kwargs = ctx.mock.calls[0]
            forwarded_body = json.loads(call_kwargs["data"])
            assert "api_key" not in forwarded_body

    @pytest.mark.asyncio
    async def test_none_api_key_normal_flow(self):
        """api_key=None goes through normal flow."""
        async with _make_mocked_client() as ctx:
            ctx.mock.enqueue(
                "start_session",
                200,
                {"api_key": "new-key", "session_id": "s1"},
            )
            resp = await ctx.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "t"},
            )
            assert resp.status_code == 200
            assert resp.json()["api_key"] == "new-key"

    @pytest.mark.asyncio
    async def test_key_pool_eviction(self):
        """Key pool evicts oldest key when exceeding key_pool_size."""
        async with _make_mocked_client(key_pool_size=2) as ctx:
            # Create 3 sessions — pool can hold only 2.
            for i in range(3):
                ctx.mock.enqueue(
                    "start_session",
                    200,
                    {"api_key": f"k{i}", "session_id": f"s{i}"},
                )
                ctx.mock.enqueue(
                    "end_session",
                    200,
                    {"message": "success", "interaction_count": 0},
                )

                resp = await ctx.post(
                    "/rl/start_session",
                    headers=_admin_headers(),
                    json={"task_id": "t"},
                )
                assert resp.status_code == 200

                # End session so the key has no active route.
                resp_end = await ctx.post(
                    "/rl/end_session",
                    headers=_session_headers(f"k{i}"),
                    json={},
                )
                assert resp_end.status_code == 200

            # k0 should have been evicted.  Sending it now should be treated
            # as an unknown key (stripped).
            ctx.mock.enqueue(
                "start_session",
                200,
                {"api_key": "fresh", "session_id": "s-new"},
            )
            resp = await ctx.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "t", "api_key": "k0"},
            )
            assert resp.status_code == 200
            assert resp.json()["api_key"] == "fresh"  # worker generated, not k0

            # k2 (most recent) should still be known.
            ctx.mock.enqueue(
                "start_session",
                200,
                {"api_key": "k2", "session_id": "s-reuse"},
            )
            resp2 = await ctx.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "t", "api_key": "k2"},
            )
            assert resp2.status_code == 200
            assert resp2.json()["api_key"] == "k2"  # reused


# ---------------------------------------------------------------------------
# Additional regression tests
# ---------------------------------------------------------------------------


class TestOrphanFutureCancellation:
    """When the backend rejects a ready worker, the future is cancelled
    and wait_for_session returns 503."""

    @pytest.mark.asyncio
    async def test_ready_worker_reject_cancels_future(self):
        """Backend returns 500 for start_session → future.cancel() → 503."""
        async with _make_mocked_client() as ctx:
            # Backend rejects start_session with 500.
            ctx.mock.enqueue(
                "start_session",
                500,
                {"error": "internal error"},
            )

            # Register a worker via wait_for_session (runs in background).
            bg = asyncio.create_task(
                ctx.post(
                    "/internal/wait_for_session",
                    headers=_admin_headers(),
                    json={"worker_addr": "http://worker-0:8000"},
                )
            )
            await asyncio.sleep(0.01)

            # start_session picks the ready worker; backend rejects;
            # gateway cancels the future.
            resp = await ctx.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "t"},
            )
            assert resp.status_code == 500

            # wait_for_session sees CancelledError → 503.
            bg_resp = await asyncio.wait_for(bg, timeout=2.0)
            assert bg_resp.status_code == 503
            assert b"cancelled" in bg_resp.content.lower()


class TestConcurrentRefreshRejected:
    """Concurrent refresh for the same key is rejected with 429."""

    @pytest.mark.asyncio
    async def test_concurrent_refresh_same_key_returns_429(self):
        """Second refresh while first is in progress returns 429."""
        async with _make_mocked_client() as ctx:
            # First session via ready queue.
            ctx.mock.enqueue(
                "start_session",
                200,
                {"api_key": "k1", "session_id": "s1"},
            )
            ctx.mock.enqueue(
                "end_session",
                200,
                {"message": "success", "interaction_count": 0},
            )

            bg_wait = asyncio.create_task(
                ctx.post(
                    "/internal/wait_for_session",
                    headers=_admin_headers(),
                    json={"worker_addr": "http://worker-0:8000"},
                )
            )
            await asyncio.sleep(0.01)

            resp1 = await ctx.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "t"},
            )
            assert resp1.status_code == 200
            assert resp1.json()["api_key"] == "k1"

            # First refresh blocks on ready_workers.get() (queue empty).
            refresh1 = asyncio.create_task(
                ctx.post(
                    "/rl/start_session",
                    headers=_admin_headers(),
                    json={"task_id": "t", "api_key": "k1"},
                )
            )
            await asyncio.sleep(0.05)  # let refresh1 enter the refresh path

            # Second refresh with same key.
            resp2 = await ctx.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "t", "api_key": "k1"},
            )
            # The _refreshing guard now rejects concurrent refreshes for the
            # same key before checking `routes`, so this reliably returns 429.
            assert resp2.status_code == 429

            # Clean up.
            refresh1.cancel()
            bg_wait.cancel()
            for t in [refresh1, bg_wait]:
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass


class TestWaitForSessionTimeoutStaleEntry:
    """After wait_for_session times out, the stale ready entry is skipped
    by the next start_session call."""

    @pytest.mark.asyncio
    async def test_timeout_stale_entry_skipped(self):
        """Timed-out worker entry is skipped; fresh worker is used."""
        import areal.experimental.openai.proxy.proxy_gateway as gw_mod

        original_timeout = gw_mod._DEFAULT_WAIT_TIMEOUT
        gw_mod._DEFAULT_WAIT_TIMEOUT = 0.1  # very short timeout
        try:
            async with _make_mocked_client() as ctx:
                # Register worker — will timeout (nobody calls start_session).
                bg_stale = asyncio.create_task(
                    ctx.post(
                        "/internal/wait_for_session",
                        headers=_admin_headers(),
                        json={"worker_addr": "http://stale-worker:8000"},
                    )
                )
                # Let it timeout.
                stale_resp = await asyncio.wait_for(bg_stale, timeout=2.0)
                assert stale_resp.status_code == 408

                # Stale entry remains in queue. Register a fresh worker.
                ctx.mock.enqueue(
                    "start_session",
                    200,
                    {"api_key": "fresh-key", "session_id": "s-fresh"},
                )

                gw_mod._DEFAULT_WAIT_TIMEOUT = 3600.0  # restore for fresh worker
                bg_fresh = asyncio.create_task(
                    ctx.post(
                        "/internal/wait_for_session",
                        headers=_admin_headers(),
                        json={"worker_addr": "http://fresh-worker:8000"},
                    )
                )
                await asyncio.sleep(0.01)

                # Should skip the stale entry and use the fresh one.
                resp = await ctx.post(
                    "/rl/start_session",
                    headers=_admin_headers(),
                    json={"task_id": "t"},
                )
                assert resp.status_code == 200
                assert resp.json()["api_key"] == "fresh-key"

                # The fresh worker's wait_for_session is still pending.
                bg_fresh.cancel()
                try:
                    await bg_fresh
                except asyncio.CancelledError:
                    pass
        finally:
            gw_mod._DEFAULT_WAIT_TIMEOUT = original_timeout
