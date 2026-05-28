"""Unit tests for the inference gateway.

Tests gateway endpoints with mocked Router and data proxy workers.
Uses ``unittest.mock.patch`` to mock ``streaming.py`` functions at module level.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest
import pytest_asyncio

from areal.experimental.inference_service.gateway.app import create_app
from areal.experimental.inference_service.gateway.config import GatewayConfig
from areal.experimental.inference_service.gateway.streaming import (
    RouterKeyRejectedError,
    RouterUnreachableError,
)

# =============================================================================
# Constants & Config
# =============================================================================

ADMIN_KEY = "test-admin-key"
SESSION_KEY = "session-key-abc123"
WORKER_ADDR = "http://worker-1:18082"
WORKER_ADDR_2 = "http://worker-2:18082"
WORKER_ADDR_3 = "http://worker-3:18082"

MODULE = "areal.experimental.inference_service.gateway.app"


@pytest.fixture
def config():
    return GatewayConfig(
        host="127.0.0.1",
        port=18080,
        admin_api_key=ADMIN_KEY,
        router_addr="http://mock-router:8081",
        router_timeout=2.0,
        forward_timeout=30.0,
    )


@pytest_asyncio.fixture
async def client(config):
    """Create gateway app and yield an httpx async client."""
    app = create_app(config)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def admin_headers():
    return {"Authorization": f"Bearer {ADMIN_KEY}"}


def session_headers():
    return {"Authorization": f"Bearer {SESSION_KEY}"}


# =============================================================================
# Health endpoint
# =============================================================================


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_no_auth_required(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    @pytest.mark.asyncio
    async def test_health_returns_router_addr(self, client, config):
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["router_addr"] == config.router_addr


# =============================================================================
# Auth rejection
# =============================================================================


class TestAuthRejection:
    @pytest.mark.asyncio
    async def test_missing_auth_chat_401(self, client):
        """POST /chat/completions without auth → 401."""
        resp = await client.post(
            "/chat/completions",
            json={"model": "sglang", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_session_key_on_admin_endpoint_rejected(self, client):
        """POST /rl/start_session with session key → 403."""
        resp = await client.post(
            "/rl/start_session",
            json={"task_id": "t1"},
            headers=session_headers(),
        )
        assert resp.status_code == 403


# =============================================================================
# Admin endpoints
# =============================================================================


class TestAdminEndpoints:
    @pytest.mark.asyncio
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_admin_chat_completions_non_streaming(
        self, mock_query_router, mock_forward, client
    ):
        """Admin key → /chat/completions (non-streaming) → response forwarded."""
        mock_query_router.return_value = WORKER_ADDR

        # Simulate data proxy response
        mock_resp = httpx.Response(
            200,
            json={"id": "chatcmpl-1", "choices": [{"message": {"content": "Hi"}}]},
        )
        mock_forward.return_value = mock_resp

        resp = await client.post(
            "/chat/completions",
            json={
                "model": "sglang",
                "messages": [{"role": "user", "content": "hello"}],
            },
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "chatcmpl-1"

    @pytest.mark.asyncio
    @patch(f"{MODULE}.forward_sse_stream")
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_admin_chat_completions_streaming(
        self, mock_query_router, mock_forward_sse, client
    ):
        """Admin key → /chat/completions (streaming) → SSE forwarded."""
        mock_query_router.return_value = WORKER_ADDR

        async def _stream():
            yield b'data: {"choices": [{"delta": {"content": "Hi"}}]}\n\n'
            yield b"data: [DONE]\n\n"

        mock_forward_sse.return_value = _stream()

        resp = await client.post(
            "/chat/completions",
            json={
                "model": "sglang",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    @pytest.mark.asyncio
    @patch(f"{MODULE}.register_session_in_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_start_session_creates_and_registers(
        self, mock_query_router, mock_forward, mock_register, client
    ):
        """Admin key → /rl/start_session → forwarded, response intercepted, session registered."""
        mock_query_router.return_value = WORKER_ADDR
        session_resp_data = {
            "group_id": "grp-test-1",
            "sessions": [{"session_id": "task-1-0", "session_api_key": "sess-key-xyz"}],
        }
        mock_forward.return_value = httpx.Response(201, json=session_resp_data)

        resp = await client.post(
            "/rl/start_session",
            json={"task_id": "task-1"},
            headers=admin_headers(),
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["group_id"] == "grp-test-1"
        assert data["sessions"] == [
            {"session_id": "task-1-0", "session_api_key": "sess-key-xyz"}
        ]

        # Verify router registration
        mock_register.assert_called_once()
        reg_args = mock_register.call_args
        assert reg_args.args[0] == "http://mock-router:8081"  # router_addr
        assert reg_args.args[1] == [
            {"session_api_key": "sess-key-xyz", "session_id": "task-1-0"}
        ]  # sessions_list
        assert reg_args.args[2] == WORKER_ADDR  # worker_addr

    @pytest.mark.asyncio
    @patch(f"{MODULE}.register_session_in_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_start_session_router_registration_fails(
        self, mock_query_router, mock_forward, mock_register, client
    ):
        """If router registration fails after session creation → 502."""
        mock_query_router.return_value = WORKER_ADDR
        mock_forward.return_value = httpx.Response(
            201,
            json={
                "group_id": "grp-test-2",
                "sessions": [{"session_id": "t-0", "session_api_key": "k"}],
            },
        )
        mock_register.side_effect = RouterUnreachableError("Router down")

        resp = await client.post(
            "/rl/start_session",
            json={"task_id": "t"},
            headers=admin_headers(),
        )
        assert resp.status_code == 502
        assert "registration failed" in resp.json()["error"]

    @pytest.mark.asyncio
    @patch(f"{MODULE}.revoke_session_in_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    async def test_admin_export_trajectories(
        self, mock_forward, mock_query_router, mock_revoke, client
    ):
        mock_query_router.return_value = WORKER_ADDR
        mock_forward.return_value = httpx.Response(200, json={"interactions": []})

        resp = await client.post(
            "/export_trajectories",
            json={
                "session_ids": ["task-1-0"],
                "group_id": "grp-test",
                "discount": 1.0,
                "style": "sft",
            },
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        mock_query_router.assert_called_once()
        assert mock_query_router.call_args.kwargs["session_id"] == "task-1-0"
        mock_revoke.assert_called_once()

    @pytest.mark.asyncio
    @patch(f"{MODULE}.revoke_session_in_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    async def test_online_export_without_group_id_skips_revoke(
        self, mock_forward, mock_query_router, mock_revoke, client
    ):
        mock_query_router.return_value = WORKER_ADDR
        mock_forward.return_value = httpx.Response(200, json={"interactions": []})

        resp = await client.post(
            "/export_trajectories",
            json={
                "session_ids": ["__hitl__"],
                "trajectory_id": 0,
                "discount": 1.0,
                "style": "individual",
            },
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        mock_revoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_export_trajectories_missing_session_id(self, client):
        """Admin key → /export_trajectories without session_id → 400."""
        resp = await client.post(
            "/export_trajectories",
            json={"discount": 1.0},
            headers=admin_headers(),
        )
        assert resp.status_code == 400
        assert "session_ids" in resp.json()["error"]


# =============================================================================
# Session endpoints
# =============================================================================


class TestSessionEndpoints:
    @pytest.mark.asyncio
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_session_chat_completions(
        self, mock_query_router, mock_forward, client
    ):
        """Session key → /chat/completions → forwarded to pinned worker."""
        mock_query_router.return_value = WORKER_ADDR
        mock_forward.return_value = httpx.Response(
            200,
            json={"id": "chatcmpl-2", "choices": [{"message": {"content": "OK"}}]},
        )

        resp = await client.post(
            "/chat/completions",
            json={
                "model": "sglang",
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers=session_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["id"] == "chatcmpl-2"

    @pytest.mark.asyncio
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_session_set_reward_finish(
        self, mock_query_router, mock_forward, client
    ):
        """Session key → /rl/set_reward (finish=True) → forwarded to pinned worker."""
        mock_query_router.return_value = WORKER_ADDR
        mock_forward.return_value = httpx.Response(
            200,
            json={"message": "success", "interaction_count": 5, "finished": True},
        )

        resp = await client.post(
            "/rl/set_reward",
            json={"reward": 0.0, "finish": True},
            headers=session_headers(),
        )
        assert resp.status_code == 200
        mock_forward.assert_called_once()

    @pytest.mark.asyncio
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_session_set_reward(self, mock_query_router, mock_forward, client):
        """Session key → /rl/set_reward → forwarded to pinned worker."""
        mock_query_router.return_value = WORKER_ADDR
        mock_forward.return_value = httpx.Response(200, json={"message": "success"})

        resp = await client.post(
            "/rl/set_reward",
            json={"reward": 1.0},
            headers=session_headers(),
        )
        assert resp.status_code == 200
        mock_forward.assert_called_once()

    @pytest.mark.asyncio
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_set_reward_returns_ready_transition_without_router_notification(
        self, mock_query_router, mock_forward, client
    ):
        mock_query_router.return_value = WORKER_ADDR
        mock_forward.return_value = httpx.Response(
            200,
            json={
                "message": "success",
                "session_id": "__hitl__",
                "trajectory_id": 0,
                "trajectory_ready": True,
                "ready_transition": True,
            },
        )

        resp = await client.post(
            "/rl/set_reward",
            json={"reward": 1.0},
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["ready_transition"] is True

    @pytest.mark.asyncio
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_set_reward_duplicate_ready_transition_is_forwarded_as_is(
        self, mock_query_router, mock_forward, client
    ):
        mock_query_router.return_value = WORKER_ADDR
        mock_forward.return_value = httpx.Response(
            200,
            json={
                "message": "success",
                "session_id": "__hitl__",
                "trajectory_id": 0,
                "trajectory_ready": True,
                "ready_transition": False,
            },
        )

        resp = await client.post(
            "/rl/set_reward",
            json={"reward": 1.0},
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["ready_transition"] is False


# =============================================================================
# Broadcast endpoints
# =============================================================================


class TestBroadcast:
    @pytest.mark.asyncio
    @patch(f"{MODULE}.broadcast_to_workers", new_callable=AsyncMock)
    @patch(f"{MODULE}.resolve_worker_addr", new_callable=AsyncMock)
    async def test_pause_generation_targets_worker(
        self, mock_resolve, mock_broadcast, client
    ):
        """Admin key → /pause_generation/{worker_id} → resolves and targets single worker."""
        mock_resolve.return_value = WORKER_ADDR
        mock_broadcast.return_value = [
            {"worker_addr": WORKER_ADDR, "status": 200, "ok": True},
        ]

        resp = await client.post(
            "/pause_generation/some-worker-id",
            content=b"{}",
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 1
        assert data["results"][0]["ok"] is True
        mock_resolve.assert_called_once()

    @pytest.mark.asyncio
    @patch(f"{MODULE}.broadcast_to_workers", new_callable=AsyncMock)
    @patch(f"{MODULE}.resolve_worker_addr", new_callable=AsyncMock)
    async def test_continue_generation_targets_worker(
        self, mock_resolve, mock_broadcast, client
    ):
        """Admin key → /continue_generation/{worker_id} → resolves and targets single worker."""
        mock_resolve.return_value = WORKER_ADDR
        mock_broadcast.return_value = [
            {"worker_addr": WORKER_ADDR, "status": 200, "ok": True},
        ]

        resp = await client.post(
            "/continue_generation/some-worker-id",
            content=b"{}",
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        assert len(resp.json()["results"]) == 1
        mock_resolve.assert_called_once()

    @pytest.mark.asyncio
    @patch(f"{MODULE}.broadcast_to_workers", new_callable=AsyncMock)
    @patch(f"{MODULE}.resolve_worker_addr", new_callable=AsyncMock)
    async def test_pause_worker_broadcast_failure(
        self, mock_resolve, mock_broadcast, client
    ):
        """Worker returns error → response shows ok=False for that worker."""
        mock_resolve.return_value = WORKER_ADDR
        mock_broadcast.return_value = [
            {
                "worker_addr": WORKER_ADDR,
                "status": 502,
                "ok": False,
                "error": "Connection refused",
            },
        ]

        resp = await client.post(
            "/pause_generation/some-worker-id",
            content=b"{}",
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert len(results) == 1
        assert results[0]["ok"] is False


# =============================================================================
# Router errors
# =============================================================================


class TestRouterErrors:
    @pytest.mark.asyncio
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_router_unreachable_502(self, mock_query_router, client):
        """Router down → 502 JSON error."""
        mock_query_router.side_effect = RouterUnreachableError(
            "Router unreachable: connect timeout"
        )

        resp = await client.post(
            "/chat/completions",
            json={
                "model": "sglang",
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers=admin_headers(),
        )
        assert resp.status_code == 502
        assert "Router unreachable" in resp.json()["error"]

    @pytest.mark.asyncio
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_no_healthy_workers_503(self, mock_query_router, client):
        """Router returns 503 (no healthy workers) → gateway returns 503."""
        mock_query_router.side_effect = RouterKeyRejectedError(
            "No healthy workers", 503
        )

        resp = await client.post(
            "/chat/completions",
            json={
                "model": "sglang",
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers=admin_headers(),
        )
        assert resp.status_code == 503
        assert "No healthy workers" in resp.json()["error"]

    @pytest.mark.asyncio
    @patch(f"{MODULE}.resolve_worker_addr", new_callable=AsyncMock)
    async def test_router_unreachable_on_pause(self, mock_resolve, client):
        """Router unreachable when resolving worker for pause → 502."""
        mock_resolve.side_effect = RouterUnreachableError("Router down")

        resp = await client.post(
            "/pause_generation/some-worker-id",
            content=b"{}",
            headers=admin_headers(),
        )
        assert resp.status_code == 502

    @pytest.mark.asyncio
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_session_not_found_for_export(self, mock_query_router, client):
        """Session not found for export_trajectories → 401."""
        mock_query_router.side_effect = RouterKeyRejectedError("Session not found", 404)

        resp = await client.post(
            "/export_trajectories",
            json={"session_ids": ["nonexistent"], "discount": 1.0, "style": "sft"},
            headers=admin_headers(),
        )
        assert resp.status_code == 401
        assert "Session not found" in resp.json()["error"]


# =============================================================================
# Capacity enforcement — /rl/start_session with capacity checks
# =============================================================================


class TestStartSessionCapacity:
    @pytest.mark.asyncio
    @patch(f"{MODULE}.register_session_in_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_start_session_full_flow(
        self, mock_query_router, mock_forward, mock_register, client
    ):
        """Full flow: route → forward → register session.

        Gateway no longer manages capacity — that is handled by the
        router's ``/register_session`` endpoint.
        """
        mock_query_router.return_value = WORKER_ADDR
        mock_forward.return_value = httpx.Response(
            201,
            json={
                "group_id": "grp-test-3",
                "sessions": [{"session_id": "t-0", "session_api_key": "k"}],
            },
        )

        resp = await client.post(
            "/rl/start_session",
            json={"task_id": "t"},
            headers=admin_headers(),
        )
        assert resp.status_code == 201

        # Verify call order: route → forward → register
        mock_query_router.assert_called_once()
        mock_forward.assert_called_once()
        mock_register.assert_called_once()
