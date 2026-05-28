"""Unit tests for training-service router and registry behavior."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import pytest_asyncio

from areal.experimental.training_service.router.app import (
    _probe_model_health,
    create_app,
)
from areal.experimental.training_service.router.config import RouterConfig

ADMIN_KEY = "router-admin-key"


def _admin_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {ADMIN_KEY}"}


@pytest.fixture
def config() -> RouterConfig:
    return RouterConfig(
        host="127.0.0.1",
        port=18081,
        admin_api_key=ADMIN_KEY,
        poll_interval=3600.0,
        worker_health_timeout=0.5,
    )


@pytest_asyncio.fixture
async def app_client(config):
    app = create_app(config)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield app, c


class TestRouterHealthAndRegistry:
    @pytest.mark.asyncio
    async def test_probe_model_health_uses_provided_client(self, app_client):
        app, _client = app_client
        model_addr = "http://worker-health:19001"
        await app.state.model_registry.register(model_addr, "health-key")

        health_client = MagicMock()
        health_client.get = AsyncMock(return_value=httpx.Response(200))

        await _probe_model_health(app.state.model_registry, model_addr, health_client)

        health_client.get.assert_awaited_once_with(f"{model_addr}/health")
        models = await app.state.model_registry.get_all()
        assert models[0].is_healthy is True
        assert models[0].consecutive_health_failures == 0

    @pytest.mark.asyncio
    async def test_probe_model_health_requires_two_consecutive_failures(
        self, app_client
    ):
        app, _client = app_client
        model_addr = "http://worker-flaky:19001"
        await app.state.model_registry.register(model_addr, "health-key")

        health_client = MagicMock()
        health_client.get = AsyncMock(side_effect=httpx.ConnectTimeout("boom"))

        await _probe_model_health(app.state.model_registry, model_addr, health_client)

        models = await app.state.model_registry.get_all()
        assert models[0].is_healthy is True
        assert models[0].consecutive_health_failures == 1

        await _probe_model_health(app.state.model_registry, model_addr, health_client)

        models = await app.state.model_registry.get_all()
        assert models[0].is_healthy is False
        assert models[0].consecutive_health_failures == 2

    @pytest.mark.asyncio
    async def test_probe_model_health_success_resets_failure_streak(self, app_client):
        app, _client = app_client
        model_addr = "http://worker-recover:19001"
        await app.state.model_registry.register(model_addr, "health-key")

        failing_client = MagicMock()
        failing_client.get = AsyncMock(side_effect=httpx.ConnectTimeout("boom"))

        await _probe_model_health(app.state.model_registry, model_addr, failing_client)

        healthy_client = MagicMock()
        healthy_client.get = AsyncMock(return_value=httpx.Response(200))

        await _probe_model_health(app.state.model_registry, model_addr, healthy_client)

        models = await app.state.model_registry.get_all()
        assert models[0].is_healthy is True
        assert models[0].consecutive_health_failures == 0

    @pytest.mark.asyncio
    async def test_health_reports_model_count(self, app_client):
        _app, client = app_client
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["models"] == 0

    @pytest.mark.asyncio
    async def test_register_then_route_success(self, app_client):
        _app, client = app_client
        model_addr = "http://worker-a:19001"
        model_api_key = "model-key-a"

        resp = await client.post(
            "/register",
            json={"model_addr": model_addr, "api_key": model_api_key},
            headers=_admin_headers(),
        )
        assert resp.status_code == 200

        resp = await client.post(
            "/route",
            json={"api_key": model_api_key},
            headers=_admin_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["model_addr"] == model_addr

    @pytest.mark.asyncio
    async def test_route_unknown_key_returns_404(self, app_client):
        _app, client = app_client
        resp = await client.post(
            "/route",
            json={"api_key": "unknown-key"},
            headers=_admin_headers(),
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_route_rejects_admin_key_as_data_key(self, app_client):
        _app, client = app_client
        resp = await client.post(
            "/route",
            json={"api_key": ADMIN_KEY},
            headers=_admin_headers(),
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_route_unhealthy_model_returns_503(self, app_client):
        app, client = app_client
        model_addr = "http://worker-b:19001"
        model_api_key = "model-key-b"

        resp = await client.post(
            "/register",
            json={"model_addr": model_addr, "api_key": model_api_key},
            headers=_admin_headers(),
        )
        assert resp.status_code == 200

        await app.state.model_registry.update_health(model_addr, False)
        await app.state.model_registry.update_health(model_addr, False)

        resp = await client.post(
            "/route",
            json={"api_key": model_api_key},
            headers=_admin_headers(),
        )
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_route_tolerates_single_transient_health_failure(self, app_client):
        app, client = app_client
        model_addr = "http://worker-c:19001"
        model_api_key = "model-key-c"

        resp = await client.post(
            "/register",
            json={"model_addr": model_addr, "api_key": model_api_key},
            headers=_admin_headers(),
        )
        assert resp.status_code == 200

        await app.state.model_registry.update_health(model_addr, False)

        resp = await client.post(
            "/route",
            json={"api_key": model_api_key},
            headers=_admin_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["model_addr"] == model_addr
