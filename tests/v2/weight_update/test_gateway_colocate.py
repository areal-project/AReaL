# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from areal.v2.inference_service.sglang.awex import register_awex_endpoints
from areal.v2.weight_update.gateway import app as gateway_app
from areal.v2.weight_update.gateway.config import WeightUpdateConfig


def _connect_payload(**overrides):
    payload = {
        "pair_name": "actor-rollout",
        "train_worker_urls": ["http://train-0"],
        "inference_worker_urls": ["http://infer-0"],
        "mode": "awex",
        "use_lora": False,
        "colocate": True,
    }
    payload.update(overrides)
    return payload


def _client():
    app = gateway_app.create_app(
        WeightUpdateConfig(host="127.0.0.1", admin_api_key="test-key")
    )
    return app, TestClient(app, headers={"Authorization": "Bearer test-key"})


@pytest.mark.asyncio
async def test_remote_worker_error_preserves_json_detail():
    """Actionable worker response text is retained by the gateway client."""

    class Response:
        status = 400

        async def json(self, *, content_type=None):
            del content_type
            return {"error": "set megatron.wrap_with_ddp=true"}

        async def text(self):
            return "unused"

    with pytest.raises(
        gateway_app.RemoteWorkerResponseError,
        match="megatron.wrap_with_ddp=true",
    ):
        await gateway_app._raise_for_remote_status(Response(), "http://train-0")


def test_colocate_lora_rejected_before_contacting_workers(monkeypatch):
    """The common AWEX contract applies before the colocated branch."""
    get_json = AsyncMock()
    post_json = AsyncMock()
    post = AsyncMock()
    monkeypatch.setattr(gateway_app, "_get_json", get_json)
    monkeypatch.setattr(gateway_app, "_post_json", post_json)
    monkeypatch.setattr(gateway_app, "_post", post)
    app, client = _client()

    with client:
        response = client.post("/connect", json=_connect_payload(use_lora=True))

    assert response.status_code == 400
    assert "does not support LoRA" in response.json()["error"]
    get_json.assert_not_awaited()
    post_json.assert_not_awaited()
    post.assert_not_awaited()
    assert app.state.registry.get_by_name("actor-rollout") is None


def test_colocate_preflight_error_has_no_metadata_or_kv_side_effects(monkeypatch):
    """An unsupported training layout fails before gateway/inference init."""
    root_error = gateway_app.RemoteWorkerResponseError(
        "http://train-0/awex/preflight_colocate_weight_update",
        400,
        "set megatron.wrap_with_ddp=true",
    )
    get_json = AsyncMock()
    post_json = AsyncMock()
    post = AsyncMock(side_effect=root_error)
    monkeypatch.setattr(gateway_app, "_get_json", get_json)
    monkeypatch.setattr(gateway_app, "_post_json", post_json)
    monkeypatch.setattr(gateway_app, "_post", post)
    app, client = _client()

    with client:
        response = client.post("/connect", json=_connect_payload())

    assert response.status_code == 400
    assert "megatron.wrap_with_ddp=true" in response.json()["error"]
    get_json.assert_not_awaited()
    post_json.assert_not_awaited()
    assert post.await_count == 1
    assert app.state.kv_store.list_keys("actor-rollout") == []
    assert app.state.registry.get_by_name("actor-rollout") is None


def test_colocate_partial_init_failure_tears_down_all_peers(monkeypatch):
    """All concurrent init calls settle before rollback clears every peer."""
    calls: list[str] = []

    async def fake_get_json(_session, url, _timeout_s):
        return {"world_size": 1}

    async def fake_post_json(_session, _url, _timeout_s, json_data=None):
        del json_data
        return {"meta": []}

    async def fake_post(_session, url, _timeout_s, json_data=None):
        del json_data
        calls.append(url)
        if url == "http://train-0/awex/init_colocate_weight_update":
            raise gateway_app.RemoteWorkerResponseError(
                url, 500, "training init failed after inference init"
            )
        if url == "http://infer-0/awex/teardown":
            raise gateway_app.RemoteWorkerResponseError(
                url, 500, "inference teardown also failed"
            )

    monkeypatch.setattr(gateway_app, "_get_json", fake_get_json)
    monkeypatch.setattr(gateway_app, "_post_json", fake_post_json)
    monkeypatch.setattr(gateway_app, "_post", fake_post)
    monkeypatch.setattr(gateway_app, "find_free_ports", lambda _count: [2345])
    app, client = _client()

    with client:
        response = client.post("/connect", json=_connect_payload())

    assert response.status_code == 502
    assert "training init failed after inference init" in response.json()["error"]
    assert "http://infer-0/awex/teardown" in calls
    assert "http://train-0/awex/teardown" in calls
    assert app.state.kv_store.list_keys("actor-rollout") == []
    assert app.state.registry.get_by_name("actor-rollout") is None


@pytest.mark.asyncio
async def test_colocate_duplicate_connect_cannot_rollback_registered_pair(monkeypatch):
    """A same-name request is rejected before it can touch shared workers."""
    calls: list[str] = []
    init_started = asyncio.Event()
    allow_init = asyncio.Event()

    async def fake_get_json(_session, _url, _timeout_s):
        return {"world_size": 1}

    async def fake_post_json(_session, _url, _timeout_s, json_data=None):
        del json_data
        return {"meta": []}

    async def fake_post(_session, url, _timeout_s, json_data=None):
        del json_data
        calls.append(url)
        if url.endswith("/awex/init_colocate_weight_update"):
            init_started.set()
            await allow_init.wait()

    monkeypatch.setattr(gateway_app, "_get_json", fake_get_json)
    monkeypatch.setattr(gateway_app, "_post_json", fake_post_json)
    monkeypatch.setattr(gateway_app, "_post", fake_post)
    monkeypatch.setattr(gateway_app, "find_free_ports", lambda _count: [2345])
    app, _ = _client()
    app.state.http_session = object()
    transport = httpx.ASGITransport(app=app)
    headers = {"Authorization": "Bearer test-key"}

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
        headers=headers,
    ) as client:
        first_task = asyncio.create_task(
            client.post("/connect", json=_connect_payload())
        )
        await asyncio.wait_for(init_started.wait(), timeout=1)

        duplicate = await asyncio.wait_for(
            client.post("/connect", json=_connect_payload()),
            timeout=1,
        )
        assert duplicate.status_code == 409
        assert calls.count("http://train-0/awex/preflight_colocate_weight_update") == 1

        allow_init.set()
        first = await asyncio.wait_for(first_task, timeout=1)

    assert first.status_code == 200
    assert app.state.registry.get_by_name("actor-rollout") is not None
    assert app.state.kv_store.list_keys("actor-rollout") == [
        "training_params_meta",
        "infer_params_meta",
    ]
    assert not any(url.endswith("/awex/teardown") for url in calls)


def test_inference_teardown_endpoint_dispatches_collective_rpc():
    """Gateway rollback can tear down partial state on inference workers."""
    app = FastAPI()
    rpc_proxy = MagicMock()
    register_awex_endpoints(app, rpc_proxy)

    response = TestClient(app).post("/awex/teardown")

    assert response.status_code == 200
    rpc_proxy.collective_rpc.assert_called_once_with(
        "awex_teardown_weight_update_group"
    )
