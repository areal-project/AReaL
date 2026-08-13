# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from starlette.testclient import TestClient

from areal.v2.inference_service.sglang.awex import register_awex_endpoints
from areal.v2.weight_update.awex.fsdp_adapter import AwexFSDPAdapter
from areal.v2.weight_update.awex.state import AwexPairState
from areal.v2.weight_update.gateway import app as gateway_app
from areal.v2.weight_update.gateway.config import WeightUpdateConfig

ADMIN_HEADERS = {"Authorization": "Bearer test-key"}
CONNECT_BODY = {
    "pair_name": "actor-rollout-v1",
    "train_worker_urls": ["http://train-0", "http://train-1"],
    "inference_worker_urls": ["http://infer-0", "http://infer-1"],
    "mode": "awex",
    "nccl_master_addr": "127.0.0.1",
    "nccl_master_port": 29500,
    "setup_timeout_s": 5.0,
    "rollback_timeout_s": 1.0,
}


def _install_gateway_rpc_stubs(monkeypatch, *, fail_init_url: str | None = None):
    calls: list[tuple[str, dict | None]] = []
    state = {"fail_teardown": False}

    async def fake_get_json(_session, _url, _timeout_s):
        return {"world_size": 1}

    async def fake_post_json(_session, _url, _timeout_s, json_data=None):
        return {"result": []}

    async def fake_post(_session, url, _timeout_s, json_data=None):
        calls.append((url, json_data))
        if fail_init_url is not None and url == fail_init_url:
            raise RuntimeError("partial init failed")
        if state["fail_teardown"] and url.endswith("/awex/teardown"):
            raise RuntimeError("teardown failed")

    monkeypatch.setattr(gateway_app, "_get_json", fake_get_json)
    monkeypatch.setattr(gateway_app, "_post_json", fake_post_json)
    monkeypatch.setattr(gateway_app, "_post", fake_post)
    return calls, state


def _create_gateway_client(*, raise_server_exceptions: bool = True):
    app = gateway_app.create_app(
        WeightUpdateConfig(
            admin_api_key="test-key",
            init_timeout_s=5,
            update_timeout_s=5,
        )
    )
    client = TestClient(app, raise_server_exceptions=raise_server_exceptions)
    return app, client


def test_disconnect_tears_down_all_train_and_inference_workers(monkeypatch):
    calls, _ = _install_gateway_rpc_stubs(monkeypatch)
    app, client = _create_gateway_client()

    with client:
        assert (
            client.post(
                "/connect", json=CONNECT_BODY, headers=ADMIN_HEADERS
            ).status_code
            == 200
        )
        response = client.post(
            "/disconnect",
            json={"pair_name": CONNECT_BODY["pair_name"]},
            headers=ADMIN_HEADERS,
        )

    assert response.status_code == 200
    teardown_calls = {
        (url, payload["pair_name"])
        for url, payload in calls
        if url.endswith("/awex/teardown") and payload is not None
    }
    assert teardown_calls == {
        ("http://train-0/awex/teardown", "actor-rollout-v1"),
        ("http://train-1/awex/teardown", "actor-rollout-v1"),
        ("http://infer-0/awex/teardown", "actor-rollout-v1"),
        ("http://infer-1/awex/teardown", "actor-rollout-v1"),
    }
    assert app.state.registry.get_by_name("actor-rollout-v1") is None


def test_partial_connect_failure_rolls_back_every_worker(monkeypatch):
    calls, _ = _install_gateway_rpc_stubs(
        monkeypatch,
        fail_init_url="http://infer-1/awex/init_weights_update_group",
    )
    app, client = _create_gateway_client(raise_server_exceptions=False)

    with client:
        response = client.post("/connect", json=CONNECT_BODY, headers=ADMIN_HEADERS)

    assert response.status_code == 500
    teardown_urls = {url for url, _ in calls if url.endswith("/awex/teardown")}
    assert teardown_urls == {
        "http://train-0/awex/teardown",
        "http://train-1/awex/teardown",
        "http://infer-0/awex/teardown",
        "http://infer-1/awex/teardown",
    }
    assert app.state.registry.get_by_name("actor-rollout-v1") is None
    assert app.state.kv_store.get("actor-rollout-v1", "training_params_meta") is None
    assert app.state.kv_store.get("actor-rollout-v1", "infer_params_meta") is None


def test_disconnect_failure_keeps_pair_registered_for_retry(monkeypatch):
    _, state = _install_gateway_rpc_stubs(monkeypatch)
    app, client = _create_gateway_client()

    with client:
        assert (
            client.post(
                "/connect", json=CONNECT_BODY, headers=ADMIN_HEADERS
            ).status_code
            == 200
        )
        state["fail_teardown"] = True
        failed = client.post(
            "/disconnect",
            json={"pair_name": CONNECT_BODY["pair_name"]},
            headers=ADMIN_HEADERS,
        )
        assert failed.status_code == 500
        assert app.state.registry.get_by_name("actor-rollout-v1") is not None

        state["fail_teardown"] = False
        retried = client.post(
            "/disconnect",
            json={"pair_name": CONNECT_BODY["pair_name"]},
            headers=ADMIN_HEADERS,
        )

    assert retried.status_code == 200
    assert app.state.registry.get_by_name("actor-rollout-v1") is None


def test_inference_teardown_endpoint_dispatches_collectively():
    app = FastAPI()
    rpc_proxy = MagicMock()
    register_awex_endpoints(app, rpc_proxy)

    response = TestClient(app).post(
        "/awex/teardown", json={"pair_name": "actor-rollout-v1"}
    )

    assert response.status_code == 200
    rpc_proxy.collective_rpc.assert_called_once_with(
        "awex_teardown_weight_update_group",
        pair_name="actor-rollout-v1",
    )


def test_adapter_teardown_is_idempotent_and_pair_scoped():
    adapter = AwexFSDPAdapter(MagicMock())
    group_a = object()
    group_b = object()
    adapter._pair_states = {
        "pair-a": AwexPairState(group_a, MagicMock(), 0),
        "pair-b": AwexPairState(group_b, MagicMock(), 1),
    }

    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.destroy_process_group") as destroy_group,
    ):
        adapter.teardown_weight_update_group("pair-a")
        adapter.teardown_weight_update_group("pair-a")

    destroy_group.assert_called_once_with(group_a)
    assert "pair-a" not in adapter._pair_states
    assert adapter._pair_states["pair-b"].weights_update_group is group_b
