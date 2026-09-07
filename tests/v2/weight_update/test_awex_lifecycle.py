# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from areal.v2.inference_service.sglang.awex import register_awex_endpoints
from areal.v2.weight_update.awex import megatron_adapter, sglang_adapter
from areal.v2.weight_update.awex.state import SGLangColocatePairState
from areal.v2.weight_update.gateway import app as gateway_app
from areal.v2.weight_update.gateway.config import WeightUpdateConfig

ADMIN_HEADERS = {"Authorization": "Bearer test-key"}
PAIR_NAME = "actor-rollout-v1"
CONNECT_BODY = {
    "pair_name": PAIR_NAME,
    "train_worker_urls": ["http://train-0", "http://train-1"],
    "inference_worker_urls": ["http://infer-0", "http://infer-1"],
    "mode": "awex",
    "nccl_master_addr": "127.0.0.1",
    "nccl_master_port": 29500,
    "setup_timeout_s": 5.0,
    "rollback_timeout_s": 1.0,
}


def _install_gateway_rpc_stubs(
    monkeypatch,
    *,
    fail_init_url: str | None = None,
):
    calls: list[tuple[str, dict | None]] = []
    state = {"fail_teardown": False, "fail_update": False}

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
        if state["fail_update"] and url.endswith("/awex/update_weights"):
            raise RuntimeError("inference rank wrote a partial payload")

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
    return app, TestClient(app, raise_server_exceptions=raise_server_exceptions)


def test_partial_connect_rollback_keeps_bookkeeping_until_retry(monkeypatch):
    _, state = _install_gateway_rpc_stubs(
        monkeypatch,
        fail_init_url="http://infer-1/awex/init_weights_update_group",
    )
    state["fail_teardown"] = True
    app, client = _create_gateway_client(raise_server_exceptions=False)

    with client:
        response = client.post("/connect", json=CONNECT_BODY, headers=ADMIN_HEADERS)
        pair_info = app.state.registry.get_by_name(PAIR_NAME)
        assert response.status_code == 500
        assert pair_info is not None
        assert pair_info.status == "cleanup_pending"
        assert app.state.kv_store.get(PAIR_NAME, "training_params_meta") == []

        state["fail_teardown"] = False
        retried = client.post(
            "/disconnect",
            json={"pair_name": PAIR_NAME},
            headers=ADMIN_HEADERS,
        )

    assert retried.status_code == 200
    assert app.state.registry.get_by_name(PAIR_NAME) is None
    assert app.state.kv_store.get(PAIR_NAME, "training_params_meta") is None


def test_disconnect_tears_down_all_workers_and_is_idempotent(monkeypatch):
    calls, _ = _install_gateway_rpc_stubs(monkeypatch)
    app, client = _create_gateway_client()

    with client:
        assert (
            client.post(
                "/connect", json=CONNECT_BODY, headers=ADMIN_HEADERS
            ).status_code
            == 200
        )
        first = client.post(
            "/disconnect", json={"pair_name": PAIR_NAME}, headers=ADMIN_HEADERS
        )
        second = client.post(
            "/disconnect", json={"pair_name": PAIR_NAME}, headers=ADMIN_HEADERS
        )

    assert first.status_code == second.status_code == 200
    teardown_calls = {
        (url, payload["pair_name"])
        for url, payload in calls
        if url.endswith("/awex/teardown") and payload is not None
    }
    assert teardown_calls == {
        ("http://train-0/awex/teardown", PAIR_NAME),
        ("http://train-1/awex/teardown", PAIR_NAME),
        ("http://infer-0/awex/teardown", PAIR_NAME),
        ("http://infer-1/awex/teardown", PAIR_NAME),
    }
    assert app.state.registry.get_by_name(PAIR_NAME) is None
    process_group_timeouts = [
        payload["process_group_timeout_s"]
        for url, payload in calls
        if "/awex/init_" in url and payload is not None
    ]
    assert process_group_timeouts
    assert all(
        0 < timeout < CONNECT_BODY["setup_timeout_s"]
        for timeout in process_group_timeouts
    )


def test_partial_transfer_error_is_marked_unsafe_via_worker_rpc(monkeypatch):
    _, state = _install_gateway_rpc_stubs(monkeypatch)
    app, client = _create_gateway_client()

    with client:
        assert (
            client.post(
                "/connect", json=CONNECT_BODY, headers=ADMIN_HEADERS
            ).status_code
            == 200
        )
        state["fail_update"] = True
        response = client.post(
            "/update_weights",
            json={"pair_name": PAIR_NAME, "version": 1},
            headers=ADMIN_HEADERS,
        )

    assert response.status_code == 200
    assert response.json()["status"] == "error"
    assert response.json()["inference_weights_may_be_mutated"] is True
    assert app.state.registry.get_by_name(PAIR_NAME).last_version == 0


@pytest.mark.parametrize(
    ("adapter_cls", "module", "adapter_setup"),
    [
        (
            megatron_adapter.AwexMegatronAdapter,
            megatron_adapter,
            lambda adapter: None,
        ),
        (
            sglang_adapter.AwexSGLangAdapter,
            sglang_adapter,
            lambda adapter: setattr(
                adapter,
                "_get_model_context",
                lambda: {"tp_size": 1, "tp_rank": 0, "pp_size": 1, "pp_rank": 0},
            ),
        ),
    ],
)
def test_candidate_init_failure_retains_group_until_teardown_retry(
    adapter_cls, module, adapter_setup, monkeypatch
):
    adapter = adapter_cls(MagicMock())
    adapter._dte_config = SimpleNamespace(enabled=False)
    adapter_setup(adapter)
    payload_group = MagicMock(name="payload_group")
    initializer = MagicMock(side_effect=[payload_group, RuntimeError("gloo failed")])
    destroy = MagicMock(side_effect=[RuntimeError("first destroy failed"), None])

    monkeypatch.setattr(
        module, "fetch_kv_metadata", lambda *args, **kwargs: (MagicMock(), MagicMock())
    )
    builder = MagicMock()
    builder.build_local_transfer_plan.return_value = MagicMock()
    monkeypatch.setattr(module, "TransferPlanBuilder", lambda **kwargs: builder)
    monkeypatch.setattr(module, "init_weights_update_group", initializer)
    monkeypatch.setattr(module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(module.dist, "destroy_process_group", destroy)

    with pytest.raises(RuntimeError, match="gloo failed"):
        adapter.init_weight_update_group(
            pair_name=PAIR_NAME,
            master_addr="127.0.0.1",
            master_port=29500,
            transfer_rank=1,
            world_size=2,
            kv_store_url="http://kv-store",
            infer_world_size=1,
            train_world_size=1,
            num_engines=1,
        )

    assert adapter._pair_states[PAIR_NAME].weights_update_group is payload_group
    adapter.teardown_weight_update_group(PAIR_NAME)
    assert PAIR_NAME not in adapter._pair_states
    assert destroy.call_args_list == [call(payload_group), call(payload_group)]


def test_sglang_colocate_teardown_retries_failed_transport(monkeypatch):
    adapter = sglang_adapter.AwexSGLangAdapter(MagicMock())
    group = MagicMock(name="colocate_group")
    client = MagicMock(name="http_client")
    transport = MagicMock(name="transport")
    transport.close.side_effect = [RuntimeError("transport busy"), None]
    adapter._colocate_pair_states[PAIR_NAME] = SGLangColocatePairState(
        weights_update_group=group,
        transfer_rank=0,
        kv_store_url="http://kv-store",
        infer_world_size=1,
        train_world_size=1,
        admin_api_key="test-key",
        timeout_s=5.0,
        http_client=client,
        transport=transport,
        train_to_infer_device_mapping={1: 0},
        infer_to_train_device_mapping={0: 1},
        send_transfer_plan=MagicMock(),
        recv_transfer_plan=MagicMock(),
    )
    monkeypatch.setattr(sglang_adapter.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(sglang_adapter.dist, "destroy_process_group", MagicMock())

    with pytest.raises(RuntimeError, match="Failed to teardown"):
        adapter.teardown_weight_update_group(PAIR_NAME)

    state = adapter._colocate_pair_states[PAIR_NAME]
    assert state.weights_update_group is None
    assert state.http_client is None
    assert state.transport is transport

    with pytest.raises(RuntimeError, match="partially initialized"):
        adapter.init_colocate_weight_update(
            pair_name=PAIR_NAME,
            kv_store_url="http://kv-store",
            transfer_rank=0,
            infer_world_size=1,
            train_world_size=1,
            num_engines=1,
            master_port=29500,
        )

    adapter.teardown_weight_update_group(PAIR_NAME)
    assert PAIR_NAME not in adapter._colocate_pair_states


def test_inference_teardown_endpoint_dispatches_collectively():
    app = FastAPI()
    rpc_proxy = MagicMock()
    register_awex_endpoints(app, rpc_proxy)

    response = TestClient(app).post("/awex/teardown", json={"pair_name": PAIR_NAME})

    assert response.status_code == 200
    rpc_proxy.collective_rpc.assert_called_once_with(
        "awex_teardown_weight_update_group",
        pair_name=PAIR_NAME,
    )
