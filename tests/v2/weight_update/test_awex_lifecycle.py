# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import flask
import httpx
import pytest
from starlette.testclient import TestClient

from areal.v2.training_service.worker.awex import create_awex_blueprint
from areal.v2.weight_update.awex import megatron_adapter, sglang_adapter
from areal.v2.weight_update.gateway import app as gateway_app
from areal.v2.weight_update.gateway.config import WeightUpdateConfig

ADMIN_HEADERS = {"Authorization": "Bearer test-key"}
PAIR_NAME = "actor-rollout-v1"
CONNECT_BODY = {
    "pair_name": PAIR_NAME,
    "operation_id": "connect-op-1",
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
    state = {"fail_teardown": False}

    async def fake_get_json(_session, _url, _timeout_s):
        return {"world_size": 1}

    async def fake_post_json(_session, _url, _timeout_s, json_data=None):
        return {"result": []}

    async def fake_post(_session, url, _timeout_s, json_data=None):
        if fail_init_url is not None and url == fail_init_url:
            raise RuntimeError("partial init failed")
        if state["fail_teardown"] and url.endswith("/awex/teardown"):
            raise RuntimeError("teardown failed")

    monkeypatch.setattr(gateway_app, "_get_json", fake_get_json)
    monkeypatch.setattr(gateway_app, "_post_json", fake_post_json)
    monkeypatch.setattr(gateway_app, "_post", fake_post)
    monkeypatch.setattr(gateway_app, "_post_once", fake_post)
    return state


def _create_gateway_client(*, raise_server_exceptions: bool = True):
    app = gateway_app.create_app(
        WeightUpdateConfig(
            admin_api_key="test-key",
            init_timeout_s=5,
            update_timeout_s=5,
        )
    )
    return app, TestClient(app, raise_server_exceptions=raise_server_exceptions)


def _create_training_worker(monkeypatch):
    queued = []
    adapter = MagicMock()
    factory = MagicMock(return_value=adapter)

    def run_endpoint(_name, submitter, **_kwargs):
        submitter()
        return flask.jsonify({"status": "ok"})

    monkeypatch.setattr(
        "areal.v2.training_service.worker.awex._create_training_adapter", factory
    )
    app = flask.Flask(__name__)
    app.register_blueprint(
        create_awex_blueprint(
            flask_module=flask,
            get_engine=lambda: MagicMock(),
            submit_to_engine_thread=lambda _name, action: queued.append(action),
            run_endpoint=run_endpoint,
        )
    )
    return app.test_client(), queued, factory, adapter


def test_partial_connect_rollback_keeps_bookkeeping_until_retry(monkeypatch):
    state = _install_gateway_rpc_stubs(
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


@pytest.mark.asyncio
async def test_disconnect_invalidates_blocked_connect_before_publish(monkeypatch):
    init_started = asyncio.Event()
    release_init = asyncio.Event()
    teardown_calls = []

    async def fake_get_json(_session, _url, _timeout_s):
        return {"world_size": 1}

    async def fake_post_json(_session, _url, _timeout_s, json_data=None):
        return {"result": []}

    async def fake_post_once(_session, url, _timeout_s, json_data=None):
        if url.endswith("/init_weights_update_group"):
            init_started.set()
            await release_init.wait()
        elif url.endswith("/teardown"):
            teardown_calls.append((url, json_data))

    monkeypatch.setattr(gateway_app, "_get_json", fake_get_json)
    monkeypatch.setattr(gateway_app, "_post_json", fake_post_json)
    monkeypatch.setattr(gateway_app, "_post_once", fake_post_once)
    app = gateway_app.create_app(
        WeightUpdateConfig(admin_api_key="test-key", init_timeout_s=5)
    )
    app.state.http_session = object()
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        connect_task = asyncio.create_task(
            client.post("/connect", json=CONNECT_BODY, headers=ADMIN_HEADERS)
        )
        await asyncio.wait_for(init_started.wait(), timeout=1)
        disconnect_task = asyncio.create_task(
            client.post(
                "/disconnect",
                json={
                    "pair_name": PAIR_NAME,
                    "operation_id": CONNECT_BODY["operation_id"],
                    "timeout_s": 2.0,
                },
                headers=ADMIN_HEADERS,
            )
        )
        for _ in range(100):
            pair = app.state.registry.get_by_name(PAIR_NAME)
            if pair is not None and pair.status == "cleanup_pending":
                break
            await asyncio.sleep(0.01)
        assert app.state.registry.get_by_name(PAIR_NAME).status == "cleanup_pending"
        release_init.set()
        connect_response, disconnect_response = await asyncio.gather(
            connect_task, disconnect_task
        )

    assert connect_response.status_code == 500
    assert disconnect_response.status_code == 200
    assert app.state.registry.get_by_name(PAIR_NAME) is None
    assert teardown_calls


@pytest.mark.asyncio
async def test_cancelled_disconnect_preserves_inflight_operation(monkeypatch):
    update_started = asyncio.Event()
    release_update = asyncio.Event()
    teardown_calls = []

    async def fake_get_json(_session, _url, _timeout_s):
        return {"world_size": 1}

    async def fake_post_json(_session, _url, _timeout_s, json_data=None):
        return {"result": []}

    async def fake_post_once(_session, url, _timeout_s, json_data=None):
        if url.endswith("/update_weights"):
            update_started.set()
            await release_update.wait()
        elif url.endswith("/teardown"):
            teardown_calls.append((url, json_data))

    monkeypatch.setattr(gateway_app, "_get_json", fake_get_json)
    monkeypatch.setattr(gateway_app, "_post_json", fake_post_json)
    monkeypatch.setattr(gateway_app, "_post_once", fake_post_once)
    app = gateway_app.create_app(WeightUpdateConfig(admin_api_key="test-key"))
    app.state.http_session = object()
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        assert (
            await client.post("/connect", json=CONNECT_BODY, headers=ADMIN_HEADERS)
        ).is_success
        update = asyncio.create_task(
            client.post(
                "/update_weights",
                json={"pair_name": PAIR_NAME, "version": 1},
                headers=ADMIN_HEADERS,
            )
        )
        await asyncio.wait_for(update_started.wait(), timeout=1)
        disconnect = asyncio.create_task(
            client.post(
                "/disconnect",
                json={"pair_name": PAIR_NAME, "timeout_s": 2.0},
                headers=ADMIN_HEADERS,
            )
        )
        for _ in range(100):
            pair = app.state.registry.get_by_name(PAIR_NAME)
            if pair is not None and pair.status == "cleanup_pending":
                break
            await asyncio.sleep(0.01)

        disconnect.cancel()
        with pytest.raises(asyncio.CancelledError):
            await disconnect

        assert not teardown_calls
        assert app.state.registry.get_by_name(PAIR_NAME) is pair
        assert pair.status == "cleanup_pending"
        assert PAIR_NAME in app.state.operations
        assert app.state.kv_store.get(PAIR_NAME, "training_params_meta") == []

        release_update.set()
        update_response = await update
        assert update_response.json()["status"] == "error"
        retried = await client.post(
            "/disconnect",
            json={"pair_name": PAIR_NAME, "timeout_s": 2.0},
            headers=ADMIN_HEADERS,
        )

    assert retried.is_success
    assert teardown_calls
    assert app.state.registry.get_by_name(PAIR_NAME) is None
    assert PAIR_NAME not in app.state.operations


@pytest.mark.asyncio
async def test_disconnect_waits_for_all_dispatched_update_requests(monkeypatch):
    blocked = asyncio.Event()
    release = asyncio.Event()
    teardown_started = asyncio.Event()

    async def fake_get_json(_session, _url, _timeout_s):
        return {"world_size": 1}

    async def fake_post_json(_session, _url, _timeout_s, json_data=None):
        return {"result": []}

    async def fake_post_once(_session, url, _timeout_s, json_data=None):
        if url == "http://train-0/awex/update_weights":
            raise RuntimeError("rank failed")
        if url.endswith("/update_weights"):
            blocked.set()
            await release.wait()
        if url.endswith("/teardown"):
            teardown_started.set()

    monkeypatch.setattr(gateway_app, "_get_json", fake_get_json)
    monkeypatch.setattr(gateway_app, "_post_json", fake_post_json)
    monkeypatch.setattr(gateway_app, "_post_once", fake_post_once)
    app = gateway_app.create_app(WeightUpdateConfig(admin_api_key="test-key"))
    app.state.http_session = object()
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        assert (
            await client.post("/connect", json=CONNECT_BODY, headers=ADMIN_HEADERS)
        ).is_success
        update = asyncio.create_task(
            client.post(
                "/update_weights",
                json={"pair_name": PAIR_NAME, "version": 1},
                headers=ADMIN_HEADERS,
            )
        )
        await asyncio.wait_for(blocked.wait(), timeout=1)
        disconnect = asyncio.create_task(
            client.post(
                "/disconnect",
                json={"pair_name": PAIR_NAME, "timeout_s": 2.0},
                headers=ADMIN_HEADERS,
            )
        )
        await asyncio.sleep(0.05)
        assert not teardown_started.is_set()
        release.set()
        update_response, disconnect_response = await asyncio.gather(update, disconnect)

    assert update_response.json()["status"] == "error"
    assert disconnect_response.is_success
    assert teardown_started.is_set()


def test_training_worker_teardown_tombstone_blocks_queued_init(monkeypatch):
    client, queued, adapter_factory, _ = _create_training_worker(monkeypatch)
    init = {
        "pair_name": PAIR_NAME,
        "operation_id": "late-op",
        "operation_ttl_s": 30.0,
    }
    assert client.post("/awex/init_weights_update_group", json=init).status_code == 200
    assert (
        client.post(
            "/awex/teardown",
            json={"pair_name": PAIR_NAME, "operation_id": "late-op"},
        ).status_code
        == 200
    )

    with pytest.raises(RuntimeError, match="was cancelled"):
        queued[0]()
    queued[1]()
    assert client.post("/awex/init_weights_update_group", json=init).status_code == 200
    with pytest.raises(RuntimeError, match="was cancelled"):
        queued[2]()
    adapter_factory.assert_not_called()


def test_training_worker_deducts_queue_time_from_pg_timeout(monkeypatch):
    client, queued, _, adapter = _create_training_worker(monkeypatch)
    clock = iter([100.0, 104.0])
    monkeypatch.setattr(
        "areal.v2.training_service.worker.awex.time.monotonic", lambda: next(clock)
    )

    response = client.post(
        "/awex/init_weights_update_group",
        json={
            "pair_name": PAIR_NAME,
            "operation_id": "queued-op",
            "operation_ttl_s": 10.0,
            "process_group_timeout_s": 9.0,
        },
    )
    assert response.status_code == 200
    queued[0]()

    assert adapter.init_weight_update_group.call_args.kwargs[
        "process_group_timeout_s"
    ] == pytest.approx(6.0)


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
