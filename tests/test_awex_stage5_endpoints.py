# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import pytest

from areal.infra.rpc.serialization import deserialize_value, serialize_value


def _worker_client(monkeypatch, adapter):
    flask = pytest.importorskip("flask")
    from areal.v2.training_service.worker import awex as awex_mod

    app = flask.Flask(__name__)
    monkeypatch.setattr(awex_mod, "_create_training_adapter", lambda engine: adapter)

    def submit_to_engine_thread(_name, action):
        return action()

    def run_endpoint(_name, action, return_result=True):
        result = action()
        if return_result:
            return flask.jsonify(
                {"status": "success", "result": serialize_value(result)}
            )
        return flask.jsonify({"status": "success", "result": None})

    app.register_blueprint(
        awex_mod.create_awex_blueprint(
            flask_module=flask,
            get_engine=lambda: object(),
            submit_to_engine_thread=submit_to_engine_thread,
            run_endpoint=run_endpoint,
        )
    )
    return app.test_client()


def test_training_worker_exposes_delta_precompute_and_seed(monkeypatch):
    events = []
    adapter = SimpleNamespace(
        precompute_delta_masks=lambda version: events.append(("precompute", version))
        or True,
        seed_delta_base=lambda version: events.append(("seed", version)),
    )
    client = _worker_client(monkeypatch, adapter)

    precompute_resp = client.post("/awex/precompute_delta_masks", json={"version": 7})
    assert precompute_resp.status_code == 200
    precompute_payload = deserialize_value(precompute_resp.get_json()["result"])
    assert precompute_payload == {"precomputed": True}

    seed_resp = client.post("/awex/seed_delta_base", json={"version": 7})
    assert seed_resp.status_code == 200

    assert events == [("precompute", 7), ("seed", 7)]


def test_training_worker_precompute_is_optional(monkeypatch):
    client = _worker_client(
        monkeypatch,
        SimpleNamespace(seed_delta_base=lambda version: None),
    )

    resp = client.post("/awex/precompute_delta_masks", json={"version": 3})

    assert resp.status_code == 200
    assert deserialize_value(resp.get_json()["result"]) == {"precomputed": False}


def test_sglang_fastapi_exposes_seed_delta_base():
    fastapi = pytest.importorskip("fastapi")
    testclient = pytest.importorskip("starlette.testclient")
    from areal.v2.inference_service.sglang.awex import register_awex_endpoints

    calls = []

    class RpcProxy:
        def collective_rpc(self, method, **kwargs):
            calls.append((method, kwargs))

    app = fastapi.FastAPI()
    register_awex_endpoints(app, RpcProxy())
    client = testclient.TestClient(app)

    resp = client.post("/awex/seed_delta_base", json={"version": 11})

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "version": 11}
    assert calls == [("awex_seed_delta_base", {"version": 11})]


def test_sglang_scheduler_bridge_binds_seed_delta_base(monkeypatch):
    monkeypatch.delenv("AREAL_AWEX_RESULT_IPC", raising=False)
    from areal.v2.inference_service.sglang.scheduler import AwexSchedulerBridge

    events = []
    scheduler = SimpleNamespace(tp_rank=0, dp_rank=0)
    bridge = AwexSchedulerBridge(scheduler)
    bridge._adapter = SimpleNamespace(
        seed_delta_base=lambda version: events.append(("seed", version))
    )

    bridge.bind()
    scheduler.awex_seed_delta_base(version=13)

    assert events == [("seed", 13)]
