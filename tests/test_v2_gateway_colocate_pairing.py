# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest import mock

import httpx
import pytest
from fastapi.testclient import TestClient

from areal.v2.weight_update.gateway import app as gateway_app
from areal.v2.weight_update.gateway.config import PairInfo


def _auth_headers():
    return {"Authorization": "Bearer areal-admin-key"}


def test_colocate_connect_uses_physical_pairing_without_legacy_weight_meta():
    reports = {
        "http://train0/awex/report_parallelism?include_device=1": {
            "world_size": 2,
            "tp_size": 1,
            "pp_size": 1,
            "ip": "192.0.2.1",
            "device_id": 0,
        },
        "http://train1/awex/report_parallelism?include_device=1": {
            "world_size": 2,
            "tp_size": 1,
            "pp_size": 1,
            "ip": "192.0.2.1",
            "device_id": 1,
        },
        "http://infer/awex/report_parallelism?include_device=1": {
            "world_size": 2,
            "tp_size": 2,
            "pp_size": 1,
            "ip": "192.0.2.1",
            "device_id": 0,
        },
    }
    init_calls = []

    async def get_json(_session, url, _timeout):
        return reports[url]

    async def post(_session, url, _timeout, json_data=None):
        init_calls.append((url, json_data))

    app = gateway_app.create_app()
    with (
        mock.patch.object(gateway_app, "_ensure_meta_server", return_value="meta:1"),
        mock.patch.object(gateway_app, "_get_json", side_effect=get_json),
        mock.patch.object(gateway_app, "_post", side_effect=post),
        mock.patch.object(gateway_app, "_post_json") as post_json,
        TestClient(app) as client,
    ):
        response = client.post(
            "/connect",
            headers=_auth_headers(),
            json={
                "pair_name": "actor-rollout",
                "train_worker_urls": ["http://train0", "http://train1"],
                "inference_worker_urls": ["http://infer"],
                "mode": "awex",
                "colocate": True,
            },
        )

    assert response.status_code == 200, response.text
    post_json.assert_not_called()
    assert {url: payload["transfer_rank"] for url, payload in init_calls} == {
        "http://infer/awex/init_colocate_weight_update": 0,
        "http://train0/awex/init_colocate_weight_update": 2,
        "http://train1/awex/init_colocate_weight_update": 3,
    }
    assert all("master_port" not in payload for _, payload in init_calls)


@pytest.mark.parametrize(
    "pair_name,workers,disconnect,status,error",
    [
        ("other-pair", ["http://train"], False, 409, "actor-rollout"),
        ("actor-rollout", ["http://train"], False, 200, None),
        ("actor-rollout", ["http://changed"], False, 409, "different worker endpoints"),
        ("actor-rollout", ["http://train"], True, 409, "gateway lifetime"),
    ],
)
def test_gateway_reconnect_preserves_pair_identity(
    pair_name, workers, disconnect, status, error
):
    app = gateway_app.create_app()
    app.state.colocate_pair_name = "actor-rollout"
    app.state.colocate_state = "active"
    app.state.registry.register(
        PairInfo(
            pair_name="actor-rollout",
            train_worker_urls=["http://train"],
            inference_worker_urls=["http://infer"],
            colocate=True,
        )
    )
    with TestClient(app) as client:
        if disconnect:
            response = client.post(
                "/disconnect",
                headers=_auth_headers(),
                json={"pair_name": "actor-rollout"},
            )
            assert response.status_code == 200
        response = client.post(
            "/connect",
            headers=_auth_headers(),
            json={
                "pair_name": pair_name,
                "train_worker_urls": workers,
                "inference_worker_urls": ["http://infer"],
                "mode": "awex",
                "colocate": True,
            },
        )
    assert response.status_code == status
    if error:
        assert error in response.json()["error"]


@pytest.mark.parametrize(
    "override,error",
    [
        ({"mode": "disk"}, "mode='awex'"),
        ({"use_lora": True}, "does not support LoRA"),
    ],
)
def test_gateway_rejects_unsupported_colocate_mode(override, error):
    with TestClient(gateway_app.create_app()) as client:
        response = client.post(
            "/connect",
            headers=_auth_headers(),
            json={
                "pair_name": "actor-rollout",
                "train_worker_urls": ["http://train"],
                "inference_worker_urls": ["http://infer"],
                "mode": "awex",
                "colocate": True,
                **override,
            },
        )
    assert response.status_code == 400
    assert error in response.json()["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "failure", "cancelled"])
async def test_colocate_connect_reserves_session_and_quarantines_failed_init(
    monkeypatch, outcome
):
    app = gateway_app.create_app()
    app.state.http_session = object()
    entered = asyncio.Event()
    release = asyncio.Event()
    init_calls = []

    async def report(*args):
        return {"ip": "192.0.2.1", "device_id": 0, "tp_size": 1, "pp_size": 1}

    async def initialize(session, url, timeout, json_data):
        init_calls.append(json_data["pair_name"])
        if "train" in url:
            entered.set()
            await release.wait()
            if outcome == "failure":
                raise RuntimeError("training worker initialization failed")

    monkeypatch.setattr(gateway_app, "_get_json", report)
    monkeypatch.setattr(gateway_app, "_post", initialize)
    monkeypatch.setattr(gateway_app, "_ensure_meta_server", lambda app: "meta:1")
    body = {
        "pair_name": "a",
        "train_worker_urls": ["http://train"],
        "inference_worker_urls": ["http://infer"],
        "mode": "awex",
        "colocate": True,
    }
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://gateway", headers=_auth_headers()
    ) as client:
        first = asyncio.create_task(client.post("/connect", json=body))
        try:
            await asyncio.wait_for(entered.wait(), 5)
            assert app.state.colocate_state == "connecting"
            for name in ("a", "b"):
                response = await client.post(
                    "/connect", json={**body, "pair_name": name}
                )
                assert response.status_code == 409
            assert init_calls == ["a", "a"]
            if outcome == "cancelled":
                first.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await first
            else:
                release.set()
                response = await first
                assert response.status_code == (200 if outcome == "success" else 500)
            expected = "active" if outcome == "success" else "failed"
            assert app.state.colocate_state == expected
            assert app.state.colocate_pair_name == "a"
            for name in ("a", "b"):
                response = await client.post(
                    "/connect", json={**body, "pair_name": name}
                )
                assert response.status_code == (
                    200 if outcome == "success" and name == "a" else 409
                )
            assert init_calls == ["a", "a"]
            assert (app.state.registry.get_by_name("a") is not None) == (
                outcome == "success"
            )
            assert app.state.registry.get_by_name("b") is None
        finally:
            release.set()
            if not first.done():
                first.cancel()
            await asyncio.gather(first, return_exceptions=True)
