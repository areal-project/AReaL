# SPDX-License-Identifier: Apache-2.0

from unittest import mock

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


def test_gateway_rejects_a_second_colocate_pair():
    app = gateway_app.create_app()
    app.state.colocate_pair_name = "actor-rollout"
    with TestClient(app) as client:
        response = client.post(
            "/connect",
            headers=_auth_headers(),
            json={
                "pair_name": "critic-rollout",
                "train_worker_urls": ["http://train"],
                "inference_worker_urls": ["http://infer"],
                "mode": "awex",
                "colocate": True,
            },
        )

    assert response.status_code == 409
    assert "actor-rollout" in response.json()["error"]


def test_gateway_retries_an_active_colocate_connect_idempotently():
    app = gateway_app.create_app()
    app.state.colocate_pair_name = "actor-rollout"
    app.state.registry.register(
        PairInfo(
            pair_name="actor-rollout",
            train_worker_urls=["http://train"],
            inference_worker_urls=["http://infer"],
            colocate=True,
        )
    )

    with TestClient(app) as client:
        response = client.post(
            "/connect",
            headers=_auth_headers(),
            json={
                "pair_name": "actor-rollout",
                "train_worker_urls": ["http://train"],
                "inference_worker_urls": ["http://infer"],
                "mode": "awex",
                "colocate": True,
            },
        )

    assert response.status_code == 200


def test_gateway_rejects_colocate_connect_with_non_awex_mode():
    app = gateway_app.create_app()
    with TestClient(app) as client:
        response = client.post(
            "/connect",
            headers=_auth_headers(),
            json={
                "pair_name": "actor-rollout",
                "train_worker_urls": ["http://train"],
                "inference_worker_urls": ["http://infer"],
                "mode": "disk",
                "colocate": True,
            },
        )

    assert response.status_code == 400
    assert "mode='awex'" in response.json()["error"]


def test_gateway_rejects_colocate_connect_with_lora():
    app = gateway_app.create_app()
    with TestClient(app) as client:
        response = client.post(
            "/connect",
            headers=_auth_headers(),
            json={
                "pair_name": "actor-rollout",
                "train_worker_urls": ["http://train"],
                "inference_worker_urls": ["http://infer"],
                "mode": "awex",
                "use_lora": True,
                "colocate": True,
            },
        )

    assert response.status_code == 400
    assert "does not support LoRA" in response.json()["error"]


def test_gateway_rejects_same_pair_name_with_changed_workers():
    app = gateway_app.create_app()
    app.state.colocate_pair_name = "actor-rollout"
    app.state.registry.register(
        PairInfo(
            pair_name="actor-rollout",
            train_worker_urls=["http://train-old"],
            inference_worker_urls=["http://infer"],
            colocate=True,
        )
    )

    with TestClient(app) as client:
        response = client.post(
            "/connect",
            headers=_auth_headers(),
            json={
                "pair_name": "actor-rollout",
                "train_worker_urls": ["http://train-new"],
                "inference_worker_urls": ["http://infer"],
                "mode": "awex",
                "colocate": True,
            },
        )

    assert response.status_code == 409
    assert "different worker endpoints" in response.json()["error"]


def test_gateway_rejects_colocate_reconnect_after_disconnect():
    app = gateway_app.create_app()
    app.state.colocate_pair_name = "actor-rollout"
    app.state.registry.register(
        PairInfo(
            pair_name="actor-rollout",
            train_worker_urls=["http://train"],
            inference_worker_urls=["http://infer"],
            colocate=True,
        )
    )

    with TestClient(app) as client:
        disconnect = client.post(
            "/disconnect",
            headers=_auth_headers(),
            json={"pair_name": "actor-rollout"},
        )
        reconnect = client.post(
            "/connect",
            headers=_auth_headers(),
            json={
                "pair_name": "actor-rollout",
                "train_worker_urls": ["http://train"],
                "inference_worker_urls": ["http://infer"],
                "mode": "awex",
                "colocate": True,
            },
        )

    assert disconnect.status_code == 200
    assert reconnect.status_code == 409
    assert "gateway lifetime" in reconnect.json()["error"]
