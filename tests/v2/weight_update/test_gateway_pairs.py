# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from areal.v2.weight_update.gateway.app import create_app
from areal.v2.weight_update.gateway.config import PairInfo, WeightUpdateConfig

ADMIN_HEADERS = {"Authorization": "Bearer test-key"}


@pytest.fixture()
def app():
    return create_app(WeightUpdateConfig(admin_api_key="test-key"))


@pytest.fixture()
def client(app):
    from starlette.testclient import TestClient

    return TestClient(app)


def _pair(name: str, **kw) -> PairInfo:
    return PairInfo(
        pair_name=name,
        train_worker_urls=[f"http://train-{name}:8000"],
        inference_worker_urls=[f"http://infer-{name}:8000"],
        train_world_size=kw.pop("train_world_size", 2),
        inference_world_size=kw.pop("inference_world_size", 1),
        **kw,
    )


class TestListPairs:
    def test_no_pairs_yields_an_empty_list(self, client):
        assert client.get("/pairs", headers=ADMIN_HEADERS).json() == {"pairs": []}

    def test_registered_pairs_are_reported_with_their_fields(self, app, client):
        app.state.registry.register(
            _pair("actor", master_addr="10.0.0.4", master_port=29500, last_version=17)
        )
        app.state.registry.register(_pair("critic", mode="disk", colocate=True))

        resp = client.get("/pairs", headers=ADMIN_HEADERS)
        assert resp.status_code == 200
        by_name = {p["pair_name"]: p for p in resp.json()["pairs"]}
        assert set(by_name) == {"actor", "critic"}

        actor = by_name["actor"]
        assert actor["last_version"] == 17
        assert actor["master_addr"] == "10.0.0.4"
        assert actor["master_port"] == 29500
        assert actor["train_world_size"] == 2
        assert actor["mode"] == "awex"
        assert by_name["critic"]["mode"] == "disk"
        assert by_name["critic"]["colocate"] is True

    def test_unregistered_pairs_disappear_from_the_listing(self, app, client):
        app.state.registry.register(_pair("actor"))
        app.state.registry.unregister("actor")
        assert client.get("/pairs", headers=ADMIN_HEADERS).json()["pairs"] == []

    def test_reading_pairs_does_not_mutate_the_registry(self, app, client):
        app.state.registry.register(_pair("actor"))
        for _ in range(3):
            client.get("/pairs", headers=ADMIN_HEADERS)
        assert app.state.registry.list_pairs() == ["actor"]


class TestAuth:
    def test_no_key_is_rejected(self, client):
        assert client.get("/pairs").status_code == 401

    def test_wrong_key_is_rejected(self, client):
        resp = client.get("/pairs", headers={"Authorization": "Bearer nope"})
        assert resp.status_code == 403

    def test_pair_contents_do_not_leak_to_an_unauthenticated_caller(self, app, client):
        app.state.registry.register(_pair("actor", master_addr="10.0.0.4"))
        assert "10.0.0.4" not in client.get("/pairs").text
