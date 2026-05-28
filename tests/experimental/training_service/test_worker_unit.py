"""Unit tests for training-service worker Flask app."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from areal.experimental.training_service.worker.config import TrainWorkerConfig
from areal.infra.rpc.serialization import deserialize_value, serialize_value

MODULE = "areal.experimental.training_service.worker.app"


@pytest.fixture(autouse=True)
def reset_worker_state():
    import areal.experimental.training_service.worker.app as worker_app

    if worker_app._engine_work_queue is not None:
        worker_app._engine_work_queue.put(None)
    if worker_app._engine_thread is not None:
        worker_app._engine_thread.join(timeout=1.0)

    worker_app._engine = None
    worker_app._node_addr = ""
    worker_app._engine_thread = None
    worker_app._engine_work_queue = None

    yield

    if worker_app._engine_work_queue is not None:
        worker_app._engine_work_queue.put(None)
    if worker_app._engine_thread is not None:
        worker_app._engine_thread.join(timeout=1.0)

    worker_app._engine = None
    worker_app._node_addr = ""
    worker_app._engine_thread = None
    worker_app._engine_work_queue = None


@pytest.fixture
def client():
    from areal.experimental.training_service.worker.app import create_app

    app = create_app(
        TrainWorkerConfig(
            host="127.0.0.1",
            port=19001,
            admin_api_key="worker-admin",
        )
    )
    return app.test_client()


class TestWorkerEngineCreation:
    def test_create_engine_requires_engine_class(self, client):
        resp = client.post(
            "/create_engine",
            json={"init_args": [], "init_kwargs": {}},
        )
        assert resp.status_code == 400
        assert "engine_class" in resp.get_json()["error"]

    def test_create_engine_success(self, client):
        resp = client.post(
            "/create_engine",
            json={
                "engine_class": "tests.experimental.training_service.fake_train_engine.FakeTrainEngine",
                "init_args": serialize_value([]),
                "init_kwargs": serialize_value({"world_size": 1}),
            },
        )
        assert resp.status_code == 200
        payload = resp.get_json()
        assert payload["status"] == "success"


class TestWorkerEndpoints:
    def test_topology_before_create_engine_returns_400(self, client):
        resp = client.get("/topology")
        assert resp.status_code == 400
        assert "Engine not created" in resp.get_json()["error"]

    def test_train_batch_after_create_engine(self, client):
        create_resp = client.post(
            "/create_engine",
            json={
                "engine_class": "tests.experimental.training_service.fake_train_engine.FakeTrainEngine",
                "init_args": serialize_value([]),
                "init_kwargs": serialize_value({"world_size": 1}),
            },
        )
        assert create_resp.status_code == 200

        train_resp = client.post(
            "/train_batch",
            json={
                "args": serialize_value(
                    [{"token_ids": [1, 2, 3], "metadata": {"weight": 2.0}}]
                ),
                "kwargs": serialize_value({}),
            },
        )
        assert train_resp.status_code == 200
        result = deserialize_value(train_resp.get_json()["result"])
        assert isinstance(result, dict)
        assert "total" in result

    def test_topology_after_create_engine(self, client):
        create_resp = client.post(
            "/create_engine",
            json={
                "engine_class": "tests.experimental.training_service.fake_train_engine.FakeTrainEngine",
                "init_args": serialize_value([]),
                "init_kwargs": serialize_value({"world_size": 1}),
            },
        )
        assert create_resp.status_code == 200

        with patch.dict(
            "os.environ",
            {"RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0"},
            clear=False,
        ):
            topo_resp = client.get("/topology")
        assert topo_resp.status_code == 200
        topo = topo_resp.get_json()
        assert topo["rank"] == 0
        assert topo["world_size"] == 1
        assert topo["dp_size"] == 1

    def test_ppo_endpoints_return_400_when_engine_method_missing(self, client):
        create_resp = client.post(
            "/create_engine",
            json={
                "engine_class": "tests.experimental.training_service.fake_train_engine.FakeTrainEngine",
                "init_args": serialize_value([]),
                "init_kwargs": serialize_value({"world_size": 1}),
            },
        )
        assert create_resp.status_code == 200

        payload = {
            "args": serialize_value([[{"token_ids": [1, 2, 3]}]]),
            "kwargs": serialize_value({}),
        }
        for path in [
            "/ppo/actor/compute_logp",
            "/ppo/actor/compute_advantages",
            "/ppo/actor/update",
            "/ppo/critic/compute_values",
            "/ppo/critic/update",
        ]:
            resp = client.post(path, json=payload)
            assert resp.status_code == 400
            assert "does not implement method" in resp.get_json()["error"]

    def test_forward_batch_after_initialize_succeeds_without_distributed_group(
        self, client
    ):
        create_resp = client.post(
            "/create_engine",
            json={
                "engine_class": "tests.experimental.training_service.fake_train_engine.FakeTrainEngine",
                "init_args": serialize_value([]),
                "init_kwargs": serialize_value({"world_size": 1}),
            },
        )
        assert create_resp.status_code == 200

        init_resp = client.post(
            "/initialize",
            json={
                "args": serialize_value([]),
                "kwargs": serialize_value({"addr": None, "ft_spec": None}),
            },
        )
        assert init_resp.status_code == 200

        forward_resp = client.post(
            "/forward_batch",
            json={
                "args": serialize_value(
                    [[{"token_ids": [1, 2, 3], "metadata": {"weight": 2.0}}]]
                ),
                "kwargs": serialize_value({"output_seqlens": [3]}),
            },
        )
        assert forward_resp.status_code == 200
        result = deserialize_value(forward_resp.get_json()["result"])
        assert isinstance(result, dict)
        assert result["output_seqlens"] == [3]

    def test_sft_route_succeeds_without_distributed_group_for_single_worker(
        self, client
    ):
        create_resp = client.post(
            "/create_engine",
            json={
                "engine_class": "tests.experimental.training_service.fake_train_engine.FakeTrainEngine",
                "init_args": serialize_value([]),
                "init_kwargs": serialize_value({"world_size": 1}),
            },
        )
        assert create_resp.status_code == 200

        resp = client.post(
            "/sft/train",
            json={
                "args": serialize_value([[{"token_ids": [1, 2, 3]}]]),
                "kwargs": serialize_value({}),
            },
        )
        assert resp.status_code == 200
        result = deserialize_value(resp.get_json()["result"])
        assert isinstance(result, dict)
        assert "total" in result

    def test_sft_route_ignores_rpc_meta_override_for_single_worker(self, client):
        create_resp = client.post(
            "/create_engine",
            json={
                "engine_class": "tests.experimental.training_service.fake_train_engine.FakeTrainEngine",
                "init_args": serialize_value([]),
                "init_kwargs": serialize_value({"world_size": 1}),
            },
        )
        assert create_resp.status_code == 200

        resp = client.post(
            "/sft/train",
            json={
                "args": serialize_value([[{"token_ids": [1, 2, 3]}]]),
                "kwargs": serialize_value({}),
                "rpc_meta": {"broadcast": False},
            },
        )
        assert resp.status_code == 200
        result = deserialize_value(resp.get_json()["result"])
        assert isinstance(result, dict)
        assert "total" in result
