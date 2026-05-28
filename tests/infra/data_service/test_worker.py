from __future__ import annotations

# pyright: reportMissingImports=false
import asyncio
import inspect
import threading
from collections.abc import Mapping
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
import pytest_asyncio
from datasets import Dataset

from areal.infra.data_service.worker.app import create_worker_app
from areal.infra.data_service.worker.config import DataWorkerConfig

DATASET_ID = "test-train"


@pytest.fixture
def config() -> DataWorkerConfig:
    return DataWorkerConfig(
        host="127.0.0.1",
        port=0,
        rank=0,
        world_size=1,
        dataloader_num_workers=0,
    )


def _make_mock_dataset(n: int = 20) -> Dataset:
    return Dataset.from_dict(
        {
            "text": [f"sample_{i}" for i in range(n)],
            "label": list(range(n)),
        }
    )


def _load_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "dataset_id": DATASET_ID,
        "dataset_path": "test/dataset",
        "dataset_type": "rl",
        "shuffle": False,
    }
    payload.update(overrides)
    return payload


def _get_route_endpoint(app, path: str, method: str = "POST"):
    for route in app.routes:
        if getattr(route, "path", None) == path and method in getattr(
            route, "methods", set()
        ):
            return route.endpoint
    raise AssertionError(f"Route not found: {method} {path}")


def _get_worker_nonlocals(app) -> Mapping[str, object]:
    endpoint = _get_route_endpoint(app, "/datasets/unload")
    return inspect.getclosurevars(endpoint).nonlocals


@pytest_asyncio.fixture
async def client(config: DataWorkerConfig):
    app = create_worker_app(config)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest_asyncio.fixture
async def loaded_client(config: DataWorkerConfig):
    with (
        patch("areal.infra.data_service.worker.app._get_custom_dataset") as mock_get,
        patch(
            "areal.infra.data_service.worker.app.load_hf_processor_and_tokenizer"
        ) as mock_load,
    ):
        ds = _make_mock_dataset(8)
        mock_get.return_value = ds
        mock_load.return_value = (None, None)

        app = create_worker_app(config)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post("/datasets/load", json=_load_payload())
            assert resp.status_code == 200
            yield c


@pytest_asyncio.fixture
async def loaded_worker(config: DataWorkerConfig):
    with (
        patch("areal.infra.data_service.worker.app._get_custom_dataset") as mock_get,
        patch(
            "areal.infra.data_service.worker.app.load_hf_processor_and_tokenizer"
        ) as mock_load,
    ):
        ds = _make_mock_dataset(8)
        mock_get.return_value = ds
        mock_load.return_value = (None, None)

        app = create_worker_app(config)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post("/datasets/load", json=_load_payload())
            assert resp.status_code == 200
            nonlocals = _get_worker_nonlocals(app)
            yield {
                "app": app,
                "client": c,
                "datasets": nonlocals["datasets"],
                "datasets_lock": nonlocals["datasets_lock"],
            }


@pytest.mark.asyncio
class TestWorkerHealth:
    async def test_health_returns_200(self, client: httpx.AsyncClient):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["datasets"] == 0

    async def test_health_shows_dataset_count(self, loaded_client: httpx.AsyncClient):
        resp = await loaded_client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["datasets"] == 1


@pytest.mark.asyncio
class TestDatasetLoading:
    async def test_load_dataset_returns_steps_per_epoch(self, config: DataWorkerConfig):
        with (
            patch(
                "areal.infra.data_service.worker.app._get_custom_dataset"
            ) as mock_get,
        ):
            ds = _make_mock_dataset(20)
            mock_get.return_value = ds

            app = create_worker_app(config)
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as c:
                resp = await c.post("/datasets/load", json=_load_payload())

        assert resp.status_code == 200
        data = resp.json()
        assert data["steps_per_epoch"] > 0
        assert data["dataset_size"] == 20

    async def test_load_dataset_duplicate_409(self, loaded_client: httpx.AsyncClient):
        resp = await loaded_client.post("/datasets/load", json=_load_payload())
        assert resp.status_code == 409

    async def test_unload_dataset_removes(self, loaded_client: httpx.AsyncClient):
        resp = await loaded_client.post(
            "/datasets/unload", json={"dataset_id": DATASET_ID}
        )
        assert resp.status_code == 200

        health = await loaded_client.get("/health")
        assert health.status_code == 200
        assert health.json()["datasets"] == 0

    async def test_unload_unknown_dataset_404(self, client: httpx.AsyncClient):
        resp = await client.post("/datasets/unload", json={"dataset_id": "unknown"})
        assert resp.status_code == 404


@pytest.mark.asyncio
class TestSampleFetch:
    async def test_fetch_samples_returns_data(self, loaded_client: httpx.AsyncClient):
        resp = await loaded_client.post(
            "/v1/samples/fetch",
            json={"dataset_id": DATASET_ID, "indices": [0, 1]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["samples"]) == 2

    async def test_fetch_samples_unknown_dataset_404(self, client: httpx.AsyncClient):
        resp = await client.post(
            "/v1/samples/fetch",
            json={"dataset_id": "unknown", "indices": [0]},
        )
        assert resp.status_code == 404

    async def test_fetch_samples_returns_distinct_items(
        self, loaded_client: httpx.AsyncClient
    ):
        resp = await loaded_client.post(
            "/v1/samples/fetch",
            json={"dataset_id": DATASET_ID, "indices": [0, 1, 2]},
        )
        assert resp.status_code == 200
        samples = resp.json()["samples"]
        assert len(samples) == 3
        assert samples[0] != samples[1]


@pytest.mark.asyncio
class TestWorkerConcurrency:
    async def test_load_dataset_returns_409_when_same_id_currently_loading(
        self, config: DataWorkerConfig
    ):
        load_started = threading.Event()
        release_load = threading.Event()

        def slow_get_dataset(*args, **kwargs):
            del args, kwargs
            load_started.set()
            assert release_load.wait(timeout=2)
            return _make_mock_dataset(8)

        with (
            patch(
                "areal.infra.data_service.worker.app._get_custom_dataset",
                side_effect=slow_get_dataset,
            ),
            patch(
                "areal.infra.data_service.worker.app.load_hf_processor_and_tokenizer"
            ) as mock_load,
        ):
            mock_load.return_value = (None, None)
            app = create_worker_app(config)
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                first_load = asyncio.create_task(
                    client.post("/datasets/load", json=_load_payload())
                )
                await asyncio.to_thread(load_started.wait, 1)

                second_load = await client.post("/datasets/load", json=_load_payload())
                assert second_load.status_code == 409
                assert "currently loading" in second_load.json()["detail"].lower()

                release_load.set()
                first_response = await first_load
                assert first_response.status_code == 200

    async def test_unload_waits_for_in_flight_state_lock(self, loaded_worker):
        client = loaded_worker["client"]
        state = loaded_worker["datasets"][DATASET_ID]

        await state.lock.acquire()
        unload_task = asyncio.create_task(
            client.post("/datasets/unload", json={"dataset_id": DATASET_ID})
        )
        await asyncio.sleep(0)
        assert not unload_task.done()

        state.lock.release()
        unload_response = await asyncio.wait_for(unload_task, timeout=1)
        assert unload_response.status_code == 200

    async def test_fetch_returns_409_after_unload_starts(self, loaded_worker):
        client = loaded_worker["client"]
        state = loaded_worker["datasets"][DATASET_ID]
        datasets_lock = loaded_worker["datasets_lock"]

        await state.lock.acquire()
        unload_task = asyncio.create_task(
            client.post("/datasets/unload", json={"dataset_id": DATASET_ID})
        )
        await asyncio.sleep(0)

        await datasets_lock.acquire()
        state.lock.release()
        await asyncio.sleep(0)

        fetch_response = await client.post(
            "/v1/samples/fetch",
            json={"dataset_id": DATASET_ID, "indices": [0]},
        )
        assert fetch_response.status_code == 409
        assert "unloading" in fetch_response.json()["detail"].lower()

        datasets_lock.release()
        unload_response = await asyncio.wait_for(unload_task, timeout=1)
        assert unload_response.status_code == 200

    async def test_unrelated_load_succeeds_while_unload_waits_on_state_lock(
        self, loaded_worker
    ):
        client = loaded_worker["client"]
        state = loaded_worker["datasets"][DATASET_ID]

        await state.lock.acquire()
        unload_task = asyncio.create_task(
            client.post("/datasets/unload", json={"dataset_id": DATASET_ID})
        )
        await asyncio.sleep(0)

        other_load = await asyncio.wait_for(
            client.post(
                "/datasets/load",
                json=_load_payload(
                    dataset_id="test-valid",
                    dataset_path="test/other-dataset",
                ),
            ),
            timeout=1,
        )
        assert other_load.status_code == 200
        assert not unload_task.done()

        state.lock.release()
        unload_response = await asyncio.wait_for(unload_task, timeout=1)
        assert unload_response.status_code == 200

        health = await client.get("/health")
        assert health.status_code == 200
        assert health.json()["datasets"] == 1


@pytest.mark.asyncio
class TestEpochReset:
    async def test_epoch_reset_updates_epoch(self, loaded_client: httpx.AsyncClient):
        reset = await loaded_client.post(
            "/epoch/reset", json={"dataset_id": DATASET_ID, "epoch": 1}
        )
        assert reset.status_code == 200
        assert reset.json()["epoch"] == 1

    async def test_epoch_reset_unknown_dataset_404(self, client: httpx.AsyncClient):
        resp = await client.post(
            "/epoch/reset", json={"dataset_id": "unknown", "epoch": 1}
        )
        assert resp.status_code == 404


@pytest.mark.asyncio
class TestStatePersistence:
    async def test_state_save_creates_file(
        self, loaded_client: httpx.AsyncClient, tmp_path: Path
    ):
        resp = await loaded_client.post(
            "/state/save", json={"dataset_id": DATASET_ID, "path": str(tmp_path)}
        )
        assert resp.status_code == 200
        out = resp.json()
        assert out["status"] == "ok"
        assert (tmp_path / "worker_0.pkl").exists()

    async def test_state_load_restores(
        self, loaded_client: httpx.AsyncClient, tmp_path: Path
    ):
        save = await loaded_client.post(
            "/state/save", json={"dataset_id": DATASET_ID, "path": str(tmp_path)}
        )
        assert save.status_code == 200

        load = await loaded_client.post(
            "/state/load", json={"dataset_id": DATASET_ID, "path": str(tmp_path)}
        )
        assert load.status_code == 200
        assert load.json()["status"] == "ok"

    async def test_state_load_missing_file_404(
        self, loaded_client: httpx.AsyncClient, tmp_path: Path
    ):
        missing = tmp_path / "does-not-exist"
        resp = await loaded_client.post(
            "/state/load", json={"dataset_id": DATASET_ID, "path": str(missing)}
        )
        assert resp.status_code == 404


@pytest.mark.asyncio
class TestTensorShardEndpoints:
    async def test_data_clear_returns_ok(self, client: httpx.AsyncClient):
        resp = await client.delete("/data/clear")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["tensor_shards"] == 0

    async def test_data_shard_not_found_404(self, client: httpx.AsyncClient):
        resp = await client.get("/data/nonexistent")
        assert resp.status_code == 404
