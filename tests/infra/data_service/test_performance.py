from __future__ import annotations

import time
from unittest.mock import patch

import httpx
import pytest

from areal.infra.data_service.worker.app import create_worker_app
from areal.infra.data_service.worker.config import DataWorkerConfig

DATASET_ID = "perf-test"
WORKER_CONFIG = DataWorkerConfig(
    host="127.0.0.1",
    port=0,
    rank=0,
    world_size=1,
    dataloader_num_workers=0,
)


def _make_mock_dataset(n: int):
    from datasets import Dataset

    return Dataset.from_dict(
        {"text": [f"sample_{i}" for i in range(n)], "label": list(range(n))}
    )


def _load_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "dataset_id": DATASET_ID,
        "dataset_path": "test/dataset",
        "dataset_type": "rl",
        "seed": 42,
        "shuffle": False,
    }
    payload.update(overrides)
    return payload


@pytest.mark.asyncio
class TestWorkerSampleFetchPerformance:
    async def test_sample_fetch_throughput(self):
        n = 100
        with (
            patch(
                "areal.infra.data_service.worker.app._get_custom_dataset"
            ) as mock_get,
            patch(
                "areal.infra.data_service.worker.app.load_hf_processor_and_tokenizer"
            ) as mock_load,
        ):
            mock_get.return_value = _make_mock_dataset(n)
            mock_load.return_value = (None, None)

            app = create_worker_app(WORKER_CONFIG)
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.post("/datasets/load", json=_load_payload())
                assert resp.status_code == 200

                batch_size = 10
                t0 = time.perf_counter()
                for start in range(0, n, batch_size):
                    indices = list(range(start, min(start + batch_size, n)))
                    resp = await client.post(
                        "/v1/samples/fetch",
                        json={"dataset_id": DATASET_ID, "indices": indices},
                    )
                    assert resp.status_code == 200
                    assert len(resp.json()["samples"]) == len(indices)
                elapsed = time.perf_counter() - t0

        per_item_ms = (elapsed / n) * 1000
        assert per_item_ms < 50, (
            f"Worker sample fetch: {per_item_ms:.1f}ms per item "
            f"(expected < 50ms for in-memory mock data via ASGI)"
        )

    async def test_single_large_sample_fetch(self):
        n = 50
        with (
            patch(
                "areal.infra.data_service.worker.app._get_custom_dataset"
            ) as mock_get,
            patch(
                "areal.infra.data_service.worker.app.load_hf_processor_and_tokenizer"
            ) as mock_load,
        ):
            mock_get.return_value = _make_mock_dataset(n)
            mock_load.return_value = (None, None)

            app = create_worker_app(WORKER_CONFIG)
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.post("/datasets/load", json=_load_payload())
                assert resp.status_code == 200

                indices = list(range(n))
                t0 = time.perf_counter()
                resp = await client.post(
                    "/v1/samples/fetch",
                    json={"dataset_id": DATASET_ID, "indices": indices},
                )
                elapsed = time.perf_counter() - t0

                assert resp.status_code == 200
                assert len(resp.json()["samples"]) == n

        assert elapsed < 5.0, (
            f"Single sample fetch of {n} items took {elapsed:.2f}s (expected < 5s)"
        )
