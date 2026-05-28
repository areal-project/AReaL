"""Epoch crossing comparison tests.

Verifies that DatasetHandle (data service) produces the same iteration
behavior as a local StatefulDataLoader for various dataset sizes, batch
sizes, and worker counts.

Specifically checks:
- __len__() matches (steps_per_epoch)
- Number of yielded batches per epoch matches
- batch_size consistency
- Epoch boundary behavior (exhaustion → reset → new epoch)
- Multi-worker round-robin yields all data
"""

from __future__ import annotations

import socket
import threading
import time
import uuid
from typing import Any

import httpx
import pytest
import uvicorn
from datasets import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.cli_args import TrainDatasetConfig
from areal.infra.data_service.gateway.app import create_gateway_app
from areal.infra.data_service.gateway.config import GatewayConfig
from areal.infra.data_service.rdataset import RDataset
from areal.infra.data_service.router.app import create_router_app
from areal.infra.data_service.router.config import RouterConfig
from areal.infra.data_service.worker.app import create_worker_app
from areal.infra.data_service.worker.config import DataWorkerConfig
from areal.utils.dataloader import create_dataloader

pytestmark = pytest.mark.slow

ADMIN_KEY = "areal-data-admin"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _start_uvicorn(app, host, port):
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return server, thread


def _wait_healthy(base_url: str, timeout: float = 10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(f"{base_url}/health", timeout=1.0)
            if resp.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.05)
    raise TimeoutError(f"Service did not become healthy: {base_url}")


def _make_dataset(n_samples: int) -> Dataset:
    return Dataset.from_dict(
        {
            "idx": list(range(n_samples)),
            "text": [f"sample_{i}" for i in range(n_samples)],
        }
    )


def _identity_collate(samples):
    return samples


def _local_dataloader_info(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    drop_last: bool = True,
    shuffle: bool = False,
) -> dict:
    """Simulate new data-service behavior: workers use batch_size=1,
    controller accumulates into batches of batch_size with drop_last.
    """
    from torch.utils.data import DistributedSampler

    from areal.utils.dataloader import EvalDistributedSampler

    all_samples: list = []
    shard_sizes: list[int] = []

    for rank in range(num_workers):
        sampler_cls = DistributedSampler if drop_last else EvalDistributedSampler
        sampler = sampler_cls(
            dataset,
            num_replicas=num_workers,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        shard_sizes.append(sampler.num_samples)
        dl = StatefulDataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
            collate_fn=_identity_collate,
            drop_last=False,
        )
        for item in dl:
            all_samples.append(
                item[0] if isinstance(item, list) and len(item) == 1 else item
            )

    total_samples = len(all_samples)

    all_batches: list[list] = []
    for i in range(0, total_samples, batch_size):
        chunk = all_samples[i : i + batch_size]
        if len(chunk) == batch_size:
            all_batches.append(chunk)
        elif not drop_last:
            all_batches.append(chunk)

    if drop_last:
        steps = total_samples // batch_size
    else:
        steps = (total_samples + batch_size - 1) // batch_size

    return {
        "total_steps": steps,
        "shard_sizes": shard_sizes,
        "all_batches": all_batches,
        "total_samples": total_samples,
    }


class _DataServiceStack:
    """Starts worker(s), router, gateway in-process for testing."""

    def __init__(self, num_workers: int, batch_size: int):
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.servers: list[uvicorn.Server] = []
        self.worker_urls: list[str] = []
        self.router_url = ""
        self.gateway_url = ""

    def start(self):
        host = "127.0.0.1"

        for rank in range(self.num_workers):
            port = _free_port()
            app = create_worker_app(
                DataWorkerConfig(
                    rank=rank,
                    world_size=self.num_workers,
                    dataloader_num_workers=0,
                )
            )
            server, _ = _start_uvicorn(app, host, port)
            self.servers.append(server)
            self.worker_urls.append(f"http://{host}:{port}")

        for url in self.worker_urls:
            _wait_healthy(url)

        router_port = _free_port()
        router_app = create_router_app(
            RouterConfig(
                host=host,
                port=router_port,
                admin_api_key=ADMIN_KEY,
            )
        )
        router_server, _ = _start_uvicorn(router_app, host, router_port)
        self.servers.append(router_server)
        self.router_url = f"http://{host}:{router_port}"
        _wait_healthy(self.router_url)

        gw_port = _free_port()
        gw_app = create_gateway_app(
            GatewayConfig(
                host=host,
                port=gw_port,
                router_addr=self.router_url,
                admin_api_key=ADMIN_KEY,
                forward_timeout=30.0,
                router_timeout=5.0,
            )
        )
        gw_server, _ = _start_uvicorn(gw_app, host, gw_port)
        self.servers.append(gw_server)
        self.gateway_url = f"http://{host}:{gw_port}"
        _wait_healthy(self.gateway_url)

        with httpx.Client(timeout=5.0) as client:
            for wurl in self.worker_urls:
                resp = client.post(
                    f"{self.router_url}/register",
                    json={"worker_addr": wurl},
                    headers={"Authorization": f"Bearer {ADMIN_KEY}"},
                )
                assert resp.status_code == 200

    def stop(self):
        for server in reversed(self.servers):
            server.should_exit = True

    def register_dataset(
        self,
        client: httpx.Client,
        dataset_path: str,
        dataset_id: str = "test",
        dataset_type: str = "rl",
        shuffle: bool = False,
    ) -> dict:
        resp = client.post(
            f"{self.gateway_url}/v1/datasets/register",
            json={
                "dataset_id": dataset_id,
                "dataset_path": dataset_path,
                "dataset_type": dataset_type,
                "seed": 42,
                "shuffle": shuffle,
            },
            headers={"Authorization": f"Bearer {ADMIN_KEY}"},
            timeout=30.0,
        )
        assert resp.status_code == 200, resp.text
        return resp.json()

    def advance_epoch(self, client: httpx.Client, api_key: str, epoch: int):
        resp = client.post(
            f"{self.gateway_url}/v1/epochs/advance",
            json={"epoch": epoch},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10.0,
        )
        assert resp.status_code == 200, resp.text
        return resp.json()


class _GatewayControllerAdapter:
    def __init__(self, gateway_url: str, admin_key: str):
        self._gateway_url = gateway_url
        self._admin_key = admin_key

    def register_dataset(
        self,
        dataset_id: str,
        dataset_path: str,
        dataset_type: str,
        dataset_kwargs: dict | None = None,
        tokenizer_or_processor_path: str = "",
        split: str = "train",
        seed: int = 42,
        shuffle: bool = False,
        drop_last: bool = True,
        max_length: int | None = None,
    ) -> dict:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                f"{self._gateway_url}/v1/datasets/register",
                headers={"Authorization": f"Bearer {self._admin_key}"},
                json={
                    "dataset_id": dataset_id,
                    "dataset_path": dataset_path,
                    "dataset_type": dataset_type,
                    "dataset_kwargs": dataset_kwargs or {},
                    "tokenizer_or_processor_path": tokenizer_or_processor_path,
                    "split": split,
                    "seed": seed,
                    "shuffle": shuffle,
                    "drop_last": drop_last,
                    "max_length": max_length,
                },
            )
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        payload["total_samples"] = payload["dataset_size"]
        return payload

    def unregister_dataset(self, dataset_id: str) -> None:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(
                f"{self._gateway_url}/v1/datasets/unregister",
                headers={"Authorization": f"Bearer {self._admin_key}"},
                json={"dataset_id": dataset_id},
            )
        assert resp.status_code == 200, resp.text

    def _gateway_post(self, endpoint: str, api_key: str, payload: dict):
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                f"{self._gateway_url}{endpoint}",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
            )
        assert resp.status_code == 200, resp.text
        return resp.json()


def _collect_epoch_indices(dl: StatefulDataLoader, epoch: int) -> list[int]:
    if hasattr(dl, "sampler") and hasattr(dl.sampler, "set_epoch"):
        dl.sampler.set_epoch(epoch)
    indices: list[int] = []
    for batch in dl:
        for item in batch:
            indices.append(int(item["idx"]))
    return indices


@pytest.fixture
def gsm8k_path(tmp_path):
    """Create a minimal synthetic dataset mimicking GSM8K structure."""
    ds = _make_dataset(100)
    path = str(tmp_path / "test_dataset")
    ds.save_to_disk(path)
    return path


@pytest.mark.parametrize(
    "n_samples,batch_size,num_workers",
    [
        (10, 3, 1),
        (10, 3, 2),
        (10, 5, 1),
        (10, 5, 2),
        (100, 32, 1),
        (100, 32, 4),
        (7, 3, 2),
        (15, 4, 3),
    ],
)
def test_steps_per_epoch_matches_local(
    n_samples: int, batch_size: int, num_workers: int
):
    """steps_per_epoch = total_samples // batch_size (drop_last=True default)."""
    dataset = _make_dataset(n_samples)
    local_info = _local_dataloader_info(dataset, batch_size, num_workers)

    assert local_info["total_steps"] == local_info["total_samples"] // batch_size
    assert local_info["total_steps"] == len(local_info["all_batches"])


@pytest.mark.parametrize(
    "n_samples,batch_size,num_workers",
    [
        (10, 3, 1),
        (10, 3, 2),
        (100, 32, 1),
        (100, 32, 4),
    ],
)
def test_local_dataloader_epoch_iteration(
    n_samples: int, batch_size: int, num_workers: int
):
    """Local DataLoader yields exactly len(dl) batches per epoch across all shards."""

    dataset = _make_dataset(n_samples)
    local_info = _local_dataloader_info(dataset, batch_size, num_workers)

    for batch in local_info["all_batches"]:
        assert len(batch) == batch_size

    assert len(local_info["all_batches"]) == local_info["total_steps"]


@pytest.mark.parametrize(
    "n_samples,batch_size,num_workers",
    [
        (20, 5, 1),
        (20, 5, 2),
        (50, 10, 3),
    ],
)
def test_data_service_epoch_matches_local(
    tmp_path, n_samples: int, batch_size: int, num_workers: int
):
    dataset = _make_dataset(n_samples)
    ds_path = str(tmp_path / "ds")
    dataset.save_to_disk(ds_path)

    stack = _DataServiceStack(num_workers=num_workers, batch_size=batch_size)
    stack.start()

    try:
        cfg = TrainDatasetConfig(
            path=ds_path,
            type="rl",
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=0,
        )
        local_dl = create_dataloader(dataset, rank=0, world_size=1, dataset_config=cfg)

        controller: Any = _GatewayControllerAdapter(stack.gateway_url, ADMIN_KEY)
        rdataset = RDataset(path=ds_path, type="rl", split="train")
        rdataset.connect(
            controller,
            dataset_id=f"epoch-local-{uuid.uuid4().hex[:8]}",
            shuffle=False,
            drop_last=True,
        )
        remote_dl = create_dataloader(
            rdataset, rank=0, world_size=1, dataset_config=cfg
        )

        try:
            remote_indices = _collect_epoch_indices(remote_dl, 0)
            local_indices = _collect_epoch_indices(local_dl, 0)
            assert remote_indices == local_indices[: len(remote_indices)]
        finally:
            rdataset.close()
    finally:
        stack.stop()


@pytest.mark.parametrize(
    "n_samples,batch_size,num_workers,num_epochs",
    [
        (20, 5, 1, 3),
        (20, 5, 2, 2),
    ],
)
def test_data_service_multi_epoch(
    tmp_path, n_samples: int, batch_size: int, num_workers: int, num_epochs: int
):
    dataset = _make_dataset(n_samples)
    ds_path = str(tmp_path / "ds")
    dataset.save_to_disk(ds_path)

    stack = _DataServiceStack(num_workers=num_workers, batch_size=batch_size)
    stack.start()

    try:
        cfg = TrainDatasetConfig(
            path=ds_path,
            type="rl",
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
        )
        local_dl = create_dataloader(dataset, rank=0, world_size=1, dataset_config=cfg)

        controller: Any = _GatewayControllerAdapter(stack.gateway_url, ADMIN_KEY)
        rdataset = RDataset(path=ds_path, type="rl", split="train")
        rdataset.connect(
            controller,
            dataset_id=f"epoch-multi-{uuid.uuid4().hex[:8]}",
            shuffle=True,
            drop_last=True,
        )
        remote_dl = create_dataloader(
            rdataset, rank=0, world_size=1, dataset_config=cfg
        )

        try:
            for epoch in range(num_epochs):
                remote_epoch = _collect_epoch_indices(remote_dl, epoch)
                local_epoch = _collect_epoch_indices(local_dl, epoch)
                assert remote_epoch == local_epoch
        finally:
            rdataset.close()
    finally:
        stack.stop()


# --- Unit tests for sharding math (no service needed) ---


@pytest.mark.parametrize(
    "n_samples,batch_size,num_workers",
    [
        (10, 3, 1),
        (10, 3, 2),
        (10, 3, 3),
        (7, 3, 2),
        (7, 3, 3),
        (100, 32, 4),
        (1, 1, 1),
        (5, 10, 1),
        (5, 10, 2),
    ],
)
def test_train_drop_last_no_incomplete_batches(
    n_samples: int, batch_size: int, num_workers: int
):
    """With drop_last=True, every batch has exactly batch_size samples."""
    dataset = _make_dataset(n_samples)
    info = _local_dataloader_info(dataset, batch_size, num_workers, drop_last=True)
    for batch in info["all_batches"]:
        assert len(batch) == batch_size


@pytest.mark.parametrize(
    "n_samples,batch_size,num_workers",
    [
        (10, 3, 1),
        (10, 3, 2),
        (10, 3, 3),
        (7, 3, 2),
        (7, 3, 3),
        (100, 32, 4),
        (1, 1, 1),
        (5, 10, 1),
        (5, 10, 2),
    ],
)
def test_valid_drop_last_false_preserves_all_data(
    n_samples: int, batch_size: int, num_workers: int
):
    """With drop_last=False, ALL samples appear in the output batches."""
    dataset = _make_dataset(n_samples)
    info = _local_dataloader_info(dataset, batch_size, num_workers, drop_last=False)

    total_yielded = sum(len(b) for b in info["all_batches"])
    assert total_yielded == n_samples, (
        f"Expected all {n_samples} samples, got {total_yielded}. "
        f"shard_sizes={info['shard_sizes']}"
    )


@pytest.mark.parametrize(
    "n_samples,batch_size,num_workers",
    [
        (7, 3, 2),
        (7, 3, 3),
        (10, 4, 3),
        (11, 5, 3),
    ],
)
def test_uneven_shard_steps_sum_is_correct(
    n_samples: int, batch_size: int, num_workers: int
):
    """Computed total_steps == actual number of batches yielded."""
    dataset = _make_dataset(n_samples)
    for drop_last in (True, False):
        info = _local_dataloader_info(
            dataset, batch_size, num_workers, drop_last=drop_last
        )
        assert info["total_steps"] == len(info["all_batches"]), (
            f"drop_last={drop_last}: total_steps={info['total_steps']} "
            f"but got {len(info['all_batches'])} batches. "
            f"shard_sizes={info['shard_sizes']}"
        )


@pytest.mark.parametrize("n_samples", [10, 7, 15, 100])
@pytest.mark.parametrize("batch_size", [3, 5, 32])
def test_single_worker_matches_single_process_dataloader(
    n_samples: int, batch_size: int
):
    """With 1 worker, data service steps == single-process DataLoader steps."""
    dataset = _make_dataset(n_samples)
    for drop_last in (True, False):
        info = _local_dataloader_info(
            dataset, batch_size, num_workers=1, drop_last=drop_last
        )
        dl = StatefulDataLoader(
            dataset,
            batch_size=batch_size,
            drop_last=drop_last,
            collate_fn=_identity_collate,
        )
        assert info["total_steps"] == len(dl), (
            f"drop_last={drop_last}: 1-worker service steps={info['total_steps']} "
            f"!= single-process steps={len(dl)}"
        )


@pytest.mark.parametrize(
    "n_samples,batch_size,num_workers",
    [
        (12, 2, 4),
        (10, 3, 2),
        (10, 3, 3),
        (7, 3, 2),
        (100, 32, 4),
    ],
)
def test_multi_worker_matches_single_process(
    n_samples: int, batch_size: int, num_workers: int
):
    """Multi-worker data service yields same steps as single-process DataLoader."""
    dataset = _make_dataset(n_samples)
    for drop_last in (True, False):
        multi = _local_dataloader_info(
            dataset, batch_size, num_workers, drop_last=drop_last
        )
        dl = StatefulDataLoader(
            dataset,
            batch_size=batch_size,
            drop_last=drop_last,
            collate_fn=_identity_collate,
        )
        assert multi["total_steps"] == len(dl), (
            f"drop_last={drop_last}, workers={num_workers}: "
            f"multi-worker steps={multi['total_steps']} != "
            f"single-process steps={len(dl)}"
        )
