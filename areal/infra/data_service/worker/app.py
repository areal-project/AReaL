# SPDX-License-Identifier: Apache-2.0

"""DataWorker FastAPI app — serves dataset samples over HTTP."""

from __future__ import annotations

import asyncio
import pickle
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from torch.utils.data import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.dataset import _get_custom_dataset
from areal.infra.data_service.types import (
    FetchSamplesRequest,
    WorkerEpochResetRequest,
    WorkerLoadDatasetRequest,
    WorkerStateLoadRequest,
    WorkerStateSaveRequest,
    WorkerUnloadDatasetRequest,
)
from areal.infra.data_service.worker.config import DataWorkerConfig
from areal.infra.rpc.serialization import serialize_value
from areal.utils import logging, seeding
from areal.utils.dataloader import EvalDistributedSampler
from areal.utils.hf_utils import load_hf_processor_and_tokenizer

logger = logging.getLogger("DataWorker")


def _identity_collate(samples: list[Any]) -> list[Any]:
    return samples


@dataclass
class _DatasetState:
    dataset_id: str
    raw_dataset: Any
    dataloader: Any
    sampler: DistributedSampler | None
    epoch: int
    exhausted: bool
    unloading: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def create_worker_app(config: DataWorkerConfig) -> FastAPI:
    # --- Concurrency model ---
    # datasets_lock : guards dict mutations (add/remove entries) and the
    #                 _loading_ids reservation set.
    # state.lock    : guards per-dataset state operations (epoch reset,
    #                 state save/load).
    # Lock ordering : datasets_lock → state.lock (never reverse).
    # Seed is set once at startup in lifespan(); per-epoch determinism
    # is handled by DistributedSampler.set_epoch(), not by re-seeding.
    datasets: dict[str, _DatasetState] = {}
    _loading_ids: set[str] = set()
    datasets_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(app: Any):
        seeding.set_random_seed(config.seed, key=f"data_worker_{config.rank}")
        app.state.config = config
        app.state.datasets = datasets
        try:
            yield
        finally:
            datasets.clear()

    app = FastAPI(title="AReaL Data Worker", lifespan=lifespan)

    def _require_dataset(dataset_id: str) -> _DatasetState:
        state = datasets.get(dataset_id)
        if state is None:
            raise HTTPException(
                status_code=404, detail=f"Unknown dataset_id: {dataset_id}"
            )
        return state

    @asynccontextmanager
    async def _locked_active_state(dataset_id: str):
        state = _require_dataset(dataset_id)
        async with state.lock:
            if state.unloading:
                raise HTTPException(
                    status_code=409,
                    detail=f"Dataset {dataset_id} is unloading",
                )
            yield state

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "rank": config.rank,
            "datasets": len(datasets),
        }

    @app.post("/datasets/load")
    async def load_dataset(body: WorkerLoadDatasetRequest):
        def _load_sync():
            _tokenizer = None
            _processor = None
            if body.tokenizer_or_processor_path:
                _processor, _tokenizer = load_hf_processor_and_tokenizer(
                    body.tokenizer_or_processor_path
                )

            _dataset = _get_custom_dataset(
                path=body.dataset_path,
                type=body.dataset_type,
                split=body.split,
                max_length=body.max_length,
                tokenizer=_tokenizer,
                processor=_processor,
                **body.dataset_kwargs,
            )

            _sampler_cls = (
                DistributedSampler if body.drop_last else EvalDistributedSampler
            )
            _sampler = _sampler_cls(
                _dataset,
                num_replicas=config.world_size,
                rank=config.rank,
                shuffle=body.shuffle,
                drop_last=body.drop_last,
            )

            _dataloader = StatefulDataLoader(
                _dataset,
                batch_size=1,
                num_workers=config.dataloader_num_workers,
                sampler=_sampler,
                drop_last=False,
                collate_fn=_identity_collate,
            )
            return _dataset, _sampler, _dataloader

        # Phase 1: Reserve the dataset ID under lock (fast).
        async with datasets_lock:
            if body.dataset_id in datasets:
                raise HTTPException(
                    status_code=409,
                    detail=f"Dataset {body.dataset_id} is already loaded",
                )
            if body.dataset_id in _loading_ids:
                raise HTTPException(
                    status_code=409,
                    detail=f"Dataset {body.dataset_id} is currently loading",
                )
            _loading_ids.add(body.dataset_id)

        # Phase 2: Load dataset outside lock (slow I/O).
        try:
            dataset, sampler, dataloader = await asyncio.to_thread(_load_sync)
        except Exception:
            async with datasets_lock:
                _loading_ids.discard(body.dataset_id)
            raise

        # Phase 3: Store the result under lock (fast).
        async with datasets_lock:
            _loading_ids.discard(body.dataset_id)
            datasets[body.dataset_id] = _DatasetState(
                dataset_id=body.dataset_id,
                raw_dataset=dataset,
                dataloader=dataloader,
                sampler=sampler,
                epoch=0,
                exhausted=False,
            )

        return {
            "status": "ok",
            "dataset_size": sampler.num_samples,
            "steps_per_epoch": len(dataloader),
        }

    @app.post("/v1/samples/fetch")
    async def fetch_samples(body: FetchSamplesRequest):
        async with _locked_active_state(body.dataset_id) as state:
            samples = [serialize_value(state.raw_dataset[idx]) for idx in body.indices]
            return {"samples": samples}

    @app.post("/datasets/unload")
    async def unload_dataset(body: WorkerUnloadDatasetRequest):
        # Phase 1: look up state under datasets_lock (fast).
        async with datasets_lock:
            if body.dataset_id in _loading_ids:
                raise HTTPException(
                    status_code=409,
                    detail=f"Dataset {body.dataset_id} is currently loading",
                )
            state = datasets.get(body.dataset_id)
            if state is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Unknown dataset_id: {body.dataset_id}",
                )

        # Phase 2: drain in-flight state ops via state.lock (may wait).
        async with state.lock:
            if state.unloading:
                raise HTTPException(
                    status_code=409,
                    detail=f"Dataset {body.dataset_id} is already unloading",
                )
            state.unloading = True

        # Phase 3: remove from dict under datasets_lock (fast).
        async with datasets_lock:
            current = datasets.get(body.dataset_id)
            if current is state:
                del datasets[body.dataset_id]
        return {"status": "ok"}

    @app.post("/epoch/reset")
    async def reset_epoch(body: WorkerEpochResetRequest):
        async with _locked_active_state(body.dataset_id) as state:
            state.epoch = body.epoch
            state.exhausted = False
            if state.sampler is not None:
                state.sampler.set_epoch(body.epoch)
        return {"status": "ok", "epoch": body.epoch}

    @app.post("/state/save")
    async def save_state(body: WorkerStateSaveRequest):
        async with _locked_active_state(body.dataset_id) as state:
            save_dir = Path(body.path)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"worker_{config.rank}.pkl"

            with save_path.open("wb") as f:
                pickle.dump(state.dataloader.state_dict(), f)

        return {"status": "ok", "path": str(save_path)}

    @app.post("/state/load")
    async def load_state(body: WorkerStateLoadRequest):
        async with _locked_active_state(body.dataset_id) as state:
            load_path = Path(body.path) / f"worker_{config.rank}.pkl"
            if not load_path.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"State file not found: {load_path}",
                )

            with load_path.open("rb") as f:
                state_dict = pickle.load(f)
            state.dataloader.load_state_dict(state_dict)
            state.exhausted = False

        return {"status": "ok", "path": str(load_path)}

    @app.delete("/data/clear")
    async def clear_data():
        return {"status": "ok", "tensor_shards": 0}

    return app
