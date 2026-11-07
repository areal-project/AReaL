# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import os
import socket
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from functools import wraps
from typing import Any

import aiohttp  # pyright: ignore[reportMissingImports]
from fastapi import FastAPI, Request  # pyright: ignore[reportMissingImports]
from fastapi.responses import JSONResponse  # pyright: ignore[reportMissingImports]
from pydantic import BaseModel  # pyright: ignore[reportMissingImports]

from areal.infra.utils.http import async_http_retry
from areal.utils import logging
from areal.utils.network import find_free_ports
from areal.v2.weight_update.gateway.auth import require_admin_key
from areal.v2.weight_update.gateway.config import (
    PairInfo,
    WeightUpdateConfig,
    WeightUpdateResult,
)
from areal.v2.weight_update.gateway.kv_store import WeightMetaStore
from areal.v2.weight_update.gateway.metadata import (
    inference_metadata_template,
    merge_parameter_metadata,
)
from areal.v2.weight_update.gateway.pair_registry import PairRegistry

logger = logging.getLogger("WeightUpdateGateway")


class ConnectRequest(BaseModel):
    pair_name: str
    train_worker_urls: list[str]
    inference_worker_urls: list[str]
    nccl_master_addr: str = ""
    nccl_master_port: int = 0
    mode: str = "awex"  # "awex" or "disk"
    save_path: str = ""
    use_lora: bool = False
    lora_name: str = ""
    lora_keep_versions: int = 0
    colocate: bool = False


class UpdateWeightsRequest(BaseModel):
    pair_name: str
    version: int = 0


class DisconnectRequest(BaseModel):
    pair_name: str
    # Recovery-only assertion by the authenticated local lifecycle owner, after
    # all training descendants have exited and their GPU pool is empty.
    training_stopped: bool = False


class KVPutBody(BaseModel):
    value: Any = None


class KVSetAddBody(BaseModel):
    value: str = ""


class HealthResponse(BaseModel):
    status: str = "healthy"


class ConnectResponse(BaseModel):
    pair_name: str


class DisconnectResponse(BaseModel):
    status: str = "ok"
    pair_name: str


class StatusResponse(BaseModel):
    status: str = "ok"


class KVDeleteResponse(BaseModel):
    status: str = "ok"
    deleted: bool


class KVGetResponse(BaseModel):
    value: Any = None


class KVSetSizeResponse(BaseModel):
    size: int


@async_http_retry
async def _get_json(session: aiohttp.ClientSession, url: str, timeout_s: float) -> Any:
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with session.get(url, timeout=timeout) as resp:
        resp.raise_for_status()
        return await resp.json()


async def _post_json(
    session: aiohttp.ClientSession,
    url: str,
    timeout_s: float,
    json_data: Any = None,
) -> Any:
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with session.post(url, json=json_data, timeout=timeout) as resp:
        resp.raise_for_status()
        return await resp.json()


async def _post(
    session: aiohttp.ClientSession,
    url: str,
    timeout_s: float,
    json_data: Any = None,
) -> None:
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with session.post(url, json=json_data, timeout=timeout) as resp:
        resp.raise_for_status()


async def _gather_all(*operations):
    """Observe every sent request before starting rollback or another collective."""
    results = await asyncio.gather(*operations, return_exceptions=True)
    for result in results:
        if isinstance(result, BaseException):
            raise result
    return results


async def _discover_parallelism(session, train_urls, inference_urls, timeout_s):
    reports = await _gather_all(
        *[
            _get_json(session, f"{url}/awex/report_parallelism", timeout_s)
            for url in train_urls + inference_urls
        ]
    )
    train_reports, infer_reports = (
        reports[: len(train_urls)],
        reports[len(train_urls) :],
    )
    keys = ("world_size", "tp_size", "pp_size", "dp_size", "ep_size", "cp_size")
    for group in (train_reports, infer_reports):
        reference = tuple(group[0].get(key, 1) for key in keys)
        if any(
            tuple(report.get(key, 1) for key in keys) != reference for report in group
        ):
            raise ValueError("Workers disagree on their parallel layout")
    if train_reports[0]["world_size"] != len(train_urls):
        raise ValueError("Training worker list does not cover the complete group")
    return train_reports[0], infer_reports[0]


def _get_own_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def _merge_training_meta_by_name(meta_list: list[dict]) -> list[dict]:
    """Normalize both partitioned FSDP tensors and replicated Megatron tensors."""
    return merge_parameter_metadata(meta_list)


def _validate_wire_dtypes(
    training_meta: list[dict], inference_meta: list[dict]
) -> None:
    """Reject incompatible P2P buffers before any rank initializes a collective."""

    def data(value: dict) -> dict:
        return value.get("data", value)

    def dtype(value: dict):
        result = value.get("dtype")
        return result.get("value") if isinstance(result, dict) else result

    by_name: dict[str, str] = {}
    for role, metadata in (("training", training_meta), ("inference", inference_meta)):
        for parameter in metadata:
            parameter = data(parameter)
            name, expected = parameter["name"], dtype(parameter)
            if expected is None:
                raise ValueError(f"Missing AWEX wire dtype for {role} parameter {name}")
            if name in by_name and by_name[name] != expected:
                raise ValueError(
                    f"AWEX wire dtype mismatch for {name}: {by_name[name]} != {expected}"
                )
            by_name[name] = expected
            shards = list(parameter.get("shards", []))
            for replica in parameter.get("replicas", []):
                shards.extend(data(replica).get("shards", []))
            for shard in shards:
                actual = dtype(data(shard))
                if actual != expected:
                    raise ValueError(
                        f"AWEX wire dtype mismatch for {role} shard {name}: "
                        f"{actual} != {expected}"
                    )


def create_app(config: WeightUpdateConfig | None = None) -> FastAPI:
    config = config or WeightUpdateConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.http_session = aiohttp.ClientSession()
        yield
        await app.state.http_session.close()

    app = FastAPI(title="Weight Update Gateway", lifespan=lifespan)

    kv_store = WeightMetaStore()
    registry = PairRegistry()

    app.state.kv_store = kv_store
    app.state.registry = registry
    app.state.config = config
    app.state.operation_lock = asyncio.Lock()
    app.state.retired_pairs = set()

    def serialized(operation):
        @wraps(operation)
        async def invoke(*args, **kwargs):
            async with app.state.operation_lock:
                return await operation(*args, **kwargs)

        return invoke

    async def teardown_pair(
        pair_info: PairInfo, session, *, training_stopped: bool = False
    ) -> list[str]:
        if pair_info.mode != "awex":
            return []
        results = await asyncio.gather(
            *[
                _post(
                    session,
                    f"{url}/awex/teardown",
                    config.init_timeout_s,
                    json_data={"pair_name": pair_info.pair_name},
                )
                for url in (
                    ([] if training_stopped else pair_info.train_worker_urls)
                    + pair_info.inference_worker_urls
                )
            ],
            return_exceptions=True,
        )
        return [str(result) for result in results if isinstance(result, BaseException)]

    def _auth(request: Request) -> None:
        require_admin_key(request, config.admin_api_key)

    @app.get("/health")
    async def health() -> HealthResponse:
        return HealthResponse()

    @app.post("/connect")
    @serialized
    async def connect(request: Request, body: ConnectRequest) -> ConnectResponse:
        _auth(request)
        if (
            not body.pair_name
            or not body.train_worker_urls
            or not body.inference_worker_urls
        ):
            return JSONResponse(
                status_code=400,
                content={"error": "Pair name and both worker lists are required"},
            )
        if body.mode not in ("awex", "disk"):
            return JSONResponse(
                status_code=400, content={"error": "Unknown weight update mode"}
            )
        existing = registry.get_by_name(body.pair_name)
        if existing is not None:
            identity = (
                "train_worker_urls",
                "inference_worker_urls",
                "mode",
                "colocate",
                "save_path",
                "use_lora",
                "lora_name",
                "lora_keep_versions",
            )
            if existing.state == "connected" and all(
                getattr(existing, key) == getattr(body, key) for key in identity
            ):
                return ConnectResponse(pair_name=body.pair_name)
            return JSONResponse(
                status_code=409,
                content={"error": "Pair already exists or requires cleanup"},
            )
        if body.pair_name in app.state.retired_pairs:
            return JSONResponse(
                status_code=409,
                content={"error": "Retired pair requires a fresh generation name"},
            )
        participants = set(body.train_worker_urls + body.inference_worker_urls)
        for name in registry.list_pairs():
            other = registry.get_by_name(name)
            if participants.intersection(
                other.train_worker_urls + other.inference_worker_urls
            ):
                return JSONResponse(
                    status_code=409,
                    content={"error": "Worker already belongs to another pair"},
                )
        pending = PairInfo(
            pair_name=body.pair_name,
            train_worker_urls=body.train_worker_urls,
            inference_worker_urls=body.inference_worker_urls,
            mode=body.mode,
            colocate=body.colocate,
            state="connecting",
        )
        registry.register(pending)
        try:
            result = await connect_impl(request, body)
            if isinstance(result, JSONResponse) and result.status_code >= 400:
                registry.unregister(body.pair_name)
                # Validation rejects have not touched workers, but the caller
                # may have lost this response and still need to disconnect.
                app.state.retired_pairs.add(body.pair_name)
            return result
        except Exception as error:
            pending.state = "failed"
            failures = await teardown_pair(pending, request.app.state.http_session)
            if not failures:
                kv_store.clear_pair(body.pair_name)
            return JSONResponse(
                status_code=503,
                content={
                    "error": str(error),
                    "state": "failed",
                    "cleanup_failures": failures,
                },
            )

    @app.get("/pairs/{pair_name}")
    async def pair_status(pair_name: str, request: Request):
        _auth(request)
        pair = registry.get_by_name(pair_name)
        if pair is None:
            return JSONResponse(status_code=404, content={"error": "Pair not found"})
        return asdict(pair)

    async def connect_impl(request: Request, body: ConnectRequest) -> ConnectResponse:
        _auth(request)
        pair_name = body.pair_name
        train_urls = body.train_worker_urls
        inference_urls = body.inference_worker_urls

        if body.colocate:
            return await _connect_colocate(
                request, pair_name, train_urls, inference_urls
            )

        if body.mode == "disk":
            if not body.save_path:
                return JSONResponse(
                    status_code=400,
                    content={"error": "save_path is required when mode='disk'"},
                )
            if not os.path.isabs(body.save_path):
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "save_path must be an absolute path when mode='disk'"
                    },
                )
            if body.use_lora and not body.lora_name:
                return JSONResponse(
                    status_code=400,
                    content={"error": "lora_name is required when use_lora=True"},
                )
            pair_info = PairInfo(
                pair_name=pair_name,
                train_worker_urls=train_urls,
                inference_worker_urls=inference_urls,
                mode="disk",
                save_path=body.save_path,
                use_lora=body.use_lora,
                lora_name=body.lora_name,
                lora_keep_versions=body.lora_keep_versions,
            )
            registry.finish_connect(pair_info)
            logger.info(
                "Connected disk pair '%s' (save_path=%s, use_lora=%s, "
                "lora_keep_versions=%d)",
                pair_name,
                body.save_path,
                body.use_lora,
                body.lora_keep_versions,
            )
            return ConnectResponse(pair_name=pair_name)

        # awex mode -- LoRA is unsupported because the NCCL P2P transfer plan
        # assumes train/infer parameter names match the HF layout, but PEFT
        # exposes ``base_model.model.*.{base_layer,lora_A,lora_B}.weight`` on
        # the train side. Fail fast with an actionable error rather than
        # bubbling up a cryptic TransferPlanBuilder key-mismatch at init.
        if body.use_lora:
            return JSONResponse(
                status_code=400,
                content={
                    "error": (
                        "awex weight update does not support LoRA; set "
                        "actor.weight_update_mode=disk in your config."
                    )
                },
            )

        session = request.app.state.http_session
        init_timeout_s = config.init_timeout_s

        train_par, infer_par = await _discover_parallelism(
            session, train_urls, inference_urls, init_timeout_s
        )

        train_world_size = train_par["world_size"]
        infer_world_size = infer_par["world_size"]
        # Each inference URL is a separate DP replica.  The adapter
        # reports per-instance parallelism (e.g. TP size) but does not
        # know how many replicas exist, so we derive num_engines from the
        # URL list the controller gave us.
        num_engines = len(inference_urls)
        total_infer_ranks = infer_world_size * num_engines
        total_world_size = total_infer_ranks + train_world_size

        train_meta_resps, infer_meta_resps = await _gather_all(
            _gather_all(
                *[
                    _post_json(
                        session, f"{url}/awex/report_weight_meta", init_timeout_s
                    )
                    for url in train_urls
                ]
            ),
            _gather_all(
                *[
                    _post_json(
                        session, f"{url}/awex/report_weight_meta", init_timeout_s
                    )
                    for url in inference_urls
                ]
            ),
        )

        training_params_meta = []
        for result in train_meta_resps:
            meta = result.get("result", result.get("meta", result))
            if isinstance(meta, list):
                training_params_meta.extend(meta)
            else:
                training_params_meta.append(meta)

        infer_params_meta = inference_metadata_template(infer_meta_resps)

        _validate_wire_dtypes(training_params_meta, infer_params_meta)
        training_params_meta = _merge_training_meta_by_name(training_params_meta)
        kv_store.put(pair_name, "training_params_meta", training_params_meta)
        kv_store.put(pair_name, "infer_params_meta", infer_params_meta)

        master_addr = body.nccl_master_addr
        master_port = body.nccl_master_port

        # Use the bound host for kv_store_url so workers can reach the
        # gateway.  When bound to 0.0.0.0 any interface works, so fall
        # back to the machine IP; otherwise use the explicit bind address.
        gateway_addr = master_addr if config.host in ("0.0.0.0", "::") else config.host
        kv_store_url = f"http://{gateway_addr}:{config.gateway_port}"

        init_payload_base = {
            "pair_name": pair_name,
            "fenced": True,
            "master_addr": master_addr,
            "master_port": master_port,
            "world_size": total_world_size,
            "kv_store_url": kv_store_url,
            "infer_world_size": total_infer_ranks,
            "train_world_size": train_world_size,
            "num_engines": num_engines,
        }

        init_tasks = []
        for i, url in enumerate(inference_urls):
            init_tasks.append(
                _post(
                    session,
                    f"{url}/awex/init_weights_update_group",
                    init_timeout_s,
                    json_data={
                        **init_payload_base,
                        "transfer_rank": i,
                        # Leave part of the HTTP budget for abort and error replies.
                        "transfer_timeout_s": config.update_timeout_s * 0.8,
                    },
                )
            )
        for i, url in enumerate(train_urls):
            init_tasks.append(
                _post(
                    session,
                    f"{url}/awex/init_weights_update_group",
                    init_timeout_s,
                    json_data={
                        **init_payload_base,
                        "transfer_rank": total_infer_ranks + i,
                    },
                )
            )
        await _gather_all(*init_tasks)

        liveness_tasks = [
            _post(
                session,
                f"{url}/awex/batch_isend_irecv",
                init_timeout_s,
                json_data={"world_size": total_world_size, "pair_name": pair_name},
            )
            for url in inference_urls + train_urls
        ]
        await _gather_all(*liveness_tasks)

        pair_info = PairInfo(
            pair_name=pair_name,
            train_worker_urls=train_urls,
            inference_worker_urls=inference_urls,
            train_world_size=train_world_size,
            inference_world_size=infer_world_size,
            master_addr=master_addr,
            master_port=master_port,
        )
        registry.finish_connect(pair_info)

        logger.info("Connected pair '%s'", pair_name)
        return ConnectResponse(pair_name=pair_name)

    async def _connect_colocate(
        request: Request,
        pair_name: str,
        train_urls: list[str],
        inference_urls: list[str],
    ) -> ConnectResponse:
        session = request.app.state.http_session
        init_timeout_s = config.init_timeout_s

        train_par, infer_par = await _discover_parallelism(
            session, train_urls, inference_urls, init_timeout_s
        )

        train_world_size = train_par["world_size"]
        num_engines = len(inference_urls)
        # report_parallelism returns per-instance world_size (e.g. TP size).
        # The total inference world for colocate NCCL groups spans all engines.
        infer_world_size = infer_par["world_size"] * num_engines

        train_meta_resps, infer_meta_resps = await _gather_all(
            _gather_all(
                *[
                    _post_json(
                        session, f"{url}/awex/report_weight_meta", init_timeout_s
                    )
                    for url in train_urls
                ]
            ),
            _gather_all(
                *[
                    _post_json(
                        session, f"{url}/awex/report_weight_meta", init_timeout_s
                    )
                    for url in inference_urls
                ]
            ),
        )

        training_params_meta = []
        for result in train_meta_resps:
            meta = result.get("result", result.get("meta", result))
            if isinstance(meta, list):
                training_params_meta.extend(meta)
            else:
                training_params_meta.append(meta)

        infer_params_meta = inference_metadata_template(infer_meta_resps)

        _validate_wire_dtypes(training_params_meta, infer_params_meta)
        training_params_meta = _merge_training_meta_by_name(training_params_meta)
        kv_store.put(pair_name, "training_params_meta", training_params_meta)
        kv_store.put(pair_name, "infer_params_meta", infer_params_meta)

        gateway_addr = (
            _get_own_ip() if config.host in ("0.0.0.0", "::") else config.host
        )
        kv_store_url = f"http://{gateway_addr}:{config.gateway_port}"

        [master_port] = find_free_ports(1)

        init_payload_base = {
            "pair_name": pair_name,
            "fenced": True,
            "kv_store_url": kv_store_url,
            "infer_world_size": infer_world_size,
            "train_world_size": train_world_size,
            "num_engines": num_engines,
            "master_port": master_port,
            "admin_api_key": config.admin_api_key,
        }

        init_tasks = []
        for i, url in enumerate(inference_urls):
            init_tasks.append(
                _post(
                    session,
                    f"{url}/awex/init_colocate_weight_update",
                    init_timeout_s,
                    json_data={**init_payload_base, "transfer_rank": i},
                )
            )
        for i, url in enumerate(train_urls):
            init_tasks.append(
                _post(
                    session,
                    f"{url}/awex/init_colocate_weight_update",
                    init_timeout_s,
                    json_data={
                        **init_payload_base,
                        "transfer_rank": infer_world_size + i,
                    },
                )
            )
        await _gather_all(*init_tasks)

        pair_info = PairInfo(
            pair_name=pair_name,
            train_worker_urls=train_urls,
            inference_worker_urls=inference_urls,
            train_world_size=train_world_size,
            inference_world_size=infer_world_size,
            colocate=True,
        )
        registry.finish_connect(pair_info)

        logger.info("Connected colocate pair '%s'", pair_name)
        return ConnectResponse(pair_name=pair_name)

    async def _colocate_transfer_weights(
        pair_info: PairInfo,
        version: int,
        session: aiohttp.ClientSession,
        timeout_s: float,
    ) -> None:
        await _gather_all(
            *[
                _post(
                    session,
                    f"{url}/awex/release_memory",
                    timeout_s,
                    json_data={"tags": ["optimizer"], "pair_name": pair_info.pair_name},
                )
                for url in pair_info.train_worker_urls
            ]
        )

        await _gather_all(
            *[
                _post(
                    session,
                    f"{url}/awex/resume_memory",
                    timeout_s,
                    json_data={"tags": ["weights"], "pair_name": pair_info.pair_name},
                )
                for url in pair_info.inference_worker_urls
            ]
        )

        await _gather_all(
            *[
                _post(
                    session,
                    f"{url}/awex/execute_colocate_weight_update",
                    timeout_s,
                    json_data={"version": version, "pair_name": pair_info.pair_name},
                )
                for url in pair_info.train_worker_urls
            ],
            *[
                _post(
                    session,
                    f"{url}/awex/execute_colocate_weight_update",
                    timeout_s,
                    json_data={"version": version, "pair_name": pair_info.pair_name},
                )
                for url in pair_info.inference_worker_urls
            ],
        )

        await _gather_all(
            *[
                _post(
                    session,
                    f"{url}/awex/release_memory",
                    timeout_s,
                    json_data={"tags": ["weights"], "pair_name": pair_info.pair_name},
                )
                for url in pair_info.train_worker_urls
            ]
        )

        await _gather_all(
            *[
                _post(
                    session,
                    f"{url}/awex/resume_memory",
                    timeout_s,
                    json_data={"tags": ["kv_cache"], "pair_name": pair_info.pair_name},
                )
                for url in pair_info.inference_worker_urls
            ]
        )

        # Flush colocate KV keys for this version to prevent accumulation
        infer_world_size = pair_info.inference_world_size
        train_world_size = pair_info.train_world_size
        for i in range(train_world_size):
            transfer_rank = infer_world_size + i
            weight_key = f"colocate_weights_rank{transfer_rank}_{version}"
            done_key = f"colocate_done_rank{transfer_rank}_{version}"
            kv_store.delete(pair_info.pair_name, weight_key)
            kv_store.delete(pair_info.pair_name, done_key)

    async def _awex_transfer_weights(
        pair_info: PairInfo,
        version: int,
        session: aiohttp.ClientSession,
        timeout_s: float,
    ) -> None:
        await _gather_all(
            *[
                _post(
                    session,
                    f"{url}/awex/update_weights",
                    timeout_s,
                    json_data={"version": version, "pair_name": pair_info.pair_name},
                )
                for url in pair_info.train_worker_urls + pair_info.inference_worker_urls
            ]
        )

    async def _disk_transfer_weights(
        pair_info: PairInfo,
        version: int,
        session: aiohttp.ClientSession,
        timeout_s: float,
    ) -> None:
        from areal.api.io_struct import SaveLoadMeta, get_versioned_lora_name
        from areal.infra.rpc.serialization import serialize_value

        save_path = os.path.join(pair_info.save_path, f"weight_update_v{version}")

        save_meta = SaveLoadMeta(path=save_path, weight_format="hf", with_optim=False)
        save_payload = {
            "args": serialize_value([save_meta]),
            "kwargs": serialize_value({}),
        }
        await _gather_all(
            *[
                _post_json(session, f"{url}/save", timeout_s, json_data=save_payload)
                for url in pair_info.train_worker_urls
            ]
        )

        if pair_info.use_lora:
            lora_name = get_versioned_lora_name(pair_info.lora_name, version)
            await _gather_all(
                *[
                    _post_json(
                        session,
                        f"{url}/load_lora_adapter",
                        timeout_s,
                        json_data={
                            "lora_name": lora_name,
                            "lora_path": save_path,
                        },
                    )
                    for url in pair_info.inference_worker_urls
                ]
            )
            # Unload the version that fell outside the retention window so
            # sglang does not accumulate one adapter per train step (leaks
            # VRAM and eventually hangs). Best-effort: the stale adapter may
            # already have been evicted or never loaded on this worker.
            keep = pair_info.lora_keep_versions
            if keep > 0 and version - keep >= 0:
                stale_name = get_versioned_lora_name(
                    pair_info.lora_name, version - keep
                )

                async def _unload(url: str) -> None:
                    try:
                        await _post_json(
                            session,
                            f"{url}/unload_lora_adapter",
                            timeout_s,
                            json_data={"lora_name": stale_name},
                        )
                    except Exception as e:
                        logger.warning(
                            "unload_lora_adapter(%s) on %s failed (best-effort): %s",
                            stale_name,
                            url,
                            e,
                        )

                await _gather_all(
                    *[_unload(url) for url in pair_info.inference_worker_urls]
                )
        else:
            await _gather_all(
                *[
                    _post_json(
                        session,
                        f"{url}/update_weights_from_disk",
                        timeout_s,
                        json_data={
                            "model_path": save_path,
                            "abort_all_requests": True,
                        },
                    )
                    for url in pair_info.inference_worker_urls
                ]
            )

    @app.post("/update_weights")
    @serialized
    async def update_weights(
        request: Request, body: UpdateWeightsRequest
    ) -> WeightUpdateResult:
        _auth(request)

        pair_info = registry.get_by_name(body.pair_name)
        if pair_info is None:
            return JSONResponse(
                status_code=404,
                content={"error": f"Pair '{body.pair_name}' not found"},
            )

        session = request.app.state.http_session
        timeout_s = config.update_timeout_s
        start = time.monotonic()

        if pair_info.state != "connected":
            return JSONResponse(
                status_code=409,
                content={"error": "Pair requires cleanup before updating"},
            )
        if body.version < 0 or (
            pair_info.has_published and body.version < pair_info.last_version
        ):
            return JSONResponse(
                status_code=409,
                content={"error": "Weight version cannot move backwards"},
            )
        if pair_info.has_published and body.version == pair_info.last_version:
            return WeightUpdateResult(status="ok", version=body.version, duration_ms=0)

        try:
            if pair_info.colocate:
                await _colocate_transfer_weights(
                    pair_info, body.version, session, timeout_s
                )
            elif pair_info.mode == "disk":
                await _disk_transfer_weights(
                    pair_info, body.version, session, timeout_s
                )
            else:
                await _awex_transfer_weights(
                    pair_info, body.version, session, timeout_s
                )
        except Exception as e:
            pair_info.state = "failed"
            duration_ms = (time.monotonic() - start) * 1000
            logger.error(
                "Weight update failed for pair '%s': %s",
                pair_info.pair_name,
                e,
            )
            return WeightUpdateResult(
                status="error",
                version=body.version,
                duration_ms=duration_ms,
                error=str(e),
            )

        duration_ms = (time.monotonic() - start) * 1000
        pair_info.last_version = body.version
        pair_info.has_published = True
        logger.info(
            "Weight update completed for pair '%s' v%d (%.1fms)",
            pair_info.pair_name,
            body.version,
            duration_ms,
        )
        return WeightUpdateResult(
            status="ok", version=body.version, duration_ms=duration_ms
        )

    @app.post("/disconnect")
    @serialized
    async def disconnect(
        request: Request, body: DisconnectRequest
    ) -> DisconnectResponse:
        _auth(request)

        pair_info = registry.get_by_name(body.pair_name)
        if pair_info is None:
            if body.pair_name in app.state.retired_pairs:
                return DisconnectResponse(pair_name=body.pair_name)
            return JSONResponse(
                status_code=404,
                content={"error": f"Pair '{body.pair_name}' not found"},
            )

        pair_info.state = "draining"
        failures = await teardown_pair(
            pair_info,
            getattr(request.app.state, "http_session", None),
            training_stopped=body.training_stopped,
        )
        if failures:
            pair_info.state = "failed"
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Weight update teardown incomplete",
                    "failures": failures,
                },
            )

        registry.unregister(pair_info.pair_name)
        kv_store.clear_pair(pair_info.pair_name)
        app.state.retired_pairs.add(pair_info.pair_name)

        return DisconnectResponse(pair_name=pair_info.pair_name)

    @app.get("/weight_meta/{pair_name}/{key}")
    async def kv_get(pair_name: str, key: str) -> KVGetResponse:
        # No auth — workers read metadata during init_weight_update_group
        # without credentials.
        value = kv_store.get(pair_name, key)
        if value is None:
            return JSONResponse(status_code=404, content={"error": "Key not found"})
        return KVGetResponse(value=value)

    @app.put("/weight_meta/{pair_name}/{key}")
    async def kv_put(
        pair_name: str, key: str, request: Request, body: KVPutBody
    ) -> StatusResponse:
        _auth(request)
        kv_store.put(pair_name, key, body.value)
        return StatusResponse()

    @app.delete("/weight_meta/{pair_name}/{key}")
    async def kv_delete(pair_name: str, key: str, request: Request) -> KVDeleteResponse:
        _auth(request)
        deleted = kv_store.delete(pair_name, key)
        return KVDeleteResponse(deleted=deleted)

    @app.put("/weight_meta/{pair_name}/set/{key}")
    async def kv_set_add(
        pair_name: str, key: str, request: Request, body: KVSetAddBody
    ) -> StatusResponse:
        _auth(request)
        kv_store.add_to_set(pair_name, key, body.value)
        return StatusResponse()

    @app.get("/weight_meta/{pair_name}/set/{key}/size")
    async def kv_set_size(
        pair_name: str, key: str, request: Request
    ) -> KVSetSizeResponse:
        _auth(request)
        size = kv_store.set_size(pair_name, key)
        return KVSetSizeResponse(size=size)

    return app
