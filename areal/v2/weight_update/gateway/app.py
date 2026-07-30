# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import os
import socket
import time
from contextlib import asynccontextmanager
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
    WeightUpdateConfig,
    WeightUpdateResult,
)
from areal.v2.weight_update.gateway.kv_store import WeightMetaStore
from areal.v2.weight_update.gateway.pair_registry import PairInfo, PairRegistry

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


class KVPutBody(BaseModel):
    value: Any = None


class KVSetAddBody(BaseModel):
    value: str = ""


class HealthResponse(BaseModel):
    status: str = "healthy"
    meta_server: str | None = None


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


@async_http_retry
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


@async_http_retry
async def _post(
    session: aiohttp.ClientSession,
    url: str,
    timeout_s: float,
    json_data: Any = None,
) -> None:
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with session.post(url, json=json_data, timeout=timeout) as resp:
        resp.raise_for_status()


async def _post_once(
    session: aiohttp.ClientSession,
    url: str,
    timeout_s: float,
    json_data: Any = None,
) -> None:
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with session.post(url, json=json_data, timeout=timeout) as resp:
        resp.raise_for_status()


def _get_own_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def _ensure_meta_server(app: FastAPI) -> str:
    """Lazily start the awex MetaServer subprocess and return its address.

    The MetaServer runs as a daemon subprocess (multiprocessing spawn
    context) next to the gateway, so it dies automatically when the
    gateway exits. It is spawned lazily — only when a colocate pair
    needs it — so non-colocate deployments never pay the cost or
    require awex to be importable.
    """
    if getattr(app.state, "meta_server_addr", ""):
        return app.state.meta_server_addr

    from awex.meta.meta_server import (  # pyright: ignore[reportMissingImports]
        start_meta_server,
    )

    host, port = start_meta_server()
    app.state.meta_server_addr = f"{host}:{port}"
    logger.info("Started awex MetaServer at %s", app.state.meta_server_addr)
    return app.state.meta_server_addr


def _merge_training_meta_by_name(meta_list: list[dict]) -> list[dict]:
    """Merge serialized training ParameterMeta entries by parameter name.

    Each FSDP worker reports metadata for its own local shard only.
    With ``dp_size > 1`` the same parameter name appears once per worker,
    each carrying a single shard.  The ``TransferPlanBuilder`` indexes
    parameters with ``{meta.name: meta}``, so duplicates silently shadow
    earlier entries.  We merge them here so the builder sees one
    ``ParameterMeta`` per name with all shards in a single replica.
    """
    by_name: dict[str, dict] = {}
    overflow: list[dict] = []
    for pm in meta_list:
        data = pm.get("data", pm) if isinstance(pm, dict) else pm
        name = data.get("name") if isinstance(data, dict) else None
        if name is None:
            logger.warning("Found parameter metadata with no name: %s", pm)
            overflow.append(pm)
            continue

        if name not in by_name:
            by_name[name] = pm
        else:
            existing_data = by_name[name].get("data", by_name[name])
            existing_data.setdefault("shards", []).extend(data.get("shards", []))
            ex_replicas = existing_data.get("replicas", [])
            new_replicas = data.get("replicas", [])
            if ex_replicas and new_replicas:
                ex_rep_data = ex_replicas[0].get("data", ex_replicas[0])
                new_rep_data = new_replicas[0].get("data", new_replicas[0])
                ex_rep_data.setdefault("shards", []).extend(
                    new_rep_data.get("shards", [])
                )
    return list(by_name.values()) + overflow


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
    # Set by _ensure_meta_server on first colocate connect; empty until then.
    app.state.meta_server_addr = ""

    def _auth(request: Request) -> None:
        require_admin_key(request, config.admin_api_key)

    @app.get("/health")
    async def health() -> HealthResponse:
        return HealthResponse(meta_server=app.state.meta_server_addr or None)

    @app.post("/connect")
    async def connect(request: Request, body: ConnectRequest) -> ConnectResponse:
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
            registry.register(pair_info)
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

        train_par, infer_par = await asyncio.gather(
            _get_json(
                session,
                f"{train_urls[0]}/awex/report_parallelism",
                init_timeout_s,
            ),
            _get_json(
                session,
                f"{inference_urls[0]}/awex/report_parallelism",
                init_timeout_s,
            ),
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

        train_meta_resps, infer_meta_resps = await asyncio.gather(
            asyncio.gather(
                *[
                    _post_json(
                        session, f"{url}/awex/report_weight_meta", init_timeout_s
                    )
                    for url in train_urls
                ]
            ),
            asyncio.gather(
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
        training_params_meta = _merge_training_meta_by_name(training_params_meta)

        infer_params_meta = []
        for result in infer_meta_resps:
            meta = result.get("result", result.get("meta", result))
            if isinstance(meta, list):
                infer_params_meta.extend(meta)
            else:
                infer_params_meta.append(meta)

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
                    json_data={**init_payload_base, "transfer_rank": i},
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
        await asyncio.gather(*init_tasks)

        liveness_tasks = [
            _post(
                session,
                f"{url}/awex/batch_isend_irecv",
                init_timeout_s,
                json_data={"world_size": total_world_size},
            )
            for url in inference_urls + train_urls
        ]
        await asyncio.gather(*liveness_tasks)

        pair_info = PairInfo(
            pair_name=pair_name,
            train_worker_urls=train_urls,
            inference_worker_urls=inference_urls,
            train_world_size=train_world_size,
            inference_world_size=infer_world_size,
            master_addr=master_addr,
            master_port=master_port,
        )
        registry.register(pair_info)

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
        meta_server_addr = _ensure_meta_server(request.app)

        train_parallelism, infer_parallelism = await asyncio.gather(
            asyncio.gather(
                *[
                    _get_json(
                        session,
                        f"{url}/awex/report_parallelism",
                        init_timeout_s,
                    )
                    for url in train_urls
                ]
            ),
            asyncio.gather(
                *[
                    _get_json(
                        session,
                        f"{url}/awex/report_parallelism",
                        init_timeout_s,
                    )
                    for url in inference_urls
                ]
            ),
        )

        def _index_workers_by_device(
            urls: list[str], reports: list[dict]
        ) -> tuple[dict[tuple[str, int], str], list[dict[str, Any]]]:
            urls_by_device: dict[tuple[str, int], list[str]] = {}
            for url, report in zip(urls, reports, strict=True):
                device = (str(report["ip"]), int(report["device_id"]))
                urls_by_device.setdefault(device, []).append(url)

            duplicates = [
                {"ip": ip, "device_id": device_id, "urls": device_urls}
                for (ip, device_id), device_urls in sorted(urls_by_device.items())
                if len(device_urls) > 1
            ]
            return (
                {
                    device: device_urls[0]
                    for device, device_urls in urls_by_device.items()
                },
                duplicates,
            )

        def _expand_inference_coverage(
            urls: list[str], reports: list[dict]
        ) -> tuple[dict[tuple[str, int], tuple[str, int, int]], list[dict[str, Any]]]:
            # One inference URL fronts tp_size*pp_size scheduler processes, one
            # per GPU, but its report carries only rank-0's device id. Expand to
            # per-GPU coverage assuming the contiguous node-local assignment
            # used when the servers were forked (base_gpu_id ordering). Each
            # device maps to (url, engine_base_offset=local rank within the
            # server, instance_world).
            coverage: dict[tuple[str, int], list[tuple[str, int, int]]] = {}
            for url, report in zip(urls, reports, strict=True):
                ip = str(report["ip"])
                base_dev = int(report["device_id"])
                instance_world = max(
                    1,
                    int(report.get("tp_size", 1)) * int(report.get("pp_size", 1)),
                )
                for local in range(instance_world):
                    coverage.setdefault((ip, base_dev + local), []).append(
                        (url, local, instance_world)
                    )
            duplicates = [
                {"ip": ip, "device_id": dev, "urls": [c[0] for c in covers]}
                for (ip, dev), covers in sorted(coverage.items())
                if len(covers) > 1
            ]
            return (
                {device: covers[0] for device, covers in coverage.items()},
                duplicates,
            )

        train_by_device, train_duplicates = _index_workers_by_device(
            train_urls, train_parallelism
        )
        infer_by_device, infer_duplicates = _expand_inference_coverage(
            inference_urls, infer_parallelism
        )
        if train_duplicates or infer_duplicates:
            return JSONResponse(
                status_code=422,
                content={
                    "error": "Duplicate colocate workers reported the same device",
                    "duplicates": {
                        "training": train_duplicates,
                        "inference": infer_duplicates,
                    },
                },
            )

        train_devices = set(train_by_device)
        infer_devices = set(infer_by_device)
        if train_devices != infer_devices:

            def _device_rows(devices: set[tuple[str, int]]) -> list[dict[str, Any]]:
                return [
                    {"ip": ip, "device_id": device_id}
                    for ip, device_id in sorted(devices)
                ]

            return JSONResponse(
                status_code=422,
                content={
                    "error": "Colocate workers must have exactly matching devices",
                    "unmatched": {
                        "training": _device_rows(train_devices - infer_devices),
                        "inference": _device_rows(infer_devices - train_devices),
                    },
                },
            )

        # Physical device pairing prevents URL ordering from crossing CUDA IPC peers.
        paired_devices = sorted(train_devices)
        paired_world_size = len(paired_devices)
        train_world_size = paired_world_size
        infer_world_size = paired_world_size
        num_engines = len(inference_urls)

        # Engine-major rank layout: servers sorted by (ip, base_gpu) get
        # engine index e; rank of (server e, local k) = e*instance_world + k.
        # With contiguous per-server coverage this must equal the globally
        # sorted (ip, gpu) position - verified below so a violated contiguity
        # assumption fails loudly instead of corrupting the IPC pairing.
        server_base: dict[str, tuple[int, int]] = {}
        server_infos = sorted(
            {
                (str(r["ip"]), int(r["device_id"]), url, iw)
                for url, r, iw in (
                    (
                        u,
                        rep,
                        max(
                            1,
                            int(rep.get("tp_size", 1)) * int(rep.get("pp_size", 1)),
                        ),
                    )
                    for u, rep in zip(inference_urls, infer_parallelism, strict=True)
                )
            }
        )
        next_base = 0
        for _ip, _dev, url, iw in server_infos:
            server_base[url] = (next_base, iw)
            next_base += iw

        pairing_table = []
        for rank, (ip, device_id) in enumerate(paired_devices):
            infer_url, infer_local, _iw = infer_by_device[(ip, device_id)]
            base_rank, _ = server_base[infer_url]
            if base_rank + infer_local != rank:
                return JSONResponse(
                    status_code=422,
                    content={
                        "error": (
                            "Inference server GPU coverage is not contiguous in "
                            "(ip, gpu) order; colocate rank layout would be "
                            "inconsistent"
                        ),
                        "device": {"ip": ip, "device_id": device_id},
                        "expected_rank": base_rank + infer_local,
                        "sorted_rank": rank,
                    },
                )
            pairing_table.append(
                {
                    "ip": ip,
                    "device_id": device_id,
                    "train_url": train_by_device[(ip, device_id)],
                    "infer_url": infer_url,
                    # The N-rank offset is the awex colocate donor/reader convention.
                    "train_rank": paired_world_size + rank,
                    "infer_rank": rank,
                }
            )

        train_meta_resps, infer_meta_resps = await asyncio.gather(
            asyncio.gather(
                *[
                    _post_json(
                        session, f"{url}/awex/report_weight_meta", init_timeout_s
                    )
                    for url in train_urls
                ]
            ),
            asyncio.gather(
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
        training_params_meta = _merge_training_meta_by_name(training_params_meta)

        infer_params_meta = []
        for result in infer_meta_resps:
            meta = result.get("result", result.get("meta", result))
            if isinstance(meta, list):
                infer_params_meta.extend(meta)
            else:
                infer_params_meta.append(meta)

        kv_store.put(pair_name, "training_params_meta", training_params_meta)
        kv_store.put(pair_name, "infer_params_meta", infer_params_meta)

        [master_port] = find_free_ports(1)

        init_payload_base = {
            "pair_name": pair_name,
            "meta_server_addr": meta_server_addr,
            "infer_world_size": infer_world_size,
            "train_world_size": train_world_size,
            "num_engines": num_engines,
            "master_port": master_port,
            "admin_api_key": config.admin_api_key,
        }

        init_tasks = []
        # Inference init goes once per SERVER with its engine-base rank: the
        # request is broadcast to every scheduler process behind the URL, and
        # each derives its own rank as base + its tp/pp local rank.
        for url, (base_rank, _iw) in server_base.items():
            init_tasks.append(
                _post(
                    session,
                    f"{url}/awex/init_colocate_weight_update",
                    init_timeout_s,
                    json_data={
                        **init_payload_base,
                        "transfer_rank": base_rank,
                    },
                )
            )
        for pair in pairing_table:
            init_tasks.append(
                _post(
                    session,
                    f"{pair['train_url']}/awex/init_colocate_weight_update",
                    init_timeout_s,
                    json_data={
                        **init_payload_base,
                        "transfer_rank": pair["train_rank"],
                    },
                )
            )
        await asyncio.gather(*init_tasks)

        pair_info = PairInfo(
            pair_name=pair_name,
            train_worker_urls=train_urls,
            inference_worker_urls=inference_urls,
            train_world_size=train_world_size,
            inference_world_size=infer_world_size,
            colocate=True,
            pairing_table=pairing_table,
        )
        registry.register(pair_info)

        for pair in pairing_table:
            logger.info(
                "[colocate-pairing] (ip=%s, gpu=%s): infer_rank=%s <- train_rank=%s",
                pair["ip"],
                pair["device_id"],
                pair["infer_rank"],
                pair["train_rank"],
            )
        logger.info(
            "Connected colocate pair '%s' (%d device pairs, meta_server=%s, "
            "master_port=%s)",
            pair_name,
            len(pairing_table),
            meta_server_addr,
            master_port,
        )
        return ConnectResponse(pair_name=pair_name)

    async def _colocate_execute_weight_update(
        pair_info: PairInfo,
        version: int,
        session: aiohttp.ClientSession,
        timeout_s: float,
    ) -> None:
        # Each adapter's execute_colocate_weight_update is self-contained:
        # both sides coordinate memory release/resume and the CUDA IPC
        # handoff among themselves via the MetaServer, and only return 200
        # once their side of the update has completed. The gateway just
        # dispatches to everyone concurrently, exactly like the awex
        # separation path. Single-shot POST: reader/writer NCCL group init
        # is not reentrant, so a blind HTTP retry would fail with
        # "group name has already been created".
        logger.info(
            "[colocate-update] v%d: dispatching execute to %d train + %d "
            "inference workers",
            version,
            len(pair_info.train_worker_urls),
            len(pair_info.inference_worker_urls),
        )
        await asyncio.gather(
            *[
                _post_once(
                    session,
                    f"{url}/awex/execute_colocate_weight_update",
                    timeout_s,
                    json_data={"version": version},
                )
                for url in pair_info.train_worker_urls + pair_info.inference_worker_urls
            ]
        )
        logger.info("[colocate-update] v%d: all workers completed", version)

    async def _awex_transfer_weights(
        pair_info: PairInfo,
        version: int,
        session: aiohttp.ClientSession,
        timeout_s: float,
    ) -> None:
        await asyncio.gather(
            *[
                _post(
                    session,
                    f"{url}/awex/update_weights",
                    timeout_s,
                    json_data={"version": version},
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
        await asyncio.gather(
            *[
                _post_json(session, f"{url}/save", timeout_s, json_data=save_payload)
                for url in pair_info.train_worker_urls
            ]
        )

        if pair_info.use_lora:
            lora_name = get_versioned_lora_name(pair_info.lora_name, version)
            await asyncio.gather(
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

                await asyncio.gather(
                    *[_unload(url) for url in pair_info.inference_worker_urls]
                )
        else:
            await asyncio.gather(
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

        try:
            if pair_info.colocate:
                await _colocate_execute_weight_update(
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
    async def disconnect(
        request: Request, body: DisconnectRequest
    ) -> DisconnectResponse:
        _auth(request)

        pair_info = registry.get_by_name(body.pair_name)
        if pair_info is None:
            return JSONResponse(
                status_code=404,
                content={"error": f"Pair '{body.pair_name}' not found"},
            )

        registry.unregister(pair_info.pair_name)
        kv_store.clear_pair(pair_info.pair_name)

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
