# SPDX-License-Identifier: Apache-2.0

"""Data Service Gateway — thin HTTP proxy with auth, routing, and forwarding."""

from __future__ import annotations

import asyncio

import aiohttp
from fastapi import FastAPI, HTTPException, Request

from areal.infra.data_service.gateway.auth import (
    DatasetKeyRegistry,
    extract_bearer_token,
    require_admin_key,
)
from areal.infra.data_service.gateway.config import GatewayConfig
from areal.utils import logging

logger = logging.getLogger("DataGateway")


async def _query_router(router_addr: str, admin_key: str, timeout: float) -> str:
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=timeout), trust_env=False
    ) as session:
        async with session.post(
            f"{router_addr}/route",
            json={},
            headers={"Authorization": f"Bearer {admin_key}"},
        ) as resp:
            if resp.status != 200:
                raise HTTPException(
                    status_code=502, detail=f"Router error: {await resp.text()}"
                )
            return (await resp.json())["worker_addr"]


async def _get_all_worker_addrs(
    router_addr: str, admin_key: str, timeout: float
) -> list[str]:
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=timeout), trust_env=False
    ) as session:
        async with session.get(
            f"{router_addr}/workers",
            headers={"Authorization": f"Bearer {admin_key}"},
        ) as resp:
            if resp.status != 200:
                raise HTTPException(
                    status_code=502, detail=f"Router error: {await resp.text()}"
                )
            return [w["addr"] for w in (await resp.json())["workers"]]


async def _broadcast_to_workers(
    worker_addrs: list[str], endpoint: str, payload: dict, timeout: float
) -> list[dict]:
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=timeout), trust_env=False
    ) as session:

        async def _send_post(addr: str) -> dict:
            try:
                async with session.post(f"{addr}{endpoint}", json=payload) as resp:
                    try:
                        data = await resp.json()
                    except Exception:
                        data = {"raw": await resp.text()}
                    return {"addr": addr, "status": resp.status, "data": data}
            except Exception as exc:
                return {"addr": addr, "status": 500, "error": str(exc)}

        results = await asyncio.gather(*(_send_post(addr) for addr in worker_addrs))
    return list(results)


def create_gateway_app(config: GatewayConfig) -> FastAPI:
    app = FastAPI(title="AReaL Data Gateway")
    registry = DatasetKeyRegistry(config.admin_api_key)

    # Helper: resolve dataset key to dataset_id, raise if invalid
    def _resolve_dataset_key(token: str) -> str:
        dataset_id = registry.resolve(token)
        if dataset_id is None:
            raise HTTPException(status_code=401, detail="Invalid dataset API key")
        return dataset_id

    def _check_broadcast_results(results: list[dict], operation: str) -> None:
        failed = [r for r in results if r["status"] != 200]
        if failed:
            details = ", ".join(
                f"{r['addr']}: {r.get('error', r.get('data'))}" for r in failed
            )
            raise HTTPException(
                status_code=502,
                detail=(
                    f"{operation} failed on {len(failed)}/{len(results)} "
                    f"workers: {details}"
                ),
            )

    # ===== Health =====
    @app.get("/health")
    async def health():
        return {"status": "ok", "router_addr": config.router_addr}

    # ===== Admin: Register Dataset =====
    @app.post("/v1/datasets/register")
    async def register_dataset(request: Request):
        require_admin_key(request, config.admin_api_key)
        body = await request.json()

        dataset_id = body.get(
            "dataset_id",
            f"{body.get('split', 'train')}-{body.get('dataset_path', 'unknown').split('/')[-1]}",
        )

        worker_addrs = await _get_all_worker_addrs(
            config.router_addr,
            config.admin_api_key,
            config.router_timeout,
        )
        if not worker_addrs:
            raise HTTPException(status_code=503, detail="No workers available")
        load_payload = {**body, "dataset_id": dataset_id}
        results = await _broadcast_to_workers(
            worker_addrs,
            "/datasets/load",
            load_payload,
            config.forward_timeout,
        )

        successful_addrs = [r["addr"] for r in results if r["status"] == 200]
        failed = [r for r in results if r["status"] != 200]
        if failed:
            rollback_error_detail = ""
            if successful_addrs:
                rollback_results = await _broadcast_to_workers(
                    successful_addrs,
                    "/datasets/unload",
                    {"dataset_id": dataset_id},
                    config.forward_timeout,
                )
                rollback_failed = [r for r in rollback_results if r["status"] != 200]
                if rollback_failed:
                    rollback_error_detail = (
                        " Rollback failed on "
                        f"{len(rollback_failed)}/{len(rollback_results)} workers."
                    )

            details = ", ".join(
                f"{r['addr']}: {r.get('error', r.get('data'))}" for r in failed
            )
            raise HTTPException(
                status_code=502,
                detail=(
                    f"register_dataset failed on {len(failed)}/{len(results)} workers: "
                    f"{details}.{rollback_error_detail}"
                ),
            )

        registry.revoke(dataset_id)
        api_key = registry.generate_key(dataset_id)

        total_size = 0
        for result in results:
            if result["status"] == 200:
                d = result.get("data", {})
                total_size += d.get("dataset_size", 0)

        return {
            "api_key": api_key,
            "dataset_id": dataset_id,
            "dataset_size": total_size,
            "num_workers": len(worker_addrs),
        }

    # ===== Admin: Unregister Dataset =====
    @app.post("/v1/datasets/unregister")
    async def unregister_dataset(request: Request):
        require_admin_key(request, config.admin_api_key)
        body = await request.json()
        dataset_id = body.get("dataset_id")
        if not dataset_id:
            raise HTTPException(status_code=400, detail="dataset_id is required")

        worker_addrs = await _get_all_worker_addrs(
            config.router_addr,
            config.admin_api_key,
            config.router_timeout,
        )
        results = await _broadcast_to_workers(
            worker_addrs,
            "/datasets/unload",
            {"dataset_id": dataset_id},
            config.forward_timeout,
        )
        _check_broadcast_results(results, "unregister_dataset")
        registry.revoke(dataset_id)
        return {"status": "ok"}

    # ===== Admin: Shutdown =====
    @app.post("/v1/shutdown")
    async def shutdown(request: Request):
        require_admin_key(request, config.admin_api_key)
        try:
            worker_addrs = await _get_all_worker_addrs(
                config.router_addr,
                config.admin_api_key,
                config.router_timeout,
            )
            dataset_ids = list(registry._dataset_to_key.keys())
            for dataset_id in dataset_ids:
                await _broadcast_to_workers(
                    worker_addrs,
                    "/datasets/unload",
                    {"dataset_id": dataset_id},
                    config.forward_timeout,
                )
                registry.revoke(dataset_id)
        except Exception as exc:
            logger.warning("Error during shutdown broadcast: %s", exc)
        return {"status": "ok"}

    # ===== Admin: Workers =====
    @app.get("/v1/workers")
    async def list_workers(request: Request):
        require_admin_key(request, config.admin_api_key)
        worker_addrs = await _get_all_worker_addrs(
            config.router_addr,
            config.admin_api_key,
            config.router_timeout,
        )
        return {"workers": [{"addr": addr} for addr in worker_addrs]}

    # ===== Consumer: Fetch Samples by Index =====
    @app.post("/v1/samples/fetch")
    async def fetch_samples(request: Request):
        token = extract_bearer_token(request)
        dataset_id = _resolve_dataset_key(token)
        body = await request.json()
        indices = body.get("indices", [])

        worker_addr = await _query_router(
            config.router_addr,
            config.admin_api_key,
            config.router_timeout,
        )
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=config.forward_timeout),
            trust_env=False,
        ) as session:
            async with session.post(
                f"{worker_addr}/v1/samples/fetch",
                json={"dataset_id": dataset_id, "indices": indices},
            ) as resp:
                if resp.status != 200:
                    raise HTTPException(
                        status_code=502,
                        detail=f"Worker fetch_samples error: {await resp.text()}",
                    )
                return await resp.json()

    # ===== Consumer: Epoch Advance =====
    @app.post("/v1/epochs/advance")
    async def epoch_advance(request: Request):
        token = extract_bearer_token(request)
        dataset_id = _resolve_dataset_key(token)
        body = await request.json()
        epoch = body.get("epoch", 0)

        worker_addrs = await _get_all_worker_addrs(
            config.router_addr,
            config.admin_api_key,
            config.router_timeout,
        )
        if not worker_addrs:
            raise HTTPException(status_code=503, detail="No workers available")
        results = await _broadcast_to_workers(
            worker_addrs,
            "/epoch/reset",
            {"dataset_id": dataset_id, "epoch": epoch},
            config.forward_timeout,
        )
        _check_broadcast_results(results, "epoch_advance")
        return {
            "status": "ok",
            "workers_reset": sum(1 for result in results if result["status"] == 200),
        }

    # ===== Consumer: State Save =====
    @app.post("/v1/state/save")
    async def state_save(request: Request):
        token = extract_bearer_token(request)
        dataset_id = _resolve_dataset_key(token)
        body = await request.json()
        path = body.get("path", "")

        worker_addrs = await _get_all_worker_addrs(
            config.router_addr,
            config.admin_api_key,
            config.router_timeout,
        )
        results = await _broadcast_to_workers(
            worker_addrs,
            "/state/save",
            {"dataset_id": dataset_id, "path": path},
            config.forward_timeout,
        )
        _check_broadcast_results(results, "state_save")
        return {"status": "ok", "path": path}

    # ===== Consumer: State Load =====
    @app.post("/v1/state/load")
    async def state_load(request: Request):
        token = extract_bearer_token(request)
        dataset_id = _resolve_dataset_key(token)
        body = await request.json()
        path = body.get("path", "")

        worker_addrs = await _get_all_worker_addrs(
            config.router_addr,
            config.admin_api_key,
            config.router_timeout,
        )
        results = await _broadcast_to_workers(
            worker_addrs,
            "/state/load",
            {"dataset_id": dataset_id, "path": path},
            config.forward_timeout,
        )
        _check_broadcast_results(results, "state_load")
        return {"status": "ok"}

    # ===== Consumer: Status =====
    @app.get("/v1/status")
    async def status(request: Request):
        token = extract_bearer_token(request)
        dataset_id = _resolve_dataset_key(token)

        try:
            worker_addr = await _query_router(
                config.router_addr,
                config.admin_api_key,
                config.router_timeout,
            )
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=config.forward_timeout),
                trust_env=False,
            ) as session:
                async with session.get(f"{worker_addr}/health") as resp:
                    if resp.status == 200:
                        payload = await resp.json()
                        payload["dataset_id"] = dataset_id
                        return payload
        except Exception:
            pass
        return {"status": "ok", "dataset_id": dataset_id}

    return app
