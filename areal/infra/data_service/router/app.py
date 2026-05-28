# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import hmac
import importlib
from contextlib import asynccontextmanager

import aiohttp

from areal.infra.data_service.router.config import RouterConfig
from areal.utils import logging

_fastapi = importlib.import_module("fastapi")
FastAPI = _fastapi.FastAPI
HTTPException = _fastapi.HTTPException
Request = _fastapi.Request
BaseModel = importlib.import_module("pydantic").BaseModel

logger = logging.getLogger("DataRouter")


def _extract_bearer_token(request: Request) -> str:
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    raise HTTPException(
        status_code=401,
        detail="Missing or malformed Authorization header.",
    )


def _require_admin_key(request: Request, admin_key: str) -> str:
    token = _extract_bearer_token(request)
    if not hmac.compare_digest(token, admin_key):
        raise HTTPException(status_code=403, detail="Invalid admin API key.")
    return token


class RegisterWorkerRequest(BaseModel):
    worker_addr: str


class UnregisterWorkerRequest(BaseModel):
    worker_addr: str


def create_router_app(config: RouterConfig) -> FastAPI:
    registered_workers: list[str] = []
    worker_healthy: dict[str, bool] = {}
    rr_idx: int = 0
    lock = asyncio.Lock()

    async def _poll_workers() -> None:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=config.worker_health_timeout),
            trust_env=False,
        ) as session:

            async def _check_health(addr: str) -> None:
                try:
                    async with session.get(f"{addr}/health") as resp:
                        worker_healthy[addr] = resp.status == 200
                except Exception:
                    worker_healthy[addr] = False

            while True:
                await asyncio.gather(
                    *(_check_health(addr) for addr in list(registered_workers))
                )
                await asyncio.sleep(config.poll_interval)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("DataRouter starting")
        poll_task = asyncio.create_task(_poll_workers())
        yield
        poll_task.cancel()
        try:
            await poll_task
        except asyncio.CancelledError:
            pass
        logger.info("DataRouter shutting down")

    app = FastAPI(title="AReaL Data Router", lifespan=lifespan)
    app.state.worker_healthy = worker_healthy

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "workers": len(registered_workers),
            "healthy": sum(1 for h in worker_healthy.values() if h),
        }

    @app.post("/register")
    async def register(body: RegisterWorkerRequest, request: Request):
        _require_admin_key(request, config.admin_api_key)
        async with lock:
            if body.worker_addr not in registered_workers:
                registered_workers.append(body.worker_addr)
                worker_healthy[body.worker_addr] = True
                logger.info(
                    "Worker registered: %s (total=%d)",
                    body.worker_addr,
                    len(registered_workers),
                )
        return {"status": "ok"}

    @app.post("/unregister")
    async def unregister(body: UnregisterWorkerRequest, request: Request):
        _require_admin_key(request, config.admin_api_key)
        async with lock:
            if body.worker_addr in registered_workers:
                registered_workers.remove(body.worker_addr)
                worker_healthy.pop(body.worker_addr, None)
                logger.info("Worker unregistered: %s", body.worker_addr)
        return {"status": "ok"}

    @app.post("/route")
    async def route(request: Request):
        nonlocal rr_idx
        _require_admin_key(request, config.admin_api_key)
        async with lock:
            healthy = [
                addr for addr in registered_workers if worker_healthy.get(addr, False)
            ]
            if not healthy:
                raise HTTPException(
                    status_code=503,
                    detail="No healthy workers available",
                )
            addr = healthy[rr_idx % len(healthy)]
            rr_idx += 1
        return {"worker_addr": addr}

    @app.get("/workers")
    async def list_workers(request: Request):
        _require_admin_key(request, config.admin_api_key)
        return {
            "workers": [
                {"addr": addr, "healthy": worker_healthy.get(addr, False)}
                for addr in registered_workers
            ]
        }

    return app
