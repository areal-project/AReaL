# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import hmac
import importlib
from contextlib import asynccontextmanager
from typing import Any

from areal.experimental.training_service.router.config import RouterConfig
from areal.experimental.training_service.router.state import ModelRegistry
from areal.utils import logging

httpx = importlib.import_module("httpx")
fastapi = importlib.import_module("fastapi")
pydantic = importlib.import_module("pydantic")
FastAPI = fastapi.FastAPI
HTTPException = fastapi.HTTPException
Request = fastapi.Request
BaseModel = pydantic.BaseModel

logger = logging.getLogger("TrainRouter")


async def _probe_model_health(
    model_registry: ModelRegistry,
    model_addr: str,
    client: Any,
) -> None:
    try:
        resp = await client.get(f"{model_addr}/health")
        healthy = resp.status_code == 200
    except Exception:
        healthy = False
    await model_registry.update_health(model_addr, healthy)


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


class RouteRequest(BaseModel):
    api_key: str | None = None


class RegisterRequest(BaseModel):
    model_addr: str
    api_key: str
    name: str = ""


class UnregisterRequest(BaseModel):
    model_addr: str


def create_app(config: RouterConfig) -> FastAPI:
    model_registry = ModelRegistry()

    async def _poll_models(client: Any) -> None:
        while True:
            models = await model_registry.get_all()
            for model in models:
                await _probe_model_health(model_registry, model.model_addr, client)
            await asyncio.sleep(config.poll_interval)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info(
            "Train router starting — poll_interval=%.1fs",
            config.poll_interval,
        )
        health_client = httpx.AsyncClient(timeout=config.worker_health_timeout)
        app.state.model_registry = model_registry
        app.state.health_client = health_client
        poll_task = asyncio.create_task(_poll_models(health_client))
        try:
            yield
        finally:
            poll_task.cancel()
            try:
                await poll_task
            except asyncio.CancelledError:
                pass
            await health_client.aclose()
            logger.info("Train router shutting down")

    app = FastAPI(title="AReaL Train Router", lifespan=lifespan)
    app.state.model_registry = model_registry

    @app.get("/health")
    async def health():
        model_count = await model_registry.count()
        return {
            "status": "ok",
            "models": model_count,
        }

    @app.post("/route")
    async def route(body: RouteRequest, request: Request):
        _require_admin_key(request, config.admin_api_key)
        if body.api_key is None:
            raise HTTPException(status_code=422, detail="api_key required")
        if hmac.compare_digest(body.api_key, config.admin_api_key):
            raise HTTPException(
                status_code=400,
                detail="Admin key cannot be used for data-plane routing",
            )
        model_info = await model_registry.lookup_by_key(body.api_key)
        if model_info is None:
            raise HTTPException(status_code=404, detail="Unknown API key")
        if not model_info.is_healthy:
            raise HTTPException(status_code=503, detail="Pinned model is unhealthy")
        return {"model_addr": model_info.model_addr}

    @app.post("/register")
    async def register(body: RegisterRequest, request: Request):
        _require_admin_key(request, config.admin_api_key)
        await model_registry.register(body.model_addr, body.api_key, body.name)
        logger.info(
            "Model registered: %s",
            body.model_addr,
        )
        return {"status": "ok"}

    @app.post("/unregister")
    async def unregister(body: UnregisterRequest, request: Request):
        _require_admin_key(request, config.admin_api_key)
        await model_registry.deregister(body.model_addr)
        logger.info(
            "Model unregistered: %s",
            body.model_addr,
        )
        return {"status": "ok"}

    @app.get("/models")
    async def list_models(request: Request):
        _require_admin_key(request, config.admin_api_key)
        models = await model_registry.get_all()
        return {
            "models": [
                {
                    "model_addr": m.model_addr,
                    "api_key": m.api_key,
                    "name": m.name,
                    "is_healthy": m.is_healthy,
                    "registered_at": m.registered_at,
                }
                for m in models
            ]
        }

    return app
