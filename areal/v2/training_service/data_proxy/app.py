# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response as RawResponse

from areal.utils import logging
from areal.v2.training_service.data_proxy.config import TrainDataProxyConfig
from areal.v2.training_service.data_proxy.dispatcher import Dispatcher
from areal.v2.training_service.data_proxy.engine import register_engine_routes
from areal.v2.training_service.data_proxy.topology import discover_topology

logger = logging.getLogger("TrainDataProxy")


def _raw_json_response(content: bytes) -> RawResponse:
    return RawResponse(content=content, media_type="application/json")


def create_app(config: TrainDataProxyConfig) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info(
            "Train data proxy starting with %d workers", len(config.worker_addrs)
        )
        topology = await discover_topology(
            config.worker_addrs,
            timeout=min(config.request_timeout, 30.0),
        )
        dispatcher = Dispatcher(
            topology=topology, request_timeout=config.request_timeout
        )

        app.state.config = config
        app.state.topology = topology
        app.state.dispatcher = dispatcher
        yield
        await app.state.dispatcher.close()
        logger.info("Train data proxy shutting down")

    app = FastAPI(title="AReaL Train Data Proxy", lifespan=lifespan)
    app.state.training_condition = asyncio.Condition()
    app.state.training_held = False
    app.state.active_requests = 0

    @app.middleware("http")
    async def training_gate(request: Request, call_next):
        if request.url.path in {
            "/health",
            "/topology",
            "/hold_training",
            "/configure_workers",
            "/release_training",
        }:
            return await call_next(request)
        condition = app.state.training_condition
        async with condition:
            await condition.wait_for(lambda: not app.state.training_held)
            app.state.active_requests += 1
        try:
            return await call_next(request)
        finally:
            async with condition:
                app.state.active_requests -= 1
                condition.notify_all()

    def require_admin(request: Request) -> None:
        if request.headers.get("Authorization") != f"Bearer {config.admin_api_key}":
            raise HTTPException(status_code=403, detail="Invalid admin API key")

    @app.post("/hold_training")
    async def hold_training(request: Request):
        require_admin(request)
        async with app.state.training_condition:
            app.state.training_held = True
            await app.state.training_condition.wait_for(
                lambda: app.state.active_requests == 0
            )
        return {"status": "ok"}

    @app.post("/configure_workers")
    async def configure_workers(request: Request):
        require_admin(request)
        body = await request.json()
        addresses = body.get("worker_addrs")
        if (
            not isinstance(addresses, list)
            or not addresses
            or any(
                not isinstance(address, str)
                or not address.startswith(("http://", "https://"))
                for address in addresses
            )
            or len(set(addresses)) != len(addresses)
        ):
            raise HTTPException(
                status_code=422, detail="Distinct worker HTTP URLs are required"
            )
        async with app.state.training_condition:
            if not app.state.training_held or app.state.active_requests:
                raise HTTPException(
                    status_code=409, detail="Hold and drain training before rebinding"
                )
            topology = await discover_topology(
                addresses, timeout=min(config.request_timeout, 30.0)
            )
            old = app.state.dispatcher
            app.state.dispatcher = Dispatcher(
                topology=topology, request_timeout=config.request_timeout
            )
            app.state.topology = topology
            app.state.config = replace(app.state.config, worker_addrs=list(addresses))
            await old.close()
        return {"status": "ok"}

    @app.post("/release_training")
    async def release_training(request: Request):
        require_admin(request)
        async with app.state.training_condition:
            app.state.training_held = False
            app.state.training_condition.notify_all()
        return {"status": "ok"}

    register_engine_routes(app, _raw_json_response=_raw_json_response)

    return app
