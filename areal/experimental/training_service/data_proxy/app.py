# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import Response as RawResponse

from areal.experimental.training_service.data_proxy.config import TrainDataProxyConfig
from areal.experimental.training_service.data_proxy.dispatcher import Dispatcher
from areal.experimental.training_service.data_proxy.engine import register_engine_routes
from areal.experimental.training_service.data_proxy.topology import discover_topology
from areal.utils import logging

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
        await dispatcher.close()
        logger.info("Train data proxy shutting down")

    app = FastAPI(title="AReaL Train Data Proxy", lifespan=lifespan)
    register_engine_routes(app, _raw_json_response=_raw_json_response)

    return app
