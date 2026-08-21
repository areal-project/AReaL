# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from areal.utils import logging

if TYPE_CHECKING:
    from areal.v2.inference_service.sglang.rpc_proxy import RpcProxy

logger = logging.getLogger("AwexInferenceEndpoints")


def register_awex_endpoints(app: FastAPI, rpc_proxy: RpcProxy) -> None:
    """Register ``/awex/*`` weight-update endpoints on the SGLang FastAPI app.

    Each endpoint dispatches to all scheduler processes via the
    :class:`RpcProxy`, which sends :class:`RpcReqInput` over ZMQ.
    The :class:`AwexSchedulerBridge` handles the methods and packs return
    values into ``RpcReqOutput.message``.
    """

    @app.post("/awex/report_weight_meta")
    async def report_weight_meta() -> JSONResponse:
        try:
            result = rpc_proxy.collective_rpc_with_result("awex_report_weight_meta")
            return JSONResponse(content={"status": "ok", "meta": result})
        except Exception as e:
            logger.error("Failed to report weight meta: %s", e)
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.get("/awex/report_parallelism")
    async def report_parallelism() -> JSONResponse:
        try:
            result = rpc_proxy.collective_rpc_with_result("awex_report_parallelism")
            if not isinstance(result, dict):
                err_msg = f"Expected dict from awex_report_parallelism, but got {type(result).__name__}"
                logger.error(err_msg)
                return JSONResponse(status_code=500, content={"error": err_msg})
            return JSONResponse(content=result)
        except Exception as e:
            logger.error("Failed to report parallelism: %s", e)
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.post("/awex/init_weights_update_group")
    async def init_weights_update_group(request: Request) -> JSONResponse:
        try:
            data = await request.json()
            rpc_proxy.collective_rpc("awex_init_weights_update_group", **data)
            return JSONResponse(content={"status": "ok"})
        except Exception as e:
            logger.error("Failed to init weights update group: %s", e)
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.post("/awex/update_weights")
    async def update_weights(request: Request) -> JSONResponse:
        try:
            data = await request.json()
            version = data.get("version", 0)
            rpc_proxy.collective_rpc("awex_execute_weight_update", version=version)
            return JSONResponse(content={"status": "ok", "version": version})
        except Exception as e:
            logger.error("Failed to update weights: %s", e)
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.post("/awex/batch_isend_irecv")
    async def batch_isend_irecv(request: Request) -> JSONResponse:
        try:
            data = await request.json()
            rpc_proxy.collective_rpc("awex_batch_isend_irecv", **data)
            return JSONResponse(content={"status": "ok"})
        except Exception as e:
            logger.error("Failed batch_isend_irecv: %s", e)
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.post("/awex/debug/get_parameters")
    async def get_parameters(request: Request) -> JSONResponse:
        try:
            data = await request.json()
            rpc_proxy.collective_rpc("awex_get_parameters", **data)
            return JSONResponse(content={"status": "ok"})
        except Exception as e:
            logger.error("Failed to get parameters: %s", e)
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.post("/awex/debug/randomize_parameters")
    async def randomize_parameters() -> JSONResponse:
        try:
            rpc_proxy.collective_rpc("awex_randomize_parameters")
            return JSONResponse(content={"status": "ok"})
        except Exception as e:
            logger.error("Failed to randomize parameters: %s", e)
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.post("/awex/init_colocate_weight_update")
    async def init_colocate_weight_update(request: Request) -> JSONResponse:
        try:
            data = await request.json()
            rpc_proxy.collective_rpc("awex_init_colocate_weight_update", **data)
            return JSONResponse(content={"status": "ok"})
        except Exception as e:
            logger.error("Failed to init colocate weight update: %s", e)
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.post("/awex/execute_colocate_weight_update")
    async def execute_colocate_weight_update(request: Request) -> JSONResponse:
        try:
            data = await request.json()
            version = data.get("version", 0)
            rpc_proxy.collective_rpc(
                "awex_execute_colocate_weight_update", version=version
            )
            return JSONResponse(content={"status": "ok", "version": version})
        except Exception as e:
            logger.error("Failed to execute colocate weight update: %s", e)
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.post("/awex/release_memory")
    async def release_memory(request: Request) -> JSONResponse:
        try:
            data = await request.json()
            tags = data.get("tags")
            rpc_proxy.collective_rpc("awex_release_memory", tags=tags)
            return JSONResponse(content={"status": "ok"})
        except Exception as e:
            logger.error("Failed to release memory: %s", e)
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.post("/awex/resume_memory")
    async def resume_memory(request: Request) -> JSONResponse:
        try:
            data = await request.json()
            tags = data.get("tags")
            rpc_proxy.collective_rpc("awex_resume_memory", tags=tags)
            return JSONResponse(content={"status": "ok"})
        except Exception as e:
            logger.error("Failed to resume memory: %s", e)
            return JSONResponse(status_code=500, content={"error": str(e)})
