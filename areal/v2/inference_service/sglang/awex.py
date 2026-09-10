# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import time
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

    def _operation_timeout(data: dict) -> float | None:
        timeout = data.pop("rpc_timeout_s", None)
        ttl = data.pop("operation_ttl_s", None)
        if ttl is not None:
            ttl = max(0.0, float(ttl))
            data["operation_expiry_monotonic"] = time.monotonic() + ttl
            timeout = ttl if timeout is None else min(float(timeout), ttl)
        return None if timeout is None else float(timeout)

    @app.post("/awex/report_weight_meta")
    async def report_weight_meta() -> JSONResponse:
        try:
            result = await rpc_proxy.collective_rpc_with_result(
                "awex_report_weight_meta"
            )
            return JSONResponse(content={"status": "ok", "meta": result})
        except Exception as e:
            logger.error("Failed to report weight meta: %s", e)
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.get("/awex/report_parallelism")
    async def report_parallelism() -> JSONResponse:
        try:
            result = await rpc_proxy.collective_rpc_with_result(
                "awex_report_parallelism"
            )
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
            timeout = _operation_timeout(data)
            await rpc_proxy.collective_rpc(
                "awex_init_weights_update_group", timeout_s=timeout, **data
            )
            return JSONResponse(content={"status": "ok"})
        except Exception as e:
            logger.error("Failed to init weights update group: %s", e)
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.post("/awex/update_weights")
    async def update_weights(request: Request) -> JSONResponse:
        try:
            data = await request.json()
            pair_name = data["pair_name"]
            version = data.get("version", 0)
            timeout = _operation_timeout(data)
            await rpc_proxy.collective_rpc(
                "awex_execute_weight_update",
                timeout_s=timeout,
                pair_name=pair_name,
                version=version,
            )
            return JSONResponse(content={"status": "ok", "version": version})
        except Exception as e:
            logger.error("Failed to update weights: %s", e)
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.post("/awex/batch_isend_irecv")
    async def batch_isend_irecv(request: Request) -> JSONResponse:
        try:
            data = await request.json()
            timeout = _operation_timeout(data)
            await rpc_proxy.collective_rpc(
                "awex_batch_isend_irecv", timeout_s=timeout, **data
            )
            return JSONResponse(content={"status": "ok"})
        except Exception as e:
            logger.error("Failed batch_isend_irecv: %s", e)
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.post("/awex/teardown")
    async def teardown(request: Request) -> JSONResponse:
        try:
            data = await request.json()
            pair_name = data["pair_name"]
            operation_id = data.get("operation_id", "")
            timeout = _operation_timeout(data)
            await rpc_proxy.collective_rpc(
                "awex_teardown_weight_update_group",
                timeout_s=timeout,
                pair_name=pair_name,
                operation_id=operation_id,
            )
            return JSONResponse(content={"status": "ok"})
        except Exception as e:
            logger.error("Failed to teardown weights update group: %s", e)
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.post("/awex/debug/get_parameters")
    async def get_parameters(request: Request) -> JSONResponse:
        try:
            data = await request.json()
            await rpc_proxy.collective_rpc("awex_get_parameters", **data)
            return JSONResponse(content={"status": "ok"})
        except Exception as e:
            logger.error("Failed to get parameters: %s", e)
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.post("/awex/debug/randomize_parameters")
    async def randomize_parameters() -> JSONResponse:
        try:
            await rpc_proxy.collective_rpc("awex_randomize_parameters")
            return JSONResponse(content={"status": "ok"})
        except Exception as e:
            logger.error("Failed to randomize parameters: %s", e)
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.post("/awex/init_colocate_weight_update")
    async def init_colocate_weight_update(request: Request) -> JSONResponse:
        try:
            data = await request.json()
            timeout = _operation_timeout(data)
            await rpc_proxy.collective_rpc(
                "awex_init_colocate_weight_update", timeout_s=timeout, **data
            )
            return JSONResponse(content={"status": "ok"})
        except Exception as e:
            logger.error("Failed to init colocate weight update: %s", e)
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.post("/awex/execute_colocate_weight_update")
    async def execute_colocate_weight_update(request: Request) -> JSONResponse:
        try:
            data = await request.json()
            pair_name = data["pair_name"]
            version = data.get("version", 0)
            timeout = _operation_timeout(data)
            await rpc_proxy.collective_rpc(
                "awex_execute_colocate_weight_update",
                timeout_s=timeout,
                pair_name=pair_name,
                version=version,
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
            timeout = _operation_timeout(data)
            await rpc_proxy.collective_rpc(
                "awex_release_memory", timeout_s=timeout, tags=tags
            )
            return JSONResponse(content={"status": "ok"})
        except Exception as e:
            logger.error("Failed to release memory: %s", e)
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.post("/awex/resume_memory")
    async def resume_memory(request: Request) -> JSONResponse:
        try:
            data = await request.json()
            tags = data.get("tags")
            timeout = _operation_timeout(data)
            await rpc_proxy.collective_rpc(
                "awex_resume_memory", timeout_s=timeout, tags=tags
            )
            return JSONResponse(content={"status": "ok"})
        except Exception as e:
            logger.error("Failed to resume memory: %s", e)
            return JSONResponse(status_code=500, content={"error": str(e)})
