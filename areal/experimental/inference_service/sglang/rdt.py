# SPDX-License-Identifier: Apache-2.0
"""RDT HTTP endpoints for IW weight update.

Reference: areal.experimental.inference_service.sglang.awex
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from areal.utils import logging

if TYPE_CHECKING:
    from areal.experimental.inference_service.sglang.rpc_proxy import RpcProxy

logger = logging.getLogger("RDTIWEndpoints")


def register_rdt_endpoints(app: FastAPI, rpc_proxy: RpcProxy) -> None:
    """Register ``/rdt/*`` weight-update endpoints on IW's FastAPI app.

    Each endpoint dispatches to all scheduler processes via RpcProxy,
    using collective_rpc_with_result or collective_rpc.

    Args:
        app: FastAPI application
        rpc_proxy: RpcProxy for scheduler subprocess communication
    """

    @app.get("/rdt/report_parallelism")
    async def report_parallelism() -> JSONResponse:
        """Report IW parallelism strategy for TransferPlan building."""
        try:
            result = rpc_proxy.collective_rpc_with_result("rdt_report_parallelism")
            if not isinstance(result, dict):
                err_msg = f"Expected dict from rdt_report_parallelism, got {type(result).__name__}"
                logger.error(err_msg)
                return JSONResponse(status_code=500, content={"error": err_msg})
            return JSONResponse(content=result)
        except Exception as e:
            logger.error("Failed to report parallelism: %s", e)
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.post("/rdt/report_weight_meta")
    async def report_weight_meta() -> JSONResponse:
        """Report IW weight metadata for TransferPlan building."""
        try:
            result = rpc_proxy.collective_rpc_with_result("rdt_report_weight_meta")
            return JSONResponse(content={"status": "ok", "meta": result})
        except Exception as e:
            logger.error("Failed to report weight meta: %s", e)
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.post("/rdt/init_weight_update_group")
    async def init_weight_update_group(request: Request) -> JSONResponse:
        """Initialize RDT weight update group.

        Args passed via JSON body:
            pair_name: TW-IW pair identifier
            kv_store_url: Gateway KV store URL
            tw_actor_bytes_b64_list: Base64-encoded TW actor handle bytes
            infer_world_size: Total IW world size
            train_world_size: Total TW world size
            num_engines: Number of IW engines
            transfer_rank: IW's transfer rank
        """
        try:
            data = await request.json()
            rpc_proxy.collective_rpc("rdt_init_weight_update_group", **data)
            return JSONResponse(content={"status": "ok"})
        except Exception as e:
            logger.error("Failed to init RDT weight update group: %s", e)
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.post("/rdt/execute_weight_update")
    async def execute_weight_update(request: Request) -> JSONResponse:
        """Execute RDT weight update - pull from TW via Ray RPC.

        Args passed via JSON body:
            version: Weight version number (optional, default 0)
        """
        try:
            data = await request.json()
            version = data.get("version", 0)
            rpc_proxy.collective_rpc("rdt_execute_weight_update", version=version)
            return JSONResponse(content={"status": "ok", "version": version})
        except Exception as e:
            logger.error("Failed to execute RDT weight update: %s", e)
            return JSONResponse(status_code=500, content={"error": str(e)})

    # ---------------------------------------------------------------------------
    # Debug endpoints for E2E testing
    # ---------------------------------------------------------------------------

    @app.post("/rdt/debug/randomize_parameters")
    async def randomize_parameters() -> JSONResponse:
        """Randomize model parameters for testing."""
        try:
            rpc_proxy.collective_rpc("rdt_randomize_parameters")
            return JSONResponse(content={"status": "ok"})
        except Exception as e:
            logger.error("Failed to randomize parameters: %s", e)
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.post("/rdt/debug/get_parameters")
    async def get_parameters(request: Request) -> JSONResponse:
        """Save parameters to disk for validation."""
        try:
            data = await request.json()
            rpc_proxy.collective_rpc("rdt_get_parameters", **data)
            return JSONResponse(content={"status": "ok"})
        except Exception as e:
            logger.error("Failed to get parameters: %s", e)
            return JSONResponse(status_code=500, content={"error": str(e)})
