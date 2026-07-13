# SPDX-License-Identifier: Apache-2.0

"""Agent Router — session-affine routing service."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse

from areal.utils import logging

from ..auth import make_admin_dependency
from ..session_keys import validate_session_key
from .config import RouterConfig

logger = logging.getLogger("AgentRouter")


def create_router_app(config: RouterConfig) -> FastAPI:
    app = FastAPI(title="AReaL Agent Router")
    auth = make_admin_dependency(config.admin_api_key)

    registered_proxies: list[str] = []
    session_map: dict[str, str] = {}
    rr_idx = 0
    lock = asyncio.Lock()

    def _session_key_from_body(body: dict[str, Any]) -> str:
        try:
            return validate_session_key(body.get("session_key"))
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "registered_proxies": len(registered_proxies),
            "active_sessions": len(session_map),
        }

    @app.post("/register", dependencies=[Depends(auth)])
    async def register(body: dict[str, Any]):
        addr = body["addr"]
        async with lock:
            if addr not in registered_proxies:
                registered_proxies.append(addr)
                logger.info(
                    "Registered DataProxy: %s (total=%d)", addr, len(registered_proxies)
                )
        return {"status": "ok"}

    @app.post("/unregister", dependencies=[Depends(auth)])
    async def unregister(body: dict[str, Any]):
        addr = body["addr"]
        async with lock:
            if addr in registered_proxies:
                registered_proxies.remove(addr)
                stale = [k for k, v in session_map.items() if v == addr]
                for k in stale:
                    del session_map[k]
                logger.info(
                    "Unregistered DataProxy: %s (removed %d sessions)", addr, len(stale)
                )
        return {"status": "ok"}

    @app.post("/route", dependencies=[Depends(auth)])
    async def route(body: dict[str, Any]):
        nonlocal rr_idx
        session_key = _session_key_from_body(body)

        async with lock:
            if session_key in session_map:
                return {"data_proxy_addr": session_map[session_key]}

            if not registered_proxies:
                return JSONResponse(
                    {"error": "No DataProxy registered"}, status_code=503
                )

            addr = registered_proxies[rr_idx % len(registered_proxies)]
            rr_idx += 1
            session_map[session_key] = addr
            logger.info("Routed session %s → %s", session_key, addr)

        return {"data_proxy_addr": addr}

    @app.post("/remove_session", dependencies=[Depends(auth)])
    async def remove_session(body: dict[str, Any]):
        session_key = _session_key_from_body(body)
        async with lock:
            session_map.pop(session_key, None)
        return {"status": "ok"}

    return app
