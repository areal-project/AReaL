# SPDX-License-Identifier: Apache-2.0

"""Data Proxy — stateful session proxy between Gateway and Worker."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from fastapi import FastAPI

from areal.utils import logging

from .config import DataProxyConfig

logger = logging.getLogger("AgentDataProxy")


@dataclass
class _SessionData:
    history: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    last_active: float = field(default_factory=time.monotonic)


def create_data_proxy_app(config: DataProxyConfig) -> FastAPI:
    app = FastAPI(title="AReaL Data Proxy")
    sessions: dict[str, _SessionData] = {}
    http_client = httpx.AsyncClient(timeout=config.request_timeout)

    async def _close_worker_session(session_key: str) -> None:
        try:
            await http_client.post(
                f"{config.worker_addr}/session/{session_key}/close", timeout=5
            )
        except Exception:
            logger.debug("Failed to close worker session %s", session_key)

    async def _reap_idle_sessions() -> None:
        while True:
            await asyncio.sleep(60)
            now = time.monotonic()
            stale = [
                k
                for k, s in sessions.items()
                if now - s.last_active > config.session_timeout
            ]
            for k in stale:
                del sessions[k]
                await _close_worker_session(k)
            if stale:
                logger.info("Reaped %d idle sessions", len(stale))

    @app.on_event("startup")
    async def startup():
        app.state.reaper_task = asyncio.create_task(_reap_idle_sessions())

    @app.on_event("shutdown")
    async def shutdown():
        await http_client.aclose()

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "active_sessions": len(sessions),
            "worker_addr": config.worker_addr,
        }

    @app.post("/session/{session_key}/turn")
    async def turn(session_key: str, body: dict[str, Any]):
        session = sessions.get(session_key)
        if session is None:
            session = _SessionData()
            sessions[session_key] = session

        message = body.get("message", "")
        run_id = body.get("run_id", "")
        queue_mode = body.get("queue_mode", "collect")
        metadata = body.get("metadata", {})

        worker_request = {
            "message": message,
            "session_key": session_key,
            "run_id": run_id,
            "history": session.history.copy(),
            "queue_mode": queue_mode,
            "metadata": metadata,
        }

        resp = await http_client.post(f"{config.worker_addr}/run", json=worker_request)
        resp.raise_for_status()
        result = resp.json()

        session.history.append({"role": "user", "content": message})

        call_counter = 0
        for evt in result.get("events", []):
            if evt.get("type") == "tool_call":
                call_id = f"call_{evt.get('name', '')}_{run_id}_{call_counter}"
                call_counter += 1
                session.history.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": evt.get("name", ""),
                                    "arguments": evt.get("args", ""),
                                },
                            }
                        ],
                    }
                )
            elif evt.get("type") == "tool_result":
                result_call_id = (
                    f"call_{evt.get('name', '')}_{run_id}_{call_counter - 1}"
                    if call_counter > 0
                    else f"call_{evt.get('name', '')}_{run_id}_0"
                )
                session.history.append(
                    {
                        "role": "tool",
                        "tool_call_id": result_call_id,
                        "content": evt.get("result", ""),
                    }
                )

        summary = result.get("summary", "")
        if summary:
            session.history.append({"role": "assistant", "content": summary})

        session.last_active = time.monotonic()
        return result

    @app.post("/session/{session_key}/close")
    async def close_session(session_key: str):
        sessions.pop(session_key, None)
        await _close_worker_session(session_key)
        return {"status": "ok"}

    @app.get("/session/{session_key}/history")
    async def get_history(session_key: str):
        session = sessions.get(session_key)
        if session is None:
            return {"history": []}
        return {"history": session.history}

    return app
