# SPDX-License-Identifier: Apache-2.0

"""Agent Gateway — public-facing HTTP/WebSocket server."""

from __future__ import annotations

import hmac
import json
import traceback

import httpx
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect

from areal.utils import logging

from ..auth import admin_headers
from ..protocol import (
    FrameType,
    RequestFrame,
    RequestMethod,
    RunStatus,
    generate_run_id,
    make_complete_response,
    make_delta_event,
    make_failed_response,
    make_tool_call_event,
    parse_frame,
    serialize_frame,
)
from .config import GatewayConfig

logger = logging.getLogger("AgentGateway")


def _make_accepted_json(request_id: str, run_id: str) -> str:
    return json.dumps(
        {
            "type": FrameType.RES,
            "id": request_id,
            "ok": True,
            "payload": {"runId": run_id, "status": RunStatus.ACCEPTED},
        }
    )


def create_gateway_app(config: GatewayConfig) -> FastAPI:
    app = FastAPI(title="AReaL Agent Gateway")
    http_client = httpx.AsyncClient(timeout=config.forward_timeout)
    _auth_headers = admin_headers(config.admin_api_key)

    async def _route(session_key: str) -> str:
        resp = await http_client.post(
            f"{config.router_addr}/route",
            json={"session_key": session_key},
            headers=_auth_headers,
            timeout=config.router_timeout,
        )
        resp.raise_for_status()
        return resp.json()["data_proxy_addr"]

    async def _send_turn(
        data_proxy_addr: str,
        session_key: str,
        message: str,
        run_id: str,
        queue_mode: str,
        metadata: dict,
    ) -> dict:
        resp = await http_client.post(
            f"{data_proxy_addr}/session/{session_key}/turn",
            json={
                "message": message,
                "run_id": run_id,
                "queue_mode": queue_mode,
                "metadata": metadata,
            },
        )
        resp.raise_for_status()
        return resp.json()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket, token: str = Query(default="")):
        if not hmac.compare_digest(token, config.admin_api_key):
            await websocket.close(code=4001, reason="Invalid admin key")
            return
        await websocket.accept()
        logger.info("WebSocket connection accepted")

        try:
            while True:
                raw = await websocket.receive_text()
                frame = parse_frame(raw)

                if not isinstance(frame, RequestFrame):
                    await websocket.send_text(
                        serialize_frame(
                            make_failed_response("unknown", "", "Expected req frame")
                        )
                    )
                    continue

                if frame.method != RequestMethod.AGENT:
                    await websocket.send_text(
                        serialize_frame(
                            make_failed_response(
                                frame.id, "", f"Unsupported method: {frame.method}"
                            )
                        )
                    )
                    continue

                session_key = frame.session_key
                if not session_key:
                    await websocket.send_text(
                        serialize_frame(
                            make_failed_response(
                                frame.id, "", "Missing sessionKey in params"
                            )
                        )
                    )
                    continue

                run_id = generate_run_id()
                await websocket.send_text(_make_accepted_json(frame.id, run_id))

                try:
                    data_proxy_addr = await _route(session_key)
                    result = await _send_turn(
                        data_proxy_addr=data_proxy_addr,
                        session_key=session_key,
                        message=frame.message,
                        run_id=run_id,
                        queue_mode=frame.queue_mode.value,
                        metadata=frame.params,
                    )

                    for evt in result.get("events", []):
                        if evt.get("type") == "delta":
                            await websocket.send_text(
                                serialize_frame(
                                    make_delta_event(run_id, evt.get("text", ""))
                                )
                            )
                        elif evt.get("type") == "tool_call":
                            await websocket.send_text(
                                serialize_frame(
                                    make_tool_call_event(
                                        run_id,
                                        evt.get("name", ""),
                                        evt.get("args", ""),
                                    )
                                )
                            )

                    await websocket.send_text(
                        serialize_frame(
                            make_complete_response(
                                frame.id, run_id, result.get("summary", "")
                            )
                        )
                    )
                except Exception as exc:
                    logger.error(
                        "Run %s failed: %s\n%s", run_id, exc, traceback.format_exc()
                    )
                    await websocket.send_text(
                        serialize_frame(
                            make_failed_response(frame.id, run_id, str(exc))
                        )
                    )

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")
        except Exception:
            logger.exception("Unexpected error in WebSocket handler")

    @app.on_event("shutdown")
    async def shutdown():
        await http_client.aclose()

    return app
