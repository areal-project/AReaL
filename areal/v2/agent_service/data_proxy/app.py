# SPDX-License-Identifier: Apache-2.0

"""Data Proxy — stateful session proxy between Gateway and Worker."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from areal.utils import logging

from ..protocol import PASSTHROUGH_HEADER
from .config import DataProxyConfig

logger = logging.getLogger("AgentDataProxy")


@dataclass
class _SessionData:
    history: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    last_active: float = field(default_factory=time.monotonic)
    reward: float | None = None
    # Per-session inference routing for self-evolution.  Holds
    # ``{"base_url", "api_key", "model"}`` where ``api_key`` is the
    # ``sk-sess-*`` the **caller** obtained itself and passed on the turn
    # (``session_api_key``).  The Agent Service never talks to the training
    # side — it only forwards these fields to the worker so the agent routes
    # its LLM calls through the inference gateway under that key.  Cached on
    # the first turn that carries them so later turns of a multi-turn session
    # can omit them.
    inference: dict[str, Any] | None = None


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

    def _resolve_inference(
        session: _SessionData, body: dict[str, Any], metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """Forward caller-supplied inference routing to the worker metadata.

        Self-evolution decouples the Agent Service from the training side: the
        **caller** mints its own per-session ``sk-sess-*`` (e.g. via its own
        ``/rl/start_session``) and passes it on the turn body.  This proxy never
        contacts the inference/training side — it merely caches the routing
        handle on the session and injects it as ``metadata['areal_inference']``
        so the agent routes its LLM calls through the inference gateway.

        The turn opts in **by the presence of the routing fields** (no separate
        flag); the required pair is:

            ``inf_base_url``     — inference gateway base URL the agent's LLM
                                    calls go to (required).
            ``session_api_key``  — the caller-minted ``sk-sess-*`` (required).
            ``inf_model``        — model id the agent should request (default "").

        ``inf_model`` is optional and never triggers self-evolution on its own;
        only the required pair does.  The handle is cached on the first turn that
        carries it, so a multi-turn session may send these fields once and omit
        them afterwards.
        """
        base_url = (body.get("inf_base_url") or "").rstrip("/")
        api_key = body.get("session_api_key") or ""
        if base_url or api_key:
            if not base_url or not api_key:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "self-evolution requires both 'inf_base_url' and "
                        "'session_api_key' in the turn body"
                    ),
                )
            session.inference = {
                "base_url": base_url,
                "api_key": api_key,
                "model": body.get("inf_model", "") or "",
            }

        if session.inference is not None:
            metadata = {**metadata, "areal_inference": dict(session.inference)}
        return metadata

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
        """Single turn endpoint for every protocol and streaming mode.

        The worker's ``/run`` decides the shape of the turn and signals it via
        the :data:`PASSTHROUGH_HEADER` response header:

        - header absent — a structured turn (``application/json``).  The body is
          read fully, conversation history is rebuilt from the emitted
          ``events``, and the JSON is returned to the caller (this backs
          ``/v1/responses`` and the WebSocket path).
        - header == ``"1"`` — a raw-passthrough turn.  The body is relayed
          **byte-for-byte** without parsing, so the caller gets the upstream's
          exact wire format (this backs ``/v1/chat/completions``, streaming or
          not).  Keying on the marker rather than ``Content-Type`` means a
          *non-streaming* passthrough — itself ``application/json`` — is still
          relayed verbatim instead of being mistaken for a structured turn.  No
          history is kept on this path; stateful callers rely on *route
          affinity* (a stable ``session_key`` pins every turn to this same
          DataProxy/Worker so the agent reuses its own state).
        """
        session = sessions.get(session_key)
        if session is None:
            session = _SessionData()
            sessions[session_key] = session

        message = body.get("message", "")
        run_id = body.get("run_id", "")
        queue_mode = body.get("queue_mode", "collect")
        metadata = body.get("metadata", {})

        # Self-evolution (opt-in): when the turn carries inference-routing
        # fields, forward them so the agent's LLM calls flow through the
        # inference gateway under the caller's own ``sk-sess-*``.  No-op for a
        # plain turn that omits them.
        metadata = _resolve_inference(session, body, metadata)

        worker_request = {
            "message": message,
            "session_key": session_key,
            "run_id": run_id,
            "history": session.history.copy(),
            "queue_mode": queue_mode,
            "metadata": metadata,
        }

        # Open the worker stream so we can inspect status/Content-Type before
        # deciding whether to parse (structured) or relay (raw passthrough).
        req = http_client.build_request(
            "POST", f"{config.worker_addr}/run", json=worker_request
        )
        resp = await http_client.send(req, stream=True)
        session.last_active = time.monotonic()

        is_passthrough = resp.headers.get(PASSTHROUGH_HEADER) == "1"

        if is_passthrough:
            # Raw-passthrough turn: relay the worker body verbatim, keep no
            # history (parsing the stream would fight the byte-exact relay).
            async def _relay():
                try:
                    async for chunk in resp.aiter_raw():
                        yield chunk
                finally:
                    await resp.aclose()
                    session.last_active = time.monotonic()

            headers = {
                k: v
                for k, v in resp.headers.items()
                if k.lower()
                not in (
                    "content-length",
                    "transfer-encoding",
                    "connection",
                    PASSTHROUGH_HEADER,
                )
            }
            return StreamingResponse(
                _relay(),
                status_code=resp.status_code,
                headers=headers,
                media_type=resp.headers.get("content-type") or None,
            )

        # Structured turn: read the full JSON body, then rebuild history.
        await resp.aread()
        status_code = resp.status_code
        result = resp.json()
        await resp.aclose()

        if status_code >= 400:
            # Worker reported an error; forward it without touching history.
            return JSONResponse(result, status_code=status_code)

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
        return JSONResponse(result, status_code=status_code)

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
