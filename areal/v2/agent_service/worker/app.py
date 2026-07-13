# SPDX-License-Identifier: Apache-2.0

"""Agent Worker — stateless HTTP server for agent execution."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from areal.utils import logging
from areal.utils.dynamic_import import import_from_string

from ..auth import is_source_visible_default_admin_key, verify_admin_key
from ..protocol import PASSTHROUGH_HEADER, QueueMode
from ..session_keys import session_key_sha256, validate_session_key
from ..types import (
    AgentRequest,
    AgentResponse,
    AgentRunnable,
    StreamResponse,
)

logger = logging.getLogger("AgentWorker")


class _CollectingEmitter:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def emit_delta(self, text: str) -> None:
        self.events.append({"type": "delta", "text": text})

    async def emit_tool_call(self, name: str, args: str) -> None:
        self.events.append({"type": "tool_call", "name": name, "args": args})

    async def emit_tool_result(self, name: str, result: str) -> None:
        self.events.append({"type": "tool_result", "name": name, "result": result})


def create_worker_app(
    agent_cls_path: str,
    **agent_kwargs: Any,
) -> FastAPI:
    """Create the backward-compatible standalone Worker application.

    Every keyword remains an Agent constructor argument.  In particular, this
    factory does not reserve ``worker_hop_api_key`` from existing plugins.
    Controller-managed deployments use
    :func:`create_worker_app_with_hop_auth` instead.
    """

    return _create_worker_app(
        agent_cls_path,
        worker_hop_api_key="",
        agent_kwargs=agent_kwargs,
    )


def create_worker_app_with_hop_auth(
    agent_cls_path: str,
    worker_hop_api_key: str,
    **agent_kwargs: Any,
) -> FastAPI:
    """Create a Worker whose state-changing HTTP routes require a pair key."""

    if type(worker_hop_api_key) is not str:
        raise TypeError("worker_hop_api_key must be a string")
    if not worker_hop_api_key.strip():
        raise ValueError("worker_hop_api_key must not be blank")
    if is_source_visible_default_admin_key(worker_hop_api_key):
        raise ValueError("worker_hop_api_key must not use a source-visible default key")
    return _create_worker_app(
        agent_cls_path,
        worker_hop_api_key=worker_hop_api_key,
        agent_kwargs=agent_kwargs,
    )


def _create_worker_app(
    agent_cls_path: str,
    *,
    worker_hop_api_key: str,
    agent_kwargs: dict[str, Any],
) -> FastAPI:
    app = FastAPI(title="AReaL Agent Worker")

    cls = import_from_string(agent_cls_path)
    agent: AgentRunnable = cls(**agent_kwargs)
    if not isinstance(agent, AgentRunnable):
        raise TypeError(
            f"Loaded class {agent_cls_path} does not satisfy AgentRunnable protocol "
            f"(missing async def run(request, *, emitter) method)"
        )
    logger.info("Agent loaded: %s", agent_cls_path)

    async def authorize_worker_hop(http_request: Request) -> None:
        if worker_hop_api_key:
            await verify_admin_key(
                http_request.headers.get("Authorization", ""),
                expected_key=worker_hop_api_key,
            )

    def validated_session_key(value: object) -> str:
        try:
            return validate_session_key(value)
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/internal/auth-check")
    async def auth_check(http_request: Request):
        """Prove that this Worker enforces the configured pair credential."""

        if not worker_hop_api_key:
            raise HTTPException(
                status_code=503,
                detail="Worker hop authentication is not enabled",
            )
        await authorize_worker_hop(http_request)
        return {"status": "ok", "worker_hop_auth": True}

    async def close_agent_session(session_key: object, http_request: Request):
        await authorize_worker_hop(http_request)
        session_key = validated_session_key(session_key)
        close_fn = getattr(agent, "close_session", None)
        if close_fn is not None:
            await close_fn(session_key)
        return {
            "status": "ok",
            "session_key_sha256": session_key_sha256(session_key),
        }

    @app.post("/sessions/close")
    async def close_session(body: dict[str, Any], http_request: Request):
        """Close an exact session identity carried as data, never URL syntax."""

        return await close_agent_session(body.get("session_key"), http_request)

    @app.post("/session/{session_key}/close", deprecated=True)
    async def close_session_legacy(session_key: str, http_request: Request):
        """Compatibility route; internal callers use ``/sessions/close``."""

        return await close_agent_session(session_key, http_request)

    @app.on_event("shutdown")
    async def shutdown():
        close_all_fn = getattr(agent, "close_all_sessions", None)
        if close_all_fn is not None:
            await close_all_fn()

    @app.post("/run")
    async def run(body: dict[str, Any], http_request: Request):
        """Single agent entry point for every protocol and streaming mode.

        Calls the agent's ``run`` and relays whichever shape it returns:

        - :class:`StreamResponse` — raw passthrough; ``status_code`` /
          ``headers`` / ``body`` are forwarded untouched so the caller gets the
          upstream's exact wire format (e.g. SSE chat completions).  The
          response carries the :data:`PASSTHROUGH_HEADER` marker so the
          DataProxy relays it verbatim instead of parsing it — this works even
          for a *non-streaming* passthrough whose body is ``application/json``.
        - :class:`AgentResponse` — structured JSON ``{summary, metadata,
          events}`` (``application/json``); the DataProxy rebuilds history from
          ``events``.
        """
        await authorize_worker_hop(http_request)
        session_key = validated_session_key(body.get("session_key"))
        request = AgentRequest(
            message=body.get("message", ""),
            session_key=session_key,
            run_id=body.get("run_id", ""),
            history=body.get("history", []),
            queue_mode=QueueMode(body.get("queue_mode", "collect")),
            metadata=body.get("metadata", {}),
        )

        emitter = _CollectingEmitter()

        try:
            response: AgentResponse | StreamResponse = await agent.run(
                request, emitter=emitter
            )
        except Exception as exc:
            logger.exception("Agent run failed (session=%s)", request.session_key)
            return JSONResponse(
                {"error": {"message": str(exc), "type": type(exc).__name__}},
                status_code=500,
            )

        if isinstance(response, StreamResponse):
            # Drop hop-by-hop / length headers that would conflict with chunked
            # relaying; FastAPI/uvicorn set framing headers themselves.
            headers = {
                k: v
                for k, v in response.headers.items()
                if k.lower()
                not in ("content-length", "transfer-encoding", "connection")
            }
            # Mark the turn as raw-passthrough so the DataProxy relays it
            # verbatim regardless of its Content-Type.
            headers[PASSTHROUGH_HEADER] = "1"
            return StreamingResponse(
                response.body,
                status_code=response.status_code,
                headers=headers,
                media_type=response.headers.get("content-type"),
            )

        return {**asdict(response), "events": emitter.events}

    return app
