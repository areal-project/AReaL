# SPDX-License-Identifier: Apache-2.0

"""Agent Worker — stateless HTTP server for agent execution."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from areal.utils import logging
from areal.utils.dynamic_import import import_from_string

from ..protocol import QueueMode
from ..types import AgentRequest, AgentResponse, AgentRunnable, TrainingContext

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
    app = FastAPI(title="AReaL Agent Worker")

    cls = import_from_string(agent_cls_path)
    agent: AgentRunnable = cls(**agent_kwargs)
    if not isinstance(agent, AgentRunnable):
        raise TypeError(
            f"Loaded class {agent_cls_path} does not satisfy AgentRunnable protocol "
            f"(missing async def run(request, *, emitter) method)"
        )
    logger.info("Agent loaded: %s", agent_cls_path)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/session/{session_key}/episode/start")
    async def episode_start(session_key: str, body: dict[str, Any]):
        hook = getattr(agent, "on_episode_start", None)
        if hook is None:
            return {"status": "noop"}
        ctx = TrainingContext(
            session_id=body.get("session_id", session_key),
            llm_base_url=body.get("llm_base_url", ""),
            llm_api_key=body.get("llm_api_key", ""),
            llm_model=body.get("llm_model", ""),
            extras=body.get("extras", {}),
        )
        try:
            await hook(session_key, ctx)
        except Exception as exc:
            logger.exception("on_episode_start failed (session=%s)", session_key)
            return JSONResponse(
                {"error": {"message": str(exc), "type": type(exc).__name__}},
                status_code=500,
            )
        return {"status": "ok"}

    @app.post("/session/{session_key}/episode/end")
    async def episode_end(session_key: str, body: dict[str, Any]):
        hook = getattr(agent, "on_episode_end", None)
        if hook is None:
            return {"status": "noop"}
        reward = body.get("reward")
        try:
            await hook(session_key, reward)
        except Exception as exc:
            logger.exception("on_episode_end failed (session=%s)", session_key)
            return JSONResponse(
                {"error": {"message": str(exc), "type": type(exc).__name__}},
                status_code=500,
            )
        return {"status": "ok"}

    @app.post("/session/{session_key}/close")
    async def close_session(session_key: str):
        close_fn = getattr(agent, "close_session", None)
        if close_fn is not None:
            await close_fn(session_key)
        return {"status": "ok"}

    @app.on_event("shutdown")
    async def shutdown():
        close_all_fn = getattr(agent, "close_all_sessions", None)
        if close_all_fn is not None:
            await close_all_fn()

    @app.post("/run")
    async def run(body: dict[str, Any]):
        request = AgentRequest(
            message=body.get("message", ""),
            session_key=body.get("session_key", ""),
            run_id=body.get("run_id", ""),
            history=body.get("history", []),
            queue_mode=QueueMode(body.get("queue_mode", "collect")),
            metadata=body.get("metadata", {}),
        )

        emitter = _CollectingEmitter()

        try:
            response: AgentResponse = await agent.run(request, emitter=emitter)
        except Exception as exc:
            logger.exception("Agent run failed (session=%s)", request.session_key)
            return JSONResponse(
                {"error": {"message": str(exc), "type": type(exc).__name__}},
                status_code=500,
            )

        return {**asdict(response), "events": emitter.events}

    return app
