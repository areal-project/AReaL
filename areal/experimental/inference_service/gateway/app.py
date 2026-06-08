# SPDX-License-Identifier: Apache-2.0

"""Inference Gateway — thin HTTP proxy with auth, routing, and forwarding.

The gateway holds only ``admin_api_key`` and ``router_addr``. All worker state,
session pinning, and routing strategies live in the Router service.
"""

from __future__ import annotations

import asyncio
import json
import traceback
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from areal.experimental.inference_service.gateway.auth import (
    extract_bearer_token,
    require_admin_key,
)
from areal.experimental.inference_service.gateway.config import GatewayConfig
from areal.experimental.inference_service.gateway.streaming import (
    RouterKeyRejectedError,
    RouterUnreachableError,
    _forwarding_headers,
    broadcast_to_workers,
    forward_request,
    forward_sse_stream,
    list_models_from_router,
    query_router,
    register_model_in_router,
    register_session_in_router,
    remove_model_from_router,
    resolve_worker_addr,
    revoke_session_in_router,
)
from areal.infra.utils.http import create_httpx_client
from areal.utils import logging

logger = logging.getLogger("InferenceGateway")


# =============================================================================
# Response models
# =============================================================================


class GatewayHealthResponse(BaseModel):
    status: str
    router_addr: str


class GatewayModelsResponse(BaseModel):
    models: list[str]


class BroadcastResultItem(BaseModel):
    worker_addr: str
    status: int
    ok: bool
    error: str | None = None


class BroadcastResponse(BaseModel):
    results: list[BroadcastResultItem]


def _router_error_response(exc: Exception) -> JSONResponse:
    """Convert router exceptions to HTTP responses."""
    if isinstance(exc, RouterUnreachableError):
        return JSONResponse({"error": str(exc)}, status_code=502)
    if isinstance(exc, RouterKeyRejectedError):
        status = 401 if exc.status_code == 404 else exc.status_code
        return JSONResponse({"error": exc.detail}, status_code=status)
    return JSONResponse({"error": str(exc)}, status_code=500)


def create_app(config: GatewayConfig) -> FastAPI:
    """Factory that creates the inference gateway FastAPI app."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.http_client = create_httpx_client(timeout=config.router_timeout)
        try:
            yield
        finally:
            await app.state.http_client.aclose()
            if _fallback_client is not None:
                await _fallback_client.aclose()

    app = FastAPI(title="AReaL Inference Gateway", lifespan=lifespan)

    # Fallback client for tests that bypass the lifespan.
    # NOTE: When testing without ASGI transport (which runs the lifespan),
    # this client will NOT be cleaned up automatically. Tests should prefer
    # using httpx.AsyncClient(transport=ASGITransport(app=app)) to ensure
    # proper lifespan management and client cleanup.
    _fallback_client: httpx.AsyncClient | None = None

    def _client() -> httpx.AsyncClient:
        nonlocal _fallback_client
        try:
            return app.state.http_client
        except AttributeError:
            if _fallback_client is None:
                _fallback_client = create_httpx_client(timeout=config.router_timeout)
            return _fallback_client

    # =========================================================================
    # Health
    # =========================================================================

    @app.get("/health", response_model=GatewayHealthResponse)
    async def health():
        return GatewayHealthResponse(status="ok", router_addr=config.router_addr)

    # =========================================================================
    # POST /chat/completions — admin OR session key, streaming or non-streaming
    # =========================================================================

    @app.post("/chat/completions")
    async def chat_completions(request: Request):
        token = extract_bearer_token(request)
        body = await request.body()
        headers = _forwarding_headers(dict(request.headers))

        model_name = None
        is_streaming = False
        try:
            body_json = json.loads(body)
            model_name = body_json.get("model")
            is_streaming = body_json.get("stream", False) or False
        except (json.JSONDecodeError, AttributeError):
            pass

        try:
            worker_addr = await query_router(
                config.router_addr,
                token,
                "/chat/completions",
                config.router_timeout,
                admin_api_key=config.admin_api_key,
                model=model_name,
                client=_client(),
            )
        except (RouterUnreachableError, RouterKeyRejectedError) as exc:
            return _router_error_response(exc)

        if is_streaming:
            return StreamingResponse(
                forward_sse_stream(
                    f"{worker_addr}/chat/completions",
                    body,
                    headers,
                    config.forward_timeout,
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        resp = await forward_request(
            f"{worker_addr}/chat/completions",
            body,
            headers,
            config.forward_timeout,
            client=_client(),
        )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type"),
        )

    @app.post("/register_model")
    async def register_model(request: Request):
        require_admin_key(request, config.admin_api_key)
        body = await request.json()
        model = body.get("model")
        url = body.get("url", "")
        api_key = body.get("api_key")
        data_proxy_addrs = body.get("data_proxy_addrs", [])
        if not model:
            return JSONResponse({"error": "model is required"}, status_code=400)
        try:
            result = await register_model_in_router(
                config.router_addr,
                model,
                url,
                api_key,
                data_proxy_addrs,
                config.admin_api_key,
                config.router_timeout,
                client=_client(),
            )
        except (RouterUnreachableError, RouterKeyRejectedError) as exc:
            return _router_error_response(exc)

        resolved_addrs = result.get("data_proxy_addrs", data_proxy_addrs)
        headers = _forwarding_headers(dict(request.headers))

        # Phase 3: parallelize data proxy registration
        async def _register_one(addr: str) -> httpx.Response:
            return await forward_request(
                f"{addr}/register_model",
                json.dumps(
                    {
                        "name": model,
                        "url": url,
                        "model": model,
                        "api_key": api_key,
                    }
                ).encode(),
                headers,
                config.forward_timeout,
                client=_client(),
            )

        responses = await asyncio.gather(
            *[_register_one(addr) for addr in resolved_addrs],
            return_exceptions=True,
        )
        # Check for failures after all tasks have completed
        failed = False
        errors = []
        for resp in responses:
            if isinstance(resp, Exception):
                logger.error("Data proxy registration raised: %s", resp)
                errors.append(str(resp))
                failed = True
            elif resp.status_code != 200:
                logger.error(
                    "Data proxy registration failed with status %d: %s",
                    resp.status_code,
                    resp.text,
                )
                errors.append(f"status {resp.status_code}: {resp.text}")
                failed = True
        if failed:
            await remove_model_from_router(
                config.router_addr,
                model,
                config.admin_api_key,
                config.router_timeout,
                client=_client(),
            )
            return JSONResponse(
                {"error": "Data proxy registration failed", "details": errors},
                status_code=502,
            )
        return result

    @app.get("/models")
    async def list_models(request: Request):
        require_admin_key(request, config.admin_api_key)
        try:
            names = await list_models_from_router(
                config.router_addr,
                config.admin_api_key,
                config.router_timeout,
                client=_client(),
            )
        except (RouterUnreachableError, RouterKeyRejectedError) as exc:
            return _router_error_response(exc)
        return GatewayModelsResponse(models=names)

    # =========================================================================
    # POST /rl/start_session — admin key ONLY, intercept response
    # =========================================================================

    @app.post("/rl/start_session")
    async def start_session(request: Request):
        token = require_admin_key(request, config.admin_api_key)

        try:
            worker_addr = await query_router(
                config.router_addr,
                token,
                "/rl/start_session",
                config.router_timeout,
                admin_api_key=config.admin_api_key,
                client=_client(),
            )
        except (RouterUnreachableError, RouterKeyRejectedError) as exc:
            return _router_error_response(exc)

        body = await request.body()
        headers = _forwarding_headers(dict(request.headers))

        resp = await forward_request(
            f"{worker_addr}/rl/start_session",
            body,
            headers,
            config.forward_timeout,
            client=_client(),
        )

        if resp.status_code == 201:
            try:
                resp_data = resp.json()
                group_id = resp_data["group_id"]
                sessions = resp_data.get("sessions", [])

                await register_session_in_router(
                    config.router_addr,
                    sessions,
                    worker_addr,
                    config.router_timeout,
                    admin_api_key=config.admin_api_key,
                    group_id=group_id,
                    client=_client(),
                )

                return JSONResponse(resp_data, status_code=201)
            except Exception as exc:
                logger.error("Failed to register session in router: %s", exc)
                traceback.print_exc()
                return JSONResponse(
                    {
                        "error": f"Session created on worker but router registration failed: {exc}"
                    },
                    status_code=502,
                )

        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type"),
        )

    # =========================================================================
    # POST /rl/set_reward — session key or admin key (HITL)
    # =========================================================================

    @app.post("/rl/set_reward")
    async def set_reward(request: Request):
        token = extract_bearer_token(request)
        body = await request.body()
        headers = _forwarding_headers(dict(request.headers))

        model = None
        try:
            body_json = json.loads(body)
            model = body_json.get("model")
        except (json.JSONDecodeError, AttributeError):
            pass

        try:
            worker_addr = await query_router(
                config.router_addr,
                token,
                "/rl/set_reward",
                config.router_timeout,
                admin_api_key=config.admin_api_key,
                model=model,
                client=_client(),
            )
        except (RouterUnreachableError, RouterKeyRejectedError) as exc:
            return _router_error_response(exc)

        resp = await forward_request(
            f"{worker_addr}/rl/set_reward",
            body,
            headers,
            config.forward_timeout,
            client=_client(),
        )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type"),
        )

    # =========================================================================
    # POST /pause_generation/{worker_id} — admin key ONLY, target single worker
    # =========================================================================

    @app.post("/pause_generation/{worker_id}", response_model=BroadcastResponse)
    async def pause_generation(worker_id: str, request: Request):
        require_admin_key(request, config.admin_api_key)
        try:
            worker_addr = await resolve_worker_addr(
                config.router_addr,
                config.admin_api_key,
                worker_id,
                config.router_timeout,
                client=_client(),
            )
        except (RouterUnreachableError, RouterKeyRejectedError) as exc:
            return _router_error_response(exc)

        body = await request.body()
        headers = _forwarding_headers(dict(request.headers))
        results = await broadcast_to_workers(
            [worker_addr], "/pause_generation", body, headers, client=_client()
        )
        return BroadcastResponse(results=[BroadcastResultItem(**r) for r in results])

    # =========================================================================
    # POST /continue_generation/{worker_id} — admin key ONLY, target single worker
    # =========================================================================

    @app.post("/continue_generation/{worker_id}", response_model=BroadcastResponse)
    async def continue_generation(worker_id: str, request: Request):
        require_admin_key(request, config.admin_api_key)
        try:
            worker_addr = await resolve_worker_addr(
                config.router_addr,
                config.admin_api_key,
                worker_id,
                config.router_timeout,
                client=_client(),
            )
        except (RouterUnreachableError, RouterKeyRejectedError) as exc:
            return _router_error_response(exc)

        body = await request.body()
        headers = _forwarding_headers(dict(request.headers))
        results = await broadcast_to_workers(
            [worker_addr], "/continue_generation", body, headers, client=_client()
        )
        return BroadcastResponse(results=[BroadcastResultItem(**r) for r in results])

    # =========================================================================
    # POST /release_memory_occupation/{worker_id} — admin key ONLY
    # =========================================================================

    @app.post(
        "/release_memory_occupation/{worker_id}", response_model=BroadcastResponse
    )
    async def release_memory_occupation(worker_id: str, request: Request):
        require_admin_key(request, config.admin_api_key)
        try:
            worker_addr = await resolve_worker_addr(
                config.router_addr,
                config.admin_api_key,
                worker_id,
                config.router_timeout,
                client=_client(),
            )
        except (RouterUnreachableError, RouterKeyRejectedError) as exc:
            return _router_error_response(exc)

        body = await request.body()
        headers = _forwarding_headers(dict(request.headers))
        results = await broadcast_to_workers(
            [worker_addr], "/release_memory_occupation", body, headers, client=_client()
        )
        return BroadcastResponse(results=[BroadcastResultItem(**r) for r in results])

    # =========================================================================
    # POST /resume_memory_occupation/{worker_id} — admin key ONLY
    # =========================================================================

    @app.post("/resume_memory_occupation/{worker_id}", response_model=BroadcastResponse)
    async def resume_memory_occupation(worker_id: str, request: Request):
        require_admin_key(request, config.admin_api_key)
        try:
            worker_addr = await resolve_worker_addr(
                config.router_addr,
                config.admin_api_key,
                worker_id,
                config.router_timeout,
                client=_client(),
            )
        except (RouterUnreachableError, RouterKeyRejectedError) as exc:
            return _router_error_response(exc)

        body = await request.body()
        headers = _forwarding_headers(dict(request.headers))
        results = await broadcast_to_workers(
            [worker_addr], "/resume_memory_occupation", body, headers, client=_client()
        )
        return BroadcastResponse(results=[BroadcastResultItem(**r) for r in results])

    # =========================================================================
    # POST /export_trajectories — admin key ONLY, route by session_ids
    # =========================================================================

    @app.post("/export_trajectories")
    async def export_trajectories(request: Request):
        require_admin_key(request, config.admin_api_key)
        body = await request.body()

        try:
            body_json = json.loads(body)
        except (json.JSONDecodeError, AttributeError):
            return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

        session_ids: list[str] = body_json.get("session_ids") or []
        group_id: str | None = body_json.get("group_id")

        if not session_ids:
            return JSONResponse({"error": "session_ids is required"}, status_code=400)

        try:
            worker_addr = await query_router(
                config.router_addr,
                timeout=config.router_timeout,
                session_id=session_ids[0],
                admin_api_key=config.admin_api_key,
                client=_client(),
            )
        except (RouterUnreachableError, RouterKeyRejectedError) as exc:
            return _router_error_response(exc)

        headers = _forwarding_headers(dict(request.headers))
        resp = await forward_request(
            f"{worker_addr}/export_trajectories",
            body,
            headers,
            config.forward_timeout,
            client=_client(),
        )

        if resp.status_code == 200 and group_id is not None:
            await revoke_session_in_router(
                config.router_addr,
                config.admin_api_key,
                group_id,
                timeout=config.router_timeout,
                client=_client(),
            )

        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type"),
        )

    # =========================================================================
    # POST /set_version/{worker_id} — admin key ONLY, target single worker
    # =========================================================================

    @app.post("/set_version/{worker_id}")
    async def set_version(worker_id: str, request: Request):
        require_admin_key(request, config.admin_api_key)
        try:
            worker_addr = await resolve_worker_addr(
                config.router_addr,
                config.admin_api_key,
                worker_id,
                config.router_timeout,
                client=_client(),
            )
        except (RouterUnreachableError, RouterKeyRejectedError) as exc:
            return _router_error_response(exc)

        body = await request.body()
        headers = _forwarding_headers(dict(request.headers))
        resp = await forward_request(
            f"{worker_addr}/set_version",
            body,
            headers,
            config.forward_timeout,
            client=_client(),
        )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type"),
        )

    # =========================================================================
    # GET /get_version/{worker_id} — admin key ONLY, target single worker
    # =========================================================================

    @app.get("/get_version/{worker_id}")
    async def get_version(worker_id: str, request: Request):
        require_admin_key(request, config.admin_api_key)
        try:
            worker_addr = await resolve_worker_addr(
                config.router_addr,
                config.admin_api_key,
                worker_id,
                config.router_timeout,
                client=_client(),
            )
        except (RouterUnreachableError, RouterKeyRejectedError) as exc:
            return _router_error_response(exc)

        try:
            resp = await _client().get(
                f"{worker_addr}/get_version",
                headers=_forwarding_headers(dict(request.headers)),
                timeout=config.forward_timeout,
            )
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type"),
            )
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=502)

    # =========================================================================
    # Compatibility aliases for RolloutCallback — map /callback/* to endpoints
    # =========================================================================
    # RolloutCallback uses /callback/* prefixed paths for generation control.
    # Gateway implements the actual handlers at unprefixed paths.  These aliases
    # register the SAME handler functions on both routes.
    # POST /callback/pause_generation/{worker_id} → pause_generation
    app.add_api_route(
        "/callback/pause_generation/{worker_id}",
        pause_generation,
        methods=["POST"],
    )

    # POST /callback/continue_generation/{worker_id} → continue_generation
    app.add_api_route(
        "/callback/continue_generation/{worker_id}",
        continue_generation,
        methods=["POST"],
    )

    # =========================================================================
    # OpenAI / OpenRouter compatibility aliases — /v1/* prefixed routes
    # =========================================================================
    app.add_api_route(
        "/v1/chat/completions",
        chat_completions,
        methods=["POST"],
    )
    app.add_api_route(
        "/v1/models",
        list_models,
        methods=["GET"],
    )

    return app
