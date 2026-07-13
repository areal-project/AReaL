# SPDX-License-Identifier: Apache-2.0

"""OpenResponses HTTP bridge — translates POST /v1/responses to DataProxy turns."""

from __future__ import annotations

import hmac
import json
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from areal.utils import logging

from ..auth import (
    DEFAULT_ADMIN_API_KEY,
    admin_headers,
    is_source_visible_default_admin_key,
    make_admin_dependency,
    verify_admin_key,
)
from ..memory_transport import (
    CHAT_REQUEST_METADATA_KEY,
    MEMORY_ASSIGNMENT_PIN_FIELD,
    MEMORY_CONTROL_AUTHORIZED_FIELD,
)
from ..protocol import generate_run_id
from ..streaming import CleanupStreamingResponse

logger = logging.getLogger("AgentBridge")

# Request/response header carrying the session identifier.  A client sends it to
# pin a multi-turn conversation onto the same DataProxy/Worker (route affinity);
# both bridges always echo the resolved key back on the response so the client
# can reuse it on the next request — including a key derived or randomly minted
# server-side when the client sent none.
SESSION_KEY_HEADER = "X-AReaL-Session-Key"


def _has_admin_authorization(request: Request, expected: dict[str, str]) -> bool:
    return hmac.compare_digest(
        request.headers.get("Authorization", ""),
        expected["Authorization"],
    )


def _dedicated_memory_control_headers(
    admin_api_key: str,
    memory_control_api_key: str,
) -> dict[str, str]:
    if type(admin_api_key) is not str:
        raise TypeError("admin_api_key must be a string")
    if not admin_api_key.strip():
        raise ValueError("admin_api_key must not be blank")
    if type(memory_control_api_key) is not str:
        raise TypeError("memory_control_api_key must be a string")
    if not memory_control_api_key:
        return {}
    if not memory_control_api_key.strip():
        raise ValueError("memory_control_api_key must not be blank")
    if is_source_visible_default_admin_key(memory_control_api_key):
        raise ValueError(
            "memory_control_api_key must not use a source-visible default key"
        )
    if hmac.compare_digest(memory_control_api_key, admin_api_key):
        raise ValueError(
            "memory_control_api_key must differ from the external admin_api_key"
        )
    return admin_headers(memory_control_api_key)


class AgentBridge(ABC):
    @abstractmethod
    async def handle_request(self, request: Request) -> Any: ...


class OpenResponsesBridge(AgentBridge):
    def __init__(
        self,
        router_addr: str,
        admin_api_key: str = DEFAULT_ADMIN_API_KEY,
        memory_control_api_key: str = "",
    ) -> None:
        self._router_addr = router_addr
        self._admin_headers = admin_headers(admin_api_key)
        self._memory_control_headers = _dedicated_memory_control_headers(
            admin_api_key,
            memory_control_api_key,
        )
        self._memory_control_enabled = bool(
            self._memory_control_headers
            and not is_source_visible_default_admin_key(admin_api_key)
        )
        self._http = httpx.AsyncClient(timeout=600.0)

    async def close(self) -> None:
        await self._http.aclose()

    async def handle_request(self, request: Request) -> Any:
        body = await request.json()
        memory_control_authorized = bool(
            self._memory_control_enabled
            and _has_admin_authorization(request, self._admin_headers)
        )
        if MEMORY_ASSIGNMENT_PIN_FIELD in body and not self._memory_control_enabled:
            return JSONResponse(
                {
                    "error": {
                        "message": (
                            "Memory assignment transport requires a non-default "
                            "external admin key and a dedicated internal hop key"
                        ),
                        "type": "server_error",
                    }
                },
                status_code=503,
            )

        input_items: list[dict[str, Any]] = body.get("input", [])
        instructions: str = body.get("instructions", "")
        model: str = body.get("model", "")
        user: str = body.get("user", "")

        # An explicit X-AReaL-Session-Key pins the conversation to one
        # DataProxy/Worker directly; only when it is absent do we fall back to
        # deriving the key from ``user`` (which is then required for affinity).
        explicit_session_key = request.headers.get(SESSION_KEY_HEADER)
        if not explicit_session_key and not user:
            return JSONResponse(
                {
                    "error": {
                        "message": (
                            "either the 'X-AReaL-Session-Key' header or the "
                            "'user' field is required for session affinity"
                        ),
                        "type": "invalid_request",
                    }
                },
                status_code=400,
            )

        message = self._extract_message(input_items, instructions)
        session_key = explicit_session_key or self._derive_session_key(user, model)
        run_id = generate_run_id()
        response_id = f"resp-{uuid.uuid4().hex[:12]}"

        metadata = {
            "input": input_items,
            "instructions": instructions,
            "tools": body.get("tools", []),
            "model": model,
            "idempotencyKey": response_id,
            **body.get("metadata", {}),
        }

        try:
            route_resp = await self._http.post(
                f"{self._router_addr}/route",
                json={"session_key": session_key},
                headers=self._admin_headers,
            )
            route_resp.raise_for_status()
            data_proxy_addr = route_resp.json()["data_proxy_addr"]

            turn_body: dict[str, Any] = {
                "message": message,
                "run_id": run_id,
                "queue_mode": "collect",
                "metadata": metadata,
                MEMORY_CONTROL_AUTHORIZED_FIELD: memory_control_authorized,
            }
            # Opt-in self-evolution: forward the caller-supplied inference
            # routing fields when present so the DataProxy hands the agent a
            # ``sk-sess-*`` (which the caller minted itself) and the agent's LLM
            # calls flow through AReaL's inference service.  The DataProxy opts
            # the turn in by the presence of these fields; absent them the turn
            # is plain.  The Agent Service never contacts the training side.
            for key in (
                "inf_base_url",
                "inf_model",
                "session_api_key",
            ):
                if key in body:
                    turn_body[key] = body[key]
            if MEMORY_ASSIGNMENT_PIN_FIELD in body:
                turn_body[MEMORY_ASSIGNMENT_PIN_FIELD] = body[
                    MEMORY_ASSIGNMENT_PIN_FIELD
                ]

            turn_resp = await self._http.post(
                f"{data_proxy_addr}/session/{session_key}/turn",
                json=turn_body,
                headers=(
                    self._memory_control_headers if memory_control_authorized else None
                ),
            )
            if 400 <= turn_resp.status_code < 500:
                try:
                    error_payload = turn_resp.json()
                except ValueError:
                    error_payload = {
                        "error": {
                            "message": turn_resp.text,
                            "type": "invalid_request",
                        }
                    }
                return JSONResponse(
                    error_payload,
                    status_code=turn_resp.status_code,
                    headers={SESSION_KEY_HEADER: session_key},
                )
            turn_resp.raise_for_status()
            result = turn_resp.json()

            output_items = self._build_output_items(result)
            response_metadata = result.get("metadata", {})

            if body.get("stream"):
                # Streaming /v1/responses: the agent ran the structured turn to
                # completion (collect); re-encode the collected output as an
                # OpenAI Responses-format SSE event stream.
                return StreamingResponse(
                    self._responses_sse(
                        response_id, model, output_items, response_metadata
                    ),
                    media_type="text/event-stream",
                    headers={SESSION_KEY_HEADER: session_key},
                )

            return JSONResponse(
                {
                    "id": response_id,
                    "object": "response",
                    "status": "completed",
                    "output": output_items,
                    "model": model,
                    "metadata": response_metadata,
                },
                headers={SESSION_KEY_HEADER: session_key},
            )
        except Exception as exc:
            logger.error("OpenResponses request failed: %s", exc)
            return JSONResponse(
                {"error": {"message": str(exc), "type": "server_error"}},
                status_code=500,
                headers={SESSION_KEY_HEADER: session_key},
            )

    @staticmethod
    def _extract_message(input_items: list[dict[str, Any]], instructions: str) -> str:
        parts: list[str] = []
        if instructions:
            parts.append(instructions)
        for item in input_items:
            if item.get("type") == "message":
                content = item.get("content", "")
                if isinstance(content, list):
                    for block in content:
                        if (
                            isinstance(block, dict)
                            and block.get("type") == "input_text"
                        ):
                            parts.append(block.get("text", ""))
                elif isinstance(content, str):
                    parts.append(content)
            elif item.get("type") == "function_call_output":
                parts.append(f"[tool result] {item.get('output', '')}")
        return "\n".join(parts)

    @staticmethod
    def _build_output_items(result: dict[str, Any]) -> list[dict[str, Any]]:
        """Translate a structured turn result into OpenAI Responses output items.

        A non-empty ``summary`` becomes an assistant ``message`` item; each
        ``tool_call`` event becomes a ``function_call`` item.
        """
        output_items: list[dict[str, Any]] = []
        summary = result.get("summary", "")
        if summary:
            output_items.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": summary}],
                }
            )
        for evt in result.get("events", []):
            if evt.get("type") == "tool_call":
                output_items.append(
                    {
                        "type": "function_call",
                        "name": evt.get("name", ""),
                        "arguments": evt.get("args", ""),
                    }
                )
        return output_items

    @staticmethod
    async def _responses_sse(
        response_id: str,
        model: str,
        output_items: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> AsyncIterator[bytes]:
        """Emit collected output items as an OpenAI Responses-format SSE stream.

        This is a re-encoding shim, not incremental generation: the structured
        turn already ran to completion, so the whole text is sent in one
        ``response.output_text.delta``.  The terminal ``response.completed``
        event carries the same full response object the non-streaming path
        returns, so a client can rely on either.
        """

        def _event(payload: dict[str, Any]) -> bytes:
            return f"data: {json.dumps(payload)}\n\n".encode()

        yield _event(
            {
                "type": "response.created",
                "response": {
                    "id": response_id,
                    "object": "response",
                    "status": "in_progress",
                    "model": model,
                },
            }
        )
        for item in output_items:
            if item.get("type") != "message":
                continue
            for block in item.get("content", []):
                text = block.get("text", "")
                if not text:
                    continue
                yield _event({"type": "response.output_text.delta", "delta": text})
                yield _event({"type": "response.output_text.done", "text": text})
        yield _event(
            {
                "type": "response.completed",
                "response": {
                    "id": response_id,
                    "object": "response",
                    "status": "completed",
                    "model": model,
                    "output": output_items,
                    "metadata": metadata,
                },
            }
        )

    @staticmethod
    def _derive_session_key(user: str, model: str) -> str:
        if user:
            return f"agent:{model or 'default'}:{user}"
        return f"agent:{model or 'default'}:{uuid.uuid4().hex[:8]}"


def mount_bridge(
    app: FastAPI,
    bridge: OpenResponsesBridge,
    admin_api_key: str = DEFAULT_ADMIN_API_KEY,
) -> None:
    auth = make_admin_dependency(admin_api_key)

    @app.post("/v1/responses", dependencies=[Depends(auth)])
    async def responses_endpoint(request: Request):
        return await bridge.handle_request(request)

    @app.on_event("shutdown")
    async def shutdown_bridge():
        await bridge.close()


# The DataProxy injects the sanitized original ``/v1/chat/completions`` request
# under :data:`CHAT_REQUEST_METADATA_KEY`.  Raw-passthrough agents read it from
# ``request.metadata`` and replay it, while Agent Service control fields travel
# separately and can never leak into that upstream request.
_CHAT_CONTROL_FIELDS = (
    MEMORY_ASSIGNMENT_PIN_FIELD,
    MEMORY_CONTROL_AUTHORIZED_FIELD,
    "inf_base_url",
    "inf_model",
    "session_api_key",
)


class ChatCompletionsBridge(AgentBridge):
    """OpenAI-compatible ``/v1/chat/completions`` → DataProxy raw stream.

    Translates a chat-completions request into a DataProxy ``turn`` and relays
    the worker/agent response **byte-for-byte** (typically SSE), so any
    OpenAI-compatible upstream can call the gateway exactly as it would call the
    backing agent directly — no client change, exact wire format preserved.  The
    DataProxy uses the Worker's ``x-areal-passthrough`` response marker to tell a
    raw stream apart from a structured turn, so both protocols share the one
    ``turn`` endpoint.

    The original request body, after removing Agent Service control fields, is
    forwarded as a trusted top-level control payload.  DataProxy injects it into
    Worker ``metadata`` under :data:`CHAT_REQUEST_METADATA_KEY` for the agent to
    replay verbatim.

    **Session model.**  ``/v1/chat/completions`` is a stateless protocol: the
    client carries the full ``messages`` history on every request, so the
    framework does not store conversation history for this path.  What the
    framework *does* provide is **route affinity** — a stable ``session_key``
    makes the Router pin all of that session's requests to the same
    DataProxy/Worker, letting the Worker-side agent reuse its per-session state
    (sandbox, agent instance, KV cache).  The key is resolved per request by
    :meth:`_resolve_session_key`:

    1. explicit ``X-AReaL-Session-Key`` request header, else
    2. derived from the OpenAI ``user`` field (``chat:{model}:{user}``).

    One of the two is **required** (mirroring :class:`OpenResponsesBridge`); a
    request carrying neither is rejected with ``400`` rather than silently
    minted a random key, so route affinity is always explicit and a multi-turn
    caller can never be split across Workers by accident.

    The resolved key is always echoed back on the ``X-AReaL-Session-Key``
    response header so the caller can pin subsequent turns to it.
    """

    def __init__(
        self,
        router_addr: str,
        admin_api_key: str = DEFAULT_ADMIN_API_KEY,
        memory_control_api_key: str = "",
    ) -> None:
        self._router_addr = router_addr
        self._admin_api_key = admin_api_key
        self._admin_headers = admin_headers(admin_api_key)
        self._memory_control_headers = _dedicated_memory_control_headers(
            admin_api_key,
            memory_control_api_key,
        )
        self._memory_control_enabled = bool(
            self._memory_control_headers
            and not is_source_visible_default_admin_key(admin_api_key)
        )
        self._http = httpx.AsyncClient(timeout=600.0)

    async def close(self) -> None:
        await self._http.aclose()

    async def handle_request(self, request: Request) -> Any:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                {"error": {"message": "invalid JSON body", "type": "invalid_request"}},
                status_code=400,
            )

        assignment_pin_present = MEMORY_ASSIGNMENT_PIN_FIELD in body
        assignment_pin = body.get(MEMORY_ASSIGNMENT_PIN_FIELD)
        memory_control_authorized = bool(
            self._memory_control_enabled
            and _has_admin_authorization(request, self._admin_headers)
        )
        if assignment_pin_present and not memory_control_authorized:
            # Ordinary Chat Completions stays OpenAI-compatible and ungated,
            # but choosing a Memory assignment is a privileged control-plane
            # operation.  Reject it before routing so an unauthenticated caller
            # cannot select another tenant's otherwise valid active assignment.
            await verify_admin_key(
                request.headers.get("Authorization", ""),
                expected_key=self._admin_api_key,
            )
        if assignment_pin_present and not self._memory_control_enabled:
            return JSONResponse(
                {
                    "error": {
                        "message": (
                            "Memory assignment transport requires a non-default "
                            "external admin key and a dedicated internal hop key"
                        ),
                        "type": "server_error",
                    }
                },
                status_code=503,
            )
        upstream_body = dict(body)
        for key in _CHAT_CONTROL_FIELDS:
            upstream_body.pop(key, None)
        messages = upstream_body.get("messages", [])
        session_key = self._resolve_session_key(request, body)
        if session_key is None:
            return JSONResponse(
                {
                    "error": {
                        "message": (
                            "either the 'X-AReaL-Session-Key' header or the "
                            "'user' field is required for session affinity"
                        ),
                        "type": "invalid_request",
                    }
                },
                status_code=400,
            )
        run_id = generate_run_id()
        message = self._extract_last_user_text(messages)
        # Memory transport is a control-plane field, not part of the OpenAI
        # request replayed by a raw-passthrough agent.
        metadata: dict[str, object] = {}

        try:
            route_resp = await self._http.post(
                f"{self._router_addr}/route",
                json={"session_key": session_key},
                headers=self._admin_headers,
            )
            route_resp.raise_for_status()
            data_proxy_addr = route_resp.json()["data_proxy_addr"]
        except Exception as exc:
            logger.error("ChatCompletions routing failed: %s", exc)
            return JSONResponse(
                {"error": {"message": str(exc), "type": "server_error"}},
                status_code=502,
                headers={SESSION_KEY_HEADER: session_key},
            )

        # Open the DataProxy stream manually so its status/headers are known
        # before we build the StreamingResponse, then relay the body verbatim.
        turn_body: dict[str, Any] = {
            "message": message,
            "run_id": run_id,
            "queue_mode": "collect",
            "metadata": metadata,
            CHAT_REQUEST_METADATA_KEY: upstream_body,
            MEMORY_CONTROL_AUTHORIZED_FIELD: memory_control_authorized,
        }
        # Opt-in self-evolution (same contract as OpenResponsesBridge): forward
        # the caller-supplied inference-routing fields when present so the
        # agent's LLM calls flow through AReaL's inference service under the
        # caller's own ``sk-sess-*``.  The DataProxy opts the turn in by the
        # presence of these fields; absent them the stream is forwarded
        # byte-for-byte exactly as before.
        for key in (
            "inf_base_url",
            "inf_model",
            "session_api_key",
        ):
            if key in body:
                turn_body[key] = body[key]
        if assignment_pin_present:
            turn_body[MEMORY_ASSIGNMENT_PIN_FIELD] = assignment_pin

        req = self._http.build_request(
            "POST",
            f"{data_proxy_addr}/session/{session_key}/turn",
            json=turn_body,
            headers=(
                self._memory_control_headers if memory_control_authorized else None
            ),
        )
        try:
            resp = await self._http.send(req, stream=True)
        except Exception as exc:
            logger.error("ChatCompletions upstream stream failed: %s", exc)
            return JSONResponse(
                {"error": {"message": str(exc), "type": "server_error"}},
                status_code=502,
                headers={SESSION_KEY_HEADER: session_key},
            )

        headers = {
            k: v
            for k, v in resp.headers.items()
            if k.lower() not in ("content-length", "transfer-encoding", "connection")
        }
        headers[SESSION_KEY_HEADER] = session_key
        return CleanupStreamingResponse(
            resp.aiter_raw(),
            cleanup=resp.aclose,
            cleanup_task_name=(f"areal-chat-downstream-cleanup:{session_key}"),
            status_code=resp.status_code,
            headers=headers,
            media_type=resp.headers.get("content-type"),
        )

    @staticmethod
    def _resolve_session_key(request: Request, body: dict[str, Any]) -> str | None:
        """Resolve the session key, in priority order (see class docstring).

        Explicit ``X-AReaL-Session-Key`` header wins so a client can pin a
        conversation onto one Worker; otherwise the OpenAI ``user`` field is
        derived into a key (mirroring :class:`OpenResponsesBridge`).  With
        neither, return ``None`` so the caller can reject the request with
        ``400`` instead of minting an implicit random key.
        """
        explicit = request.headers.get(SESSION_KEY_HEADER)
        if explicit:
            return explicit
        user = body.get("user", "")
        if user:
            model = body.get("model") or "default"
            return f"chat:{model}:{user}"
        return None

    @staticmethod
    def _extract_last_user_text(messages: list[dict[str, Any]]) -> str:
        for item in reversed(messages):
            if item.get("role") != "user":
                continue
            content = item.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = [
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                return "".join(parts)
        return ""


def mount_chat_bridge(app: FastAPI, bridge: ChatCompletionsBridge) -> None:
    """Mount the chat-completions bridge.

    Ordinary chat requests are deliberately **not** admin-gated: upstreams call
    ``/v1/chat/completions`` exactly as they call the backing agent today (no
    AReaL admin key).  A request that selects ``memory_assignment_pin`` must
    carry the admin key.  Every later turn on that Memory-bound session must
    also carry it, even when the immutable pin is omitted for reuse; the
    DataProxy rejects an untrusted per-turn marker.  The internal gateway →
    router ``/route`` hop still carries the admin header.
    """

    @app.post("/v1/chat/completions")
    async def chat_completions_endpoint(request: Request):
        return await bridge.handle_request(request)

    @app.on_event("shutdown")
    async def shutdown_chat_bridge():
        await bridge.close()
