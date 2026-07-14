# SPDX-License-Identifier: Apache-2.0

"""Data Proxy — stateful session proxy between Gateway and Worker."""

from __future__ import annotations

import asyncio
import time
from contextlib import suppress
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from areal.utils import logging

from ..auth import admin_headers, verify_admin_key
from ..memory_transport import (
    AREAL_INFERENCE_METADATA_KEY,
    CHAT_REQUEST_METADATA_KEY,
    MEMORY_ASSIGNMENT_PIN_FIELD,
    MEMORY_CONTROL_AUTHORIZED_FIELD,
    MemoryAssignmentPinWireV1,
    MemoryPinTransportError,
    MemorySessionPinCache,
    copy_user_metadata,
    inject_memory_assignment_pin,
)
from ..protocol import PASSTHROUGH_HEADER
from ..session_keys import session_key_sha256, validate_session_key
from ..streaming import CleanupStreamingResponse
from .config import DataProxyConfig

logger = logging.getLogger("AgentDataProxy")

_CHAT_CONTROL_FIELDS = (
    MEMORY_ASSIGNMENT_PIN_FIELD,
    MEMORY_CONTROL_AUTHORIZED_FIELD,
    "inf_base_url",
    "inf_model",
    "session_api_key",
)


class _SessionSecurityMode(StrEnum):
    ORDINARY = "ordinary"
    MEMORY = "memory"


@dataclass
class _SessionData:
    history: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    last_active: float = field(default_factory=time.monotonic)
    reward: float | None = None
    # Fixed by the first admitted turn.  A caller must close the whole
    # incarnation before changing between ordinary Agent state and a
    # principal/grant-gated Memory session; otherwise DataProxy history and
    # routing state could be smuggled across the security boundary.  The
    # future Worker runtime must independently bind its own state/incarnation.
    security_mode: _SessionSecurityMode | None = None
    # Per-session inference routing for self-evolution.  Holds
    # ``{"base_url", "api_key", "model"}`` where ``api_key`` is the
    # ``sk-sess-*`` the **caller** obtained itself and passed on the turn
    # (``session_api_key``).  The Agent Service never talks to the training
    # side — it only forwards these fields to the worker so the agent routes
    # its LLM calls through the inference gateway under that key.  Cached on
    # the first turn that carries them so later turns of a multi-turn session
    # can omit them.
    inference: dict[str, Any] | None = None
    lifecycle_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    active_turns: int = 0
    active_turns_drained: asyncio.Event = field(
        default_factory=asyncio.Event,
        repr=False,
    )
    closing: bool = False

    def __post_init__(self) -> None:
        self.active_turns_drained.set()


_INFERENCE_OMITTED = object()


def create_data_proxy_app(config: DataProxyConfig) -> FastAPI:
    app = FastAPI(title="AReaL Data Proxy")
    sessions: dict[str, _SessionData] = {}
    session_close_tasks: dict[str, asyncio.Task[bool]] = {}
    memory_pin_cache = MemorySessionPinCache()
    app.state.memory_pin_cache = memory_pin_cache
    http_client = httpx.AsyncClient(timeout=config.request_timeout)
    worker_hop_headers = (
        admin_headers(config.worker_hop_api_key) if config.worker_hop_api_key else {}
    )

    async def _close_worker_session(session_key: str) -> bool:
        try:
            session_key = validate_session_key(session_key)
            response = await http_client.post(
                f"{config.worker_addr}/sessions/close",
                json={"session_key": session_key},
                headers=worker_hop_headers or None,
                timeout=5,
            )
            used_legacy_endpoint = response.status_code in {404, 405}
            if used_legacy_endpoint:
                # Rolling upgrades may temporarily pair a new DataProxy with
                # an old Worker.  A validated key is safe in one path segment;
                # fall back only when the fixed endpoint is absent, never for
                # authorization or lifecycle failures.
                response = await http_client.post(
                    f"{config.worker_addr}/session/{session_key}/close",
                    headers=worker_hop_headers or None,
                    timeout=5,
                )
            response.raise_for_status()
            receipt = response.json()
            exact_receipt = {
                "status": "ok",
                "session_key_sha256": session_key_sha256(session_key),
            }
            valid_receipt = receipt == exact_receipt or (
                used_legacy_endpoint and receipt == {"status": "ok"}
            )
            if not valid_receipt:
                raise ValueError("Worker returned a mismatched session close receipt")
        except Exception:
            logger.warning("Failed to close worker session %s", session_key)
            return False
        return True

    def _validate_session_key(session_key: object) -> str:
        try:
            return validate_session_key(session_key)
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    def _parse_inference(body: dict[str, Any]) -> dict[str, Any] | object:
        """Validate optional routing without mutating a live session.

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
        base_url_value = body.get("inf_base_url", "")
        api_key = body.get("session_api_key", "")
        model = body.get("inf_model", "")
        if any(type(value) is not str for value in (base_url_value, api_key, model)):
            raise HTTPException(
                status_code=400,
                detail="self-evolution routing fields must be strings",
            )
        base_url = base_url_value.rstrip("/")
        if base_url or api_key:
            if not base_url or not api_key:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "self-evolution requires both 'inf_base_url' and "
                        "'session_api_key' in the turn body"
                    ),
                )
            return {
                "base_url": base_url,
                "api_key": api_key,
                "model": model,
            }
        return _INFERENCE_OMITTED

    async def _release_turn(session: _SessionData) -> None:
        async with session.lifecycle_lock:
            if session.active_turns <= 0:  # pragma: no cover - internal invariant
                raise RuntimeError("session turn lease underflow")
            session.active_turns -= 1
            session.last_active = time.monotonic()
            if session.active_turns == 0:
                session.active_turns_drained.set()

    async def _close_response_and_release(
        response: httpx.Response | None,
        session: _SessionData,
    ) -> None:
        """Close a Worker response without letting close failure leak a lease."""

        try:
            if response is not None:
                await response.aclose()
        finally:
            await _release_turn(session)

    async def _finish_response(
        response: httpx.Response | None,
        session: _SessionData,
    ) -> None:
        """Run response cleanup in a task that caller cancellation cannot kill."""

        cleanup = asyncio.create_task(
            _close_response_and_release(response, session),
            name="areal-data-proxy-response-cleanup",
        )
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            # ``shield`` leaves cleanup running.  Consume a later exception so
            # an upstream close failure does not become an unobserved task.
            def _consume_result(task: asyncio.Task[None]) -> None:
                if task.cancelled():
                    return
                with suppress(Exception):
                    task.result()

            cleanup.add_done_callback(_consume_result)
            raise

    def _parse_chat_request(body: dict[str, Any]) -> dict[str, object] | None:
        """Validate and sanitize the trusted raw-chat control payload."""

        if CHAT_REQUEST_METADATA_KEY not in body:
            return None
        value = body[CHAT_REQUEST_METADATA_KEY]
        if type(value) is not dict or any(type(key) is not str for key in value):
            raise HTTPException(
                status_code=400,
                detail=f"'{CHAT_REQUEST_METADATA_KEY}' must be a JSON object",
            )
        result = dict(value)
        # Defense in depth: Agent Service control fields must never be replayed
        # verbatim to an external OpenAI-compatible upstream.
        for key in _CHAT_CONTROL_FIELDS:
            result.pop(key, None)
        return result

    def _parse_memory_control_authorized(body: dict[str, Any]) -> bool:
        value = body.get(MEMORY_CONTROL_AUTHORIZED_FIELD, False)
        if type(value) is not bool:
            raise HTTPException(
                status_code=400,
                detail=f"'{MEMORY_CONTROL_AUTHORIZED_FIELD}' must be a boolean",
            )
        return value

    async def _admit_turn(
        session_key: str,
        *,
        submitted_pin_present: bool,
        submitted_pin: MemoryAssignmentPinWireV1 | None,
        inference: dict[str, Any] | object,
        user_metadata: dict[str, object],
        chat_request: dict[str, object] | None,
        memory_control_authorized: bool,
    ) -> tuple[_SessionData, dict[str, object], list[dict[str, Any]]]:
        """Atomically bind transport state and acquire one turn lease."""

        while True:
            session = sessions.get(session_key)
            if session is None:
                session = sessions.setdefault(session_key, _SessionData())
            async with session.lifecycle_lock:
                if sessions.get(session_key) is not session:
                    continue
                if session.closing:
                    raise HTTPException(status_code=409, detail="session is closing")

                # The pin cache is a transport CAS, not an authorization store.
                # Reject a submitted pin before it can win that CAS.  For an
                # omitted pin, resolve and authorize its reuse under this same
                # lifecycle lock so close/rebind cannot race the decision.
                if submitted_pin_present and not memory_control_authorized:
                    raise HTTPException(
                        status_code=403,
                        detail="Memory assignment use requires trusted ingress",
                    )

                # Reject an authorized ordinary→Memory upgrade before the
                # submitted pin can win the transport CAS.  Authentication is
                # checked first so an untrusted caller cannot use the status
                # code as a session-mode oracle.
                if (
                    session.security_mode is _SessionSecurityMode.ORDINARY
                    and submitted_pin_present
                ):
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            "session security mode cannot change; close the "
                            "session before rebinding"
                        ),
                    )

                assignment_pin = (
                    memory_pin_cache.resolve(session_key, submitted_pin)
                    if submitted_pin_present
                    else memory_pin_cache.resolve(session_key)
                )
                if assignment_pin is not None and not memory_control_authorized:
                    raise HTTPException(
                        status_code=403,
                        detail="Memory assignment use requires trusted ingress",
                    )
                requested_mode = (
                    _SessionSecurityMode.MEMORY
                    if assignment_pin is not None
                    else _SessionSecurityMode.ORDINARY
                )
                if (
                    session.security_mode is not None
                    and session.security_mode is not requested_mode
                ):
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            "session security mode cannot change; close the "
                            "session before rebinding"
                        ),
                    )
                resolved_metadata = inject_memory_assignment_pin(
                    user_metadata,
                    assignment_pin,
                )
                effective_inference = (
                    session.inference if inference is _INFERENCE_OMITTED else inference
                )
                if effective_inference is not None:
                    if type(effective_inference) is not dict:  # pragma: no cover
                        raise RuntimeError("validated inference routing is not a dict")
                    resolved_metadata[AREAL_INFERENCE_METADATA_KEY] = dict(
                        effective_inference
                    )
                if chat_request is not None:
                    resolved_metadata[CHAT_REQUEST_METADATA_KEY] = dict(chat_request)

                # Commit mutable session routing only after pin CAS and all
                # metadata construction have succeeded.
                if session.security_mode is None:
                    session.security_mode = requested_mode
                if inference is not _INFERENCE_OMITTED:
                    session.inference = dict(inference)
                session.active_turns += 1
                session.active_turns_drained.clear()
                session.last_active = time.monotonic()
                return session, resolved_metadata, session.history.copy()

    async def _close_session_state(
        session_key: str,
        session: _SessionData,
    ) -> bool:
        """Drain one incarnation and clear its pin only after Worker close."""

        current_task = asyncio.current_task()
        try:
            await session.active_turns_drained.wait()
            if not await _close_worker_session(session_key):
                # Keep the closing tombstone and pin.  A later close/reaper
                # retries instead of letting a new incarnation reuse Worker
                # state whose cleanup was never confirmed.
                return False
            async with session.lifecycle_lock:
                if sessions.get(session_key) is session:
                    sessions.pop(session_key, None)
                    memory_pin_cache.clear(session_key)
            return True
        finally:
            if session_close_tasks.get(session_key) is current_task:
                session_close_tasks.pop(session_key, None)

    async def _begin_session_close(
        session_key: str,
        *,
        idle_only: bool,
    ) -> asyncio.Task[bool] | None:
        """Install one closing tombstone/task for a session incarnation."""

        while True:
            session = sessions.get(session_key)
            if session is None:
                # The idle reaper walks a detached key snapshot.  Another
                # close may retire a later key while this scan is waiting on
                # an earlier one; do not resurrect that stale snapshot entry.
                if idle_only:
                    return None
                # Even an unknown local session may still exist in the Worker
                # after a prior partial failure.  The tombstone prevents a new
                # incarnation from racing ahead of that explicit cleanup
                # attempt.
                candidate = _SessionData()
                session = sessions.setdefault(session_key, candidate)
            async with session.lifecycle_lock:
                if sessions.get(session_key) is not session:
                    continue
                existing = session_close_tasks.get(session_key)
                if existing is not None:
                    return existing
                now = time.monotonic()
                if idle_only and not session.closing:
                    if (
                        session.active_turns != 0
                        or now - session.last_active <= config.session_timeout
                    ):
                        return None
                session.closing = True
                task = asyncio.create_task(
                    _close_session_state(session_key, session),
                    name=f"areal-data-proxy-close:{session_key}",
                )
                session_close_tasks[session_key] = task
                return task

    async def _reap_idle_sessions() -> None:
        while True:
            await asyncio.sleep(60)
            close_tasks: list[tuple[str, asyncio.Task[bool]]] = []
            for session_key in tuple(sessions):
                task = await _begin_session_close(session_key, idle_only=True)
                if task is not None:
                    close_tasks.append((session_key, task))
            if not close_tasks:
                continue

            # Start every eligible close in this scan before waiting.  A slow
            # Worker cleanup for one session must not head-of-line block the
            # remaining idle sessions.  Shielding preserves the existing
            # shutdown contract: cancelling the reaper stops future scans but
            # does not cancel an already installed exact close task.
            results = await asyncio.gather(
                *(asyncio.shield(task) for _, task in close_tasks),
                return_exceptions=True,
            )
            for (session_key, _), result in zip(close_tasks, results, strict=True):
                if isinstance(result, BaseException):
                    logger.error(
                        "Unexpected failure while reaping idle session %s: %r",
                        session_key,
                        result,
                    )
            reaped = sum(result is True for result in results)
            if reaped:
                logger.info("Reaped %d idle sessions", reaped)

    @app.on_event("startup")
    async def startup():
        if config.worker_hop_api_key:
            try:
                response = await http_client.get(
                    f"{config.worker_addr}/internal/auth-check",
                    headers=worker_hop_headers,
                    timeout=5,
                )
                response.raise_for_status()
                payload = response.json()
                valid_receipt = (
                    type(payload) is dict
                    and set(payload) == {"status", "worker_hop_auth"}
                    and type(payload["status"]) is str
                    and payload["status"] == "ok"
                    and type(payload["worker_hop_auth"]) is bool
                    and payload["worker_hop_auth"] is True
                )
                if not valid_receipt:
                    raise ValueError(
                        "Worker returned a malformed hop-authentication receipt"
                    )
            except Exception as error:
                await http_client.aclose()
                raise RuntimeError("Worker hop authentication check failed") from error
        app.state.reaper_task = asyncio.create_task(_reap_idle_sessions())

    @app.on_event("shutdown")
    async def shutdown():
        reaper_task = getattr(app.state, "reaper_task", None)
        if reaper_task is not None:
            reaper_task.cancel()
            with suppress(asyncio.CancelledError):
                await reaper_task
        close_tasks: list[asyncio.Task[bool]] = []
        for session_key in tuple(sessions):
            task = await _begin_session_close(session_key, idle_only=False)
            if task is not None:
                close_tasks.append(task)
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)
        memory_pin_cache.clear_all()
        await http_client.aclose()

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "active_sessions": len(sessions),
            "worker_addr": config.worker_addr,
        }

    @app.post("/session/{session_key}/turn")
    async def turn(session_key: str, body: dict[str, Any], request: Request):
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
        _validate_session_key(session_key)
        message = body.get("message", "")
        run_id = body.get("run_id", "")
        queue_mode = body.get("queue_mode", "collect")
        try:
            # ``areal_memory`` is written only by this proxy.  Parse the
            # optional top-level pin before mutating any session binding, so a
            # malformed request cannot reserve a session.
            user_metadata = copy_user_metadata(body.get("metadata", {}))
            submitted_pin_present = MEMORY_ASSIGNMENT_PIN_FIELD in body
            submitted_pin = (
                MemoryAssignmentPinWireV1.from_wire(body[MEMORY_ASSIGNMENT_PIN_FIELD])
                if submitted_pin_present
                else None
            )
            inference = _parse_inference(body)
            chat_request = _parse_chat_request(body)
            memory_control_asserted = _parse_memory_control_authorized(body)
            if memory_control_asserted:
                # The JSON marker controls protocol flow but is forgeable.  A
                # dedicated Gateway→DataProxy hop must authenticate it before
                # any session creation or pin CAS.  Never reuse the externally
                # configured Agent admin key for this cross-scope capability.
                if not config.memory_control_api_key:
                    raise HTTPException(
                        status_code=503,
                        detail="Memory assignment transport is not configured",
                    )
                await verify_admin_key(
                    request.headers.get("Authorization", ""),
                    expected_key=config.memory_control_api_key,
                )
            memory_control_authorized = memory_control_asserted
            if submitted_pin_present and not config.memory_control_api_key:
                raise HTTPException(
                    status_code=503,
                    detail="Memory assignment transport is not configured",
                )
            if submitted_pin_present and not memory_control_authorized:
                raise HTTPException(
                    status_code=403,
                    detail="Memory assignment use requires trusted ingress",
                )
        except MemoryPinTransportError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        try:
            session, metadata, history = await _admit_turn(
                session_key,
                submitted_pin_present=submitted_pin_present,
                submitted_pin=submitted_pin,
                inference=inference,
                user_metadata=user_metadata,
                chat_request=chat_request,
                memory_control_authorized=memory_control_authorized,
            )
        except MemoryPinTransportError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

        worker_request = {
            "message": message,
            "session_key": session_key,
            "run_id": run_id,
            "history": history,
            "queue_mode": queue_mode,
            "metadata": metadata,
        }

        # A turn lease spans the complete structured request or raw response
        # body.  Close/reaper cannot retire the incarnation while either is
        # still using its Worker-side state.
        lease_transferred = False
        resp: httpx.Response | None = None
        try:
            req = http_client.build_request(
                "POST",
                f"{config.worker_addr}/run",
                json=worker_request,
                headers=worker_hop_headers or None,
            )
            resp = await http_client.send(req, stream=True)
            is_passthrough = resp.headers.get(PASSTHROUGH_HEADER) == "1"

            if is_passthrough:
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
                response = CleanupStreamingResponse(
                    resp.aiter_raw(),
                    cleanup=lambda: _close_response_and_release(resp, session),
                    cleanup_task_name=(
                        f"areal-data-proxy-stream-cleanup:{session_key}"
                    ),
                    status_code=resp.status_code,
                    headers=headers,
                    media_type=resp.headers.get("content-type") or None,
                )
                lease_transferred = True
                return response

            # Structured turn: read the full JSON body, then rebuild history.
            await resp.aread()
            status_code = resp.status_code
            try:
                result = resp.json()
            except ValueError:
                result = None

            if not isinstance(result, dict):
                # A structured Worker response is an object by contract.  Do
                # not turn an upstream HTML page, empty body, or JSON array
                # into an unhandled DataProxy exception, and never append a
                # turn to history when its response cannot be interpreted.
                return JSONResponse(
                    {"detail": "worker returned an invalid structured response"},
                    status_code=status_code if status_code >= 400 else 502,
                )

            if status_code >= 400:
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
            return JSONResponse(result, status_code=status_code)
        finally:
            if not lease_transferred:
                await _finish_response(resp, session)

    async def _authorize_session_state_access(request: Request) -> None:
        if config.memory_control_api_key:
            # History reveals conversation state, while close destroys the
            # session's pin and lifecycle state.  Both belong to the same
            # Gateway→DataProxy trust boundary as a privileged Memory turn.
            # Authenticate before key validation or lookup so unauthorized
            # callers cannot use either route as a state oracle.  Standalone
            # deployments retain anonymous access when Memory transport is
            # disabled.
            await verify_admin_key(
                request.headers.get("Authorization", ""),
                expected_key=config.memory_control_api_key,
            )

    async def _close_session(session_key: object):
        session_key = _validate_session_key(session_key)
        task = await _begin_session_close(session_key, idle_only=False)
        if task is None:  # pragma: no cover - explicit close always creates one
            raise HTTPException(
                status_code=409, detail="session close was not admitted"
            )
        if not await asyncio.shield(task):
            raise HTTPException(
                status_code=503,
                detail="worker session close failed; retry is required",
            )
        return {"status": "ok"}

    @app.post("/sessions/close")
    async def close_session(request: Request, body: dict[str, Any]):
        """Close a session without interpreting its identity as URL syntax."""

        await _authorize_session_state_access(request)
        return await _close_session(body.get("session_key"))

    @app.post("/session/{session_key}/close", deprecated=True)
    async def close_session_legacy(session_key: str, request: Request):
        """Compatibility route; internal callers use ``/sessions/close``."""

        await _authorize_session_state_access(request)
        return await _close_session(session_key)

    @app.get("/session/{session_key}/history")
    async def get_history(session_key: str, request: Request):
        await _authorize_session_state_access(request)
        session_key = _validate_session_key(session_key)
        session = sessions.get(session_key)
        if session is None:
            return {"history": []}
        return {"history": session.history}

    return app
