# SPDX-License-Identifier: Apache-2.0

"""Proxy gateway server: lightweight, stateless FastAPI gateway.

Routes external user requests (OpenAI/Anthropic SDK) to session-pinned
backend proxy workers. Holds NO session data — only routing state and
a coordination queue for the online training workflow.
"""

from __future__ import annotations

import asyncio
import hmac
import json
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass

import aiohttp
from fastapi import FastAPI, Request, Response

from areal.utils import logging

from .server import (
    ANTHROPIC_MESSAGES_PATHNAME,
    CHAT_COMPLETIONS_PATHNAME,
    DEFAULT_ADMIN_API_KEY,
    INTERNAL_WAIT_FOR_SESSION_PATHNAME,
    RESPONSES_PATHNAME,
    RL_END_SESSION_PATHNAME,
    RL_SET_REWARD_PATHNAME,
    RL_START_SESSION_PATHNAME,
    WaitForSessionRequest,
    WaitForSessionResponse,
)

logger = logging.getLogger("ProxyGateway")

# Timeout for forwarding requests to backend proxy workers.
_CLIENT_TIMEOUT = aiohttp.ClientTimeout(total=120)

# Default timeout for /internal/wait_for_session (1 hour).
_DEFAULT_WAIT_TIMEOUT = 3600.0


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class _ReadyWorkerEntry:
    """A proxy worker registered as ready to accept an external session.

    Created by ``_OnlineAgent`` via ``/internal/wait_for_session``.
    Consumed by ``/rl/start_session`` when an external user arrives.

    Attributes
    ----------
    worker_addr : str
        HTTP address of the backend proxy worker.
    future : asyncio.Future[CompletedSessionInfo]
        Resolved when the session ends via ``/rl/end_session``.
    """

    worker_addr: str
    future: asyncio.Future  # Future[CompletedSessionInfo]


@dataclass
class CompletedSessionInfo:
    """Credentials of a completed external session.

    Returned by ``/internal/wait_for_session`` and used by
    ``_OnlineAgent`` to pass session credentials back to
    ``OpenAIProxyWorkflow`` for ``export_trajectories``.

    Attributes
    ----------
    session_api_key : str
        The session-scoped API key issued by the backend proxy worker.
    session_id : str
        Unique session identifier on the backend proxy worker.
    worker_addr : str
        HTTP address of the backend proxy worker that hosted this session.
    """

    session_api_key: str
    session_id: str
    worker_addr: str


@dataclass
class _SessionRoute:
    """Routing entry for an active session.

    Attributes
    ----------
    worker_addr : str
        HTTP address of the backend proxy worker.
    session_id : str
        Session identifier on the backend worker.
    pending_future : asyncio.Future[CompletedSessionInfo] | None
        If set, this session was initiated via the online-mode readiness
        queue. Resolved when ``/rl/end_session`` is called.
    """

    worker_addr: str
    session_id: str
    pending_future: asyncio.Future | None = None  # Future[CompletedSessionInfo]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_bearer_token(request: Request) -> str | None:
    """Extract the bearer token from the Authorization header."""
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:]
    return None


def _forwarding_headers(raw_headers: dict[str, str]) -> dict[str, str]:
    """Select headers to forward to the backend."""
    out: dict[str, str] = {}
    for key, value in raw_headers.items():
        if key.lower() in ("authorization", "content-type"):
            out[key] = value
    return out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_proxy_gateway_app(
    proxy_addrs: list[str],
    admin_api_key: str = DEFAULT_ADMIN_API_KEY,
    refresh_timeout: float = 120.0,
    key_pool_size: int = 4096,
) -> FastAPI:
    """Create and configure the proxy gateway FastAPI app.

    Parameters
    ----------
    proxy_addrs : list[str]
        HTTP addresses of all backend proxy workers.
    admin_api_key : str
        Admin API key shared with proxy workers.
    refresh_timeout : float
        Maximum seconds to wait for a ready worker during session refresh.
        Returns HTTP 429 on timeout.
    key_pool_size : int
        Maximum number of AReaL-issued API keys to track for reuse.
        Uses LRU eviction when full.

    Returns
    -------
    FastAPI
        Configured application ready to be served.
    """

    # -- Shared state (scoped to this app instance) --
    routes: dict[str, _SessionRoute] = {}
    ready_workers: asyncio.Queue[_ReadyWorkerEntry] = asyncio.Queue()
    rr_index: list[int] = [0]  # mutable counter for round-robin
    known_keys: OrderedDict[str, None] = OrderedDict()  # LRU key pool
    _refreshing: set[str] = set()  # keys currently mid-refresh
    _bg_tasks: set[asyncio.Task] = set()  # prevent GC of fire-and-forget tasks

    # -- Lifespan: shared aiohttp session --

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # noqa: ARG001
        logger.info(
            "Proxy gateway starting — %d backend worker(s): %s",
            len(proxy_addrs),
            ", ".join(proxy_addrs),
        )
        async with aiohttp.ClientSession() as session:
            app.state.http_session = session
            yield
        logger.info("Proxy gateway shutting down")

    app = FastAPI(title="AReaL Proxy Gateway", lifespan=lifespan)

    # -- Forwarding helper --

    async def _forward(
        worker_addr: str,
        path: str,
        body: bytes,
        headers: dict[str, str],
    ) -> Response:
        """Forward an HTTP request to a backend proxy worker."""
        url = f"{worker_addr}/{path}"
        fwd_headers = _forwarding_headers(headers)
        http_session: aiohttp.ClientSession = app.state.http_session
        try:
            async with http_session.post(
                url, data=body, headers=fwd_headers, timeout=_CLIENT_TIMEOUT
            ) as resp:
                data = await resp.read()
                return Response(
                    content=data,
                    status_code=resp.status,
                    media_type=resp.content_type,
                )
        except (TimeoutError, aiohttp.ClientError, RuntimeError) as exc:
            logger.error("Failed to forward to %s: %s", worker_addr, exc)
            return Response(status_code=502, content=b"Backend unreachable")

    # -- Future settlement helpers --

    def _resolve_future(
        future: asyncio.Future | None,
        info: CompletedSessionInfo,
    ) -> None:
        """Resolve an online-mode future with session info, if unsettled."""
        if future is not None and not future.done():
            future.set_result(info)

    def _reject_future(
        future: asyncio.Future | None,
        reason: str,
    ) -> None:
        """Reject an online-mode future with a RuntimeError, if unsettled."""
        if future is not None and not future.done():
            future.set_exception(RuntimeError(reason))

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    @app.get("/health")
    async def health():
        active = len(routes)
        queued = ready_workers.qsize()
        logger.info(
            "Health check — %d worker(s), %d active session(s), %d queued worker(s)",
            len(proxy_addrs),
            active,
            queued,
        )
        return {"status": "ok", "workers": len(proxy_addrs)}

    # -- Key pool management --

    def _track_key(api_key: str) -> list[tuple[str, _SessionRoute]]:
        """Record *api_key* in the bounded LRU pool of known keys.

        Returns list of ``(evicted_key, evicted_route)`` pairs whose
        backend sessions should be ended asynchronously by the caller.
        """
        known_keys[api_key] = None
        known_keys.move_to_end(api_key)
        evicted: list[tuple[str, _SessionRoute]] = []
        while len(known_keys) > key_pool_size:
            evicted_key, _ = known_keys.popitem(last=False)
            evicted_route = routes.pop(evicted_key, None)
            if evicted_route is not None:
                _reject_future(
                    evicted_route.pending_future,
                    f"Session route evicted from key pool: {evicted_key}",
                )
                evicted.append((evicted_key, evicted_route))
                logger.warning(
                    "[_track_key] Evicted key %s (active route to %s)",
                    evicted_key,
                    evicted_route.worker_addr,
                )
        return evicted

    async def _end_evicted_sessions(
        evicted: list[tuple[str, _SessionRoute]],
    ) -> None:
        """Best-effort end backend sessions for evicted routes."""
        for evicted_key, route in evicted:
            try:
                headers = {
                    "authorization": f"Bearer {evicted_key}",
                    "content-type": "application/json",
                }
                await _forward(
                    route.worker_addr, RL_END_SESSION_PATHNAME, b"{}", headers
                )
            except Exception:
                logger.debug(
                    "Failed to end evicted session on %s (best-effort)",
                    route.worker_addr,
                )

    def _schedule_eviction_cleanup(
        evicted: list[tuple[str, _SessionRoute]],
    ) -> None:
        """Fire-and-forget cleanup for evicted sessions."""
        if not evicted:
            return
        task = asyncio.create_task(_end_evicted_sessions(evicted))
        _bg_tasks.add(task)
        task.add_done_callback(_bg_tasks.discard)

    # -- Refresh helper (end old session, resolve future) --

    async def _end_and_resolve(api_key: str, old_route: _SessionRoute) -> None:
        """End the old session on the worker and resolve the online-mode future."""
        end_headers = {
            "authorization": f"Bearer {api_key}",
            "content-type": "application/json",
        }
        end_resp = await _forward(
            old_route.worker_addr, RL_END_SESSION_PATHNAME, b"{}", end_headers
        )

        if end_resp.status_code >= 400:
            logger.warning(
                "[refresh] end_session failed on worker %s (status %d)",
                old_route.worker_addr,
                end_resp.status_code,
            )
            _reject_future(
                old_route.pending_future,
                f"end_session failed during refresh (status {end_resp.status_code})",
            )
            return

        try:
            end_data = json.loads(end_resp.body)
            n = end_data.get("interaction_count", -1)
        except (json.JSONDecodeError, AttributeError):
            n = -1
        logger.info(
            "[refresh] Old session %s ended (%d interactions)",
            old_route.session_id,
            n,
        )
        _resolve_future(
            old_route.pending_future,
            CompletedSessionInfo(
                session_api_key=api_key,
                session_id=old_route.session_id,
                worker_addr=old_route.worker_addr,
            ),
        )

    # -- start_session (admin auth, refresh / reuse / new) --------------

    @app.post(f"/{RL_START_SESSION_PATHNAME}")
    async def start_session(request: Request):
        token = _extract_bearer_token(request)
        if not hmac.compare_digest(token or "", admin_api_key):
            return Response(status_code=401, content=b"Unauthorized")
        body = await request.body()
        raw_headers = dict(request.headers)

        parsed = json.loads(body) if body else {}
        task_id = parsed.get("task_id", "?")
        requested_key = parsed.get("api_key") or None

        logger.info(
            "[start_session] Request received (task_id=%s, api_key=%s)",
            task_id,
            "reuse" if requested_key else "new",
        )

        # Reject if a refresh is already in flight for this key.
        # Must be checked BEFORE `in routes` since refresh pops the route.
        if requested_key and requested_key in _refreshing:
            return Response(
                status_code=429,
                content=json.dumps(
                    {"detail": "A refresh is already in progress for this key."}
                ).encode(),
            )

        # ---- REFRESH PATH ----
        # Known key with an active route → end old session, wait for
        # the training pipeline to cycle, start a new session.
        if requested_key and requested_key in known_keys and requested_key in routes:
            _refreshing.add(requested_key)
            ready_entry: _ReadyWorkerEntry | None = None
            try:
                old_route = routes.pop(requested_key)
                logger.info(
                    "[refresh] Ending session %s on worker %s",
                    old_route.session_id,
                    old_route.worker_addr,
                )

                await _end_and_resolve(requested_key, old_route)

                # Skip stale ready entries within the deadline.
                deadline = asyncio.get_running_loop().time() + refresh_timeout
                ready_entry = None
                while True:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        break
                    try:
                        candidate = await asyncio.wait_for(
                            ready_workers.get(), timeout=remaining
                        )
                    except TimeoutError:
                        break
                    if candidate.future.done():
                        logger.debug(
                            "[refresh] Skipping stale ready entry for worker %s",
                            candidate.worker_addr,
                        )
                        continue
                    ready_entry = candidate
                    break

                if ready_entry is None:
                    logger.warning(
                        "[refresh] Timed out after %.0fs waiting for ready worker",
                        refresh_timeout,
                    )
                    known_keys.pop(requested_key, None)
                    return Response(
                        status_code=429,
                        content=json.dumps(
                            {
                                "detail": (
                                    f"Refresh timed out ({refresh_timeout:.0f}s). "
                                    "Your old session was ended. Retry to get a new session."
                                )
                            }
                        ).encode(),
                    )

                # Forward start_session to new ready worker.
                resp = await _forward(
                    ready_entry.worker_addr,
                    RL_START_SESSION_PATHNAME,
                    body,
                    raw_headers,
                )
                if resp.status_code == 200:
                    resp_data = json.loads(resp.body)
                    api_key = resp_data["api_key"]
                    session_id = resp_data["session_id"]
                    routes[api_key] = _SessionRoute(
                        worker_addr=ready_entry.worker_addr,
                        session_id=session_id,
                        pending_future=ready_entry.future,
                    )
                    _schedule_eviction_cleanup(_track_key(api_key))
                    logger.info(
                        "[refresh] New session %s started on worker %s (same key)",
                        session_id,
                        ready_entry.worker_addr,
                    )
                else:
                    # Worker rejected — cancel the orphaned future so the
                    # _OnlineAgent that placed this entry doesn't hang for 1h.
                    ready_entry.future.cancel()
                    known_keys.pop(requested_key, None)
                    logger.warning(
                        "[refresh] Worker %s rejected start_session (status %d)",
                        ready_entry.worker_addr,
                        resp.status_code,
                    )
                return resp
            except Exception:
                known_keys.pop(requested_key, None)
                _reject_future(old_route.pending_future, "Refresh failed unexpectedly")
                # Also settle the new worker's future if we consumed one
                if ready_entry is not None:
                    _reject_future(ready_entry.future, "Refresh failed unexpectedly")
                raise
            finally:
                _refreshing.discard(requested_key)

        # ---- REUSE / NEW PATHS ----
        # Prepare the body forwarded to the worker.
        if requested_key and requested_key in known_keys:
            # Known key, no active route (previously ended) → keep api_key
            # in body so the worker reuses it.
            pass
        else:
            # Unknown or absent key → strip so the worker generates a new one.
            if "api_key" in parsed:
                parsed.pop("api_key")
                body = json.dumps(parsed).encode()

        # 1. Try readiness queue (online mode — pre-granted capacity).
        # Drain stale entries whose future is already resolved.
        ready_entry: _ReadyWorkerEntry | None = None
        while not ready_workers.empty():
            try:
                candidate = ready_workers.get_nowait()
            except asyncio.QueueEmpty:
                break
            if candidate.future.done():
                logger.debug(
                    "[start_session] Skipping stale ready entry for worker %s",
                    candidate.worker_addr,
                )
                continue
            ready_entry = candidate
            break
        if ready_entry is not None:
            resp = await _forward(
                ready_entry.worker_addr,
                RL_START_SESSION_PATHNAME,
                body,
                raw_headers,
            )
            if resp.status_code == 200:
                resp_data = json.loads(resp.body)
                api_key = resp_data["api_key"]
                session_id = resp_data["session_id"]
                routes[api_key] = _SessionRoute(
                    worker_addr=ready_entry.worker_addr,
                    session_id=session_id,
                    pending_future=ready_entry.future,
                )
                _schedule_eviction_cleanup(_track_key(api_key))
                logger.info(
                    "[start_session] Session %s started via ready worker %s (online mode)",
                    session_id,
                    ready_entry.worker_addr,
                )
            else:
                # Backend rejected despite being "ready" — cancel the orphaned
                # future so the _OnlineAgent doesn't hang for 1 hour.
                ready_entry.future.cancel()
                logger.warning(
                    "Ready worker %s rejected start_session (status %d)",
                    ready_entry.worker_addr,
                    resp.status_code,
                )
            return resp

        # 2. Round-robin fallback.
        last_resp: Response | None = None
        for _ in range(len(proxy_addrs)):
            idx = rr_index[0] % len(proxy_addrs)
            rr_index[0] += 1
            worker_addr = proxy_addrs[idx]

            resp = await _forward(
                worker_addr, RL_START_SESSION_PATHNAME, body, raw_headers
            )
            if resp.status_code == 200:
                resp_data = json.loads(resp.body)
                api_key = resp_data["api_key"]
                session_id = resp_data["session_id"]
                routes[api_key] = _SessionRoute(
                    worker_addr=worker_addr,
                    session_id=session_id,
                    pending_future=None,
                )
                _schedule_eviction_cleanup(_track_key(api_key))
                logger.info(
                    "[start_session] Session %s started via round-robin worker %s",
                    session_id,
                    worker_addr,
                )
                return resp
            if resp.status_code != 429:
                # Non-capacity error — propagate immediately.
                return resp
            last_resp = resp

        # All workers returned 429.
        if last_resp is not None:
            return last_resp
        return Response(status_code=429, content=b"No capacity available")

    # -- Session-routed forwarding endpoints ---------------------------

    async def _session_forward(request: Request, path: str) -> Response:
        token = _extract_bearer_token(request)
        if token is None or token not in routes:
            return Response(status_code=401, content=b"Invalid session key")
        route = routes[token]
        body = await request.body()
        return await _forward(route.worker_addr, path, body, dict(request.headers))

    @app.post(f"/{CHAT_COMPLETIONS_PATHNAME}")
    async def chat_completions(request: Request):
        token = _extract_bearer_token(request)
        route = routes.get(token) if token else None
        if route:
            logger.info(
                "[chat/completions] Session %s → worker %s",
                route.session_id,
                route.worker_addr,
            )
        return await _session_forward(request, CHAT_COMPLETIONS_PATHNAME)

    @app.post(f"/{RESPONSES_PATHNAME}")
    async def responses(request: Request):
        token = _extract_bearer_token(request)
        route = routes.get(token) if token else None
        if route:
            logger.info(
                "[responses] Session %s → worker %s",
                route.session_id,
                route.worker_addr,
            )
        return await _session_forward(request, RESPONSES_PATHNAME)

    @app.post(f"/{ANTHROPIC_MESSAGES_PATHNAME}")
    async def messages(request: Request):
        token = _extract_bearer_token(request)
        route = routes.get(token) if token else None
        if route:
            logger.info(
                "[messages] Session %s → worker %s",
                route.session_id,
                route.worker_addr,
            )
        return await _session_forward(request, ANTHROPIC_MESSAGES_PATHNAME)

    @app.post(f"/{RL_SET_REWARD_PATHNAME}")
    async def set_reward(request: Request):
        token = _extract_bearer_token(request)
        route = routes.get(token) if token else None
        if route:
            logger.info(
                "[set_reward] Session %s → worker %s",
                route.session_id,
                route.worker_addr,
            )
        return await _session_forward(request, RL_SET_REWARD_PATHNAME)

    # -- end_session (session auth, resolve online-mode future) --------

    @app.post(f"/{RL_END_SESSION_PATHNAME}")
    async def end_session(request: Request):
        token = _extract_bearer_token(request)
        if token is None or token not in routes:
            return Response(status_code=401, content=b"Invalid session key")

        route = routes[token]
        body = await request.body()

        resp = await _forward(
            route.worker_addr, RL_END_SESSION_PATHNAME, body, dict(request.headers)
        )

        # Remove route — session is done regardless of status.
        routes.pop(token, None)

        if resp.status_code < 400:
            _resolve_future(
                route.pending_future,
                CompletedSessionInfo(
                    session_api_key=token,
                    session_id=route.session_id,
                    worker_addr=route.worker_addr,
                ),
            )
            mode = "online" if route.pending_future is not None else "direct"
            logger.info(
                "[end_session] Session %s ended (%s mode, %d active)",
                route.session_id,
                mode,
                len(routes),
            )
        else:
            _reject_future(
                route.pending_future,
                f"end_session failed (status {resp.status_code})",
            )
            log = logger.error if resp.status_code >= 500 else logger.warning
            log(
                "[end_session] Error %d for session %s on worker %s",
                resp.status_code,
                route.session_id,
                route.worker_addr,
            )

        return resp

    # -- Internal: wait_for_session (admin auth, blocks) ---------------

    @app.post(f"/{INTERNAL_WAIT_FOR_SESSION_PATHNAME}")
    async def wait_for_session(request: Request):
        token = _extract_bearer_token(request)
        if not hmac.compare_digest(token or "", admin_api_key):
            return Response(status_code=401, content=b"Unauthorized")

        req_data = WaitForSessionRequest(**(await request.json()))

        loop = asyncio.get_running_loop()
        future: asyncio.Future[CompletedSessionInfo] = loop.create_future()

        entry = _ReadyWorkerEntry(
            worker_addr=req_data.worker_addr,
            future=future,
        )
        await ready_workers.put(entry)
        logger.info(
            "[wait_for_session] Worker %s registered in readiness queue (queue size: %d)",
            req_data.worker_addr,
            ready_workers.qsize(),
        )

        try:
            result = await asyncio.wait_for(future, timeout=_DEFAULT_WAIT_TIMEOUT)
        except asyncio.CancelledError:
            # Future was cancelled (e.g. backend rejected the ready worker).
            logger.warning(
                "[wait_for_session] Worker %s session assignment was cancelled",
                req_data.worker_addr,
            )
            return Response(
                status_code=503,
                content=b"Session assignment was cancelled",
            )
        except TimeoutError:
            logger.warning(
                "[wait_for_session] Worker %s timed out after %.0fs",
                req_data.worker_addr,
                _DEFAULT_WAIT_TIMEOUT,
            )
            return Response(
                status_code=408, content=b"Timeout waiting for session completion"
            )
        except RuntimeError as exc:
            logger.warning(
                "[wait_for_session] Worker %s session failed: %s",
                req_data.worker_addr,
                exc,
            )
            return Response(
                status_code=500,
                content=f"Session failed: {exc}".encode(),
            )

        logger.info(
            "[wait_for_session] Session %s completed on worker %s — notifying _OnlineAgent",
            result.session_id,
            result.worker_addr,
        )
        return WaitForSessionResponse(
            session_api_key=result.session_api_key,
            session_id=result.session_id,
            worker_addr=result.worker_addr,
        ).model_dump()

    return app
