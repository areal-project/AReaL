# SPDX-License-Identifier: Apache-2.0

import asyncio
import concurrent.futures
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from http import HTTPStatus
from typing import Any

import aiohttp
import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from areal.utils import logging
from areal.utils.network import format_hostport, gethostip, split_hostport

DEFAULT_RETRIES = 1
DEFAULT_REQUEST_TIMEOUT = 3600

DEFAULT_ADMIN_API_KEY = "areal-admin-key"
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})

# ---------------------------------------------------------------------------
# Shared capacity defaults for httpx clients and uvicorn servers
# ---------------------------------------------------------------------------
HTTPX_MAX_CONNECTIONS = 4096
HTTPX_MAX_KEEPALIVE_CONNECTIONS = 1024
HTTPX_KEEPALIVE_EXPIRY = 30
HTTPX_RETRIES = 3

UVICORN_BACKLOG = 4096
UVICORN_LIMIT_CONCURRENCY = 4096


def get_default_httpx_limits() -> httpx.Limits:
    """Return shared httpx.Limits for high-concurrency services."""
    return httpx.Limits(
        max_connections=HTTPX_MAX_CONNECTIONS,
        max_keepalive_connections=HTTPX_MAX_KEEPALIVE_CONNECTIONS,
        keepalive_expiry=HTTPX_KEEPALIVE_EXPIRY,
    )


def create_httpx_client(timeout: float | None = 120.0, **kwargs) -> httpx.AsyncClient:
    """Create an httpx.AsyncClient with shared pool limits and transport-level retries."""
    transport = httpx.AsyncHTTPTransport(
        retries=HTTPX_RETRIES,
        limits=get_default_httpx_limits(),
    )
    return httpx.AsyncClient(timeout=timeout, transport=transport, **kwargs)


class HttpxAsyncClientCleanupState(Enum):
    OPEN = auto()
    CLOSING = auto()
    SUCCEEDED = auto()
    FAILED = auto()


@dataclass
class HttpxAsyncClientCleanup:
    """Single-flight cleanup state for one loop-owned async client.

    ``httpx.AsyncClient.is_closed`` only describes HTTPX's logical state.  It can
    become true before the transport's ``aclose()`` has completed (or even when
    transport cleanup subsequently fails), so it cannot acknowledge resource
    cleanup.  This state object records the outcome of the one real close call
    and lets concurrent callers await that same outcome.
    """

    client: httpx.AsyncClient
    owner_loop: asyncio.AbstractEventLoop
    _state: HttpxAsyncClientCleanupState = field(
        default=HttpxAsyncClientCleanupState.OPEN, init=False
    )
    _error: BaseException | None = field(default=None, init=False)
    _completion: concurrent.futures.Future[None] | None = field(
        default=None, init=False
    )
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    @property
    def succeeded(self) -> bool:
        with self._lock:
            return self._state is HttpxAsyncClientCleanupState.SUCCEEDED

    @property
    def error(self) -> BaseException | None:
        with self._lock:
            return self._error

    def snapshot(
        self,
    ) -> tuple[HttpxAsyncClientCleanupState, BaseException | None]:
        with self._lock:
            return self._state, self._error

    def terminal_result(self) -> bool:
        """Return whether cleanup succeeded, or re-raise its stable failure."""
        state, error = self.snapshot()
        if state is HttpxAsyncClientCleanupState.FAILED:
            assert error is not None
            raise error
        return state is HttpxAsyncClientCleanupState.SUCCEEDED

    async def close(self) -> None:
        current_loop = asyncio.get_running_loop()
        with self._lock:
            state = self._state
            error = self._error
            if state is HttpxAsyncClientCleanupState.SUCCEEDED:
                return
            if state is HttpxAsyncClientCleanupState.FAILED:
                assert error is not None
                raise error
            if current_loop is not self.owner_loop:
                raise RuntimeError(
                    "httpx.AsyncClient cleanup must run on its owner event loop"
                )
            if state is HttpxAsyncClientCleanupState.OPEN:
                completion: concurrent.futures.Future[None] = (
                    concurrent.futures.Future()
                )
                # ``asyncio.wrap_future`` propagates waiter cancellation to its
                # source future.  Mark this shared acknowledgement as running so
                # one cancelled follower cannot cancel the leader's completion.
                if not completion.set_running_or_notify_cancel():
                    raise RuntimeError(
                        "new async-client cleanup completion was unexpectedly cancelled"
                    )
                self._completion = completion
                self._state = HttpxAsyncClientCleanupState.CLOSING
                leader = True
            else:
                assert state is HttpxAsyncClientCleanupState.CLOSING
                assert self._completion is not None
                completion = self._completion
                leader = False

        if not leader:
            await asyncio.wrap_future(completion)
            return

        try:
            await self.client.aclose()
        except BaseException as exc:
            with self._lock:
                self._error = exc
                self._state = HttpxAsyncClientCleanupState.FAILED
            completion.set_exception(exc)
            raise
        else:
            with self._lock:
                self._state = HttpxAsyncClientCleanupState.SUCCEEDED
            completion.set_result(None)


async def close_httpx_client_on_owner_loop(
    cleanup: HttpxAsyncClientCleanup,
) -> None:
    """Close an async client without driving loop-bound transports elsewhere."""
    if cleanup.terminal_result():
        return

    owner_loop = cleanup.owner_loop
    current_loop = asyncio.get_running_loop()
    if owner_loop is current_loop:
        await cleanup.close()
        return
    if owner_loop.is_closed():
        raise RuntimeError(
            "httpx.AsyncClient owner event loop closed before cleanup succeeded"
        )
    if owner_loop.is_running():
        close_future = asyncio.run_coroutine_threadsafe(cleanup.close(), owner_loop)
        await asyncio.wrap_future(close_future)
        return
    raise RuntimeError("httpx.AsyncClient owner event loop is not running")


def register_httpx_client_loop_cleanup(
    cleanup: HttpxAsyncClientCleanup,
    *,
    on_closed: Callable[[], None] | None = None,
) -> None:
    """Close a client on its owner loop immediately before loop shutdown."""
    from areal.infra.utils.concurrent import register_loop_cleanup

    async def close_client() -> None:
        await cleanup.close()
        if on_closed is not None:
            on_closed()

    register_loop_cleanup(close_client, loop=cleanup.owner_loop)


def close_httpx_client_from_sync(
    cleanup: HttpxAsyncClientCleanup,
) -> None:
    """Synchronously close a client without deadlocking its running owner loop."""
    if cleanup.terminal_result():
        return
    owner_loop = cleanup.owner_loop
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None
    if current_loop is owner_loop:
        raise RuntimeError(
            "cannot synchronously close httpx.AsyncClient from its owner event loop"
        )

    from areal.infra.utils.concurrent import run_async_task

    run_async_task(close_httpx_client_on_owner_loop, cleanup)


def get_default_uvicorn_kwargs() -> dict[str, Any]:
    """Return shared uvicorn capacity kwargs to spread into uvicorn.run()."""
    return {
        "backlog": UVICORN_BACKLOG,
        "limit_concurrency": UVICORN_LIMIT_CONCURRENCY,
    }


def validate_admin_api_key(
    host: str,
    admin_api_key: str,
    default_key: str = DEFAULT_ADMIN_API_KEY,
    config_field: str = "admin_api_key",
) -> None:
    """Refuse to start an HTTP service on a non-loopback bind with the default admin key.

    The default admin API key is publicly documented in the source tree, so
    a server that listens on a routable interface with the default key
    effectively has no admin authentication. Operators must either set a
    unique ``admin_api_key`` or opt in via ``AREAL_ALLOW_DEFAULT_ADMIN_KEY=1``
    when they knowingly accept the risk on a trusted network.

    A non-default key, or a loopback-only bind, is always accepted.
    """
    if admin_api_key != default_key:
        return

    resolved_host = host
    if host in ("0.0.0.0", "::"):
        resolved_host = gethostip()

    allow_override = os.environ.get("AREAL_ALLOW_DEFAULT_ADMIN_KEY", "0") == "1"
    if resolved_host in _LOOPBACK_HOSTS or allow_override:
        logger.warning(
            "Using default admin API key. Change '%s' before exposing this "
            "server on a network.",
            config_field,
        )
        return

    raise RuntimeError(
        f"Refusing to start server on non-loopback host {resolved_host!r} "
        f"with the default admin API key ({default_key!r}). Set "
        f"'{config_field}' to a unique secret, or set "
        "AREAL_ALLOW_DEFAULT_ADMIN_KEY=1 to acknowledge the risk in a "
        "trusted environment."
    )


async_http_retry = retry(
    stop=stop_after_attempt(8),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    retry=retry_if_exception_type((aiohttp.ClientError, OSError, RuntimeError)),
    reraise=True,
)

# Retry decorator for httpx-based calls (gateway forwarding).
async_httpx_retry = retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    retry=retry_if_exception_type(httpx.TransportError),
    reraise=True,
)


logger = logging.getLogger("HTTPUtils")


def get_default_connector():
    return aiohttp.TCPConnector(limit=0, use_dns_cache=False, force_close=True)


async def arequest_with_retry(
    addr: str,
    endpoint: str,
    payload: dict[str, Any] | None = None,
    session: aiohttp.ClientSession | None = None,
    method: str = "POST",
    max_retries: int | None = None,
    timeout: float | None = None,
    retry_delay: float = 1.0,
    verbose=False,
) -> dict | str | bytes:
    if timeout is None:
        timeout = DEFAULT_REQUEST_TIMEOUT
    last_exception = None
    max_retries = max_retries or DEFAULT_RETRIES
    try:
        host, port = split_hostport(addr)
        base_url = f"http://{format_hostport(host, port)}"
    except ValueError:
        base_url = f"http://{addr}"
    url = f"{base_url}{endpoint}"

    timeo = aiohttp.ClientTimeout(
        total=timeout,
        sock_connect=timeout,
        connect=timeout,
    )
    if session is None:
        _session = aiohttp.ClientSession(
            timeout=timeo,
            read_bufsize=1024 * 1024 * 10,
            connector=get_default_connector(),
        )
    else:
        _session = session

    for attempt in range(max_retries):
        try:
            if verbose:
                logger.info("enter client session, start sending requests")
            if method.upper() == "GET":
                ctx = _session.get(url, timeout=timeo)
            elif method.upper() == "POST":
                ctx = _session.post(url, json=payload, timeout=timeo)
            elif method.upper() == "PUT":
                ctx = _session.put(url, json=payload, timeout=timeo)
            elif method.upper() == "DELETE":
                ctx = _session.delete(url, timeout=timeo)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            async with ctx as response:
                if verbose:
                    logger.info("http requests return")
                response.raise_for_status()
                ctype = response.content_type or ""
                if ctype == "application/json":
                    res = await response.json()
                elif ctype.startswith("text/"):
                    res = await response.text()
                else:
                    res = await response.read()
                if verbose:
                    logger.info("get http result")
                if session is None:
                    await _session.close()
                return res
        except (TimeoutError, aiohttp.ClientError, aiohttp.ClientResponseError) as e:
            if isinstance(e, asyncio.TimeoutError):
                logger.warning(
                    "HTTP request to %s%s timed out after %.2fs (attempt %d/%d)",
                    addr,
                    endpoint,
                    timeout,
                    attempt + 1,
                    max_retries,
                )
            else:
                logger.warning(
                    "HTTP request to %s%s failed with %s: %s (attempt %d/%d)",
                    addr,
                    endpoint,
                    e.__class__.__name__,
                    str(e),
                    attempt + 1,
                    max_retries,
                )
            last_exception = e
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
            continue
    if session is None:
        await _session.close()
    raise RuntimeError(
        f"Failed after {max_retries} retries each. "
        f"Payload: {payload}. Addr: {addr}. Endpoint: {endpoint}. "
        f"Last error: {repr(last_exception)}"
    )


def response_ok(http_code: int) -> bool:
    return http_code == HTTPStatus.OK


def response_retryable(http_code: int) -> bool:
    return http_code == HTTPStatus.REQUEST_TIMEOUT


def ensure_end_with_slash(url: str) -> str:
    if not url.endswith("/"):
        return url + "/"
    return url
