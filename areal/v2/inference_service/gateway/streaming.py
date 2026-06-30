# SPDX-License-Identifier: Apache-2.0

"""Router communication and request forwarding utilities for the gateway."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import httpx

from areal.infra.utils.http import async_httpx_retry, create_httpx_client
from areal.utils import logging

logger = logging.getLogger("InferenceGateway")


class RouterUnreachableError(Exception):
    """Router service is unreachable or returned an error."""

    pass


class RouterKeyRejectedError(Exception):
    """Router rejected the API key (unknown key) or has no healthy workers."""

    def __init__(self, detail: str, status_code: int = 404):
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


@asynccontextmanager
async def _use_client(
    client: httpx.AsyncClient | None, timeout: float
) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Yield the shared *client* if provided, otherwise create a temporary one."""
    if client is not None:
        yield client
    else:
        async with create_httpx_client(timeout=timeout) as c:
            yield c


@async_httpx_retry
async def query_router(
    router_addr: str,
    api_key: str | None = None,
    path: str | None = None,
    timeout: float = 2.0,
    *,
    session_id: str | None = None,
    admin_api_key: str | None = None,
    model: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> str:
    """Ask the Router for a worker address.

    POST ``{router_addr}/route`` with ``{"api_key": ..., "path": ...}``
    or ``{"session_id": ...}``.
    Returns the ``worker_addr`` string.

    Parameters
    ----------
    admin_api_key : str | None
        When set, sent as ``Authorization: Bearer <key>`` so the Router
        can authenticate the request.
    session_id : str | None
        Pin routing to the session's worker.
    model : str | None
        Route to a specific model's data proxies.
    client : httpx.AsyncClient | None
        Shared HTTP client.  When ``None``, a per-request client is created
        (backwards-compatible, but less efficient).

    Raises
    ------
    RouterUnreachableError
        Router is unreachable or returned an unexpected HTTP error.
    RouterKeyRejectedError
        Router returned 404 (unknown key / session) or 503 (no healthy workers).
    """
    payload: dict[str, str] = {}
    if model is not None:
        payload["model"] = model
    if session_id is not None:
        payload["session_id"] = session_id
    else:
        if api_key is not None:
            payload["api_key"] = api_key
        if path is not None:
            payload["path"] = path
    try:
        headers = {}
        if admin_api_key is not None:
            headers["Authorization"] = f"Bearer {admin_api_key}"

        async with _use_client(client, timeout) as c:
            resp = await c.post(
                f"{router_addr}/route", json=payload, headers=headers, timeout=timeout
            )

        if resp.status_code == 404:
            data = resp.json()
            raise RouterKeyRejectedError(
                data.get("detail", data.get("error", "Not found")), 404
            )
        if resp.status_code == 503:
            data = resp.json()
            raise RouterKeyRejectedError(
                data.get("detail", data.get("error", "No healthy workers")), 503
            )
        resp.raise_for_status()
        return resp.json()["worker_addr"]
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        raise RouterUnreachableError(f"Router unreachable: {exc}") from exc
    except httpx.TimeoutException as exc:
        raise RouterUnreachableError(f"Router timed out: {exc}") from exc
    except httpx.TransportError as exc:
        raise RouterUnreachableError(f"Router transport error: {exc}") from exc
    except httpx.HTTPStatusError as exc:
        raise RouterUnreachableError(
            f"Router returned HTTP {exc.response.status_code}: {exc}"
        ) from exc


@async_httpx_retry
async def register_session_in_router(
    router_addr: str,
    sessions: list[dict[str, str]],
    worker_addr: str,
    timeout: float,
    admin_api_key: str | None = None,
    *,
    group_id: str,
    client: httpx.AsyncClient | None = None,
) -> None:
    """Register session(s) and their group in the Router atomically."""
    try:
        headers = {}
        if admin_api_key is not None:
            headers["Authorization"] = f"Bearer {admin_api_key}"

        payload: dict[str, Any] = {
            "sessions": sessions,
            "worker_addr": worker_addr,
            "group_id": group_id,
        }

        async with _use_client(client, timeout) as c:
            resp = await c.post(
                f"{router_addr}/register_session",
                json=payload,
                headers=headers,
                timeout=timeout,
            )

        resp.raise_for_status()
    except httpx.TransportError as exc:
        logger.error("Failed to register session in router: %s", exc)
        raise RouterUnreachableError(f"Failed to register session: {exc}") from exc
    except Exception as exc:
        logger.error("Failed to register session in router: %s", exc)
        raise RouterUnreachableError(f"Failed to register session: {exc}") from exc


async def revoke_session_in_router(
    router_addr: str,
    admin_api_key: str,
    group_id: str,
    timeout: float = 2.0,
    *,
    client: httpx.AsyncClient | None = None,
) -> None:
    """Best-effort removal of a session group from the Router's registry."""
    try:
        async with _use_client(client, timeout) as c:
            resp = await c.post(
                f"{router_addr}/remove_session",
                json={"group_id": group_id},
                headers={"Authorization": f"Bearer {admin_api_key}"},
                timeout=timeout,
            )
        if resp.status_code != 200:
            logger.warning(
                "remove_session returned %d: %s", resp.status_code, resp.text
            )
    except Exception as exc:
        logger.warning("Failed to remove group %s in router: %s", group_id, exc)


@async_httpx_retry
async def get_all_worker_addrs(
    router_addr: str,
    admin_api_key: str,
    timeout: float,
    *,
    client: httpx.AsyncClient | None = None,
) -> list[str]:
    """Fetch all worker addresses from the Router (for broadcast).

    GET ``{router_addr}/workers`` with admin key auth.
    """
    try:
        async with _use_client(client, timeout) as c:
            resp = await c.get(
                f"{router_addr}/workers",
                headers={"Authorization": f"Bearer {admin_api_key}"},
                timeout=timeout,
            )
        resp.raise_for_status()
        data = resp.json()
        return [w["addr"] for w in data.get("workers", [])]
    except Exception as exc:
        raise RouterUnreachableError(f"Failed to get workers: {exc}") from exc


@async_httpx_retry
async def register_model_in_router(
    router_addr: str,
    model: str,
    url: str,
    api_key: str | None,
    data_proxy_addrs: list[str],
    admin_api_key: str,
    timeout: float,
    *,
    client: httpx.AsyncClient | None = None,
) -> dict:
    try:
        async with _use_client(client, timeout) as c:
            resp = await c.post(
                f"{router_addr}/register_model",
                json={
                    "model": model,
                    "url": url,
                    "api_key": api_key,
                    "data_proxy_addrs": data_proxy_addrs,
                },
                headers={"Authorization": f"Bearer {admin_api_key}"},
                timeout=timeout,
            )
        if resp.status_code == 503:
            raise RouterKeyRejectedError("No healthy workers", 503)
        resp.raise_for_status()
        return resp.json()
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        raise RouterUnreachableError(f"Router unreachable: {exc}") from exc


@async_httpx_retry
async def route_external_model(
    router_addr: str,
    name: str,
    admin_api_key: str,
    timeout: float,
    *,
    client: httpx.AsyncClient | None = None,
) -> dict:
    try:
        async with _use_client(client, timeout) as c:
            resp = await c.post(
                f"{router_addr}/route",
                json={"model": name},
                headers={"Authorization": f"Bearer {admin_api_key}"},
                timeout=timeout,
            )
        if resp.status_code == 404:
            raise RouterKeyRejectedError(f"Model '{name}' not found", 404)
        if resp.status_code == 503:
            raise RouterKeyRejectedError(f"No healthy workers for model '{name}'", 503)
        resp.raise_for_status()
        return resp.json()
    except RouterKeyRejectedError:
        raise
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        raise RouterUnreachableError(f"Router unreachable: {exc}") from exc


@async_httpx_retry
async def list_models_from_router(
    router_addr: str,
    admin_api_key: str,
    timeout: float,
    *,
    client: httpx.AsyncClient | None = None,
) -> list[str]:
    try:
        async with _use_client(client, timeout) as c:
            resp = await c.get(
                f"{router_addr}/models",
                headers={"Authorization": f"Bearer {admin_api_key}"},
                timeout=timeout,
            )
        resp.raise_for_status()
        return resp.json().get("models", [])
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        raise RouterUnreachableError(f"Router unreachable: {exc}") from exc


@async_httpx_retry
async def remove_model_from_router(
    router_addr: str,
    name: str,
    admin_api_key: str,
    timeout: float,
    *,
    client: httpx.AsyncClient | None = None,
) -> None:
    """Remove an external model from the router registry (best-effort rollback)."""
    try:
        async with _use_client(client, timeout) as c:
            resp = await c.post(
                f"{router_addr}/remove_model",
                json={"name": name},
                headers={"Authorization": f"Bearer {admin_api_key}"},
                timeout=timeout,
            )
        resp.raise_for_status()
    except Exception:
        pass  # Best-effort rollback; swallow errors


@async_httpx_retry
async def resolve_worker_addr(
    router_addr: str,
    admin_api_key: str,
    worker_id: str,
    timeout: float,
    *,
    client: httpx.AsyncClient | None = None,
) -> str:
    """Resolve a worker_id to its address via the Router.

    GET ``{router_addr}/resolve_worker/{worker_id}`` with admin key auth.

    Raises
    ------
    RouterKeyRejectedError
        Worker ID not found (404).
    RouterUnreachableError
        Router connection failed.
    """
    try:
        async with _use_client(client, timeout) as c:
            resp = await c.get(
                f"{router_addr}/resolve_worker/{worker_id}",
                headers={"Authorization": f"Bearer {admin_api_key}"},
                timeout=timeout,
            )
        if resp.status_code == 404:
            data = resp.json()
            raise RouterKeyRejectedError(
                data.get("detail", f"Worker ID {worker_id} not found"), 404
            )
        resp.raise_for_status()
        return resp.json()["worker_addr"]
    except RouterKeyRejectedError:
        raise
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        raise RouterUnreachableError(
            f"Router unreachable for resolve_worker: {exc}"
        ) from exc
    except httpx.TransportError as exc:
        raise RouterUnreachableError(
            f"Router transport error for resolve_worker: {exc}"
        ) from exc
    except Exception as exc:
        raise RouterUnreachableError(
            f"Failed to resolve worker {worker_id}: {exc}"
        ) from exc


def _forwarding_headers(raw_headers: dict[str, str]) -> dict[str, str]:
    """Build headers to forward to data proxy.

    Strips hop-by-hop headers (``host``, ``content-length``,
    ``transfer-encoding``) that are incompatible with proxied requests.
    """
    skip = {"host", "content-length", "transfer-encoding"}
    return {k: v for k, v in raw_headers.items() if k.lower() not in skip}


async def forward_sse_stream(
    upstream_url: str,
    body: bytes,
    headers: dict[str, str],
    timeout: float | None = None,
    *,
    client: httpx.AsyncClient | None = None,
) -> AsyncGenerator[bytes, None]:
    """True SSE streaming proxy — yields bytes as they arrive from upstream.

    Uses ``httpx.AsyncClient.stream()`` so the client sees tokens
    as soon as the data proxy emits them.

    On upstream HTTP errors or mid-stream failures, an SSE error event
    (``data: {"error": ...}``) is emitted so clients can distinguish a
    clean end-of-stream from a backend failure.

    Note: streaming requires owning the client context for the duration
    of the stream, so a per-request client is always created here.
    The ``client`` parameter is accepted for API consistency but not used.
    """
    import json as _json

    fwd_headers = _forwarding_headers(headers)
    try:
        async with create_httpx_client(timeout=httpx.Timeout(timeout)) as c:
            async with c.stream(
                "POST", upstream_url, content=body, headers=fwd_headers
            ) as resp:
                if resp.status_code != 200:
                    error_body = await resp.aread()
                    try:
                        detail = _json.loads(error_body)
                    except Exception:
                        detail = error_body.decode("utf-8", errors="replace")
                    error_event = _json.dumps(
                        {"error": detail, "status_code": resp.status_code}
                    )
                    yield f"data: {error_event}\n\n".encode()
                    return
                async for chunk in resp.aiter_bytes():
                    yield chunk
    except httpx.TransportError as exc:
        logger.error("SSE transport error from %s: %s", upstream_url, exc)
        error_event = _json.dumps({"error": str(exc)})
        yield f"data: {error_event}\n\n".encode()
    except Exception as exc:
        logger.error("SSE stream error from %s: %s", upstream_url, exc)
        error_event = _json.dumps({"error": str(exc)})
        yield f"data: {error_event}\n\n".encode()


@async_httpx_retry
async def forward_request(
    upstream_url: str,
    body: bytes,
    headers: dict[str, str],
    timeout: float = 120.0,
    *,
    client: httpx.AsyncClient | None = None,
) -> httpx.Response:
    """Forward a non-streaming request to upstream, return full response."""
    fwd_headers = _forwarding_headers(headers)
    async with _use_client(client, timeout) as c:
        return await c.post(
            upstream_url, content=body, headers=fwd_headers, timeout=timeout
        )


async def broadcast_to_workers(
    worker_addrs: list[str],
    path: str,
    body: bytes,
    headers: dict[str, str],
    timeout: float = 10.0,
    *,
    client: httpx.AsyncClient | None = None,
) -> list[dict[str, Any]]:
    """Broadcast a request to all workers (best-effort).

    Returns a list of per-worker result dicts::

        [{"worker_addr": "...", "status": 200, "ok": True}, ...]

    Failed workers get ``ok=False`` with an ``error`` field.
    """

    async def _call(addr: str) -> dict[str, Any]:
        try:
            async with _use_client(client, timeout) as c:
                resp = await c.post(
                    f"{addr}{path}",
                    content=body,
                    headers=_forwarding_headers(headers),
                    timeout=timeout,
                )
            return {
                "worker_addr": addr,
                "status": resp.status_code,
                "ok": resp.status_code < 400,
            }
        except Exception as exc:
            return {
                "worker_addr": addr,
                "status": 502,
                "ok": False,
                "error": str(exc),
            }

    tasks = [_call(addr) for addr in worker_addrs]
    results = await asyncio.gather(*tasks)
    return list(results)
