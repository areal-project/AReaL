# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import httpx

from areal.utils import logging

logger = logging.getLogger("TrainGateway")


class RouterUnreachableError(Exception):
    pass


class RouterKeyRejectedError(Exception):
    def __init__(self, detail: str, status_code: int = 404):
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


async def query_router(
    router_addr: str,
    api_key: str,
    timeout: float = 2.0,
    *,
    admin_api_key: str | None = None,
    client: httpx.AsyncClient,
) -> str:
    payload = {"api_key": api_key}
    try:
        headers = {}
        if admin_api_key is not None:
            headers["Authorization"] = f"Bearer {admin_api_key}"
        resp = await client.post(
            f"{router_addr}/route",
            json=payload,
            headers=headers,
            timeout=timeout,
        )
        if resp.status_code in {404, 503}:
            try:
                data = resp.json()
                detail = data.get("detail", data.get("error", resp.text))
            except Exception:
                detail = resp.text
            raise RouterKeyRejectedError(detail, resp.status_code)
        resp.raise_for_status()
        return resp.json()["model_addr"]
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        raise RouterUnreachableError(f"Router unreachable: {exc}") from exc
    except httpx.TimeoutException as exc:
        raise RouterUnreachableError(f"Router timed out: {exc}") from exc
    except httpx.HTTPStatusError as exc:
        raise RouterUnreachableError(
            f"Router returned HTTP {exc.response.status_code}: {exc}"
        ) from exc


def _forwarding_headers(raw_headers: dict[str, str]) -> dict[str, str]:
    skip = {"host", "content-length", "transfer-encoding"}
    return {k: v for k, v in raw_headers.items() if k.lower() not in skip}


async def forward_request(
    upstream_url: str,
    body: bytes,
    headers: dict[str, str],
    timeout: float = 600.0,
    *,
    client: httpx.AsyncClient,
) -> httpx.Response:
    fwd_headers = _forwarding_headers(headers)
    resp = await client.post(
        upstream_url,
        content=body,
        headers=fwd_headers,
        timeout=timeout,
    )
    return resp
