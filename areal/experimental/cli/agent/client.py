# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class AgentHTTPError(Exception):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"HTTP {status}: {body}")
        self.status = status
        self.body = body


class AgentUnreachable(Exception):
    pass


def _request(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    bearer: str | None = None,
    timeout: float = 5.0,
) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode()
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        raise AgentHTTPError(exc.code, exc.read().decode(errors="replace")) from exc
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
        raise AgentUnreachable(str(exc)) from exc


class GatewayClient:
    def __init__(self, base_url: str, admin_api_key: str) -> None:
        self.base = base_url.rstrip("/")
        self.key = admin_api_key

    def health(self, *, timeout: float = 2.0) -> dict[str, Any]:
        return _request(f"{self.base}/health", timeout=timeout)


class RouterClient:
    def __init__(self, base_url: str, admin_api_key: str) -> None:
        self.base = base_url.rstrip("/")
        self.key = admin_api_key

    def health(self, *, timeout: float = 2.0) -> dict[str, Any]:
        return _request(f"{self.base}/health", timeout=timeout)

    def register_proxy(self, addr: str, *, timeout: float = 10.0) -> dict[str, Any]:
        return _request(
            f"{self.base}/register",
            method="POST",
            payload={"addr": addr},
            bearer=self.key,
            timeout=timeout,
        )


class DataProxyClient:
    def __init__(self, base_url: str) -> None:
        self.base = base_url.rstrip("/")

    def health(self, *, timeout: float = 2.0) -> dict[str, Any]:
        return _request(f"{self.base}/health", timeout=timeout)
