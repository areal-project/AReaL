# SPDX-License-Identifier: Apache-2.0

"""Minimal stdlib HTTP client for the inference gateway.

Uses ``urllib.request`` so the CLI stays light. Only the endpoints the CLI
needs are wrapped; for ``areal inf run`` that's just ``GET /health``.
"""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class GatewayError(Exception):
    pass


class GatewayUnreachable(GatewayError):
    pass


class GatewayStatusError(GatewayError):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"gateway returned HTTP {status}: {body[:200]}")
        self.status = status
        self.body = body


@dataclass
class GatewayClient:
    url: str
    admin_api_key: str | None = None
    timeout: float = 5.0

    def _request(
        self, path: str, method: str = "GET", body: dict | None = None
    ) -> Any:
        full = self.url.rstrip("/") + path
        headers = {"Accept": "application/json"}
        data: bytes | None = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if self.admin_api_key:
            headers["Authorization"] = f"Bearer {self.admin_api_key}"

        req = urllib.request.Request(full, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            raw = e.read() if hasattr(e, "read") else b""
            raise GatewayStatusError(e.code, raw.decode("utf-8", "replace")) from e
        except (urllib.error.URLError, socket.timeout, ConnectionError) as e:
            raise GatewayUnreachable(f"gateway at {self.url} unreachable: {e}") from e
        if not raw:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return raw.decode("utf-8", "replace")

    def health(self) -> dict:
        return self._request("/health")
