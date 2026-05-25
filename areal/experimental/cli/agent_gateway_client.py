# SPDX-License-Identifier: Apache-2.0

"""Thin HTTP client for the *agent* gateway.

Talks to ``areal/experimental/agent_service/gateway/app.py`` plus its
``OpenResponsesBridge`` mount. Two endpoints matter for the CLI today:

- ``GET /health`` — liveness
- ``POST /v1/responses`` — single-turn chat keyed by ``user`` (the session key)

The agent gateway's primary chat protocol is a WebSocket frame channel at
``WS /ws`` (token in query string). We deliberately stick to the REST bridge
for the CLI surface so the client can be stdlib-only (urllib has no
WebSocket support and pulling in a WS lib would break the lightness
guarantee). REPL streaming uses the bridge synchronously — no token-by-token
delta for now; the bridge already aggregates events into a complete
response.
"""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class AgentGatewayError(Exception):
    """Base for all agent-gateway client errors."""


class AgentGatewayUnreachable(AgentGatewayError):
    """Connection failed or timed out (no HTTP response)."""


class AgentGatewayAuthError(AgentGatewayError):
    """Agent gateway returned 401/403."""


class AgentGatewayStatusError(AgentGatewayError):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"agent gateway returned HTTP {status}: {body[:200]}")
        self.status = status
        self.body = body


@dataclass
class AgentGatewayClient:
    url: str
    admin_api_key: str | None = None
    timeout: float = 60.0

    def _request(
        self,
        path: str,
        method: str = "GET",
        body: dict | None = None,
        admin: bool = False,
    ) -> Any:
        full = self.url.rstrip("/") + path
        headers = {"Accept": "application/json"}
        data: bytes | None = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if admin and self.admin_api_key:
            headers["Authorization"] = f"Bearer {self.admin_api_key}"

        req = urllib.request.Request(full, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            raw = e.read() if hasattr(e, "read") else b""
            if e.code in (401, 403):
                raise AgentGatewayAuthError(
                    f"agent gateway rejected admin key (HTTP {e.code})"
                ) from e
            raise AgentGatewayStatusError(
                e.code, raw.decode("utf-8", "replace")
            ) from e
        except (urllib.error.URLError, socket.timeout, ConnectionError) as e:
            raise AgentGatewayUnreachable(
                f"agent gateway at {self.url} unreachable: {e}"
            ) from e
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise AgentGatewayError(
                f"non-JSON response from agent gateway ({len(raw)} bytes): {raw[:200]!r}"
            ) from e

    def health(self) -> dict:
        return self._request("/health")

    def responses(
        self,
        *,
        session_key: str,
        message: str,
        model: str = "",
        instructions: str = "",
        tools: list | None = None,
        metadata: dict | None = None,
    ) -> dict:
        """POST /v1/responses — single agent turn.

        ``session_key`` is sent in the ``user`` field; the bridge derives
        affinity from it. Returns the full response dict (OpenResponses
        shape: ``{"id", "output": [...], "status", ...}``).
        """
        body = {
            "user": session_key,
            "model": model,
            "input": [
                {"role": "user", "content": message},
            ],
            "instructions": instructions,
            "tools": tools or [],
            "metadata": metadata or {},
        }
        return self._request("/v1/responses", method="POST", body=body, admin=True)


def extract_response_text(resp: dict) -> str:
    """Pull plain assistant text out of an OpenResponses payload."""
    out: list[str] = []
    for item in (resp or {}).get("output") or []:
        if item.get("type") == "message":
            for chunk in item.get("content") or []:
                if chunk.get("type") in ("output_text", "text"):
                    text = chunk.get("text", "")
                    if text:
                        out.append(text)
    return "".join(out)
