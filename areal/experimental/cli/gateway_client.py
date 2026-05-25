# SPDX-License-Identifier: Apache-2.0

"""Thin HTTP client for the inference gateway.

Uses ``urllib.request`` (stdlib) on purpose so the CLI stays light — no
``httpx`` / ``aiohttp`` dependency. Only the small set of endpoints the CLI
needs is wrapped: ``GET /health``, ``GET /models``, ``POST /register_model``,
``POST /deregister_model``, and streaming ``POST /chat/completions``.
"""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any


class GatewayError(Exception):
    """Base for all gateway client errors."""


class GatewayUnreachable(GatewayError):
    """Connection failed or timed out (no HTTP response)."""


class GatewayAuthError(GatewayError):
    """Gateway returned 401/403 (admin key wrong or missing)."""


class GatewayStatusError(GatewayError):
    """Gateway returned a non-2xx response we can't classify more precisely."""

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
                raise GatewayAuthError(
                    f"gateway rejected admin key (HTTP {e.code})"
                ) from e
            raise GatewayStatusError(e.code, raw.decode("utf-8", "replace")) from e
        except (urllib.error.URLError, socket.timeout, ConnectionError) as e:
            raise GatewayUnreachable(f"gateway at {self.url} unreachable: {e}") from e
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise GatewayError(
                f"non-JSON response from gateway ({len(raw)} bytes): {raw[:200]!r}"
            ) from e

    # ------------------------------------------------------------------
    # Read-only endpoints
    # ------------------------------------------------------------------

    def health(self) -> dict:
        return self._request("/health")

    def models(self) -> list[str]:
        data = self._request("/models", admin=True)
        if isinstance(data, dict) and "models" in data:
            return list(data["models"])
        if isinstance(data, list):
            return list(data)
        return []

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def register_model(
        self,
        name: str,
        *,
        url: str = "",
        api_key: str | None = None,
        data_proxy_addrs: list[str] | None = None,
    ) -> dict:
        """POST /register_model.

        For external models, pass ``url`` (provider base URL) and ``api_key``.
        For internal models, pass the launched ``data_proxy_addrs`` (and leave
        ``url`` empty so the router uses the data proxies as workers).
        """
        body: dict = {"model": name, "url": url}
        if api_key is not None:
            body["api_key"] = api_key
        if data_proxy_addrs:
            body["data_proxy_addrs"] = list(data_proxy_addrs)
        return self._request(
            "/register_model", method="POST", body=body, admin=True
        )

    def deregister_model(self, name: str) -> dict:
        """POST /deregister_model (added in design §10.2)."""
        return self._request(
            "/deregister_model",
            method="POST",
            body={"model": name},
            admin=True,
        )

    # ------------------------------------------------------------------
    # RL session coordination (called from `areal agent ...`)
    # ------------------------------------------------------------------

    def start_rl_session(self, *, model: str = "", **extra: Any) -> dict:
        """POST /rl/start_session.

        Returns the worker's response (HTTP 201) containing ``group_id`` and
        ``sessions`` (each with an ``api_key`` field). The caller decides
        whether to retain the session key.
        """
        body: dict = {}
        if model:
            body["model"] = model
        body.update(extra)
        return self._request(
            "/rl/start_session", method="POST", body=body, admin=True
        )

    def set_rl_reward(
        self,
        *,
        session_id: str,
        reward: float,
        model: str | None = None,
        **extra: Any,
    ) -> dict:
        """POST /rl/set_reward."""
        body: dict = {"session_id": session_id, "reward": reward}
        if model:
            body["model"] = model
        body.update(extra)
        return self._request(
            "/rl/set_reward", method="POST", body=body, admin=True
        )

    # ------------------------------------------------------------------
    # Chat completions — non-streaming and streaming
    # ------------------------------------------------------------------

    def chat_completion(
        self,
        *,
        model: str,
        messages: list[dict],
        stream: bool = False,
        **gen_kwargs: Any,
    ) -> dict | Iterator[str]:
        """POST /chat/completions.

        - ``stream=False``: returns the full OpenAI-style response dict.
        - ``stream=True``: returns an iterator that yields *content deltas*
          (string fragments) as the gateway streams them via SSE.
        """
        body: dict = {"model": model, "messages": messages, "stream": stream}
        body.update(gen_kwargs)
        if not stream:
            return self._request(
                "/chat/completions", method="POST", body=body, admin=True
            )
        return self._stream_chat(body)

    def _stream_chat(self, body: dict) -> Iterator[str]:
        full = self.url.rstrip("/") + "/chat/completions"
        headers = {
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
        }
        if self.admin_api_key:
            headers["Authorization"] = f"Bearer {self.admin_api_key}"
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(full, data=data, headers=headers, method="POST")
        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            raw = e.read() if hasattr(e, "read") else b""
            if e.code in (401, 403):
                raise GatewayAuthError(
                    f"gateway rejected admin key (HTTP {e.code})"
                ) from e
            raise GatewayStatusError(e.code, raw.decode("utf-8", "replace")) from e
        except (urllib.error.URLError, socket.timeout, ConnectionError) as e:
            raise GatewayUnreachable(f"gateway at {self.url} unreachable: {e}") from e

        with resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", "replace").rstrip("\r\n")
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:") :].strip()
                if payload == "[DONE]":
                    return
                if not payload:
                    continue
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                for ch in obj.get("choices") or []:
                    delta = ch.get("delta") or {}
                    content = delta.get("content")
                    if content:
                        yield content
