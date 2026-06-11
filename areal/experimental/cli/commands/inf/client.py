# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import urllib.error
import urllib.request


class GatewayUnreachable(Exception):
    pass


class GatewayHTTPError(Exception):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"HTTP {status}: {body}")
        self.status = status
        self.body = body


def _request(
    url: str,
    *,
    method: str = "GET",
    payload: dict | None = None,
    bearer: str | None = None,
    timeout: float = 5.0,
) -> dict:
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
    except urllib.error.HTTPError as e:
        raise GatewayHTTPError(e.code, e.read().decode(errors="replace")) from e
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
        raise GatewayUnreachable(str(e)) from e


class GatewayClient:
    def __init__(self, base_url: str, admin_api_key: str) -> None:
        self.base = base_url.rstrip("/")
        self.key = admin_api_key

    def health(self, *, timeout: float = 2.0) -> dict:
        return _request(f"{self.base}/health", timeout=timeout)

    def list_models(self, *, timeout: float = 5.0) -> dict:
        return _request(
            f"{self.base}/v1/models",
            bearer=self.key,
            timeout=timeout,
        )

    def register_model(self, payload: dict, *, timeout: float = 30.0) -> dict:
        return _request(
            f"{self.base}/register_model",
            method="POST",
            payload=payload,
            bearer=self.key,
            timeout=timeout,
        )


class RouterClient:
    def __init__(self, base_url: str, admin_api_key: str) -> None:
        self.base = base_url.rstrip("/")
        self.key = admin_api_key

    def health(self, *, timeout: float = 2.0) -> dict:
        return _request(f"{self.base}/health", timeout=timeout)

    def register_worker(self, addr: str, *, timeout: float = 10.0) -> dict:
        return _request(
            f"{self.base}/register",
            method="POST",
            payload={"worker_addr": addr},
            bearer=self.key,
            timeout=timeout,
        )

    def unregister_worker(self, addr: str, *, timeout: float = 10.0) -> dict:
        return _request(
            f"{self.base}/unregister",
            method="POST",
            payload={"worker_addr": addr},
            bearer=self.key,
            timeout=timeout,
        )

    def remove_model(self, name: str, *, timeout: float = 10.0) -> dict:
        return _request(
            f"{self.base}/remove_model",
            method="POST",
            payload={"name": name},
            bearer=self.key,
            timeout=timeout,
        )
