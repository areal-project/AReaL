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
            f"{self.base}/models",
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

    def start_session(
        self,
        *,
        model: str,
        task_id: str,
        group_size: int,
        timeout: float = 30.0,
    ) -> dict:
        return _request(
            f"{self.base}/rl/start_session",
            method="POST",
            payload={"model": model, "task_id": task_id, "group_size": group_size},
            bearer=self.key,
            timeout=timeout,
        )

    def set_reward(
        self,
        *,
        session_api_key: str,
        reward: float,
        model: str | None = None,
        timeout: float = 10.0,
    ) -> dict:
        payload: dict = {"reward": reward}
        if model:
            payload["model"] = model
        return _request(
            f"{self.base}/rl/set_reward",
            method="POST",
            payload=payload,
            bearer=session_api_key,
            timeout=timeout,
        )

    def export_trajectories(
        self,
        *,
        session_ids: list[str],
        group_id: str | None = None,
        remove_session: bool = True,
        discount: float = 1.0,
        style: str = "individual",
        timeout: float = 30.0,
    ) -> dict:
        payload: dict = {
            "session_ids": session_ids,
            "remove_session": remove_session,
            "discount": discount,
            "style": style,
        }
        if group_id:
            payload["group_id"] = group_id
        return _request(
            f"{self.base}/export_trajectories",
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
