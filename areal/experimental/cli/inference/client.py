# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from areal.experimental.cli.client import (
    BaseHTTPClient,
    ServiceHTTPError,
    ServiceUnreachable,
)

# Legacy aliases — preserve the old names imported by inference subcommands
# (and downstream callers) so this swap is mechanical. New code should use
# the scaffold names directly.
GatewayHTTPError = ServiceHTTPError
GatewayUnreachable = ServiceUnreachable


class GatewayClient(BaseHTTPClient):
    def list_models(self, *, timeout: float = 5.0) -> dict[str, Any]:
        return self._get("/models", timeout=timeout)

    def register_model(self, payload: dict, *, timeout: float = 30.0) -> dict[str, Any]:
        return self._post("/register_model", payload=payload, timeout=timeout)

    def start_session(
        self,
        *,
        model: str,
        task_id: str,
        group_size: int,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        return self._post(
            "/rl/start_session",
            payload={"model": model, "task_id": task_id, "group_size": group_size},
            timeout=timeout,
        )

    def set_reward(
        self,
        *,
        session_api_key: str,
        reward: float,
        model: str | None = None,
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        payload: dict = {"reward": reward}
        if model:
            payload["model"] = model
        # Per-session bearer token, not the gateway admin key — bypass the
        # base client's auth and pass session_api_key explicitly.
        from areal.experimental.cli.client import request_json

        return request_json(
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
    ) -> dict[str, Any]:
        payload: dict = {
            "session_ids": session_ids,
            "remove_session": remove_session,
            "discount": discount,
            "style": style,
        }
        if group_id:
            payload["group_id"] = group_id
        return self._post("/export_trajectories", payload=payload, timeout=timeout)


class RouterClient(BaseHTTPClient):
    def register_worker(self, addr: str, *, timeout: float = 10.0) -> dict[str, Any]:
        return self._post("/register", payload={"worker_addr": addr}, timeout=timeout)

    def unregister_worker(self, addr: str, *, timeout: float = 10.0) -> dict[str, Any]:
        return self._post("/unregister", payload={"worker_addr": addr}, timeout=timeout)

    def remove_model(self, name: str, *, timeout: float = 10.0) -> dict[str, Any]:
        return self._post("/remove_model", payload={"name": name}, timeout=timeout)
