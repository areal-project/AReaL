# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from areal.v2.cli.client import (
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


class RouterClient(BaseHTTPClient):
    def register_worker(self, addr: str, *, timeout: float = 10.0) -> dict[str, Any]:
        return self._post("/register", payload={"worker_addr": addr}, timeout=timeout)

    def unregister_worker(self, addr: str, *, timeout: float = 10.0) -> dict[str, Any]:
        return self._post("/unregister", payload={"worker_addr": addr}, timeout=timeout)

    def remove_model(self, name: str, *, timeout: float = 10.0) -> dict[str, Any]:
        return self._post("/remove_model", payload={"name": name}, timeout=timeout)
