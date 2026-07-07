# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from areal.v2.cli.client import BaseHTTPClient


class RouterClient(BaseHTTPClient):
    def register_proxy(self, addr: str, *, timeout: float = 10.0) -> dict[str, Any]:
        return self._post("/register", payload={"addr": addr}, timeout=timeout)
