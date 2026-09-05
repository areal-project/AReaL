# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from areal.v2.cli.client import BaseHTTPClient


class WeightUpdateClient(BaseHTTPClient):
    """Read-only view of a weight-update gateway."""

    def pairs(self, *, timeout: float = 5.0) -> list[dict[str, Any]]:
        return self._get("/pairs", timeout=timeout).get("pairs", [])
