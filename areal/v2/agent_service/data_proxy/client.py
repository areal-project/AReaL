# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import httpx


class DataProxyClient:
    def __init__(self, data_proxy_addr: str) -> None:
        self._addr = data_proxy_addr
        self._http = httpx.AsyncClient(timeout=600.0)

    async def turn(
        self,
        session_key: str,
        message: str,
        run_id: str = "",
        queue_mode: str = "collect",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resp = await self._http.post(
            f"{self._addr}/session/{session_key}/turn",
            json={
                "message": message,
                "run_id": run_id,
                "queue_mode": queue_mode,
                "metadata": metadata or {},
            },
        )
        resp.raise_for_status()
        return resp.json()

    async def close_session(self, session_key: str) -> None:
        resp = await self._http.post(f"{self._addr}/session/{session_key}/close")
        resp.raise_for_status()

    async def get_history(self, session_key: str) -> list[dict[str, Any]]:
        resp = await self._http.get(f"{self._addr}/session/{session_key}/history")
        resp.raise_for_status()
        return resp.json()["history"]

    async def close(self) -> None:
        await self._http.aclose()
