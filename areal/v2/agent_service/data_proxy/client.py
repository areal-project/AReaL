# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import httpx

from ..auth import DEFAULT_ADMIN_API_KEY, admin_headers
from ..memory_transport import (
    MEMORY_ASSIGNMENT_PIN_FIELD,
    MEMORY_CONTROL_AUTHORIZED_FIELD,
    MemoryAssignmentPinWireV1,
)


class DataProxyClient:
    def __init__(
        self,
        data_proxy_addr: str,
        admin_api_key: str = DEFAULT_ADMIN_API_KEY,
    ) -> None:
        self._addr = data_proxy_addr
        self._admin_headers = admin_headers(admin_api_key)
        self._http = httpx.AsyncClient(timeout=600.0)

    async def turn(
        self,
        session_key: str,
        message: str,
        run_id: str = "",
        queue_mode: str = "collect",
        metadata: dict[str, Any] | None = None,
        memory_assignment_pin: MemoryAssignmentPinWireV1 | None = None,
        memory_control_authorized: bool = False,
    ) -> dict[str, Any]:
        if type(memory_control_authorized) is not bool:
            raise TypeError("memory_control_authorized must be a bool")
        if memory_assignment_pin is not None and not memory_control_authorized:
            raise ValueError(
                "memory_assignment_pin requires trusted Memory control authorization"
            )
        body: dict[str, Any] = {
            "message": message,
            "run_id": run_id,
            "queue_mode": queue_mode,
            "metadata": metadata or {},
        }
        if memory_control_authorized:
            body[MEMORY_CONTROL_AUTHORIZED_FIELD] = True
        if memory_assignment_pin is not None:
            if type(memory_assignment_pin) is not MemoryAssignmentPinWireV1:
                raise TypeError(
                    "memory_assignment_pin must be a MemoryAssignmentPinWireV1"
                )
            body[MEMORY_ASSIGNMENT_PIN_FIELD] = memory_assignment_pin.to_wire()
        resp = await self._http.post(
            f"{self._addr}/session/{session_key}/turn",
            json=body,
            headers=(self._admin_headers if memory_control_authorized else None),
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
