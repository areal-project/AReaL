# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import httpx

from ..auth import admin_headers, is_source_visible_default_admin_key
from ..memory_transport import (
    MEMORY_ASSIGNMENT_PIN_FIELD,
    MEMORY_CONTROL_AUTHORIZED_FIELD,
    MemoryAssignmentPinWireV1,
)


class DataProxyClient:
    def __init__(
        self,
        data_proxy_addr: str,
        memory_control_api_key: str = "",
    ) -> None:
        if type(memory_control_api_key) is not str:
            raise TypeError("memory_control_api_key must be a string")
        if memory_control_api_key and not memory_control_api_key.strip():
            raise ValueError("memory_control_api_key must not be blank")
        if is_source_visible_default_admin_key(memory_control_api_key):
            raise ValueError(
                "memory_control_api_key must not use a source-visible default key"
            )
        self._addr = data_proxy_addr
        self._memory_control_api_key = memory_control_api_key
        self._memory_control_headers = (
            admin_headers(memory_control_api_key) if memory_control_api_key else {}
        )
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
        if memory_control_authorized and not self._memory_control_api_key:
            raise ValueError(
                "memory_control_authorized requires a dedicated Memory control key"
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
            headers=(
                self._memory_control_headers if memory_control_authorized else None
            ),
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
