# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

from ..auth import DEFAULT_ADMIN_API_KEY, is_source_visible_default_admin_key


@dataclass
class GatewayConfig:
    host: str = "0.0.0.0"
    port: int = 8080
    admin_api_key: str = DEFAULT_ADMIN_API_KEY
    memory_control_api_key: str = ""
    router_addr: str = "http://localhost:8081"
    router_timeout: float = 2.0
    forward_timeout: float = 120.0
    log_level: str = "warning"

    def __post_init__(self) -> None:
        if type(self.admin_api_key) is not str:
            raise TypeError("admin_api_key must be a string")
        if not self.admin_api_key.strip():
            raise ValueError("admin_api_key must not be blank")
        if type(self.memory_control_api_key) is not str:
            raise TypeError("memory_control_api_key must be a string")
        if self.memory_control_api_key and not self.memory_control_api_key.strip():
            raise ValueError("memory_control_api_key must not be blank")
        if is_source_visible_default_admin_key(self.memory_control_api_key):
            raise ValueError(
                "memory_control_api_key must not use a source-visible default key"
            )
        if (
            self.memory_control_api_key
            and self.memory_control_api_key == self.admin_api_key
        ):
            raise ValueError(
                "memory_control_api_key must differ from the external admin_api_key"
            )

    @property
    def memory_control_enabled(self) -> bool:
        return bool(self.memory_control_api_key) and not (
            is_source_visible_default_admin_key(self.admin_api_key)
        )
