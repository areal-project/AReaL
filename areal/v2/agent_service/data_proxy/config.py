# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

from ..auth import is_source_visible_default_admin_key


@dataclass
class DataProxyConfig:
    host: str = "0.0.0.0"
    port: int = 9100
    worker_addr: str = "http://localhost:9000"
    # Empty disables privileged Memory assignment transport while preserving
    # ordinary Agent turns for standalone DataProxy users.
    memory_control_api_key: str = ""
    request_timeout: float = 600.0
    session_timeout: int = 3600
    log_level: str = "warning"

    def __post_init__(self) -> None:
        if type(self.memory_control_api_key) is not str:
            raise TypeError("memory_control_api_key must be a string")
        if self.memory_control_api_key and not self.memory_control_api_key.strip():
            raise ValueError("memory_control_api_key must not be blank")
        if is_source_visible_default_admin_key(self.memory_control_api_key):
            raise ValueError(
                "memory_control_api_key must not use a source-visible default key"
            )
