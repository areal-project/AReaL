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
    # Appended for positional-call compatibility.  Empty preserves standalone
    # use; Controller-managed pairs always use an independently generated key.
    worker_hop_api_key: str = ""

    def __post_init__(self) -> None:
        for field_name in ("worker_hop_api_key", "memory_control_api_key"):
            value = getattr(self, field_name)
            if type(value) is not str:
                raise TypeError(f"{field_name} must be a string")
            if value and not value.strip():
                raise ValueError(f"{field_name} must not be blank")
            if is_source_visible_default_admin_key(value):
                raise ValueError(
                    f"{field_name} must not use a source-visible default key"
                )
        if (
            self.worker_hop_api_key
            and self.worker_hop_api_key == self.memory_control_api_key
        ):
            raise ValueError(
                "worker_hop_api_key must differ from memory_control_api_key"
            )
        if self.memory_control_api_key and not self.worker_hop_api_key:
            raise ValueError(
                "memory_control_api_key requires an independent worker_hop_api_key"
            )
