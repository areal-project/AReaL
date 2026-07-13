# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

from ..auth import DEFAULT_ADMIN_API_KEY


@dataclass
class DataProxyConfig:
    host: str = "0.0.0.0"
    port: int = 9100
    worker_addr: str = "http://localhost:9000"
    admin_api_key: str = DEFAULT_ADMIN_API_KEY
    request_timeout: float = 600.0
    session_timeout: int = 3600
    log_level: str = "warning"
