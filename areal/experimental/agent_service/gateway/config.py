# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

from ..auth import DEFAULT_ADMIN_API_KEY


@dataclass
class GatewayConfig:
    host: str = "0.0.0.0"
    port: int = 8080
    admin_api_key: str = DEFAULT_ADMIN_API_KEY
    router_addr: str = "http://localhost:8081"
    router_timeout: float = 2.0
    forward_timeout: float = 120.0
    log_level: str = "warning"
