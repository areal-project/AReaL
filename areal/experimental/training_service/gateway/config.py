# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GatewayConfig:
    host: str = "0.0.0.0"
    port: int = 9080
    router_addr: str = ""
    admin_api_key: str = "areal-admin-key"
    log_level: str = "warning"
    router_timeout: float = 2.0
    forward_timeout: float = 600.0
