# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GatewayConfig:
    host: str = "0.0.0.0"
    port: int = 8090
    router_addr: str = ""
    admin_api_key: str = "areal-data-admin"
    forward_timeout: float = 60.0
    router_timeout: float = 2.0
