# SPDX-License-Identifier: Apache-2.0

"""Configuration for the Inference Gateway service."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GatewayConfig:
    """Configuration for the inference gateway.

    The gateway only needs ``admin_api_key`` and ``router_addr`` —
    all worker state and session pinning live in the Router service.
    """

    host: str = "0.0.0.0"
    port: int = 8080
    admin_api_key: str = "areal-admin-key"
    router_addr: str = "http://localhost:8081"
    router_timeout: float = 2.0  # seconds for /route call
    forward_timeout: float = 120.0  # seconds for forwarding to data proxy
    log_level: str = "warning"
