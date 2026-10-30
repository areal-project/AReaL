# SPDX-License-Identifier: Apache-2.0

"""Configuration for the Router service."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RouterConfig:
    """Configuration for the routing service.

    The router owns worker registry, session pinning, and routing
    strategy. It is a separate FastAPI process from the gateway.
    """

    host: str = "0.0.0.0"
    port: int = 8081
    admin_api_key: str = "areal-admin-key"
    poll_interval: float = 5.0  # seconds between health polls
    worker_health_timeout: float = 2.0  # seconds per health check
    backend_health_timeout: float = 10.0  # native inference can delay health probes
    routing_strategy: str = "round_robin"  # "round_robin" (only supported strategy)
    log_level: str = "warning"
