# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RouterConfig:
    host: str = "0.0.0.0"
    port: int = 9081
    admin_api_key: str = "areal-admin-key"
    log_level: str = "warning"
    poll_interval: float = 5.0
    worker_health_timeout: float = 2.0
