# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

from ..auth import DEFAULT_ADMIN_API_KEY


@dataclass
class RouterConfig:
    host: str = "0.0.0.0"
    port: int = 8081
    admin_api_key: str = DEFAULT_ADMIN_API_KEY
    poll_interval: float = 5.0
    worker_health_timeout: float = 2.0
    log_level: str = "warning"
