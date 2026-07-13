# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RouterConfig:
    host: str = "0.0.0.0"
    port: int = 8091
    admin_api_key: str = "areal-data-admin"
    poll_interval: float = 5.0
    worker_health_timeout: float = 3.0
