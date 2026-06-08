# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WeightUpdateControllerConfig:
    host: str = "127.0.0.1"
    port: int = 0
    admin_api_key: str = "areal-admin-key"
    log_level: str = "warning"
    setup_timeout: float = 30.0
    request_timeout: float = 300.0
    init_timeout_s: float = 300.0
    update_timeout_s: float = 120.0
