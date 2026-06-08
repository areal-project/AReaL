# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrainDataProxyConfig:
    host: str = "0.0.0.0"
    port: int = 9082
    worker_addrs: list[str] = field(default_factory=list)
    admin_api_key: str = "areal-admin-key"
    log_level: str = "warning"
    request_timeout: float = 600.0
    warmup_timeout: float = 120.0
