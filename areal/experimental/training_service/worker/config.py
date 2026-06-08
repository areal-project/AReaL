# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainWorkerConfig:
    host: str = "0.0.0.0"
    port: int = 0
    admin_api_key: str = "areal-admin-key"
    log_level: str = "warning"
