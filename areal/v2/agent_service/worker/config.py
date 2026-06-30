# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WorkerConfig:
    host: str = "0.0.0.0"
    port: int = 9000
    agent_cls_path: str = ""
    log_level: str = "warning"
