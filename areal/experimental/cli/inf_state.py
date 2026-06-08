# SPDX-License-Identifier: Apache-2.0

"""State files for ``areal inf`` services.

A service is a long-running ``inf_supervisor`` process that holds the
``RolloutControllerV2`` (which in turn manages sglang workers + router +
gateway + data-proxies). State is the minimum needed to find and stop
the supervisor and to display useful diagnostics; the supervisor itself
is the source of truth for its children.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from areal.experimental.cli.state import areal_home, pid_alive, sanitize_name


def inf_dir() -> Path:
    d = areal_home() / "inf"
    d.mkdir(parents=True, exist_ok=True)
    return d


def services_dir() -> Path:
    d = inf_dir() / "services"
    d.mkdir(parents=True, exist_ok=True)
    return d


def service_logs_dir(name: str) -> Path:
    d = inf_dir() / "logs" / sanitize_name(name)
    d.mkdir(parents=True, exist_ok=True)
    return d


def service_state_path(name: str) -> Path:
    return services_dir() / f"{sanitize_name(name)}.json"


def service_ready_marker(name: str) -> Path:
    return services_dir() / f"{sanitize_name(name)}.ready"


@dataclass
class ServiceState:
    name: str
    supervisor_pid: int
    config_path: str
    overrides: list[str] = field(default_factory=list)
    gateway_addr: str = ""
    router_addr: str = ""
    server_addrs: list[str] = field(default_factory=list)
    created_at: float = 0.0
    ready_at: float = 0.0

    def save(self) -> None:
        p = service_state_path(self.name)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(asdict(self), f, indent=2)
        os.replace(tmp, p)

    @classmethod
    def load(cls, name: str) -> ServiceState:
        p = service_state_path(name)
        if not p.exists():
            raise FileNotFoundError(f"No service state for {name!r} at {p}")
        with open(p) as f:
            return cls(**json.load(f))

    def remove(self) -> None:
        p = service_state_path(self.name)
        if p.exists():
            p.unlink()
        ready = service_ready_marker(self.name)
        if ready.exists():
            ready.unlink()


def supervisor_alive(state: ServiceState) -> bool:
    return pid_alive(state.supervisor_pid)


def get_current_service() -> str | None:
    p = inf_dir() / "current-service"
    if not p.exists():
        return None
    name = p.read_text().strip()
    return name or None


def set_current_service(name: str | None) -> None:
    p = inf_dir() / "current-service"
    p.parent.mkdir(parents=True, exist_ok=True)
    if name is None:
        if p.exists():
            p.unlink()
        return
    p.write_text(name + "\n")
