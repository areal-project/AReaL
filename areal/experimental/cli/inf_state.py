# SPDX-License-Identifier: Apache-2.0

"""State files for ``areal inf`` services.

Layout under ``$AREAL_HOME`` (default ``~/.areal``)::

    inf/
      current-service          plain text: name of the default service
      services/<name>.json     service info (PIDs, ports, admin key, ...)
      logs/<name>/             gateway.log, router.log, <model>.log

Service state is updated by ``areal inf run`` on launch and consulted by
``stop`` / ``status`` / ``ps`` / ``models`` / ``logs``. There is no hidden
supervisor process — reconciliation works by comparing local state against
live PIDs and the gateway's HTTP health endpoint.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from areal.experimental.cli.state import areal_home, pid_alive


def inf_dir() -> Path:
    d = areal_home() / "inf"
    d.mkdir(parents=True, exist_ok=True)
    return d


def services_dir() -> Path:
    d = inf_dir() / "services"
    d.mkdir(parents=True, exist_ok=True)
    return d


def service_logs_dir(name: str) -> Path:
    d = inf_dir() / "logs" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def current_service_file() -> Path:
    return inf_dir() / "current-service"


def service_state_path(name: str) -> Path:
    return services_dir() / f"{name}.json"


@dataclass
class ServiceState:
    name: str
    gateway_host: str
    gateway_port: int
    router_host: str
    router_port: int
    gateway_pid: int
    router_pid: int
    admin_api_key: str
    mode: str = "detached"  # detached | interactive
    log_level: str = "info"
    routing_strategy: str = "round_robin"
    created_at: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def gateway_url(self) -> str:
        host = "127.0.0.1" if self.gateway_host in ("0.0.0.0", "::") else self.gateway_host
        return f"http://{host}:{self.gateway_port}"

    @property
    def router_url(self) -> str:
        host = "127.0.0.1" if self.router_host in ("0.0.0.0", "::") else self.router_host
        return f"http://{host}:{self.router_port}"

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
            data = json.load(f)
        return cls(**data)

    def remove(self) -> None:
        p = service_state_path(self.name)
        if p.exists():
            p.unlink()


def list_service_names() -> list[str]:
    d = services_dir()
    return sorted(p.stem for p in d.glob("*.json"))


def load_all_services() -> list[ServiceState]:
    out: list[ServiceState] = []
    for name in list_service_names():
        try:
            out.append(ServiceState.load(name))
        except (ValueError, FileNotFoundError, TypeError):
            continue
    return out


def get_current_service() -> str | None:
    p = current_service_file()
    if not p.exists():
        return None
    name = p.read_text().strip()
    return name or None


def set_current_service(name: str | None) -> None:
    p = current_service_file()
    p.parent.mkdir(parents=True, exist_ok=True)
    if name is None:
        if p.exists():
            p.unlink()
        return
    p.write_text(name + "\n")


def resolve_service_name(explicit: str | None) -> str:
    """Apply design §11.1 resolution rules for a target service name.

    Order: explicit arg > ``current-service`` file > sole known service > error.
    Does NOT contact the gateway; callers do health checks separately.
    """
    if explicit:
        return explicit
    current = get_current_service()
    if current:
        return current
    names = list_service_names()
    if len(names) == 1:
        return names[0]
    if not names:
        raise SystemExit(
            "No inference service found. "
            "Start one with `areal inf run` first."
        )
    raise SystemExit(
        "Multiple inference services exist; specify one with --service. "
        f"Known: {', '.join(names)}."
    )


def liveness_summary(state: ServiceState) -> dict[str, bool]:
    """Cheap process-level liveness probe (no HTTP)."""
    return {
        "gateway_pid_alive": pid_alive(state.gateway_pid),
        "router_pid_alive": pid_alive(state.router_pid),
    }
