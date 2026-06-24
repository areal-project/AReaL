# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_SERVICE = "default"


def areal_home() -> Path:
    env = os.environ.get("AREAL_HOME")
    root = Path(env).expanduser() if env else Path.home() / ".areal"
    root.mkdir(parents=True, exist_ok=True)
    return root


def agent_root() -> Path:
    root = areal_home() / "agent"
    root.mkdir(parents=True, exist_ok=True)
    return root


def services_dir() -> Path:
    path = agent_root() / "services"
    path.mkdir(parents=True, exist_ok=True)
    return path


def logs_root() -> Path:
    path = agent_root() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def service_logs_dir(service: str) -> Path:
    path = logs_root() / service
    path.mkdir(parents=True, exist_ok=True)
    return path


def service_state_path(service: str) -> Path:
    return services_dir() / f"{service}.json"


def current_service_path() -> Path:
    return agent_root() / "current-service"


def atomic_write_json(path: Path, data: Any, *, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        f.write(json.dumps(data, indent=indent) + "\n")
    os.replace(tmp, path)


@dataclass
class ProcessState:
    component: str
    pid: int
    url: str
    log_file: str


@dataclass
class PairState:
    index: int
    worker: ProcessState
    data_proxy: ProcessState


@dataclass
class ServiceState:
    service: str
    launch_mode: str
    agent: str
    admin_api_key: str
    gateway: ProcessState
    router: ProcessState
    pairs: list[PairState]
    inf_addr: str = ""
    inf_api_key: str = ""
    inf_model: str = ""
    session_timeout: float = 1800.0
    health_poll_interval: float = 5.0
    drain_timeout: float = 30.0
    created_at: float = field(default_factory=time.time)

    def save(self) -> None:
        atomic_write_json(service_state_path(self.service), asdict(self))
        current_service_path().write_text(self.service + "\n")

    @classmethod
    def load(cls, service: str) -> ServiceState:
        with open(service_state_path(service)) as f:
            raw = json.load(f)
        raw["gateway"] = ProcessState(**raw["gateway"])
        raw["router"] = ProcessState(**raw["router"])
        raw["pairs"] = [
            PairState(
                index=pair["index"],
                worker=ProcessState(**pair["worker"]),
                data_proxy=ProcessState(**pair["data_proxy"]),
            )
            for pair in raw.get("pairs", [])
        ]
        return cls(**raw)

    @classmethod
    def remove(cls, service: str) -> None:
        path = service_state_path(service)
        if path.exists():
            path.unlink()
        clear_current_service(service)

    def all_pids(self) -> list[int]:
        pids = [self.gateway.pid, self.router.pid]
        for pair in self.pairs:
            pids.extend([pair.worker.pid, pair.data_proxy.pid])
        return pids


def resolve_service_name(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    path = current_service_path()
    if path.exists():
        value = path.read_text().strip()
        if value:
            return value
    services = list_service_names()
    if len(services) == 1:
        return services[0]
    return DEFAULT_SERVICE


def list_service_names() -> list[str]:
    return sorted(path.stem for path in services_dir().glob("*.json"))


def clear_current_service(service: str) -> None:
    path = current_service_path()
    if path.exists() and path.read_text().strip() == service:
        path.unlink()
