# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from areal.experimental.cli.process import pid_alive
from areal.experimental.cli.state import areal_home, atomic_write_json


def inf_root() -> Path:
    d = areal_home() / "inf"
    d.mkdir(parents=True, exist_ok=True)
    return d


def state_path() -> Path:
    return inf_root() / "state.json"


def logs_dir() -> Path:
    d = inf_root() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class ModelEntry:
    kind: str = "internal"
    backend: str = ""
    api_url: str = ""
    base_gpu_id: int = 0
    gpu_count: int = 0
    pids: list[int] = field(default_factory=list)
    proxy_addrs: list[str] = field(default_factory=list)
    inference_server_addrs: list[str] = field(default_factory=list)


@dataclass
class DaemonState:
    gateway_pid: int
    gateway_url: str
    router_pid: int
    router_url: str
    admin_api_key: str
    started_at: float
    next_gpu_id: int = 0
    models: dict[str, ModelEntry] = field(default_factory=dict)

    def save(self) -> None:
        atomic_write_json(state_path(), asdict(self))

    @classmethod
    def load(cls) -> DaemonState:
        p = state_path()
        if not p.exists():
            raise FileNotFoundError(f"No daemon state at {p}")
        with open(p) as f:
            raw = json.load(f)
        models = {
            name: ModelEntry(**entry)
            for name, entry in (raw.pop("models", None) or {}).items()
        }
        return cls(models=models, **raw)

    @classmethod
    def remove(cls) -> None:
        p = state_path()
        if p.exists():
            p.unlink()

    def all_worker_pids(self) -> list[int]:
        return [pid for entry in self.models.values() for pid in entry.pids]


def gateway_alive(state: DaemonState) -> bool:
    return pid_alive(state.gateway_pid)
