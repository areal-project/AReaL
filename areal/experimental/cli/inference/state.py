# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

from areal.experimental.cli.process import pid_alive
from areal.experimental.cli.state import areal_home, atomic_write_json

DEFAULT_SERVICE = "default"


def inf_root() -> Path:
    d = areal_home() / "inf"
    d.mkdir(parents=True, exist_ok=True)
    return d


def services_dir() -> Path:
    d = inf_root() / "services"
    d.mkdir(parents=True, exist_ok=True)
    return d


def models_dir() -> Path:
    d = inf_root() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def logs_root() -> Path:
    d = inf_root() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def logs_dir(service: str = DEFAULT_SERVICE) -> Path:
    d = logs_root() / service
    d.mkdir(parents=True, exist_ok=True)
    return d


def service_state_path(service: str = DEFAULT_SERVICE) -> Path:
    return services_dir() / f"{service}.json"


def models_state_path(service: str = DEFAULT_SERVICE) -> Path:
    return models_dir() / f"{service}.json"


def models_lock_path(service: str = DEFAULT_SERVICE) -> Path:
    return models_dir() / f"{service}.lock"


def current_service_path() -> Path:
    return inf_root() / "current-service"


# Backward-compatible helper name. New code should use service_state_path().
def state_path(service: str = DEFAULT_SERVICE) -> Path:
    return service_state_path(service)


@dataclass
class ModelEntry:
    kind: str = "internal"
    backend: str = ""
    api_url: str = ""
    base_gpu_id: int = 0
    gpu_count: int = 0
    pids: list[int] = field(default_factory=list)
    engine_pids: list[int] = field(default_factory=list)
    proxy_pids: list[int] = field(default_factory=list)
    proxy_addrs: list[str] = field(default_factory=list)
    inference_server_addrs: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.pids and not self.engine_pids and not self.proxy_pids:
            self.engine_pids = self.pids[0::2]
            self.proxy_pids = self.pids[1::2]
        elif not self.pids and (self.engine_pids or self.proxy_pids):
            self.pids = [
                pid for pair in zip(self.engine_pids, self.proxy_pids) for pid in pair
            ]
            if len(self.engine_pids) > len(self.proxy_pids):
                self.pids.extend(self.engine_pids[len(self.proxy_pids) :])
            elif len(self.proxy_pids) > len(self.engine_pids):
                self.pids.extend(self.proxy_pids[len(self.engine_pids) :])

    def all_pids(self) -> list[int]:
        if self.pids:
            return self.pids
        return [*self.engine_pids, *self.proxy_pids]


@dataclass
class ServiceState:
    gateway_pid: int
    gateway_url: str
    router_pid: int
    router_url: str
    admin_api_key: str
    started_at: float
    service: str = DEFAULT_SERVICE
    launch_mode: str = "detached"

    def save(self) -> None:
        atomic_write_json(service_state_path(self.service), asdict(self))
        set_current_service(self.service)

    @classmethod
    def load(cls, service: str = DEFAULT_SERVICE) -> ServiceState:
        p = service_state_path(service)
        if not p.exists():
            raise FileNotFoundError(f"No service state at {p}")
        with open(p) as f:
            raw = json.load(f)
        raw.setdefault("service", service)
        raw.setdefault("launch_mode", "detached")
        return cls(**raw)

    @classmethod
    def remove(cls, service: str = DEFAULT_SERVICE) -> None:
        p = service_state_path(service)
        if p.exists():
            p.unlink()
        clear_current_service(service)

    def all_service_pids(self) -> list[int]:
        return [self.gateway_pid, self.router_pid]


@dataclass
class ModelState:
    service: str = DEFAULT_SERVICE
    next_gpu_id: int = 0
    default_model: str = ""
    models: dict[str, ModelEntry] = field(default_factory=dict)

    def save(self) -> None:
        if not self.default_model and self.models:
            self.default_model = next(iter(self.models))
        atomic_write_json(models_state_path(self.service), asdict(self))

    @classmethod
    def load(cls, service: str = DEFAULT_SERVICE) -> ModelState:
        p = models_state_path(service)
        if not p.exists():
            return cls(service=service)
        with open(p) as f:
            raw = json.load(f)
        models = {
            name: ModelEntry(**entry)
            for name, entry in (raw.pop("models", None) or {}).items()
        }
        raw.setdefault("service", service)
        raw.setdefault("next_gpu_id", 0)
        raw.setdefault("default_model", "")
        state = cls(models=models, **raw)
        if state.default_model and state.default_model not in state.models:
            state.default_model = next(iter(state.models), "")
        return state

    @classmethod
    def remove(cls, service: str = DEFAULT_SERVICE) -> None:
        p = models_state_path(service)
        if p.exists():
            p.unlink()

    def all_worker_pids(self) -> list[int]:
        return [pid for entry in self.models.values() for pid in entry.all_pids()]

    def all_engine_pids(self) -> list[int]:
        return [pid for entry in self.models.values() for pid in entry.engine_pids]

    def all_proxy_pids(self) -> list[int]:
        return [pid for entry in self.models.values() for pid in entry.proxy_pids]

    def set_default_if_empty(self, model: str) -> None:
        if not self.default_model:
            self.default_model = model

    def promote_default_after_remove(self, model: str) -> None:
        if self.default_model == model:
            self.default_model = next(
                (name for name in self.models if name != model), ""
            )


@dataclass
class RuntimeState:
    service_state: ServiceState
    model_state: ModelState

    @property
    def service(self) -> str:
        return self.service_state.service

    @property
    def gateway_pid(self) -> int:
        return self.service_state.gateway_pid

    @property
    def gateway_url(self) -> str:
        return self.service_state.gateway_url

    @property
    def router_pid(self) -> int:
        return self.service_state.router_pid

    @property
    def router_url(self) -> str:
        return self.service_state.router_url

    @property
    def admin_api_key(self) -> str:
        return self.service_state.admin_api_key

    @property
    def started_at(self) -> float:
        return self.service_state.started_at

    @property
    def models(self) -> dict[str, ModelEntry]:
        return self.model_state.models

    def save(self) -> None:
        self.service_state.save()
        self.model_state.save()

    def all_worker_pids(self) -> list[int]:
        return self.model_state.all_worker_pids()

    def all_pids(self) -> list[int]:
        return [*self.service_state.all_service_pids(), *self.all_worker_pids()]


# Compatibility name for older inf CLI code paths.
DaemonState = ServiceState


def set_current_service(service: str) -> None:
    current_service_path().write_text(service + "\n")


def clear_current_service(service: str) -> None:
    path = current_service_path()
    if path.exists() and path.read_text().strip() == service:
        path.unlink()


def resolve_service_name(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    current = current_service_path()
    if current.exists():
        value = current.read_text().strip()
        if value:
            return value
    running = [name for name in list_service_names() if service_running(name)]
    if len(running) == 1:
        return running[0]
    return DEFAULT_SERVICE


def list_service_names() -> list[str]:
    return sorted(path.stem for path in services_dir().glob("*.json"))


def service_running(service: str) -> bool:
    try:
        return gateway_alive(ServiceState.load(service))
    except Exception:
        return False


def gateway_alive(state: ServiceState | RuntimeState) -> bool:
    return pid_alive(state.gateway_pid)


def load_runtime_state(service: str = DEFAULT_SERVICE) -> RuntimeState:
    service_state = ServiceState.load(service)
    return RuntimeState(
        service_state=service_state,
        model_state=ModelState.load(service_state.service),
    )


@contextmanager
def locked_model_state(service: str = DEFAULT_SERVICE) -> Iterator[None]:
    import fcntl

    path = models_lock_path(service)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def recover_pids_from_raw_state(service: str = DEFAULT_SERVICE) -> list[int]:
    pids: list[int] = []

    def add(value) -> None:
        if isinstance(value, int) and value > 0:
            pids.append(value)
        elif isinstance(value, list):
            for item in value:
                add(item)

    def walk(value) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {
                    "pid",
                    "pids",
                    "gateway_pid",
                    "router_pid",
                    "engine_pids",
                    "proxy_pids",
                }:
                    add(item)
                else:
                    walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    for path in (service_state_path(service), models_state_path(service)):
        if not path.exists():
            continue
        with open(path) as f:
            walk(json.load(f))

    seen: set[int] = set()
    unique: list[int] = []
    for pid in pids:
        if pid not in seen:
            seen.add(pid)
            unique.append(pid)
    return unique
