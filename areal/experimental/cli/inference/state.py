# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

from areal.experimental.cli.inference.scheduler import TaskHandle
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


def _handle_from_dict(raw: dict) -> TaskHandle:
    return TaskHandle(
        host=raw["host"],
        ports=list(raw.get("ports", [])),
        gpu_devices=list(raw.get("gpu_devices", [])),
        ref=dict(raw.get("ref", {})),
    )


@dataclass
class ModelReplica:
    """One DP replica: data-proxy in front, worker behind. Field order
    matches the request-flow direction used by stop / status output."""

    data_proxy: TaskHandle
    worker: TaskHandle


@dataclass
class ModelEntry:
    backend: str = ""
    replicas: list[ModelReplica] = field(default_factory=list)

    def all_workers(self) -> list[TaskHandle]:
        return [r.worker for r in self.replicas]

    def all_data_proxies(self) -> list[TaskHandle]:
        return [r.data_proxy for r in self.replicas]

    def gpu_devices(self) -> list[int]:
        return [g for w in self.all_workers() for g in w.gpu_devices]

    @classmethod
    def from_dict(cls, raw: dict) -> ModelEntry:
        replicas = [
            ModelReplica(
                data_proxy=_handle_from_dict(r["data_proxy"]),
                worker=_handle_from_dict(r["worker"]),
            )
            for r in raw.get("replicas", [])
        ]
        return cls(backend=raw.get("backend", ""), replicas=replicas)


@dataclass
class ServiceState:
    service: str
    backend: str
    gateway_handle: TaskHandle
    router_handle: TaskHandle
    admin_api_key: str
    started_at: float
    launch_mode: str = "detached"

    @property
    def gateway_pid(self) -> int:
        return self.gateway_handle.pid

    @property
    def gateway_url(self) -> str:
        return self.gateway_handle.addr

    @property
    def router_pid(self) -> int:
        return self.router_handle.pid

    @property
    def router_url(self) -> str:
        return self.router_handle.addr

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
        return cls(
            service=raw.get("service", service),
            backend=raw["backend"],
            gateway_handle=_handle_from_dict(raw["gateway_handle"]),
            router_handle=_handle_from_dict(raw["router_handle"]),
            admin_api_key=raw["admin_api_key"],
            started_at=float(raw["started_at"]),
            launch_mode=raw.get("launch_mode", "detached"),
        )

    @classmethod
    def remove(cls, service: str = DEFAULT_SERVICE) -> None:
        p = service_state_path(service)
        if p.exists():
            p.unlink()
        clear_current_service(service)


@dataclass
class ModelState:
    service: str = DEFAULT_SERVICE
    models: dict[str, ModelEntry] = field(default_factory=dict)

    def save(self) -> None:
        atomic_write_json(models_state_path(self.service), asdict(self))

    @classmethod
    def load(cls, service: str = DEFAULT_SERVICE) -> ModelState:
        p = models_state_path(service)
        if not p.exists():
            return cls(service=service)
        with open(p) as f:
            raw = json.load(f)
        models = {
            name: ModelEntry.from_dict(entry)
            for name, entry in (raw.pop("models", None) or {}).items()
        }
        return cls(service=raw.get("service", service), models=models)

    @classmethod
    def remove(cls, service: str = DEFAULT_SERVICE) -> None:
        p = models_state_path(service)
        if p.exists():
            p.unlink()

    def all_workers(self) -> list[TaskHandle]:
        return [h for entry in self.models.values() for h in entry.all_workers()]

    def all_data_proxies(self) -> list[TaskHandle]:
        return [h for entry in self.models.values() for h in entry.all_data_proxies()]

    def occupied_gpus(self) -> set[int]:
        return {g for entry in self.models.values() for g in entry.gpu_devices()}


@dataclass
class RuntimeState:
    service_state: ServiceState
    model_state: ModelState

    @property
    def service(self) -> str:
        return self.service_state.service

    @property
    def backend(self) -> str:
        return self.service_state.backend

    @property
    def gateway_handle(self) -> TaskHandle:
        return self.service_state.gateway_handle

    @property
    def router_handle(self) -> TaskHandle:
        return self.service_state.router_handle

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
    # Non-local backends will need their own liveness probe — PID-based
    # check returns False for them (no pid in handle.ref).
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
    """Best-effort PID extraction from a possibly-malformed state file —
    walks the JSON tree for ``pid``/``pids`` keys so ``run --force`` can
    still clean up children when the dataclass parse fails."""

    pids: list[int] = []
    pid_keys = {"pid", "pids"}

    def add(value) -> None:
        if isinstance(value, int) and value > 0:
            pids.append(value)
        elif isinstance(value, list):
            for item in value:
                add(item)

    def walk(value) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in pid_keys:
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
