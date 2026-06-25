# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

from areal.experimental.cli.inference.scheduler import TaskHandle
from areal.experimental.cli.process import pid_alive
from areal.experimental.cli.state import (
    DEFAULT_SERVICE,
    ServiceStateBase,
    SupportsComponentProbe,
    atomic_write_json,
    clear_current_service,
    namespace_root,
    service_state_path,
    set_current_service,
)
from areal.experimental.cli.utils import file_lock

INF_NAMESPACE = "inf"

# ``DEFAULT_SERVICE`` is re-exported above so existing subcommand imports
# (`from ...inference.state import DEFAULT_SERVICE`) keep working.
__all__ = [
    "DEFAULT_SERVICE",
    "INF_NAMESPACE",
    "ModelEntry",
    "ModelReplica",
    "ModelState",
    "RuntimeState",
    "ServiceState",
    "locked_model_state",
    "models_dir",
    "models_lock_path",
    "models_state_path",
    "recover_pids_from_raw_state",
]


def models_dir() -> Path:
    """``$AREAL_HOME/inf/models`` — separate file per service holds the
    model registry, so we can lock and rewrite it independently of the
    service state file."""

    d = namespace_root(INF_NAMESPACE) / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def models_state_path(service: str) -> Path:
    return models_dir() / f"{service}.json"


def models_lock_path(service: str) -> Path:
    return models_dir() / f"{service}.lock"


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
        atomic_write_json(service_state_path(INF_NAMESPACE, self.service), asdict(self))
        set_current_service(INF_NAMESPACE, self.service)

    @classmethod
    def load(cls, service: str) -> ServiceState:
        p = service_state_path(INF_NAMESPACE, service)
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
    def remove(cls, service: str) -> None:
        p = service_state_path(INF_NAMESPACE, service)
        if p.exists():
            p.unlink()
        clear_current_service(INF_NAMESPACE, service)


@dataclass
class ModelState:
    service: str
    models: dict[str, ModelEntry] = field(default_factory=dict)

    def save(self) -> None:
        atomic_write_json(models_state_path(self.service), asdict(self))

    @classmethod
    def load(cls, service: str) -> ModelState:
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
    def remove(cls, service: str) -> None:
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
class RuntimeState(ServiceStateBase):
    """Composite of ServiceState (gateway/router) and ModelState (model
    registry). The single object the CLI verbs operate on; the lifecycle
    contract is implemented against ``RuntimeState`` so scaffold helpers
    enumerate every component in one pass."""

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
    def launch_mode(self) -> str:
        return self.service_state.launch_mode

    @property
    def models(self) -> dict[str, ModelEntry]:
        return self.model_state.models

    def save(self) -> None:
        self.service_state.save()
        self.model_state.save()

    @classmethod
    def load(cls, service: str) -> RuntimeState:
        service_state = ServiceState.load(service)
        return cls(
            service_state=service_state,
            model_state=ModelState.load(service_state.service),
        )

    # --- ServiceStateBase contract ----------------------------------------

    def gateway_alive(self) -> bool:
        # Non-local backends will need their own liveness probe — PID-based
        # check returns False for them (no pid in handle.ref).
        return pid_alive(self.gateway_pid)

    def components(self) -> Iterable[tuple[str, SupportsComponentProbe]]:
        yield "gateway", self.gateway_handle
        yield "router", self.router_handle
        for name, entry in self.model_state.models.items():
            for i, replica in enumerate(entry.replicas):
                yield f"data_proxy[{name}/{i}]", replica.data_proxy
                yield f"worker[{name}/{i}]", replica.worker


@contextmanager
def locked_model_state(service: str) -> Iterator[None]:
    """Per-service flock on the model-registry file. Held across
    register / deregister so concurrent CLI calls cannot race the
    JSON read-modify-write."""

    with file_lock(models_lock_path(service)):
        yield


def recover_pids_from_raw_state(service: str) -> list[int]:
    """Best-effort PID extraction from possibly-malformed state files —
    walks BOTH service-state and model-state JSON for ``pid`` / ``pids``
    keys so ``run --force`` cleans up children when the dataclass parse
    fails. Inference owns its own walker because it has the second
    (model-state) file scaffold's default helper doesn't know about."""

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

    for path in (
        service_state_path(INF_NAMESPACE, service),
        models_state_path(service),
    ):
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
