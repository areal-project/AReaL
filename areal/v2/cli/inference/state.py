# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

from areal.v2.cli.inference.scheduler import TaskHandle
from areal.v2.cli.process import pid_alive
from areal.v2.cli.state import (
    DEFAULT_SERVICE,
    NamespacedStateStore,
    ServiceStateBase,
    SupportsComponentProbe,
    atomic_write_json,
)
from areal.v2.cli.utils import file_lock

INF_NAMESPACE = "inf"


class InferenceStateStore(NamespacedStateStore):
    """Inference-specific extension of the scaffold ``NamespacedStateStore``.

    The inference service splits its state across two files per service —
    the service-state (gateway / router handles, inherited from
    scaffold) and the model-state (model registry, locked for
    register / deregister). This class adds the model-state file paths
    + lock + overrides ``recover_pids_from_raw_state`` to walk both
    files when ``run --force`` falls back to raw-JSON PID recovery.
    """

    def __init__(self, namespace: str = INF_NAMESPACE) -> None:
        super().__init__(namespace)

    # --- model-state file layout ----------------------------------------

    def models_dir(self) -> Path:
        """``$AREAL_HOME/<namespace>/models`` — created lazily."""

        d = self.root() / "models"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def models_state_path(self, service: str) -> Path:
        return self.models_dir() / f"{service}.json"

    def models_lock_path(self, service: str) -> Path:
        return self.models_dir() / f"{service}.lock"

    # --- locking --------------------------------------------------------

    @contextmanager
    def lock_model_state(self, service: str) -> Iterator[None]:
        """Per-service flock on the model-registry file. Held across
        register / deregister so concurrent CLI calls cannot race the
        JSON read-modify-write."""

        with file_lock(self.models_lock_path(service)):
            yield

    # --- best-effort recovery (override) --------------------------------

    def recover_pids_from_raw_state(self, service: str) -> list[int]:
        """Walk BOTH service-state and model-state JSON for ``pid`` /
        ``pids`` keys. Inference overrides scaffold's single-file walker
        because the second (model-state) file is invisible to it."""

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

        for path in (self.service_state_path(service), self.models_state_path(service)):
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


# Module-level singleton — every dataclass / verb routes path resolution
# through this. Tests use ``AREAL_HOME`` to isolate; the store re-resolves
# on every call so the env override takes effect without rebuilding.
store = InferenceStateStore()


# ``DEFAULT_SERVICE`` is re-exported above so existing subcommand imports
# (`from ...inference.state import DEFAULT_SERVICE`) keep working.
__all__ = [
    "DEFAULT_SERVICE",
    "INF_NAMESPACE",
    "InferenceStateStore",
    "ModelEntry",
    "ModelReplica",
    "ModelState",
    "RuntimeState",
    "ServiceState",
    "store",
]


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
        atomic_write_json(store.service_state_path(self.service), asdict(self))
        store.set_current_service(self.service)

    @classmethod
    def load(cls, service: str) -> ServiceState:
        p = store.service_state_path(service)
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
        p = store.service_state_path(service)
        if p.exists():
            p.unlink()
        store.clear_current_service(service)


@dataclass
class ModelState:
    service: str
    models: dict[str, ModelEntry] = field(default_factory=dict)

    def save(self) -> None:
        atomic_write_json(store.models_state_path(self.service), asdict(self))

    @classmethod
    def load(cls, service: str) -> ModelState:
        p = store.models_state_path(service)
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
        p = store.models_state_path(service)
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
