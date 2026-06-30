# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field

from areal.v2.cli.process import pid_alive
from areal.v2.cli.state import (
    NamespacedStateStore,
    ServiceStateBase,
    SupportsComponentProbe,
    atomic_write_json,
)

AGENT_NAMESPACE = "agent"

# Module-level store — every save/load/remove routes paths through it,
# tests with ``AREAL_HOME`` work because the store re-resolves on each
# call.
store = NamespacedStateStore(AGENT_NAMESPACE)


@dataclass
class ProcessState:
    component: str
    pid: int
    url: str
    log_file: str

    @property
    def addr(self) -> str:
        return self.url


@dataclass
class PairState:
    index: int
    worker: ProcessState
    data_proxy: ProcessState


@dataclass
class ServiceState(ServiceStateBase):
    service: str
    launch_mode: str
    agent: str
    admin_api_key: str
    gateway: ProcessState
    router: ProcessState
    pairs: list[PairState]
    session_timeout: float = 1800.0
    health_poll_interval: float = 5.0
    drain_timeout: float = 30.0
    started_at: float = field(default_factory=time.time)

    def save(self) -> None:
        atomic_write_json(store.service_state_path(self.service), asdict(self))
        store.set_current_service(self.service)

    @classmethod
    def load(cls, service: str) -> ServiceState:
        with open(store.service_state_path(service)) as f:
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
        # ``created_at`` is the legacy field name; accept it for forward
        # compatibility with any state files written before the rename.
        if "created_at" in raw and "started_at" not in raw:
            raw["started_at"] = raw.pop("created_at")
        raw.pop("created_at", None)
        for legacy in ("inf_addr", "inf_api_key", "inf_model"):
            raw.pop(legacy, None)
        return cls(**raw)

    @classmethod
    def remove(cls, service: str) -> None:
        path = store.service_state_path(service)
        if path.exists():
            path.unlink()
        store.clear_current_service(service)

    def gateway_alive(self) -> bool:
        # The CLI treats "running" as "gateway process still alive". The
        # gateway is the entry point — without it, the rest of the stack
        # is unreachable even if some children survived.
        return pid_alive(self.gateway.pid)

    def components(self) -> Iterable[tuple[str, SupportsComponentProbe]]:
        yield "gateway", self.gateway
        yield "router", self.router
        for pair in self.pairs:
            yield pair.worker.component, pair.worker
            yield pair.data_proxy.component, pair.data_proxy
