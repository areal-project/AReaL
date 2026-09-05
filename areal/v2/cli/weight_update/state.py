# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field

from areal.v2.cli.client import ServiceHTTPError, ServiceUnreachable, request_json
from areal.v2.cli.state import (
    NamespacedStateStore,
    ServiceStateBase,
    SupportsComponentProbe,
    atomic_write_json,
)

WU_NAMESPACE = "weight-update"

store = NamespacedStateStore(WU_NAMESPACE)


@dataclass
class GatewayHandle:
    url: str
    pid: int = 0

    @property
    def addr(self) -> str:
        return self.url


@dataclass
class ServiceState(ServiceStateBase):
    """What the operator needs to reach a weight-update gateway.

    Written by ``WeightUpdateController`` when it brings the gateway up, not by
    this CLI -- in the v2 flow the gateway belongs to the training side. The
    controller picks an ephemeral port when ``config.port`` is 0, so without
    this file the address is only ever known inside that one process.
    """

    service: str
    launch_mode: str
    admin_api_key: str
    gateway: GatewayHandle
    started_at: float = field(default_factory=time.time)

    def save(self) -> None:
        atomic_write_json(store.service_state_path(self.service), asdict(self))
        store.set_current_service(self.service)

    @classmethod
    def load(cls, service: str) -> ServiceState:
        with open(store.service_state_path(service)) as f:
            raw = json.load(f)
        raw["gateway"] = GatewayHandle(**raw["gateway"])
        return cls(**raw)

    @classmethod
    def remove(cls, service: str) -> None:
        path = store.service_state_path(service)
        if path.exists():
            path.unlink()
        store.clear_current_service(service)

    def gateway_alive(self) -> bool:
        """Probe ``/health`` rather than the recorded PID.

        The CLI does not own this process. A PID written by a controller on
        another host means nothing here, and a PID on this host may have been
        reused since. Reachability is the only claim the operator can act on.
        """
        try:
            request_json(f"{self.gateway.url}/health", timeout=2.0)
            return True
        except (ServiceUnreachable, ServiceHTTPError):
            return False

    def components(self) -> Iterable[tuple[str, SupportsComponentProbe]]:
        yield "gateway", self.gateway
