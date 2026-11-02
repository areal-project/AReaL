# SPDX-License-Identifier: Apache-2.0

"""TP-local GPU leases and explicit startup, state and link timing inputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

from areal.elastic.contracts import OnlineLayout, nonnegative, positive_int
from areal.elastic.layout import ExecutionLayout


class ActorMemoryRejected(ValueError):
    """The declared Actor recipe cannot safely train this completed batch."""


@dataclass(frozen=True)
class NativePlacement:
    layout: OnlineLayout
    execution: ExecutionLayout


@dataclass(frozen=True)
class HandoffConfig:
    node_gpus: tuple[int, ...]
    actor_startup_seconds: float
    rollout_startup_seconds: float
    actor_warmup_seconds: float
    rollout_warmup_seconds: float
    actor_state_bytes: int
    weight_bytes: int
    effective_bandwidth_GBps: float
    actor_sync_setup_seconds: float
    rollout_sync_setup_seconds: float
    provenance: str
    mode: str = "hot"
    native_source: ExecutionLayout | None = None
    native_devices: tuple[int, ...] = ()
    native_placements: tuple[NativePlacement, ...] = ()
    phase_seconds: tuple[tuple[str, float], ...] = ()
    native_checkpoint_version: int | None = None

    @classmethod
    def from_dict(cls, raw):
        values = dict(raw)
        values["node_gpus"] = tuple(values["node_gpus"])
        values["native_devices"] = tuple(values.get("native_devices", ()))
        values["phase_seconds"] = tuple(
            tuple(x) for x in values.get("phase_seconds", ())
        )
        if values.get("native_source") is not None:
            values["native_source"] = ExecutionLayout.decode(values["native_source"])
        values["native_placements"] = tuple(
            NativePlacement(
                OnlineLayout(**p["layout"]), ExecutionLayout.decode(p["execution"])
            )
            for p in values.get("native_placements", ())
        )
        return cls(**values)

    def __post_init__(self):
        if self.mode not in {"hot", "cold"}:
            raise ValueError("Handoff mode must be hot or cold")
        if len(dict(self.phase_seconds)) != len(self.phase_seconds):
            raise ValueError("Native transition timing names must be unique")
        for name, seconds in self.phase_seconds:
            nonnegative(name, seconds)
        if self.native_source is not None:
            if len(set(self.native_devices)) != len(self.native_devices) or sum(
                self.node_gpus
            ) != len(self.native_devices):
                raise ValueError(
                    "Native topology must contain exactly the immutable reservation"
                )
            for placement in self.native_placements:
                target = placement.execution
                if (
                    placement.layout.actor_tp != 1
                    or placement.layout.rollout_tp != 1
                    or target.world_size != placement.layout.actor_dp
                    or len(target.rollout_devices) != placement.layout.rollout_replicas
                ):
                    raise ValueError(
                        "Native prediction requires matching FSDP DP / SGLang TP1 layouts"
                    )
                if not set(target.actor_devices + target.rollout_devices) <= set(
                    self.native_devices
                ):
                    raise ValueError(
                        "Native candidate exceeds the original reservation"
                    )
        if self.native_checkpoint_version is not None and (
            type(self.native_checkpoint_version) is not int
            or self.native_checkpoint_version < 0
        ):
            raise ValueError("Native checkpoint version must be nonnegative or unknown")
        if not self.node_gpus or not self.provenance:
            raise ValueError(
                "Explicit pool topology and timing provenance are required"
            )
        for count in self.node_gpus:
            positive_int("node_gpus", count)
            if count > 8:
                raise ValueError("A modeled node contains at most eight GPUs")
        for name in (
            "actor_startup_seconds",
            "rollout_startup_seconds",
            "actor_warmup_seconds",
            "rollout_warmup_seconds",
            "actor_sync_setup_seconds",
            "rollout_sync_setup_seconds",
        ):
            nonnegative(name, getattr(self, name))
        for name in ("actor_state_bytes", "weight_bytes"):
            positive_int(name, getattr(self, name))
        nonnegative(
            "effective_bandwidth_GBps", self.effective_bandwidth_GBps, positive=True
        )


@dataclass
class GPUAllocator:
    nodes: tuple[int, ...]
    leases: dict[str, tuple[str, tuple[tuple[int, int], ...]]] = field(
        default_factory=dict
    )
    gpu_seconds: dict[str, float] = field(
        default_factory=lambda: {"actor": 0.0, "rollout": 0.0}
    )
    now: float = 0.0
    peak_gpus: int = 0
    lease_gpu_seconds: dict[str, float] = field(default_factory=dict)

    def acquire(
        self, owner: str, role: str, count: int, *, tp: int = 1
    ) -> tuple[tuple[int, int], ...] | None:
        if owner in self.leases or role not in self.gpu_seconds:
            raise ValueError("A deployment needs a unique owner and known role")
        if tp not in (1, 2, 4, 8) or count % tp:
            raise ValueError("GPU count must contain complete node-local TP groups")
        occupied = {gpu for _, devices in self.leases.values() for gpu in devices}
        available = []
        for node, capacity in enumerate(self.nodes):
            free = tuple(
                (node, rank) for rank in range(capacity) if (node, rank) not in occupied
            )
            available.append(free)
            if len(free) >= count:
                return self._lease(owner, role, free[:count])
        # Preserve the existing node-local contract for roles of <=8 GPUs.
        # Larger roles span two nodes, with every TP group kept inside a node.
        if count > 8:
            for left, right in combinations(available, 2):
                left_count = min(count, len(left) // tp * tp)
                right_count = count - left_count
                if right_count <= len(right) // tp * tp:
                    return self._lease(
                        owner, role, left[:left_count] + right[:right_count]
                    )
        return None

    def acquire_exact(self, owner, role, devices):
        devices = tuple(devices)
        occupied = {d for _, lease in self.leases.values() for d in lease}
        if (
            owner in self.leases
            or role not in self.gpu_seconds
            or len(set(devices)) != len(devices)
            or any(
                d in occupied
                or node < 0
                or node >= len(self.nodes)
                or not 0 <= rank < self.nodes[node]
                for d in devices
                for node, rank in (d,)
            )
        ):
            raise ValueError(
                "Native target leases must fit the immutable reservation without overlap"
            )
        return self._lease(owner, role, devices)

    def _lease(self, owner, role, devices):
        self.leases[owner] = role, devices
        self.peak_gpus = max(
            self.peak_gpus, sum(len(gpus) for _, gpus in self.leases.values())
        )
        return devices

    def release(self, owner: str):
        del self.leases[owner]

    def count(self, role: str) -> int:
        return sum(
            len(devices) for name, devices in self.leases.values() if name == role
        )

    def advance(self, when: float):
        if when < self.now - 1e-12:
            raise ValueError("Resource accounting cannot move backwards")
        for owner, (role, devices) in self.leases.items():
            cost = (when - self.now) * len(devices)
            self.gpu_seconds[role] += cost
            self.lease_gpu_seconds[owner] = (
                self.lease_gpu_seconds.get(owner, 0.0) + cost
            )
        self.now = when


@dataclass
class SerializedLink:
    """Conservative shared-link prior: startup/state/publication transfers serialize.

    Reserving a fitted publication's complete wall span also serializes its host
    preparation. This is explicit headroom, not a measured network contention fit.
    """

    bandwidth_GBps: float
    available_at: float = 0.0

    def reserve(
        self,
        now: float,
        *,
        byte_count: int = 0,
        setup_seconds: float = 0.0,
        fitted_seconds: float | None = None,
    ) -> tuple[float, float]:
        seconds = (
            byte_count / (self.bandwidth_GBps * 1e9) + setup_seconds
            if fitted_seconds is None
            else fitted_seconds
        )
        nonnegative("transfer_seconds", seconds)
        start = max(now, self.available_at)
        self.available_at = start + seconds
        return start, self.available_at
