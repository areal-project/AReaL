# SPDX-License-Identifier: Apache-2.0

"""Explicit source/target transition DAGs, without free missing operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..core import (
    Estimate,
    TimedTask,
    critical_path,
    evidence_of,
    fingerprint,
    nonnegative,
)


@dataclass(frozen=True)
class AllocationChange:
    task: str
    when: Literal["start", "end"]
    action: Literal["acquire", "release"]
    role: Literal["actor", "rollout"]
    devices: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "devices", tuple(self.devices))
        if (
            self.when not in {"start", "end"}
            or self.action not in {"acquire", "release"}
            or self.role not in {"actor", "rollout"}
        ):
            raise ValueError("Invalid allocation change")
        if (
            not self.devices
            or len(set(self.devices)) != len(self.devices)
            or any(not d for d in self.devices)
        ):
            raise ValueError("Device identities must be nonempty and unique")


@dataclass(frozen=True)
class TransferTask:
    name: str
    predecessors: tuple[str, ...] = ()
    measured: Estimate | None = None
    transferred_bytes: float | None = None
    bandwidth_bytes_s: float | None = None
    setup_seconds: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "predecessors", tuple(self.predecessors))
        if not self.name:
            raise ValueError("Transfer task name is required")
        for name in ("transferred_bytes", "bandwidth_bytes_s", "setup_seconds"):
            value = getattr(self, name)
            if value is not None:
                nonnegative(name, value, positive=name == "bandwidth_bytes_s")
        if self.measured is not None and any(
            x is not None
            for x in (
                self.transferred_bytes,
                self.bandwidth_bytes_s,
                self.setup_seconds,
            )
        ):
            raise ValueError("Choose explicit duration or a transfer model, not both")

    def estimate(self) -> Estimate:
        if self.measured is not None:
            return self.measured
        if any(
            x is None
            for x in (
                self.transferred_bytes,
                self.bandwidth_bytes_s,
                self.setup_seconds,
            )
        ):
            return Estimate(
                None,
                "unsupported",
                notes=(f"Missing transfer measurements/parameters: {self.name}",),
            )
        return Estimate(
            self.setup_seconds + self.transferred_bytes / self.bandwidth_bytes_s,
            "assumed",
            notes=("Explicit latency plus bytes/bandwidth prior",),
        )


@dataclass(frozen=True)
class TransitionProfile:
    action: str
    source: str | None
    target: str | None
    tasks: tuple[TransferTask, ...]
    allocation_changes: tuple[AllocationChange, ...] = ()
    blocks: tuple[str, ...] = ("actor",)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tasks", tuple(self.tasks))
        object.__setattr__(self, "allocation_changes", tuple(self.allocation_changes))
        object.__setattr__(self, "blocks", tuple(self.blocks))
        if not self.action or not self.tasks:
            raise ValueError("Even a zero-cost transition needs an explicit task")
        names = {t.name for t in self.tasks}
        if any(c.task not in names for c in self.allocation_changes):
            raise ValueError("Allocation event refers to an unknown transition task")
        if any(b not in {"actor", "generation", "admission"} for b in self.blocks):
            raise ValueError("Unknown transition blocking scope")


@dataclass(frozen=True)
class TransitionEstimate:
    duration: Estimate
    spans: tuple[tuple[str, float, float], ...] = ()
    allocation_events: tuple[tuple[float, AllocationChange], ...] = ()
    blocks: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExpertTransitionModel:
    profiles: tuple[TransitionProfile, ...] = ()
    version: str = "expert-transition-v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "profiles", tuple(self.profiles))
        keys = [(p.action, p.source, p.target) for p in self.profiles]
        if len(set(keys)) != len(keys) or not self.version:
            raise ValueError("Transition identities must be unique and versioned")

    @property
    def snapshot_id(self) -> str:
        return fingerprint(self.version, self.profiles)

    def estimate(
        self, action: str, source: str | None, target: str | None
    ) -> TransitionEstimate:
        matches = [
            p
            for p in self.profiles
            if (p.action, p.source, p.target) == (action, source, target)
        ]
        if not matches:
            return TransitionEstimate(
                Estimate(
                    None,
                    "unsupported",
                    notes=(
                        f"No transition profile for {action}: {source} -> {target}",
                    ),
                )
            )
        profile = matches[0]
        estimates = tuple(t.estimate() for t in profile.tasks)
        if any(e.seconds is None for e in estimates):
            return TransitionEstimate(
                Estimate(
                    None,
                    "unsupported",
                    notes=tuple(n for e in estimates for n in e.notes),
                )
            )
        if any(e.feasibility == "infeasible" for e in estimates):
            return TransitionEstimate(
                Estimate(
                    None,
                    "unsupported",
                    "infeasible",
                    ("Transition task is infeasible",),
                )
            )
        spans = critical_path(
            tuple(
                TimedTask(t.name, e, t.predecessors)
                for t, e in zip(profile.tasks, estimates, strict=True)
            )
        )
        times = {name: (start, end) for name, start, end in spans}
        changes = tuple(
            sorted(
                (
                    (times[c.task][c.when == "end"], c)
                    for c in profile.allocation_changes
                ),
                key=lambda row: (row[0], row[1].action != "release", row[1].task),
            )
        )
        return TransitionEstimate(
            Estimate(
                max(end for _, _, end in spans),
                evidence_of(tuple(e.evidence for e in estimates)),
                notes=tuple(n for e in estimates for n in e.notes),
                source_ids=tuple(
                    dict.fromkeys(s for e in estimates for s in e.source_ids)
                ),
            ),
            spans,
            changes,
            profile.blocks,
        )
