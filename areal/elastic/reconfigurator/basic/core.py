# SPDX-License-Identifier: Apache-2.0

"""Immutable estimates and calibration records, independent of Basic revisions."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, is_dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

Evidence = Literal["measured", "calibrated", "assumed", "unsupported"]
Feasibility = Literal["feasible", "infeasible", "unknown"]
Features = tuple[tuple[str, float], ...]


def nonnegative(name: str, value: float, *, positive: bool = False) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    if positive and value == 0:
        raise ValueError(f"{name} must be positive")


def integer(name: str, value: int, *, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def fingerprint(*values: object) -> str:
    def encode(value):
        if is_dataclass(value):
            return asdict(value)
        raise TypeError(f"Cannot fingerprint {type(value).__name__}")

    payload = json.dumps(values, default=encode, sort_keys=True, allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


@lru_cache(maxsize=1)
def code_revision() -> str:
    """Hash this reference implementation once per interpreter/model session."""
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    # WorkloadStats validation/length reconstruction also affects predictions.
    digest.update((root.parents[1] / "contracts.py").read_bytes())
    return digest.hexdigest()


def evidence_of(values: tuple[Evidence, ...]) -> Evidence:
    order = ("measured", "calibrated", "assumed", "unsupported")
    return max(values, key=order.index, default="measured")


@dataclass(frozen=True)
class Estimate:
    seconds: float | None
    evidence: Evidence
    feasibility: Feasibility = "unknown"
    notes: tuple[str, ...] = ()
    source_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.evidence not in {"measured", "calibrated", "assumed", "unsupported"}:
            raise ValueError("Unknown estimate evidence")
        if self.feasibility not in {"feasible", "infeasible", "unknown"}:
            raise ValueError("Unknown feasibility")
        if (self.seconds is None) != (self.evidence == "unsupported"):
            raise ValueError("Unsupported estimates must have unknown duration")
        if self.seconds is not None:
            nonnegative("seconds", self.seconds)
        object.__setattr__(self, "notes", tuple(self.notes))
        object.__setattr__(self, "source_ids", tuple(self.source_ids))

    def require_seconds(self) -> float:
        if self.seconds is None or self.feasibility == "infeasible":
            raise UnsupportedEstimate("; ".join(self.notes) or "Unavailable estimate")
        return self.seconds


class UnsupportedEstimate(ValueError):
    """A missing model or known infeasibility, never an implicit zero duration."""


@dataclass(frozen=True)
class TimedTask:
    name: str
    estimate: Estimate
    predecessors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Task name is required")
        object.__setattr__(self, "predecessors", tuple(self.predecessors))


def critical_path(tasks: tuple[TimedTask, ...]) -> tuple[tuple[str, float, float], ...]:
    """Schedule a dependency DAG; resource serialization must be explicit edges."""
    pending = {task.name: task for task in tasks}
    if len(pending) != len(tasks):
        raise ValueError("Duplicate task name")
    if any(parent not in pending for t in tasks for parent in t.predecessors):
        raise ValueError("Unknown task dependency")
    end: dict[str, float] = {}
    spans = []
    while pending:
        ready = [
            task
            for task in pending.values()
            if all(p in end for p in task.predecessors)
        ]
        if not ready:
            raise ValueError("Cyclic task dependencies")
        for task in ready:
            start = max((end[p] for p in task.predecessors), default=0.0)
            end[task.name] = start + task.estimate.require_seconds()
            spans.append((task.name, start, end[task.name]))
            del pending[task.name]
    return tuple(spans)


@dataclass(frozen=True)
class CalibrationPoint:
    sample_id: str
    profile: str
    phase: str
    predicted_seconds: float
    measured_seconds: float
    features: Features
    timing_boundary: str = ""

    def __post_init__(self) -> None:
        if not all((self.sample_id, self.profile, self.phase)):
            raise ValueError(
                "Calibration requires sample, profile and phase identities"
            )
        nonnegative("predicted_seconds", self.predicted_seconds, positive=True)
        nonnegative("measured_seconds", self.measured_seconds, positive=True)
        object.__setattr__(self, "features", tuple(tuple(x) for x in self.features))
        if not self.features or len(dict(self.features)) != len(self.features):
            raise ValueError("Calibration features must have unique names")
        for name, value in self.features:
            nonnegative(name, value)


@dataclass(frozen=True)
class PhaseCalibration:
    version: str
    profile: str
    phase: str
    scale: float
    bounds: tuple[tuple[str, float, float], ...]
    sample_ids: tuple[str, ...]
    timing_boundary: str = ""

    def __post_init__(self) -> None:
        nonnegative("scale", self.scale, positive=True)
        if not all(
            (self.version, self.profile, self.phase, self.sample_ids, self.bounds)
        ):
            raise ValueError("Calibration needs version, scope, bounds and provenance")
        object.__setattr__(self, "bounds", tuple(tuple(b) for b in self.bounds))
        object.__setattr__(self, "sample_ids", tuple(self.sample_ids))
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("Duplicate calibration sample IDs")
        if len({b[0] for b in self.bounds}) != len(self.bounds):
            raise ValueError("Duplicate calibration feature bounds")
        for name, low, high in self.bounds:
            nonnegative(name, low)
            nonnegative(name, high)
            if high < low:
                raise ValueError("Reversed calibration bounds")

    @classmethod
    def fit(
        cls, version: str, points: tuple[CalibrationPoint, ...]
    ) -> PhaseCalibration:
        """Fit one positive scale; no evaluator labels, acceptance or search here."""
        if not points:
            raise ValueError("No calibration points")
        first = points[0]
        names = sorted(dict(first.features))
        if any(
            (p.profile, p.phase, p.timing_boundary)
            != (first.profile, first.phase, first.timing_boundary)
            or sorted(dict(p.features)) != names
            for p in points
        ):
            raise ValueError(
                "Calibration points must share full profile, phase, timing boundary and features"
            )
        scale = sum(p.predicted_seconds * p.measured_seconds for p in points) / sum(
            p.predicted_seconds**2 for p in points
        )
        return cls(
            version,
            first.profile,
            first.phase,
            scale,
            tuple(
                (
                    name,
                    min(dict(p.features)[name] for p in points),
                    max(dict(p.features)[name] for p in points),
                )
                for name in names
            ),
            tuple(p.sample_id for p in points),
            first.timing_boundary,
        )

    def apply(
        self,
        seconds: float,
        profile: str,
        phase: str,
        features: Features,
        *,
        timing_boundary: str = "",
    ) -> Estimate:
        nonnegative("seconds", seconds)
        values = dict(features)
        violations = []
        if (profile, phase) != (self.profile, self.phase):
            violations.append("Calibration profile or phase mismatch")
        if timing_boundary != self.timing_boundary:
            violations.append("Calibration timing boundary mismatch")
        for name, low, high in self.bounds:
            if name not in values:
                violations.append(f"Missing calibration feature: {name}")
            elif not low <= values[name] <= high:
                violations.append(
                    f"Outside calibration bound: {name}={values[name]} not in [{low}, {high}]"
                )
        if violations:
            return Estimate(
                None,
                "unsupported",
                notes=tuple(violations),
                source_ids=self.sample_ids,
            )
        return Estimate(seconds * self.scale, "calibrated", source_ids=self.sample_ids)


def apply_calibration(
    seconds: float,
    profile: str,
    phase: str,
    features: Features,
    calibration: tuple[PhaseCalibration, ...],
    *,
    timing_boundary: str = "",
) -> Estimate:
    matching = [c for c in calibration if (c.profile, c.phase) == (profile, phase)]
    if len(matching) > 1:
        raise ValueError("Ambiguous calibration for profile and phase")
    if not matching:
        return Estimate(
            seconds, "assumed", notes=("Analytical prior; no matching calibration",)
        )
    return matching[0].apply(
        seconds, profile, phase, features, timing_boundary=timing_boundary
    )
