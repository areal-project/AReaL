# SPDX-License-Identifier: Apache-2.0

"""Fit whole-phase wall time from summary logs, without inventing components."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from ....contracts import WorkloadStats
from ..core import Estimate, Features, PhaseCalibration, nonnegative
from .advantage import ADVANTAGE_BOUNDARY, AdvantageEstimate


def _features(workload: WorkloadStats) -> Features:
    values = [
        ("advantage_sequence_tokens", workload.sequence_tokens),
        ("advantage_n_seqs", float(workload.n_seqs)),
        ("advantage_mean_seq_len", workload.sequence_tokens / workload.n_seqs),
    ]
    maximum = workload.max_seq_len
    if maximum is None and workload.sequence_lengths is not None:
        maximum = max(workload.sequence_lengths)
    if maximum is None and workload.per_dp_sequence_lengths is not None:
        maximum = max(max(row) for row in workload.per_dp_sequence_lengths)
    if maximum is not None:
        values.append(("advantage_max_seq_len", float(maximum)))
    return tuple(values)


@dataclass(frozen=True)
class AdvantageObservation:
    """One calibration-only phase label with an explicitly audited timing scope.

    sample_id identifies the source run/step. Workload uses global attention
    tokens once, excluding TP/CP/PP copies and repeated PPO uses.
    """

    sample_id: str
    profile: str
    workload: WorkloadStats
    measured_seconds: float
    timing_boundary: str

    def __post_init__(self) -> None:
        if not self.sample_id or not self.profile or not self.timing_boundary:
            raise ValueError(
                "Advantage labels require source, profile and timing boundary"
            )
        nonnegative("measured_seconds", self.measured_seconds, positive=True)


@dataclass(frozen=True)
class AggregateAdvantageModel:
    """Explicit alternative to the component prior, scoped to one fitted profile.

    execution_id identifies controller/device/GAE/normalization options and label
    aggregation. A run-scoped identity is appropriate when these cannot yet be
    separated. Parameters describe the whole phase, never identified components.
    calibration has scale=1 and only enforces scope/bounds and records sources;
    do not fit another phase scale on top of this already calibrated predictor.
    """

    execution_id: str
    form: str = "affine_tokens"
    timing_boundary: str = ADVANTAGE_BOUNDARY
    calibration: PhaseCalibration | None = None
    intercept_seconds: float | None = None
    seconds_per_token: float | None = None

    def __post_init__(self) -> None:
        if not self.execution_id or self.form not in {"affine_tokens", "constant"}:
            raise ValueError(
                "Aggregate advantage requires execution identity and known form"
            )
        if self.timing_boundary != ADVANTAGE_BOUNDARY:
            raise ValueError(
                "Aggregate advantage requires trainer host wall time with stats"
            )
        fitted = self.calibration is not None
        if any(
            (value is not None) != fitted
            for value in (self.intercept_seconds, self.seconds_per_token)
        ):
            raise ValueError(
                "Aggregate coefficients and calibration must be supplied together"
            )
        if not fitted:
            return
        nonnegative("intercept_seconds", self.intercept_seconds)
        nonnegative("seconds_per_token", self.seconds_per_token)
        if self.intercept_seconds == self.seconds_per_token == 0:
            raise ValueError("A positive phase label cannot fit a zero predictor")
        if self.form == "constant" and self.seconds_per_token != 0:
            raise ValueError("Constant model cannot have a token slope")
        scope = self.calibration
        if (
            scope.phase != "compute_advantage"
            or scope.scale != 1
            or scope.timing_boundary != self.timing_boundary
        ):
            raise ValueError(
                "Aggregate calibration must only guard the matching phase boundary"
            )
        required = {
            "advantage_sequence_tokens",
            "advantage_n_seqs",
            "advantage_mean_seq_len",
        }
        if not required <= {name for name, _, _ in scope.bounds}:
            raise ValueError(
                "Aggregate calibration is missing required workload bounds"
            )

    @property
    def scope_spec(self) -> tuple[str, ...]:
        """Stable before/after fitting; coefficients remain in simulator snapshots."""
        return (
            "aggregate-advantage-v1",
            self.execution_id,
            self.form,
            self.timing_boundary,
        )

    def fit(
        self, version: str, observations: tuple[AdvantageObservation, ...]
    ) -> AggregateAdvantageModel:
        """Nonnegative least squares for one preselected form, using calibration only."""
        if len(observations) < 2:
            raise ValueError("At least two calibration observations are required")
        first = observations[0]
        features = [_features(point.workload) for point in observations]
        names = tuple(name for name, _ in features[0])
        if any(
            point.profile != first.profile
            or point.timing_boundary != self.timing_boundary
            for point in observations
        ):
            raise ValueError(
                "Aggregate labels must share the full profile and trainer wall boundary"
            )
        if any(tuple(name for name, _ in row) != names for row in features):
            raise ValueError("Aggregate labels must share available summary features")
        # Reuse the established scope/bounds checker, without a second fitted scale.
        guard = PhaseCalibration(
            version,
            first.profile,
            "compute_advantage",
            1.0,
            tuple(
                (
                    name,
                    min(dict(row)[name] for row in features),
                    max(dict(row)[name] for row in features),
                )
                for name in names
            ),
            tuple(point.sample_id for point in observations),
            self.timing_boundary,
        )
        x = [point.workload.sequence_tokens for point in observations]
        y = [point.measured_seconds for point in observations]
        # Scale both axes to keep squared errors/normal equations well conditioned.
        xscale, yscale = max(x), max(y)
        x, y = [v / xscale for v in x], [v / yscale for v in y]
        mean_x, mean_y = math.fsum(x) / len(x), math.fsum(y) / len(y)
        candidates = [(mean_y, 0.0)]
        if self.form == "affine_tokens":
            variance = math.fsum((v - mean_x) ** 2 for v in x)
            if variance <= 1e-24 * len(x):
                raise ValueError(
                    "Affine fit needs distinct token counts; select constant explicitly"
                )
            slope = (
                math.fsum(
                    (a - mean_x) * (b - mean_y) for a, b in zip(x, y, strict=True)
                )
                / variance
            )
            intercept = mean_y - slope * mean_x
            if slope >= 0 and intercept >= 0:
                candidates.append((intercept, slope))
            # Both boundary optima are required; clipping unconstrained a/b is wrong.
            candidates.append(
                (
                    0.0,
                    math.fsum(a * b for a, b in zip(x, y, strict=True))
                    / math.fsum(a * a for a in x),
                )
            )
        intercept, slope = min(
            candidates,
            key=lambda pair: math.fsum(
                (pair[0] + pair[1] * a - b) ** 2 for a, b in zip(x, y, strict=True)
            ),
        )
        return replace(
            self,
            calibration=guard,
            intercept_seconds=intercept * yscale,
            seconds_per_token=slope * yscale / xscale,
        )

    def estimate(self, workload: WorkloadStats, profile: str) -> AdvantageEstimate:
        features = _features(workload)
        notes = (
            "Aggregate empirical advantage phase; no identified component decomposition",
            "Same-profile empirical fit within observed feature bounds; no cross-configuration accuracy claim",
        )
        if self.calibration is None:
            result = Estimate(
                None, "unsupported", notes=notes + ("Missing aggregate advantage fit",)
            )
        else:
            checked = self.calibration.apply(
                1.0,
                profile,
                "compute_advantage",
                features,
                timing_boundary=self.timing_boundary,
            )
            if checked.seconds is not None:
                checked = replace(
                    checked,
                    seconds=self.intercept_seconds
                    + self.seconds_per_token * workload.sequence_tokens,
                )
            result = replace(checked, notes=checked.notes + notes)
        return AdvantageEstimate(result, (), features)
