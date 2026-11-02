# SPDX-License-Identifier: Apache-2.0

"""Calibration-only empirical diagnostics, never the default actor predictor."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from ..core import (
    Estimate,
    Features,
    PhaseCalibration,
    UnsupportedEstimate,
    nonnegative,
)
from ._actor.packing import PackingPlan
from .advantage import ADVANTAGE_BOUNDARY
from .packing_observations import ObservedPackingWork, PackingComparison, _plan_calls


def _features(work: ObservedPackingWork, count: int) -> Features:
    workload = work.workload
    values = [
        ("microbatch_count", float(count)),
        ("sequence_tokens", workload.sequence_tokens),
        ("n_seqs", float(workload.n_seqs)),
    ]
    maximum = workload.max_seq_len
    if maximum is None and workload.sequence_lengths is not None:
        maximum = max(workload.sequence_lengths)
    if maximum is None and workload.per_dp_sequence_lengths is not None:
        maximum = max(max(rank) for rank in workload.per_dp_sequence_lengths)
    if maximum is not None:
        values.append(("max_seq_len", float(maximum)))
    return tuple(values)


@dataclass(frozen=True)
class PackingTimingObservation:
    work: ObservedPackingWork
    phase: str
    measured_seconds: float
    timing_boundary: str

    def __post_init__(self) -> None:
        if self.phase not in {"recompute_logp", "train_step"}:
            raise ValueError("Packing timing diagnostic requires logprob or train")
        if self.timing_boundary != ADVANTAGE_BOUNDARY:
            raise ValueError(
                "Packing timing diagnostic requires trainer host wall time with stats"
            )
        nonnegative("measured_seconds", self.measured_seconds, positive=True)
        self.work.phase_microbatches(self.phase)

    @property
    def sample_id(self) -> str:
        from ..core import fingerprint

        record = self.work.records[0]
        return fingerprint(record.run_id, record.global_step, self.phase)


@dataclass(frozen=True)
class PackingTimingResult:
    """The SAME frozen empirical g evaluated at observed and predicted MB counts.

    Both estimates are empirical diagnostics, not Transformer/communication validation.
    Missing coverage or out-of-range inputs stay unsupported.
    """

    sample_id: str
    measured_seconds: float
    packing_error: PackingComparison
    given_observed_packing: Estimate
    given_predicted_packing: Estimate

    @property
    def conditional_timing_error(self) -> float | None:
        """Absolute relative error conditioned on observed synchronized MB count."""
        predicted = self.given_observed_packing.seconds
        return (
            None
            if predicted is None
            else abs(predicted - self.measured_seconds) / self.measured_seconds
        )

    @property
    def packing_timing_delta(self) -> float | None:
        """Signed g(predicted count)-g(observed count), not an additive WAPE."""
        a, b = self.given_predicted_packing.seconds, self.given_observed_packing.seconds
        return None if a is None or b is None else a - b


@dataclass(frozen=True)
class PackingTimingDiagnostic:
    """Explicit same-profile T=a+b*nmb diagnostic; no optimizer/simulator coupling.

    nmb is the sum of synchronized counts over PPO calls, not over DP or replicas.
    Fit only on calibration rows. Constant requires explicit form selection.
    """

    phase: str
    form: str = "affine_microbatches"
    calibration: PhaseCalibration | None = None
    intercept_seconds: float | None = None
    seconds_per_microbatch: float | None = None

    def __post_init__(self) -> None:
        if self.phase not in {"recompute_logp", "train_step"} or self.form not in {
            "affine_microbatches",
            "constant",
        }:
            raise ValueError("Known packing phase and diagnostic form required")
        fitted = self.calibration is not None
        if any(
            (v is not None) != fitted
            for v in (self.intercept_seconds, self.seconds_per_microbatch)
        ):
            raise ValueError(
                "Diagnostic coefficients and calibration must be supplied together"
            )
        if fitted:
            nonnegative("intercept_seconds", self.intercept_seconds)
            nonnegative("seconds_per_microbatch", self.seconds_per_microbatch)
            if self.intercept_seconds == self.seconds_per_microbatch == 0:
                raise ValueError("Positive labels cannot fit a zero diagnostic")
            if self.form == "constant" and self.seconds_per_microbatch != 0:
                raise ValueError("Constant diagnostic cannot have a microbatch slope")
            guard = self.calibration
            if (
                guard.phase != self.phase
                or guard.scale != 1
                or guard.timing_boundary != ADVANTAGE_BOUNDARY
                or not {"microbatch_count", "sequence_tokens", "n_seqs"}
                <= {name for name, _, _ in guard.bounds}
            ):
                raise ValueError(
                    "Diagnostic calibration requires matching scope and workload bounds"
                )

    def fit(
        self, version: str, observations: tuple[PackingTimingObservation, ...]
    ) -> PackingTimingDiagnostic:
        if len(observations) < 2:
            raise ValueError("At least two calibration observations are required")
        first = observations[0]
        if any(
            o.phase != self.phase or o.work.profile != first.work.profile
            for o in observations
        ):
            raise ValueError("Diagnostic labels must share profile and phase")
        x = [o.work.phase_microbatches(self.phase) for o in observations]
        y = [o.measured_seconds for o in observations]
        features = [_features(o.work, n) for o, n in zip(observations, x, strict=True)]
        names = tuple(name for name, _ in features[0])
        if any(tuple(name for name, _ in row) != names for row in features):
            raise ValueError("Diagnostic labels must share available workload features")
        guard = PhaseCalibration(
            version,
            first.work.profile,
            self.phase,
            1.0,
            tuple(
                (
                    name,
                    min(dict(r)[name] for r in features),
                    max(dict(r)[name] for r in features),
                )
                for name in names
            ),
            tuple(o.sample_id for o in observations),
            ADVANTAGE_BOUNDARY,
        )
        xs, ys = max(x), max(y)
        x, y = [v / xs for v in x], [v / ys for v in y]
        mx, my = math.fsum(x) / len(x), math.fsum(y) / len(y)
        candidates = [(my, 0.0)]
        if self.form == "affine_microbatches":
            variance = math.fsum((v - mx) ** 2 for v in x)
            if variance <= 1e-24 * len(x):
                raise ValueError(
                    "Affine fit needs distinct counts; select constant explicitly"
                )
            slope = (
                math.fsum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
                / variance
            )
            intercept = my - slope * mx
            if slope >= 0 and intercept >= 0:
                candidates.append((intercept, slope))
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
            intercept_seconds=intercept * ys,
            seconds_per_microbatch=slope * ys / xs,
        )

    def predict(
        self,
        work: ObservedPackingWork,
        *,
        predicted_plan: PackingPlan | None = None,
        extrapolate_count: bool = False,
    ) -> Estimate:
        notes = (
            "Empirical given-microbatch-count diagnostic; not analytical actor model validation",
        )
        if self.calibration is None:
            return Estimate(
                None, "unsupported", notes=notes + ("Missing diagnostic fit",)
            )
        try:
            # Even paired prediction requires complete observed phase coverage.
            count = work.phase_microbatches(self.phase)
        except UnsupportedEstimate as error:
            return Estimate(None, "unsupported", notes=notes + (str(error),))
        if predicted_plan is not None:
            work.compare(predicted_plan)  # Validate same workload and complete layout.
            count = sum(
                len(rank)
                for (phase, _, dp), rank in _plan_calls(predicted_plan).items()
                if phase == self.phase and dp == 0
            )
        if type(extrapolate_count) is not bool:
            raise ValueError("extrapolate_count must be an explicit boolean")
        guard = self.calibration
        count_bounds = next(b for b in guard.bounds if b[0] == "microbatch_count")
        extrapolated = not count_bounds[1] <= count <= count_bounds[2]
        if extrapolate_count and extrapolated:
            # Only this calibrated feature may extrapolate. Profile, phase,
            # timing boundary and every other workload bound remain mandatory.
            guard = replace(
                guard,
                bounds=tuple(b for b in guard.bounds if b[0] != "microbatch_count"),
            )
        checked = guard.apply(
            self.intercept_seconds + self.seconds_per_microbatch * count,
            work.profile,
            self.phase,
            _features(work, count),
            timing_boundary=ADVANTAGE_BOUNDARY,
        )
        if extrapolated and extrapolate_count and checked.seconds is not None:
            checked = replace(
                checked,
                evidence="assumed",
                notes=checked.notes
                + (
                    f"Extrapolated microbatch_count={count} outside calibration [{count_bounds[1]}, {count_bounds[2]}]; this is not a runtime capacity limit",
                ),
            )
        return replace(checked, notes=checked.notes + notes)

    def evaluate(
        self,
        observation: PackingTimingObservation,
        predicted_plan: PackingPlan,
        *,
        extrapolate_count: bool = False,
    ) -> PackingTimingResult:
        if observation.phase != self.phase:
            raise ValueError("Diagnostic phase mismatch")
        if (
            self.calibration is not None
            and observation.sample_id in self.calibration.sample_ids
        ):
            raise ValueError(
                "Held-out evaluation cannot reuse calibration run/step labels"
            )
        return PackingTimingResult(
            observation.sample_id,
            observation.measured_seconds,
            observation.work.compare(predicted_plan),
            self.predict(observation.work, extrapolate_count=extrapolate_count),
            self.predict(
                observation.work,
                predicted_plan=predicted_plan,
                extrapolate_count=extrapolate_count,
            ),
        )
