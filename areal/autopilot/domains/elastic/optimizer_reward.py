# SPDX-License-Identifier: Apache-2.0

"""Acceptance and reward for an Optimizer's decisions on fixed replay windows."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite

from areal.autopilot.revision.contracts import Reward
from areal.elastic.reconfigurator.optimizer import Prediction


@dataclass(frozen=True)
class WindowResult:
    name: str
    slo_target: float
    request_fraction: float
    completed_requests: int
    completed_updates: int
    actor_sustainable: bool
    gpu_seconds: float
    evaluations: int

    @classmethod
    def from_prediction(
        cls, name: str, target: float, prediction: Prediction, evaluations: int
    ) -> WindowResult:
        if (
            not prediction.supported
            or prediction.request_fraction is None
            or not 0 <= prediction.request_fraction <= 1
            or prediction.actor_sustainable is None
            or prediction.gpu_seconds is None
            or not isfinite(prediction.gpu_seconds)
            or prediction.gpu_seconds < 0
            or prediction.completed_requests is None
            or prediction.completed_updates is None
        ):
            raise ValueError(
                f"{name}: selected layout has no complete replay prediction"
            )
        return cls(
            name,
            target,
            prediction.request_fraction,
            prediction.completed_requests,
            prediction.completed_updates,
            prediction.actor_sustainable,
            prediction.gpu_seconds,
            evaluations,
        )


def optimizer_reward(
    candidate: tuple[WindowResult, ...], incumbent: tuple[WindowResult, ...]
) -> Reward:
    """Check every window, then prefer lower GPU-seconds and fewer queries."""
    if not incumbent or [(w.name, w.slo_target) for w in candidate] != [
        (w.name, w.slo_target) for w in incumbent
    ]:
        raise ValueError(
            "Reward requires the same nonempty replay windows and SLO targets"
        )
    saved_gpus = sum(
        old.gpu_seconds - new.gpu_seconds
        for new, old in zip(candidate, incumbent, strict=True)
    )
    saved_evaluations = sum(
        old.evaluations - new.evaluations
        for new, old in zip(candidate, incumbent, strict=True)
    )
    metrics = {"gpu_seconds_saved": saved_gpus, "evaluations_saved": saved_evaluations}
    for new, old in zip(candidate, incumbent, strict=True):
        slo_floor = min(old.slo_target, old.request_fraction)
        if new.request_fraction < slo_floor and not isclose(
            new.request_fraction, slo_floor, rel_tol=0, abs_tol=1e-12
        ):
            reason = "Request SLO fell below the acceptance floor"
        elif new.completed_requests < old.completed_requests:
            reason = "Fewer requests completed"
        elif new.completed_updates < old.completed_updates or (
            old.actor_sustainable and not new.actor_sustainable
        ):
            reason = "Actor training progress regressed"
        elif new.gpu_seconds > old.gpu_seconds and not isclose(
            new.gpu_seconds, old.gpu_seconds, rel_tol=1e-12, abs_tol=1e-9
        ):
            reason = "GPU-seconds increased"
        else:
            continue
        return Reward(False, f"{old.name}: {reason}", metrics)
    if saved_gpus > 1e-9:
        return Reward(True, "Lower GPU-seconds", metrics)
    if abs(saved_gpus) <= 1e-9 and saved_evaluations > 0:
        return Reward(
            True,
            "Same GPU-seconds with fewer evaluations",
            metrics,
        )
    return Reward(False, "No improvement", metrics)
