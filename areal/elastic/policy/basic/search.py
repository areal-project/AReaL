# SPDX-License-Identifier: Apache-2.0

"""Joint selection and conservative bounds; no future-workload access."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .forecast import OnlineLayout
    from .online import CandidatePrediction


def meets_target(prediction: CandidatePrediction, target: float) -> bool:
    """Historical candidates must meet SLO and replenish admission credits."""
    return (
        (prediction.actor_memory is None or prediction.actor_memory.admissible)
        and prediction.worst_request_fraction is not None
        and prediction.worst_request_fraction + 1e-12 >= target
        and all(s.sustainable is True for s in prediction.actor_supply)
    )


def credit_deficit(prediction: CandidatePrediction) -> float:
    """Only deficits break SLO ties; surplus capacity does not earn free GPUs."""
    return max(
        (
            max(0.0, -s.credit_balance_rate)
            if s.credit_balance_rate is not None
            else (0.0 if s.sustainable else inf)
            for s in prediction.actor_supply
        ),
        default=0.0,  # Legacy sampled forecasts have no supply constraint.
    )


def select_prediction(predictions, current, target, *, trigger="pressure"):
    """Apply the trigger's SLO filter, then minimize switching plus window cost."""
    if not predictions:
        return None
    work = predictions[0].work_ids
    runnable = [
        p
        for p in predictions
        if p.supported
        and (p.actor_memory is None or p.actor_memory.admissible)
        and not p.pruned
        and p.work_ids == work
        and p.worst_request_fraction is not None
        and p.mean_gpu_seconds is not None
    ]
    eligible = [p for p in runnable if meets_target(p, target)]
    if trigger == "pressure":
        eligible = [p for p in eligible if p.layout != current]
    if eligible:
        return min(
            eligible, key=lambda p: (p.mean_gpu_seconds, p.layout != current, p.layout)
        )
    if trigger == "slack":
        return next((p for p in runnable if p.layout == current), None)
    return min(
        runnable,
        key=lambda p: (
            -p.worst_request_fraction,
            credit_deficit(p),
            p.mean_gpu_seconds,
            p.layout != current,
            p.layout,
        ),
        default=None,
    )


@dataclass(frozen=True)
class SearchBound:
    context_id: tuple
    target: float
    best: CandidatePrediction
    fraction_pruning: bool = True

    def rejection(self, fraction_upper: float, mean_cost_lower: float) -> str:
        best = self.best
        if meets_target(best, self.target):
            if self.fraction_pruning and fraction_upper + 1e-12 < self.target:
                return "Request-fraction upper bound is below an eligible incumbent"
            if mean_cost_lower > best.mean_gpu_seconds + 1e-8:
                return "Score lower bound exceeds eligible incumbent"
        elif (
            not best.actor_supply
            and self.fraction_pruning
            and fraction_upper + 1e-12 < best.worst_request_fraction
        ):
            return "Request-fraction upper bound is below the best runnable incumbent"
        # Preserve all ties and candidates that could still reach the target.
        return ""


def candidate_order(layouts, current, preferred, score):
    """Prioritize useful seeds/neighbors; every original candidate remains."""

    def distance(layout: OnlineLayout):
        return sum(
            a != b
            for a, b in zip(
                (
                    layout.actor_dp,
                    layout.actor_tp,
                    layout.rollout_replicas,
                    layout.rollout_tp,
                ),
                (
                    current.actor_dp,
                    current.actor_tp,
                    current.rollout_replicas,
                    current.rollout_tp,
                ),
                strict=True,
            )
        )

    return tuple(
        sorted(
            enumerate(layouts),
            key=lambda item: (
                item[1] != current,
                item[1] not in preferred,
                distance(item[1]) != 1,
                score(item[1]),
                item[1],
                item[0],
            ),
        )
    )
