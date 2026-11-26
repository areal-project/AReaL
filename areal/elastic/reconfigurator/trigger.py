# SPDX-License-Identifier: Apache-2.0

"""Per-role demand-to-capacity trigger with asymmetric soft scale-in."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .evaluator import CapacityEvaluator
    from .evaluator.rollout import RolloutCapacity
    from .evaluator.workload import OnlineObservation, WorkloadPattern
    from .optimizer import Layout

RoleAction = Literal["hold", "scale_out", "scale_in"]


@dataclass(frozen=True)
class ScaleSignal:
    """Per-role scaling trigger with current-layout capacity snapshot."""

    rate: float
    pattern: WorkloadPattern
    publication: float
    actor_capacity: float | None
    rollout_capacity: RolloutCapacity
    actor_pressure: float
    rollout_pressure: float
    actor_action: RoleAction
    rollout_action: RoleAction

    @property
    def reasons(self) -> tuple[str, ...]:
        items = []
        if self.actor_action != "hold":
            items.append(f"actor:{self.actor_action}")
        if self.rollout_action != "hold":
            items.append(f"rollout:{self.rollout_action}")
        return tuple(items)


def _pressure(rate: float, capacity: float | None) -> float:
    if capacity is None or not math.isfinite(capacity) or capacity <= 0:
        return math.inf
    return rate / capacity


class CapacityTrigger:
    """HexGen-3-style per-role pressure trigger (fast scale-out, soft scale-in)."""

    window_seconds = 10
    window_count = 3
    theta_up = 0.90
    theta_down = 0.70
    scale_in_windows = 3

    def __init__(self, cohort_size: int = 128):
        if cohort_size < 1:
            raise ValueError("Expected a positive workload cohort size")
        self.cohort_size = cohort_size
        self._low_streak: dict[str, int] = {"actor": 0, "rollout": 0}

    def reset(self) -> None:
        """Clear soft scale-in hysteresis after a layout change."""
        self._low_streak["actor"] = 0
        self._low_streak["rollout"] = 0

    def _classify(
        self, role: str, pressure: float, current_gpus: int, min_gpus: int
    ) -> RoleAction:
        if pressure > self.theta_up:
            self._low_streak[role] = 0
            return "scale_out"
        if pressure < self.theta_down and current_gpus > min_gpus:
            self._low_streak[role] += 1
            if self._low_streak[role] >= self.scale_in_windows:
                return "scale_in"
            return "hold"
        self._low_streak[role] = 0
        return "hold"

    def observe(
        self,
        view: OnlineObservation,
        current: Layout,
        evaluator: CapacityEvaluator,
        episode_ids: tuple[str, ...] | None = None,
        *,
        min_actor_gpus: int = 1,
        min_rollout_gpus: int = 1,
    ) -> ScaleSignal | None:
        from .evaluator.rollout import probe_capacity
        from .evaluator.workload import WorkloadPattern

        now = view.now
        period = self.window_count * self.window_seconds
        if view.input_closed or now < period:
            return None
        arrivals = sorted(
            (e.available_at, e.family)
            for e in view.episodes
            if now - period < e.available_at <= now
        )
        if not arrivals:
            return None

        completed = {
            e.episode_id: e
            for e in view.episodes
            if e.available_at <= now
            and e.completed_at is not None
            and e.completed_at <= now
            and e.completed_turns
        }
        if episode_ids is None:
            batch = tuple(
                sorted(
                    completed.values(), key=lambda e: (e.completed_at, e.episode_id)
                )[-self.cohort_size :]
            )
        else:
            if any(key not in completed for key in episode_ids):
                return None
            batch = tuple(completed[key] for key in episode_ids)
        if not batch:
            return None

        rate = len(arrivals) / period
        offsets = tuple(
            sorted(((at - now) % period, family) for at, family in arrivals)
        )
        pattern = WorkloadPattern(period, offsets, batch)
        publication = evaluator.publication_duration()
        actor_cap = evaluator.actor_capacity(current.actor, pattern, publication)
        target = evaluator.slo.minimum_request_fraction
        rollout_cap = probe_capacity(
            lambda load: evaluator.rollout_slo(current.rollout, pattern, load),
            rate,
            target,
        )
        actor_p = _pressure(rate, actor_cap)
        rollout_p = _pressure(rate, rollout_cap.lower)
        actor_action = self._classify(
            "actor", actor_p, current.actor.gpus, min_actor_gpus
        )
        rollout_action = self._classify(
            "rollout", rollout_p, current.rollout.gpus, min_rollout_gpus
        )
        if actor_action == "hold" and rollout_action == "hold":
            return None
        return ScaleSignal(
            rate=rate,
            pattern=pattern,
            publication=publication,
            actor_capacity=actor_cap,
            rollout_capacity=rollout_cap,
            actor_pressure=actor_p,
            rollout_pressure=rollout_p,
            actor_action=actor_action,
            rollout_action=rollout_action,
        )
