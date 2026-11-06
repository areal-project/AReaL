# SPDX-License-Identifier: Apache-2.0

"""Observed workload windows and confirmed triggers for Optimizer revision."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean, variance

from areal.autopilot.domains.elastic.contracts import (
    LengthStatistics,
    WorkloadChange,
    WorkloadProfile,
)
from areal.elastic.contracts import OnlineObservation


@dataclass(frozen=True)
class RevisionTriggerConfig:
    window_seconds: float = 120.0
    check_interval_seconds: float = 10.0
    relative_change: float = 0.15
    change_checks: int = 3
    cooldown_seconds: float = 120.0
    minimum_length_samples: int = 30
    noise_multiplier: float = 2.0

    def __post_init__(self):
        for name in (
            "window_seconds",
            "check_interval_seconds",
            "cooldown_seconds",
            "noise_multiplier",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 < self.relative_change <= 1:
            raise ValueError("relative_change must be in (0, 1]")
        for name in ("change_checks", "minimum_length_samples"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")


def _length_statistics(values):
    values = tuple(values)
    if not values:
        return LengthStatistics(0, None, None, None)
    mean = fmean(values)
    spread = variance(values) if len(values) > 1 else 0.0
    return LengthStatistics(
        len(values), mean, spread, math.sqrt(spread) / mean if mean else 0.0
    )


class RevisionTrigger:
    """Emit one change, then refresh the anchor after revision and cooldown."""

    def __init__(self, config: RevisionTriggerConfig | None = None):
        self.config = config or RevisionTriggerConfig()
        self.reference: WorkloadProfile | None = None
        self.current: WorkloadProfile | None = None
        self.next_check = 0.0
        self.triggered_at: float | None = None
        self.revision_running = False
        self._checks: dict[str, tuple[int, int]] = {}

    def profile(self, view: OnlineObservation) -> WorkloadProfile:
        start = max(0.0, view.now - self.config.window_seconds)
        recent_episodes = [
            episode
            for episode in view.episodes
            if start <= episode.available_at < view.now
        ]
        completed_episodes = [
            episode
            for episode in view.episodes
            if episode.completed_at is not None
            and start <= episode.completed_at < view.now
        ]
        tools = [
            turn.tool_seconds
            for episode in completed_episodes
            for turn in episode.completed_turns
            if turn.tool_seconds is not None
        ]
        return WorkloadProfile(
            started_at=start,
            ended_at=view.now,
            arrivals=len(recent_episodes),
            inputs=_length_statistics(
                request.input_tokens
                for request in view.requests
                if start <= request.ready_at < view.now
            ),
            outputs=_length_statistics(
                request.generated
                for request in view.requests
                if request.completed_at is not None
                and start <= request.completed_at < view.now
            ),
            mean_turns=fmean(len(e.completed_turns) for e in completed_episodes)
            if completed_episodes
            else None,
            mean_tool_seconds=fmean(tools) if tools else None,
        )

    def _changes(self):
        current, reference, config = self.current, self.reference, self.config
        difference = current.arrivals - reference.arrivals
        result = {}
        if (
            abs(difference)
            >= max(
                config.relative_change * max(reference.arrivals, 1),
                config.noise_multiplier
                * math.sqrt(current.arrivals + reference.arrivals),
            )
            and difference
        ):
            result["arrival_rate"] = difference / max(reference.arrivals, 1)
        for name in ("inputs", "outputs"):
            recent, anchor = getattr(current, name), getattr(reference, name)
            if min(recent.count, anchor.count) < config.minimum_length_samples:
                continue
            difference = recent.mean - anchor.mean
            standard_error = math.sqrt(
                recent.variance / recent.count + anchor.variance / anchor.count
            )
            if difference and abs(difference) >= max(
                config.relative_change * abs(anchor.mean),
                config.noise_multiplier * standard_error,
            ):
                result[name] = difference / max(abs(anchor.mean), 1e-12)
        return result

    def observe(self, view: OnlineObservation) -> WorkloadChange | None:
        if view.input_closed or view.now < self.next_check:
            return None
        self.next_check = view.now + self.config.check_interval_seconds
        self.current = self.profile(view)
        if self.reference is None:
            if view.now >= self.config.window_seconds:
                self.reference = self.current
            return None
        if self.triggered_at is not None:
            if (
                not self.revision_running
                and view.now >= self.triggered_at + self.config.cooldown_seconds
            ):
                self.reference = self.current
                self.triggered_at = None
                self._checks.clear()
            return None
        changes = self._changes()
        confirmed = []
        for name in ("arrival_rate", "inputs", "outputs"):
            ratio = changes.get(name, 0.0)
            if not ratio:
                self._checks.pop(name, None)
                continue
            direction = 1 if ratio > 0 else -1
            previous_direction, count = self._checks.get(name, (direction, 0))
            count = count + 1 if direction == previous_direction else 1
            self._checks[name] = direction, count
            if count >= self.config.change_checks:
                confirmed.append(name)
        if not confirmed:
            return None
        self.triggered_at = view.now
        self.revision_running = True
        return WorkloadChange(
            view.now,
            self.reference,
            self.current,
            tuple(confirmed),
            tuple((name, changes[name]) for name in confirmed),
        )

    def finish(self):
        self.revision_running = False
