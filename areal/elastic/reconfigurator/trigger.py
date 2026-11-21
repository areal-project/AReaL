# SPDX-License-Identifier: Apache-2.0

"""Detect persistent changes in offered demand at an observation boundary."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .evaluator.workload import OnlineObservation, WorkloadPattern


@dataclass(frozen=True)
class TriggerConfig:
    rate_window_seconds: float = 15
    minimum_arrivals: int = 8
    rate_stability_fraction: float = 0.15
    rate_change_fraction: float = 0.35
    cohort_size: int = 128
    ks_distance: float = 0.25
    quantile_change: float = 0.35

    def __post_init__(self):
        if not (
            self.rate_window_seconds > 0
            and self.minimum_arrivals > 0
            and self.cohort_size > 1
            and 0 <= self.rate_stability_fraction < self.rate_change_fraction < 1
            and 0 < self.ks_distance <= 1
            and self.quantile_change > 0
        ):
            raise ValueError("Invalid workload detection thresholds")


Distribution = tuple[int, ...]


@dataclass(frozen=True)
class WorkloadChange:
    reasons: tuple[str, ...]
    rate: float
    shapes: dict[str, Distribution]
    pattern: WorkloadPattern


def _relative(value, reference):
    return abs(value - reference) / max(abs(reference), 1e-9)


def _changed(a: Distribution, b: Distribution, config: TriggerConfig) -> bool:
    for q in (0.5, 0.9):
        if (
            _relative(a[int((len(a) - 1) * q)], b[int((len(b) - 1) * q)])
            >= config.quantile_change
        ):
            break
    else:
        return False
    return any(
        abs(bisect_right(a, x) / len(a) - bisect_right(b, x) / len(b))
        >= config.ks_distance
        for x in set(a).union(b)
    )


def _cohorts(episodes, now, size):
    """Use the newest two fully observed arrival cohorts for each signal."""
    inputs, trajectories = [], []
    for end in range(len(episodes) // size * size, size - 1, -size):
        group = episodes[end - size : end]
        if len(inputs) < 2 and all(e.initial_input_tokens is not None for e in group):
            inputs.append(group)
        if len(trajectories) < 2 and all(
            e.completed_at is not None and e.completed_at <= now and e.completed_turns
            for e in group
        ):
            trajectories.append(group)
        if len(inputs) == len(trajectories) == 2:
            break
    return inputs, trajectories


def _distributions(inputs, trajectories):
    if len(inputs) == 2:
        yield (
            "initial_input_tokens",
            [tuple(sorted(e.initial_input_tokens for e in group)) for group in inputs],
        )
    if len(trajectories) == 2:
        trajectories = [[e.completed_turns for e in group] for group in trajectories]
        for name, values in (
            (
                "turn_input_tokens",
                lambda group: (t.input_tokens for ep in group for t in ep),
            ),
            (
                "turn_output_tokens",
                lambda group: (t.output_tokens for ep in group for t in ep),
            ),
            (
                "episode_output_tokens",
                lambda group: (sum(t.output_tokens for t in ep) for ep in group),
            ),
            ("turn_count", lambda group: (len(ep) for ep in group)),
        ):
            yield name, [tuple(sorted(values(group))) for group in trajectories]


class WorkloadTrigger:
    """Compare external arrivals and observed token/turn shapes with the last search."""

    def __init__(self, config: TriggerConfig = TriggerConfig()):
        self.config = config
        self.arrivals: tuple[tuple[float, str], ...] = ()
        self.rate: float | None = None
        self.shapes: dict[str, Distribution] = {}

    def observe(self, view: OnlineObservation) -> WorkloadChange | None:
        from .evaluator.workload import WorkloadPattern

        now, cfg = view.now, self.config
        width = cfg.rate_window_seconds
        if view.input_closed or now < 2 * width:
            return None
        episodes = sorted(
            (e for e in view.episodes if e.available_at <= now),
            key=lambda e: (e.available_at, e.episode_id),
        )
        times = [e.available_at for e in episodes]
        split = bisect_right(times, now - width)
        start = 0 if now == 2 * width else bisect_right(times, now - 2 * width)
        recent = (len(times) - split) / width
        previous = (split - start) / width
        enough = len(times) - start >= cfg.minimum_arrivals
        stable = enough and _relative(recent, previous) <= cfg.rate_stability_fraction
        if stable or recent == previous == 0:
            # (now-width, now] is the same period that supplied the confirmed rate.
            self.arrivals = tuple(
                sorted(
                    (
                        ((e.available_at - now) % width, e.family)
                        for e in episodes[split:]
                    ),
                    key=lambda slot: slot[0],
                )
            )
        reasons = []
        if self.rate is None:
            if not stable:
                return None
            reasons.append("initial_workload")
        elif (stable or (recent == previous == 0)) and all(
            _relative(rate, self.rate) >= cfg.rate_change_fraction
            for rate in (recent, previous)
        ):
            reasons.append("arrival_rate")

        updates = {}
        inputs, trajectories = _cohorts(episodes, now, cfg.cohort_size)
        for name, (new, old) in _distributions(inputs, trajectories):
            if _changed(new, old, cfg):
                continue
            reference = self.shapes.get(name)
            if reference is None:
                pooled = tuple(sorted(new + old))
                if self.rate is None:
                    updates[name] = pooled
                else:
                    self.shapes[name] = (
                        pooled  # First observation calibrates the feature.
                    )
            elif _changed(old, reference, cfg) and _changed(new, reference, cfg):
                reasons.append(name)
                updates[name] = tuple(sorted(new + old))
        if not reasons:
            return None
        # An input-length shift may be visible before a whole newer cohort finishes.
        # Use completed examples from that input regime. For startup, use available
        # completions until there is a full cohort; never read an unfinished answer.
        if trajectories:
            pool = [e for group in reversed(trajectories) for e in group]
        else:
            pool = episodes[-2 * cfg.cohort_size :]
        if len(inputs) == 2:
            latest_inputs = [e for group in reversed(inputs) for e in group]
            latest = tuple(sorted(e.initial_input_tokens for e in latest_inputs))
            observed = tuple(
                sorted(
                    e.completed_turns[0].input_tokens
                    for e in pool
                    if e.completed_at is not None
                    and e.completed_at <= now
                    and e.completed_turns
                )
            )
            if (
                "initial_input_tokens" in reasons
                or not observed
                or _changed(latest, observed, cfg)
            ):
                pool = latest_inputs
        templates = tuple(
            e
            for e in pool
            if e.completed_at is not None
            and e.completed_at <= now
            and e.completed_turns
        )
        pattern = WorkloadPattern(width, self.arrivals, templates)
        return WorkloadChange(
            tuple(reasons), len(self.arrivals) / width, updates, pattern
        )

    def searched(self, change: WorkloadChange) -> None:
        """A completed search establishes the baseline even when it retains the layout."""
        if "initial_workload" in change.reasons or "arrival_rate" in change.reasons:
            self.rate = change.rate
        self.shapes.update(change.shapes)
