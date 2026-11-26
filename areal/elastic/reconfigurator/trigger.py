# SPDX-License-Identifier: Apache-2.0

"""Detect workload changes against the observation bound to the deployed layout."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .evaluator.workload import EpisodeView, OnlineObservation, WorkloadPattern


@dataclass(frozen=True)
class WorkloadChange:
    reasons: tuple[str, ...]
    rate: float
    shapes: dict[str, float]
    pattern: WorkloadPattern
    score: float | None = None


class StableWorkloadTrigger:
    """Use three arrival windows and observed, complete rollout shapes."""

    window_seconds = 10
    window_count = 3
    stability_spread = 0.20
    change_threshold = 0.20
    tool_time_floor_seconds = 1.0

    def __init__(self, cohort_size: int = 128):
        if cohort_size < 1:
            raise ValueError("Expected a positive workload cohort size")
        self.cohort_size = cohort_size
        self.baseline: WorkloadChange | None = None

    def bind(self, change: WorkloadChange | None) -> None:
        """Bind the observation that selected the active layout."""
        self.baseline = change

    @staticmethod
    def _shape(batch: tuple[EpisodeView, ...]) -> dict[str, float]:
        turns = tuple(turn for episode in batch for turn in episode.completed_turns)
        return {
            "mean_turns": len(turns) / len(batch),
            "mean_input_tokens": sum(turn.input_tokens for turn in turns) / len(turns),
            "mean_output_tokens": sum(turn.output_tokens for turn in turns)
            / len(turns),
            "mean_tool_seconds": sum(turn.tool_seconds or 0 for turn in turns)
            / len(turns),
        }

    def observe(
        self, view: OnlineObservation, episode_ids: tuple[str, ...] | None = None
    ) -> WorkloadChange | None:
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
        total = len(arrivals)
        if not total:
            return None
        times = [at for at, _ in arrivals]
        edges = [
            bisect_right(times, now - i * self.window_seconds)
            for i in range(self.window_count - 1, 0, -1)
        ]
        boundaries = (0, *edges, total)
        counts = [end - start for start, end in zip(boundaries, boundaries[1:])]
        if (
            max(counts) - min(counts)
            > self.stability_spread * total / self.window_count
        ):
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
            # Direct callers without Actor take events use the latest complete cohort.
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

        shapes = self._shape(batch)
        baseline = self.baseline
        score = None
        reason = "initial_workload"
        if baseline is not None:
            prior = len(baseline.pattern.arrivals)
            rate_delta = abs(total - prior) / prior
            shape_deltas = []
            for name, value in shapes.items():
                reference = baseline.shapes[name]
                denominator = (
                    max(reference, self.tool_time_floor_seconds)
                    if name == "mean_tool_seconds"
                    else reference
                )
                shape_deltas.append(abs(value - reference) / denominator)
            score = (rate_delta + sum(shape_deltas)) / (1 + len(shape_deltas))
            if score < self.change_threshold:
                return None
            reason = "workload_change"
        offsets = tuple(
            sorted(((at - now) % period, family) for at, family in arrivals)
        )
        return WorkloadChange(
            (reason,),
            total / period,
            shapes,
            WorkloadPattern(period, offsets, batch),
            score=score,
        )
