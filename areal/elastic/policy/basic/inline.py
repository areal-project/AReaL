# SPDX-License-Identifier: Apache-2.0

"""Fixed-budget, equal-work completion-time reconfiguration policy."""

import math
from dataclasses import dataclass
from typing import Protocol

from .inline_forecast import InlineForecaster
from .inline_trigger import InlineSearchTrigger
from .observations import OnlineLayout


@dataclass(frozen=True)
class InlinePolicyConfig:
    every_updates: int = 2
    lookahead_batches: int = 8
    minimum_completed_episodes: int = 8
    minimum_relative_improvement: float = 0.05
    max_evaluations: int = 64
    decision_delay_seconds: float = 0.0
    workload_trigger_enabled: bool = False
    feature_window_updates: int = 2
    workload_change_ratio: float = 0.3
    workload_change_checks: int = 2
    max_search_interval_updates: int = 16
    forecast_seed_by_update: bool = False

    def __post_init__(self):
        for name in (
            "every_updates",
            "lookahead_batches",
            "minimum_completed_episodes",
            "max_evaluations",
            "feature_window_updates",
            "workload_change_checks",
            "max_search_interval_updates",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"Positive {name} required")
        if not 0 <= self.minimum_relative_improvement < 1:
            raise ValueError("Relative improvement must be in [0, 1)")
        if not 0 < self.workload_change_ratio <= 1:
            raise ValueError("Workload change ratio must be in (0, 1]")
        if any(
            type(value) is not bool
            for value in (self.workload_trigger_enabled, self.forecast_seed_by_update)
        ):
            raise ValueError("Inline trigger and forecast switches must be boolean")
        if (
            not math.isfinite(self.decision_delay_seconds)
            or self.decision_delay_seconds < 0
        ):
            raise ValueError("Decision delay must be finite and nonnegative")


@dataclass(frozen=True)
class InlinePrediction:
    layout: OnlineLayout
    supported: bool
    seconds: float | None
    target_update: int
    published_update: int
    gpu_seconds: float
    peak_gpus: int
    reason: str


class InlineModel(Protocol):
    batch_size: int
    pipeline_episodes: int

    def evaluate(
        self, view, forecast, layout, target_update: int
    ) -> InlinePrediction: ...


@dataclass(frozen=True)
class InlineDecision:
    observed_at: float
    observed_version: int
    target: OnlineLayout
    target_update: int
    predictions: tuple[InlinePrediction, ...]
    forecast_seed: int | None
    work_ids: tuple[str, ...]
    relative_improvement: float
    reason: str


class InlinePolicy:
    def __init__(
        self, candidates, forecaster: InlineForecaster, config: InlinePolicyConfig
    ):
        self.candidates = tuple(dict.fromkeys(candidates))
        self.forecaster, self.config = forecaster, config
        self.last_checked_update = 0
        self.decision_index = 0
        self.search_trigger = InlineSearchTrigger(
            change_ratio=config.workload_change_ratio,
            change_checks=config.workload_change_checks,
            maximum_gap=config.max_search_interval_updates,
        )

    @property
    def next_check_update(self):
        previous = (
            self.search_trigger.last_probe_update
            if self.config.workload_trigger_enabled
            else self.last_checked_update
        )
        return previous + self.config.every_updates

    def due(self, view):
        return (
            not view.busy
            and view.provider_version
            >= self.last_checked_update + self.config.every_updates
            and sum(e.completed_at is not None for e in view.episodes)
            >= self.config.minimum_completed_episodes
        )

    def decide(
        self, view, pending, model: InlineModel, total_updates: int, *, targets=None
    ):
        self.decision_index += 1
        target_update = min(
            total_updates, view.training_version + self.config.lookahead_batches
        )
        # Bound the forecast to the same work horizon plus existing pipeline credits.
        needed = max(
            0,
            target_update * model.batch_size
            + model.pipeline_episodes
            - len(view.episodes),
        )
        forecast_index = (
            view.provider_version
            if self.config.forecast_seed_by_update
            else self.decision_index
        )
        forecast = self.forecaster.predict(view, pending[:needed], forecast_index)
        if forecast is None or target_update <= view.training_version:
            return InlineDecision(
                view.now,
                view.provider_version,
                view.layout,
                target_update,
                (),
                None,
                (),
                0.0,
                "Insufficient history or no remaining updates",
            )
        layouts = tuple(
            dict.fromkeys(
                (view.layout,)
                + tuple(targets if targets is not None else self.candidates)
            )
        )
        # Current is always included. The finite evaluation budget is explicit.
        layouts = layouts[: self.config.max_evaluations]
        scores = tuple(
            model.evaluate(view, forecast, c, target_update) for c in layouts
        )
        current = scores[0]
        supported = [s for s in scores if s.supported and s.seconds is not None]
        winner = (
            min(supported, key=lambda s: (s.seconds, s.layout != view.layout, s.layout))
            if supported
            else current
        )
        gain = (
            (current.seconds - winner.seconds) / max(current.seconds, 1e-9)
            if current.supported and winner.supported
            else (1.0 if winner.supported else 0.0)
        )
        target = (
            winner.layout
            if winner.supported
            and gain >= self.config.minimum_relative_improvement
            and winner.layout != view.layout
            else view.layout
        )
        return InlineDecision(
            view.now,
            view.provider_version,
            target,
            target_update,
            scores,
            forecast.seed,
            tuple(e.episode_id for e in forecast.episodes),
            gain,
            "Predicted equal-work time improvement"
            if target != view.layout
            else "Keep current layout",
        )
