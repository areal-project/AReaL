# SPDX-License-Identifier: Apache-2.0

"""Deterministic historical workload replay over a fixed serving window."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from statistics import fmean

from .cost_model.rollout import nonnegative, positive_int
from .forecast import (
    EpisodeSample,
    Forecast,
    ForecastConfig,
    HistoricalForecaster,
    OnlineObservation,
    ReplayWindow,
)


@dataclass(frozen=True)
class HistoryReplayConfig:
    history_seconds: float = 120.0
    feature_window_seconds: float = 30.0
    change_ratio: float = 0.5
    change_checks: int = 2
    initial_regime_seconds: float = 120.0
    max_new_tokens: int = 32768
    unseen_tool_residual_seconds: float = 5.0

    def __post_init__(self):
        for name in (
            "history_seconds",
            "feature_window_seconds",
            "initial_regime_seconds",
            "change_ratio",
            "unseen_tool_residual_seconds",
        ):
            nonnegative(name, getattr(self, name), positive=True)
        positive_int("change_checks", self.change_checks)
        positive_int("max_new_tokens", self.max_new_tokens)


@dataclass(frozen=True)
class WorkloadFeatures:
    arrivals_per_second: float
    mean_initial_input: float
    families: tuple[tuple[str, float], ...]

    def distance(self, other: WorkloadFeatures) -> float:
        def relative(a, b):
            return abs(a - b) / max(abs(a), abs(b), 1e-12)

        left, right = dict(self.families), dict(other.families)
        mixture = (
            sum(
                abs(left.get(k, 0) - right.get(k, 0))
                for k in left.keys() | right.keys()
            )
            / 2
        )
        return max(
            relative(self.arrivals_per_second, other.arrivals_per_second),
            relative(self.mean_initial_input, other.mean_initial_input),
            mixture,
        )


class _LatestObserved:
    """Use the latest compatible observed example, with no random sampling."""

    @staticmethod
    def choice(values):
        return values[-1]


class HistoricalReplay:
    def __init__(self, config: HistoryReplayConfig):
        self.config = config
        self.regime_started_at = 0.0
        self.previous_regime_seconds: float | None = None
        self.regime_index = 0
        self.anchor: WorkloadFeatures | None = None
        self.features: WorkloadFeatures | None = None
        self._changed_at: float | None = None
        self._change_count = 0
        self._observed_at: float | None = None
        self._confirmed_at: float | None = None
        # Reuse only the observed-progress completion rules, with a deterministic
        # selector. This path never calls the stochastic predict() method.
        self._completion = HistoricalForecaster(
            ForecastConfig(
                config.history_seconds,
                config.initial_regime_seconds,
                1,
                1,
                0,
                config.max_new_tokens,
                config.unseen_tool_residual_seconds,
            )
        )

    def observe(self, view: OnlineObservation, *, confirm_change: bool = True):
        if self._observed_at == view.now and (
            not confirm_change or self._confirmed_at == view.now
        ):
            return
        self._observed_at = view.now
        cfg = self.config
        rows = [
            e
            for e in view.episodes
            if e.available_at >= max(0.0, view.now - cfg.feature_window_seconds)
        ]
        width = min(view.now, cfg.feature_window_seconds)
        families = Counter(e.family for e in rows)
        prompts = [
            e.completed_turns[0].input_tokens
            if e.completed_turns
            else e.current.input_tokens
            for e in rows
        ]
        self.features = WorkloadFeatures(
            len(rows) / max(width, 1e-9),
            fmean(prompts) if prompts else 0.0,
            tuple(sorted((k, n / len(rows)) for k, n in families.items())),
        )
        if not confirm_change:
            return  # CPU completion is not an extra periodic confirmation check.
        self._confirmed_at = view.now
        if self.anchor is None:
            if view.now >= cfg.feature_window_seconds:
                self.anchor = self.features
            return
        if self.features.distance(self.anchor) < cfg.change_ratio:
            self._changed_at, self._change_count = None, 0
            return
        if self._changed_at is None:
            self._changed_at = view.now
        self._change_count += 1
        if self._change_count >= cfg.change_checks:
            boundary = self._changed_at
            self.previous_regime_seconds = boundary - self.regime_started_at
            self.regime_started_at = boundary
            self.regime_index += 1
            self.anchor = self.features
            self._changed_at, self._change_count = None, 0

    def predict(
        self,
        view: OnlineObservation,
        decision: int,
        *,
        confirm_change: bool = True,
    ) -> tuple[Forecast, ...]:
        self.observe(view, confirm_change=confirm_change)
        cfg = self.config
        history = sorted(
            (e for e in view.episodes if e.completed_at is not None),
            key=lambda e: (e.completed_at, e.available_at, e.episode_id),
        )
        # Already completed batches remain usable even after the recent window
        # empties; an Actor recovery does not require new arriving episodes.
        if not history:
            return ()
        # The history length is an evaluation budget, not a remaining lifetime.
        remaining = cfg.history_seconds
        source = "fixed_history_window"
        if view.input_closed:
            source += "_observed_input_closed"
        start = max(0.0, view.now - cfg.history_seconds)
        span = max(0.0, view.now - start)
        recent = [e for e in view.episodes if start <= e.available_at < view.now]
        turns = {}
        substitutions = 0
        for episode in view.episodes:
            if episode.completed_at is not None:
                turns[episode.episode_id] = episode.completed_turns
            else:
                compatible = [
                    e for e in history if e.family == episode.family
                ] or history
                prototype = compatible[-1].completed_turns
                tool_history = tuple(t for e in compatible for t in e.completed_turns)
                turns[episode.episode_id] = self._completion._continuation(
                    episode,
                    prototype,
                    _LatestObserved(),
                    view.now,
                    tool_history,
                )
                substitutions += 1
        episodes = [
            EpisodeSample(e.episode_id, e.family, e.available_at, turns[e.episode_id])
            for e in view.episodes
        ]
        # Repeat the actual arrival slots in order. An unfinished arrival uses
        # the latest completed same-family example, with its known initial prompt.
        cycle = 0
        while not view.input_closed and remaining > 0 and span > 0 and recent:
            for index, episode in enumerate(recent):
                when = view.now + cycle * span + episode.available_at - start
                if when >= view.now + remaining:
                    break
                prototype = episode.completed_turns
                if episode.completed_at is None:
                    compatible = [
                        e for e in history if e.family == episode.family
                    ] or history
                    prototype = compatible[-1].completed_turns
                    initial = (
                        episode.completed_turns[0].input_tokens
                        if episode.completed_turns
                        else episode.current.input_tokens
                    )
                    sequence = [replace(prototype[0], input_tokens=initial)]
                    for turn in prototype[1:]:
                        previous = sequence[-1]
                        sequence.append(
                            replace(
                                turn,
                                input_tokens=previous.input_tokens
                                + previous.output_tokens
                                + previous.tool_output_tokens
                                + previous.next_input_overhead,
                            )
                        )
                    prototype = tuple(sequence)
                episodes.append(
                    EpisodeSample(
                        f"history-replay-{decision}-{cycle}-{index}",
                        episode.family,
                        when,
                        prototype,
                    )
                )
            cycle += 1
            if cycle * span >= remaining:
                break
        evidence = ReplayWindow(
            self.regime_started_at,
            self.previous_regime_seconds,
            remaining,
            source,
            start,
            span,
            len(recent),
            sum(e.completed_at is not None for e in recent),
            substitutions,
        )
        return (
            Forecast(
                0,
                tuple(episodes),
                view.now + remaining,
                (
                    "Deterministic replay of observed arrival slots and historical task shapes",
                    "Unfinished work uses the latest compatible completed history; no true suffix or RNG",
                    "Fixed serving window; regime age never reduces the evaluation horizon",
                ),
                evidence,
            ),
        )
