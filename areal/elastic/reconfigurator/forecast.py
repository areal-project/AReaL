# SPDX-License-Identifier: Apache-2.0

"""Observed workload history, replay and sampled continuations."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, replace
from statistics import fmean

from areal.elastic.contracts import (
    EpisodeSample,
    EpisodeView,
    Forecast,
    OnlineObservation,
    ReplayWindow,
    Turn,
    WorkloadFeatures,
    nonnegative,
    positive_int,
)


def continue_episode(
    config,
    episode: EpisodeView,
    prototype: tuple[Turn, ...],
    rng,
    now: float,
    tool_history: tuple[Turn, ...],
) -> tuple[Turn, ...]:
    current = episode.current
    assert current is not None
    index = len(episode.completed_turns)
    tail = list(prototype[index:] or (replace(prototype[-1], tool_seconds=None),))
    first = tail[0]
    if current.completed_at is None:
        eligible = [
            t.output_tokens for t in prototype if t.output_tokens > current.generated
        ]
        output = (
            rng.choice(eligible)
            if eligible
            else current.generated + max(1, first.output_tokens)
        )
        output = min(config.max_new_tokens, max(current.generated + 1, output))
        first = replace(first, input_tokens=current.input_tokens, output_tokens=output)
    else:
        tool = episode.tool
        assert tool is not None
        elapsed = (
            max(0.0, now - tool.started_at) if tool.started_at is not None else 0.0
        )
        tools = [
            t
            for t in tool_history
            if t.tool_seconds is not None and t.tool_pool == tool.pool
        ]
        survivors = [t for t in tools if t.tool_seconds > elapsed]
        if survivors:
            source = rng.choice(survivors)
            total = float(source.tool_seconds)
        else:
            positive = [t for t in tools if t.tool_seconds > 0]
            source = (
                rng.choice(positive)
                if positive
                else Turn(
                    current.input_tokens,
                    current.generated,
                    config.unseen_tool_residual_seconds,
                    0,
                    tool.pool,
                )
            )
            total = elapsed + float(source.tool_seconds)
        first = replace(
            source,
            input_tokens=current.input_tokens,
            output_tokens=current.generated,
            tool_seconds=total,
            tool_pool=tool.pool,
        )
        if len(tail) == 1:
            tail.append(replace(prototype[-1], tool_seconds=None))
    tail[0] = first
    for i in range(1, len(tail)):
        previous = tail[i - 1]
        tail[i] = replace(
            tail[i],
            input_tokens=previous.input_tokens
            + previous.output_tokens
            + previous.tool_output_tokens
            + previous.next_input_overhead,
        )
    return episode.completed_turns + tuple(tail)


@dataclass(frozen=True)
class ForecastConfig:
    history_seconds: float
    horizon_seconds: float
    samples: int
    minimum_completed_episodes: int
    seed: int
    max_new_tokens: int
    unseen_tool_residual_seconds: float

    def __post_init__(self):
        for name in ("history_seconds", "horizon_seconds"):
            nonnegative(name, getattr(self, name), positive=True)
        for name in ("samples", "minimum_completed_episodes", "max_new_tokens"):
            positive_int(name, getattr(self, name))
        nonnegative(
            "unseen_tool_residual_seconds",
            self.unseen_tool_residual_seconds,
            positive=True,
        )


class HistoricalForecaster:
    def __init__(self, config: ForecastConfig):
        self.config = config

    def predict(self, view: OnlineObservation, decision: int) -> tuple[Forecast, ...]:
        cfg = self.config
        recent = tuple(
            e for e in view.episodes if e.available_at >= view.now - cfg.history_seconds
        )
        history = tuple(
            e
            for e in view.episodes
            if e.completed_at is not None
            and e.completed_at >= view.now - cfg.history_seconds
        )
        if len(history) < cfg.minimum_completed_episodes:
            return ()
        start = max(0.0, view.now - cfg.history_seconds)
        rate = len(recent) / max(view.now - start, 1e-09)
        result = []
        for sample in range(cfg.samples):
            seed = cfg.seed + decision * 1000003 + sample
            rng = random.Random(seed)
            episodes = []
            for episode in view.episodes:
                if episode.completed_at is not None:
                    turns = episode.completed_turns
                else:
                    same_family = [e for e in history if e.family == episode.family]
                    prototype = rng.choice(same_family or history)
                    turns = self._continuation(
                        episode,
                        prototype.completed_turns,
                        rng,
                        view.now,
                        tuple(
                            t for e in same_family or history for t in e.completed_turns
                        ),
                    )
                episodes.append(
                    EpisodeSample(
                        episode.episode_id, episode.family, episode.available_at, turns
                    )
                )
            when = view.now
            if not view.input_closed and rate:
                while True:
                    when += rng.expovariate(rate)
                    if when > view.now + cfg.horizon_seconds:
                        break
                    prototype = rng.choice(history)
                    episodes.append(
                        EpisodeSample(
                            f"forecast-{decision}-{sample}-{len(episodes)}",
                            prototype.family,
                            when,
                            prototype.completed_turns,
                        )
                    )
            result.append(
                Forecast(seed, tuple(episodes), view.now + cfg.horizon_seconds)
            )
        return tuple(result)

    def _continuation(self, episode, prototype, rng, now, tool_history):
        return continue_episode(self.config, episode, prototype, rng, now, tool_history)


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
            len(rows) / max(width, 1e-09),
            fmean(prompts) if prompts else 0.0,
            tuple(sorted(((k, n / len(rows)) for k, n in families.items()))),
        )
        if not confirm_change:
            return
        self._confirmed_at = view.now
        if self.anchor is None:
            if view.now >= cfg.feature_window_seconds:
                self.anchor = self.features
            return
        if self.features.distance(self.anchor) < cfg.change_ratio:
            self._changed_at, self._change_count = (None, 0)
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
            self._changed_at, self._change_count = (None, 0)

    def predict(
        self, view: OnlineObservation, decision: int, *, confirm_change: bool = True
    ) -> tuple[Forecast, ...]:
        self.observe(view, confirm_change=confirm_change)
        cfg = self.config
        history = sorted(
            (e for e in view.episodes if e.completed_at is not None),
            key=lambda e: (e.completed_at, e.available_at, e.episode_id),
        )
        if not history:
            return ()
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
                    episode, prototype, _LatestObserved(), view.now, tool_history
                )
                substitutions += 1
        episodes = [
            EpisodeSample(e.episode_id, e.family, e.available_at, turns[e.episode_id])
            for e in view.episodes
        ]
        cycle = 0
        while not view.input_closed and remaining > 0 and (span > 0) and recent:
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
