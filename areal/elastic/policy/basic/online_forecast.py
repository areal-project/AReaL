# SPDX-License-Identifier: Apache-2.0

"""Causal arrival-window forecasts from completed episode history."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .cost_model.rollout import nonnegative, positive_int
from .forecast_workload import continue_episode
from .observations import EpisodeSample, Forecast, OnlineObservation


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
            return ()  # Explicit cold-start behavior: keep the initial deployment.
        start = max(0.0, view.now - cfg.history_seconds)
        rate = len(recent) / max(view.now - start, 1e-9)
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
                            t
                            for e in (same_family or history)
                            for t in e.completed_turns
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
