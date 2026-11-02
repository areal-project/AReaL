# SPDX-License-Identifier: Apache-2.0

"""Causal online observations and an empirical, candidate-neutral forecast.

This module does not import Reference or accept a ground-truth trace. A forecast
contains sampled work, never authoritative future outcomes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

from areal.elastic.contracts import nonnegative, positive_int

from ...contracts import (
    EpisodeSample,
    EpisodeView,
    Forecast,
    OnlineObservation,
    Turn,
)


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

    def _continuation(
        self,
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
                t.output_tokens
                for t in prototype
                if t.output_tokens > current.generated
            ]
            output = (
                rng.choice(eligible)
                if eligible
                else current.generated + max(1, first.output_tokens)
            )
            output = min(self.config.max_new_tokens, max(current.generated + 1, output))
            first = replace(
                first, input_tokens=current.input_tokens, output_tokens=output
            )
        else:
            # A tool invocation is already observable, even if a sampled prototype ended here.
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
                        self.config.unseen_tool_residual_seconds,
                        0,
                        tool.pool,
                    )
                )
                # Outside observed support use an explicit positive residual,
                # rather than equating survival with immediate completion.
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
