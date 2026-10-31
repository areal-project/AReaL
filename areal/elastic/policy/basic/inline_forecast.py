# SPDX-License-Identifier: Apache-2.0

"""Forecast a finite amount of training work using causal episode history."""

import random
from dataclasses import dataclass, replace

from .forecast_workload import continue_episode
from .observations import EpisodeSample, Forecast, OnlineObservation


@dataclass(frozen=True)
class InlineForecastConfig:
    seed: int = 42
    max_new_tokens: int = 32768
    unseen_tool_residual_seconds: float = 5.0


class InlineForecaster:
    def __init__(self, config: InlineForecastConfig):
        self.config = config

    def predict(
        self, view: OnlineObservation, pending, decision: int
    ) -> Forecast | None:
        history = tuple(e for e in view.episodes if e.completed_at is not None)
        if not history:
            return None
        seed = self.config.seed + decision * 1000003
        rng = random.Random(seed)
        rows = []
        for episode in view.episodes:
            if episode.completed_at is not None:
                turns = episode.completed_turns
            else:
                family = tuple(e for e in history if e.family == episode.family)
                prototypes = family or history
                turns = continue_episode(
                    self.config,
                    episode,
                    rng.choice(prototypes).completed_turns,
                    rng,
                    view.now,
                    tuple(t for e in prototypes for t in e.completed_turns),
                )
            rows.append(
                EpisodeSample(
                    episode.episode_id, episode.family, episode.available_at, turns
                )
            )
        for task in pending:
            family = tuple(e for e in history if e.family == task.family)
            prototype = rng.choice(family or history).completed_turns
            turns = [replace(prototype[0], input_tokens=task.input_tokens)]
            for turn in prototype[1:]:
                previous = turns[-1]
                turns.append(
                    replace(
                        turn,
                        input_tokens=previous.input_tokens
                        + previous.output_tokens
                        + previous.tool_output_tokens
                        + previous.next_input_overhead,
                    )
                )
            rows.append(
                EpisodeSample(task.episode_id, task.family, view.now, tuple(turns))
            )
        return Forecast(
            seed,
            tuple(rows),
            view.now,
            notes=(
                "Finite ordered task metadata; unknown outputs and tools resampled from completed history",
                "Candidate-neutral forecast; no true future outputs or wall-clock arrival prior",
            ),
        )
