# SPDX-License-Identifier: Apache-2.0

"""Causal online observations and an empirical, candidate-neutral forecast.

This module does not import Reference or accept a ground-truth trace. A forecast
contains sampled work, never authoritative future outcomes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

from .cost_model.rollout import nonnegative, positive_int


@dataclass(frozen=True, order=True)
class OnlineLayout:
    actor_dp: int
    actor_tp: int
    rollout_replicas: int
    rollout_tp: int

    def __post_init__(self):
        for name in ("actor_dp", "actor_tp", "rollout_replicas", "rollout_tp"):
            positive_int(name, getattr(self, name))
        if max(self.actor_gpus, self.rollout_gpus) > 16:
            raise ValueError("An online role uses at most sixteen GPUs")
        if self.actor_tp not in (1, 2, 4, 8) or self.rollout_tp not in (1, 2, 4, 8):
            raise ValueError("Online TP candidates are 1, 2, 4 or 8")

    @property
    def actor_gpus(self):
        return self.actor_dp * self.actor_tp

    @property
    def rollout_gpus(self):
        return self.rollout_replicas * self.rollout_tp

    @property
    def key(self):
        return f"a{self.actor_dp}t{self.actor_tp}-r{self.rollout_replicas}t{self.rollout_tp}"


@dataclass(frozen=True)
class Turn:
    """A completed historical turn, or a sampled turn in a forecast.

    tool_seconds=None denotes a terminal response. A live observation only puts
    fully observed turns here; its current unfinished turn has a separate view.
    """

    input_tokens: int
    output_tokens: int
    tool_seconds: float | None = None
    tool_output_tokens: int = 0
    tool_pool: str = "python"
    next_input_overhead: int = 0

    def __post_init__(self):
        positive_int("input_tokens", self.input_tokens)
        positive_int("output_tokens", self.output_tokens)
        if self.tool_seconds is not None:
            nonnegative("tool_seconds", self.tool_seconds)
        for name in ("tool_output_tokens", "next_input_overhead"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"Invalid {name}")


@dataclass(frozen=True)
class EpisodeSample:
    episode_id: str
    family: str
    available_at: float
    turns: tuple[Turn, ...]

    def __post_init__(self):
        if (
            not self.episode_id
            or not self.turns
            or self.turns[-1].tool_seconds is not None
        ):
            raise ValueError("An episode needs an identity and a terminal response")
        if any(t.tool_seconds is None for t in self.turns[:-1]):
            raise ValueError("Forecast episodes strictly alternate inference and tools")
        nonnegative("available_at", self.available_at)


@dataclass(frozen=True)
class RequestView:
    request_id: str
    episode_id: str
    turn: int
    input_tokens: int
    ready_at: float
    prefilled: int = 0
    generated: int = 0
    first_token_at: float | None = None
    last_token_at: float | None = None
    maximum_tbt_seconds: float = 0.0
    completed_at: float | None = None
    deployment: int | None = None
    replica: int | None = None
    version_counts: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class ToolView:
    pool: str
    ready_at: float
    started_at: float | None


@dataclass(frozen=True)
class EpisodeView:
    episode_id: str
    family: str
    available_at: float
    admitted: bool
    completed_at: float | None
    completed_turns: tuple[Turn, ...]
    current: RequestView | None
    tool: ToolView | None


@dataclass(frozen=True)
class ServiceView:
    replica: int
    started_at: float
    # id, known context, query, phase, emits a token. No final output length.
    slices: tuple[tuple[str, int, int, str, bool], ...]
    version: int


@dataclass(frozen=True)
class ActorWaitView:
    batch: tuple[str, ...]
    deployment: int
    since: float
    reason: str


@dataclass(frozen=True)
class ActorView:
    batch: tuple[str, ...] = ()
    phase: int = 0
    phase_started_at: float = 0.0
    waiting: ActorWaitView | None = None


@dataclass(frozen=True)
class OnlineObservation:
    now: float
    layout: OnlineLayout
    episodes: tuple[EpisodeView, ...]
    requests: tuple[RequestView, ...]
    services: tuple[ServiceView, ...]
    actor: ActorView
    consumed_episodes: int
    training_version: int
    provider_version: int
    busy: bool
    input_closed: bool
    actor_deployment: int = 0
    rollout_deployment: int = 1
    leases: tuple[tuple[int, str, tuple[tuple[int, int], ...]], ...] = ()


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


@dataclass(frozen=True)
class ReplayWindow:
    regime_started_at: float
    previous_regime_seconds: float | None
    expected_remaining_seconds: float
    duration_source: str
    template_started_at: float
    template_seconds: float
    observed_arrivals: int
    complete_templates: int
    historical_substitutions: int


@dataclass(frozen=True)
class Forecast:
    seed: int
    episodes: tuple[EpisodeSample, ...]
    input_closes_at: float
    notes: tuple[str, ...] = (
        "Empirical complete-episode resampling; short-term stationary arrival prior",
        "Completed-episode history has censoring/selection bias; not a calibrated tail forecast",
        "Tools use duration samples conditional on survival; beyond observed support use a sampled positive residual, or the explicit unseen-tool fallback",
    )
    replay: ReplayWindow | None = None


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
