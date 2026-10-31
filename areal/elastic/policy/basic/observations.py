# SPDX-License-Identifier: Apache-2.0

"""Immutable layouts, task samples and causal observations shared by policies."""

from __future__ import annotations

from dataclasses import dataclass

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
class ActorLifecycleView:
    state: str
    first_check_at: float
    first_check_done: bool
    check_pending: bool = False
    wake_at: float | None = None
    snapshot: tuple[int, int, int] | None = None


@dataclass(frozen=True)
class ActorView:
    batch: tuple[str, ...] = ()
    phase: int = 0
    phase_started_at: float = 0.0
    waiting: ActorWaitView | None = None
    lifecycle: ActorLifecycleView | None = None


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
