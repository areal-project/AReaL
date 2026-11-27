# SPDX-License-Identifier: Apache-2.0

"""Workload snapshots, candidate execution settings and serving objectives."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property


def _nonnegative(name: str, value: float, *, positive: bool = False) -> None:
    if (
        isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
        or (positive and value == 0)
    ):
        raise ValueError(
            f"{name} must be finite and {'positive' if positive else 'nonnegative'}"
        )


def positive_int(name: str, value: int) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class WorkloadStats:
    """One accepted actor batch, after trajectory/turn expansion, before PPO splitting."""

    n_seqs: int
    sequence_tokens: float
    loss_tokens: float | None = None
    sequence_lengths: tuple[int, ...] | None = None
    trajectory_group_sizes: tuple[int, ...] | None = None
    trajectory_dispatch_weights: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        for name in (
            "sequence_lengths",
            "trajectory_group_sizes",
            "trajectory_dispatch_weights",
        ):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, tuple(getattr(self, name)))
        positive_int("n_seqs", self.n_seqs)
        _nonnegative("sequence_tokens", self.sequence_tokens, positive=True)
        if self.sequence_tokens / self.n_seqs < 1:
            raise ValueError("Each actor sequence must contain at least one token")
        if (
            self.sequence_lengths is not None
            and len(self.sequence_lengths) != self.n_seqs
        ):
            raise ValueError("sequence_lengths must have n_seqs entries")
        if (
            self.trajectory_group_sizes is not None
            and sum(self.trajectory_group_sizes) != self.n_seqs
        ):
            raise ValueError("trajectory_group_sizes must sum to n_seqs")

    @classmethod
    def from_batch(cls, batch: tuple[tuple[Turn, ...], ...]) -> WorkloadStats:
        """Expand each episode into one Actor sequence per turn, keeping dispatch groups."""
        lengths = tuple(
            turn.input_tokens + turn.output_tokens
            for episode in batch
            for turn in episode
        )
        return cls(
            len(lengths),
            sum(lengths),
            loss_tokens=sum(
                turn.output_tokens for episode in batch for turn in episode
            ),
            sequence_lengths=lengths,
            trajectory_group_sizes=tuple(len(episode) for episode in batch),
            trajectory_dispatch_weights=tuple(
                len(episode)
                * max(turn.input_tokens + turn.output_tokens for turn in episode)
                for episode in batch
            ),
        )

    def lengths(self) -> list[int]:
        if self.sequence_lengths is None:
            raise ValueError("Actor packing requires per-sequence lengths")
        return list(self.sequence_lengths)


def _dimensions(text: str, allowed: str) -> dict[str, int]:
    pairs = re.findall(r"([a-z])(\d+)", text)
    if not pairs or "".join(k + v for k, v in pairs) != text:
        raise ValueError("Use native backend:parallelism notation")
    if len(dict(pairs)) != len(pairs) or set(dict(pairs)) - set(allowed):
        raise ValueError("Repeated or unsupported parallel dimension")
    result = {k: int(v) for k, v in pairs}
    for name, value in result.items():
        positive_int(name, value)
    return result


def parse_actor_parallel(backend: str) -> tuple[int, int, int, int, int, int]:
    """Return Megatron DP, TP, PP, CP, EP and expert TP for dense or MoE layouts."""
    text = backend.lower().replace(" ", "")
    if not text.startswith("megatron:"):
        raise ValueError("Elastic candidate evaluation supports Megatron Actors")
    text = text.split(":", 1)[1]
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1]
    if text.startswith("attn:"):
        attention, expert = text.split("|ffn:")
        attn = _dimensions(attention[5:], "dtpc")
        ffn = _dimensions(expert, "dtpe")
    else:
        attn, ffn = _dimensions(text, "dtpc"), {}
    dp, tp, pp, cp = (attn.get(k, 1) for k in "dtpc")
    ep, etp = ffn.get("e", 1), ffn.get("t", 1)
    if dp * tp * cp % (ep * etp):
        raise ValueError("Expert EP×TP must divide the attention stage world size")
    edp = dp * tp * cp // (ep * etp)
    if ffn.get("p", pp) != pp or ffn.get("d", edp) != edp:
        raise ValueError("Attention and FFN must have the same GPU count and PP")
    return dp, tp, pp, cp, ep, etp


@dataclass(frozen=True)
class ExecutionConfig:
    """One candidate using AReaL's native backend:parallelism syntax."""

    actor_backend: str
    rollout_backend: str
    actor_n_mbs: int | None = None
    actor_max_tokens_per_mb: int | None = None
    actor_pad_to_maximum: bool = False
    sglang_dp_size: int = 1
    sglang_enable_dp_attention: bool = False
    sglang_max_running_requests: int | None = None
    sglang_mem_fraction_static: float | None = None
    sglang_max_prefill_tokens: int = 16384
    rollout_max_concurrent_rollouts: int | None = None

    @cached_property
    def actor_parallel(self) -> tuple[int, int, int, int, int, int]:
        return parse_actor_parallel(self.actor_backend)

    @cached_property
    def rollout_parallel(self) -> tuple[int, int, int]:
        text = self.rollout_backend.lower().replace(" ", "")
        if not text.startswith("sglang:"):
            raise ValueError("Elastic candidate evaluation supports SGLang rollouts")
        dims = _dimensions(text.split(":", 1)[1], "dtp")
        return dims.get("d", 1), dims.get("t", 1), dims.get("p", 1)

    def __post_init__(self) -> None:
        _, _, _, _, ep, etp = self.actor_parallel
        if ep != 1 or etp != 1:
            raise ValueError(
                "EP/ETP search is suspended pending reconfiguration validation"
            )
        _, rtp, _ = self.rollout_parallel
        for name in (
            "actor_n_mbs",
            "actor_max_tokens_per_mb",
            "sglang_max_running_requests",
            "rollout_max_concurrent_rollouts",
        ):
            value = getattr(self, name)
            if value is not None:
                positive_int(name, value)
        positive_int("sglang_dp_size", self.sglang_dp_size)
        positive_int("sglang_max_prefill_tokens", self.sglang_max_prefill_tokens)
        if self.sglang_dp_size > 1 and not self.sglang_enable_dp_attention:
            raise ValueError("sglang_dp_size > 1 requires sglang_enable_dp_attention")
        if rtp % self.sglang_dp_size:
            raise ValueError("SGLang attention DP must divide rollout TP")
        if self.sglang_mem_fraction_static is not None and (
            not math.isfinite(self.sglang_mem_fraction_static)
            or not 0 < self.sglang_mem_fraction_static <= 1
        ):
            raise ValueError("sglang_mem_fraction_static must be in (0, 1]")

    @property
    def actor_dp(self) -> int:
        return self.actor_parallel[0]

    @property
    def actor_tp(self) -> int:
        return self.actor_parallel[1]

    @property
    def actor_pp(self) -> int:
        return self.actor_parallel[2]

    @property
    def actor_cp(self) -> int:
        return self.actor_parallel[3]

    @property
    def rollout_replicas(self) -> int:
        return self.rollout_parallel[0]

    @property
    def rollout_tp(self) -> int:
        return self.rollout_parallel[1]

    @property
    def rollout_pp(self) -> int:
        return self.rollout_parallel[2]

    @property
    def actor_gpus(self) -> int:
        return math.prod(self.actor_parallel[:4])

    @property
    def rollout_gpus(self) -> int:
        return math.prod(self.rollout_parallel)


@dataclass(frozen=True)
class Turn:
    """A completed historical turn, or a sampled turn in a forecast."""

    input_tokens: int
    output_tokens: int
    tool_seconds: float | None = None
    tool_output_tokens: int = 0
    tool_pool: str = "python"
    next_input_overhead: int = 0

    def __post_init__(self) -> None:
        positive_int("input_tokens", self.input_tokens)
        positive_int("output_tokens", self.output_tokens)
        if self.tool_seconds is not None:
            _nonnegative("tool_seconds", self.tool_seconds)


@dataclass(frozen=True)
class EpisodeSample:
    episode_id: str
    family: str
    available_at: float
    turns: tuple[Turn, ...]

    def __post_init__(self) -> None:
        if (
            not self.episode_id
            or not self.turns
            or self.turns[-1].tool_seconds is not None
        ):
            raise ValueError("An episode needs an identity and a terminal response")
        if any(t.tool_seconds is None for t in self.turns[:-1]):
            raise ValueError("Forecast episodes strictly alternate inference and tools")
        _nonnegative("available_at", self.available_at)


@dataclass(frozen=True)
class EpisodeView:
    episode_id: str
    family: str
    available_at: float
    completed_at: float | None
    completed_turns: tuple[Turn, ...]
    initial_input_tokens: int | None = None


@dataclass(frozen=True)
class OnlineObservation:
    now: float
    episodes: tuple[EpisodeView, ...]
    training_version: int
    provider_version: int
    busy: bool
    input_closed: bool


@dataclass(frozen=True)
class Forecast:
    episodes: tuple[EpisodeSample, ...]
    input_closes_at: float


@dataclass(frozen=True)
class WorkloadPattern:
    """One confirmed arrival period and whole, observed episodes for its request mix."""

    period_seconds: float
    arrivals: tuple[tuple[float, str], ...]
    templates: tuple[EpisodeView, ...]

    def forecast(
        self, start: float, horizon_seconds: float, *, episode_count: int | None = None
    ) -> Forecast | None:
        _nonnegative("horizon_seconds", horizon_seconds, positive=True)
        by_family: dict[str, list[tuple[Turn, ...]]] = defaultdict(list)
        for episode in self.templates:
            by_family[episode.family].append(episode.completed_turns)
        if not self.arrivals or any(
            family not in by_family for _, family in self.arrivals
        ):
            return None
        positions: dict[str, int] = defaultdict(int)
        samples = []
        cycles = math.ceil(horizon_seconds / self.period_seconds)
        time_scale = 1.0
        if episode_count is not None:
            if episode_count <= 0:
                raise ValueError("episode_count must be positive")
            full, tail = divmod(episode_count, len(self.arrivals))
            cycles = full + bool(tail)
            duration = full * self.period_seconds
            if tail:
                # End the partial period one mean arrival interval after its last offer.
                duration += self.arrivals[tail - 1][0] + self.period_seconds / len(
                    self.arrivals
                )
            time_scale = horizon_seconds / duration
        for cycle in range(cycles):
            for offset, family in self.arrivals:
                delay = (cycle * self.period_seconds + offset) * time_scale
                if episode_count is not None and len(samples) == episode_count:
                    break
                if episode_count is None and delay >= horizon_seconds:
                    break
                pool = by_family[family]
                turns = pool[positions[family] % len(pool)]
                positions[family] += 1
                samples.append(
                    EpisodeSample(
                        f"pattern-{len(samples)}", family, start + delay, turns
                    )
                )
        return Forecast(tuple(samples), start + horizon_seconds)


class ActorMemoryRejected(ValueError):
    """The declared Actor recipe cannot safely train this completed batch."""


@dataclass(frozen=True)
class ServingSLO:
    ttft_seconds: float
    tpot_seconds: float
    minimum_request_fraction: float

    def __post_init__(self) -> None:
        _nonnegative("ttft_seconds", self.ttft_seconds, positive=True)
        _nonnegative("tpot_seconds", self.tpot_seconds, positive=True)
        _nonnegative(
            "minimum_request_fraction", self.minimum_request_fraction, positive=True
        )
        if self.minimum_request_fraction > 1:
            raise ValueError("minimum_request_fraction must be at most one")


@dataclass(frozen=True)
class Request:
    name: str
    input_tokens: int
    output_tokens: int
    predecessors: tuple[str, ...] = ()
    cached_prefix_tokens: int = 0
