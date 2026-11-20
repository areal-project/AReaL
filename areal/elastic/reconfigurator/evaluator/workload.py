# SPDX-License-Identifier: Apache-2.0

"""Workload snapshots, candidate execution settings and serving objectives."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from functools import cached_property


def nonnegative(name: str, value: float, *, positive: bool = False) -> None:
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


def _positive_stat(name: str, value: float) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class WorkloadStats:
    """One accepted actor batch, after trajectory/turn expansion, before PPO splitting.

    sequence_tokens are attention_mask tokens, not outer n_tokens or loss tokens.
    Exact lengths and actual synchronized microbatch count are optional refinements.
    The scalar mb count is only unambiguous for ppo_n_minibatches=1.
    Optional trajectory_group_sizes correspond to TrajBatchMeta.traj_group_sizes.
    Per-DP lengths bypass unknown atomic trajectory dispatch; neither is required.
    """

    n_seqs: int
    sequence_tokens: float
    max_seq_len: float | None = None
    min_seq_len: float | None = None
    loss_tokens: float | None = None
    outer_padded_tokens: float | None = None
    sequence_lengths: tuple[int, ...] | None = None
    train_microbatches: int | None = None
    trajectory_group_sizes: tuple[int, ...] | None = None
    trajectory_dispatch_weights: tuple[float, ...] | None = None
    per_dp_sequence_lengths: tuple[tuple[int, ...], ...] | None = None

    def __post_init__(self) -> None:
        for name in (
            "sequence_lengths",
            "trajectory_group_sizes",
            "trajectory_dispatch_weights",
        ):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.per_dp_sequence_lengths is not None:
            object.__setattr__(
                self,
                "per_dp_sequence_lengths",
                tuple(tuple(rank) for rank in self.per_dp_sequence_lengths),
            )
        if type(self.n_seqs) is not int or self.n_seqs < 1:
            raise ValueError("n_seqs must be a positive actual actor sequence count")
        _positive_stat("sequence_tokens", self.sequence_tokens)
        mean = self.sequence_tokens / self.n_seqs
        if mean < 1:
            raise ValueError("Each actor sequence must contain at least one token")
        for name in (
            "min_seq_len",
            "max_seq_len",
            "loss_tokens",
            "outer_padded_tokens",
        ):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.min_seq_len is not None and not 1 <= self.min_seq_len <= mean:
            raise ValueError("min_seq_len must be between one and mean length")
        if self.max_seq_len is not None and self.max_seq_len < mean:
            raise ValueError("max_seq_len is smaller than mean length")
        if self.loss_tokens is not None and self.loss_tokens > self.sequence_tokens + 1:
            raise ValueError("Loss tokens cannot exceed attended tokens")
        if (
            self.outer_padded_tokens is not None
            and self.outer_padded_tokens + 1 < self.sequence_tokens
        ):
            raise ValueError(
                "Outer padded tokens cannot be smaller than attended tokens"
            )
        if self.sequence_lengths is not None:
            if len(self.sequence_lengths) != self.n_seqs or any(
                type(n) is not int or n < 1 for n in self.sequence_lengths
            ):
                raise ValueError("sequence_lengths must have n_seqs positive integers")
            if not math.isclose(
                sum(self.sequence_lengths), self.sequence_tokens, abs_tol=1
            ):
                raise ValueError("Exact lengths disagree with sequence_tokens")
            if (
                self.max_seq_len is not None
                and max(self.sequence_lengths) != self.max_seq_len
            ):
                raise ValueError("Exact lengths disagree with max_seq_len")
            if (
                self.min_seq_len is not None
                and min(self.sequence_lengths) != self.min_seq_len
            ):
                raise ValueError("Exact lengths disagree with min_seq_len")
        if self.train_microbatches is not None and (
            type(self.train_microbatches) is not int or self.train_microbatches < 1
        ):
            raise ValueError(
                "train_microbatches must be a positive synchronized per-rank count"
            )
        if self.trajectory_group_sizes is not None:
            if (
                any(type(n) is not int or n < 1 for n in self.trajectory_group_sizes)
                or sum(self.trajectory_group_sizes) != self.n_seqs
            ):
                raise ValueError(
                    "trajectory_group_sizes must partition n_seqs into nonempty atomic trajectories"
                )
        if self.trajectory_dispatch_weights is not None:
            if self.trajectory_group_sizes is None or len(
                self.trajectory_dispatch_weights
            ) != len(self.trajectory_group_sizes):
                raise ValueError(
                    "Dispatch weights require matching trajectory_group_sizes"
                )
            for weight in self.trajectory_dispatch_weights:
                _positive_stat("trajectory_dispatch_weight", weight)
        if self.per_dp_sequence_lengths is not None:
            flat = [n for rank in self.per_dp_sequence_lengths for n in rank]
            if (
                any(not rank for rank in self.per_dp_sequence_lengths)
                or len(flat) != self.n_seqs
                or any(type(n) is not int or n < 1 for n in flat)
                or not math.isclose(sum(flat), self.sequence_tokens, abs_tol=1)
            ):
                raise ValueError(
                    "per_dp_sequence_lengths must contain all attended sequences on nonempty DP ranks"
                )
            if (
                self.max_seq_len is not None
                and max(flat) != self.max_seq_len
                or self.min_seq_len is not None
                and min(flat) != self.min_seq_len
            ):
                raise ValueError("Per-DP lengths disagree with summary extrema")
            if self.sequence_lengths is not None and sorted(flat) != sorted(
                self.sequence_lengths
            ):
                raise ValueError("Per-DP lengths disagree with global sequence_lengths")

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

    def lengths(self, *, tail: bool = False, capacity: int | None = None) -> list[int]:
        """Return the supplied per-sequence or per-rank lengths for packing."""
        if self.sequence_lengths is not None:
            return list(self.sequence_lengths)
        if self.per_dp_sequence_lengths is not None:
            return [n for rank in self.per_dp_sequence_lengths for n in rank]
        raise ValueError("Actor packing requires per-sequence lengths")


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
    """One candidate using AReaL's native backend:parallelism syntax.

    d/t/p/c denote data/tensor/pipeline/context parallelism.
    EP/ETP search is suspended pending reconfiguration validation.
    The optimizer varies the backends and SGLang Attention DP. The other
    seven settings come from the run template for every candidate.
    None leaves an optional setting for the run/backend to resolve.
    """

    # Megatron: d/t/p/c. The parser retains MoE attn/ffn syntax for later use.
    actor_backend: str
    # sglang: dimensions d/t/p; d=replicas, t=TP and p=PP per replica.
    rollout_backend: str

    # Microbatch count per DP shard; a minimum when a token cap is also set.
    actor_n_mbs: int | None = 1
    # Packed-token cap per microbatch, before context-parallel splitting.
    actor_max_tokens_per_mb: int | None = None
    # Pad each packed microbatch to its configured token upper bound.
    actor_pad_to_maximum: bool = False
    # Sequence parallelism follows TP. Virtual pipeline stages are not searched.

    # In-flight rollout workflows; None falls back to consumer_batch_size.
    rollout_max_concurrent_rollouts: int | None = None
    # Enable data-parallel attention inside each SGLang instance.
    sglang_enable_dp_attention: bool = False
    # Internal attention DP degree; used when DP attention is enabled.
    sglang_dp_size: int = 1
    # Expert-parallel settings are reserved until reconfiguration is validated.
    # sglang_enable_ep_moe: bool = False
    # sglang_ep_size: int = 1
    # Active-request budget per SGLang instance; None = auto.
    # DP Attention divides this budget among its internal DP groups.
    sglang_max_running_requests: int | None = None
    # Prefill-token budget per internal SGLang DP scheduler; chunk size stays fixed.
    sglang_max_prefill_tokens: int = 32768
    # GPU memory fraction for model weights and KV cache; 0.8 means 80%.
    sglang_mem_fraction_static: float | None = 0.8

    @cached_property
    def actor_parallel(self):
        return parse_actor_parallel(self.actor_backend)

    @cached_property
    def rollout_parallel(self):
        """Independent SGLang replicas, tensor parallelism, pipeline parallelism."""
        text = self.rollout_backend.lower().replace(" ", "")
        if not text.startswith("sglang:"):
            raise ValueError("Elastic candidate evaluation supports SGLang Rollout")
        dimensions = _dimensions(text.split(":", 1)[1], "dtp")
        return tuple(dimensions.get(k, 1) for k in "dtp")

    def __post_init__(self):
        dp, tp, pp, cp, ep, etp = self.actor_parallel
        actor = f"d{dp}t{tp}p{pp}c{cp}"
        if ep != 1 or etp != 1:
            actor = f"(attn:{actor}|ffn:d{dp * tp * cp // (ep * etp)}t{etp}p{pp}e{ep})"
        object.__setattr__(self, "actor_backend", "megatron:" + actor)
        rd, rt, rp = self.rollout_parallel
        object.__setattr__(self, "rollout_backend", f"sglang:d{rd}t{rt}p{rp}")
        for name in (
            "actor_n_mbs",
            "actor_max_tokens_per_mb",
            "rollout_max_concurrent_rollouts",
            "sglang_max_running_requests",
            "sglang_max_prefill_tokens",
            "sglang_dp_size",
        ):
            if getattr(self, name) is not None:
                positive_int(name, getattr(self, name))
        for name in (
            "actor_pad_to_maximum",
            "sglang_enable_dp_attention",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if not self.sglang_enable_dp_attention and self.sglang_dp_size != 1:
            raise ValueError("SGLang internal DP requires DP Attention")
        if rt % self.sglang_dp_size:
            raise ValueError("SGLang internal DP must divide each instance's TP")
        fraction = self.sglang_mem_fraction_static
        if fraction is not None and (
            not math.isfinite(fraction) or not 0 < fraction < 1
        ):
            raise ValueError("SGLang static memory fraction must be in (0, 1)")

    actor_dp = property(lambda self: self.actor_parallel[0])
    actor_tp = property(lambda self: self.actor_parallel[1])
    actor_pp = property(lambda self: self.actor_parallel[2])
    actor_cp = property(lambda self: self.actor_parallel[3])
    actor_ep = property(lambda self: self.actor_parallel[4])
    actor_etp = property(lambda self: self.actor_parallel[5])
    rollout_replicas = property(lambda self: self.rollout_parallel[0])
    rollout_tp = property(lambda self: self.rollout_parallel[1])
    rollout_pp = property(lambda self: self.rollout_parallel[2])

    @property
    def actor_gpus(self):
        return math.prod(self.actor_parallel[:4])

    @property
    def rollout_gpus(self):
        return math.prod(self.rollout_parallel)


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

    @property
    def actor_key(self):
        return self.actor_dp, self.actor_tp

    @property
    def rollout_key(self):
        return self.rollout_replicas, self.rollout_tp

    def with_rollout(self, rollout):
        return OnlineLayout(
            self.actor_dp, self.actor_tp, rollout.rollout_replicas, rollout.rollout_tp
        )

    @property
    def order_key(self):
        return self.actor_dp, self.actor_tp, self.rollout_replicas, self.rollout_tp, ""


@dataclass(frozen=True)
class Turn:
    """A completed historical turn, or a sampled turn in a forecast.

    tool_seconds=None denotes a terminal response.
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
class EpisodeView:
    """Historical arrival with a complete turn sequence once it has finished."""

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
    busy: bool  # Actor update or reconfiguration in progress
    input_closed: bool


@dataclass(frozen=True)
class Forecast:
    episodes: tuple[EpisodeSample, ...]
    input_closes_at: float


class ActorMemoryRejected(ValueError):
    """The declared Actor recipe cannot safely train this completed batch."""


@dataclass(frozen=True)
class ServingSLO:
    ttft_seconds: float
    tpot_seconds: float
    minimum_request_fraction: float

    def __post_init__(self) -> None:
        nonnegative("ttft_seconds", self.ttft_seconds, positive=True)
        nonnegative("tpot_seconds", self.tpot_seconds, positive=True)
        nonnegative(
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "predecessors", tuple(self.predecessors))
        positive_int("input_tokens", self.input_tokens)
        positive_int("output_tokens", self.output_tokens)
        if type(self.cached_prefix_tokens) is not int or self.cached_prefix_tokens < 0:
            raise ValueError("cached_prefix_tokens must be a nonnegative integer")
        if not self.name or self.cached_prefix_tokens >= self.input_tokens:
            raise ValueError(
                "Request needs a name and at least one uncached input token"
            )


@dataclass(frozen=True)
class HistoryReplayConfig:
    history_seconds: float = 120.0

    def __post_init__(self):
        nonnegative("history_seconds", self.history_seconds, positive=True)


def _with_initial_input(turns: tuple[Turn, ...], input_tokens: int) -> tuple[Turn, ...]:
    result = [replace(turns[0], input_tokens=input_tokens)]
    for turn in turns[1:]:
        previous = result[-1]
        result.append(
            replace(
                turn,
                input_tokens=previous.input_tokens
                + previous.output_tokens
                + previous.tool_output_tokens
                + previous.next_input_overhead,
            )
        )
    return tuple(result)


@dataclass(frozen=True)
class HistoricalReplay:
    """Replay recent arrival slots using the latest compatible completed trajectories."""

    config: HistoryReplayConfig

    def predict(self, view: OnlineObservation, decision: int) -> tuple[Forecast, ...]:
        history = sorted(
            (
                e
                for e in view.episodes
                if e.completed_at is not None and e.completed_at <= view.now
            ),
            key=lambda e: (e.completed_at, e.available_at, e.episode_id),
        )
        if not history:
            return ()

        start = max(0.0, view.now - self.config.history_seconds)
        span = view.now - start
        end = view.now + self.config.history_seconds
        latest = {episode.family: episode for episode in history}

        episodes = []
        recent = []
        for episode in view.episodes:
            if view.input_closed or not start <= episode.available_at < view.now:
                continue
            if episode.completed_at is not None and episode.completed_at <= view.now:
                turns = episode.completed_turns
            else:
                turns = latest.get(episode.family, history[-1]).completed_turns
                if episode.initial_input_tokens is not None:
                    turns = _with_initial_input(turns, episode.initial_input_tokens)
            recent.append((episode.available_at, episode.family, turns))

        recent.sort(key=lambda slot: slot[0])
        cycle = 0
        while recent and cycle * span < self.config.history_seconds:
            for index, (available_at, family, turns) in enumerate(recent):
                when = view.now + cycle * span + available_at - start
                if when >= end:
                    break
                episodes.append(
                    EpisodeSample(
                        f"history-replay-{decision}-{cycle}-{index}",
                        family,
                        when,
                        turns,
                    )
                )
            cycle += 1
        return (Forecast(tuple(episodes), end),)
