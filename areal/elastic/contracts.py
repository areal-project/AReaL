# SPDX-License-Identifier: Apache-2.0

"Shared observations, predictions, layouts and execution interfaces."

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from areal.elastic.reconfigurator.reference.evaluator.cost_model.contracts import (
        RolloutConfig,
        ServiceSlice,
    )

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, replace
from functools import cached_property
from itertools import product
from typing import Protocol

from areal.elastic.layout import ExecutionLayout


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


@dataclass(frozen=True, order=True)
class TrainingConfig:
    """Resource identity; empty profile_id preserves the pure-DP runtime contract.

    Named profiles identify complete planning configs and require an executor
    that explicitly validates that profile. Equal card counts are not identity.
    """

    world_size: int
    profile_id: str = ""

    def __post_init__(self) -> None:
        if type(self.world_size) is not int or self.world_size < 1:
            raise ValueError("world_size must be a positive integer")
        if (
            not isinstance(self.profile_id, str)
            or self.profile_id.strip() != self.profile_id
        ):
            raise ValueError("profile_id must be a string without outer whitespace")


@dataclass(frozen=True)
class BatchWorkload:
    """One logical update; counts never include replicated rank copies."""

    samples: int
    effective_tokens: int
    sequence_tokens: int
    rollout_seconds: float = 0.0
    actor_stats: WorkloadStats | None = None

    def __post_init__(self) -> None:
        if any(
            type(x) is not int or x < 1
            for x in (self.samples, self.effective_tokens, self.sequence_tokens)
        ):
            raise ValueError("Workload counts must be positive integers")
        if self.effective_tokens > self.sequence_tokens:
            raise ValueError("Effective tokens exceed sequence tokens")
        if not math.isfinite(self.rollout_seconds) or self.rollout_seconds < 0:
            raise ValueError("Invalid rollout duration")
        if self.actor_stats is not None:
            stats = self.actor_stats
            if isinstance(stats, dict):
                stats = WorkloadStats(**stats)
                object.__setattr__(self, "actor_stats", stats)
            if not isinstance(stats, WorkloadStats):
                raise ValueError("actor_stats must be WorkloadStats or its JSON object")
            if not math.isclose(stats.sequence_tokens, self.sequence_tokens, abs_tol=1):
                raise ValueError("Actor stats and workload sequence tokens disagree")
            if stats.loss_tokens is not None and not math.isclose(
                stats.loss_tokens, self.effective_tokens, abs_tol=1
            ):
                raise ValueError("Actor loss tokens and effective_tokens disagree")


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
    def from_areal_metrics(
        cls, metrics: dict, prefix: str = "ppo_actor"
    ) -> WorkloadStats:
        """Consume one AReaL exported stats row; no GPU access or new counters.

        Deliberately never use outer n_tokens as attended tokens. update/n_tokens
        is also not used implicitly because it may aggregate repeated exposure.
        """

        def get(key: str):
            return metrics.get(f"{prefix}/{key}")

        count, avg = get("n_seqs"), get("seq_len/avg")
        if count is None or avg is None:
            raise ValueError(
                "Need ppo_actor/n_seqs and ppo_actor/seq_len/avg from one stats row"
            )
        if not math.isfinite(count) or int(count) != count:
            raise ValueError("n_seqs must be an integer global DP sum")
        return cls(
            n_seqs=int(count),
            sequence_tokens=count * avg,
            max_seq_len=get("seq_len/max"),
            min_seq_len=get("seq_len/min"),
            loss_tokens=get("n_valid_tokens"),
            outer_padded_tokens=get("n_tokens"),
        )

    def lengths(self, *, tail: bool = False, capacity: int | None = None) -> list[int]:
        """Deterministic surrogate when logs lack lengths; never label it observed.

        The nominal surrogate has the smallest variance compatible with summary
        extrema. tail=True redistributes to bounded extremes for sensitivity tests.
        """
        if self.sequence_lengths is not None:
            return list(self.sequence_lengths)
        if self.per_dp_sequence_lengths is not None:
            return [n for rank in self.per_dp_sequence_lengths for n in rank]
        total = round(self.sequence_tokens)
        low = max(1, math.floor(self.min_seq_len or 1))
        high = max(
            math.ceil(self.sequence_tokens / self.n_seqs),
            math.ceil(self.max_seq_len or total - self.n_seqs + 1),
        )
        if capacity is not None:
            high = min(high, capacity)
        if high * self.n_seqs < total:
            raise ValueError("Mean sequence length exceeds microbatch capacity")
        if self.max_seq_len is not None and self.max_seq_len > high:
            raise ValueError("Maximum sequence length exceeds microbatch capacity")
        fixed = []
        if self.max_seq_len is not None:
            fixed.append(math.ceil(self.max_seq_len))
        if self.min_seq_len is not None and len(fixed) < self.n_seqs:
            fixed.append(low)
        free = self.n_seqs - len(fixed)
        remaining = total - sum(fixed) - free * low
        if remaining < 0 or remaining > free * (high - low):
            raise ValueError("Summary extrema cannot describe this actor batch")
        if not free:
            return fixed
        if tail:
            result = [low] * free
            for i in range(free):
                extra = min(high - low, remaining)
                result[i] += extra
                remaining -= extra
        else:
            quotient, extra = divmod(remaining, free)
            result = [low + quotient + (i < extra) for i in range(free)]
        return fixed + result


class Reconfigurator(Protocol):
    def decide(
        self, observation: OnlineObservation, evaluator: ConfigurationEvaluator
    ) -> OnlineDecision: ...


@dataclass(frozen=True)
class ExecutionConfig:
    """One candidate using AReaL's native backend:parallelism syntax.

    d/t/p/c/e denote data/tensor/pipeline/context/expert parallelism.
    MoE may use separate attn/ffn layouts; ffn:t denotes Expert TP.
    None leaves an optional setting for the run/backend to resolve.
    """

    # Megatron: d/t/p/c; MoE: megatron:(attn:<d/t/p/c>|ffn:<d/t/p/e>).
    # FFN d/t = Expert DP/TP; both layouts share PP and GPU count.
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
    # Enable expert parallelism for a MoE model inside each SGLang instance.
    sglang_enable_ep_moe: bool = False
    # Internal MoE expert-parallel group size.
    sglang_ep_size: int = 1
    # Active-request budget per SGLang instance; None = auto.
    # DP Attention divides this budget among its internal DP groups.
    sglang_max_running_requests: int | None = None
    # Prefill-token budget per internal SGLang DP scheduler; chunk size stays fixed.
    sglang_max_prefill_tokens: int = 32768
    # GPU memory fraction for model weights and KV cache; 0.9 means 90%.
    sglang_mem_fraction_static: float | None = 0.9

    # Example grid: 32 GPUs (4x8), Actor + Rollout <= 32.
    # Each cell: (parallel layouts only, all raw parameter combinations).
    # TP/ETP: 1/2/4/8; PP/CP/EP: 1/2/4/8/16; DP/replicas: positive integers.
    # Actor n_mbs/token_cap/pad: 4/4/2 choices; rollout concurrency: 4.
    # SGLang running/prefill/memory: 4/3/3 choices.
    # Model, batch, memory and physical-placement filters are not applied.
    # Rollout backend: SGLang.
    #
    # | Actor backend | Dense             | MoE                 |
    # | ------------- | ----------------- | ------------------- |
    # | Megatron      | (13980, 64419840)  | (171136, 788594688)  |

    @staticmethod
    def _dimensions(text, allowed):
        pairs = re.findall(r"([a-z])(\d+)", text)
        if not pairs or "".join(k + v for k, v in pairs) != text:
            raise ValueError("Use native backend:parallelism notation")
        if len(dict(pairs)) != len(pairs) or set(dict(pairs)) - set(allowed):
            raise ValueError("Repeated or unsupported parallel dimension")
        result = {k: int(v) for k, v in pairs}
        for name, value in result.items():
            positive_int(name, value)
        return result

    @cached_property
    def actor_parallel(self):
        """DP, TP, PP, CP, EP, Expert TP; expert DP is derived, not a GPU multiplier."""
        text = self.actor_backend.lower().replace(" ", "")
        if not text.startswith("megatron:"):
            raise ValueError("Elastic candidate evaluation supports Megatron Actors")
        text = text.split(":", 1)[1]
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1]
        if text.startswith("attn:"):
            attention, expert = text.split("|ffn:")
            a = self._dimensions(attention[5:], "dtpc")
            e = self._dimensions(expert, "dtpe")
        else:
            a, e = self._dimensions(text, "dtpc"), {}
        dp, tp, pp, cp = (a.get(k, 1) for k in "dtpc")
        ep, etp = e.get("e", 1), e.get("t", 1)
        if dp * tp * cp % (ep * etp):
            raise ValueError("Expert EP×TP must divide the attention stage world size")
        edp = dp * tp * cp // (ep * etp)
        if e.get("p", pp) != pp or e.get("d", edp) != edp:
            raise ValueError("Attention and FFN must have the same GPU count and PP")
        return dp, tp, pp, cp, ep, etp

    @cached_property
    def rollout_parallel(self):
        """Independent SGLang replicas, tensor parallelism, pipeline parallelism."""
        text = self.rollout_backend.lower().replace(" ", "")
        if not text.startswith("sglang:"):
            raise ValueError("Elastic candidate evaluation supports SGLang Rollout")
        dimensions = self._dimensions(text.split(":", 1)[1], "dtp")
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
            "sglang_ep_size",
        ):
            if getattr(self, name) is not None:
                positive_int(name, getattr(self, name))
        for name in (
            "actor_pad_to_maximum",
            "sglang_enable_dp_attention",
            "sglang_enable_ep_moe",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if not self.sglang_enable_dp_attention and self.sglang_dp_size != 1:
            raise ValueError("SGLang internal DP requires DP Attention")
        if not self.sglang_enable_ep_moe and self.sglang_ep_size != 1:
            raise ValueError("SGLang expert groups require MoE EP")
        if rt % self.sglang_dp_size or rt % self.sglang_ep_size:
            raise ValueError("SGLang internal DP and EP must divide each instance's TP")
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

    @property
    def actor_key(self):
        return (
            self.actor_backend,
            self.actor_n_mbs,
            self.actor_max_tokens_per_mb,
            self.actor_pad_to_maximum,
        )

    @property
    def rollout_key(self):
        return tuple(
            (k, v) for k, v in asdict(self).items() if not k.startswith("actor_")
        )

    @property
    def key(self):
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True).encode()
        ).hexdigest()[:24]

    @property
    def order_key(self):
        return (
            self.actor_dp,
            self.actor_tp,
            self.rollout_replicas,
            self.rollout_tp,
            self.key,
        )

    def with_rollout(self, rollout):
        return replace(
            self,
            **{k: v for k, v in asdict(rollout).items() if not k.startswith("actor_")},
        )

    @classmethod
    def generate(
        cls, *, batch_size, gpu_budget, actor, rollout, parameters=None, moe=False
    ):
        """Lazily enumerate a declared grid; model/packing checks stay in Evaluator.

        actor keys: dp/tp/pp/cp/ep/etp; rollout keys: replicas/tp/pp.
        parameters maps remaining ExecutionConfig fields to their allowed values.
        """
        positive_int("batch_size", batch_size)
        positive_int("gpu_budget", gpu_budget)
        if set(actor) - {"dp", "tp", "pp", "cp", "ep", "etp"} or set(rollout) - {
            "replicas",
            "tp",
            "pp",
        }:
            raise ValueError("Unknown parallel dimension in candidate grid")
        parameters = parameters or {}
        actor = {k: tuple(v) for k, v in actor.items()}
        rollout = {k: tuple(v) for k, v in rollout.items()}
        parameters = {k: tuple(v) for k, v in parameters.items()}
        for dimensions in (actor, rollout):
            for name, values in dimensions.items():
                for value in values:
                    positive_int(name, value)
        if set(parameters) - (
            set(cls.__dataclass_fields__) - {"actor_backend", "rollout_backend"}
        ):
            raise ValueError("Unknown execution parameter in candidate grid")
        names = tuple(parameters)
        for dp, tp, pp, cp, ep, etp in product(
            *(actor.get(k, (1,)) for k in ("dp", "tp", "pp", "cp", "ep", "etp"))
        ):
            if batch_size % dp or (not moe and (ep != 1 or etp != 1)):
                continue
            actor_gpus = dp * tp * pp * cp
            if actor_gpus >= gpu_budget or dp * tp * cp % (ep * etp):
                continue
            backend = f"megatron:(attn:d{dp}t{tp}p{pp}c{cp}|ffn:t{etp}e{ep})"
            for replicas, rtp, rpp in product(
                *(rollout.get(k, (1,)) for k in ("replicas", "tp", "pp"))
            ):
                if actor_gpus + replicas * rtp * rpp > gpu_budget:
                    continue
                for values in product(*(parameters[k] for k in names)):
                    try:
                        candidate = cls(
                            backend,
                            f"sglang:d{replicas}t{rtp}p{rpp}",
                            **dict(zip(names, values, strict=True)),
                        )
                    except ValueError:
                        continue
                    if not moe and candidate.sglang_ep_size != 1:
                        continue
                    yield candidate


# Causal online planning contracts shared by Basic and Reference.
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
    running: bool = True


@dataclass(frozen=True)
class OnlineObservation:
    now: float
    layout: ExecutionConfig | OnlineLayout
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


@dataclass(frozen=True)
class RollingSLO:
    completed: int
    pending: int
    decided: int
    jointly_passed: int
    fraction: float | None
    ttft_quantile_seconds: float
    tpot_quantile_seconds: float


@dataclass(frozen=True)
class ActorSupply:
    """Conditional target-only capacity; excludes old Actor work and idle time."""

    offered_episodes: int
    history_seconds: float
    arrival_rate: float
    actor_seconds_per_batch: float | None
    publication_seconds_per_batch: float | None
    capacity_rate: float | None
    credit_balance_rate: float | None
    sustainable: bool | None
    representative_batches: int
    padded_episodes: int
    substituted_episodes: int
    input_closed: bool
    reason: str


@dataclass(frozen=True)
class ActorMemorySafety:
    """Candidate-only margin; evaluation batches can contain replay templates."""

    admissible: bool
    reserve_fraction: float
    checked_batches: int
    peak_gib: float | None
    capacity_gib: float | None
    allowed_gib: float | None
    worst_batch: tuple[str, ...]
    observed_through: float
    reason: str
    scope: str = "all_completed"


@dataclass(frozen=True)
class CandidatePrediction:
    """Historical mean_gpu_seconds aliases score_gpu_seconds (K + H * GPUs).

    Request/update counts describe the serving window, not a completed workload.
    Legacy sampled forecasts retain their complete-work mean cost.
    """

    layout: ExecutionConfig | OnlineLayout
    supported: bool
    worst_request_fraction: float | None
    mean_gpu_seconds: float | None
    work_ids: tuple[str, ...]
    reason: str = ""
    source_ids: tuple[str, ...] = ()
    window_gpu_seconds: float | None = None
    tail_gpu_seconds: float | None = None
    actor_coverage: str = "legacy_sampled_execution"
    pruned: bool = False
    fraction_upper_bound: float | None = None
    mean_cost_lower_bound: float | None = None
    pruning_incumbent: tuple[str, float, float] | None = None
    switching_gpu_seconds: float | None = None
    running_gpu_seconds_per_second: float | None = None
    evaluation_seconds: float | None = None
    running_gpu_seconds: float | None = None
    score_gpu_seconds: float | None = None
    completed_requests: int | None = None
    pending_requests: int | None = None
    failed_pending_requests: int | None = None
    completed_updates: int | None = None
    retained_episodes: int | None = None
    actor_supply: tuple[ActorSupply, ...] = ()
    admission_credits_start: int | None = None
    admission_credits_end: float | None = None
    admission_credit_blocked_seconds: float | None = None
    actor_memory: ActorMemorySafety | None = None


@dataclass(frozen=True)
class CandidateWave:
    """Model-owned results and measured execution costs for one requested wave."""

    indices: tuple[int, ...]
    predictions: tuple[CandidatePrediction, ...]
    worker_pids: tuple[int, ...] = ()
    task_cpu_seconds: float = 0.0
    input_preparations: int = 0
    chunks: int = 0


class ConfigurationEvaluator(Protocol):
    def evaluate(
        self,
        view: OnlineObservation,
        forecasts: tuple[Forecast, ...],
        layout: ExecutionConfig | OnlineLayout,
    ) -> CandidatePrediction: ...


@dataclass(frozen=True)
class OnlineDecision:
    observed_at: float
    trigger: str
    target: ExecutionConfig | OnlineLayout
    predictions: tuple[CandidatePrediction, ...]
    forecast_seeds: tuple[int, ...]
    statistics: RollingSLO
    search_cpu_seconds: float
    reason: str
    replay_window: ReplayWindow | None = None
    workload_features: WorkloadFeatures | None = None
    net_window_gpu_seconds: float | None = None
    net_total_gpu_seconds: float | None = None


class ControlBusy(RuntimeError):
    """The native lifecycle is executing another operation."""


@dataclass(frozen=True)
class ControlSnapshot:
    """Live applied/published progress and independent durable checkpoint progress.

    ``boundary_ready`` requires completed publication and no active update;
    it does not require a checkpoint at the applied version.
    """

    closed: bool
    boundary_ready: bool
    generation: int
    applied_version: int
    published_version: int
    boundary_id: str
    checkpoint_version: int = 0


class CostModels(Protocol):
    source_ids: tuple[str, ...]

    def rollout_config(
        self, layout: ExecutionConfig | OnlineLayout
    ) -> RolloutConfig: ...
    def service(
        self,
        layout: ExecutionConfig | OnlineLayout,
        kv: int,
        slices: tuple[ServiceSlice, ...],
    ) -> float: ...
    def decode_interval(
        self,
        layout: ExecutionConfig | OnlineLayout,
        kv: int,
        slices: tuple[ServiceSlice, ...],
        max_steps: int,
    ) -> tuple[float, float, int]:
        """First iteration seconds, per-step increment and valid segment length."""
        ...

    def actor(
        self,
        layout: ExecutionConfig | OnlineLayout,
        batch: tuple[tuple[Turn, ...], ...],
    ) -> tuple[tuple[str, float], ...]: ...
    def publication(
        self,
        layout: ExecutionConfig | OnlineLayout,
        recipients: tuple[ExecutionConfig | OnlineLayout, ...],
    ) -> float: ...


@dataclass(frozen=True)
class SimulationConfig:
    batch_size: int
    staleness: int
    client_concurrency: int
    tool_capacities: tuple[tuple[str, int], ...]
    max_new_tokens: int
    max_context_tokens: int
    max_events: int
    max_simulated_seconds: float

    def __post_init__(self):
        for name in (
            "batch_size",
            "client_concurrency",
            "max_new_tokens",
            "max_context_tokens",
            "max_events",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"Invalid {name}")
        if self.staleness < 0 or self.max_simulated_seconds <= 0:
            raise ValueError("Explicit staleness and simulation budget are required")
        if not self.tool_capacities or any(n < 1 for _, n in self.tool_capacities):
            raise ValueError("Explicit positive tool pool capacities are required")


class ActorMemoryRejected(ValueError):
    """The declared Actor recipe cannot safely train this completed batch."""


@dataclass(frozen=True)
class NativePlacement:
    layout: ExecutionConfig | OnlineLayout
    execution: ExecutionLayout


@dataclass(frozen=True)
class HandoffConfig:
    node_gpus: tuple[int, ...]
    actor_startup_seconds: float
    rollout_startup_seconds: float
    actor_warmup_seconds: float
    rollout_warmup_seconds: float
    actor_state_bytes: int
    weight_bytes: int
    effective_bandwidth_GBps: float
    actor_sync_setup_seconds: float
    rollout_sync_setup_seconds: float
    provenance: str
    mode: str = "hot"
    native_source: ExecutionLayout | None = None
    native_devices: tuple[int, ...] = ()
    native_placements: tuple[NativePlacement, ...] = ()
    phase_seconds: tuple[tuple[str, float], ...] = ()
    native_checkpoint_version: int | None = None
    # Target residency budget; hot preparation may use the explicit spare pool.
    steady_gpu_budget: int | None = None

    @classmethod
    def from_dict(cls, raw):
        values = dict(raw)
        values["node_gpus"] = tuple(values["node_gpus"])
        values["native_devices"] = tuple(values.get("native_devices", ()))
        values["phase_seconds"] = tuple(
            tuple(x) for x in values.get("phase_seconds", ())
        )
        if values.get("native_source") is not None:
            values["native_source"] = ExecutionLayout.decode(values["native_source"])
        values["native_placements"] = tuple(
            NativePlacement(
                (
                    ExecutionConfig(**p["layout"])
                    if "actor_backend" in p["layout"]
                    else OnlineLayout(**p["layout"])
                ),
                ExecutionLayout.decode(p["execution"]),
            )
            for p in values.get("native_placements", ())
        )
        return cls(**values)

    def __post_init__(self):
        if self.steady_gpu_budget is None:
            object.__setattr__(self, "steady_gpu_budget", min(16, sum(self.node_gpus)))
        positive_int("steady_gpu_budget", self.steady_gpu_budget)
        if self.steady_gpu_budget > sum(self.node_gpus):
            raise ValueError("Steady GPU budget exceeds the physical pool")
        if self.mode not in {"hot", "cold"}:
            raise ValueError("Handoff mode must be hot or cold")
        if len(dict(self.phase_seconds)) != len(self.phase_seconds):
            raise ValueError("Native transition timing names must be unique")
        for name, seconds in self.phase_seconds:
            nonnegative(name, seconds)
        if self.native_source is not None:
            if len(set(self.native_devices)) != len(self.native_devices) or sum(
                self.node_gpus
            ) != len(self.native_devices):
                raise ValueError(
                    "Native topology must contain exactly the immutable reservation"
                )
            for placement in self.native_placements:
                target = placement.execution
                if (
                    target.world_size != placement.layout.actor_gpus
                    or len(target.rollout_devices) != placement.layout.rollout_gpus
                ):
                    raise ValueError(
                        "Physical mapping must match the candidate's Actor and Rollout GPU counts"
                    )
                if not set(target.actor_devices + target.rollout_devices) <= set(
                    self.native_devices
                ):
                    raise ValueError(
                        "Native candidate exceeds the original reservation"
                    )
        if self.native_checkpoint_version is not None and (
            type(self.native_checkpoint_version) is not int
            or self.native_checkpoint_version < 0
        ):
            raise ValueError("Native checkpoint version must be nonnegative or unknown")
        if not self.node_gpus or not self.provenance:
            raise ValueError(
                "Explicit pool topology and timing provenance are required"
            )
        for count in self.node_gpus:
            positive_int("node_gpus", count)
            if count > 8:
                raise ValueError("A modeled node contains at most eight GPUs")
        for name in (
            "actor_startup_seconds",
            "rollout_startup_seconds",
            "actor_warmup_seconds",
            "rollout_warmup_seconds",
            "actor_sync_setup_seconds",
            "rollout_sync_setup_seconds",
        ):
            nonnegative(name, getattr(self, name))
        for name in ("actor_state_bytes", "weight_bytes"):
            positive_int(name, getattr(self, name))
        nonnegative(
            "effective_bandwidth_GBps", self.effective_bandwidth_GBps, positive=True
        )
