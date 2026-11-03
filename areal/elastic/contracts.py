# SPDX-License-Identifier: Apache-2.0

"Shared observations, predictions, layouts and execution interfaces."

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from areal.elastic.reconfigurator.reference.evaluator.cost_model.contracts import (
        RolloutConfig,
        ServiceSlice,
    )

import math
from dataclasses import dataclass
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

    layout: OnlineLayout
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
        layout: OnlineLayout,
    ) -> CandidatePrediction: ...


@dataclass(frozen=True)
class OnlineDecision:
    observed_at: float
    trigger: str
    target: OnlineLayout
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

    def rollout_config(self, layout: OnlineLayout) -> RolloutConfig: ...
    def service(
        self, layout: OnlineLayout, kv: int, slices: tuple[ServiceSlice, ...]
    ) -> float: ...
    def actor(
        self, layout: OnlineLayout, batch: tuple[tuple[Turn, ...], ...]
    ) -> tuple[tuple[str, float], ...]: ...
    def publication(
        self, layout: OnlineLayout, recipients: tuple[OnlineLayout, ...]
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
    layout: OnlineLayout
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
                OnlineLayout(**p["layout"]), ExecutionLayout.decode(p["execution"])
            )
            for p in values.get("native_placements", ())
        )
        return cls(**values)

    def __post_init__(self):
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
                    placement.layout.actor_tp != 1
                    or placement.layout.rollout_tp != 1
                    or target.world_size != placement.layout.actor_dp
                    or len(target.rollout_devices) != placement.layout.rollout_replicas
                ):
                    raise ValueError(
                        "Native prediction requires matching FSDP DP / SGLang TP1 layouts"
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
