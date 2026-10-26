# SPDX-License-Identifier: Apache-2.0

"""Immutable offline joint-planning contracts, independent of Expert/runtime."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal, Protocol

from ...contracts import BatchWorkload, TrainingConfig, WorkloadStats
from .cost_model.rollout import (
    RolloutConfig,
    RolloutWorkload,
    nonnegative,
    positive_int,
)
from .serving import ServingMetrics


@dataclass(frozen=True)
class JointConfig:
    actor: TrainingConfig
    rollout: RolloutConfig

    @property
    def world_size(self) -> int:
        return self.actor.world_size + self.rollout.world_size


class JointWork(Protocol):
    actor: BatchWorkload


class JointWindow(Protocol):
    """Common search boundary; each model validates its own timeline contract."""

    current: JointConfig | None
    batches: tuple[JointWork, ...]


@dataclass(frozen=True)
class JointBatch:
    actor: BatchWorkload
    rollout: RolloutWorkload | None
    groups: int
    source_actor: TrainingConfig | None = None
    logical_stats: WorkloadStats | None = None
    inferred_lengths: bool = field(init=False)
    _prediction_stats: WorkloadStats = field(init=False, repr=False)

    def __post_init__(self) -> None:
        positive_int("groups", self.groups)
        stats = self.actor.actor_stats
        if stats is None:
            raise ValueError("Joint comparison requires actor sequence statistics")
        ranked = (
            stats.per_dp_sequence_lengths is not None
            or stats.train_microbatches is not None
        )
        if ranked and self.source_actor is None:
            raise ValueError("Rank/packing observations require source_actor")
        logical = self.logical_stats
        if logical is None:
            if (
                stats.per_dp_sequence_lengths is not None
                and stats.sequence_lengths is None
            ):
                # Rank-flattening loses trajectory order. Require caller attribution.
                raise ValueError(
                    "Provide logical_stats in original trajectory order for rank-only observations"
                )
            logical = replace(
                stats, per_dp_sequence_lengths=None, train_microbatches=None
            )
        if (
            logical.per_dp_sequence_lengths is not None
            or logical.train_microbatches is not None
        ):
            raise ValueError("logical_stats must precede rank dispatch and packing")
        if (logical.n_seqs, logical.sequence_tokens, logical.loss_tokens) != (
            stats.n_seqs,
            stats.sequence_tokens,
            stats.loss_tokens,
        ):
            raise ValueError("Logical and observed actor work disagree")
        if (
            stats.sequence_lengths is not None
            and logical.sequence_lengths != stats.sequence_lengths
        ):
            raise ValueError("logical_stats must preserve observed sequence order")
        for name in ("trajectory_group_sizes", "trajectory_dispatch_weights"):
            if getattr(stats, name) is not None and getattr(logical, name) != getattr(
                stats, name
            ):
                raise ValueError(
                    "logical_stats must preserve observed trajectory metadata"
                )
        if stats.per_dp_sequence_lengths is not None and (
            logical.sequence_lengths is None
            or sorted(logical.sequence_lengths)
            != sorted(n for rank in stats.per_dp_sequence_lengths for n in rank)
        ):
            raise ValueError(
                "logical_stats must preserve the observed rank sequence lengths"
            )
        # Retain the input provenance across dataclasses.replace; only the
        # prediction view materializes a shared surrogate.
        object.__setattr__(self, "logical_stats", logical)
        # Freeze one surrogate before any candidate can influence token capacity.
        object.__setattr__(self, "inferred_lengths", logical.sequence_lengths is None)
        if logical.sequence_lengths is None:
            lengths = tuple(logical.lengths())
            logical = replace(
                logical,
                sequence_lengths=lengths,
                min_seq_len=min(lengths),
                max_seq_len=max(lengths),
            )
        object.__setattr__(self, "_prediction_stats", logical)

    def stats_for(self, config: TrainingConfig) -> WorkloadStats:
        if self.source_actor == config:
            return replace(
                self._prediction_stats,
                per_dp_sequence_lengths=self.actor.actor_stats.per_dp_sequence_lengths,
                train_microbatches=self.actor.actor_stats.train_microbatches,
            )
        return self._prediction_stats


@dataclass(frozen=True)
class JointScenario:
    batches: tuple[JointBatch, ...]
    current: JointConfig | None
    samples_per_group: int
    staleness: int
    max_actor_gpus: int
    max_rollout_gpus: int
    max_total_gpus: int
    ready_batches: int = 0
    provider_version: int = 0
    accepted_batches: int = 0
    checkpoint_every: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "batches", tuple(self.batches))
        if not self.batches:
            raise ValueError("Joint planning needs a nonempty fixed workload window")
        for name in (
            "samples_per_group",
            "max_actor_gpus",
            "max_rollout_gpus",
            "max_total_gpus",
        ):
            positive_int(name, getattr(self, name))
        for name in (
            "staleness",
            "ready_batches",
            "provider_version",
            "accepted_batches",
            "checkpoint_every",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if (
            self.ready_batches > len(self.batches)
            or self.accepted_batches != self.provider_version + self.ready_batches
        ):
            raise ValueError("Initial ready/accepted batch counts are inconsistent")
        groups = self.batches[0].groups
        if any(
            b.groups != groups or b.actor.samples != groups * self.samples_per_group
            for b in self.batches
        ):
            raise ValueError("The window must preserve logical batch and samples/group")
        if any(b.rollout is None for b in self.batches[self.ready_batches :]):
            raise ValueError(
                "Future batches need separate inference workload, not historical actor wait"
            )


@dataclass(frozen=True)
class JointEvent:
    time: float
    kind: str
    batch: int = -1
    provider_version: int = 0
    accepted_batches: int = 0


@dataclass(frozen=True)
class JointPrediction:
    status: Literal["complete", "unsupported", "incomplete", "infeasible"]
    config: JointConfig
    seconds: float = 0.0
    actor_gpu_seconds: float = 0.0
    rollout_gpu_seconds: float = 0.0
    completed_batches: int = 0
    samples: int = 0
    effective_tokens: int = 0
    sequence_tokens: int = 0
    queue_wait_seconds: float = 0.0
    reason: str = ""
    events: tuple[JointEvent, ...] = ()
    assumptions: tuple[str, ...] = ()
    source_ids: tuple[str, ...] = ()
    serving: ServingMetrics | None = None

    def __post_init__(self) -> None:
        if self.status not in {"complete", "unsupported", "incomplete", "infeasible"}:
            raise ValueError("Unknown joint prediction status")
        for name in (
            "seconds",
            "actor_gpu_seconds",
            "rollout_gpu_seconds",
            "queue_wait_seconds",
        ):
            nonnegative(name, getattr(self, name))
        for name in (
            "completed_batches",
            "samples",
            "effective_tokens",
            "sequence_tokens",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("events", "assumptions", "source_ids"):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    @property
    def total_gpu_seconds(self) -> float:
        return self.actor_gpu_seconds + self.rollout_gpu_seconds


@dataclass(frozen=True)
class JointCandidateResult:
    config: JointConfig
    prediction: JointPrediction
    eligible: bool
    reason: str = ""


@dataclass(frozen=True)
class JointSearchResult:
    config: JointConfig | None
    prediction: JointPrediction | None
    candidates: tuple[JointCandidateResult, ...]
    unevaluated: int
    elapsed_seconds: float
    reason: str

    @property
    def exhaustive(self) -> bool:
        return self.unevaluated == 0
