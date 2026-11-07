# SPDX-License-Identifier: Apache-2.0

"""Basic aggregate inference rates, scoped to full configurations and workloads.

This intentionally does not reproduce request-level continuous batching. Rates
are per-replica aggregate throughput; TP speedups are never inferred from counts.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Literal


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


@dataclass(frozen=True)
class RolloutConfig:
    model_id: str
    hardware_id: str
    runtime_id: str
    sampling_id: str
    replicas: int
    tp: int
    max_batch_sequences: int
    max_batch_tokens: int
    kv_capacity_tokens: int
    max_inflight_batches: int
    backend: str = "sglang"

    def __post_init__(self) -> None:
        for name in ("model_id", "hardware_id", "runtime_id", "sampling_id", "backend"):
            if (
                not isinstance(getattr(self, name), str)
                or not getattr(self, name).strip()
            ):
                raise ValueError(f"{name} is required for full rollout identity")
        for name in (
            "replicas",
            "tp",
            "max_batch_sequences",
            "max_batch_tokens",
            "kv_capacity_tokens",
            "max_inflight_batches",
        ):
            positive_int(name, getattr(self, name))

    @property
    def world_size(self) -> int:
        return self.replicas * self.tp

    @property
    def profile_id(self) -> str:
        return (
            "rollout-"
            + hashlib.sha256(
                json.dumps(asdict(self), sort_keys=True).encode()
            ).hexdigest()
        )


@dataclass(frozen=True)
class RolloutStage:
    """Aggregate GPU work followed by an external wait.

    parallel_requests explicitly bounds usable replicas within this stage;
    unknown request parallelism defaults to one, never the whole batch size.
    """

    prefill_tokens: int
    decode_tokens: int
    external_seconds: float = 0.0
    parallel_requests: int = 1

    def __post_init__(self) -> None:
        for name in ("prefill_tokens", "decode_tokens"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        nonnegative("external_seconds", self.external_seconds)
        positive_int("parallel_requests", self.parallel_requests)


@dataclass(frozen=True)
class RolloutWorkload:
    stages: tuple[RolloutStage, ...]
    requests: int
    max_request_tokens: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "stages", tuple(self.stages))
        if not self.stages or any(not isinstance(s, RolloutStage) for s in self.stages):
            raise ValueError("Rollout workload requires explicit service/wait stages")
        positive_int("requests", self.requests)
        positive_int("max_request_tokens", self.max_request_tokens)
        if self.total_tokens < self.requests:
            raise ValueError("Request count exceeds total inference tokens")
        if any(s.parallel_requests > self.requests for s in self.stages):
            raise ValueError("Stage parallel requests exceed total request count")

    @property
    def total_tokens(self) -> int:
        return sum(s.prefill_tokens + s.decode_tokens for s in self.stages)


@dataclass(frozen=True)
class RolloutProfile:
    config: RolloutConfig
    prefill_tokens_per_second: float
    decode_tokens_per_second: float
    min_requests: int
    max_requests: int
    min_tokens: int
    max_tokens: int
    max_request_tokens: int
    source_id: str
    evidence: Literal["assumed", "measured"] = "assumed"

    def __post_init__(self) -> None:
        for name in ("prefill_tokens_per_second", "decode_tokens_per_second"):
            nonnegative(name, getattr(self, name), positive=True)
        for name in (
            "min_requests",
            "max_requests",
            "min_tokens",
            "max_tokens",
            "max_request_tokens",
        ):
            positive_int(name, getattr(self, name))
        if self.min_requests > self.max_requests or self.min_tokens > self.max_tokens:
            raise ValueError("Rollout profile bounds are reversed")
        if not self.source_id or self.evidence not in {"assumed", "measured"}:
            raise ValueError("Profile needs explicit source and evidence")


@dataclass(frozen=True)
class RolloutEstimate:
    supported: bool
    stages: tuple[
        tuple[float, float], ...
    ] = ()  # GPU service wall seconds, external wait.
    reason: str = ""
    source_id: str = ""
    evidence: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "stages", tuple(tuple(s) for s in self.stages))
        if self.supported and not self.stages:
            raise ValueError("Supported rollout estimate needs service stages")
        for gpu, external in self.stages:
            nonnegative("inference service", gpu)
            nonnegative("external wait", external)


@dataclass(frozen=True)
class RolloutCostModel:
    profiles: tuple[RolloutProfile, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "profiles", tuple(self.profiles))
        if len({p.config for p in self.profiles}) != len(self.profiles):
            raise ValueError("Duplicate rollout profile")

    def predict(
        self, config: RolloutConfig, workload: RolloutWorkload
    ) -> RolloutEstimate:
        if workload.max_request_tokens > config.kv_capacity_tokens:
            return RolloutEstimate(
                False, reason="A request exceeds per-replica KV capacity"
            )
        profile = next((p for p in self.profiles if p.config == config), None)
        if profile is None:
            return RolloutEstimate(False, reason="No exact rollout profile")
        if not (
            profile.min_requests <= workload.requests <= profile.max_requests
            and profile.min_tokens <= workload.total_tokens <= profile.max_tokens
            and workload.max_request_tokens <= profile.max_request_tokens
        ):
            return RolloutEstimate(
                False,
                reason="Rollout workload outside profile bounds",
                source_id=profile.source_id,
            )
        return RolloutEstimate(
            True,
            tuple(
                (
                    (
                        s.prefill_tokens / profile.prefill_tokens_per_second
                        + s.decode_tokens / profile.decode_tokens_per_second
                    )
                    / min(config.replicas, s.parallel_requests),
                    s.external_seconds,
                )
                for s in workload.stages
            ),
            source_id=profile.source_id,
            evidence=profile.evidence,
        )
