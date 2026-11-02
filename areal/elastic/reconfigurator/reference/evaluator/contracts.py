# SPDX-License-Identifier: Apache-2.0

"""Candidate evaluation inputs shared by component assembly and execution."""

from __future__ import annotations

from dataclasses import dataclass

from areal.elastic.reconfigurator.reference.evaluator.cost_model.contracts import (
    ActorConfig,
    RolloutConfig,
    integer,
)
from areal.elastic.reconfigurator.reference.identity import fingerprint


@dataclass(frozen=True)
class Request:
    name: str
    input_tokens: int
    output_tokens: int
    predecessors: tuple[str, ...] = ()
    cached_prefix_tokens: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "predecessors", tuple(self.predecessors))
        integer("input_tokens", self.input_tokens, minimum=1)
        integer("output_tokens", self.output_tokens, minimum=1)
        integer("cached_prefix_tokens", self.cached_prefix_tokens)
        if not self.name or self.cached_prefix_tokens >= self.input_tokens:
            raise ValueError(
                "Request needs a name and at least one uncached input token"
            )


@dataclass(frozen=True)
class ExecutionConfig:
    actor: ActorConfig
    rollout: RolloutConfig
    actor_devices: tuple[str, ...]
    rollout_devices: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "actor_devices", tuple(self.actor_devices))
        object.__setattr__(self, "rollout_devices", tuple(self.rollout_devices))
        devices = self.actor_devices + self.rollout_devices
        if (
            len(self.actor_devices) != self.actor.parallel.world_size
            or len(self.rollout_devices) != self.rollout.world_size
        ):
            raise ValueError("Device identities must match each configured world size")
        if len(set(devices)) != len(devices) or any(not d for d in devices):
            raise ValueError(
                "This simulator requires disjoint actor/rollout device identities"
            )

    @property
    def key(self) -> str:
        return fingerprint(self)
