# SPDX-License-Identifier: Apache-2.0

"""Explicit sequential transition intervals with total residency per GPU pool."""

from dataclasses import dataclass

from ..joint import JointConfig
from .aggregate_rollout import nonnegative


@dataclass(frozen=True)
class TransitionPhase:
    seconds: float
    actor_gpus: int
    rollout_gpus: int

    def __post_init__(self) -> None:
        nonnegative("transition seconds", self.seconds)
        for name in ("actor_gpus", "rollout_gpus"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(
                    f"{name} must be a nonnegative unique allocation count"
                )


@dataclass(frozen=True)
class TransitionProfile:
    action: str
    source: JointConfig | None
    target: JointConfig | None
    phases: tuple[TransitionPhase, ...]
    source_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "phases", tuple(self.phases))
        if self.action not in {
            "start",
            "reconfigure",
            "checkpoint",
            "publish",
            "shutdown",
        }:
            raise ValueError("Unknown joint transition action")
        if not self.phases or not self.source_id:
            raise ValueError(
                "Even zero-cost transitions require explicit phases and provenance"
            )
        if self.action == "start" and (self.source is not None or self.target is None):
            raise ValueError("Start requires an empty source and resident target")
        if self.action == "shutdown" and (
            self.source is None or self.target is not None
        ):
            raise ValueError("Shutdown requires a resident source and empty target")
        if self.action == "reconfigure" and (
            self.source is None or self.target is None or self.source == self.target
        ):
            raise ValueError(
                "Reconfigure requires distinct complete source/target configs"
            )
        if self.action in {"publish", "checkpoint"}:
            if self.source is None or self.source != self.target:
                raise ValueError(
                    "Publication/checkpoint requires the same resident config"
                )
            if any(
                p.actor_gpus < self.target.actor.world_size
                or p.rollout_gpus < self.target.rollout.world_size
                for p in self.phases
            ):
                raise ValueError(
                    "Resident GPUs cannot disappear during publication/checkpoint"
                )


@dataclass(frozen=True)
class JointTransitionModel:
    profiles: tuple[TransitionProfile, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "profiles", tuple(self.profiles))
        keys = [(p.action, p.source, p.target) for p in self.profiles]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate directed joint transition profile")

    def lookup(
        self, action: str, source: JointConfig | None, target: JointConfig | None
    ) -> TransitionProfile | None:
        return next(
            (
                p
                for p in self.profiles
                if (p.action, p.source, p.target) == (action, source, target)
            ),
            None,
        )
