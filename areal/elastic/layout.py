# SPDX-License-Identifier: Apache-2.0

"""Pure layout comparison and ordered native reconfiguration plans."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionLayout:
    """Actor DP and physical GPU placement for Actor and Rollout."""

    world_size: int
    actor_devices: tuple[int, ...]
    rollout_devices: tuple[int, ...]

    def __post_init__(self):
        devices = self.actor_devices + self.rollout_devices
        if (
            type(self.world_size) is not int
            or not 1 <= self.world_size <= len(self.actor_devices)
            or not self.rollout_devices
            or len(set(devices)) != len(devices)
            or any(type(device) is not int or device < 0 for device in devices)
        ):
            raise ValueError(
                "Require disjoint nonempty GPU leases and an Actor DP count fitting its lease"
            )

    @classmethod
    def decode(cls, value):
        return cls(
            value["world_size"],
            tuple(value["actor_devices"]),
            tuple(value["rollout_devices"]),
        )


@dataclass(frozen=True)
class ReconfigurationPlan:
    source: ExecutionLayout
    target: ExecutionLayout
    actor_running: bool
    steps: tuple[str, ...]

    @property
    def rebuilds_rollout(self) -> bool:
        return "stop_rollout" in self.steps


class ReconfigurationPlanner:
    """Generate a sequential baseline; creating a plan performs no engine operations."""

    def plan(
        self,
        source: ExecutionLayout,
        target: ExecutionLayout,
        *,
        actor_running: bool,
        force_actor: bool = False,
        force_rollout: bool = False,
    ) -> ReconfigurationPlan:
        actor_changed = (
            source.world_size != target.world_size
            or source.actor_devices != target.actor_devices
            or force_actor
        )
        rollout_changed = (
            source.rollout_devices != target.rollout_devices or force_rollout
        )
        if not actor_changed and not rollout_changed:
            return ReconfigurationPlan(source, target, actor_running, ())
        steps = []
        if rollout_changed:
            steps.append("drain_model_calls")
        steps.append("close_admission")
        if actor_changed and actor_running:
            steps.append("release_actor")
        elif rollout_changed and actor_running:
            steps.append("disconnect_weights")
        if rollout_changed:
            steps.append("stop_rollout")
        steps.append("set_layout")
        if rollout_changed:
            steps.append("rebuild_rollout")
        steps.append("rebind_runtime")
        if actor_changed or not actor_running:
            steps.append("restore_actor")
        steps.append("connect_weights")
        if not rollout_changed:
            steps.append("pause_rollout")
        steps.append("publish_weights")
        return ReconfigurationPlan(source, target, actor_running, tuple(steps))
