# SPDX-License-Identifier: Apache-2.0

"""Ordinary policy assembly and decisions for native fixed-layout execution.

The Monitor supplies shared execution facts and history. The default adapter
selects the Basic timeout baseline; revision hooks remain optional.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

from areal.utils.atomic import atomic_json

from ..contracts import ResourceDecision, TrainingConfig
from ..control.policy_hooks import PolicyHooks
from ..identity import measurement_identity
from ..monitor.collector import SCOPE
from ..state.batch_queue import QueuedBatch
from . import build_policy
from .basic.optimizer import BasicOptimizer
from .basic.policy import BasicPolicy, IdleTimeoutPolicy
from .basic.profile import fixed_layout_simulator, workflow_from_profile


class PolicyAdapter:
    """Bind timeout or measured Basic decisions to one native coordinator.

    A measured profile must match this execution's identity. AutoPilot extends
    this adapter for revision without introducing another decision entry point.
    """

    supports_policy_activation = False
    hooks: PolicyHooks | None = None

    def __init__(self, *, profile=None, settings=None):
        self._provided_profile = profile is not None
        self.profile = None if profile is None else json.loads(json.dumps(profile))
        self.settings = dataclasses.asdict(
            BasicOptimizer(**dict(settings or {"horizon": 1}))
        )
        if self.settings.get("horizon", 1) != 1:
            raise ValueError("Native fixed-layout planning requires horizon=1")
        self.owner: Any = None
        self.policy = None
        self.execution_id: str | None = None
        self.closed = False

    @property
    def supports_actor_dp_resize(self) -> bool:
        return type(self) is PolicyAdapter and not self._provided_profile

    def rebind_actor_layout(self, owner: Any) -> None:
        """Start fresh measurements after an explicit default-policy DP change."""
        if not self.supports_actor_dp_resize or self.closed:
            raise ValueError(
                "Measured/revision adapters require separately accepted target profiles"
            )
        self.owner = self.profile = self.execution_id = self.policy = None
        self.settings["candidate_configs"] = None
        self.bind(owner)

    def bind(self, owner: Any) -> None:
        if self.owner is not None or self.closed:
            raise RuntimeError("A policy adapter must be unbound and open")
        self.owner = owner
        identity = self.execution_id = measurement_identity(owner)
        world_size = getattr(owner, "actor_world_size", len(owner.gpu_uuids))
        current = (TrainingConfig(world_size),)
        optimizer = BasicOptimizer(**self.settings)
        if (
            optimizer.candidate_configs is not None
            and optimizer.candidate_configs != current
        ):
            raise ValueError(
                "Native planning candidates must match the fixed Actor layout"
            )
        self.settings = json.loads(
            json.dumps(
                dataclasses.asdict(
                    dataclasses.replace(optimizer, candidate_configs=current)
                )
            )
        )
        if self.profile is None:
            self.profile = {
                "execution_scope": SCOPE,
                "native_execution_id": identity,
                "native_actor_world_size": world_size,
                "compatibility": owner.compatibility,
                "inference_gpus": len(
                    getattr(owner, "rollout_devices", owner.options.rollout_devices)
                ),
                "measurements": [],
            }
            self.policy = build_policy(
                owner.options.policy, idle_timeout=owner.options.idle_timeout
            )
        else:
            if (
                self.profile.get("execution_scope") != SCOPE
                or self.profile.get("native_execution_id") != identity
                or self.profile.get("native_actor_world_size") != world_size
                or self.profile.get("compatibility") != owner.compatibility
                or self.profile.get("inference_gpus")
                != len(getattr(owner, "rollout_devices", owner.options.rollout_devices))
            ):
                raise ValueError(
                    "Measured profile differs from the native execution identity"
                )
            self.policy = BasicPolicy(
                fixed_layout_simulator(
                    workflow_from_profile(self.profile), self.profile
                ),
                optimizer=BasicOptimizer(**self.settings),
                fallback=IdleTimeoutPolicy(owner.options.idle_timeout),
            )
        owner.monitor.bind(identity)
        try:
            atomic_json(owner.root / "native-profile.json", self.profile)
        except OSError as error:
            owner.monitor.mark_unavailable(error)
        if hasattr(self.policy, "history"):
            self.policy.history = self.history

    @property
    def applied_version(self) -> int:
        return self.owner.applied_version

    @property
    def published_version(self) -> int:
        return self.owner.published_version

    @property
    def checkpoint_version(self) -> int:
        return self.owner.checkpoint_version

    @property
    def history(self):
        return self.owner.monitor.history

    def observe(self, batch: QueuedBatch | None = None, *, now: float | None = None):
        observation = self.owner.monitor.observe(batch, now=now)
        return dataclasses.replace(
            observation, gate_available=not self.closed and observation.gate_available
        )

    def policy_metadata(self) -> dict:
        return {
            "policy_name": type(self.policy).__name__,
            "decision_mode": "policy"
            if self.owner.monitor.metrics_available
            else "timeout_fallback",
        }

    def _action(self, decision, observation):
        if isinstance(decision, ResourceDecision):
            if decision.action == "reconfigure" or (
                decision.config is not None
                and decision.config != observation.training_config
            ):
                raise ValueError(
                    "Native execution currently has a fixed Actor/Rollout layout"
                )
            if decision.action == "launch" and not observation.batch_ready:
                raise ValueError("Native Actor cannot prewarm without a durable batch")
            if decision.train and not observation.batch_ready:
                raise ValueError("Training requires a durable eligible batch")
            return (
                "train"
                if decision.train
                else "release"
                if decision.action == "release"
                else "keep"
            )
        if decision not in {"train", "keep", "release"}:
            raise ValueError("Invalid native resource decision")
        return decision

    def _require_decision_boundary(self) -> None:
        if self.owner is None or self.closed:
            raise RuntimeError("Policy decisions require a bound, open adapter")

    def decide(self, batch: QueuedBatch | None, *, now: float | None = None):
        self._require_decision_boundary()
        observation = self.observe(batch, now=now)
        if hasattr(self.policy, "history"):
            self.policy.history = self.history
        if self.owner.monitor.metrics_available:
            decision = self.policy.decide(observation)
        else:
            decision = IdleTimeoutPolicy(self.owner.options.idle_timeout).decide(
                observation
            )
        action = self._action(decision, observation)
        self.owner.monitor.record_decision(
            batch, observation, decision, action, policy=self.policy_metadata()
        )
        return action

    def close(self) -> None:
        self.closed = True
