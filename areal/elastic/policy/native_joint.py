# SPDX-License-Identifier: Apache-2.0

"""Opt-in joint planning at empty, published native training boundaries."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from typing import Any

from areal.utils.atomic import atomic_json, json_digest

from ..control.boundary import ControlBusy
from ..control.config_application import ConfigRequest
from ..control.resource_reconfiguration import ResourceLayout
from .basic.joint import JointConfig, JointScenario
from .basic.policy import JointPolicy


@dataclass(frozen=True)
class JointPlacement:
    config: JointConfig
    layout: ResourceLayout

    def __post_init__(self):
        if (
            self.config.actor.profile_id
            or self.config.actor.world_size != self.layout.world_size
            or len(self.layout.actor_devices) != self.layout.world_size
            or self.config.rollout.backend != "sglang"
            or self.config.rollout.tp != 1
            or self.config.rollout.replicas != len(self.layout.rollout_devices)
        ):
            raise ValueError(
                "Native joint candidates require unnamed pure DP, exact Actor leases and SGLang TP1 replicas"
            )


class NativeJointPlanner:
    """Bridge an explicit forecast/model to the existing resource transaction.

    Lifecycle decisions stay with PolicyAdapter. Forecasts are caller supplied;
    this bridge neither predicts rewards nor turns future traces into live facts.
    poll() is called before claiming the next native batch, once per completed
    version when no training episode credits remain. step() also supports an
    explicit driver. Dedicated inference services must not admit unrelated eval.
    """

    def __init__(
        self,
        policy: JointPolicy,
        placements: tuple[JointPlacement, ...],
        *,
        model_id: str,
        forecast_source: str,
        forecast: Callable[[Any], JointScenario | None] | None = None,
    ):
        if not model_id or not forecast_source:
            raise ValueError("Explicit model and forecast provenance are required")
        self.model_id, self.forecast_source = model_id, forecast_source
        self.policy, self.placements, self.forecast = (
            policy,
            tuple(placements),
            forecast,
        )
        configs = tuple(p.config for p in self.placements)
        layouts = tuple(p.layout for p in self.placements)
        if (
            not configs
            or len(set(configs)) != len(configs)
            or len(set(layouts)) != len(layouts)
            or set(configs) != set(policy.optimizer.candidates)
        ):
            raise ValueError("Require one explicit placement per optimizer candidate")
        reference = replace(configs[0].rollout, replicas=1)
        if any(replace(c.rollout, replicas=1) != reference for c in configs):
            raise ValueError("Native joint planning may only change DP and replicas")
        self.binding_id = json_digest([asdict(p) for p in self.placements])
        self.last_version: int | None = None
        self.last_result: dict | None = None
        self._execution_id: str | None = None

    @staticmethod
    def _observation(owner) -> dict:
        state = owner.control_snapshot()
        application = (
            owner.resource_config_application or owner.actor_config_application
        )
        journal = application.state()
        if journal["pending"] is not None:
            raise ControlBusy("Recover the pending configuration before planning")
        return {
            "run": owner.identity,
            "queue_id": owner.inbox.queue.queue_id,
            "authority": "resource"
            if owner.resource_config_application is not None
            else "actor",
            "config_generation": journal["generation"],
            "weights_version": state.applied_version,
            "published_version": state.published_version,
            "boundary_id": state.boundary_id,
            "layout": asdict(owner.resource_layout),
            "cursor": owner.consumer.cursor.copy(),
            "data_progress": owner.data_progress.copy(),
        }

    def poll(self, owner) -> dict | None:
        """Optional native hook; never runs a forecast on partial/in-flight work."""
        with owner.control_operation():
            state = owner.control_snapshot()
            status = owner.inbox.status()
            if (
                self.forecast is None
                or owner.actor is None
                or not state.boundary_ready
                or state.applied_version < 1
                or state.applied_version == self.last_version
                or not status["admission_open"]
                or status["active_credits"]
            ):
                return None
            with self._quiet_boundary(owner) as quiet:
                if not quiet:
                    return None
                scenario = self.forecast(owner)
                if scenario is None:
                    return None
                result = self._step(owner, scenario)
                self.last_version = state.applied_version
                return result

    def step(self, owner, scenario: JointScenario) -> dict:
        """Plan and apply atomically relative to native update/lifecycle control."""
        with owner.control_operation():
            if (
                not owner.control_snapshot().boundary_ready
                or owner.applied_version < 1
                or not owner.inbox.status()["admission_open"]
            ):
                raise ControlBusy("Another operation owns episode admission")
            with self._quiet_boundary(owner) as quiet:
                if not quiet:
                    raise ControlBusy(
                        "A training episode arrived before model-call drain"
                    )
                return self._step(owner, scenario)

    @staticmethod
    @contextmanager
    def _quiet_boundary(owner):
        owner.validate_resource_layout(owner.resource_layout)
        # A model call may already be crossing the online_admit callback even
        # when credits were just zero. Drain proxies with episode admission OPEN
        # first, matching native resource transaction ordering. Late arrivals
        # acquire their normal lease; defer planning rather than return a 503.
        owner.rollout.drain_model_calls()
        try:
            quiet = owner.inbox.status()["active_credits"] == 0
            if quiet:
                owner.inbox.set_admission_open(False)
            yield quiet
        finally:
            applications = (
                owner.actor_config_application,
                owner.resource_config_application,
            )
            if owner.control_snapshot().boundary_ready and all(
                app is None or app.state()["pending"] is None for app in applications
            ):
                owner.inbox.set_admission_open(True)
                owner.rollout.open_model_calls()

    def _step(self, owner, scenario: JointScenario) -> dict:
        state = owner.control_snapshot()
        if (
            not state.boundary_ready
            or owner.actor is None
            or state.applied_version < 1
            or owner.inbox.status()["active_credits"]
        ):
            raise ControlBusy(
                "Joint planning requires a resident Actor and a published, empty boundary"
            )
        execution_id = json_digest(
            {"run": owner.identity, "queue": owner.inbox.queue.queue_id}
        )
        if self._execution_id not in (None, execution_id):
            raise ValueError("A native joint planner is bound to one training run")
        self._execution_id = execution_id
        current = next(
            (p.config for p in self.placements if p.layout == owner.resource_layout),
            None,
        )
        if (
            current is None
            or scenario.current != current
            or scenario.ready_batches != 0
            or scenario.provider_version != state.applied_version
            or scenario.accepted_batches != state.applied_version
            or scenario.staleness != owner.inbox.max_staleness
            or scenario.samples_per_group != 1
            or any(b.actor.samples != owner.inbox.batch_size for b in scenario.batches)
        ):
            raise ValueError("Forecast does not match the native source boundary/batch")
        # This check also covers the unchanged candidate; unsupported execution
        # must never be made eligible by an optimistic cost prediction.
        eligible, excluded = [], []
        for placement in self.placements:
            try:
                owner.validate_resource_layout(placement.layout)
            except ValueError as error:
                excluded.append({"placement": asdict(placement), "reason": str(error)})
            else:
                eligible.append(placement.config)
        if current not in eligible:
            raise ValueError(
                "The current native layout is outside planner capabilities"
            )
        observation = self._observation(owner)
        search = JointPolicy(
            self.policy.simulator,
            replace(self.policy.optimizer, candidates=tuple(eligible)),
        ).plan(scenario)
        if self._observation(owner) != observation:
            raise ControlBusy("Native source boundary changed during joint search")
        self.policy.last_search = search
        inputs = {
            "observation": observation,
            "binding_id": self.binding_id,
            "placements": [asdict(p) for p in self.placements],
            "model_id": self.model_id,
            "forecast_source": self.forecast_source,
            "scenario": asdict(scenario),
            "search_settings": asdict(self.policy.optimizer),
        }
        decision_id = json_digest(inputs)
        request_id = f"joint-{decision_id[:48]}"
        result = {
            "format_version": "native-joint-v1",
            "decision_id": decision_id,
            **inputs,
            "search": asdict(search),
            "excluded": excluded,
            "status": "planned",
            "receipt": None,
        }
        path = owner.root / "joint-planner" / f"{request_id}.json"
        path.parent.mkdir(exist_ok=True)
        atomic_json(path, result)
        if search.config is None:
            result["status"] = "no_feasible_plan"
        else:
            target = next(
                p.layout for p in self.placements if p.config == search.config
            )
            if target == owner.resource_layout:
                result["status"] = "unchanged"
            else:
                # Validate request identity/provenance before entering native GPU
                # operations. The runtime repeats CAS against its durable journal.
                request = ConfigRequest(
                    request_id,
                    target,
                    observation["config_generation"],
                    observation["weights_version"],
                    f"native-joint-{self.binding_id}",
                )
                result["receipt"] = owner.reconfigure_resources(
                    target=target,
                    request_id=request.request_id,
                    expected_generation=request.expected_generation,
                    expected_weights_version=request.expected_weights_version,
                    policy_bundle_id=request.policy_bundle_id,
                )
                result["status"] = result["receipt"]["status"]
        actual = self._observation(owner)
        for key in (
            "run",
            "queue_id",
            "weights_version",
            "published_version",
            "cursor",
            "data_progress",
        ):
            if actual[key] != observation[key]:
                raise RuntimeError(
                    "Joint reconfiguration changed native training progress"
                )
        result["actual"] = actual
        atomic_json(path, result)
        self.last_result = result
        return result
