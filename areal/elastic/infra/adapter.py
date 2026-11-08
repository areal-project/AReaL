# SPDX-License-Identifier: Apache-2.0

"Adapt native training observations, placements and events to a Reconfigurator."

from __future__ import annotations

import heapq
import json
from dataclasses import asdict, replace
from typing import Any

from areal.elastic.contracts import ControlBusy, OnlineLayout
from areal.elastic.infra.monitor.identity import measurement_identity
from areal.elastic.reconfigurator import build_reconfigurator, component_module
from areal.utils.atomic import atomic_json


def validate_unbound_adapter(adapter: Any) -> None:
    """Reject shared/closed helpers before a new trainer takes cleanup ownership."""
    if adapter is not None and (
        getattr(adapter, "owner", None) is not None or getattr(adapter, "closed", False)
    ):
        raise ValueError(
            "Reconfigurator adapter is already bound or closed; create a new instance"
        )


class NativePlacements:
    def __init__(self, owner):
        self.owner = owner

    def current(self):
        parallel = self.owner.allocation.parallel
        from areal.api.alloc_mode import ModelAllocation

        rollout = ModelAllocation.from_str(self.owner.config.rollout.backend).parallel
        return OnlineLayout(
            parallel.dp_size,
            parallel.tp_size,
            len(self.owner.rollout_devices) // rollout.tp_size,
            rollout.tp_size,
        )

    def resolve(self, layout):
        if layout == self.current():
            return self.owner.execution_layout
        raise ValueError(
            "Native Megatron layout changes are not implemented; use CPU Simulation"
        )

    def candidates(self, proposed=None):
        # Keep native observations available while dynamic Megatron execution
        # is developed separately. No other-backend mapping is reused.
        return (self.current(),)

    def prediction_handoff(self, handoff):
        from areal.elastic.contracts import NativePlacement

        devices = tuple(self.owner.resource_reservation)
        if len(devices) > 8:
            raise ValueError(
                "Native Runtime prediction requires its explicit local GPU reservation"
            )
        if handoff.mode != "cold":
            raise ValueError("Native execution cannot apply hot handoff predictions")
        return replace(
            handoff,
            node_gpus=(len(devices),),
            native_source=self.owner.execution_layout,
            native_devices=devices,
            native_placements=tuple(
                NativePlacement(c, self.resolve(c))
                for c in self.candidates(self.owner.reconfigurator_adapter.proposed)
            ),
            native_checkpoint_version=self.owner.checkpoint_version,
        )


class NativeControlEnvironment:
    """Host-time event adapter; decision/search rules stay in the selected version."""

    def __init__(self, adapter):
        self.adapter = adapter
        self.owner = adapter.owner
        self.reconfigurator = adapter.reconfigurator
        self.cost_models = adapter.cost_models
        self.config = adapter.run_config
        self.slo = adapter.reconfigurator.slo
        self.candidate_executor = None
        self.decisions = []
        self.events = []
        self._events = []
        self._sequence = 0
        self._now = adapter.observations.now()
        self.pending_handoff = None
        self.decision_pending = False

    @property
    def handoff(self):
        return self.adapter.placements.prediction_handoff(self.adapter.handoff)

    @property
    def now(self):
        # Controller computations use their captured observation time. The host
        # event pump compares deadlines with the actual clock, avoiding double charging CPU time.
        return self._now

    @property
    def layout(self):
        return self.adapter.placements.current()

    @property
    def publication(self):
        pending = self.owner.trainer._control_phase == "publication"
        return "pending" if pending else "idle"

    def observe(self):
        view = self.adapter.observe()
        self._now = view.now
        return view

    def emit(self, kind, **data):
        self.events.append(dict(kind=kind, time=self.now, **data))
        self.events = self.events[-128:]

    def schedule_at(self, when, kind, data=None, priority=10):
        self._sequence += 1
        heapq.heappush(self._events, (when, priority, self._sequence, kind, data))

    def request_handoff(self, layout):
        owner = self.owner
        if (
            self.adapter.closed
            or not owner.control_snapshot().boundary_ready
            or owner.applied_version < 1
        ):
            return False
        target = self.adapter.placements.resolve(layout)
        source = owner.execution_layout
        if target == source:
            return False
        self.pending_handoff = target
        try:
            generation = owner.reconfiguration_application.state()["generation"]
            receipt = owner.trainer.reconfigure(
                target=target,
                request_id=f"online-{self.reconfigurator.implementation}-{self._sequence}-{owner.applied_version}-{generation}",
                expected_generation=generation,
                expected_weights_version=owner.applied_version,
                reconfigurator_bundle_id=self.adapter.bundle_id,
            )
            self._now = self.adapter.observations.now()
            if receipt["status"] != "applied":
                return False
            self.reconfigurator.switched(self.now)
            self.adapter.last_receipt = receipt
            return True
        except ControlBusy:
            return False
        finally:
            self.pending_handoff = None

    def poll(self, controller):
        if not self._events:
            self.schedule_at(self.adapter.observations.now(), "reconfigurator")
        clock = self.adapter.observations.now()
        while self._events and self._events[0][0] <= clock:
            _, _, _, kind, data = heapq.heappop(self._events)
            self._now = clock
            if kind == "reconfigurator":
                controller.poll()
            elif kind == "search_complete":
                controller.complete(data)
            elif (
                self.reconfigurator.implementation == "basic"
                and kind == "verification_complete"
            ):
                controller.verified(data)
            elif (
                self.reconfigurator.implementation == "basic"
                and kind == "decision_ready"
            ):
                controller.ready(data)
            else:
                raise ValueError(f"Unknown native Online event: {kind}")
            clock = self.adapter.observations.now()
        return controller.pending is not None

    def snapshot(self):
        return dict(
            implementation=self.reconfigurator.implementation,
            layout=asdict(self.layout),
            decisions=self.decisions[-32:],
            events=self.events,
            last_receipt=self.adapter.last_receipt,
        )


class ReconfiguratorAdapter:
    """Supply native facts/capabilities, drive the selected version, execute its intent.

    cost_models/run_config/handoff may be supplied for candidate prediction. Missing
    native token timing stays unknown; no second timeout or joint optimizer is assembled.
    """

    def __init__(
        self,
        *,
        implementation=None,
        settings=None,
        replay=None,
        forecast=None,
        slo=None,
        candidates=None,
        cost_models=None,
        run_config=None,
        handoff=None,
        observation_provider=None,
        profile=None,
    ):
        self.implementation = implementation
        self.settings = dict(settings or {})
        self.replay = replay
        self.forecast = forecast
        self.slo = slo
        self.proposed = None if candidates is None else tuple(candidates)
        self.cost_models = cost_models
        self.run_config = run_config
        self.handoff = handoff
        self.observation_provider = observation_provider
        self.profile = None if profile is None else json.loads(json.dumps(profile))
        self.owner: Any = None
        self.reconfigurator = None
        self.execution_id = None
        self.closed = False
        self.environment = self.controller = None
        self.last_receipt = None
        self.bundle_id = "builtin-online"
        self._polling = False

    def bind(self, owner):
        if self.owner is not None or self.closed:
            raise RuntimeError("A reconfigurator adapter must be unbound and open")
        self.owner = owner
        self.implementation = self.implementation or owner.options.reconfigurator
        self.observations = owner.monitor.online_observations
        self.placements = NativePlacements(owner)
        profile = self.profile or {}
        if "measurements" in profile:
            raise ValueError(
                "Legacy phase-only profiles must be replaced by an Online configuration"
            )
        self.settings = {
            "decision_delay_seconds": 1.0,
            **profile.get("reconfigurator", {}),
            **self.settings,
        }
        replay = self.replay if self.replay is not None else profile.get("replay")
        forecast = (
            self.forecast if self.forecast is not None else profile.get("forecast")
        )
        self.reconfigurator = build_reconfigurator(
            self.implementation,
            candidates=self.placements.candidates(self.proposed),
            settings=self.settings,
            replay=replay,
            forecast=forecast,
            slo=self.slo or profile.get("slo"),
        )
        self.execution_id = measurement_identity(owner)
        self.profile = {
            **profile,
            "implementation": self.implementation,
            "native_execution_id": self.execution_id,
            "native_actor_world_size": owner.actor_world_size,
            "compatibility": owner.compatibility,
        }
        owner.monitor.bind(self.execution_id)
        try:
            atomic_json(owner.root / "native-profile.json", self.profile)
        except OSError as error:
            owner.monitor.mark_unavailable(error)
        cost_model_config = profile.get("cost_models", profile.get("model"))
        if self.cost_models is None and cost_model_config:
            from pathlib import Path

            self.cost_models = component_module(
                self.implementation, "cost_models"
            ).CalibratedCostModels(
                Path(cost_model_config["bundle"]),
                Path(cost_model_config["mechanism_fits"]),
                Path(cost_model_config["publication_metadata"]),
                float(profile["handoff"]["effective_bandwidth_GBps"]),
            )
        if self.run_config is None and profile.get("run"):
            from areal.elastic.contracts import SimulationConfig

            values = dict(profile["run"])
            values["tool_capacities"] = tuple(
                tuple(v) for v in values["tool_capacities"]
            )
            self.run_config = SimulationConfig(**values)
        if self.handoff is None and profile.get("handoff"):
            from areal.elastic.contracts import HandoffConfig

            self.handoff = HandoffConfig.from_dict(
                {"mode": "cold", **profile["handoff"]}
            )
        if self.cost_models is not None:
            actor_template = getattr(self.cost_models, "actor_template", None)
            if (
                actor_template is not None
                and actor_template.backend != owner.allocation.backend
            ):
                raise ValueError(
                    "Calibrated Actor backend does not match the native execution backend"
                )
            if self.run_config is None or self.handoff is None:
                raise ValueError(
                    "Native candidate prediction requires run and handoff inputs"
                )
            if (
                self.run_config.batch_size != owner.config.train_dataset.batch_size
                or self.run_config.staleness
                != owner.config.rollout.max_head_offpolicyness
            ):
                raise ValueError(
                    "Native forecast must use the real batch and staleness"
                )
            if self.handoff.mode != "cold":
                raise ValueError(
                    "Native Runtime currently executes cold handoff; set handoff.mode='cold'"
                )
            self.environment = NativeControlEnvironment(self)
            self.controller = self.reconfigurator.bind(self.environment)

    def role_released(self, role):
        self.reconfigurator.role_switched(role, self.observations.now())

    def rebind_actor_layout(self, owner):
        if self.owner is not owner or self.closed:
            raise ValueError("Layout rebind requires the same active native owner")
        self.execution_id = measurement_identity(owner)
        self.reconfigurator.candidates = self.placements.candidates(self.proposed)
        self.profile.update(
            native_execution_id=self.execution_id,
            native_actor_world_size=owner.actor_world_size,
        )
        owner.monitor.bind(self.execution_id)
        self.reconfigurator.search_invalidated()
        atomic_json(owner.root / "native-profile.json", self.profile)

    @property
    def applied_version(self):
        return self.owner.applied_version

    @property
    def published_version(self):
        return self.owner.published_version

    @property
    def checkpoint_version(self):
        return self.owner.checkpoint_version

    @property
    def history(self):
        return self.owner.monitor.history

    def observe(self, batch=None, *, now=None):
        if self.observation_provider is not None:
            view = self.observation_provider(self.owner)
            return self.observations.validate(view)
        return self.observations.observe(now=now)

    def reconfigurator_metadata(self):
        return dict(
            reconfigurator_name=type(self.reconfigurator).__name__,
            reconfigurator_implementation=self.implementation,
            decision_mode="online",
            handoff_mode="cold",
            native_token_slo="unavailable"
            if self.observation_provider is None
            else "provider",
            reconfigurator_bundle_id=self.bundle_id,
        )

    def _require_decision_boundary(self):
        if self.owner is None or self.closed:
            raise RuntimeError("Reconfigurator decisions require a bound, open adapter")

    def poll(self):
        self._require_decision_boundary()
        if (
            self.controller is None
            or self._polling
            or not self.owner.monitor.metrics_available
        ):
            return False
        self._polling = True
        try:
            busy = self.environment.poll(self.controller)
            if self.environment.decisions:
                atomic_json(
                    self.owner.root / "online-decisions.json",
                    self.environment.snapshot(),
                )
            return busy
        finally:
            self._polling = False

    def close(self):
        self.closed = True
        if self.controller is not None:
            self.controller.cancel()
