# SPDX-License-Identifier: Apache-2.0

"""Release-before-deploy execution over the same native layout planner."""

from __future__ import annotations

from dataclasses import asdict

from areal.elastic.layout import ExecutionLayout, ReconfigurationPlanner

from .online_simulation import OnlineSimulation
from .online_transitions import GPUAllocator
from .simulation_core import _Deployment


class ColdSimulation(OnlineSimulation):
    """Execution mechanism only; trigger/search/selection remain version-owned.

    GPU-bearing native phases require explicit durations. Pure host bookkeeping
    defaults to zero in this conditional model; provenance remains in HandoffConfig.
    """

    _host_steps = {"close_admission", "set_layout", "rebind_runtime", "pause_rollout"}

    def __init__(self, episodes, initial, models, config, handoff, *args, **kwargs):
        self._logical = initial
        self._native_step = None
        self._model_gate = "open"
        self._drain_requests = set()
        self._admission_open = True
        self._checkpoint_version = handoff.native_checkpoint_version
        super().__init__(episodes, initial, models, config, handoff, *args, **kwargs)
        self.execution_layout = handoff.native_source or self._physical_source()
        if handoff.native_source is not None:
            source = handoff.native_source
            if (
                initial.actor_tp != 1
                or initial.rollout_tp != 1
                or initial.actor_dp != source.world_size
                or initial.rollout_replicas != len(source.rollout_devices)
            ):
                raise ValueError(
                    "Native initial state disagrees with the physical source layout"
                )
            self.pool.leases = {}
            self.pool.peak_gpus = 0
            self.pool.acquire_exact(
                "0", "actor", self._positions(source.actor_devices[: source.world_size])
            )
            self.pool.acquire_exact(
                "1", "rollout", self._positions(source.rollout_devices)
            )
            self.events[0]["leases"] = dict(self.pool.leases)

    @property
    def layout(self):
        return self._logical

    def _physical_source(self):
        source = self.pool.leases
        actor = source.get(str(self.actor_id))
        if actor is None:
            probe = GPUAllocator(self.handoff.node_gpus, leases=dict(source))
            devices = probe.acquire(
                "source-actor", "actor", self.layout.actor_gpus, tp=self.layout.actor_tp
            )
            if devices is None:
                raise ValueError(
                    "Stopped Actor recovery does not fit the declared pool"
                )
        else:
            devices = actor[1]
        return ExecutionLayout(
            self.layout.actor_gpus,
            self._ids(devices),
            self._ids(source[str(self.rollout_id)][1]),
        )

    def _ids(self, positions):
        all_positions = tuple(
            (node, rank)
            for node, n in enumerate(self.handoff.node_gpus)
            for rank in range(n)
        )
        identifiers = self.handoff.native_devices or tuple(range(len(all_positions)))
        mapping = dict(zip(all_positions, identifiers, strict=True))
        return tuple(mapping[p] for p in positions)

    def _positions(self, identifiers):
        all_positions = tuple(
            (node, rank)
            for node, n in enumerate(self.handoff.node_gpus)
            for rank in range(n)
        )
        devices = self.handoff.native_devices or tuple(range(len(all_positions)))
        mapping = dict(zip(devices, all_positions, strict=True))
        try:
            return tuple(mapping[d] for d in identifiers)
        except KeyError as error:
            raise ValueError(
                "Target GPU is outside the immutable reservation"
            ) from error

    @classmethod
    def from_forecast(cls, view, forecast, *args, **kwargs):
        sim = cls.restore(view, forecast, *args, **kwargs)
        sim.execution_layout = sim.handoff.native_source or sim._physical_source()
        return sim

    def _target(self, layout):
        if self.handoff.native_source is not None:
            target = next(
                (
                    p.execution
                    for p in self.handoff.native_placements
                    if p.layout == layout
                ),
                None,
            )
            if target is None:
                raise ValueError(
                    "Native prediction has no physical mapping for this candidate"
                )
            return target
        actor_changed = (layout.actor_dp, layout.actor_tp) != (
            self.layout.actor_dp,
            self.layout.actor_tp,
        )
        rollout_changed = (layout.rollout_replicas, layout.rollout_tp) != (
            self.layout.rollout_replicas,
            self.layout.rollout_tp,
        )
        probe = GPUAllocator(self.handoff.node_gpus, leases=dict(self.pool.leases))
        for identity, changed in (
            (self.actor_id, actor_changed),
            (self.rollout_id, rollout_changed),
        ):
            if changed and str(identity) in probe.leases:
                probe.release(str(identity))
        actor = (
            probe.acquire(
                "target-actor", "actor", layout.actor_gpus, tp=layout.actor_tp
            )
            if actor_changed or str(self.actor_id) not in probe.leases
            else probe.leases[str(self.actor_id)][1]
        )
        rollout = (
            probe.acquire(
                "target-rollout", "rollout", layout.rollout_gpus, tp=layout.rollout_tp
            )
            if rollout_changed
            else probe.leases[str(self.rollout_id)][1]
        )
        if actor is None or rollout is None:
            raise ValueError("Cold target does not fit the declared TP-local pool")
        return ExecutionLayout(layout.actor_gpus, self._ids(actor), self._ids(rollout))

    def _seconds(self, step):
        values = dict(self.handoff.phase_seconds)
        if step in self._host_steps:
            return values.get(step, 0.0)
        if step not in values:
            raise ValueError(f"Cold transition timing unavailable for {step}")
        return values[step]

    def request_handoff(self, target):
        if self.pending_handoff is not None or target == self.layout:
            return False
        if (
            target.actor_gpus + target.rollout_gpus > 16
            or self.config.batch_size % target.actor_dp
        ):
            raise ValueError(
                "Cold target must fit the steady GPU budget and logical batch"
            )
        physical = self._target(target)
        plan = ReconfigurationPlanner().plan(
            self.execution_layout,
            physical,
            actor_running=self.actor_id in self.deployments,
            force_actor=(target.actor_dp, target.actor_tp)
            != (self.layout.actor_dp, self.layout.actor_tp),
            force_rollout=(target.rollout_replicas, target.rollout_tp)
            != (self.layout.rollout_replicas, self.layout.rollout_tp),
        )
        steps = plan.steps
        for step in steps:
            if step not in {"drain_model_calls", "publish_weights"}:
                self._seconds(step)
        self.pending_handoff = dict(
            target=target,
            plan=plan,
            steps=list(steps),
            new_actor=None,
            new_rollout=None,
            actor_ready=False,
            rollout_fence=True,
            started=False,
        )
        self.emit(
            "cold_handoff_start",
            source=asdict(self.layout),
            target=asdict(target),
            plan=asdict(plan),
        )
        return True

    def before_work(self):
        if self.pending_handoff is None or self._native_step is not None:
            return
        if self.actor_batch or self.publication != "idle":
            return
        steps = self.pending_handoff["steps"]
        if not self.pending_handoff["started"]:
            self.pending_handoff["started"] = True
            if self._checkpoint_version != self.training_version:
                self._seconds("checkpoint")
                steps.insert(0, "checkpoint")
        if not steps:
            self._admission_open = True
            self._model_gate = "open"
            self.pending_handoff = None
            self.emit("handoff_complete", layout=asdict(self.layout), mode="cold")
            if self.reconfigurator is not None:
                self.reconfigurator.switched(self.now)
            return
        self._native_step = steps.pop(0)
        step = self._native_step
        self.emit("cold_phase_start", phase=step)
        if step == "drain_model_calls":
            self._model_gate = "draining"
            self._drain_requests = {
                k
                for k, r in self._live_requests.items()
                if self.episodes[r.view.episode_id].admitted
            }
        elif step == "publish_weights":
            self._model_gate = "paused"
        else:
            if step == "close_admission":
                self._admission_open = False
            if step == "pause_rollout":
                self._model_gate = "paused"
            if step in {"restore_actor", "rebuild_rollout"}:
                self._allocate(step)
            self.schedule_at(
                self.now + self._seconds(step), "cold_phase_complete", step
            )

    def after_work(self):
        if self._native_step == "drain_model_calls" and not (
            self._drain_requests & self._live_requests.keys()
        ):
            self._model_gate = "closed"
            self._complete_step("drain_model_calls")
        elif self._native_step == "publish_weights" and self.publication == "idle":
            if not any(d.busy for d in self.deployments.values()):
                self.start_publication()

    def _allocate(self, step):
        role = "actor" if step == "restore_actor" else "rollout"
        plan = self.pending_handoff["plan"]
        devices = (
            plan.target.actor_devices[: plan.target.world_size]
            if role == "actor"
            else plan.target.rollout_devices
        )
        identity = self.next_deployment
        self.next_deployment += 1
        self.pool.acquire_exact(str(identity), role, self._positions(devices))
        self.deployments[identity] = _Deployment(
            identity,
            role,
            self.pending_handoff["target"],
            "preparing",
            self.provider_version,
        )
        self.pending_handoff["new_" + role] = identity

    def _complete_step(self, step):
        pending = self.pending_handoff
        if step == "checkpoint":
            self._checkpoint_version = self.training_version
        elif step == "release_actor":
            self.release_deployment(self.actor_id)
        elif step == "stop_rollout":
            self.release_deployment(self.rollout_id)
        elif step == "set_layout":
            self.execution_layout = pending["plan"].target
            self._logical = pending["target"]
            for d in self.deployments.values():
                d.layout = self.layout
        elif step == "restore_actor":
            old = self.actor_id
            self.actor_id = pending["new_actor"]
            self.deployments[self.actor_id].status = "active"
            self.emit(
                "actor_cutover",
                source=old,
                target=self.actor_id,
                version=self.provider_version,
                cursor=self.consumed,
            )
        elif step == "rebuild_rollout":
            old = self.rollout_id
            self.rollout_id = pending["new_rollout"]
            self.deployments[self.rollout_id].status = "active"
            self.emit(
                "rollout_cutover",
                source=old,
                target=self.rollout_id,
                version=self.provider_version,
            )
        self.emit("cold_phase_complete", phase=step)
        self._native_step = None
        # Host-only zero-duration steps still create events; never spin inside a tick.
        self.schedule_at(self.now, "cold_continue")

    def handle_transition_event(self, kind, data):
        if kind == "cold_phase_complete":
            self._complete_step(data)
        elif kind == "cold_continue":
            pass
        else:
            super().handle_transition_event(kind, data)

    def finish_publication(self, data):
        super().finish_publication(data)
        if self._native_step == "publish_weights":
            self._complete_step("publish_weights")

    def deployment_released(self, identity, deployment):
        if self.pending_handoff is not None and self.reconfigurator is not None:
            self.reconfigurator.role_switched(deployment.role, self.now)

    def switching_gpu_seconds(self):
        return 0.0  # No duplicate generation leases in release-before-deploy execution.

    def admit(self):
        if self._admission_open:
            super().admit()

    def service(self, local_services=None):
        if (
            self._model_gate in {"closed", "paused"}
            or self.rollout_id not in self.deployments
        ):
            return
        if self._model_gate != "draining":
            return super().service(local_services)
        hidden = {
            k: r for k, r in self._unassigned.items() if k not in self._drain_requests
        }
        self._unassigned = {
            k: r for k, r in self._unassigned.items() if k in self._drain_requests
        }
        try:
            super().service(local_services)
        finally:
            self._unassigned.update(hidden)

    def start_actor(self):
        if self.pending_handoff is None and not self.decision_pending:
            super().start_actor()

    def reject_unrepairable_forecast_wait(self):
        if (
            self.pending_handoff is not None
            and "restore_actor" in self.pending_handoff["plan"].steps
        ):
            return
        super().reject_unrepairable_forecast_wait()
