# SPDX-License-Identifier: Apache-2.0

"""Hot preparation and Cold release-before-deploy transitions for experiments."""

from __future__ import annotations

from dataclasses import asdict

from areal.elastic.contracts import ActorMemoryRejected, OnlineLayout
from areal.elastic.layout import ExecutionLayout, ReconfigurationPlanner
from areal.elastic.simulation_primitives import GPUAllocator

from .state import _Deployment


class SimulationReconfiguration:
    _host_steps = {"close_admission", "set_layout", "rebind_runtime", "pause_rollout"}

    def __init__(self, simulation):
        self.sim = simulation

    @property
    def cold(self):
        return self.sim.handoff.mode == "cold"

    @property
    def layout(self):
        return self.sim._logical if self.cold else self._hot_layout()

    def request_handoff(self, target):
        return (
            self._cold_request_handoff(target)
            if self.cold
            else self._hot_request_handoff(target)
        )

    def switching_gpu_seconds(self):
        return 0.0 if self.cold else self._hot_switching_gpu_seconds()

    def before_work(self):
        if self.cold:
            self._cold_before_work()
        else:
            self.maybe_actor_handoff()
            self.maybe_rollout_cutover()

    def after_work(self):
        if self.cold:
            self._cold_after_work()
        else:
            self.retire_rollouts()
            self.maybe_finish_handoff()

    def deployment_released(self, identity, deployment):
        if self.cold:
            self._cold_deployment_released(identity, deployment)
        else:
            self._hot_deployment_released(identity, deployment)

    def handle_event(self, kind, data):
        if self.cold and kind == "cold_phase_complete":
            self._complete_step(data)
        elif self.cold and kind == "cold_continue":
            pass
        else:
            self._hot_handle_event(kind, data)

    def publication_complete(self):
        if self.cold and self.sim._native_step == "publish_weights":
            self._complete_step("publish_weights")

    def admit(self):
        if not self.cold or self.sim._admission_open:
            self.sim.execution.admit()

    def service_base(self, local_services=None):
        return self.sim.execution.service(local_services)

    def service(self, local_services=None):
        if self.cold:
            return self._cold_service(local_services)
        return self.service_base(local_services)

    def start_actor(self):
        if not self.cold or (
            self.sim.pending_handoff is None and not self.sim.decision_pending
        ):
            self.sim.execution.start_actor()

    def reject_unrepairable_forecast_wait(self):
        if self.cold:
            return self._cold_reject_unrepairable_forecast_wait()
        return self._hot_reject_unrepairable_forecast_wait()

    def initialize(self, initial):
        sim = self.sim
        handoff = sim.handoff
        if handoff.mode != "cold":
            return
        sim._logical = initial
        sim._native_step = None
        sim._model_gate = "open"
        sim._drain_requests = set()
        sim._admission_open = True
        sim._checkpoint_version = handoff.native_checkpoint_version
        sim.execution_layout = handoff.native_source or self._physical_source()
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
            sim.pool.leases = {}
            sim.pool.peak_gpus = 0
            sim.pool.acquire_exact(
                "0", "actor", self._positions(source.actor_devices[: source.world_size])
            )
            sim.pool.acquire_exact(
                "1", "rollout", self._positions(source.rollout_devices)
            )
            sim.events[0]["leases"] = dict(sim.pool.leases)

    def _hot_deployment_released(self, identity, deployment):
        sim = self.sim
        if sim.pending_handoff is not None and str(identity) in sim._handoff_old_costs:
            sim.emit(
                "role_handoff_complete",
                role=deployment.role,
                source=identity,
                target=sim.actor_id if deployment.role == "actor" else sim.rollout_id,
            )
            if sim.reconfigurator is not None:
                sim.reconfigurator.role_switched(deployment.role, sim.now)

    def _hot_reject_unrepairable_forecast_wait(self):
        sim = self.sim
        target = sim._forecast_target
        if target is None:
            return  # A real reconfigurator or externally scheduled action may repair it.
        changes_actor = (target.actor_dp, target.actor_tp) != (
            sim.layout.actor_dp,
            sim.layout.actor_tp,
        )
        pending = sim.pending_handoff
        repair_planned = sim.decision_pending or (
            pending is not None
            and pending["new_actor"] is not None
            and pending["new_actor"] != sim.actor_id
        )
        if not changes_actor or not repair_planned:
            raise ActorMemoryRejected(
                sim.actor_wait.reason
                + "; no planned Actor migration can repair this batch"
            )

    def _hot_request_handoff(self, target: OnlineLayout):
        sim = self.sim
        if not sim.reconfiguration_allowed():
            return False
        if target.actor_gpus + target.rollout_gpus > 16:
            raise ValueError(
                "Target Actor and Rollout share a sixteen-GPU steady budget"
            )
        if sim.pending_handoff is not None or target == sim.layout:
            return False
        if sim.config.batch_size % target.actor_dp:
            raise ValueError("Target Actor DP must divide the configured episode batch")
        current = sim.layout
        if (
            sim.reconfigurator is not None
            and sim.reconfigurator.config.independent_role_history
        ):
            allowed = sim.reconfigurator.allowed_roles(sim.now)
            if (
                not allowed["actor"]
                and (target.actor_dp, target.actor_tp)
                != (current.actor_dp, current.actor_tp)
            ) or (
                not allowed["rollout"]
                and (target.rollout_replicas, target.rollout_tp)
                != (current.rollout_replicas, current.rollout_tp)
            ):
                return False
        if sim.actor_id not in sim.deployments:
            return False
        roles = []
        if (current.actor_dp, current.actor_tp) != (target.actor_dp, target.actor_tp):
            roles.append(("actor", target.actor_gpus))
        if (current.rollout_replicas, current.rollout_tp) != (
            target.rollout_replicas,
            target.rollout_tp,
        ):
            roles.append(("rollout", target.rollout_gpus))
        acquired = {}
        previous_peak = sim.pool.peak_gpus
        for role, count in roles:
            identity = sim.next_deployment
            sim.next_deployment += 1
            tp = target.actor_tp if role == "actor" else target.rollout_tp
            if sim.pool.acquire(str(identity), role, count, tp=tp) is None:
                for previous in acquired.values():
                    sim.pool.release(str(previous))
                sim.pool.peak_gpus = previous_peak
                sim.emit(
                    "handoff_deferred",
                    target=asdict(target),
                    reason="Insufficient spare GPUs for node-local TP groups",
                )
                return False
            acquired[role] = identity
        sim.pending_handoff = dict(
            target=target,
            new_actor=acquired.get("actor"),
            new_rollout=acquired.get("rollout"),
            actor_ready=False,
            actor_syncing=False,
            rollout_fence=False,
            rollout_switched="rollout" not in acquired,
        )
        sim._handoff_old_costs = {
            str(
                sim.actor_id if role == "actor" else sim.rollout_id
            ): sim.pool.lease_gpu_seconds.get(
                str(sim.actor_id if role == "actor" else sim.rollout_id), 0.0
            )
            for role in acquired
        }
        for role, identity in acquired.items():
            sim.deployments[identity] = _Deployment(
                identity, role, target, "preparing", sim.provider_version
            )
            seconds = (
                sim.handoff.actor_startup_seconds
                if role == "actor"
                else sim.handoff.rollout_startup_seconds
            )
            sim.schedule_at(sim.now + seconds, "process_started", identity)
        sim.emit(
            "handoff_start",
            source=asdict(current),
            target=asdict(target),
            leases=dict(sim.pool.leases),
        )
        return True

    def _hot_switching_gpu_seconds(self):
        """Old changed-role leases beyond target residency; no double counting."""
        sim = self.sim
        return sum(
            sim.pool.lease_gpu_seconds.get(owner, 0.0) - before
            for owner, before in sim._handoff_old_costs.items()
        )

    def load_deployment(self, identity: int):
        sim = self.sim
        deployment = sim.deployments[identity]
        size = (
            sim.handoff.actor_state_bytes
            if deployment.role == "actor"
            else sim.handoff.weight_bytes
        )
        if deployment.role == "rollout":
            # One host-side model image per destination node; replica startup
            # within a node retains the existing shared-image assumption.
            size *= len({node for node, _ in sim.pool.leases[str(identity)][1]})
        start, end = sim.link.reserve(sim.now, byte_count=size)
        sim.emit(
            "deployment_load",
            deployment=identity,
            role=deployment.role,
            snapshot_version=sim.provider_version,
            transfer_start=start,
            ends_at=end,
            bytes=size,
        )
        sim.schedule_at(end, "deployment_loaded", (identity, sim.provider_version))

    def add_restoring_actor(self, identity):
        sim = self.sim
        sim.deployments[identity] = _Deployment(
            identity, "actor", sim.layout, "preparing", sim.provider_version
        )

    def prepared(self, identity: int, version: int):
        sim = self.sim
        deployment = sim.deployments[identity]
        deployment.version, deployment.status = version, "prepared"
        if deployment.role == "actor":
            if sim._restoring_actor == identity:
                sim.actor_id = identity
                deployment.status = "active"
                sim._restoring_actor = None
            else:
                sim.pending_handoff["actor_ready"] = True
        sim.emit(
            "deployment_prepared",
            deployment=identity,
            role=deployment.role,
            version=version,
        )

    def maybe_rollout_cutover(self):
        sim = self.sim
        pending = sim.pending_handoff
        if not pending or pending["new_rollout"] is None or pending["rollout_switched"]:
            return
        new = sim.deployments[pending["new_rollout"]]
        if new.status != "prepared" or sim.publication != "idle":
            return
        if new.version != sim.provider_version:
            # Finish the current update, but fence further updates while final
            # synchronization catches up with its last published version.
            pending["rollout_fence"] = True
            new.status = "syncing"
            new.transfers += 1
            nodes = len({node for node, _ in sim.pool.leases[str(new.identity)][1]})
            byte_count = sim.handoff.weight_bytes * nodes
            start, end = sim.link.reserve(
                sim.now,
                byte_count=byte_count,
                setup_seconds=sim.handoff.rollout_sync_setup_seconds,
            )
            sim.emit(
                "rollout_catchup_start",
                deployment=new.identity,
                version=sim.provider_version,
                transfer_start=start,
                ends_at=end,
                bytes=byte_count,
                nodes=nodes,
            )
            sim.schedule_at(
                end, "rollout_synchronized", (new.identity, sim.provider_version)
            )
            return
        old = sim.deployments[sim.rollout_id]
        old.status, new.status = "draining", "active"
        sim.rollout_id = new.identity
        pending["rollout_switched"] = True
        pending["rollout_fence"] = False
        sim.emit(
            "rollout_cutover",
            source=old.identity,
            target=new.identity,
            version=new.version,
        )

    def rollout_synchronized(self, identity: int, version: int):
        sim = self.sim
        deployment = sim.deployments[identity]
        deployment.version, deployment.status = version, "prepared"
        deployment.transfers -= 1
        sim.emit("rollout_catchup_complete", deployment=identity, version=version)

    def maybe_actor_handoff(self):
        sim = self.sim
        pending = sim.pending_handoff
        if (
            not pending
            or not pending["actor_ready"]
            or pending["actor_syncing"]
            or pending["new_actor"] == sim.actor_id
            or sim.actor_batch
            or sim.publication != "idle"
        ):
            return
        pending["actor_syncing"] = True
        start, end = sim.link.reserve(
            sim.now,
            byte_count=sim.handoff.actor_state_bytes,
            setup_seconds=sim.handoff.actor_sync_setup_seconds,
        )
        snapshot = (sim.provider_version, sim.consumed)
        sim.emit(
            "actor_handoff_start",
            source=sim.actor_id,
            target=pending["new_actor"],
            version=snapshot[0],
            cursor=snapshot[1],
            transfer_start=start,
            ends_at=end,
        )
        sim.schedule_at(end, "actor_synchronized", snapshot)

    def actor_synchronized(self, snapshot):
        sim = self.sim
        if snapshot != (sim.provider_version, sim.consumed) or sim.actor_batch:
            raise ValueError(
                "Training authority changed while its boundary snapshot was transferring"
            )
        pending = sim.pending_handoff
        old = sim.actor_id
        sim.actor_id = pending["new_actor"]
        sim.deployments[sim.actor_id].status = "active"
        sim.deployments[sim.actor_id].version = sim.provider_version
        pending["actor_ready"] = False
        sim.execution.release_deployment(old)
        sim.emit(
            "actor_cutover",
            source=old,
            target=sim.actor_id,
            version=sim.provider_version,
            cursor=sim.consumed,
        )

    def retire_rollouts(self):
        sim = self.sim
        for identity, deployment in tuple(sim.deployments.items()):
            if (
                deployment.role != "rollout"
                or deployment.status != "draining"
                or deployment.busy
                or deployment.transfers
            ):
                continue
            if not any(group.rows for group in deployment.replicas.values()):
                sim.execution.release_deployment(identity)

    def maybe_finish_handoff(self):
        sim = self.sim
        pending = sim.pending_handoff
        if not pending:
            return
        actor_done = (
            pending["new_actor"] is None or pending["new_actor"] == sim.actor_id
        )
        active = {sim.actor_id, sim.rollout_id}
        if sim._restoring_actor is not None:
            active.add(sim._restoring_actor)
        if (
            actor_done
            and pending["rollout_switched"]
            and set(sim.deployments) <= active
        ):
            sim.emit("handoff_complete", layout=asdict(sim.layout))
            sim.pending_handoff = None
            if sim.reconfigurator:
                sim.reconfigurator.switched(sim.now)

    def _hot_handle_event(self, kind, data):
        sim = self.sim
        if kind == "process_started":
            self.load_deployment(data)
        elif kind == "deployment_loaded":
            deployment, version = data
            warmup = (
                sim.handoff.actor_warmup_seconds
                if sim.deployments[deployment].role == "actor"
                else sim.handoff.rollout_warmup_seconds
            )
            sim.schedule_at(
                sim.now + warmup, "deployment_prepared", (deployment, version)
            )
        elif kind == "deployment_prepared":
            self.prepared(*data)
        elif kind == "rollout_synchronized":
            self.rollout_synchronized(*data)
        elif kind == "actor_synchronized":
            self.actor_synchronized(data)
        else:
            raise ValueError(f"Unknown online event {kind}")

    def _physical_source(self):
        sim = self.sim
        source = sim.pool.leases
        actor = source.get(str(sim.actor_id))
        if actor is None:
            probe = GPUAllocator(sim.handoff.node_gpus, leases=dict(source))
            devices = probe.acquire(
                "source-actor", "actor", sim.layout.actor_gpus, tp=sim.layout.actor_tp
            )
            if devices is None:
                raise ValueError(
                    "Stopped Actor recovery does not fit the declared pool"
                )
        else:
            devices = actor[1]
        return ExecutionLayout(
            sim.layout.actor_gpus,
            self._ids(devices),
            self._ids(source[str(sim.rollout_id)][1]),
        )

    def _ids(self, positions):
        sim = self.sim
        all_positions = tuple(
            (node, rank)
            for node, n in enumerate(sim.handoff.node_gpus)
            for rank in range(n)
        )
        identifiers = sim.handoff.native_devices or tuple(range(len(all_positions)))
        mapping = dict(zip(all_positions, identifiers, strict=True))
        return tuple(mapping[p] for p in positions)

    def _positions(self, identifiers):
        sim = self.sim
        all_positions = tuple(
            (node, rank)
            for node, n in enumerate(sim.handoff.node_gpus)
            for rank in range(n)
        )
        devices = sim.handoff.native_devices or tuple(range(len(all_positions)))
        mapping = dict(zip(devices, all_positions, strict=True))
        try:
            return tuple(mapping[d] for d in identifiers)
        except KeyError as error:
            raise ValueError(
                "Target GPU is outside the immutable reservation"
            ) from error

    def _target(self, layout):
        sim = self.sim
        if sim.handoff.native_source is not None:
            target = next(
                (
                    p.execution
                    for p in sim.handoff.native_placements
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
            sim.layout.actor_dp,
            sim.layout.actor_tp,
        )
        rollout_changed = (layout.rollout_replicas, layout.rollout_tp) != (
            sim.layout.rollout_replicas,
            sim.layout.rollout_tp,
        )
        probe = GPUAllocator(sim.handoff.node_gpus, leases=dict(sim.pool.leases))
        for identity, changed in (
            (sim.actor_id, actor_changed),
            (sim.rollout_id, rollout_changed),
        ):
            if changed and str(identity) in probe.leases:
                probe.release(str(identity))
        actor = (
            probe.acquire(
                "target-actor", "actor", layout.actor_gpus, tp=layout.actor_tp
            )
            if actor_changed or str(sim.actor_id) not in probe.leases
            else probe.leases[str(sim.actor_id)][1]
        )
        rollout = (
            probe.acquire(
                "target-rollout", "rollout", layout.rollout_gpus, tp=layout.rollout_tp
            )
            if rollout_changed
            else probe.leases[str(sim.rollout_id)][1]
        )
        if actor is None or rollout is None:
            raise ValueError("Cold target does not fit the declared TP-local pool")
        return ExecutionLayout(layout.actor_gpus, self._ids(actor), self._ids(rollout))

    def _seconds(self, step):
        sim = self.sim
        values = dict(sim.handoff.phase_seconds)
        if step in self._host_steps:
            return values.get(step, 0.0)
        if step not in values:
            raise ValueError(f"Cold transition timing unavailable for {step}")
        return values[step]

    def _cold_request_handoff(self, target):
        sim = self.sim
        if sim.pending_handoff is not None or target == sim.layout:
            return False
        if (
            target.actor_gpus + target.rollout_gpus > 16
            or sim.config.batch_size % target.actor_dp
        ):
            raise ValueError(
                "Cold target must fit the steady GPU budget and logical batch"
            )
        physical = self._target(target)
        plan = ReconfigurationPlanner().plan(
            sim.execution_layout,
            physical,
            actor_running=sim.actor_id in sim.deployments,
            force_actor=(target.actor_dp, target.actor_tp)
            != (sim.layout.actor_dp, sim.layout.actor_tp),
            force_rollout=(target.rollout_replicas, target.rollout_tp)
            != (sim.layout.rollout_replicas, sim.layout.rollout_tp),
        )
        steps = plan.steps
        for step in steps:
            if step not in {"drain_model_calls", "publish_weights"}:
                self._seconds(step)
        sim.pending_handoff = dict(
            target=target,
            plan=plan,
            steps=list(steps),
            new_actor=None,
            new_rollout=None,
            actor_ready=False,
            rollout_fence=True,
            started=False,
        )
        sim.emit(
            "cold_handoff_start",
            source=asdict(sim.layout),
            target=asdict(target),
            plan=asdict(plan),
        )
        return True

    def _cold_before_work(self):
        sim = self.sim
        if sim.pending_handoff is None or sim._native_step is not None:
            return
        if sim.actor_batch or sim.publication != "idle":
            return
        steps = sim.pending_handoff["steps"]
        if not sim.pending_handoff["started"]:
            sim.pending_handoff["started"] = True
            if sim._checkpoint_version != sim.training_version:
                self._seconds("checkpoint")
                steps.insert(0, "checkpoint")
        if not steps:
            sim._admission_open = True
            sim._model_gate = "open"
            sim.pending_handoff = None
            sim.emit("handoff_complete", layout=asdict(sim.layout), mode="cold")
            if sim.reconfigurator is not None:
                sim.reconfigurator.switched(sim.now)
            return
        sim._native_step = steps.pop(0)
        step = sim._native_step
        sim.emit("cold_phase_start", phase=step)
        if step == "drain_model_calls":
            sim._model_gate = "draining"
            sim._drain_requests = {
                k
                for k, r in sim._live_requests.items()
                if sim.episodes[r.view.episode_id].admitted
            }
        elif step == "publish_weights":
            sim._model_gate = "paused"
        else:
            if step == "close_admission":
                sim._admission_open = False
            if step == "pause_rollout":
                sim._model_gate = "paused"
            if step in {"restore_actor", "rebuild_rollout"}:
                self._allocate(step)
            sim.schedule_at(sim.now + self._seconds(step), "cold_phase_complete", step)

    def _cold_after_work(self):
        sim = self.sim
        if sim._native_step == "drain_model_calls" and not (
            sim._drain_requests & sim._live_requests.keys()
        ):
            sim._model_gate = "closed"
            self._complete_step("drain_model_calls")
        elif sim._native_step == "publish_weights" and sim.publication == "idle":
            if not any(d.busy for d in sim.deployments.values()):
                sim.execution.start_publication()

    def _allocate(self, step):
        sim = self.sim
        role = "actor" if step == "restore_actor" else "rollout"
        plan = sim.pending_handoff["plan"]
        devices = (
            plan.target.actor_devices[: plan.target.world_size]
            if role == "actor"
            else plan.target.rollout_devices
        )
        identity = sim.next_deployment
        sim.next_deployment += 1
        sim.pool.acquire_exact(str(identity), role, self._positions(devices))
        sim.deployments[identity] = _Deployment(
            identity,
            role,
            sim.pending_handoff["target"],
            "preparing",
            sim.provider_version,
        )
        sim.pending_handoff["new_" + role] = identity

    def _complete_step(self, step):
        sim = self.sim
        pending = sim.pending_handoff
        if step == "checkpoint":
            sim._checkpoint_version = sim.training_version
        elif step == "release_actor":
            sim.execution.release_deployment(sim.actor_id)
        elif step == "stop_rollout":
            sim.execution.release_deployment(sim.rollout_id)
        elif step == "set_layout":
            sim.execution_layout = pending["plan"].target
            sim._logical = pending["target"]
            for d in sim.deployments.values():
                d.layout = sim.layout
        elif step == "restore_actor":
            old = sim.actor_id
            sim.actor_id = pending["new_actor"]
            sim.deployments[sim.actor_id].status = "active"
            sim.emit(
                "actor_cutover",
                source=old,
                target=sim.actor_id,
                version=sim.provider_version,
                cursor=sim.consumed,
            )
        elif step == "rebuild_rollout":
            old = sim.rollout_id
            sim.rollout_id = pending["new_rollout"]
            sim.deployments[sim.rollout_id].status = "active"
            sim.emit(
                "rollout_cutover",
                source=old,
                target=sim.rollout_id,
                version=sim.provider_version,
            )
        sim.emit("cold_phase_complete", phase=step)
        sim._native_step = None
        # Host-only zero-duration steps still create events; never spin inside a tick.
        sim.schedule_at(sim.now, "cold_continue")

    def _cold_deployment_released(self, identity, deployment):
        sim = self.sim
        if sim.pending_handoff is not None and sim.reconfigurator is not None:
            sim.reconfigurator.role_switched(deployment.role, sim.now)

    def _cold_service(self, local_services=None):
        sim = self.sim
        if (
            sim._model_gate in {"closed", "paused"}
            or sim.rollout_id not in sim.deployments
        ):
            return
        if sim._model_gate != "draining":
            return self.service_base(local_services)
        hidden = {
            k: r for k, r in sim._unassigned.items() if k not in sim._drain_requests
        }
        sim._unassigned = {
            k: r for k, r in sim._unassigned.items() if k in sim._drain_requests
        }
        try:
            self.service_base(local_services)
        finally:
            sim._unassigned.update(hidden)

    def _cold_reject_unrepairable_forecast_wait(self):
        sim = self.sim
        if (
            sim.pending_handoff is not None
            and "restore_actor" in sim.pending_handoff["plan"].steps
        ):
            return
        self._hot_reject_unrepairable_forecast_wait()

    def _hot_layout(self):
        sim = self.sim
        if sim.rollout_id not in sim.deployments:
            return sim._last_layout
        actor = (
            sim.deployments[sim.actor_id].layout
            if sim.actor_id in sim.deployments
            else sim._last_layout
        )
        rollout = sim.deployments[sim.rollout_id].layout
        sim._last_layout = OnlineLayout(
            actor.actor_dp, actor.actor_tp, rollout.rollout_replicas, rollout.rollout_tp
        )
        return sim._last_layout
