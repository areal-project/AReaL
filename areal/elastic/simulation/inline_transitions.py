# SPDX-License-Identifier: Apache-2.0

"""Drain/save/release/load handoff under a hard, whole-run GPU cap."""

from dataclasses import asdict

from .online_transitions import GPUAllocator
from .simulation_core import _Deployment


class InlineTransition:
    def __init__(self, simulation, target):
        self.sim, self.target = simulation, target
        self.phase = "draining"
        self.started_at = simulation.now
        current = simulation.layout
        self.changed = tuple(
            role
            for role, changed in (
                (
                    "actor",
                    (current.actor_dp, current.actor_tp)
                    != (target.actor_dp, target.actor_tp),
                ),
                (
                    "rollout",
                    (current.rollout_replicas, current.rollout_tp)
                    != (target.rollout_replicas, target.rollout_tp),
                ),
            )
            if changed
        )
        self.old = {role: getattr(simulation, role + "_id") for role in self.changed}
        self.ready = set()
        self.new = {}
        self.snapshot = None

    @staticmethod
    def feasible(sim, target):
        if (
            target.actor_gpus + target.rollout_gpus > sim.gpu_budget
            or sim.config.batch_size % target.actor_dp
        ):
            return False
        preview = GPUAllocator(sim.handoff.node_gpus)
        current = sim.layout
        for role, unchanged in (
            (
                "actor",
                (current.actor_dp, current.actor_tp)
                == (target.actor_dp, target.actor_tp),
            ),
            (
                "rollout",
                (current.rollout_replicas, current.rollout_tp)
                == (target.rollout_replicas, target.rollout_tp),
            ),
        ):
            if unchanged:
                owner = str(getattr(sim, role + "_id"))
                preview.leases[owner] = sim.pool.leases[owner]
        for role, count, tp in (
            ("actor", target.actor_gpus, target.actor_tp),
            ("rollout", target.rollout_gpus, target.rollout_tp),
        ):
            if not any(r == role for r, _ in preview.leases.values()):
                if preview.acquire("target-" + role, role, count, tp=tp) is None:
                    return False
        return True

    def tick(self):
        sim = self.sim
        if (
            self.phase != "draining"
            or sim._active_episodes
            or sim.actor_batch
            or sim.publication != "idle"
            or any(d.busy or d.transfers for d in sim.deployments.values())
        ):
            return
        sim.flush_decode()
        self.snapshot = (
            sim.training_version,
            sim.provider_version,
            sim.consumed,
            sim.task_pool.cursor,
        )
        self.phase = "saving"
        end = sim.now
        if "actor" in self.changed:
            _, end = sim.link.reserve(
                sim.now,
                byte_count=sim.handoff.actor_state_bytes,
                setup_seconds=sim.handoff.actor_sync_setup_seconds,
            )
        sim.emit(
            "inline_save_start", snapshot=self.snapshot, roles=self.changed, ends_at=end
        )
        sim.schedule_at(end, "inline_saved")

    def saved(self):
        sim = self.sim
        self._check_snapshot()
        # Release all changing roles before acquiring any target role.
        for identity in self.old.values():
            sim.release_deployment(identity)
        self.phase = "loading"
        for role in self.changed:
            identity = sim.next_deployment
            sim.next_deployment += 1
            count = (
                self.target.actor_gpus if role == "actor" else self.target.rollout_gpus
            )
            tp = self.target.actor_tp if role == "actor" else self.target.rollout_tp
            if sim.pool.acquire(str(identity), role, count, tp=tp) is None:
                raise ValueError(
                    "Preflighted inline target no longer fits the hard GPU pool"
                )
            sim.deployments[identity] = _Deployment(
                identity, role, self.target, "loading", sim.provider_version
            )
            setattr(sim, role + "_id", identity)
            self.new[role] = identity
            sim.emit(
                "allocation",
                deployment=identity,
                role=role,
                leases=dict(sim.pool.leases),
            )
            startup = getattr(sim.handoff, role + "_startup_seconds")
            sim.schedule_at(sim.now + startup, "inline_process_started", role)

    def process_started(self, role):
        sim = self.sim
        byte_count = (
            sim.handoff.actor_state_bytes
            if role == "actor"
            else sim.handoff.weight_bytes
            * len({n for n, _ in sim.pool.leases[str(self.new[role])][1]})
        )
        setup = (
            sim.handoff.actor_sync_setup_seconds
            if role == "actor"
            else sim.handoff.rollout_sync_setup_seconds
        )
        start, end = sim.link.reserve(
            sim.now, byte_count=byte_count, setup_seconds=setup
        )
        ready_at = end + getattr(sim.handoff, role + "_warmup_seconds")
        sim.emit(
            "inline_load_start",
            role=role,
            deployment=self.new[role],
            bytes=byte_count,
            transfer_start=start,
            ends_at=end,
            ready_at=ready_at,
        )
        sim.schedule_at(ready_at, "inline_role_ready", role)

    def role_ready(self, role):
        self.ready.add(role)
        if self.ready != set(self.changed):
            return
        sim = self.sim
        self._check_snapshot()
        for identity in self.new.values():
            sim.deployments[identity].status = "active"
        sim._last_layout = self.target
        sim.emit(
            "handoff_complete",
            layout=asdict(sim.layout),
            snapshot=self.snapshot,
            seconds=sim.now - self.started_at,
        )
        sim.pending_handoff = None
        sim.transition = None

    def _check_snapshot(self):
        sim = self.sim
        if self.snapshot != (
            sim.training_version,
            sim.provider_version,
            sim.consumed,
            sim.task_pool.cursor,
        ):
            raise ValueError(
                "Inline training authority or task cursor changed during save/restore"
            )
