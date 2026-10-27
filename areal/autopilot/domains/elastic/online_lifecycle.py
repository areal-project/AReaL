# SPDX-License-Identifier: Apache-2.0

"""Causal Actor residency decisions, serialized with online reconfiguration."""

from __future__ import annotations

import math

from areal.elastic.policy.basic.forecast import ActorLifecycleView


class ActorLifecycle:
    def __init__(self, simulation, view=None):
        self.sim = simulation
        self.state = view.state if view is not None else "running"
        self.first_check_at = (
            view.first_check_at
            if view is not None
            else simulation.now + simulation.config.actor_lifecycle_history_seconds
        )
        self.first_check_done = view.first_check_done if view is not None else False
        self.check_pending = view.check_pending if view is not None else False
        self.wake_at = view.wake_at if view is not None else None
        self.restore_id = None
        self.snapshot = view.snapshot if view is not None else None
        if not self.first_check_done:
            simulation.schedule_at(
                max(simulation.now, self.first_check_at),
                "actor_lifecycle_check",
                priority=8,
            )
        if self.state == "stopped" and self.wake_at is not None:
            simulation.schedule_at(
                max(simulation.now, self.wake_at), "actor_lifecycle_wake", self.wake_at
            )

    def observe(self):
        return ActorLifecycleView(
            self.state,
            self.first_check_at,
            self.first_check_done,
            self.check_pending,
            self.wake_at,
            self.snapshot,
        )

    def request_check(self):
        self.check_pending = True

    def next_batch(self):
        sim = self.sim
        return sim.arrival_order[sim.consumed : sim.consumed + sim.config.batch_size]

    def batch_ready(self):
        sim, batch = self.sim, self.next_batch()
        return len(batch) == sim.config.batch_size and all(
            sim.episodes[e].completed_at is not None for e in batch
        )

    def estimate(self):
        """Use only observed episode arrivals and completed end-to-end durations."""
        sim = self.sim
        window = sim.config.actor_lifecycle_history_seconds
        start = max(self.first_check_at - window, sim.now - window)
        recent = [
            e
            for e in sim.arrival_order
            if start <= sim._truth[e].available_at <= sim.now
        ]
        rate = len(recent) / max(sim.now - start, 1e-9)
        durations = [
            e.completed_at - sim._truth[e.episode_id].available_at
            for e in sim.episodes.values()
            if e.completed_at is not None and e.completed_at >= start
        ]
        latency = sum(durations) / len(durations) if durations else 0.0
        batch = self.next_batch()
        missing = sim.config.batch_size - len(batch)
        arrival_wait = missing / rate if rate else (math.inf if missing else 0.0)
        remaining = max(
            (
                max(0.001, latency - (sim.now - sim._truth[e].available_at))
                for e in batch
                if sim.episodes[e].completed_at is None
            ),
            default=0.0,
        )
        return max(arrival_wait + (latency if missing else 0.0), remaining), rate

    def restore_seconds(self):
        h = self.sim.handoff
        return (
            h.actor_startup_seconds
            + h.actor_state_bytes / (h.effective_bandwidth_GBps * 1e9)
            + h.actor_warmup_seconds
        )

    def tick(self):
        sim = self.sim
        if self.state == "stopped":
            if self.batch_ready():
                self.wake("complete_batch")
            elif not sim.input_closed:
                wait, _ = self.estimate()
                when = sim.now + max(
                    0.0,
                    wait
                    - self.restore_seconds()
                    - sim.config.actor_lifecycle_margin_seconds,
                )
                if math.isfinite(when) and (
                    self.wake_at is None or when < self.wake_at - 1e-9
                ):
                    self.wake_at = when
                    sim.schedule_at(when, "actor_lifecycle_wake", when)
            return
        if self.state != "running" or not self.check_pending:
            return
        if (
            sim.actor_batch
            or sim.publication != "idle"
            or sim.pending_handoff is not None
            or sim.decision_pending
            or (sim.control is not None and sim.control.pending is not None)
        ):
            return
        if sim.now < self.first_check_at:
            return
        self.first_check_done, self.check_pending = True, False
        if (
            self.batch_ready()
            or sim.actor_wait is not None
            or sim.admission_credits() < sim.config.batch_size
            or (sim.input_closed and len(self.next_batch()) < sim.config.batch_size)
        ):
            return
        wait, rate = self.estimate()
        h = sim.handoff
        release = (
            max(0.0, sim.link.available_at - sim.now)
            + h.actor_sync_setup_seconds
            + h.actor_state_bytes / (h.effective_bandwidth_GBps * 1e9)
        )
        threshold = (
            release + self.restore_seconds() + sim.config.actor_lifecycle_margin_seconds
        )
        sim.emit(
            "actor_residency_check",
            arrival_rate=rate,
            batch_wait_seconds=wait if math.isfinite(wait) else None,
            release_restore_seconds=threshold,
            release=wait > threshold,
        )
        if wait <= threshold:
            return
        self.state = "releasing"
        self.snapshot = (sim.training_version, sim.provider_version, sim.consumed)
        sim._last_layout = sim.layout
        deployment = sim.deployments[sim.actor_id]
        deployment.transfers += 1
        start, end = sim.link.reserve(
            sim.now,
            byte_count=h.actor_state_bytes,
            setup_seconds=h.actor_sync_setup_seconds,
        )
        sim.emit(
            "actor_suspend_start",
            deployment=sim.actor_id,
            snapshot=self.snapshot,
            bytes=h.actor_state_bytes,
            transfer_start=start,
            ends_at=end,
        )
        sim.schedule_at(end, "actor_lifecycle_saved")

    def saved(self):
        sim = self.sim
        if self.snapshot != (
            sim.training_version,
            sim.provider_version,
            sim.consumed,
        ):
            raise ValueError("Actor state changed while saving its boundary snapshot")
        sim.deployments[sim.actor_id].transfers -= 1
        sim.release_deployment(sim.actor_id)
        self.state = "stopped"
        sim.emit("actor_suspended", snapshot=self.snapshot)

    def wake(self, reason):
        if self.state != "stopped":
            return
        sim = self.sim
        identity = sim.next_deployment
        if (
            sim.pool.acquire(
                str(identity), "actor", sim.layout.actor_gpus, tp=sim.layout.actor_tp
            )
            is None
        ):
            sim.schedule_at(sim.now + 1.0, "actor_lifecycle_wake", self.wake_at)
            return
        self.state, self.restore_id, self.wake_at = "restoring", identity, None
        sim.next_deployment += 1
        sim.add_restoring_actor(identity)
        sim.emit(
            "actor_resume_start",
            deployment=identity,
            reason=reason,
            snapshot=self.snapshot,
            layout=sim.layout.key,
            leases=dict(sim.pool.leases),
        )
        sim.schedule_at(
            sim.now + sim.handoff.actor_startup_seconds, "process_started", identity
        )

    def prepared(self, identity):
        sim = self.sim
        if self.snapshot != (
            sim.training_version,
            sim.provider_version,
            sim.consumed,
        ):
            raise ValueError("Restored Actor snapshot disagrees with the live cursor")
        sim.actor_id = identity
        sim.deployments[identity].status = "active"
        self.state, self.restore_id = "running", None
        sim.emit("actor_resumed", deployment=identity, snapshot=self.snapshot)
