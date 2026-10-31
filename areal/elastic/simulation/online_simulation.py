# SPDX-License-Identifier: Apache-2.0

"""External-arrival simulation with SLO control and hot deployment handoff."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from areal.elastic.policy.basic.observations import (
    OnlineLayout,
    RequestView,
)
from areal.elastic.policy.basic.search import (
    SearchBound,
)
from areal.elastic.policy.reference.core import fingerprint

from .online_transitions import (
    ActorMemoryRejected,
)
from .simulation_core import SimulationConfig as OnlineRunConfig
from .simulation_core import SimulationCore, _Deployment

__all__ = ["OnlineRunConfig", "OnlineSimulation"]


@dataclass
class _ForecastSLOGuard:
    excluded: set[str]
    total: int
    failed: set[str] = field(default_factory=set)
    check_after: float = 0.0


@dataclass
class _CandidateGuard:
    bound: SearchBound
    excluded: frozenset[str]
    total: int
    completed_cost: float
    fraction_cap: float
    samples: int
    failed: set[str] = field(default_factory=set)
    check_after: float = 0.0
    upper: float = 1.0
    lower: float = 0.0
    fixed_running_cost: float | None = None


class OnlineSimulation(SimulationCore):
    """Externally arriving service requests with SLO-driven hot reconfiguration."""

    def __init__(
        self,
        *args,
        policy=None,
        measured_search_time=False,
        search_clock=None,
        **kwargs,
    ):
        super().__init__(*args, policy=policy, **kwargs)
        self._forecast_slo_guard = None
        self._candidate_guard = None
        if measured_search_time and policy is not None:
            from .online_control import OnlineControl

            self.control = OnlineControl(self, search_clock)
        if policy:
            self.schedule_at(
                self.now + policy.config.check_interval_seconds, "policy", priority=10
            )

    def before_work(self):
        self.maybe_actor_handoff()
        self.maybe_rollout_cutover()

    def after_work(self):
        self.retire_rollouts()
        self.maybe_finish_handoff()

    def check_prediction_bounds(self):
        self.check_forecast_slo()
        self.check_candidate_bound()

    @classmethod
    def from_forecast(cls, view, forecast, models, config, handoff, slo, **kwargs):
        return cls.restore(view, forecast, models, config, handoff, slo, **kwargs)

    def result(self):
        result = super().result()
        rows = tuple(r.view for r in self.requests.values())
        passed = [
            r
            for r in rows
            if r.completed_at is not None
            and r.first_token_at - r.ready_at <= self.slo.ttft_seconds + 1e-12
            and r.maximum_tbt_seconds <= self.slo.tbt_seconds + 1e-12
        ]
        result.update(
            jointly_passed_requests=len(passed),
            joint_request_fraction=len(passed) / len(rows) if rows else None,
        )
        return result

    def deployment_released(self, identity, deployment):
        if (
            self.pending_handoff is not None
            and str(identity) in self._handoff_old_costs
        ):
            self.emit(
                "role_handoff_complete",
                role=deployment.role,
                source=identity,
                target=self.actor_id if deployment.role == "actor" else self.rollout_id,
            )
            if self.policy is not None:
                self.policy.role_switched(deployment.role, self.now)

    def handle_transition_event(self, kind, data):
        if kind == "actor_lifecycle_check":
            self.lifecycle.request_check()
        elif kind == "actor_lifecycle_saved":
            self.lifecycle.saved()
        elif kind == "actor_lifecycle_wake":
            if self.lifecycle.wake_at == data:
                self.lifecycle.wake("estimated_batch_ready")
        elif kind == "policy":
            self.poll_policy()
        elif kind == "search_complete":
            self.control.complete(data)
        elif kind == "verification_complete":
            self.control.verified(data)
        elif kind == "decision_ready":
            self.control.ready(data)
        elif kind == "decision_apply":
            self.decision_pending = False
            self.request_handoff(data)
        elif kind == "process_started":
            self.load_deployment(data)
        elif kind == "deployment_loaded":
            deployment, version = data
            warmup = (
                self.handoff.actor_warmup_seconds
                if self.deployments[deployment].role == "actor"
                else self.handoff.rollout_warmup_seconds
            )
            self.schedule_at(
                self.now + warmup, "deployment_prepared", (deployment, version)
            )
        elif kind == "deployment_prepared":
            self.prepared(*data)
        elif kind == "rollout_synchronized":
            self.rollout_synchronized(*data)
        elif kind == "actor_synchronized":
            self.actor_synchronized(data)
        else:
            raise ValueError(f"Unknown online event {kind}")

    def check_forecast_slo(self):
        guard = self._forecast_slo_guard
        if guard is None or self.done or self.now < guard.check_after:
            return
        guard.check_after = self.now + 1.0  # CPU pruning cadence; no new time events.
        failed = len(guard.failed) + sum(
            self.request_has_failed(r.view) for r in self._live_requests.values()
        )
        upper = 1 - failed / guard.total if guard.total else 1.0
        if upper + 1e-12 < self.slo.minimum_request_fraction:
            self.status = "forecast_slo_infeasible"
            self.reason = (
                f"Sampled branch has irreversible SLO failures: {failed}/{guard.total}; "
                f"joint_fraction_upper_bound={upper:.12g}; stopped before full cost prediction"
            )
            self.done = True

    def request_has_failed(self, view: RequestView):
        ttft = (
            view.first_token_at if view.first_token_at is not None else self.now
        ) - view.ready_at
        gap = max(
            view.maximum_tbt_seconds,
            self.now - view.last_token_at
            if view.completed_at is None and view.last_token_at is not None
            else 0.0,
        )
        return (
            ttft > self.slo.ttft_seconds + 1e-12 or gap > self.slo.tbt_seconds + 1e-12
        )

    def check_candidate_bound(self):
        guard = self._candidate_guard
        if guard is None or self.done or self.now < guard.check_after:
            return
        guard.check_after = self.now + 1.0
        guard.lower = (
            guard.completed_cost
            + (
                guard.fixed_running_cost + self.switching_gpu_seconds()
                if guard.fixed_running_cost is not None
                else sum(self.pool.gpu_seconds.values())
            )
        ) / guard.samples
        reason = guard.bound.rejection(guard.fraction_cap, guard.lower)
        guard.upper = guard.fraction_cap
        if not reason and guard.bound.fraction_pruning:
            self.flush_decode()
            failed = len(guard.failed) + sum(
                r.view.request_id not in guard.excluded
                and self.request_has_failed(r.view)
                for r in self._live_requests.values()
            )
            guard.upper = min(
                guard.fraction_cap, 1 - failed / guard.total if guard.total else 1.0
            )
            reason = guard.bound.rejection(guard.upper, guard.lower)
        if reason:
            self.status, self.reason, self.done = "candidate_pruned", reason, True

    def reject_unrepairable_forecast_wait(self):
        target = self._forecast_target
        if target is None:
            return  # A real policy or externally scheduled action may repair it.
        changes_actor = (target.actor_dp, target.actor_tp) != (
            self.layout.actor_dp,
            self.layout.actor_tp,
        )
        pending = self.pending_handoff
        repair_planned = self.decision_pending or (
            pending is not None
            and pending["new_actor"] is not None
            and pending["new_actor"] != self.actor_id
        )
        if not changes_actor or not repair_planned:
            raise ActorMemoryRejected(
                self.actor_wait.reason
                + "; no planned Actor migration can repair this batch"
            )

    def request_handoff(self, target: OnlineLayout):
        if not self.reconfiguration_allowed():
            return False
        if target.actor_gpus + target.rollout_gpus > 16:
            raise ValueError(
                "Target Actor and Rollout share a sixteen-GPU steady budget"
            )
        if self.pending_handoff is not None or target == self.layout:
            return False
        if self.config.batch_size % target.actor_dp:
            raise ValueError("Target Actor DP must divide the configured episode batch")
        current = self.layout
        if self.policy is not None and self.policy.config.independent_role_history:
            allowed = self.policy.allowed_roles(self.now)
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
        stopped = self.lifecycle is not None and self.lifecycle.state == "stopped"
        if stopped and (target.actor_dp, target.actor_tp) != (
            current.actor_dp,
            current.actor_tp,
        ):
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
        previous_peak = self.pool.peak_gpus
        for role, count in roles:
            identity = self.next_deployment
            self.next_deployment += 1
            tp = target.actor_tp if role == "actor" else target.rollout_tp
            if self.pool.acquire(str(identity), role, count, tp=tp) is None:
                for previous in acquired.values():
                    self.pool.release(str(previous))
                self.pool.peak_gpus = previous_peak
                self.emit(
                    "handoff_deferred",
                    target=asdict(target),
                    reason="Insufficient spare GPUs for node-local TP groups",
                )
                return False
            acquired[role] = identity
        recovery_placement = None
        if stopped:
            # Probe recovery while both Rollout generations still occupy GPUs.
            # This is a feasibility check, not a charged or persistent lease.
            actual_peak = self.pool.peak_gpus
            recovery_placement = self.pool.acquire(
                "actor-recovery-probe", "actor", current.actor_gpus, tp=current.actor_tp
            )
            if recovery_placement is None:
                for previous in acquired.values():
                    self.pool.release(str(previous))
                self.pool.peak_gpus = previous_peak
                self.emit(
                    "handoff_deferred",
                    target=asdict(target),
                    reason="Rollout overlap would prevent Actor recovery",
                )
                return False
            self.pool.release("actor-recovery-probe")
            self.pool.peak_gpus = actual_peak
        self.pending_handoff = dict(
            target=target,
            new_actor=acquired.get("actor"),
            new_rollout=acquired.get("rollout"),
            actor_ready=False,
            actor_syncing=False,
            rollout_fence=False,
            rollout_switched="rollout" not in acquired,
        )
        self._handoff_old_costs = {
            str(
                self.actor_id if role == "actor" else self.rollout_id
            ): self.pool.lease_gpu_seconds.get(
                str(self.actor_id if role == "actor" else self.rollout_id), 0.0
            )
            for role in acquired
        }
        for role, identity in acquired.items():
            self.deployments[identity] = _Deployment(
                identity, role, target, "preparing", self.provider_version
            )
            seconds = (
                self.handoff.actor_startup_seconds
                if role == "actor"
                else self.handoff.rollout_startup_seconds
            )
            self.schedule_at(self.now + seconds, "process_started", identity)
        self.emit(
            "handoff_start",
            source=asdict(current),
            target=asdict(target),
            leases=dict(self.pool.leases),
            actor_stopped=stopped,
            actor_recovery_placement=recovery_placement,
        )
        return True

    def switching_gpu_seconds(self):
        """Old changed-role leases beyond target residency; no double counting."""
        return sum(
            self.pool.lease_gpu_seconds.get(owner, 0.0) - before
            for owner, before in self._handoff_old_costs.items()
        )

    def load_deployment(self, identity: int):
        deployment = self.deployments[identity]
        size = (
            self.handoff.actor_state_bytes
            if deployment.role == "actor"
            else self.handoff.weight_bytes
        )
        if deployment.role == "rollout":
            # One host-side model image per destination node; replica startup
            # within a node retains the existing shared-image assumption.
            size *= len({node for node, _ in self.pool.leases[str(identity)][1]})
        start, end = self.link.reserve(self.now, byte_count=size)
        self.emit(
            "deployment_load",
            deployment=identity,
            role=deployment.role,
            snapshot_version=self.provider_version,
            transfer_start=start,
            ends_at=end,
            bytes=size,
        )
        self.schedule_at(end, "deployment_loaded", (identity, self.provider_version))

    def add_restoring_actor(self, identity):
        self.deployments[identity] = _Deployment(
            identity, "actor", self.layout, "preparing", self.provider_version
        )

    def prepared(self, identity: int, version: int):
        deployment = self.deployments[identity]
        deployment.version, deployment.status = version, "prepared"
        if deployment.role == "actor":
            if self.lifecycle is not None and self.lifecycle.restore_id == identity:
                self.lifecycle.prepared(identity)
            else:
                self.pending_handoff["actor_ready"] = True
        self.emit(
            "deployment_prepared",
            deployment=identity,
            role=deployment.role,
            version=version,
        )

    def maybe_rollout_cutover(self):
        pending = self.pending_handoff
        if not pending or pending["new_rollout"] is None or pending["rollout_switched"]:
            return
        new = self.deployments[pending["new_rollout"]]
        if new.status != "prepared" or self.publication != "idle":
            return
        if new.version != self.provider_version:
            # Finish the current update, but fence further updates while final
            # synchronization catches up with its last published version.
            pending["rollout_fence"] = True
            new.status = "syncing"
            new.transfers += 1
            nodes = len({node for node, _ in self.pool.leases[str(new.identity)][1]})
            byte_count = self.handoff.weight_bytes * nodes
            start, end = self.link.reserve(
                self.now,
                byte_count=byte_count,
                setup_seconds=self.handoff.rollout_sync_setup_seconds,
            )
            self.emit(
                "rollout_catchup_start",
                deployment=new.identity,
                version=self.provider_version,
                transfer_start=start,
                ends_at=end,
                bytes=byte_count,
                nodes=nodes,
            )
            self.schedule_at(
                end, "rollout_synchronized", (new.identity, self.provider_version)
            )
            return
        old = self.deployments[self.rollout_id]
        old.status, new.status = "draining", "active"
        self.rollout_id = new.identity
        pending["rollout_switched"] = True
        pending["rollout_fence"] = False
        self.emit(
            "rollout_cutover",
            source=old.identity,
            target=new.identity,
            version=new.version,
        )

    def rollout_synchronized(self, identity: int, version: int):
        deployment = self.deployments[identity]
        deployment.version, deployment.status = version, "prepared"
        deployment.transfers -= 1
        self.emit("rollout_catchup_complete", deployment=identity, version=version)

    def maybe_actor_handoff(self):
        pending = self.pending_handoff
        if (
            not pending
            or not pending["actor_ready"]
            or pending["actor_syncing"]
            or pending["new_actor"] == self.actor_id
            or self.actor_batch
            or self.publication != "idle"
        ):
            return
        pending["actor_syncing"] = True
        start, end = self.link.reserve(
            self.now,
            byte_count=self.handoff.actor_state_bytes,
            setup_seconds=self.handoff.actor_sync_setup_seconds,
        )
        snapshot = (self.provider_version, self.consumed)
        self.emit(
            "actor_handoff_start",
            source=self.actor_id,
            target=pending["new_actor"],
            version=snapshot[0],
            cursor=snapshot[1],
            transfer_start=start,
            ends_at=end,
        )
        self.schedule_at(end, "actor_synchronized", snapshot)

    def actor_synchronized(self, snapshot):
        if snapshot != (self.provider_version, self.consumed) or self.actor_batch:
            raise ValueError(
                "Training authority changed while its boundary snapshot was transferring"
            )
        pending = self.pending_handoff
        old = self.actor_id
        self.actor_id = pending["new_actor"]
        self.deployments[self.actor_id].status = "active"
        self.deployments[self.actor_id].version = self.provider_version
        pending["actor_ready"] = False
        self.release_deployment(old)
        self.emit(
            "actor_cutover",
            source=old,
            target=self.actor_id,
            version=self.provider_version,
            cursor=self.consumed,
        )

    def retire_rollouts(self):
        for identity, deployment in tuple(self.deployments.items()):
            if (
                deployment.role != "rollout"
                or deployment.status != "draining"
                or deployment.busy
                or deployment.transfers
            ):
                continue
            if not any(group.rows for group in deployment.replicas.values()):
                self.release_deployment(identity)

    def maybe_finish_handoff(self):
        pending = self.pending_handoff
        if not pending:
            return
        actor_done = (
            pending["new_actor"] is None or pending["new_actor"] == self.actor_id
        )
        active = {self.actor_id, self.rollout_id}
        if self.lifecycle is not None and self.lifecycle.restore_id is not None:
            active.add(self.lifecycle.restore_id)
        if (
            actor_done
            and pending["rollout_switched"]
            and set(self.deployments) <= active
        ):
            self.emit("handoff_complete", layout=asdict(self.layout))
            self.pending_handoff = None
            if self.policy:
                self.policy.switched(self.now)

    def poll_policy(self):
        from .online_candidates import OnlineForecastModel

        if not self.policy or self.done:
            return
        if self.lifecycle is not None:
            self.lifecycle.tick()
        if self.control is not None:
            return self.control.poll()
        view = self.observe()
        model = OnlineForecastModel(
            self.models,
            self.config,
            self.handoff,
            self.slo,
            self.policy.config.decision_delay_seconds,
            executor=self.candidate_executor,
            actor_memory_reserve_fraction=self.policy.config.actor_memory_reserve_fraction,
            actor_memory_scope=self.policy.config.actor_memory_scope,
        )
        decision = self.policy.decide(view, model)
        record = asdict(decision)
        record["actor_lifecycle_state"] = (
            view.actor.lifecycle.state if view.actor.lifecycle is not None else None
        )
        record["role_history"] = self.policy.role_history_status(view.now)
        record["observation_id"] = fingerprint(view)
        record["evaluation"] = model.execution_stats
        self.decisions.append(record)
        self.emit("decision", **record)
        if decision.target != view.layout:
            self.decision_pending = True
            self.schedule_at(
                self.now + self.policy.config.decision_delay_seconds,
                "decision_apply",
                decision.target,
                priority=9,
            )
        self.schedule_at(
            self.now + self.policy.config.check_interval_seconds, "policy", priority=10
        )

    def prediction_guard_active(self):
        return self._forecast_slo_guard is not None

    def request_finished(self, view):
        guard = self._forecast_slo_guard
        if (
            guard is not None
            and view.request_id not in guard.excluded
            and self.request_has_failed(view)
        ):
            guard.failed.add(view.request_id)
        bound = self._candidate_guard
        if (
            bound is not None
            and view.request_id not in bound.excluded
            and self.request_has_failed(view)
        ):
            bound.failed.add(view.request_id)
