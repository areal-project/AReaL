# SPDX-License-Identifier: Apache-2.0

"""CPU decision availability on the serving timeline, with causal revalidation."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

from areal.elastic.policy.basic.history import HistoricalReplay
from areal.elastic.policy.basic.search import select_prediction
from areal.elastic.policy.reference.core import fingerprint


@dataclass
class _Search:
    identity: int
    view: object
    decision: object
    record: dict
    regime: int | None
    stage: str = "search"
    validation: tuple = ()
    validation_view: object = None
    validation_features: object = None
    elapsed: float = 0.0


class OnlineControl:
    def __init__(self, simulation, clock=None):
        self.sim = simulation
        self.clock = clock or time.perf_counter
        self.pending: _Search | None = None
        self.sequence = 0
        self.delay_estimate = simulation.policy.config.decision_delay_seconds
        self.verification_estimate = self.delay_estimate
        self.stats = dict(
            searches=0,
            search_wall_seconds=0.0,
            verifications=0,
            verification_wall_seconds=0.0,
            application_wait_seconds=0.0,
            applied=0,
            discarded=0,
            cancelled=0,
        )

    def model(self, delay):
        from .online_simulation import OnlineForecastModel

        sim = self.sim
        return OnlineForecastModel(
            sim.models,
            sim.config,
            sim.handoff,
            sim.slo,
            delay,
            executor=sim.candidate_executor,
        )

    def observe_history(self, view, *, confirm_change=True):
        forecaster = self.sim.policy.forecaster
        if isinstance(forecaster, HistoricalReplay):
            forecaster.observe(view, confirm_change=confirm_change)
        return getattr(forecaster, "regime_index", None)

    def changed_features(self, features):
        history = self.sim.policy.forecaster
        return (
            isinstance(history, HistoricalReplay)
            and features is not None
            and history.features is not None
            and features.distance(history.features) >= history.config.change_ratio
        )

    def poll(self):
        sim, policy = self.sim, self.sim.policy
        view = sim.observe()
        regime = self.observe_history(view)
        if self.pending is not None:
            sim.emit("search_pending", search=self.pending.identity)
        else:
            started = self.clock()
            model = self.model(self.delay_estimate)
            decision = policy.decide(view, model)
            elapsed = max(0.0, self.clock() - started)
            record = dict(
                **asdict(decision),
                observation_id=fingerprint(view),
                evaluation=dict(model.execution_stats),
                search_wall_seconds=elapsed,
                delay_estimate_seconds=self.delay_estimate,
            )
            if not decision.predictions:
                record["application_status"] = "no_search"
                sim.decisions.append(record)
                sim.emit("decision", **record)
            else:
                self.sequence += 1
                self.pending = _Search(self.sequence, view, decision, record, regime)
                self.stats["searches"] += 1
                self.stats["search_wall_seconds"] += elapsed
                sim.emit(
                    "search_start",
                    search=self.sequence,
                    observation_id=record["observation_id"],
                )
                sim.schedule_at(
                    sim.now + elapsed, "search_complete", self.sequence, priority=9
                )
        sim.schedule_at(
            sim.now + policy.config.check_interval_seconds, "policy", priority=10
        )

    def complete(self, identity):
        job, sim = self.pending, self.sim
        if job is None or job.identity != identity or job.stage != "search":
            return
        self.delay_estimate = job.record["search_wall_seconds"]
        job.record["search_available_at"] = sim.now
        view = sim.observe()
        regime = self.observe_history(view, confirm_change=False)
        sim.emit(
            "search_complete",
            search=identity,
            observed_at=job.view.now,
            seconds=self.delay_estimate,
        )
        if (
            view.busy
            or view.layout != job.view.layout
            or view.leases != job.view.leases
            or regime != job.regime
            or self.changed_features(job.decision.workload_features)
        ):
            self.finish(
                "discarded",
                view.layout,
                "Deployment/publication or workload regime changed during search",
            )
            return
        if job.decision.target == view.layout:
            self.finish("retained", view.layout, "Search retained current deployment")
            return
        forecaster = sim.policy.forecaster
        forecasts = (
            forecaster.predict(view, sim.policy.decision_index, confirm_change=False)
            if isinstance(forecaster, HistoricalReplay)
            else forecaster.predict(view, sim.policy.decision_index)
        )
        if not forecasts:
            self.finish(
                "discarded",
                view.layout,
                "No causal history for latest-state verification",
            )
            return
        started = self.clock()
        model = self.model(self.verification_estimate)
        job.validation = model.evaluate_many(
            view, forecasts, (view.layout, job.decision.target)
        )
        job.elapsed = max(0.0, self.clock() - started)
        job.validation_view, job.stage = view, "verification"
        job.record["verification_observation_id"] = fingerprint(view)
        job.record["verification_started_at"] = sim.now
        job.record["verification_wall_seconds"] = job.elapsed
        job.record["verification_delay_estimate_seconds"] = self.verification_estimate
        job.record["verification_input_closes_at"] = min(
            forecast.input_closes_at for forecast in forecasts
        )
        job.validation_features = getattr(forecaster, "features", None)
        job.record["verification_features"] = (
            asdict(job.validation_features)
            if job.validation_features is not None
            else None
        )
        job.record["verification_evaluation"] = dict(model.execution_stats)
        job.record["validation_predictions"] = tuple(asdict(p) for p in job.validation)
        self.stats["verifications"] += 1
        self.stats["verification_wall_seconds"] += job.elapsed
        sim.emit(
            "verification_start", search=identity, observation_id=fingerprint(view)
        )
        sim.schedule_at(
            sim.now + job.elapsed, "verification_complete", identity, priority=9
        )

    @staticmethod
    def preconditions(view):
        return (
            view.layout,
            view.leases,
            view.training_version,
            view.provider_version,
            view.consumed_episodes,
            view.actor.batch,
            view.actor.waiting,
            view.input_closed,
        )

    def verified(self, identity):
        job, sim = self.pending, self.sim
        if job is None or job.identity != identity or job.stage != "verification":
            return
        self.verification_estimate = max(
            sim.policy.config.decision_delay_seconds, job.elapsed * 1.25
        )
        sim.emit("verification_complete", search=identity, seconds=job.elapsed)
        job.record["verification_completed_at"] = sim.now
        if job.elapsed > job.record["verification_delay_estimate_seconds"] + 1e-9:
            self.finish(
                "discarded",
                sim.layout,
                "Verification exceeded its modeled CPU delay; retain serving deployment",
            )
            return
        # Starting early is not necessarily cheaper: the new Actor may occupy
        # GPUs while still waiting for the same old update/publication boundary.
        planned = (
            job.record["verification_started_at"]
            + job.record["verification_delay_estimate_seconds"]
        )
        job.record["application_scheduled_at"] = planned
        self.stats["application_wait_seconds"] += max(0.0, planned - sim.now)
        job.stage = "application_wait"
        sim.schedule_at(max(sim.now, planned), "decision_ready", identity, priority=9)

    def ready(self, identity):
        job, sim = self.pending, self.sim
        if job is None or job.identity != identity or job.stage != "application_wait":
            return
        view = sim.observe()
        regime = self.observe_history(view, confirm_change=False)
        if (
            view.busy
            or regime != job.regime
            or self.changed_features(job.validation_features)
            or self.preconditions(view) != self.preconditions(job.validation_view)
            or sim.now - job.validation_view.now
            > sim.policy.config.check_interval_seconds
        ):
            self.finish(
                "discarded",
                view.layout,
                "Verification preconditions changed or expired; retain serving deployment",
            )
            return
        best = select_prediction(
            job.validation,
            view.layout,
            sim.slo.minimum_request_fraction,
            trigger=job.decision.trigger
            if isinstance(sim.policy.forecaster, HistoricalReplay)
            else "slack",
        )
        if (
            not isinstance(sim.policy.forecaster, HistoricalReplay)
            and best is not None
            and best.worst_request_fraction + 1e-12 < sim.slo.minimum_request_fraction
        ):
            best = None  # Preserve the legacy sampled policy's feasibility rule.
        if best is None or best.layout == view.layout:
            self.finish(
                "retained",
                view.layout,
                "Latest-state verification retained current deployment",
            )
        elif sim.request_handoff(best.layout):
            self.stats["applied"] += 1
            self.finish(
                "applied", best.layout, "Latest-state verification accepted target"
            )
        else:
            self.finish(
                "discarded", view.layout, "Target resources unavailable at application"
            )

    def finish(self, status, target, reason):
        job, sim = self.pending, self.sim
        if job is None:
            return
        record = dict(job.record)
        record.update(
            target=asdict(target),
            proposed_target=asdict(job.decision.target),
            application_status=status,
            decision_available_at=sim.now,
            applied_at=sim.now if status == "applied" else None,
            reason=job.decision.reason + "; " + reason,
        )
        sim.decisions.append(record)
        sim.emit("decision", **record)
        self.pending = None
        if status in {"discarded", "cancelled"}:
            self.stats[status] += 1
            invalidate = getattr(sim.policy, "search_invalidated", None)
            if invalidate is not None:
                invalidate()

    def cancel(self):
        if self.pending is not None:
            self.finish(
                "cancelled",
                self.sim.layout,
                "Finite workload completed before CPU decision application",
            )
