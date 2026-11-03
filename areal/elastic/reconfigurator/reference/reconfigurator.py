# SPDX-License-Identifier: Apache-2.0

"""Online reconfiguration: observe, search, revalidate and apply on one timeline."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field

from areal.elastic.contracts import (
    ConfigurationEvaluator,
    OnlineDecision,
    OnlineLayout,
    OnlineObservation,
    RollingSLO,
    nonnegative,
    positive_int,
)
from areal.elastic.reconfigurator.reference import trigger as trigger_rules
from areal.elastic.reconfigurator.reference.forecast import (
    HistoricalForecaster,
    HistoricalReplay,
)
from areal.elastic.reconfigurator.reference.identity import fingerprint
from areal.elastic.reconfigurator.reference.optimizer import (
    OnlineOptimizer,
    meets_target,
)
from areal.elastic.serving import ServingSLO, rolling_slo


@dataclass(frozen=True)
class OnlineReconfiguratorConfig:
    check_interval_seconds: float
    slo_window_seconds: float
    warning_ratio: float = field(default=0.8, kw_only=True)
    slack_ratio: float = field(default=0.5, kw_only=True)
    pressure_checks: int = field(default=1, kw_only=True)
    slack_checks: int = field(default=1, kw_only=True)
    cooldown_seconds: float
    decision_delay_seconds: float
    minimum_completed_requests: int
    max_evaluations: int
    optimize_search: bool = True
    deduplicate_searches: bool = True
    pressure_request_fraction: float = 0.8
    slack_request_fraction: float = 0.9
    actor_memory_reserve_fraction: float = 0.0
    actor_memory_scope: str = "all_completed"
    independent_role_history: bool = False

    def __post_init__(self):
        for name in ("check_interval_seconds", "slo_window_seconds"):
            nonnegative(name, getattr(self, name), positive=True)
        for name in ("cooldown_seconds", "decision_delay_seconds"):
            nonnegative(name, getattr(self, name))
        if not 0 <= self.pressure_request_fraction < self.slack_request_fraction <= 1:
            raise ValueError(
                "Require 0 <= pressure_request_fraction < slack_request_fraction <= 1"
            )
        if not 0 <= self.actor_memory_reserve_fraction < 1:
            raise ValueError("Actor memory reserve must be in [0, 1)")
        if self.actor_memory_scope not in ("all_completed", "evaluation"):
            raise ValueError("Actor memory scope must be all_completed or evaluation")
        for name in (
            "pressure_checks",
            "slack_checks",
            "minimum_completed_requests",
            "max_evaluations",
        ):
            positive_int(name, getattr(self, name))


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
    publication_wait_started: float | None = None
    publication_resume_stage: str | None = None


class OnlineReconfigurator:
    implementation = "reference"

    def __init__(
        self,
        candidates: tuple[OnlineLayout, ...],
        forecaster: HistoricalForecaster | HistoricalReplay,
        config: OnlineReconfiguratorConfig,
        slo: ServingSLO,
    ):
        if not candidates:
            raise ValueError("Explicit online candidates are required")
        self.candidates = tuple(dict.fromkeys(candidates))
        self.optimizer = OnlineOptimizer()
        self.forecaster, self.config, self.slo = (forecaster, config, slo)
        self.decision_index = 0
        self.last_switch = -math.inf
        self.last_role_switch = {"actor": -math.inf, "rollout": -math.inf}
        self.last_search = -math.inf
        self._last_search_state = None
        self._winners = []
        self.environment = None

    def search_invalidated(self):
        self._last_search_state = None

    def search_state(self, view, evaluator):
        return trigger_rules.search_state(self, view, evaluator)

    def redundant_search(self, view, evaluator):
        return trigger_rules.redundant_search(self, view, evaluator)

    def switched(self, when: float):
        """Called after handoff completion, including old-deployment release."""
        self.last_switch = when

    def role_switched(self, role: str, when: float):
        """One changed side has taken over and its old deployment is released."""
        self.last_role_switch[role] = when
        if self.config.independent_role_history:
            self.search_invalidated()

    def allowed_roles(self, when: float) -> dict[str, bool]:
        return trigger_rules.allowed_roles(self, when)

    def role_history_status(self, when: float):
        if not self.config.independent_role_history:
            return None
        return dict(
            last_completed={
                r: t if math.isfinite(t) else None
                for r, t in self.last_role_switch.items()
            },
            allowed=self.allowed_roles(when),
        )

    def _trigger(self, stats: RollingSLO) -> str:
        return trigger_rules.trigger(self, stats)

    def decide(
        self, view: OnlineObservation, evaluator: ConfigurationEvaluator
    ) -> OnlineDecision:
        cfg, slo = (self.config, self.slo)
        historical = isinstance(self.forecaster, HistoricalReplay)
        if historical:
            self.forecaster.observe(view)
        stats = rolling_slo(view, slo, cfg.slo_window_seconds)
        started = time.perf_counter()
        evidence = None

        def keep(reason, trigger="none", predictions=(), seeds=()):
            return OnlineDecision(
                view.now,
                trigger,
                view.layout,
                tuple(predictions),
                tuple(seeds),
                stats,
                time.perf_counter() - started,
                reason,
                replay_window=evidence,
                workload_features=self.forecaster.features if historical else None,
            )

        roles = self.allowed_roles(view.now) if cfg.independent_role_history else None
        if roles is not None and (not any(roles.values())):
            return keep("Waiting for each role's own full history window")
        if (
            not cfg.independent_role_history
            and historical
            and (
                view.now - max(0.0, self.last_switch)
                < self.forecaster.config.history_seconds
            )
        ):
            return keep(
                "Waiting for a full history window since startup or completed handoff before judging service"
            )
        if view.busy or (
            not cfg.independent_role_history
            and view.now - self.last_switch < cfg.cooldown_seconds
        ):
            return keep("Transition/publication/decision pending or switch cooldown")
        trigger = self._trigger(stats)
        if trigger == "none":
            return keep("No SLO pressure or slack trigger")
        if (
            historical
            and cfg.deduplicate_searches
            and self.redundant_search(view, evaluator)
        ):
            return keep("Unchanged recent search evidence; retain deployment", trigger)
        forecasts = self.forecaster.predict(view, self.decision_index)
        self.decision_index += 1
        if not forecasts:
            return keep(
                "Insufficient completed-episode history; retain initial/current deployment",
                trigger,
            )
        evidence = forecasts[0].replay
        if historical:
            self.last_search = view.now
            self._last_search_state = self.search_state(view, evaluator)
        seeds = () if historical else tuple(f.seed for f in forecasts)
        candidates = (view.layout,) + tuple(
            c for c in self.candidates if c != view.layout
        )
        if roles is not None:
            candidates = tuple(
                c
                for c in candidates
                if (
                    roles["actor"]
                    or (c.actor_dp, c.actor_tp)
                    == (view.layout.actor_dp, view.layout.actor_tp)
                )
                and (
                    roles["rollout"]
                    or (c.rollout_replicas, c.rollout_tp)
                    == (view.layout.rollout_replicas, view.layout.rollout_tp)
                )
            )
        layouts = candidates[: cfg.max_evaluations]
        if cfg.independent_role_history and len(layouts) == 1:
            return keep("No alternative layout under role history constraints", trigger)
        evaluate_many = getattr(evaluator, "evaluate_many", None)
        if hasattr(evaluator, "optimizer"):
            evaluator.optimizer = self.optimizer
        search = getattr(evaluator, "search", None)
        preferred = ()
        if historical and self._winners and (self.forecaster.features is not None):
            preferred = (
                min(
                    self._winners,
                    key=lambda row: row[0].distance(self.forecaster.features),
                )[1],
            )
        predictions = (
            tuple(
                search(view, forecasts, layouts, preferred=preferred, trigger=trigger)
            )
            if historical and cfg.optimize_search and (search is not None)
            else tuple(evaluate_many(view, forecasts, layouts))
            if evaluate_many is not None
            else tuple(evaluator.evaluate(view, forecasts, c) for c in layouts)
        )
        expected = predictions[0].work_ids
        runnable = [
            p
            for p in predictions
            if p.supported
            and p.work_ids == expected
            and (p.worst_request_fraction is not None)
            and (p.mean_gpu_seconds is not None)
        ]
        eligible = [
            p
            for p in runnable
            if p.worst_request_fraction + 1e-12 >= slo.minimum_request_fraction
        ]
        if historical:
            current = next((p for p in runnable if p.layout == view.layout), None)
            if not runnable:
                return keep(
                    "No executable candidate with verified Actor capacity",
                    trigger,
                    predictions,
                    seeds,
                )
            best = self.optimizer.select(
                predictions, view.layout, slo.minimum_request_fraction, trigger=trigger
            )
            if best is None:
                return keep(
                    "No scored candidate; retain current deployment",
                    trigger,
                    predictions,
                    seeds,
                )
            if meets_target(best, slo.minimum_request_fraction):
                reason = (
                    "Minimum switching plus fixed-window GPU cost among SLO-target and sustainable-Actor candidates; "
                    + (
                        "current excluded when a feasible replacement exists"
                        if trigger == "pressure"
                        else "current included under slack"
                    )
                )
            elif trigger == "slack":
                reason = "No SLO-target and sustainable-Actor candidate under slack; retain current deployment"
            else:
                reason = "Best executable request fraction, then minimum credit deficit and switching plus fixed-window GPU cost; SLO/capacity target remains unmet"
            if self.forecaster.features is not None:
                self._winners = (
                    self._winners + [(self.forecaster.features, best.layout)]
                )[-16:]
            return OnlineDecision(
                view.now,
                trigger,
                best.layout,
                predictions,
                seeds,
                stats,
                time.perf_counter() - started,
                reason,
                evidence,
                self.forecaster.features,
                current.window_gpu_seconds - best.window_gpu_seconds
                if current
                and current.window_gpu_seconds is not None
                and (best.window_gpu_seconds is not None)
                else None,
                current.mean_gpu_seconds - best.mean_gpu_seconds if current else None,
            )
        if not eligible:
            return keep(
                "No predicted SLO-feasible candidate; continue serving current deployment",
                trigger,
                predictions,
                seeds,
            )
        best = min(
            eligible,
            key=lambda p: (p.mean_gpu_seconds, p.layout != view.layout, p.layout),
        )
        return OnlineDecision(
            view.now,
            trigger,
            best.layout,
            predictions,
            seeds,
            stats,
            time.perf_counter() - started,
            "Minimum predicted GPU seconds among SLO-feasible sampled continuations",
        )

    def make_evaluator(self, *args, **kwargs):
        from areal.elastic.reconfigurator.reference.evaluator import Evaluator

        return Evaluator(*args, optimizer=self.optimizer, **kwargs)

    def bind(self, environment, clock=None):
        """Attach an execution environment; this object also drives its callbacks."""
        self.environment = environment
        self.clock = clock or time.perf_counter
        self.pending: _Search | None = None
        self.sequence = 0
        self.delay_estimate = self.config.decision_delay_seconds
        self.verification_estimate = self.delay_estimate
        self.stats = dict(
            searches=0,
            search_wall_seconds=0.0,
            verifications=0,
            verification_wall_seconds=0.0,
            application_wait_seconds=0.0,
            publication_waits=0,
            publication_wait_seconds=0.0,
            applied=0,
            discarded=0,
            cancelled=0,
        )
        return self

    def evaluator(self, delay):
        sim = self.environment
        return self.make_evaluator(
            getattr(sim, "prediction_cost_models", sim.cost_models),
            sim.config,
            sim.handoff,
            sim.slo,
            delay,
            executor=sim.candidate_executor,
            actor_memory_reserve_fraction=self.config.actor_memory_reserve_fraction,
            actor_memory_scope=self.config.actor_memory_scope,
        )

    def observe_history(self, view, *, confirm_change=True):
        forecaster = self.forecaster
        if isinstance(forecaster, HistoricalReplay):
            forecaster.observe(view, confirm_change=confirm_change)
        return getattr(forecaster, "regime_index", None)

    def changed_features(self, features):
        history = self.forecaster
        return (
            isinstance(history, HistoricalReplay)
            and features is not None
            and (history.features is not None)
            and (features.distance(history.features) >= history.config.change_ratio)
        )

    def poll(self):
        sim = self.environment
        view = sim.observe()
        regime = self.observe_history(view)
        if self.pending is not None:
            sim.emit("search_pending", search=self.pending.identity)
        else:
            started = self.clock()
            evaluator = self.evaluator(self.delay_estimate)
            decision = self.decide(view, evaluator)
            elapsed = max(0.0, self.clock() - started)
            record = dict(
                **asdict(decision),
                observation_id=fingerprint(view),
                evaluation=dict(evaluator.execution_stats),
                search_wall_seconds=elapsed,
                delay_estimate_seconds=self.delay_estimate,
                actor_running=view.actor.running,
                role_history=self.role_history_status(view.now),
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
            sim.now + self.config.check_interval_seconds, "reconfigurator", priority=10
        )

    def complete(self, identity):
        job, sim = (self.pending, self.environment)
        if job is None or job.identity != identity or job.stage != "search":
            return
        self.delay_estimate = job.record["search_wall_seconds"]
        job.record["search_available_at"] = sim.now
        sim.emit(
            "search_complete",
            search=identity,
            observed_at=job.view.now,
            seconds=self.delay_estimate,
        )
        self.verify(identity)

    def verify(self, identity):
        job, sim = (self.pending, self.environment)
        if job is None or job.identity != identity or job.stage != "search":
            return
        view = sim.observe()
        regime = self.observe_history(view, confirm_change=False)
        if (
            sim.pending_handoff is not None
            or sim.decision_pending
            or view.layout != job.view.layout
            or (view.leases != job.view.leases)
            or (regime != job.regime)
            or self.changed_features(job.decision.workload_features)
        ):
            self.finish(
                "discarded",
                view.layout,
                "Deployment or workload regime changed during search",
            )
            return
        if job.decision.target == view.layout:
            self.finish("retained", view.layout, "Search retained current deployment")
            return
        if self.wait_for_publication("search"):
            return
        forecaster = self.forecaster
        forecasts = (
            forecaster.predict(view, self.decision_index, confirm_change=False)
            if isinstance(forecaster, HistoricalReplay)
            else forecaster.predict(view, self.decision_index)
        )
        if not forecasts:
            self.finish(
                "discarded",
                view.layout,
                "No causal history for latest-state verification",
            )
            return
        started = self.clock()
        evaluator = self.evaluator(self.verification_estimate)
        job.validation = evaluator.evaluate_many(
            view, forecasts, (view.layout, job.decision.target)
        )
        job.elapsed = max(0.0, self.clock() - started)
        job.validation_view, job.stage = (view, "verification")
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
        job.record["verification_evaluation"] = dict(evaluator.execution_stats)
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
            view.actor.waiting,
            view.input_closed,
            view.actor.running,
        )

    def wait_for_publication(self, resume_stage):
        job, sim = (self.pending, self.environment)
        if sim.publication == "idle":
            return False
        job.publication_wait_started = sim.now
        job.publication_resume_stage = resume_stage
        job.stage = "publication_wait"
        self.stats["publication_waits"] += 1
        sim.emit("decision_publication_wait", search=job.identity, stage=resume_stage)
        return True

    def publication_complete(self):
        job, sim = (self.pending, self.environment)
        if job is None or job.stage != "publication_wait":
            return
        seconds = sim.now - job.publication_wait_started
        stage = job.publication_resume_stage
        key = (
            "verification_publication_wait_seconds"
            if stage == "search"
            else "application_publication_wait_seconds"
        )
        job.record[key] = job.record.get(key, 0.0) + seconds
        self.stats["publication_wait_seconds"] += seconds
        job.stage = stage
        job.publication_wait_started = job.publication_resume_stage = None
        sim.emit(
            "decision_publication_resume",
            search=job.identity,
            stage=stage,
            seconds=seconds,
            version=sim.provider_version,
            cursor=sim.consumed,
        )
        if stage == "search":
            self.verify(job.identity)
        else:
            self.ready(job.identity)

    def verified(self, identity):
        job, sim = (self.pending, self.environment)
        if job is None or job.identity != identity or job.stage != "verification":
            return
        self.verification_estimate = max(
            self.config.decision_delay_seconds, job.elapsed * 1.25
        )
        sim.emit("verification_complete", search=identity, seconds=job.elapsed)
        job.record["verification_completed_at"] = sim.now
        if job.elapsed > job.record["verification_delay_estimate_seconds"] + 1e-09:
            self.finish(
                "discarded",
                sim.layout,
                "Verification exceeded its modeled CPU delay; retain serving deployment",
            )
            return
        planned = (
            job.record["verification_started_at"]
            + job.record["verification_delay_estimate_seconds"]
        )
        job.record["application_scheduled_at"] = planned
        self.stats["application_wait_seconds"] += max(0.0, planned - sim.now)
        job.stage = "application_wait"
        sim.schedule_at(max(sim.now, planned), "decision_ready", identity, priority=9)

    def ready(self, identity):
        job, sim = (self.pending, self.environment)
        if job is None or job.identity != identity or job.stage != "application_wait":
            return
        view = sim.observe()
        regime = self.observe_history(view, confirm_change=False)
        if (
            sim.pending_handoff is not None
            or sim.decision_pending
            or regime != job.regime
            or self.changed_features(job.validation_features)
            or (self.preconditions(view) != self.preconditions(job.validation_view))
            or (
                sim.now
                - job.validation_view.now
                - job.record.get("application_publication_wait_seconds", 0.0)
                > self.config.check_interval_seconds
            )
        ):
            self.finish(
                "discarded",
                view.layout,
                "Verification preconditions changed or expired; retain serving deployment",
            )
            return
        best = self.optimizer.select(
            job.validation,
            view.layout,
            sim.slo.minimum_request_fraction,
            trigger=job.decision.trigger
            if isinstance(self.forecaster, HistoricalReplay)
            else "slack",
        )
        if (
            not isinstance(self.forecaster, HistoricalReplay)
            and best is not None
            and (best.worst_request_fraction + 1e-12 < sim.slo.minimum_request_fraction)
        ):
            best = None
        if best is None or best.layout == view.layout:
            self.finish(
                "retained",
                view.layout,
                "Latest-state verification retained current deployment",
            )
        elif self.wait_for_publication("application_wait"):
            return
        elif sim.request_handoff(best.layout):
            job.record["application_state"] = dict(
                training_version=view.training_version,
                provider_version=view.provider_version,
                consumed_episodes=view.consumed_episodes,
            )
            self.stats["applied"] += 1
            self.finish(
                "applied", best.layout, "Latest-state verification accepted target"
            )
        else:
            self.finish(
                "discarded", view.layout, "Target resources unavailable at application"
            )

    def finish(self, status, target, reason):
        job, sim = (self.pending, self.environment)
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
            self.search_invalidated()

    def cancel(self):
        if self.pending is not None:
            self.finish(
                "cancelled",
                self.environment.layout,
                "Finite workload completed before CPU decision application",
            )
