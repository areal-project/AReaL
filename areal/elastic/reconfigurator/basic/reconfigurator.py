# SPDX-License-Identifier: Apache-2.0

"""SLO-triggered joint search over causal observations and fixed serving windows."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from areal.elastic.contracts import nonnegative, positive_int

from ...contracts import (
    ConfigurationEvaluator,
    OnlineDecision,
    OnlineLayout,
    OnlineObservation,
    RollingSLO,
)
from ...serving import ServingSLO, rolling_slo
from .forecast import HistoricalForecaster
from .history import HistoricalReplay
from .optimizer import OnlineOptimizer


@dataclass(frozen=True)
class OnlineReconfiguratorConfig:
    check_interval_seconds: float
    slo_window_seconds: float
    # Accepted for old JSON configs; triggering uses request fractions below.
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
    independent_role_history: bool = False
    actor_memory_reserve_fraction: float = 0.0
    actor_memory_scope: str = "all_completed"

    def __post_init__(self):
        if (
            self.independent_role_history
            or self.actor_memory_reserve_fraction
            or self.actor_memory_scope != "all_completed"
        ):
            raise ValueError(
                "Basic v1 does not support Reference role history or memory-reserve options"
            )
        for name in ("check_interval_seconds", "slo_window_seconds"):
            nonnegative(name, getattr(self, name), positive=True)
        for name in ("cooldown_seconds", "decision_delay_seconds"):
            nonnegative(name, getattr(self, name))
        if not 0 <= self.pressure_request_fraction < self.slack_request_fraction <= 1:
            raise ValueError(
                "Require 0 <= pressure_request_fraction < slack_request_fraction <= 1"
            )
        for name in (
            "pressure_checks",
            "slack_checks",
            "minimum_completed_requests",
            "max_evaluations",
        ):
            positive_int(name, getattr(self, name))


class OnlineReconfigurator:
    implementation = "basic"

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
        self.forecaster, self.config, self.slo = forecaster, config, slo
        self.decision_index = 0
        self.last_switch = -math.inf
        self.last_search = -math.inf
        self._last_search_state = None
        self._winners = []

    def search_invalidated(self):
        self._last_search_state = None

    def search_state(self, view, evaluator):
        active = [r for r in view.requests if r.completed_at is None]
        recent = [
            t
            for e in view.episodes
            if e.completed_at is not None
            and e.completed_at >= view.now - self.config.slo_window_seconds
            for t in e.completed_turns
        ]
        statistics = rolling_slo(view, self.slo, self.config.slo_window_seconds)
        return (
            view.layout,
            view.leases,
            getattr(evaluator, "source_ids", ()),
            view.actor.waiting,
            self.forecaster.regime_index,
            self.forecaster.features,
            (
                len(active),
                sum(max(0, r.input_tokens - r.prefilled) for r in active),
                sum(t.output_tokens for t in recent) / max(1, len(recent)),
                sum(t.tool_seconds or 0 for t in recent) / max(1, len(recent)),
                getattr(evaluator, "delay", 0.0),
                statistics.ttft_quantile_seconds,
                statistics.tpot_quantile_seconds,
            ),
            (
                statistics.fraction is not None
                and statistics.fraction >= self.slo.minimum_request_fraction,
                self._trigger(statistics),
            ),
        )

    def redundant_search(self, view, evaluator):
        old = self._last_search_state
        state = self.search_state(view, evaluator)
        if old is None or state[:5] != old[:5] or state[7] != old[7]:
            return False
        if view.now - self.last_search >= self.config.slo_window_seconds:
            return False
        ratio = self.forecaster.config.change_ratio
        if state[5] is not None and old[5] is not None:
            if state[5].distance(old[5]) >= ratio:
                return False
        return all(
            abs(a - b) / max(abs(a), abs(b), 1e-9) < ratio
            for a, b in zip(state[6], old[6], strict=True)
        )

    def switched(self, when: float):
        """Called after handoff completion, including old-deployment release."""
        self.last_switch = when

    def _trigger(self, stats: RollingSLO) -> str:
        if stats.fraction is None:
            return "none"
        if stats.fraction < self.config.pressure_request_fraction:
            return "pressure"
        if (
            stats.completed >= self.config.minimum_completed_requests
            and stats.fraction > self.config.slack_request_fraction
        ):
            return "slack"
        return "none"

    def decide(
        self, view: OnlineObservation, evaluator: ConfigurationEvaluator
    ) -> OnlineDecision:
        cfg, slo = self.config, self.slo
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

        if (
            historical
            and view.now - max(0.0, self.last_switch)
            < self.forecaster.config.history_seconds
        ):
            return keep(
                "Waiting for a full history window since startup or completed handoff before judging service"
            )
        if view.busy or view.now - self.last_switch < cfg.cooldown_seconds:
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
        layouts = candidates[: cfg.max_evaluations]
        evaluate_many = getattr(evaluator, "evaluate_many", None)
        if hasattr(evaluator, "optimizer"):
            evaluator.optimizer = self.optimizer
        search = getattr(evaluator, "search", None)
        preferred = ()
        if historical and self._winners and self.forecaster.features is not None:
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
            if (historical and cfg.optimize_search and search is not None)
            else (
                tuple(evaluate_many(view, forecasts, layouts))
                if evaluate_many is not None
                else tuple(evaluator.evaluate(view, forecasts, c) for c in layouts)
            )
        )
        expected = predictions[0].work_ids
        runnable = [
            p
            for p in predictions
            if p.supported
            and p.work_ids == expected
            and p.worst_request_fraction is not None
            and p.mean_gpu_seconds is not None
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
            if best.worst_request_fraction + 1e-12 >= slo.minimum_request_fraction:
                reason = (
                    "Minimum switching plus fixed-window GPU cost among SLO-target candidates; "
                    + (
                        "current excluded when a feasible replacement exists"
                        if trigger == "pressure"
                        else "current included under slack"
                    )
                )
            elif trigger == "slack":
                reason = (
                    "No SLO-target candidate under slack; retain current deployment"
                )
            else:
                reason = "Best executable request fraction below SLO target, then minimum switching plus fixed-window GPU cost; target remains unmet"
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
                and best.window_gpu_seconds is not None
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
        from .prediction import JointEvaluator

        reserve = kwargs.pop("actor_memory_reserve_fraction", 0.0)
        kwargs.pop("actor_memory_scope", None)
        if reserve:
            raise ValueError("Basic v1 has no Actor memory-reserve policy")
        return JointEvaluator(*args, optimizer=self.optimizer, **kwargs)

    def bind(self, environment, clock=None):
        from .controller import OnlineControl

        return OnlineControl(environment, clock)

    def role_switched(self, role, when):
        """v1 has global handoff history only; per-role notifications are informational."""

    def role_history_status(self, when):
        return None
