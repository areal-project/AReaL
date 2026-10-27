# SPDX-License-Identifier: Apache-2.0

"""SLO-triggered joint search over causal observations and fixed serving windows."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Protocol

from .cost_model.rollout import nonnegative, positive_int
from .forecast import (
    Forecast,
    HistoricalForecaster,
    OnlineLayout,
    OnlineObservation,
    ReplayWindow,
)
from .history import HistoricalReplay, WorkloadFeatures
from .search import meets_target, select_prediction
from .serving import ServingSLO


@dataclass(frozen=True)
class OnlinePolicyConfig:
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


@dataclass(frozen=True)
class RollingSLO:
    completed: int
    pending: int
    decided: int
    jointly_passed: int
    fraction: float | None
    ttft_quantile_seconds: float
    tbt_quantile_seconds: float


def rolling_slo(view: OnlineObservation, slo: ServingSLO, seconds: float) -> RollingSLO:
    rows = [
        r
        for r in view.requests
        if r.completed_at is None or r.completed_at >= view.now - seconds
    ]
    complete = passed = failed_pending = 0
    ttfts, gaps = [], []
    for request in rows:
        first = request.first_token_at
        ttft = (first if first is not None else view.now) - request.ready_at
        gap = request.maximum_tbt_seconds
        if request.completed_at is None and request.last_token_at is not None:
            gap = max(gap, view.now - request.last_token_at)
        ttfts.append(ttft)
        gaps.append(gap)
        meets = ttft <= slo.ttft_seconds + 1e-12 and gap <= slo.tbt_seconds + 1e-12
        if request.completed_at is not None:
            complete += 1
            passed += meets
        else:
            failed_pending += not meets
    decided = complete + failed_pending

    def quantile(values):
        return (
            sorted(values)[
                max(0, math.ceil(len(values) * slo.minimum_request_fraction) - 1)
            ]
            if values
            else 0.0
        )

    return RollingSLO(
        complete,
        len(rows) - complete,
        decided,
        passed,
        passed / decided if decided else None,
        quantile(ttfts),
        quantile(gaps),
    )


@dataclass(frozen=True)
class ActorSupply:
    """Conditional target-only capacity; excludes old Actor work and idle time."""

    offered_episodes: int
    history_seconds: float
    arrival_rate: float
    actor_seconds_per_batch: float | None
    publication_seconds_per_batch: float | None
    capacity_rate: float | None
    credit_balance_rate: float | None
    sustainable: bool | None
    representative_batches: int
    padded_episodes: int
    substituted_episodes: int
    input_closed: bool
    reason: str


@dataclass(frozen=True)
class ActorMemorySafety:
    """Candidate-only margin; evaluation batches can contain replay templates."""

    admissible: bool
    reserve_fraction: float
    checked_batches: int
    peak_gib: float | None
    capacity_gib: float | None
    allowed_gib: float | None
    worst_batch: tuple[str, ...]
    observed_through: float
    reason: str
    scope: str = "all_completed"


@dataclass(frozen=True)
class CandidatePrediction:
    """Historical mean_gpu_seconds aliases score_gpu_seconds (K + H * GPUs).

    Request/update counts describe the serving window, not a completed workload.
    Legacy sampled forecasts retain their complete-work mean cost.
    """

    layout: OnlineLayout
    supported: bool
    worst_request_fraction: float | None
    mean_gpu_seconds: float | None
    work_ids: tuple[str, ...]
    reason: str = ""
    source_ids: tuple[str, ...] = ()
    window_gpu_seconds: float | None = None
    tail_gpu_seconds: float | None = None
    actor_coverage: str = "legacy_sampled_execution"
    pruned: bool = False
    fraction_upper_bound: float | None = None
    mean_cost_lower_bound: float | None = None
    pruning_incumbent: tuple[str, float, float] | None = None
    switching_gpu_seconds: float | None = None
    running_gpu_seconds_per_second: float | None = None
    evaluation_seconds: float | None = None
    running_gpu_seconds: float | None = None
    score_gpu_seconds: float | None = None
    completed_requests: int | None = None
    pending_requests: int | None = None
    failed_pending_requests: int | None = None
    completed_updates: int | None = None
    retained_episodes: int | None = None
    actor_supply: tuple[ActorSupply, ...] = ()
    admission_credits_start: int | None = None
    admission_credits_end: float | None = None
    admission_credit_blocked_seconds: float | None = None
    actor_memory: ActorMemorySafety | None = None


class OnlineModel(Protocol):
    def evaluate(
        self,
        view: OnlineObservation,
        forecasts: tuple[Forecast, ...],
        layout: OnlineLayout,
    ) -> CandidatePrediction: ...


@dataclass(frozen=True)
class OnlineDecision:
    observed_at: float
    trigger: str
    target: OnlineLayout
    predictions: tuple[CandidatePrediction, ...]
    forecast_seeds: tuple[int, ...]
    statistics: RollingSLO
    search_cpu_seconds: float
    reason: str
    replay_window: ReplayWindow | None = None
    workload_features: WorkloadFeatures | None = None
    net_window_gpu_seconds: float | None = None
    net_total_gpu_seconds: float | None = None


class OnlinePolicy:
    def __init__(
        self,
        candidates: tuple[OnlineLayout, ...],
        forecaster: HistoricalForecaster | HistoricalReplay,
        config: OnlinePolicyConfig,
        slo: ServingSLO,
    ):
        if not candidates:
            raise ValueError("Explicit online candidates are required")
        self.candidates = tuple(dict.fromkeys(candidates))
        self.forecaster, self.config, self.slo = forecaster, config, slo
        self.decision_index = 0
        self.last_switch = -math.inf
        self.last_role_switch = {"actor": -math.inf, "rollout": -math.inf}
        self.last_search = -math.inf
        self._last_search_state = None
        self._winners = []

    def search_invalidated(self):
        self._last_search_state = None

    def search_state(self, view, model):
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
            getattr(model, "source_ids", ()),
            view.actor.waiting,
            self.forecaster.regime_index,
            self.forecaster.features,
            (
                len(active),
                sum(max(0, r.input_tokens - r.prefilled) for r in active),
                sum(t.output_tokens for t in recent) / max(1, len(recent)),
                sum(t.tool_seconds or 0 for t in recent) / max(1, len(recent)),
                getattr(model, "delay", 0.0),
                statistics.ttft_quantile_seconds,
                statistics.tbt_quantile_seconds,
            ),
            (
                statistics.fraction is not None
                and statistics.fraction >= self.slo.minimum_request_fraction,
                self._trigger(statistics),
            ),
        )

    def redundant_search(self, view, model):
        if self.config.independent_role_history and self.allowed_roles(
            view.now
        ) != self.allowed_roles(self.last_search):
            return False
        old = self._last_search_state
        state = self.search_state(view, model)
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

    def role_switched(self, role: str, when: float):
        """One changed side has taken over and its old deployment is released."""
        self.last_role_switch[role] = when
        if self.config.independent_role_history:
            self.search_invalidated()

    def allowed_roles(self, when: float) -> dict[str, bool]:
        seconds = max(
            self.config.cooldown_seconds,
            self.forecaster.config.history_seconds
            if isinstance(self.forecaster, HistoricalReplay)
            else 0.0,
        )
        return {
            role: when - max(0.0, completed) >= seconds
            for role, completed in self.last_role_switch.items()
        }

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

    def decide(self, view: OnlineObservation, model: OnlineModel) -> OnlineDecision:
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

        roles = self.allowed_roles(view.now) if cfg.independent_role_history else None
        if roles is not None and not any(roles.values()):
            return keep("Waiting for each role's own full history window")
        if (
            not cfg.independent_role_history
            and historical
            and view.now - max(0.0, self.last_switch)
            < self.forecaster.config.history_seconds
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
            and self.redundant_search(view, model)
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
            self._last_search_state = self.search_state(view, model)
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
        if view.actor.lifecycle is not None and view.actor.lifecycle.state == "stopped":
            candidates = tuple(
                c
                for c in candidates
                if (c.actor_dp, c.actor_tp)
                == (view.layout.actor_dp, view.layout.actor_tp)
            )
        layouts = candidates[: cfg.max_evaluations]
        if cfg.independent_role_history and len(layouts) == 1:
            return keep(
                "No alternative layout under role history and lifecycle constraints",
                trigger,
            )
        evaluate_many = getattr(model, "evaluate_many", None)
        search = getattr(model, "search", None)
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
                else tuple(model.evaluate(view, forecasts, c) for c in layouts)
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
            best = select_prediction(
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
