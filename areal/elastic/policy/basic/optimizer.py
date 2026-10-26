# SPDX-License-Identifier: Apache-2.0

"""Bounded exhaustive search over resource ownership and arbitrary GPU counts."""

import itertools
import math
import time
from dataclasses import dataclass
from typing import Literal, Protocol

from ...contracts import (
    ElasticObservation,
    ResourceDecision,
    SearchResult,
    TrainingConfig,
)
from .cost_model.rollout import nonnegative, positive_int
from .joint import (
    JointCandidateResult,
    JointConfig,
    JointPrediction,
    JointSearchResult,
    JointWindow,
)
from .serving import ServingSLO
from .simulator import (
    HistorySnapshot,
    PlanStep,
    SimulationResult,
    WorkflowSimulator,
    forecast_window,
)


class Optimizer(Protocol):
    def optimize(
        self,
        observation: ElasticObservation,
        simulator: WorkflowSimulator,
        history: HistorySnapshot,
    ) -> SearchResult: ...


class JointModel(Protocol):
    """Prediction boundary for later model revisions and independent evaluation."""

    def evaluate(
        self, scenario: JointWindow, config: JointConfig
    ) -> JointPrediction: ...


@dataclass(frozen=True)
class JointOptimizer:
    """Enumerate fixed joint configurations on an identical finite window.

    Budget limits candidate evaluations, never workload or horizon. The current
    configuration is evaluated first. Costs include transitions and cleanup;
    max_seconds bounds predicted job time, not search runtime.
    """

    candidates: tuple[JointConfig, ...]
    max_evaluations: int = 128
    max_seconds: float | None = None
    switch_margin: float = 0.05
    serving_slo: ServingSLO | None = None
    expected_serving_requests: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(dict.fromkeys(self.candidates)))
        if not self.candidates:
            raise ValueError("Joint search requires explicit candidate configurations")
        positive_int("max_evaluations", self.max_evaluations)
        nonnegative("switch_margin", self.switch_margin)
        if self.switch_margin >= 1:
            raise ValueError("Switch margin must be in [0, 1)")
        if self.max_seconds is not None:
            nonnegative("max_seconds", self.max_seconds, positive=True)
        if self.serving_slo is not None:
            positive_int("expected_serving_requests", self.expected_serving_requests)
        elif self.expected_serving_requests is not None:
            raise ValueError("expected_serving_requests requires serving_slo")
        identities = {
            (c.rollout.model_id, c.rollout.hardware_id, c.rollout.sampling_id)
            for c in self.candidates
        }
        if len(identities) != 1:
            raise ValueError(
                "Candidates must share model, hardware cost basis and sampling semantics"
            )

    def optimize(
        self, scenario: JointWindow, simulator: JointModel
    ) -> JointSearchResult:
        started = time.perf_counter()
        configs = tuple(
            dict.fromkeys(
                (() if scenario.current is None else (scenario.current,))
                + self.candidates
            )
        )
        expected = (
            len(scenario.batches),
            sum(b.actor.samples for b in scenario.batches),
            sum(b.actor.effective_tokens for b in scenario.batches),
            sum(b.actor.sequence_tokens for b in scenario.batches),
        )
        evaluated = []
        for config in configs[: self.max_evaluations]:
            prediction = simulator.evaluate(scenario, config)
            reason = ""
            if prediction.config != config:
                reason = "Prediction returned a different configuration"
            elif prediction.status != "complete":
                reason = prediction.reason or prediction.status
            elif (
                prediction.completed_batches,
                prediction.samples,
                prediction.effective_tokens,
                prediction.sequence_tokens,
            ) != expected:
                reason = "Prediction did not complete the identical workload window"
            elif self.max_seconds is not None and prediction.seconds > self.max_seconds:
                reason = "Prediction exceeds job time bound"
            elif self.serving_slo is not None:
                reason = self.serving_slo.rejection(
                    prediction.serving, self.expected_serving_requests
                )
            evaluated.append(
                JointCandidateResult(config, prediction, not reason, reason)
            )
        eligible = [item for item in evaluated if item.eligible]
        best = min(
            eligible,
            key=lambda item: (
                item.prediction.total_gpu_seconds,
                item.config != scenario.current,
                item.prediction.seconds,
            ),
            default=None,
        )
        incumbent = next(
            (item for item in eligible if item.config == scenario.current), None
        )
        reason = "No candidate has a supported complete prediction within constraints"
        if best is not None:
            reason = "Lowest total GPU seconds among eligible evaluated candidates"
            if (
                incumbent is not None
                and best.config != incumbent.config
                and (
                    best.prediction.total_gpu_seconds
                    >= incumbent.prediction.total_gpu_seconds * (1 - self.switch_margin)
                )
            ):
                best = incumbent
                reason = "Keep current configuration: improvement does not exceed switch margin"
        return JointSearchResult(
            None if best is None else best.config,
            None if best is None else best.prediction,
            tuple(evaluated),
            len(configs) - len(evaluated),
            time.perf_counter() - started,
            reason,
        )


@dataclass(frozen=True)
class BasicOptimizer:
    horizon: int = 2
    cost_scope: Literal["total", "training"] = "total"
    switch_margin: float = 0.05
    max_candidates: int = 100_000
    max_seconds: float | None = None

    candidate_configs: tuple[TrainingConfig, ...] | None = None

    def __post_init__(self) -> None:
        if self.candidate_configs is not None:
            configs = tuple(
                TrainingConfig(**value) if isinstance(value, dict) else value
                for value in self.candidate_configs
            )
            if (
                not configs
                or any(not isinstance(value, TrainingConfig) for value in configs)
                or len(set(configs)) != len(configs)
            ):
                raise ValueError("Candidate configurations must be explicit and unique")
            object.__setattr__(self, "candidate_configs", configs)
        if type(self.horizon) is not int or self.horizon < 1:
            raise ValueError("Horizon must be a positive integer")
        if self.cost_scope not in {"total", "training"}:
            raise ValueError("Unknown GPU cost scope")
        if not math.isfinite(self.switch_margin) or not 0 <= self.switch_margin < 1:
            raise ValueError("Switch margin must be in [0, 1)")
        if self.max_candidates < 1:
            raise ValueError("Search budget must be positive")
        if self.max_seconds is not None and (
            not math.isfinite(self.max_seconds) or self.max_seconds <= 0
        ):
            raise ValueError("Time bound must be finite and positive")

    def optimize(
        self,
        observation: ElasticObservation,
        simulator: WorkflowSimulator,
        history: HistorySnapshot,
    ) -> SearchResult:
        if observation.pending_publication or not observation.committed:
            return SearchResult(
                ResourceDecision("keep", reason="Uncommitted work or publication"),
                None,
                0,
            )
        forecast = forecast_window(observation, history, self.horizon)
        if not forecast:
            return SearchResult(
                ResourceDecision("keep", reason="No workload history"),
                None,
                0,
                fallback_allowed=True,
            )
        configs = self.candidate_configs or tuple(
            TrainingConfig(n) for n in range(1, observation.max_train_gpus + 1)
        )
        if any(config.world_size > observation.max_train_gpus for config in configs):
            raise ValueError("Candidate configuration exceeds the assigned GPU pool")
        # Include every card count even when search must shorten its lookahead.
        depth = len(forecast)
        while depth > 1 and (2 * len(configs)) ** depth > self.max_candidates:
            depth -= 1
        if not configs or 2 * len(configs) > self.max_candidates:
            raise ValueError("Search budget cannot cover the GPU pool")
        forecast = forecast[:depth]
        options = tuple(
            PlanStep(config, release) for config in configs for release in (False, True)
        )
        candidates = []
        rejected = set()
        evaluated = 0
        fallback_allowed = True
        for plan in itertools.product(options, repeat=depth):
            evaluated += 1
            prediction = simulator.evaluate(observation, forecast, plan, history)
            if not prediction.feasible:
                fallback_allowed = fallback_allowed and prediction.unmeasured
                rejected.add(prediction.reason)
                continue
            if self.max_seconds is not None and prediction.seconds > self.max_seconds:
                fallback_allowed = False
                rejected.add("Forecast exceeds time bound")
                continue
            cost = (
                prediction.total_gpu_seconds
                if self.cost_scope == "total"
                else prediction.training_gpu_seconds
            )
            candidates.append((cost, plan, prediction))
        if not candidates:
            return SearchResult(
                ResourceDecision("keep", reason="No feasible measured plan"),
                None,
                evaluated,
                tuple(sorted(rejected)),
                fallback_allowed=fallback_allowed,
            )
        candidates.sort(
            key=lambda x: (
                x[0],
                sum(s.release_before for s in x[1]),
                tuple(s.config.world_size for s in x[1]),
            )
        )
        best = candidates[0]
        if observation.training_active:
            steady = [
                x
                for x in candidates
                if x[1][0].config == observation.training_config
                and not x[1][0].release_before
            ]
            if steady and best[0] >= steady[0][0] * (1 - self.switch_margin):
                best = steady[0]
        _, plan, prediction = best
        first = plan[0]
        ready = observation.batch_ready and observation.gate_available
        if (
            observation.training_active
            and first.release_before
            and not observation.batch_ready
        ):
            decision = ResourceDecision(
                "release", reason="Release before predicted workload"
            )
        elif (
            observation.training_active and first.config != observation.training_config
        ):
            if observation.batch_ready and observation.gate_available:
                decision = ResourceDecision(
                    "reconfigure", first.config, True, "Lower forecast GPU cost"
                )
            else:
                decision = ResourceDecision(
                    "keep",
                    observation.training_config,
                    reason="Wait for ready data before configuration change",
                )
        elif not observation.training_active:
            # Initial version starts on ready data; prewarming can be added later.
            decision = (
                ResourceDecision(
                    "launch", first.config, True, "Lowest measured forecast cost"
                )
                if ready
                else ResourceDecision("keep", reason="Waiting for data")
            )
        else:
            decision = ResourceDecision(
                "keep", first.config, ready, "Retain current configuration"
            )
        return SearchResult(
            decision, prediction, evaluated, tuple(sorted(rejected)), plan
        )


@dataclass(frozen=True)
class _Prefix:
    plan: tuple[PlanStep, ...]
    prediction: SimulationResult
    retained_first: bool = False


@dataclass(frozen=True)
class LifecycleOptimizer:
    """Finite-horizon lifecycle/config search using simulator transition costs.

    Dynamic programming keeps a cost/time Pareto frontier for each live profile
    and first-action retention flag. Candidates consume the same forecast data;
    all complete plans end suspended. No future trace, reward oracle or profiling.
    """

    configurations: tuple[TrainingConfig, ...]
    horizon: int = 3
    cost_scope: Literal["total", "training"] = "total"
    switch_margin: float = 0.05
    max_evaluations: int = 100_000
    max_seconds: float | None = None

    def __post_init__(self) -> None:
        configs = tuple(dict.fromkeys(self.configurations))
        if not configs or any(
            not isinstance(config, TrainingConfig) for config in configs
        ):
            raise ValueError(
                "Lifecycle optimizer requires explicit TrainingConfig candidates"
            )
        object.__setattr__(self, "configurations", configs)
        BasicOptimizer(
            self.horizon,
            self.cost_scope,
            self.switch_margin,
            self.max_evaluations,
            self.max_seconds,
        )
        if type(self.max_evaluations) is not int:
            raise ValueError("Evaluation budget must be an integer")

    def _cost(self, prediction: SimulationResult) -> float:
        return (
            prediction.total_gpu_seconds
            if self.cost_scope == "total"
            else prediction.training_gpu_seconds
        )

    def _rank(self, prefix: _Prefix):
        return (
            self._cost(prefix.prediction),
            prefix.prediction.seconds,
            sum(step.release_before for step in prefix.plan),
            tuple(
                (step.config.world_size, step.config.profile_id) for step in prefix.plan
            ),
        )

    def _frontier(self, prefixes: list[_Prefix]) -> list[_Prefix]:
        groups = {}
        for prefix in sorted(prefixes, key=self._rank):
            state = prefix.plan[-1].config, prefix.retained_first
            frontier = groups.setdefault(state, [])
            cost, seconds = self._cost(prefix.prediction), prefix.prediction.seconds
            if any(
                self._cost(other.prediction) <= cost
                and (self.max_seconds is None or other.prediction.seconds <= seconds)
                for other in frontier
            ):
                continue
            frontier[:] = [
                other
                for other in frontier
                if not (
                    cost <= self._cost(other.prediction)
                    and (
                        self.max_seconds is None or seconds <= other.prediction.seconds
                    )
                )
            ]
            frontier.append(prefix)
        return [prefix for frontier in groups.values() for prefix in frontier]

    def optimize(
        self,
        observation: ElasticObservation,
        simulator: WorkflowSimulator,
        history: HistorySnapshot,
    ) -> SearchResult:
        from dataclasses import replace

        def stopped(reason, evaluated=0, rejected=()):
            return SearchResult(
                ResourceDecision("keep", reason=reason),
                None,
                evaluated,
                tuple(sorted(rejected)),
            )

        if observation.pending_publication or not observation.committed:
            return stopped("Uncommitted work or publication")
        if observation.training_active and observation.training_config is None:
            return stopped("Active training configuration is unknown")
        if observation.batch_ready and not observation.gate_available:
            return stopped("Ready batch waits for admission gate")
        if observation.batch_ready and observation.next_batch is None:
            return stopped("Ready batch workload is unavailable")
        forecast = forecast_window(
            observation, history, self.horizon, preserve_workload=True
        )
        if not forecast:
            return stopped("No completed workload history or ready batch")
        configs = tuple(
            config
            for config in self.configurations
            if config.world_size <= observation.max_train_gpus
        )
        if not configs:
            return stopped("No candidate fits the training GPU pool")
        count = len(configs)
        if self.max_evaluations < 2 * count:
            raise ValueError("Evaluation budget cannot cover every lifecycle candidate")
        depth = len(forecast)
        # Without deadline frontiers there are at most two first-action tags/profile.
        # Deadline Pareto frontiers can grow; an exhausted search returns no decision.
        while (
            depth > 1
            and 2 * count + (depth - 1) * 4 * count * count > self.max_evaluations
        ):
            depth -= 1
        forecast = forecast[:depth]
        frontier = [_Prefix((), SimulationResult(True))]
        cache, rejected = {}, set()
        evaluated = 0
        for index, future in enumerate(forecast):
            candidates = []
            for prefix in frontier:
                state = observation
                if index:
                    current = prefix.plan[-1].config
                    state = replace(
                        observation,
                        training_active=True,
                        training_config=current,
                        has_checkpoint=True,
                        checkpoint_config=current,
                    )
                for target in configs:
                    for release in (False, True):
                        step = PlanStep(target, release)
                        key = (
                            index,
                            state.training_active,
                            state.training_config,
                            state.has_checkpoint,
                            state.checkpoint_config,
                            target,
                            release,
                        )
                        if key not in cache:
                            if evaluated >= self.max_evaluations:
                                return stopped(
                                    "Search budget exhausted before completing the horizon",
                                    evaluated,
                                    rejected,
                                )
                            cache[key] = simulator.evaluate(
                                state,
                                (future,),
                                (step,),
                                history,
                                terminal_release=index == depth - 1,
                            )
                            evaluated += 1
                        edge = cache[key]
                        if not edge.feasible:
                            rejected.add(edge.reason)
                            continue
                        before = prefix.prediction
                        prediction = SimulationResult(
                            True,
                            before.seconds + edge.seconds,
                            before.training_gpu_seconds + edge.training_gpu_seconds,
                            before.inference_gpu_seconds + edge.inference_gpu_seconds,
                            before.extra_gpu_seconds + edge.extra_gpu_seconds,
                            before.samples + edge.samples,
                            before.effective_tokens + edge.effective_tokens,
                            events=before.events + edge.events,
                            profile_events=before.profile_events + edge.profile_events,
                        )
                        if (
                            self.max_seconds is not None
                            and prediction.seconds > self.max_seconds
                        ):
                            rejected.add("Forecast exceeds time bound")
                            continue
                        retained = (
                            prefix.retained_first
                            if index
                            else (
                                observation.training_active
                                and target == observation.training_config
                                and not release
                            )
                        )
                        candidates.append(
                            _Prefix(prefix.plan + (step,), prediction, retained)
                        )
            if not candidates:
                return stopped("No feasible lifecycle plan", evaluated, rejected)
            frontier = self._frontier(candidates)
        best = min(frontier, key=self._rank)
        steady = [prefix for prefix in frontier if prefix.retained_first]
        if steady:
            retained = min(steady, key=self._rank)
            if self._cost(best.prediction) >= self._cost(retained.prediction) * (
                1 - self.switch_margin
            ):
                best = retained
        first = best.plan[0]
        ready = observation.batch_ready and observation.gate_available
        if (
            observation.training_active
            and first.release_before
            and not observation.batch_ready
        ):
            decision = ResourceDecision(
                "release", reason="Release during predicted idle interval"
            )
        elif (
            observation.training_active and first.config != observation.training_config
        ):
            decision = (
                ResourceDecision(
                    "reconfigure", first.config, True, "Lower lifecycle GPU cost"
                )
                if ready
                else ResourceDecision(
                    "keep",
                    observation.training_config,
                    reason="Wait for ready data before configuration change",
                )
            )
        elif not observation.training_active:
            decision = (
                ResourceDecision(
                    "launch", first.config, True, "Lowest lifecycle GPU cost"
                )
                if ready
                else ResourceDecision("keep", reason="Waiting for data")
            )
        else:
            decision = ResourceDecision(
                "keep", first.config, ready, "Retain current profile"
            )
        return SearchResult(
            decision, best.prediction, evaluated, tuple(sorted(rejected)), best.plan
        )
