# SPDX-License-Identifier: Apache-2.0

"""Evaluate Optimizer revisions with parent-owned configuration predictions."""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from areal.autopilot.domains.elastic.contracts import OnlineProblem, decode
from areal.autopilot.domains.elastic.cycle import (
    ElasticRevisionBenchmark,
    candidate_runner,
)
from areal.autopilot.domains.elastic.versions import AREAL_ROOT, protected_files
from areal.autopilot.revision.contracts import RevisionReport, digest, source_digest
from areal.autopilot.revision.execution import load_source
from areal.elastic.reconfigurator.reference.optimizer import (
    OnlineOptimizer,
    SearchBound,
    credit_deficit,
    meets_target,
    select_prediction,
)


@dataclass(frozen=True)
class OptimizerRevisionCriteria:
    min_relative_improvement: float = 0.005
    max_cost_regression: float = 0.0

    def __post_init__(self):
        if (
            not 0 <= self.min_relative_improvement < 1
            or not math.isfinite(self.max_cost_regression)
            or self.max_cost_regression < 0
        ):
            raise ValueError("Invalid Online optimizer acceptance thresholds")


class SearchPort:
    """Restrict search to one immutable problem and Evaluator-produced predictions."""

    def __init__(self, evaluator, problem, *, max_queries, max_seconds):
        self.evaluator, self.problem = evaluator, problem
        self.max_queries, self.max_seconds = max_queries, max_seconds
        self.queries = 0
        self.issued = []
        self.violations = []
        self.started = time.monotonic()
        self.evaluator.prepare_context(problem.observation, problem.forecasts)

    def reject(self, reason):
        self.violations.append(reason)
        raise ValueError(reason)

    def check(self, view, forecasts):
        if view != self.problem.observation or forecasts != self.problem.forecasts:
            self.reject("Search changed its frozen observation or forecasts")
        if time.monotonic() - self.started > self.max_seconds:
            self.reject("Search exceeded its wall-time budget")

    def prepare_context(self, view, forecasts):
        self.check(view, forecasts)

    @property
    def historical(self):
        return self.evaluator.historical

    @property
    def context_id(self):
        return self.evaluator.context_id

    @property
    def slo(self):
        return self.evaluator.slo

    @property
    def parallel_available(self):
        return self.evaluator.parallel_available

    @property
    def execution_stats(self):
        return self.evaluator.execution_stats

    @execution_stats.setter
    def execution_stats(self, value):
        # Candidate-reported work statistics are never used by acceptance.
        pass

    def order_candidates(self, view, forecasts, layouts, preferred=()):
        self.check(view, forecasts)
        if tuple(layouts) != self.problem.candidates or not set(preferred) <= set(
            layouts
        ):
            self.reject("Search changed its candidate catalog")
        return self.evaluator.order_candidates(view, forecasts, layouts, preferred)

    def evaluate_wave(self, view, forecasts, indexed, *, bound=None, chunk_size=2):
        self.check(view, forecasts)
        indexed = tuple(indexed)
        if any(c not in self.problem.candidates for _, c in indexed):
            self.reject("Search queried a layout outside the frozen catalog")
        if self.queries + len(indexed) > self.max_queries:
            self.reject("Search exceeded its Evaluator-query budget")
        if bound is not None:
            if (
                bound.best not in self.issued
                or bound.target != self.slo.minimum_request_fraction
            ):
                self.reject(
                    "Pruning requires an Evaluator-issued incumbent and the fixed SLO target"
                )
            bound = SearchBound(
                self.context_id, bound.target, bound.best, bound.fraction_pruning
            )
        self.queries += len(indexed)
        result = self.evaluator.evaluate_wave(
            view, forecasts, indexed, bound=bound, chunk_size=chunk_size
        )
        self.issued.extend(result.predictions)
        self.check(view, forecasts)
        return result

    def evaluate(self, view, forecasts, layout, *, bound=None):
        return self.evaluate_wave(
            view, forecasts, ((0, layout),), bound=bound
        ).predictions[0]

    def evaluate_many(self, view, forecasts, layouts):
        return self.evaluate_wave(
            view, forecasts, tuple(enumerate(layouts)), chunk_size=8
        ).predictions

    def finish(self, predictions):
        predictions = tuple(predictions)
        if self.violations:
            raise ValueError("; ".join(self.violations))
        if not predictions or len({p.layout for p in predictions}) != len(predictions):
            raise ValueError("Search must return unique Evaluator-issued predictions")
        if any(
            p not in self.issued for p in predictions
        ) or self.problem.observation.layout not in {p.layout for p in predictions}:
            raise ValueError(
                "Search omitted the current configuration or fabricated predictions"
            )
        self.check(self.problem.observation, self.problem.forecasts)
        return predictions


class VerifiedOnlineOptimizer:
    """Deploy the evaluated search factory with the same port and fixed selection rule."""

    select = staticmethod(select_prediction)

    def __init__(
        self,
        source,
        settings=None,
        *,
        max_queries=64,
        max_seconds=30,
        runner=None,
        directory=None,
    ):
        self.source = source
        self.settings = copy.deepcopy(settings or {})
        self.runner = candidate_runner(runner) if runner is not None else None
        self.directory = Path(directory) if directory is not None else None
        if self.runner is not None and self.directory is None:
            raise ValueError("Subprocess Optimizer searches need an output directory")
        if self.runner is None:
            self.searcher = load_source(source).build_optimizer(self.settings)
            if not callable(getattr(self.searcher, "search", None)):
                raise ValueError(
                    "build_optimizer must return an Online search implementation"
                )
        self.max_queries, self.max_seconds = max_queries, max_seconds
        self.searches = self.query_count = 0
        self.search_wall_seconds = 0.0

    def search(
        self, evaluator, view, forecasts, layouts, *, preferred=(), trigger="pressure"
    ):
        started = time.perf_counter()
        problem = OnlineProblem(
            view, tuple(forecasts), tuple(layouts), tuple(preferred), trigger
        )
        port = SearchPort(
            evaluator,
            problem,
            max_queries=self.max_queries,
            max_seconds=self.max_seconds,
        )
        self.searches += 1
        if self.runner is None:
            searcher = load_source(self.source).build_optimizer(
                copy.deepcopy(self.settings)
            )
            predictions = port.finish(
                searcher.search(
                    port, view, forecasts, layouts, preferred=preferred, trigger=trigger
                )
            )
        else:
            from areal.autopilot.domains.elastic.evaluation.worker import (
                run_optimizer_search,
            )

            predictions = run_optimizer_search(
                self.source,
                (port,),
                self.settings,
                self.runner,
                self.directory / f"search-{self.searches:04d}",
            )[0]
        self.query_count += port.queries
        self.search_wall_seconds += time.perf_counter() - started
        evaluator.execution_stats = dict(
            mode="verified_online_revision",
            model_queries=port.queries,
            source_sha256=source_digest(self.source),
        )
        return predictions

    def evaluate_many(self, evaluator, view, forecasts, layouts):
        # Application revalidation evaluates the supplied layouts directly.
        layouts = tuple(layouts)
        self.query_count += len(layouts)
        return OnlineOptimizer().evaluate_many(evaluator, view, forecasts, layouts)


class OptimizerRevisionBenchmark(ElasticRevisionBenchmark):
    target = "optimizer"

    def __init__(
        self,
        cases,
        *,
        evaluator_factory,
        context,
        settings=None,
        criteria=None,
        runner=None,
        max_queries=64,
        max_seconds=30,
        overhead_weight=0.05,
    ):
        if (
            type(max_queries) is not int
            or max_queries < 1
            or not math.isfinite(max_seconds)
            or max_seconds <= 0
        ):
            raise ValueError("Online search requires positive query and time budgets")
        if not math.isfinite(overhead_weight) or overhead_weight < 0:
            raise ValueError("Search overhead weight must be nonnegative")
        self.cases = tuple(
            copy.deepcopy(decode(OnlineProblem, c) if isinstance(c, dict) else c)
            for c in cases
        )
        if not self.cases:
            raise ValueError("Freeze at least one Online optimization problem")
        self.evaluator_factory = evaluator_factory
        self.context = copy.deepcopy(context)
        self.settings = copy.deepcopy(settings or {})
        self.criteria = criteria or OptimizerRevisionCriteria()
        self.runner = candidate_runner(runner)
        self.max_queries, self.max_seconds, self.overhead_weight = (
            max_queries,
            max_seconds,
            overhead_weight,
        )
        self.context.update(
            optimizer_settings=self.settings,
            optimizer_limits=dict(max_queries=max_queries, max_seconds=max_seconds),
        )
        self._files = protected_files()
        self._code = {
            str(p.relative_to(AREAL_ROOT)): source_digest(p.read_text())
            for p in self._files
        }
        self.benchmark_sha256 = digest(
            dict(
                cases=[asdict(c) for c in self.cases],
                context=self.context,
                settings=self.settings,
                criteria=asdict(self.criteria),
                code=self._code,
                limits=(max_queries, max_seconds, overhead_weight),
            )
        )
        self._reports = set()
        self._oracle = []
        for case in self.cases:
            evaluator = self.evaluator_factory()
            predictions = tuple(
                evaluator.evaluate(case.observation, case.forecasts, c)
                for c in case.candidates
            )
            chosen = select_prediction(
                predictions,
                case.observation.layout,
                evaluator.slo.minimum_request_fraction,
                trigger=case.trigger,
            )
            if chosen is None:
                raise ValueError("Each Online case needs a runnable reference choice")
            self._oracle.append(chosen)

    @property
    def calibration(self):
        return {}

    def assert_unchanged(self):
        if any(
            not (AREAL_ROOT / p).is_file()
            or source_digest((AREAL_ROOT / p).read_text()) != h
            for p, h in self._code.items()
        ):
            raise ValueError("Online evaluation/runtime code changed")

    def evaluate(self, source, directory):
        errors, rows = [], []
        ports = [
            SearchPort(
                self.evaluator_factory(),
                c,
                max_queries=self.max_queries,
                max_seconds=self.max_seconds,
            )
            for c in self.cases
        ]
        try:
            self.assert_unchanged()
            from areal.autopilot.domains.elastic.evaluation.worker import (
                run_optimizer_search,
            )

            results = run_optimizer_search(
                source, ports, self.settings, self.runner, directory
            )
            for index, (case, predictions) in enumerate(
                zip(self.cases, results, strict=True)
            ):
                port = ports[index]
                selected = select_prediction(
                    predictions,
                    case.observation.layout,
                    port.slo.minimum_request_fraction,
                    trigger=case.trigger,
                )
                oracle = self._oracle[index]
                if selected is None:
                    raise ValueError("Search returned no runnable layout")
                # Cost/feasibility are recomputed without candidate pruning or self-reported values.
                actual = self.evaluator_factory().evaluate(
                    case.observation, case.forecasts, selected.layout
                )
                if not actual.supported or actual.mean_gpu_seconds is None:
                    raise ValueError("Selected layout is unsupported")
                if meets_target(
                    oracle, port.slo.minimum_request_fraction
                ) and not meets_target(actual, port.slo.minimum_request_fraction):
                    raise ValueError(
                        "Search missed the feasible SLO/Actor-supply target"
                    )
                if not meets_target(oracle, port.slo.minimum_request_fraction) and (
                    actual.worst_request_fraction + 1e-12
                    < oracle.worst_request_fraction
                ):
                    raise ValueError("Search reduced best attainable request SLO")
                if (
                    case.trigger == "pressure"
                    and oracle.layout != case.observation.layout
                    and meets_target(oracle, port.slo.minimum_request_fraction)
                    and actual.layout == case.observation.layout
                ):
                    raise ValueError(
                        "Pressure search must retain an eligible alternative when one exists"
                    )
                if (
                    not meets_target(oracle, port.slo.minimum_request_fraction)
                    and abs(
                        actual.worst_request_fraction - oracle.worst_request_fraction
                    )
                    <= 1e-12
                    and credit_deficit(actual) > credit_deficit(oracle) + 1e-12
                ):
                    raise ValueError(
                        "Search increased the Actor-supply deficit at the same SLO"
                    )
                rows.append(
                    dict(
                        layout=asdict(actual.layout),
                        cost=actual.mean_gpu_seconds,
                        relative_cost=actual.mean_gpu_seconds
                        / max(oracle.mean_gpu_seconds, 1e-12),
                        queries=port.queries,
                    )
                )
            self.assert_unchanged()
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")

        def mean(key):
            return sum(r[key] for r in rows) / len(rows) if rows else None

        metrics = dict(
            mean_gpu_seconds=mean("cost"),
            mean_evaluator_queries=mean("queries"),
            selection_sha256=digest([r["layout"] for r in rows]),
        )
        score = (
            None
            if errors
            else mean("relative_cost")
            + self.overhead_weight * mean("queries") / self.max_queries
        )
        report = RevisionReport(
            self.target,
            source_digest(source),
            self.benchmark_sha256,
            not errors,
            score,
            metrics,
            tuple(errors),
        )
        self._reports.add(digest(asdict(report)))
        return report

    def compare(self, candidate, incumbent):
        self.assert_unchanged()
        if any(digest(asdict(r)) not in self._reports for r in (candidate, incumbent)):
            raise ValueError("Comparison requires issued Online reports")
        if not incumbent.valid:
            raise ValueError("Incumbent search must be valid")
        if not candidate.valid:
            return False, "Candidate search failed"
        if (
            candidate.metrics["mean_gpu_seconds"]
            > incumbent.metrics["mean_gpu_seconds"]
            * (1 + self.criteria.max_cost_regression)
            + 1e-9
        ):
            return False, "Reference GPU cost regressed"
        improved = incumbent.score - candidate.score > max(
            1e-12, abs(incumbent.score) * self.criteria.min_relative_improvement
        )
        return (
            improved,
            "Improved Online cost/query score"
            if improved
            else "Insufficient Online improvement",
        )

    def feedback(self, report):
        if digest(asdict(report)) not in self._reports:
            raise ValueError("Unknown Online report")
        return dict(
            score=report.score,
            metrics={
                k: v for k, v in report.metrics.items() if k != "selection_sha256"
            },
            errors=report.errors,
        )


@dataclass(frozen=True)
class OptimizerSimulationCriteria:
    """Preserve service and training, then improve cost or tied-cost queries."""

    cost_relative_tolerance: float = 1e-9
    slo_tolerance: float = 1e-12

    def __post_init__(self):
        if any(
            not math.isfinite(value) or value < 0
            for value in (self.cost_relative_tolerance, self.slo_tolerance)
        ):
            raise ValueError("Simulation comparison tolerances must be nonnegative")


class OptimizerSimulationBenchmark(ElasticRevisionBenchmark):
    """Evaluate complete reconfiguration runs on frozen, observation-derived work."""

    target = "optimizer"

    def __init__(
        self,
        cases,
        *,
        assembly,
        cost_models,
        prediction_cost_models=None,
        settings=None,
        criteria=None,
        runner=None,
        max_queries=64,
        max_seconds=30,
    ):
        from areal.autopilot.domains.elastic.contracts import OptimizerSimulationCase

        self.cases = tuple(
            copy.deepcopy(
                decode(OptimizerSimulationCase, case)
                if isinstance(case, dict)
                else case
            )
            for case in cases
        )
        if not self.cases:
            raise ValueError("Freeze at least one complete Simulation case")
        if (
            type(max_queries) is not int
            or max_queries < 1
            or not math.isfinite(max_seconds)
            or max_seconds <= 0
        ):
            raise ValueError("Optimizer requires positive query and time budgets")
        self.assembly = json.loads(json.dumps(assembly, allow_nan=False))
        self.cost_models = cost_models
        self.prediction_cost_models = prediction_cost_models or cost_models
        self.settings = copy.deepcopy(settings or {})
        self.criteria = criteria or OptimizerSimulationCriteria()
        self.runner = candidate_runner(runner)
        self.max_queries, self.max_seconds = max_queries, max_seconds
        self.context = dict(
            evaluation="complete_simulation",
            assembly=self.assembly,
            cost_model_source_ids=tuple(cost_models.source_ids),
            prediction_source_ids=tuple(self.prediction_cost_models.source_ids),
            optimizer_settings=self.settings,
            optimizer_limits=dict(max_queries=max_queries, max_seconds=max_seconds),
        )
        self.context = json.loads(json.dumps(self.context, allow_nan=False))
        self._code = {
            str(path.relative_to(AREAL_ROOT)): source_digest(path.read_text())
            for path in protected_files()
        }
        self.benchmark_sha256 = digest(
            dict(
                cases=[asdict(case) for case in self.cases],
                context=self.context,
                criteria=asdict(self.criteria),
                code=self._code,
            )
        )
        self._reports = set()

    @property
    def calibration(self):
        return {}

    def assert_unchanged(self):
        if any(
            not (AREAL_ROOT / name).is_file()
            or source_digest((AREAL_ROOT / name).read_text()) != expected
            for name, expected in self._code.items()
        ):
            raise ValueError("Simulation evaluation code changed")
        if tuple(self.cost_models.source_ids) != tuple(
            self.context["cost_model_source_ids"]
        ) or tuple(self.prediction_cost_models.source_ids) != tuple(
            self.context["prediction_source_ids"]
        ):
            raise ValueError("Frozen Simulation model identities changed")

    def evaluate(self, source, directory):
        from areal.elastic.contracts import (
            HandoffConfig,
            OnlineLayout,
            SimulationConfig,
        )
        from areal.elastic.reconfigurator import build_reconfigurator
        from areal.elastic.serving import ServingSLO, request_slo_status
        from areal.elastic.simulation.simulation import OnlineSimulation

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "candidate.py").write_text(source)
        errors, rows = [], []
        try:
            self.assert_unchanged()
            compile(source, str(directory / "candidate.py"), "exec")
            run = decode(SimulationConfig, self.assembly["run"])
            handoff = decode(HandoffConfig, self.assembly["handoff"])
            slo = decode(ServingSLO, self.assembly["slo"])
            candidates = tuple(
                decode(OnlineLayout, value) for value in self.assembly["candidates"]
            )
            for index, case in enumerate(self.cases):
                optimizer = VerifiedOnlineOptimizer(
                    source,
                    self.settings,
                    max_queries=self.max_queries,
                    max_seconds=self.max_seconds,
                    runner=self.runner,
                    directory=directory / f"case-{index:03d}",
                )
                reconfigurator = build_reconfigurator(
                    "reference",
                    candidates=candidates,
                    settings=self.assembly.get("reconfigurator"),
                    replay=self.assembly.get("replay"),
                    forecast=self.assembly.get("forecast"),
                    slo=slo,
                    optimizer=optimizer,
                )
                simulation = OnlineSimulation.from_forecast(
                    case.observation,
                    case.workload,
                    self.cost_models,
                    run,
                    handoff,
                    slo,
                    reconfigurator=reconfigurator,
                    prediction_cost_models=self.prediction_cost_models,
                    coalesce_decode=True,
                )
                reconfigurator.observe_history(case.observation)
                result = simulation.run()
                if result["status"] != "complete":
                    raise ValueError(f"Simulation did not complete: {result['reason']}")
                if optimizer.searches == 0:
                    raise ValueError(
                        "Simulation case did not exercise Optimizer.search; "
                        "supply observed history and an active decision window"
                    )
                completed_before = {
                    request.request_id
                    for request in case.observation.requests
                    if request.completed_at is not None
                }
                requests = [
                    request.view
                    for request in simulation.requests.values()
                    if request.view.request_id not in completed_before
                ]
                passed = sum(
                    request_slo_status(request, slo, now=request.completed_at) is True
                    for request in requests
                )
                rows.append(
                    dict(
                        total_gpu_seconds=result["total_gpu_seconds"],
                        joint_request_fraction=passed / len(requests)
                        if requests
                        else 1.0,
                        completed_requests=len(requests),
                        completed_updates=result["completed_updates"]
                        - case.observation.training_version,
                        trained_episodes=result["trained_episodes"]
                        - case.observation.consumed_episodes,
                        evaluator_queries=optimizer.query_count,
                        searches=optimizer.searches,
                        search_wall_seconds=optimizer.search_wall_seconds,
                        final_layout=result["final_layout"],
                    )
                )
            self.assert_unchanged()
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")
        metrics = dict(
            total_gpu_seconds=sum(row["total_gpu_seconds"] for row in rows),
            evaluator_queries=sum(row["evaluator_queries"] for row in rows),
            search_wall_seconds=sum(row["search_wall_seconds"] for row in rows),
            cases=rows,
            selection_sha256=digest(
                [
                    {
                        key: value
                        for key, value in row.items()
                        if key != "search_wall_seconds"
                    }
                    for row in rows
                ]
            ),
        )
        report = RevisionReport(
            self.target,
            source_digest(source),
            self.benchmark_sha256,
            not errors,
            metrics["total_gpu_seconds"] if not errors else None,
            metrics,
            tuple(errors),
        )
        self._reports.add(digest(asdict(report)))
        return report

    def compare(self, candidate, incumbent):
        self.assert_unchanged()
        if any(
            digest(asdict(report)) not in self._reports
            for report in (candidate, incumbent)
        ):
            raise ValueError("Comparison requires issued Simulation reports")
        if not incumbent.valid:
            raise ValueError("Current Optimizer must complete the Simulation cases")
        if not candidate.valid:
            return False, "Candidate Simulation failed"
        for current, proposed in zip(
            incumbent.metrics["cases"], candidate.metrics["cases"], strict=True
        ):
            if (
                proposed["joint_request_fraction"] + self.criteria.slo_tolerance
                < current["joint_request_fraction"]
            ):
                return False, "Simulation SLO regressed"
            if any(
                proposed[key] < current[key]
                for key in (
                    "completed_updates",
                    "trained_episodes",
                    "completed_requests",
                )
            ):
                return False, "Simulation training or serving progress regressed"
        old_cost, new_cost = incumbent.score, candidate.score
        tolerance = self.criteria.cost_relative_tolerance * max(
            1.0, abs(old_cost), abs(new_cost)
        )
        if new_cost > old_cost + tolerance:
            return False, "Simulation GPU cost regressed"
        if new_cost < old_cost - tolerance:
            return True, "Improved Simulation GPU cost"
        if (
            candidate.metrics["evaluator_queries"]
            < incumbent.metrics["evaluator_queries"]
        ):
            return True, "Same Simulation GPU cost with fewer Evaluator queries"
        return False, "No Simulation cost or query improvement"

    def feedback(self, report):
        if digest(asdict(report)) not in self._reports:
            raise ValueError("Unknown Simulation report")
        return dict(
            score=report.score,
            metrics={
                key: value
                for key, value in report.metrics.items()
                if key != "selection_sha256"
            },
            errors=report.errors,
        )
