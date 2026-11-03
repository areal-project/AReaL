# SPDX-License-Identifier: Apache-2.0

"""Evaluate Optimizer revisions with parent-owned configuration predictions."""

from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict, dataclass

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

    def __init__(self, source, settings=None, *, max_queries=64, max_seconds=30):
        self.source = source
        self.settings = copy.deepcopy(settings or {})
        self.searcher = load_source(source).build_optimizer(self.settings)
        if not callable(getattr(self.searcher, "search", None)):
            raise ValueError(
                "build_optimizer must return an Online search implementation"
            )
        self.max_queries, self.max_seconds = max_queries, max_seconds

    def search(
        self, evaluator, view, forecasts, layouts, *, preferred=(), trigger="pressure"
    ):
        problem = OnlineProblem(
            view, tuple(forecasts), tuple(layouts), tuple(preferred), trigger
        )
        port = SearchPort(
            evaluator,
            problem,
            max_queries=self.max_queries,
            max_seconds=self.max_seconds,
        )
        searcher = load_source(self.source).build_optimizer(
            copy.deepcopy(self.settings)
        )
        predictions = port.finish(
            searcher.search(
                port, view, forecasts, layouts, preferred=preferred, trigger=trigger
            )
        )
        evaluator.execution_stats = dict(
            mode="verified_online_revision",
            model_queries=port.queries,
            source_sha256=source_digest(self.source),
        )
        return predictions

    def evaluate_many(self, evaluator, view, forecasts, layouts):
        # Application revalidation evaluates the supplied layouts directly.
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
        # Start each case's budget when the child begins it, excluding process startup.
        begun = set()
        finished = {}

        def simulate(message):
            index = message["case"]
            if type(index) is not int or not 0 <= index < len(ports):
                raise ValueError("Unknown Online case")
            port, case = ports[index], self.cases[index]
            operation = message["operation"]
            if operation == "start":
                if index in begun:
                    return port.reject("Search case started twice")
                begun.add(index)
                port.started = time.monotonic()
                return dict(
                    context_id=port.context_id,
                    historical=port.historical,
                    slo=asdict(port.slo),
                    parallel_available=port.parallel_available,
                )
            if index not in begun:
                return port.reject("Search queried before starting its case")
            if operation == "order":
                return {
                    "order": [
                        (i, asdict(c))
                        for i, c in port.order_candidates(
                            case.observation,
                            case.forecasts,
                            case.candidates,
                            case.preferred,
                        )
                    ]
                }
            if operation == "wave":
                from areal.elastic.contracts import CandidatePrediction, OnlineLayout

                indexed = tuple(
                    (i, decode(OnlineLayout, c)) for i, c in message["indexed"]
                )
                raw = message.get("bound")
                bound = (
                    None
                    if raw is None
                    else SearchBound(
                        port.context_id,
                        raw["target"],
                        decode(CandidatePrediction, raw["best"]),
                        raw["fraction_pruning"],
                    )
                )
                result = port.evaluate_wave(
                    case.observation,
                    case.forecasts,
                    indexed,
                    bound=bound,
                    chunk_size=message.get("chunk_size", 2),
                )
                return {"wave": asdict(result)}
            if operation == "finish":
                from areal.elastic.contracts import CandidatePrediction

                finished[index] = port.finish(
                    tuple(
                        decode(CandidatePrediction, p) for p in message["predictions"]
                    )
                )
                return {"finished": True}
            return port.reject("Unknown configuration Evaluator operation")

        try:
            self.assert_unchanged()
            output = self.runner.run(
                source,
                dict(
                    target=self.target,
                    settings=self.settings,
                    cases=[asdict(c) for c in self.cases],
                ),
                directory,
                simulate,
            )
            if output.get("error"):
                raise ValueError(output["error"])
            if len(output["predictions"]) != len(self.cases):
                raise ValueError("Search omitted cases")
            from areal.elastic.contracts import CandidatePrediction

            for index, (case, raw) in enumerate(
                zip(self.cases, output["predictions"], strict=True)
            ):
                port = ports[index]
                predictions = tuple(decode(CandidatePrediction, p) for p in raw)
                if (
                    index not in finished
                    or predictions != finished[index]
                    or port.violations
                ):
                    raise ValueError(
                        "Search result lacks its completed parent-owned evaluation"
                    )
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
