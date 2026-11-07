# SPDX-License-Identifier: Apache-2.0

"""Evaluate the Evaluator's CostModels revisions against historical measurements."""

from __future__ import annotations

import copy
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from areal.autopilot.domains.elastic.contracts import BASELINE_SOURCE, decode
from areal.autopilot.domains.elastic.cycle import (
    ElasticRevisionBenchmark,
    candidate_runner,
)
from areal.autopilot.domains.elastic.versions import protected_files
from areal.autopilot.revision.contracts import RevisionReport, digest, source_digest
from areal.autopilot.revision.execution import load_source
from areal.elastic.contracts import ExecutionConfig, OnlineLayout, Turn
from areal.elastic.reconfigurator.reference.evaluator.cost_model.contracts import (
    ServiceSlice,
)
from areal.elastic.reconfigurator.reference.evaluator.cost_model.loader import (
    CalibratedCostModels,
)
from areal.utils.atomic import atomic_json

METRICS = {
    "actor": frozenset(
        {
            "recompute_logp_seconds",
            "compute_advantage_seconds",
            "train_step_seconds",
            "actor_total_seconds",
        }
    ),
    "service": frozenset({"service_seconds"}),
    "publication": frozenset({"publication_seconds"}),
}


def _number(name: str, value: float) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class HistorySpan:
    """Inclusive original run/step interval; rank/phase aliases share this identity."""

    run_id: str
    first_step: int
    last_step: int

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("Original run identity is required")
        if (
            any(type(n) is not int or n < 0 for n in (self.first_step, self.last_step))
            or self.last_step < self.first_step
        ):
            raise ValueError("History span requires ordered nonnegative integer steps")

    def overlaps(self, other: HistorySpan) -> bool:
        return self.run_id == other.run_id and max(
            self.first_step, other.first_step
        ) <= min(self.last_step, other.last_step)


@dataclass(frozen=True)
class HistoricalInput:
    """Causal component work at its actual layout, without private timing labels."""

    kind: str
    layout: ExecutionConfig | OnlineLayout
    execution_id: str
    timing_boundary: str
    batch: tuple[tuple[Turn, ...], ...] = ()
    kv_tokens: int = 0
    slices: tuple[ServiceSlice, ...] = ()
    rollout_layouts: tuple[ExecutionConfig | OnlineLayout, ...] = ()
    input_mode: str = "given_work"

    def __post_init__(self):
        if (
            self.kind not in METRICS
            or not self.execution_id
            or not self.timing_boundary
        ):
            raise ValueError(
                "A component input needs an operation, execution identity and timing boundary"
            )
        if self.input_mode != "given_work" or self.kv_tokens < 0:
            raise ValueError(
                "Online component timing is conditional on supplied observed work"
            )
        if self.kind == "actor" and not self.batch:
            raise ValueError("Actor timing requires a complete observed batch")
        if self.kind == "service" and not self.slices:
            raise ValueError("Service timing requires observed forward slices")
        if self.kind == "publication" and not self.rollout_layouts:
            raise ValueError("Publication timing requires destination layouts")

    @property
    def config_id(self):
        return digest((asdict(self.layout), self.execution_id))

    @property
    def unsupported_reason(self):
        return ""


@dataclass(frozen=True)
class HistoricalCase:
    sample_id: str
    span: HistorySpan
    split: str
    inputs: HistoricalInput
    measured: tuple[tuple[str, float | None], ...]
    source_id: str
    workload_group: str = "all"

    def __post_init__(self) -> None:
        object.__setattr__(self, "measured", tuple(tuple(x) for x in self.measured))
        if self.split not in {"calibration", "validation", "final_holdout"}:
            raise ValueError(
                "Explicit calibration/validation/final_holdout split required"
            )
        if not all(
            isinstance(x, str) and x.strip()
            for x in (self.sample_id, self.source_id, self.workload_group)
        ):
            raise ValueError(
                "Sample identity, measurement source and workload group are required"
            )
        if not self.measured or len(dict(self.measured)) != len(self.measured):
            raise ValueError("Historical metric labels must have unique names")
        for name, value in self.measured:
            if name not in METRICS[self.inputs.kind]:
                raise ValueError(
                    f"Unknown measured metric for {self.inputs.kind}: {name}"
                )
            if value is not None:
                _number(name, value)


@dataclass(frozen=True)
class HistoricalPrediction:
    values: tuple[tuple[str, float | None], ...] = ()
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", tuple(tuple(x) for x in self.values))
        if len(dict(self.values)) != len(self.values):
            raise ValueError("Duplicate predicted metric")
        for name, value in self.values:
            if value is not None:
                _number(name, value)


class HistoricalPredictor(Protocol):
    def predict(self, inputs: HistoricalInput) -> HistoricalPrediction: ...


@dataclass(frozen=True)
class HistoryRow:
    sample_id: str
    kind: str
    boundary: str
    input_mode: str
    metric: str
    config_id: str
    workload_group: str
    measured: float | None
    predicted: float | None
    reason: str = ""

    @property
    def cell(self) -> tuple[str, str]:
        return self.sample_id, self.metric

    @property
    def covered(self) -> bool:
        return self.measured is not None and self.predicted is not None

    @property
    def metric_key(self) -> tuple[str, ...]:
        return self.kind, self.boundary, self.input_mode, self.metric


@dataclass(frozen=True)
class HistoryMetric:
    kind: str
    boundary: str
    input_mode: str
    metric: str
    group_by: str
    group: str
    cells: int
    labelled: int
    covered: int
    absolute_error: float
    measured_sum: float
    wape: float | None
    mae: float | None
    p90_relative_error: float | None
    relative_error_samples: int
    normalized_error: float | None

    @property
    def key(self) -> tuple[str, ...]:
        return (
            self.kind,
            self.boundary,
            self.input_mode,
            self.metric,
            self.group_by,
            self.group,
        )


def _summaries(
    rows: tuple[HistoryRow, ...], group_by: str = "all"
) -> tuple[HistoryMetric, ...]:
    groups = defaultdict(list)
    for row in rows:
        group = "all" if group_by == "all" else getattr(row, group_by)
        groups[(*row.metric_key, group)].append(row)
    result = []
    for (*key, group), members in sorted(groups.items()):
        known = [row for row in members if row.covered]
        absolute = math.fsum(abs(row.predicted - row.measured) for row in known)
        measured = math.fsum(row.measured for row in known)
        relative = sorted(
            abs(row.predicted - row.measured) / row.measured
            for row in known
            if row.measured > 0
        )
        # Keep undefined WAPE distinct from error on all-zero labels. For scoring,
        # that special case uses MAE in the metric's native unit (seconds/GPU-s).
        result.append(
            HistoryMetric(
                *key,
                group_by,
                group,
                len(members),
                sum(r.measured is not None for r in members),
                len(known),
                absolute,
                measured,
                absolute / measured if measured else None,
                absolute / len(known) if known else None,
                relative[math.ceil(0.9 * len(relative)) - 1] if relative else None,
                len(relative),
                absolute / (measured or len(known)) if known else None,
            )
        )
    return tuple(result)


def _score(
    metrics: tuple[HistoryMetric, ...], primary: tuple[str, ...]
) -> float | None:
    selected = [m for m in metrics if m.metric in primary]
    if (
        not selected
        or {m.metric for m in selected} != set(primary)
        or any(m.normalized_error is None for m in selected)
    ):
        return None
    return math.fsum(m.normalized_error for m in selected) / len(selected)


@dataclass(frozen=True)
class CostModelAcceptanceCriteria:
    min_relative_improvement: float = 0.005
    max_metric_regression: float = 0.0
    max_slice_regression: float = 0.0
    max_added_error: float = 0.1
    bootstrap_max_error: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "min_relative_improvement",
            "max_metric_regression",
            "max_slice_regression",
            "max_added_error",
        ):
            _number(name, getattr(self, name))
        if self.min_relative_improvement >= 1:
            raise ValueError("Minimum relative improvement must be below one")
        if self.bootstrap_max_error is not None:
            _number("bootstrap_max_error", self.bootstrap_max_error)


@dataclass(frozen=True)
class HistoryReport:
    candidate_version: str
    benchmark_sha256: str
    valid: bool
    score: float | None
    metrics: tuple[HistoryMetric, ...]
    slices: tuple[HistoryMetric, ...]
    rows: tuple[HistoryRow, ...]
    errors: tuple[str, ...]
    fitted_on: tuple[HistorySpan, ...]
    reserved_cases: int

    def feedback(self) -> dict:
        """Aggregate feedback; suppress small slices and absolute error statistics.

        This limits direct label disclosure, not inference from repeated queries.
        Validation is development data; final_holdout must remain untouched.
        """

        def public(metric):
            values = asdict(metric)
            for name in ("absolute_error", "measured_sum", "mae", "normalized_error"):
                values.pop(name)
            values["error_suppressed"] = metric.covered < 5
            if values["error_suppressed"]:
                values["wape"] = values["p90_relative_error"] = None
            return values

        return {
            "candidate_version": self.candidate_version,
            "benchmark_sha256": self.benchmark_sha256,
            "valid": self.valid,
            "score": self.score
            if self.metrics and all(m.covered >= 5 for m in self.metrics if m.labelled)
            else None,
            "errors": list(self.errors),
            "metrics": [public(m) for m in self.metrics],
            "slices": [public(m) for m in self.slices],
        }


@dataclass(frozen=True)
class CostModelComparison:
    accepted: bool
    reason: str
    incumbent_score: float | None
    candidate_score: float | None
    retained_cells: int
    added_cells: int
    lost_cells: int


class CostModelHistoryBenchmark:
    """Frozen history benchmark; repeated validation feedback is development data.

    final_holdout is fingerprinted and protected from fitting but never predicted
    or scored here. Supplied fitted_on provenance is a trusted caller declaration,
    not proof of what arbitrary Python previously read. No fitting occurs here.
    """

    def __init__(
        self,
        cases: tuple[HistoricalCase, ...],
        *,
        primary_metrics: tuple[str, ...],
        criteria: CostModelAcceptanceCriteria | None = None,
    ):
        self._cases = copy.deepcopy(tuple(sorted(cases, key=lambda c: c.sample_id)))
        self._primary_metrics = tuple(primary_metrics)
        self._criteria = criteria or CostModelAcceptanceCriteria()
        if not self._cases or len({c.sample_id for c in self._cases}) != len(
            self._cases
        ):
            raise ValueError("Historical cases require unique sample IDs")
        if not self.primary_metrics or len(set(self.primary_metrics)) != len(
            self.primary_metrics
        ):
            raise ValueError("Freeze a nonempty unique set of primary metrics")
        if "external_seconds" in self.primary_metrics:
            raise ValueError(
                "Known external waits cannot be a primary prediction metric"
            )
        if "rollout_seconds" in self.primary_metrics:
            if "inference_seconds" not in self.primary_metrics:
                raise ValueError(
                    "Rollout total acceptance also requires inference_seconds"
                )
            for case in self._cases:
                labels = dict(case.measured)
                if (
                    case.split == "validation"
                    and labels.get("rollout_seconds") is not None
                    and labels.get("inference_seconds") is None
                ):
                    raise ValueError(
                        "Each scored Rollout total needs an independent inference label"
                    )
        seen = set()
        for i, case in enumerate(self._cases):
            for other in self._cases[:i]:
                if case.split != other.split and case.span.overlaps(other.span):
                    raise ValueError(
                        "Calibration/validation/final history spans overlap"
                    )
            for metric, _ in case.measured:
                key = (
                    case.span,
                    case.inputs.kind,
                    case.inputs.timing_boundary,
                    case.inputs.input_mode,
                    metric,
                )
                if key in seen:
                    raise ValueError(
                        "Duplicate original measurement under different sample IDs"
                    )
                seen.add(key)
        validation = [c for c in self._cases if c.split == "validation"]
        available = {
            name for c in validation for name, value in c.measured if value is not None
        }
        if not set(self.primary_metrics) <= available:
            raise ValueError("Every primary metric needs a measured validation label")
        # Do not claim an improvement by mixing conditional and logical modes or
        # incompatible measurement boundaries in a single acceptance decision.
        for kind in METRICS:
            scopes = {
                (c.inputs.timing_boundary, c.inputs.input_mode)
                for c in validation
                if c.inputs.kind == kind
            }
            if len(scopes) > 1:
                raise ValueError(
                    "Use separate benchmarks for timing boundaries/input modes"
                )
        self._reference_files = protected_files()
        self._code = self._code_fingerprint()
        self._benchmark_sha256 = digest(
            {
                "cases": [asdict(c) for c in self._cases],
                "primary_metrics": self.primary_metrics,
                "criteria": asdict(self.criteria),
                "reference_code": self._code,
            }
        )
        self._issued_reports: set[str] = set()

    def _code_fingerprint(self) -> tuple[str, ...]:
        return tuple(source_digest(path.read_text()) for path in self._reference_files)

    @property
    def primary_metrics(self) -> tuple[str, ...]:
        return self._primary_metrics

    @property
    def criteria(self) -> CostModelAcceptanceCriteria:
        return self._criteria

    @property
    def benchmark_sha256(self) -> str:
        return self._benchmark_sha256

    @property
    def calibration(self) -> tuple[HistoricalCase, ...]:
        return copy.deepcopy(tuple(c for c in self._cases if c.split == "calibration"))

    @property
    def prediction_inputs(self) -> tuple[HistoricalInput, ...]:
        """Ordered prediction requests; benchmark-owned unsupported traces stay out."""
        return copy.deepcopy(
            tuple(
                c.inputs
                for c in self._cases
                if c.split == "validation" and not c.inputs.unsupported_reason
            )
        )

    def validate_fit(self, fitted_on: tuple[HistorySpan, ...]) -> None:
        protected = [c for c in self._cases if c.split != "calibration"]
        if any(not isinstance(span, HistorySpan) for span in fitted_on):
            raise ValueError("Fitting provenance requires original HistorySpan values")
        if any(span.overlaps(case.span) for span in fitted_on for case in protected):
            raise ValueError("Model fit overlaps validation or final holdout history")

    def assert_unchanged(self) -> None:
        if self._code_fingerprint() != self._code:
            raise ValueError(
                "History benchmark/reference code changed; create a new benchmark context"
            )

    def evaluate(
        self,
        predictor: HistoricalPredictor,
        version: str,
        *,
        fitted_on: tuple[HistorySpan, ...],
    ) -> HistoryReport:
        if not isinstance(version, str) or not version.strip():
            raise ValueError("Candidate version/snapshot identity is required")
        self.assert_unchanged()
        fitted_on = tuple(fitted_on)
        self.validate_fit(fitted_on)
        rows, errors = [], []
        for case in self._cases:
            if case.split != "validation":
                continue
            values, reason = {}, ""
            try:
                prediction = (
                    HistoricalPrediction(reason=case.inputs.unsupported_reason)
                    if case.inputs.unsupported_reason
                    else predictor.predict(copy.deepcopy(case.inputs))
                )
                if not isinstance(prediction, HistoricalPrediction):
                    raise ValueError("Candidate must return HistoricalPrediction")
                # Revalidate even if trusted code mutates an otherwise frozen value.
                prediction = HistoricalPrediction(prediction.values, prediction.reason)
                values, reason = dict(prediction.values), prediction.reason
                if not values.keys() <= METRICS[case.inputs.kind]:
                    raise ValueError("Candidate returned unknown metric names")
                if (
                    "rollout_seconds" in self.primary_metrics
                    and values.get("rollout_seconds") is not None
                    and values.get("inference_seconds") is None
                ):
                    raise ValueError(
                        "Scored Rollout total requires a paired inference prediction"
                    )
            except Exception as error:
                reason = f"{type(error).__name__}: {error}"
                errors.append(f"{case.sample_id}: {reason}")
                values = {}
            for name, measured in case.measured:
                value = values.get(name)
                rows.append(
                    HistoryRow(
                        case.sample_id,
                        case.inputs.kind,
                        case.inputs.timing_boundary,
                        case.inputs.input_mode,
                        name,
                        case.inputs.config_id,
                        case.workload_group,
                        measured,
                        value,
                        (reason or "Metric prediction unavailable")
                        if value is None
                        else "",
                    )
                )
        rows = tuple(rows)
        if self._code_fingerprint() != self._code:
            errors.append("History benchmark/reference code changed during prediction")
        metrics = _summaries(rows)
        report = HistoryReport(
            version,
            self.benchmark_sha256,
            not errors,
            _score(metrics, self.primary_metrics),
            metrics,
            _summaries(rows, "config_id") + _summaries(rows, "workload_group"),
            rows,
            tuple(errors),
            fitted_on,
            sum(c.split == "final_holdout" for c in self._cases),
        )
        self._issued_reports.add(digest(asdict(report)))
        return report

    def compare(
        self, candidate: HistoryReport, incumbent: HistoryReport
    ) -> CostModelComparison:
        if any(
            r.benchmark_sha256 != self.benchmark_sha256 for r in (candidate, incumbent)
        ):
            raise ValueError("Cannot compare different historical evaluation contexts")
        if self._code_fingerprint() != self._code:
            raise ValueError(
                "History benchmark/reference code changed before comparison"
            )
        if any(
            digest(asdict(r)) not in self._issued_reports
            for r in (candidate, incumbent)
        ):
            raise ValueError(
                "Comparison requires unchanged reports issued by this benchmark"
            )
        if not incumbent.valid:
            raise ValueError("Incumbent execution must be valid before comparison")
        old = {r.cell: r for r in incumbent.rows if r.covered}
        new = {r.cell: r for r in candidate.rows if r.covered}
        retained, added, lost = (
            old.keys() & new.keys(),
            new.keys() - old.keys(),
            old.keys() - new.keys(),
        )
        before = _score(_summaries(tuple(old.values())), self.primary_metrics)
        after = _score(
            _summaries(tuple(new[k] for k in sorted(retained))), self.primary_metrics
        )

        def decision(accepted, reason):
            return CostModelComparison(
                accepted, reason, before, after, len(retained), len(added), len(lost)
            )

        if not candidate.valid:
            return decision(False, "Candidate execution or output failed")
        if lost:
            return decision(
                False, "Candidate dropped previously covered sample/metric cells"
            )
        if not old:
            after = candidate.score
            limit = self.criteria.bootstrap_max_error
            accepted = (
                limit is not None
                and candidate.score is not None
                and all(
                    m.covered == m.labelled and m.normalized_error <= limit
                    for m in (*candidate.metrics, *candidate.slices)
                    if m.labelled
                )
            )
            return decision(
                accepted,
                "Bootstrap accuracy/coverage passed"
                if accepted
                else "Empty incumbent coverage requires explicit bootstrap accuracy and full coverage",
            )
        added_rows = tuple(new[k] for k in sorted(added))
        added_metrics = tuple(
            m
            for grouping in ("all", "config_id", "workload_group")
            for m in _summaries(added_rows, grouping)
        )
        if any(
            m.normalized_error > self.criteria.max_added_error for m in added_metrics
        ):
            return decision(False, "New coverage exceeds its fixed error limit")
        for grouping, limit in (
            ("all", self.criteria.max_metric_regression),
            ("config_id", self.criteria.max_slice_regression),
            ("workload_group", self.criteria.max_slice_regression),
        ):
            old_metrics = {m.key: m for m in _summaries(tuple(old.values()), grouping)}
            new_metrics = _summaries(tuple(new[k] for k in sorted(retained)), grouping)
            if any(
                m.normalized_error > old_metrics[m.key].normalized_error + limit + 1e-12
                for m in new_metrics
            ):
                return decision(False, f"Previously covered {grouping} error regressed")
        if before is None or after is None:
            return decision(False, "Primary metrics need common covered measurements")
        improvement = before - after
        if improvement <= max(1e-12, before * self.criteria.min_relative_improvement):
            return decision(
                False, "Insufficient improvement on identical covered measurements"
            )
        return decision(
            True, "Historical prediction error improved without coverage loss"
        )


def calibrated_cost_models(settings):
    cost_model_config = settings.get("cost_models", settings.get("model"))
    return CalibratedCostModels(
        Path(cost_model_config["bundle"]),
        Path(cost_model_config["mechanism_fits"]),
        Path(cost_model_config["publication_metadata"]),
        float(cost_model_config["effective_bandwidth_GBps"]),
        inter_node_dp_GBps=cost_model_config.get("inter_node_dp_GBps"),
    )


def validate_cost_models(cost_models):
    if not getattr(cost_models, "source_ids", ()) or not all(
        callable(getattr(cost_models, name, None))
        for name in ("rollout_config", "service", "actor", "publication")
    ):
        raise ValueError(
            "build_cost_models must return the complete CostModels interface"
        )
    return (
        cost_models
        if isinstance(cost_models, FrozenCostModels)
        else FrozenCostModels(cost_models)
    )


class FrozenCostModels:
    """Keep factory behavior when evaluating accepted CostModels.

    A baseline bundle worker_spec reloads baseline code, not the revised factory.
    An inherited decode-affine shortcut may likewise bypass a revised service().
    Until those exports have factory-aware reconstruction/probes, use the actual
    component callbacks for both historical and candidate execution.
    """

    def __init__(self, cost_models):
        self._cost_models = cost_models

    def __getattr__(self, name):
        if name in {"worker_spec", "decode_affine"}:
            raise AttributeError(name)
        return getattr(self._cost_models, name)


class CostModelHistoryPredictor:
    def __init__(self, cost_models):
        self.cost_models = validate_cost_models(cost_models)

    def predict(self, inputs):
        try:
            return self._predict(inputs)
        except ValueError as error:
            return HistoricalPrediction(reason=str(error))

    def _predict(self, inputs):
        if inputs.kind == "actor":
            stages = tuple(self.cost_models.actor(inputs.layout, inputs.batch))
            names = {
                "logp": "recompute_logp_seconds",
                "advantage": "compute_advantage_seconds",
                "train": "train_step_seconds",
                "recompute_logp": "recompute_logp_seconds",
                "compute_advantage": "compute_advantage_seconds",
                "train_step": "train_step_seconds",
            }
            values = tuple(
                (
                    names.get(
                        name, name if name.endswith("_seconds") else name + "_seconds"
                    ),
                    seconds,
                )
                for name, seconds in stages
            )
            return HistoricalPrediction(
                values + (("actor_total_seconds", sum(t for _, t in stages)),)
            )
        if inputs.kind == "service":
            return HistoricalPrediction(
                (
                    (
                        "service_seconds",
                        self.cost_models.service(
                            inputs.layout, inputs.kv_tokens, inputs.slices
                        ),
                    ),
                )
            )
        return HistoricalPrediction(
            (
                (
                    "publication_seconds",
                    self.cost_models.publication(inputs.layout, inputs.rollout_layouts),
                ),
            )
        )


def historical_input_from_dict(data):
    return decode(HistoricalInput, data)


def historical_case_from_dict(data):
    return decode(HistoricalCase, data)


def prediction_fingerprint(rows):
    return digest(
        [
            sorted(dict(HistoricalPrediction(**r["prediction"]).values).items())
            for r in rows
        ]
    )


class EvaluatorRevisionBenchmark(ElasticRevisionBenchmark):
    """Benchmark Evaluator revisions that currently replace only its CostModels.

    build_cost_models(calibration_cases, settings) returns the CostModels interface.
    Only calibration labels enter the worker. Historical probes and the Reference
    configuration Evaluator use the same returned object.
    """

    target = "evaluator"

    def __init__(
        self, history, *, settings, fitted_on=(), runner=None, initial_source=None
    ):
        self.history = history
        self.runner = candidate_runner(runner)
        self._settings_json = json.dumps(settings, sort_keys=True, allow_nan=False)
        self._fitted_on = tuple(
            sorted(
                set(fitted_on) | {c.span for c in history.calibration},
                key=lambda s: (s.run_id, s.first_step, s.last_step),
            )
        )
        history.validate_fit(self._fitted_on)
        self._initial = initial_source or BASELINE_SOURCE["evaluator"]
        self._reports = {}
        self.context = {
            "api_version": "online-evaluator-cost-models-v2",
            "history_sha256": history.benchmark_sha256,
            "settings": self.settings,
            "fitted_on": [asdict(s) for s in self._fitted_on],
            "worker_limits": {
                "timeout_seconds": self.runner.timeout_seconds,
                "max_output_bytes": self.runner.max_output_bytes,
            },
        }
        self.benchmark_sha256 = digest(self.context)

    @property
    def initial_source(self):
        return self._initial

    @property
    def settings(self):
        return json.loads(self._settings_json)

    @property
    def criteria(self):
        return self.history.criteria

    @property
    def calibration(self):
        return {"cases": [asdict(c) for c in self.history.calibration]}

    def assert_unchanged(self):
        self.history.assert_unchanged()

    def evaluate(self, source, directory):
        errors, metrics, raw = [], {}, None
        violations = []

        def no_reference(message):
            violations.append(
                "CostModel history benchmark cannot query reference labels"
            )
            raise ValueError(violations[-1])

        try:
            self.assert_unchanged()
            request = dict(
                target=self.target,
                settings=self.settings,
                calibration=self.calibration,
                cases=[asdict(i) for i in self.history.prediction_inputs],
            )
            output = self.runner.run(source, request, directory, no_reference)
            if output.get("error") or violations:
                raise ValueError(output.get("error") or violations[0])
            rows = output["rows"]
            if len(rows) != len(request["cases"]):
                raise ValueError("Model must predict every historical input")
            fingerprint = prediction_fingerprint(rows)
            repeated = self.runner.run(
                source, request, directory / "reproduce", no_reference
            )
            if (
                repeated.get("error")
                or violations
                or prediction_fingerprint(repeated["rows"]) != fingerprint
                or repeated.get("source_ids") != output.get("source_ids")
            ):
                raise ValueError("Model factory predictions are not reproducible")
            self.assert_unchanged()

            class Predictions:
                def __init__(self):
                    self.rows = iter(rows)

                def predict(self, inputs):
                    return HistoricalPrediction(**next(self.rows)["prediction"])

            raw = self.history.evaluate(
                Predictions(), source_digest(source), fitted_on=self._fitted_on
            )
            atomic_json(directory / "history-evaluation.json", asdict(raw))
            metrics = {
                **raw.feedback(),
                "prediction_sha256": fingerprint,
                "source_ids": output["source_ids"],
            }
            errors.extend(raw.errors)
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")
        report = RevisionReport(
            self.target,
            source_digest(source),
            self.benchmark_sha256,
            not errors,
            None if raw is None else raw.score,
            metrics,
            tuple(errors),
        )
        self._reports[digest(asdict(report))] = raw
        return report

    def _verified(self, report):
        self.assert_unchanged()
        key = digest(asdict(report))
        if report.benchmark_sha256 != self.benchmark_sha256 or key not in self._reports:
            raise ValueError("Comparison requires an unchanged issued report")
        return self._reports[key]

    def compare(self, candidate, incumbent):
        old, new = self._verified(incumbent), self._verified(candidate)
        if not incumbent.valid or old is None:
            raise ValueError("Incumbent Evaluator revision must benchmark successfully")
        if not candidate.valid or new is None:
            return False, "Candidate Evaluator revision benchmark failed"
        result = self.history.compare(new, old)
        return result.accepted, result.reason

    def feedback(self, report):
        self._verified(report)
        metrics = {k: v for k, v in report.metrics.items() if k != "prediction_sha256"}
        return {
            "score": metrics.get("score"),
            "metrics": metrics,
            "errors": report.errors,
        }

    def build_cost_models(self, source):
        self.assert_unchanged()
        cost_models = validate_cost_models(
            load_source(source).build_cost_models(
                self.history.calibration, self.settings
            )
        )
        self.assert_unchanged()
        return cost_models

    def verify_cost_models(self, cost_models, expected_sha, expected_source_ids=None):
        self.assert_unchanged()
        if expected_source_ids is not None and tuple(cost_models.source_ids) != tuple(
            expected_source_ids
        ):
            raise ValueError("Loaded CostModel data identities changed")
        for _ in range(2):
            predictor = CostModelHistoryPredictor(cost_models)
            rows = [
                {"prediction": asdict(predictor.predict(copy.deepcopy(i)))}
                for i in self.history.prediction_inputs
            ]
            if not expected_sha or prediction_fingerprint(rows) != expected_sha:
                raise ValueError("Deployed CostModels differ from accepted predictions")
        self.assert_unchanged()
