# SPDX-License-Identifier: Apache-2.0

"""Online component factories: private historical scoring and reproducible deployment."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict

from areal.autopilot.revision.contracts import EvaluationReport, digest, source_digest
from areal.autopilot.revision.execution import load_source
from areal.utils.atomic import atomic_json

from .adapter import ElasticEvaluator, candidate_runner
from .contracts import BASELINE_SOURCE, decode
from .model_adapters import OnlineHistoryPredictor, validate_components
from .model_evaluator import HistoricalCase, HistoricalInput, HistoricalPrediction


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


class HistoricalSourceEvaluator(ElasticEvaluator):
    """build_model(calibration_cases, settings) returns the complete Online model.

    Only calibration labels enter the worker. The same returned object is used
    by historical component probes and by the reference candidate simulator.
    """

    target = "model"

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
        self._initial = initial_source or BASELINE_SOURCE["model"]
        self._reports = {}
        self.context = {
            "api_version": "online-components-v1",
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
                "Historical model evaluation cannot query reference labels"
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
        report = EvaluationReport(
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
            raise ValueError("Incumbent model must evaluate successfully")
        if not candidate.valid or new is None:
            return False, "Candidate model evaluation failed"
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

    def build(self, source):
        self.assert_unchanged()
        models = validate_components(
            load_source(source).build_model(self.history.calibration, self.settings)
        )
        self.assert_unchanged()
        return models

    def verify_loaded(self, models, expected_sha, expected_source_ids=None):
        self.assert_unchanged()
        if expected_source_ids is not None and tuple(models.source_ids) != tuple(
            expected_source_ids
        ):
            raise ValueError("Loaded Online model data identities changed")
        for _ in range(2):
            predictor = OnlineHistoryPredictor(models)
            rows = [
                {"prediction": asdict(predictor.predict(copy.deepcopy(i)))}
                for i in self.history.prediction_inputs
            ]
            if not expected_sha or prediction_fingerprint(rows) != expected_sha:
                raise ValueError(
                    "Deployed components differ from accepted model predictions"
                )
        self.assert_unchanged()
