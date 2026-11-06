# SPDX-License-Identifier: Apache-2.0

"""TIR-specific calibration adapter for existing model/Harness evaluation reports."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from areal.v2.data_engineering import (
    Attempt,
    CalibrationBatch,
    CalibrationContext,
    PublicTask,
)


class RecordedResponses:
    """Replay independent TIR evaluation reports; stored scores are never trusted."""

    def __init__(
        self,
        reports: list[list[str]],
        model_ref: str,
        harness_ref: str,
        backend_model_digest: str,
        verifier_digest: str,
    ):
        if not reports or any(
            not batch or any(not isinstance(path, str) for path in batch)
            for batch in reports
        ):
            raise ValueError("reports must contain one list of report paths per batch")
        if not backend_model_digest or not verifier_digest:
            raise ValueError(
                "Recorded calibration needs pinned model and verifier digests"
            )
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (model_ref, harness_ref)
        ):
            raise ValueError(
                "TIR calibration needs nonempty model and Harness references"
            )
        self.reports = [
            [Path(path).expanduser().resolve() for path in batch] for batch in reports
        ]
        self.model_ref = model_ref
        self.harness_ref = harness_ref
        self.backend_model_digest = backend_model_digest
        self.verifier_digest = verifier_digest

    def source_files(self) -> tuple[Path, ...]:
        return tuple(path for batch in self.reports for path in batch)

    async def __call__(
        self, tasks: Sequence[PublicTask], context: CalibrationContext
    ) -> CalibrationBatch:
        if (
            context.batch_index >= len(self.reports)
            or len(self.reports[context.batch_index]) != context.repeats
        ):
            raise ValueError(
                "Recorded calibration needs one report per repeat and batch"
            )
        attempts = []
        previous_identity = None
        trials = set()
        for repeat, path in enumerate(self.reports[context.batch_index]):
            report = json.loads(path.read_bytes())
            if (
                report.get("model") != self.model_ref
                or report.get("harness_id") != self.harness_ref
                or report.get("backend_model_digest") != self.backend_model_digest
                or report.get("verifier_digest") != self.verifier_digest
            ):
                raise ValueError(
                    "Recorded report changed model, Harness, or verifier identity"
                )
            identity = (report.get("task_set_id"), report.get("executable_digest"))
            if any(value is None for value in identity) or (
                previous_identity is not None and identity != previous_identity
            ):
                raise ValueError("Repeated reports changed task set or executable")
            previous_identity = identity
            trial = report.get("trial_id")
            if not isinstance(trial, str) or not trial or trial in trials:
                raise ValueError("Calibration repeats must use distinct trial IDs")
            trials.add(trial)
            cases = report["cases"]
            by_id = {case["task_id"]: case for case in cases}
            if len(by_id) != len(cases):
                raise ValueError("Recorded report contains duplicate tasks")
            for task in tasks:
                if task.task_id not in by_id:
                    raise ValueError("Recorded report omitted a requested task")
                case = by_id[task.task_id]
                attempts.append(
                    Attempt(
                        task.task_id,
                        repeat,
                        case.get("response"),
                        case.get("failure_reason"),
                    )
                )
        return CalibrationBatch(
            tuple(attempts),
            {
                "model_ref": self.model_ref,
                "harness_ref": self.harness_ref,
                "backend_model_digest": self.backend_model_digest,
                "verifier_digest": self.verifier_digest,
            },
        )


def recorded_responses(**params: Any) -> RecordedResponses:
    return RecordedResponses(**params)
