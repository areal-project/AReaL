# SPDX-License-Identifier: Apache-2.0

"""One resumable offline cycle: Online component model followed by Online optimizer."""

from __future__ import annotations

import fcntl
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

from areal.autopilot.domains.elastic.contracts import BASELINE_SOURCE
from areal.autopilot.revision.contracts import digest, source_digest
from areal.autopilot.revision.loop import RevisionLoop
from areal.utils.atomic import atomic_json


def _fixed_json(path: Path, value: dict) -> None:
    value = json.loads(json.dumps(value, allow_nan=False))
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Saved {path.name} differs from the fixed revision cycle")
    else:
        atomic_json(path, value)


def _initial_source(path: Path, source: str) -> None:
    if path.exists():
        if path.read_text() != source:
            raise ValueError("Stage initial source changed")
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w") as stream:
            stream.write(source)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class _Stage:
    root: Path
    source: Path
    record: dict

    def check(self):
        if (
            source_digest(self.source.read_text()) != self.record["source_sha256"]
            or digest(json.loads((self.root / self.record["report"]).read_text()))
            != self.record["report_sha256"]
        ):
            raise ValueError(
                "Completed stage source/evaluation changed after verification"
            )


def _stage(loop: RevisionLoop, budget: int) -> _Stage:
    """Recover durable child results; count started attempts against the total cap."""
    root, evaluator = loop.store.root, loop.evaluator
    initial = loop.initial_source or BASELINE_SOURCE[evaluator.target]
    initial_path = root / "initial.py"
    receipt_path = root / "stage.json"
    with (root / "revision.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        context_path = root / "context.json"
        if context_path.exists():
            if json.loads(context_path.read_text()) != loop.context:
                raise ValueError(
                    "Child stage belongs to another fixed evaluation context"
                )
        elif any(
            (root / name).exists()
            for name in (
                "active.json",
                "stage.json",
                "run.json",
                "baseline.json",
                "attempts",
            )
        ):
            raise ValueError("Child stage is missing its evaluation context")
        _initial_source(initial_path, initial)
        attempts = sorted((root / "attempts").glob("round-*"))
        if len(attempts) > budget:
            raise ValueError("Started revision attempts exceed the cycle budget")
        active = loop.store.active()
        # A promotion may have crashed after its acceptance receipt was durable.
        pending = [p for p in attempts if (p / "acceptance.json").exists()]
        if active is None and pending:
            if len(pending) != 1 or receipt_path.exists():
                raise ValueError("Missing active pointer for an already accepted stage")
            candidate = pending[0] / "candidate.py"
            loop.store.promote(
                candidate, json.loads((pending[0] / "evaluation.json").read_text())
            )
            active = loop.store.active()
        complete = (
            receipt_path.exists() or active is not None or (root / "run.json").exists()
        )
        remaining = budget - len(attempts)
        if not complete and remaining:
            # Already hold the same lock used by RevisionLoop.run; avoid relocking
            # and keep the budget check atomic with any new proposals.
            loop._run(remaining, True)
            active = loop.store.active()
        if not context_path.exists() or not (root / "baseline.json").exists():
            raise ValueError("Stage has no completed baseline evaluation")
        if (root / "run.json").exists():
            run = json.loads((root / "run.json").read_text())
            if (
                run["target"] != evaluator.target
                or run["benchmark_sha256"] != evaluator.benchmark_sha256
            ):
                raise ValueError(
                    "Child run result belongs to another evaluation context"
                )
            if active is None and run.get("active_source") is not None:
                raise ValueError("Accepted child run is missing its active source")
        source = active[0] if active is not None else initial_path
        report_path = (
            source.parent / "evaluation.json"
            if active is not None
            else root / "baseline.json"
        )
        report = json.loads(report_path.read_text())
        baseline_path = root / "baseline.json"
        record = {
            "target": evaluator.target,
            "context_sha256": digest(loop.context),
            "source": str(source.relative_to(root)),
            "source_sha256": source_digest(source.read_text()),
            "report": str(report_path.relative_to(root)),
            "report_sha256": digest(report),
            "baseline_sha256": digest(json.loads(baseline_path.read_text())),
            "accepted": active is not None,
        }
        if receipt_path.exists() and json.loads(receipt_path.read_text()) != record:
            raise ValueError("Completed stage source/receipt changed after handoff")
        if (
            not report["valid"]
            or report["target"] != evaluator.target
            or report["benchmark_sha256"] != evaluator.benchmark_sha256
            or report["candidate_sha256"] != record["source_sha256"]
        ):
            raise ValueError("Effective stage source has no matching valid evaluation")
        checked = evaluator.evaluate(
            source.read_text(), root / f"verify-{uuid.uuid4().hex}"
        )
        keys = (
            ("prediction_sha256",)
            if evaluator.target == "model"
            else ("prediction_sha256",)
            if evaluator.target in {"simulator", "serial_model"}
            else ("selection_sha256",)
        )
        if not checked.valid or any(
            checked.metrics.get(k) != report["metrics"].get(k) for k in keys
        ):
            raise ValueError(
                "Effective stage predictions/selections changed since evaluation"
            )
        _fixed_json(receipt_path, record)
        return _Stage(root, source, record)


class ReconfiguratorRevisionCycle:
    """Freeze a model revision, then evaluate Online search against that same model."""

    def __init__(
        self,
        model_evaluator,
        optimizer_cases,
        root,
        *,
        assembly,
        initial_model_source=None,
        initial_optimizer_source=None,
        optimizer_settings=None,
        optimizer_limits=None,
        parent_bundle=None,
    ):
        self.model = model_evaluator
        self.cases = tuple(optimizer_cases)
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.assembly = json.loads(json.dumps(assembly, allow_nan=False))
        from areal.elastic.contracts import OnlineLayout

        from .contracts import OnlineProblem, decode

        catalog = tuple(decode(OnlineLayout, c) for c in self.assembly["candidates"])
        if any(
            not set((decode(OnlineProblem, c) if isinstance(c, dict) else c).candidates)
            <= set(catalog)
            for c in self.cases
        ):
            raise ValueError(
                "Optimizer problems must use the assembly's frozen candidate catalog"
            )
        self.model_source = initial_model_source or model_evaluator.initial_source
        self.optimizer_source = initial_optimizer_source or BASELINE_SOURCE["optimizer"]
        self.optimizer_settings = optimizer_settings or {}
        self.optimizer_limits = optimizer_limits or {}
        self.parent_bundle = parent_bundle

    def run(
        self,
        model_agent,
        optimizer_agent,
        *,
        max_model_rounds=3,
        max_optimizer_rounds=3,
    ):
        from dataclasses import asdict

        from areal.elastic.contracts import HandoffConfig, SimulationConfig
        from areal.elastic.reconfigurator import build_reconfigurator
        from areal.elastic.serving import ServingSLO

        from .bundle import export_accepted_bundle
        from .contracts import decode
        from .optimizer_evaluator import OnlineOptimizerEvaluator

        with (self.root / "cycle.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            _fixed_json(
                self.root / "context.json",
                dict(
                    model=self.model.benchmark_sha256,
                    cases=[
                        asdict(c) if hasattr(c, "__dataclass_fields__") else c
                        for c in self.cases
                    ],
                    assembly=self.assembly,
                    model_source=source_digest(self.model_source),
                    optimizer_source=source_digest(self.optimizer_source),
                    settings=self.optimizer_settings,
                    limits=self.optimizer_limits,
                    budgets=[max_model_rounds, max_optimizer_rounds],
                    parent_bundle_id=self.parent_bundle,
                ),
            )
            model_stage = _stage(
                RevisionLoop(
                    model_agent,
                    self.model,
                    self.root / "model",
                    initial_source=self.model_source,
                ),
                max_model_rounds,
            )
            model_source = model_stage.source.read_text()
            model_report = json.loads(
                (model_stage.root / model_stage.record["report"]).read_text()
            )
            models = self.model.build(model_source)
            self.model.verify_loaded(
                models,
                model_report["metrics"]["prediction_sha256"],
                model_report["metrics"]["source_ids"],
            )
            run = decode(SimulationConfig, self.assembly["run"])
            handoff = decode(HandoffConfig, self.assembly["handoff"])
            slo = decode(ServingSLO, self.assembly["slo"])
            context = dict(
                api_version="online-reference-optimizer-v1",
                assembly_sha256=digest(self.assembly),
                model_source_sha256=source_digest(model_source),
                model_prediction_sha256=model_report["metrics"]["prediction_sha256"],
                model_settings_sha256=digest(self.model.settings),
            )
            from areal.elastic.contracts import OnlineLayout

            prototype = build_reconfigurator(
                "reference",
                candidates=tuple(
                    decode(OnlineLayout, c) for c in self.assembly["candidates"]
                ),
                settings=self.assembly.get("reconfigurator"),
                replay=self.assembly.get("replay"),
                forecast=self.assembly.get("forecast"),
                slo=slo,
            )

            def make_model():
                return prototype.make_model(
                    models,
                    run,
                    handoff,
                    slo,
                    prototype.config.decision_delay_seconds,
                    actor_memory_reserve_fraction=prototype.config.actor_memory_reserve_fraction,
                    actor_memory_scope=prototype.config.actor_memory_scope,
                )

            evaluator = OnlineOptimizerEvaluator(
                self.cases,
                model_factory=make_model,
                context=context,
                settings=self.optimizer_settings,
                runner=self.model.runner,
                **self.optimizer_limits,
            )
            optimizer_stage = _stage(
                RevisionLoop(
                    optimizer_agent,
                    evaluator,
                    self.root / "optimizer",
                    initial_source=self.optimizer_source,
                ),
                max_optimizer_rounds,
            )
            path = export_accepted_bundle(
                self.root / "bundles",
                model_evaluator=self.model,
                model_stage=model_stage,
                optimizer_evaluator=evaluator,
                optimizer_stage=optimizer_stage,
                assembly=self.assembly,
                parent_bundle_id=self.parent_bundle,
            )
            result = dict(
                model_accepted=model_stage.record["accepted"],
                optimizer_accepted=optimizer_stage.record["accepted"],
                bundle=None if path is None else str(path),
                model_source=str(model_stage.source),
                optimizer_source=str(optimizer_stage.source),
            )
            _fixed_json(self.root / "result.json", result)
            return result
