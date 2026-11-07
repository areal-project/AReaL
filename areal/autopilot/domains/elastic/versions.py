# SPDX-License-Identifier: Apache-2.0

"""Accepted component identities, version bundles, verification and loading."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from areal.autopilot.domains.elastic.contracts import decode
from areal.autopilot.revision.contracts import digest, source_digest
from areal.autopilot.revision.execution import load_source
from areal.autopilot.revision.loop import RevisionStore
from areal.elastic.contracts import ExecutionConfig, OnlineLayout
from areal.elastic.reconfigurator import build_reconfigurator
from areal.utils.atomic import atomic_json

AREAL_ROOT = Path(__file__).resolve().parents[3]


def protected_files() -> tuple[Path, ...]:
    roots = (
        AREAL_ROOT / "autopilot/revision",
        AREAL_ROOT / "autopilot/domains/elastic",
        AREAL_ROOT / "elastic",
    )
    paths = {
        p for root in roots for p in root.rglob("*") if p.suffix in {".py", ".json"}
    }
    paths.update(
        AREAL_ROOT / name
        for name in (
            "utils/atomic.py",
            "trainer/rl_trainer.py",
            "v2/training_service/checkpoint.py",
            "v2/training_service/runtime_state.py",
        )
    )
    return tuple(sorted(paths))


def _stage_evidence(stage):
    stage.check()
    report = json.loads((stage.root / stage.record["report"]).read_text())
    source = stage.source.read_text()
    return dict(
        source=source,
        sha256=source_digest(source),
        accepted=stage.record["accepted"],
        context=json.loads((stage.root / "context.json").read_text()),
        report=report,
        acceptance=json.loads((stage.source.parent / "acceptance.json").read_text())
        if stage.record["accepted"]
        else None,
    )


def export_accepted_bundle(
    destination,
    *,
    evaluator_benchmark,
    evaluator_stage,
    optimizer_benchmark,
    optimizer_stage,
    assembly,
    parent_bundle_id=None,
):
    evaluator_benchmark.assert_unchanged()
    optimizer_benchmark.assert_unchanged()
    if (
        not evaluator_stage.record["accepted"]
        and not optimizer_stage.record["accepted"]
    ):
        return None
    payload = dict(
        format_version=2,
        kind="online-reference-reconfigurator-v2",
        implementation="reference",
        assembly=assembly,
        cost_model_settings=evaluator_benchmark.settings,
        calibration_cases=evaluator_benchmark.calibration["cases"],
        cost_model_probes=[
            asdict(i) for i in evaluator_benchmark.history.prediction_inputs
        ],
        components=dict(
            evaluator=_stage_evidence(evaluator_stage),
            optimizer=_stage_evidence(optimizer_stage),
        ),
        cost_model_source_ids=json.loads(
            (evaluator_stage.root / evaluator_stage.record["report"]).read_text()
        )["metrics"]["source_ids"],
        optimizer_settings=optimizer_benchmark.settings,
        optimizer_limits=dict(
            max_queries=optimizer_benchmark.max_queries,
            max_seconds=optimizer_benchmark.max_seconds,
        ),
        parent_bundle_id=parent_bundle_id,
        protected_files={
            str(p.relative_to(AREAL_ROOT)): source_digest(p.read_text())
            for p in protected_files()
        },
    )
    value = json.loads(
        json.dumps(dict(payload, bundle_id=digest(payload)), allow_nan=False)
    )
    validate_bundle(value)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / (value["bundle_id"] + ".json")
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError("Immutable Online bundle changed")
    if not path.exists():
        atomic_json(path, value)
    return path


def validate_bundle(value):
    payload = {k: v for k, v in value.items() if k != "bundle_id"}
    if (
        value.get("kind") != "online-reference-reconfigurator-v2"
        or value.get("implementation") != "reference"
        or value.get("format_version") != 2
        or value.get("bundle_id") != digest(payload)
    ):
        raise ValueError("Unsupported or changed Online bundle")
    components = value["components"]
    if set(components) != {"evaluator", "optimizer"} or not any(
        c["accepted"] for c in components.values()
    ):
        raise ValueError("An Online bundle needs an accepted component")
    for name, c in components.items():
        report, context = c["report"], c["context"]
        if (
            source_digest(c["source"]) != c["sha256"]
            or not report["valid"]
            or report["target"] != name
            or context["target"] != name
            or report["candidate_sha256"] != c["sha256"]
            or report["benchmark_sha256"] != context["benchmark_sha256"]
        ):
            raise ValueError("Component source/evaluation identity differs")
        if c["accepted"]:
            a = c["acceptance"]
            if (
                not a
                or a.get("accepted") is not True
                or a["candidate_sha256"] != c["sha256"]
                or a["benchmark_sha256"] != report["benchmark_sha256"]
            ):
                raise ValueError("Component has no matching acceptance")
        elif context.get("initial_sha256") != c["sha256"]:
            raise ValueError("Retained component differs from its revalidated baseline")
    evaluator_context = components["evaluator"]["context"]["deployment_context"]
    optimizer_context = components["optimizer"]["context"]["deployment_context"]
    if (
        evaluator_context["settings"] != value["cost_model_settings"]
        or optimizer_context.get("assembly_sha256") != digest(value["assembly"])
        or optimizer_context.get("evaluator_source_sha256")
        != components["evaluator"]["sha256"]
        or optimizer_context.get("evaluator_prediction_sha256")
        != components["evaluator"]["report"]["metrics"]["prediction_sha256"]
        or optimizer_context.get("cost_model_settings_sha256")
        != digest(value["cost_model_settings"])
    ):
        raise ValueError(
            "Model and optimizer were not validated under this shared Online contract"
        )
    if (
        optimizer_context.get("optimizer_settings") != value["optimizer_settings"]
        or optimizer_context.get("optimizer_limits") != value["optimizer_limits"]
        or value["cost_model_source_ids"]
        != components["evaluator"]["report"]["metrics"]["source_ids"]
    ):
        raise ValueError(
            "Online factory settings, limits or CostModel data identities changed"
        )
    if not value["cost_model_probes"] or not value["protected_files"]:
        raise ValueError(
            "Online bundle requires deployment probes and runtime identity"
        )
    for name, expected in value["protected_files"].items():
        path = (AREAL_ROOT / name).resolve()
        if (
            not path.is_relative_to(AREAL_ROOT)
            or not path.is_file()
            or source_digest(path.read_text()) != expected
        ):
            raise ValueError("Online evaluation/runtime code changed")


@dataclass(frozen=True)
class LoadedOnlineBundle:
    reconfigurator: object
    cost_models: object
    config: object
    handoff: object
    slo: object
    bundle_id: str


def load_bundle(path):
    from areal.autopilot.domains.elastic.evaluation.evaluator import (
        CostModelHistoryPredictor,
        HistoricalCase,
        HistoricalInput,
        prediction_fingerprint,
        validate_cost_models,
    )
    from areal.autopilot.domains.elastic.evaluation.optimizer import (
        VerifiedOnlineOptimizer,
    )
    from areal.elastic.contracts import HandoffConfig, SimulationConfig
    from areal.elastic.serving import ServingSLO

    value = json.loads(Path(path).read_text())
    validate_bundle(value)
    components = value["components"]
    cost_models = validate_cost_models(
        load_source(components["evaluator"]["source"]).build_cost_models(
            tuple(decode(HistoricalCase, c) for c in value["calibration_cases"]),
            value["cost_model_settings"],
        )
    )
    if tuple(cost_models.source_ids) != tuple(value["cost_model_source_ids"]):
        raise ValueError("Loaded CostModels have different frozen data identities")
    predictor = CostModelHistoryPredictor(cost_models)
    expected = components["evaluator"]["report"]["metrics"]["prediction_sha256"]
    for _ in range(2):
        rows = [
            dict(prediction=asdict(predictor.predict(decode(HistoricalInput, i))))
            for i in value["cost_model_probes"]
        ]
        if prediction_fingerprint(rows) != expected:
            raise ValueError("Loaded CostModels differ from benchmarked predictions")
    spec = value["assembly"]
    optimizer = VerifiedOnlineOptimizer(
        components["optimizer"]["source"],
        value["optimizer_settings"],
        **value["optimizer_limits"],
    )
    reconfigurator = build_reconfigurator(
        "reference",
        candidates=tuple(
            decode(ExecutionConfig | OnlineLayout, c) for c in spec["candidates"]
        ),
        settings=spec.get("reconfigurator"),
        replay=spec.get("replay"),
        forecast=spec.get("forecast"),
        slo=spec["slo"],
        optimizer=optimizer,
    )
    validate_bundle(value)
    return LoadedOnlineBundle(
        reconfigurator,
        cost_models,
        decode(SimulationConfig, spec["run"]),
        decode(HandoffConfig, spec["handoff"]),
        decode(ServingSLO, spec["slo"]),
        value["bundle_id"],
    )


def _accepted(root, benchmark):
    benchmark.assert_unchanged()
    store = RevisionStore(Path(root))
    context = json.loads((store.root / "context.json").read_text())
    if (
        context["target"] != benchmark.target
        or context["benchmark_sha256"] != benchmark.benchmark_sha256
    ):
        raise ValueError(
            "Accepted component belongs to another Online evaluation context"
        )
    active = store.active()
    if active is None:
        raise ValueError("No accepted Online revision")
    return active[0], json.loads((active[0].parent / "evaluation.json").read_text())


def load_accepted_cost_models(root, benchmark):
    path, report = _accepted(root, benchmark)
    cost_models = benchmark.build_cost_models(path.read_text())
    benchmark.verify_cost_models(
        cost_models,
        report["metrics"]["prediction_sha256"],
        report["metrics"]["source_ids"],
    )
    return cost_models


def load_accepted_optimizer(root, benchmark):
    from areal.autopilot.domains.elastic.evaluation.optimizer import (
        VerifiedOnlineOptimizer,
    )

    path, _ = _accepted(root, benchmark)
    return VerifiedOnlineOptimizer(
        path.read_text(),
        benchmark.settings,
        max_queries=benchmark.max_queries,
        max_seconds=benchmark.max_seconds,
    )
