# SPDX-License-Identifier: Apache-2.0

"""Accepted Online Reference factories and shared native/simulation assembly."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from areal.autopilot.revision.contracts import digest, source_digest
from areal.autopilot.revision.execution import load_source
from areal.elastic.contracts import OnlineLayout
from areal.elastic.infra.activation import PreparedReconfigurator
from areal.elastic.infra.activation import ReconfiguratorApplication as ActivationStore
from areal.elastic.reconfigurator import build_reconfigurator
from areal.utils.atomic import atomic_json

from .contracts import decode, deployment_contract
from .identity import AREAL_ROOT, protected_files
from .model_adapters import OnlineHistoryPredictor, validate_components
from .model_evaluator import HistoricalCase, HistoricalInput
from .model_execution import prediction_fingerprint
from .optimizer_evaluator import VerifiedOnlineOptimizer


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
    model_evaluator,
    model_stage,
    optimizer_evaluator,
    optimizer_stage,
    assembly,
    parent_bundle_id=None,
):
    model_evaluator.assert_unchanged()
    optimizer_evaluator.assert_unchanged()
    if not model_stage.record["accepted"] and not optimizer_stage.record["accepted"]:
        return None
    payload = dict(
        format_version=1,
        kind="online-reference-reconfigurator-v1",
        implementation="reference",
        assembly=assembly,
        model_settings=model_evaluator.settings,
        calibration_cases=model_evaluator.calibration["cases"],
        model_probes=[asdict(i) for i in model_evaluator.history.prediction_inputs],
        components=dict(
            model=_stage_evidence(model_stage),
            optimizer=_stage_evidence(optimizer_stage),
        ),
        model_source_ids=json.loads(
            (model_stage.root / model_stage.record["report"]).read_text()
        )["metrics"]["source_ids"],
        optimizer_settings=optimizer_evaluator.settings,
        optimizer_limits=dict(
            max_queries=optimizer_evaluator.max_queries,
            max_seconds=optimizer_evaluator.max_seconds,
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
        value.get("kind") != "online-reference-reconfigurator-v1"
        or value.get("implementation") != "reference"
        or value.get("format_version") != 1
        or value.get("bundle_id") != digest(payload)
    ):
        raise ValueError("Unsupported or changed Online bundle")
    components = value["components"]
    if set(components) != {"model", "optimizer"} or not any(
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
    model_context = components["model"]["context"]["deployment_context"]
    optimizer_context = components["optimizer"]["context"]["deployment_context"]
    if (
        model_context["settings"] != value["model_settings"]
        or optimizer_context.get("assembly_sha256") != digest(value["assembly"])
        or optimizer_context.get("model_source_sha256") != components["model"]["sha256"]
        or optimizer_context.get("model_prediction_sha256")
        != components["model"]["report"]["metrics"]["prediction_sha256"]
        or optimizer_context.get("model_settings_sha256")
        != digest(value["model_settings"])
    ):
        raise ValueError(
            "Model and optimizer were not validated under this shared Online contract"
        )
    if (
        optimizer_context.get("optimizer_settings") != value["optimizer_settings"]
        or optimizer_context.get("optimizer_limits") != value["optimizer_limits"]
        or value["model_source_ids"]
        != components["model"]["report"]["metrics"]["source_ids"]
    ):
        raise ValueError(
            "Online factory settings, limits or model data identities changed"
        )
    if not value["model_probes"] or not value["protected_files"]:
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
    models: object
    config: object
    handoff: object
    slo: object
    bundle_id: str


def load_bundle(path):
    from areal.elastic.serving import ServingSLO
    from areal.elastic.simulation.online_transitions import HandoffConfig
    from areal.elastic.simulation.simulation_core import SimulationConfig

    value = json.loads(Path(path).read_text())
    validate_bundle(value)
    components = value["components"]
    models = validate_components(
        load_source(components["model"]["source"]).build_model(
            tuple(decode(HistoricalCase, c) for c in value["calibration_cases"]),
            value["model_settings"],
        )
    )
    if tuple(models.source_ids) != tuple(value["model_source_ids"]):
        raise ValueError("Loaded Online model has different frozen data identities")
    predictor = OnlineHistoryPredictor(models)
    expected = components["model"]["report"]["metrics"]["prediction_sha256"]
    for _ in range(2):
        rows = [
            dict(prediction=asdict(predictor.predict(decode(HistoricalInput, i))))
            for i in value["model_probes"]
        ]
        if prediction_fingerprint(rows) != expected:
            raise ValueError("Loaded Online model differs from evaluated predictions")
    spec = value["assembly"]
    optimizer = VerifiedOnlineOptimizer(
        components["optimizer"]["source"],
        value["optimizer_settings"],
        **value["optimizer_limits"],
    )
    reconfigurator = build_reconfigurator(
        "reference",
        candidates=tuple(decode(OnlineLayout, c) for c in spec["candidates"]),
        settings=spec.get("reconfigurator"),
        replay=spec.get("replay"),
        forecast=spec.get("forecast"),
        slo=spec["slo"],
        optimizer=optimizer,
    )
    validate_bundle(value)
    return LoadedOnlineBundle(
        reconfigurator,
        models,
        decode(SimulationConfig, spec["run"]),
        decode(HandoffConfig, spec["handoff"]),
        decode(ServingSLO, spec["slo"]),
        value["bundle_id"],
    )


class ReconfiguratorApplication(ActivationStore):
    """Native activation store loads the exact same complete Online bundle."""

    def __init__(
        self,
        root,
        *,
        compatibility,
        execution_context,
        initial_bundle_id="constructor-reconfigurator",
    ):
        super().__init__(
            root,
            compatibility=compatibility,
            execution_context=execution_context,
            initial_bundle_id=initial_bundle_id,
            prepare_bundle=self._prepare_online,
            validate_bundle=validate_bundle,
        )

    def _prepare_online(self, path):
        value = json.loads(Path(path).read_text())
        validate_bundle(value)
        deployment = value["assembly"].get("deployment", {})
        if deployment.get("compatibility") != self.compatibility or deployment.get(
            "native_runtime_id"
        ) != self.execution_context.get("native_runtime_id"):
            raise ValueError(
                "Online bundle belongs to another native execution contract"
            )
        if self.execution_context.get("online_contract_sha256") != digest(
            deployment_contract(value["assembly"])
        ):
            raise ValueError(
                "Online bundle changes the native workload/candidate contract"
            )
        loaded = load_bundle(path)
        directory = self.root / "bundles" / loaded.bundle_id
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(directory / "bundle.json", value)
        return PreparedReconfigurator(
            json.dumps(value, sort_keys=True), self.context, loaded
        )
