# SPDX-License-Identifier: Apache-2.0

"""Normal Online component revision, fixed-reference search and deployable bundles."""

import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from examples.autopilot.elastic.revision_demo import (
    GOOD_MODEL_SOURCE,
    REDUNDANT_OPTIMIZER_SOURCE,
    demo_evaluator,
    demo_inputs,
)

from areal.autopilot.domains.elastic.bundle import load_bundle
from areal.autopilot.domains.elastic.contracts import (
    BASELINE_SOURCE,
    OnlineProblem,
    decode,
    proposal_context,
)
from areal.autopilot.domains.elastic.cycle import ReconfiguratorRevisionCycle
from areal.autopilot.domains.elastic.loaders import (
    load_accepted_model,
    load_accepted_optimizer,
)
from areal.autopilot.revision import CandidateRunner, Proposal, RevisionLoop
from areal.autopilot.revision.execution import load_source
from areal.elastic.serving import ServingSLO
from areal.elastic.simulation.online_simulation import OnlineSimulation
from areal.elastic.simulation.online_transitions import HandoffConfig
from areal.elastic.simulation.simulation_core import SimulationConfig


@pytest.fixture
def runner():
    root = str(Path(__file__).resolve().parents[1])
    bootstrap = f"import sys,types,runpy; p=types.ModuleType('areal'); p.__path__=[{root!r}+'/areal'];sys.modules['areal']=p;runpy.run_module('areal.autopilot.domains.elastic.worker',run_name='__main__')"
    return CandidateRunner(command=(sys.executable, "-c", bootstrap))


class Agent:
    def propose(self, request):
        return Proposal(
            GOOD_MODEL_SOURCE
            if request.target == "model"
            else BASELINE_SOURCE["optimizer"]
        )


def test_online_revision_json_roundtrip_preserves_frozen_work():
    _, problems, _ = demo_inputs()
    assert (
        decode(OnlineProblem, json.loads(json.dumps(asdict(problems[0]))))
        == problems[0]
    )
    assert "reference/optimizer.py" in " ".join(
        proposal_context("optimizer")["reference_sources"]
    )


def test_model_revision_scores_private_history_and_loads_same_online_components(
    runner, tmp_path
):
    evaluator = demo_evaluator("model", runner)
    result = RevisionLoop(Agent(), evaluator, tmp_path / "model").run(max_rounds=1)
    assert result["active_source"] is not None
    models = load_accepted_model(tmp_path / "model", evaluator)
    case = evaluator.history.prediction_inputs[0]
    assert models.actor(case.layout, case.batch) == load_source(
        GOOD_MODEL_SOURCE
    ).build_model((), {}).actor(case.layout, case.batch)
    request = json.loads(
        next((tmp_path / "model").glob("attempts/round-*/request.json")).read_text()
    )
    assert all("measured" not in c for c in request["cases"])
    assert all(c["split"] == "calibration" for c in request["calibration"]["cases"])
    assert len(request["cases"]) == 7  # final holdout is never predicted by revision.


def test_optimizer_revision_reduces_parent_counted_queries_and_loads_same_search(
    runner, tmp_path
):
    evaluator = demo_evaluator("optimizer", runner)
    result = RevisionLoop(
        Agent(),
        evaluator,
        tmp_path / "optimizer",
        initial_source=REDUNDANT_OPTIMIZER_SOURCE,
    ).run(max_rounds=1)
    assert result["active_source"] is not None
    baseline = json.loads((tmp_path / "optimizer/baseline.json").read_text())
    report = json.loads(
        (Path(result["active_source"]).parent / "evaluation.json").read_text()
    )
    assert baseline["metrics"]["mean_model_queries"] == 6
    assert report["metrics"]["mean_model_queries"] == 3
    search = load_accepted_optimizer(tmp_path / "optimizer", evaluator)
    case = evaluator.cases[0]
    predictions = search.search(
        evaluator.model_factory(), case.observation, case.forecasts, case.candidates
    )
    target = search.select(predictions, case.observation.layout, 0.8)
    assert asdict(target.layout) == evaluator._oracle[0].layout.__dict__


def test_model_optimizer_cycle_loads_full_reference_and_simulates_same_work(
    runner, tmp_path
):
    model = demo_evaluator("model", runner)
    _, problems, assembly = demo_inputs()
    cycle = ReconfiguratorRevisionCycle(
        model,
        problems,
        tmp_path / "cycle",
        assembly=assembly,
        initial_optimizer_source=REDUNDANT_OPTIMIZER_SOURCE,
        optimizer_limits=dict(max_queries=12, overhead_weight=0.5),
    )
    result = cycle.run(Agent(), Agent(), max_model_rounds=1, max_optimizer_rounds=1)
    assert (
        result["model_accepted"] and result["optimizer_accepted"] and result["bundle"]
    )
    loaded = load_bundle(Path(result["bundle"]))
    assert loaded.reconfigurator.implementation == "reference"
    case = problems[0]
    model = loaded.reconfigurator.make_model(
        loaded.models, loaded.config, loaded.handoff, loaded.slo
    )
    predictions = model.search(case.observation, case.forecasts, case.candidates)
    assert len(predictions) == len(case.candidates)
    first = OnlineSimulation(
        case.forecasts[0].episodes,
        case.observation.layout,
        load_source(GOOD_MODEL_SOURCE).build_model((), {}),
        loaded.config,
        loaded.handoff,
        loaded.slo,
        input_closes_at=20,
        prediction_models=loaded.models,
    ).run()
    oracle = OnlineSimulation(
        case.forecasts[0].episodes,
        case.observation.layout,
        load_source(GOOD_MODEL_SOURCE).build_model((), {}),
        decode(SimulationConfig, assembly["run"]),
        decode(HandoffConfig, assembly["handoff"]),
        decode(ServingSLO, assembly["slo"]),
        input_closes_at=20,
    ).run()
    assert first["status"] == oracle["status"] == "complete"
    assert first["seconds"] == oracle["seconds"]
    assert first["total_gpu_seconds"] == oracle["total_gpu_seconds"]
    assert (
        cycle.run(Agent(), Agent(), max_model_rounds=1, max_optimizer_rounds=1)
        == result
    )
