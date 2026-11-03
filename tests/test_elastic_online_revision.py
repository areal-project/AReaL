# SPDX-License-Identifier: Apache-2.0

"""Normal Online component revision, fixed-reference search and deployable bundles."""

import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from examples.autopilot.elastic.revision_demo import (
    GOOD_EVALUATOR_SOURCE,
    REDUNDANT_OPTIMIZER_SOURCE,
    demo_benchmark,
    demo_inputs,
)

from areal.autopilot.domains.elastic.contracts import (
    BASELINE_SOURCE,
    OnlineProblem,
    decode,
    proposal_context,
)
from areal.autopilot.domains.elastic.cycle import ReconfiguratorRevisionCycle
from areal.autopilot.domains.elastic.versions import (
    load_accepted_cost_models,
    load_accepted_optimizer,
    load_bundle,
)
from areal.autopilot.revision import CandidateRunner, Proposal, RevisionLoop
from areal.autopilot.revision.execution import load_source
from areal.elastic.contracts import HandoffConfig, SimulationConfig
from areal.elastic.serving import ServingSLO
from areal.elastic.simulation.simulation import OnlineSimulation


@pytest.fixture
def runner():
    root = str(Path(__file__).resolve().parents[1])
    bootstrap = f"import sys,types,runpy; p=types.ModuleType('areal'); p.__path__=[{root!r}+'/areal'];sys.modules['areal']=p;runpy.run_module('areal.autopilot.domains.elastic.evaluation.worker',run_name='__main__')"
    return CandidateRunner(command=(sys.executable, "-c", bootstrap))


class Agent:
    def propose(self, request):
        return Proposal(
            GOOD_EVALUATOR_SOURCE
            if request.target == "evaluator"
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


def test_evaluator_revision_scores_private_history_and_loads_same_cost_models(
    runner, tmp_path
):
    benchmark = demo_benchmark("evaluator", runner)
    result = RevisionLoop(Agent(), benchmark, tmp_path / "evaluator").run(max_rounds=1)
    assert result["active_source"] is not None
    cost_models = load_accepted_cost_models(tmp_path / "evaluator", benchmark)
    case = benchmark.history.prediction_inputs[0]
    assert cost_models.actor(case.layout, case.batch) == load_source(
        GOOD_EVALUATOR_SOURCE
    ).build_cost_models((), {}).actor(case.layout, case.batch)
    request = json.loads(
        next((tmp_path / "evaluator").glob("attempts/round-*/request.json")).read_text()
    )
    assert all("measured" not in c for c in request["cases"])
    assert all(c["split"] == "calibration" for c in request["calibration"]["cases"])
    assert len(request["cases"]) == 7  # final holdout is never predicted by revision.


def test_optimizer_revision_reduces_parent_counted_queries_and_loads_same_search(
    runner, tmp_path
):
    benchmark = demo_benchmark("optimizer", runner)
    result = RevisionLoop(
        Agent(),
        benchmark,
        tmp_path / "optimizer",
        initial_source=REDUNDANT_OPTIMIZER_SOURCE,
    ).run(max_rounds=1)
    assert result["active_source"] is not None
    baseline = json.loads((tmp_path / "optimizer/baseline.json").read_text())
    report = json.loads(
        (Path(result["active_source"]).parent / "evaluation.json").read_text()
    )
    assert baseline["metrics"]["mean_evaluator_queries"] == 6
    assert report["metrics"]["mean_evaluator_queries"] == 3
    search = load_accepted_optimizer(tmp_path / "optimizer", benchmark)
    case = benchmark.cases[0]
    predictions = search.search(
        benchmark.evaluator_factory(), case.observation, case.forecasts, case.candidates
    )
    target = search.select(predictions, case.observation.layout, 0.8)
    assert asdict(target.layout) == benchmark._oracle[0].layout.__dict__


def test_evaluator_optimizer_cycle_loads_full_reference_and_simulates_same_work(
    runner, tmp_path
):
    evaluator_benchmark = demo_benchmark("evaluator", runner)
    _, problems, assembly = demo_inputs()
    cycle = ReconfiguratorRevisionCycle(
        evaluator_benchmark,
        problems,
        tmp_path / "cycle",
        assembly=assembly,
        initial_optimizer_source=REDUNDANT_OPTIMIZER_SOURCE,
        optimizer_limits=dict(max_queries=12, overhead_weight=0.5),
    )
    result = cycle.run(Agent(), Agent(), max_evaluator_rounds=1, max_optimizer_rounds=1)
    assert (
        result["evaluator_accepted"]
        and result["optimizer_accepted"]
        and result["bundle"]
    )
    loaded = load_bundle(Path(result["bundle"]))
    bundle = json.loads(Path(result["bundle"]).read_text())
    assert bundle["format_version"] == 2
    assert set(bundle["components"]) == {"evaluator", "optimizer"}
    assert loaded.reconfigurator.implementation == "reference"
    case = problems[0]
    candidate_evaluator = loaded.reconfigurator.make_evaluator(
        loaded.cost_models, loaded.config, loaded.handoff, loaded.slo
    )
    predictions = candidate_evaluator.search(
        case.observation, case.forecasts, case.candidates
    )
    assert len(predictions) == len(case.candidates)
    first = OnlineSimulation(
        case.forecasts[0].episodes,
        case.observation.layout,
        load_source(GOOD_EVALUATOR_SOURCE).build_cost_models((), {}),
        loaded.config,
        loaded.handoff,
        loaded.slo,
        input_closes_at=20,
        prediction_cost_models=loaded.cost_models,
    ).run()
    oracle = OnlineSimulation(
        case.forecasts[0].episodes,
        case.observation.layout,
        load_source(GOOD_EVALUATOR_SOURCE).build_cost_models((), {}),
        decode(SimulationConfig, assembly["run"]),
        decode(HandoffConfig, assembly["handoff"]),
        decode(ServingSLO, assembly["slo"]),
        input_closes_at=20,
    ).run()
    assert first["status"] == oracle["status"] == "complete"
    assert first["seconds"] == oracle["seconds"]
    assert first["total_gpu_seconds"] == oracle["total_gpu_seconds"]
    assert (
        cycle.run(Agent(), Agent(), max_evaluator_rounds=1, max_optimizer_rounds=1)
        == result
    )
