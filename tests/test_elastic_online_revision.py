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
)
from areal.autopilot.domains.elastic.cycle import ReconfiguratorRevisionCycle
from areal.autopilot.domains.elastic.prompt import proposal_context
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
        assert request.context["factory"] == (
            "build_cost_models(calibration_cases, settings)"
            if request.target == "evaluator"
            else "build_optimizer(settings)"
        )
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
    result = RevisionLoop(
        Agent(),
        benchmark,
        tmp_path / "evaluator",
        proposal_context=proposal_context("evaluator"),
    ).run(max_rounds=1)
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
        proposal_context=proposal_context("optimizer"),
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


def test_revision_history_reaches_next_round_and_resumed_observation(runner, tmp_path):
    """A measured rejection and acceptance both return to the next Agent call."""
    history_root = tmp_path / "training-001" / "revision"
    output = history_root / "cycle-001" / "optimizer"
    benchmark = demo_benchmark("optimizer", runner)
    requests = []
    sources = iter(
        (
            REDUNDANT_OPTIMIZER_SOURCE + "\n# Same search order.\n",
            BASELINE_SOURCE["optimizer"],
            BASELINE_SOURCE["optimizer"] + "\n# Same accepted search.\n",
        )
    )

    class HistoryAgent:
        def propose(self, request):
            requests.append(request)
            return Proposal(next(sources), "Try fewer Evaluator queries")

    def loop():
        return RevisionLoop(
            HistoryAgent(),
            benchmark,
            output,
            initial_source=REDUNDANT_OPTIMIZER_SOURCE,
            proposal_context=proposal_context("optimizer"),
            history_root=history_root,
        )

    first = loop().run(max_rounds=2)
    assert first["accepted"] == 1
    assert requests[0].revision_history == ()
    previous = requests[1].revision_history
    assert len(previous) == 1
    assert previous[0]["rationale"] == "Try fewer Evaluator queries"
    assert previous[0]["baseline"]["metrics"]["mean_evaluator_queries"] == 6
    assert previous[0]["outcome"]["metrics"]["mean_evaluator_queries"] == 6
    assert previous[0]["outcome"]["status"] == "rejected"

    second = loop().run(max_rounds=1)
    assert second["accepted"] == 0
    assert requests[2].round == 3
    assert len(requests[2].revision_history) == 2
    accepted = requests[2].revision_history[1]
    assert accepted["baseline"]["metrics"]["mean_evaluator_queries"] == 6
    assert accepted["outcome"]["metrics"]["mean_evaluator_queries"] == 3
    assert accepted["outcome"]["status"] == "accepted"
    assert requests[2].feedback[0]["metrics"]["mean_evaluator_queries"] == 3
    history = json.loads((history_root / "history.json").read_text())["attempts"]
    assert len(history) == 3
    assert len({item["attempt"] for item in history}) == 3
    assert all("reference_sources" not in item["context"] for item in history)
    assert all("selection_sha256" not in item["outcome"]["metrics"] for item in history)
    saved = json.loads(
        (history_root / history[-1]["attempt"] / "proposal-request.json").read_text()
    )
    assert saved["revision_history"] == history[:2]


def test_revision_history_is_shared_by_cycles_within_one_training(runner, tmp_path):
    """Explicit training roots accumulate cycle records and keep new runs separate."""
    benchmark = demo_benchmark("optimizer", runner)
    requests = []

    class HistoryAgent:
        def propose(self, request):
            requests.append(request)
            return Proposal(BASELINE_SOURCE["optimizer"], "Remove repeated evaluation")

    for training, cycle in (
        ("training-001", "cycle-001"),
        ("training-001", "cycle-002"),
        ("training-002", "cycle-001"),
    ):
        history_root = tmp_path / training / "revision"
        result = RevisionLoop(
            HistoryAgent(),
            benchmark,
            history_root / cycle / "optimizer",
            initial_source=REDUNDANT_OPTIMIZER_SOURCE,
            history_root=history_root,
        ).run(max_rounds=1)
        assert result["accepted"] == 1

    assert [len(r.revision_history) for r in requests] == [0, 1, 0]
    first = json.loads((tmp_path / "training-001/revision/history.json").read_text())[
        "attempts"
    ]
    second = json.loads((tmp_path / "training-002/revision/history.json").read_text())[
        "attempts"
    ]
    assert len(first) == 2 and len(second) == 1
    assert first[0]["attempt"].startswith("cycle-001/optimizer/")
    assert first[1]["attempt"].startswith("cycle-002/optimizer/")


def test_evaluator_optimizer_cycle_loads_full_reference_and_simulates_same_work(
    runner, tmp_path
):
    evaluator_benchmark = demo_benchmark("evaluator", runner)
    _, problems, assembly = demo_inputs()
    history_root = tmp_path / "training" / "revision"
    cycle = ReconfiguratorRevisionCycle(
        evaluator_benchmark,
        problems,
        history_root / "cycle-001",
        assembly=assembly,
        initial_optimizer_source=REDUNDANT_OPTIMIZER_SOURCE,
        optimizer_limits=dict(max_queries=12, overhead_weight=0.5),
        history_root=history_root,
    )
    result = cycle.run(Agent(), Agent(), max_evaluator_rounds=1, max_optimizer_rounds=1)
    assert (
        result["evaluator_accepted"]
        and result["optimizer_accepted"]
        and result["bundle"]
    )
    history = json.loads((history_root / "history.json").read_text())["attempts"]
    assert [item["target"] for item in history] == ["evaluator", "optimizer"]
    for item in history:
        request = json.loads(
            (history_root / item["attempt"] / "proposal-request.json").read_text()
        )
        assert request["revision_history"] == []
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
    assert (
        json.loads((history_root / "history.json").read_text())["attempts"] == history
    )
