# SPDX-License-Identifier: Apache-2.0

"""Native CPU coordinator uses the complete reference and accepted Online factories."""

from dataclasses import replace

import pytest

from examples.autopilot.elastic.revision_demo import (
    GOOD_MODEL_SOURCE,
    REDUNDANT_OPTIMIZER_SOURCE,
    demo_evaluator,
    demo_inputs,
)
from tests.test_elastic_checkpoint import (
    _complete_native_update,
    _ready_native_episodes,
)
from tests.test_elastic_checkpoint import native_lifecycle as native_lifecycle
from tests.test_elastic_online_revision import runner as runner

from areal.autopilot.domains.elastic.contracts import decode
from areal.autopilot.domains.elastic.cycle import ReconfiguratorRevisionCycle
from areal.autopilot.domains.elastic.native import RevisionAdapter
from areal.autopilot.revision import Proposal
from areal.autopilot.revision.execution import load_source
from areal.elastic.infra.activation import ReconfiguratorRequest
from areal.elastic.infra.adapter import ReconfiguratorAdapter
from areal.elastic.simulation.online_transitions import HandoffConfig
from areal.elastic.simulation.simulation_core import SimulationConfig


@pytest.mark.usefixtures("native_lifecycle")
@pytest.mark.parametrize("implementation", ["basic", "reference"])
def test_native_uses_selected_complete_online_and_existing_observations(
    request, implementation
):
    f = request.getfixturevalue("native_lifecycle")
    f.manager.close()
    f.config.actor_lifecycle.reconfigurator = implementation
    adapter = ReconfiguratorAdapter()
    manager = f.create(reconfigurator_adapter=adapter)
    assert adapter.reconfigurator.implementation == implementation
    _ready_native_episodes(manager, 0, 0)
    manager.next_batch()
    manager.actor.updates += 1
    _complete_native_update(manager, 1)
    view = adapter.observe()
    assert (
        view.training_version == view.provider_version == 1
        and view.consumed_episodes == 4
    )
    assert (
        not view.requests
    )  # No per-token observations are synthesized from response chunks.
    assert all(e.completed_turns for e in view.episodes)
    assert not hasattr(manager, "resource_planner")
    manager.close()


@pytest.mark.usefixtures("native_lifecycle", "runner")
def test_native_activates_same_online_bundle_and_continues_published_training(
    request, tmp_path
):
    fixture = request.getfixturevalue("native_lifecycle")
    cpu_runner = request.getfixturevalue("runner")

    class Agent:
        def propose(self, request):
            from areal.autopilot.domains.elastic.contracts import BASELINE_SOURCE

            return Proposal(
                GOOD_MODEL_SOURCE
                if request.target == "model"
                else BASELINE_SOURCE["optimizer"]
            )

    _, problems, example = demo_inputs()
    models = load_source(GOOD_MODEL_SOURCE).build_model((), {})
    adapter = RevisionAdapter(
        models=models,
        run_config=decode(SimulationConfig, example["run"]),
        handoff=replace(decode(HandoffConfig, example["handoff"]), mode="cold"),
    )
    fixture.manager.close()
    manager = fixture.create(reconfigurator_adapter=adapter)
    _ready_native_episodes(manager, 0, 0)
    manager.next_batch()
    manager.actor.updates += 1
    _complete_native_update(manager, 1)
    # The committed native assembly is the activation contract; held-out model work remains explicit.
    from areal.elastic.simulation.online_simulation import OnlineSimulation

    p = problems[0]
    observed = OnlineSimulation.create(
        p.forecasts[0].episodes,
        adapter.placements.current(),
        models,
        adapter.run_config,
        adapter.handoff,
        adapter.reconfigurator.slo,
        input_closes_at=20,
    ).observe()
    problems = (
        replace(p, observation=observed, candidates=adapter.reconfigurator.candidates),
    )
    evaluator = demo_evaluator("model", cpu_runner)
    cycle = ReconfiguratorRevisionCycle(
        evaluator,
        problems,
        tmp_path / "online-cycle",
        assembly=adapter.assembly(),
        initial_optimizer_source=REDUNDANT_OPTIMIZER_SOURCE,
        optimizer_limits=dict(max_queries=12, overhead_weight=0.5),
    )
    result = cycle.run(Agent(), Agent(), max_model_rounds=1, max_optimizer_rounds=1)
    old_actor = manager.actor
    forecaster = adapter.reconfigurator.forecaster
    prepared = adapter.prepare_reconfigurator(result["bundle"])
    receipt = adapter.apply_reconfigurator(
        ReconfiguratorRequest("online-accept", prepared.bundle_id, 0, 1), prepared
    )
    assert receipt["status"] == "applied" and adapter.reconfigurator_generation == 1
    assert (
        adapter.reconfigurator.forecaster is forecaster and manager.actor is old_actor
    )
    assert (
        adapter.reconfigurator.optimizer.source
        == prepared.reconfigurator.reconfigurator.optimizer.source
    )
    _ready_native_episodes(manager, 4, 1)
    manager.next_batch()
    manager.actor.updates += 1
    _complete_native_update(manager, 2)
    assert (
        manager.applied_version == manager.published_version == 2
        and manager.actor.updates == 2
    )
    assert manager.consumer.cursor["through_sequence"] == 2
    assert manager.data_progress["samples"] == 8
    manager.close()
    resumed_adapter = RevisionAdapter(bundle=result["bundle"])
    resumed = fixture.create(reconfigurator_adapter=resumed_adapter)
    try:
        assert resumed_adapter.reconfigurator_generation == 1
        assert resumed.applied_version == resumed.checkpoint_version == 2
        assert resumed_adapter.bundle_id == prepared.bundle_id
        _ready_native_episodes(resumed, 8, 2)
        resumed.next_batch()
        resumed.actor.updates += 1
        _complete_native_update(resumed, 3)
        assert resumed.applied_version == resumed.published_version == 3
        assert resumed.data_progress["samples"] == 12
    finally:
        resumed.close()


@pytest.mark.usefixtures("native_lifecycle")
@pytest.mark.parametrize("native_lifecycle", [(1, 4)], indirect=True)
@pytest.mark.parametrize("implementation", ["basic", "reference"])
def test_native_and_counterfactual_use_same_reconfigurator_and_cold_plan(
    request, tmp_path, implementation
):
    """Controlled serving observations test decision/execution parity, not GPU timing."""
    import json
    from dataclasses import asdict

    from tests.test_elastic_online_simulation import _KnownCosts, config, handoff

    from areal.elastic.contracts import RequestView
    from areal.elastic.reconfigurator import build_reconfigurator, component_module

    fixture = request.getfixturevalue("native_lifecycle")
    fixture.manager.close()
    settings = dict(
        minimum_completed_requests=1,
        decision_delay_seconds=0,
        deduplicate_searches=False,
    )
    replay = dict(history_seconds=120, initial_regime_seconds=120, max_new_tokens=32)
    phases = tuple(
        (name, 0.01)
        for name in (
            "checkpoint",
            "release_actor",
            "disconnect_weights",
            "stop_rollout",
            "rebuild_rollout",
            "restore_actor",
            "connect_weights",
        )
    )
    adapter = ReconfiguratorAdapter(
        implementation=implementation,
        settings=settings,
        replay=replay,
        models=_KnownCosts(),
        run_config=config(),
        handoff=replace(handoff(), mode="cold", phase_seconds=phases),
    )
    manager = fixture.create(reconfigurator_adapter=adapter)
    try:
        _ready_native_episodes(manager, 0, 0)
        manager.next_batch()
        manager.actor.updates += 1
        _complete_native_update(manager, 1)
        base = adapter.observe()
        # Inject four internally causal test requests; production missing TTFT/TBT stays unknown.
        episodes = tuple(
            replace(e, family="fixture", available_at=190 + i, completed_at=199.1)
            for i, e in enumerate(base.episodes)
        )
        requests = tuple(
            RequestView(
                f"{e.episode_id}/fixture-0",
                e.episode_id,
                0,
                e.completed_turns[0].input_tokens,
                192,
                prefilled=e.completed_turns[0].input_tokens,
                generated=e.completed_turns[0].output_tokens,
                first_token_at=199,
                last_token_at=199.09,
                maximum_tbt_seconds=0.09,
                completed_at=199.1,
                deployment=1,
                replica=0,
                version_counts=((0, e.completed_turns[0].output_tokens),),
            )
            for e in episodes
        )
        snapshot = replace(base, now=200, episodes=episodes, requests=requests)
        adapter.observation_provider = lambda owner: snapshot
        native_handoff = adapter.environment.handoff
        native_model = adapter.reconfigurator.make_model(
            adapter.models,
            adapter.run_config,
            native_handoff,
            adapter.reconfigurator.slo,
        )
        native_decision = adapter.reconfigurator.decide(snapshot, native_model)
        independent = build_reconfigurator(
            implementation,
            candidates=adapter.reconfigurator.candidates,
            settings=settings,
            replay=replay,
            slo=adapter.reconfigurator.slo,
        )
        sim_model = independent.make_model(
            adapter.models, adapter.run_config, native_handoff, independent.slo
        )
        simulated_decision = independent.decide(snapshot, sim_model)
        assert native_decision.target == simulated_decision.target != snapshot.layout
        assert [
            (p.layout, p.supported, p.mean_gpu_seconds)
            for p in native_decision.predictions
        ] == [
            (p.layout, p.supported, p.mean_gpu_seconds)
            for p in simulated_decision.predictions
        ]
        forecasts = independent.forecaster.predict(snapshot, 0)
        if implementation == "basic":
            cls = component_module(implementation, "simulator").JointSimulator
        else:
            from areal.elastic.reconfigurator.reference.evaluator import Evaluator

            cls = Evaluator
        factory = cls.from_forecast if implementation == "basic" else cls._from_forecast
        branch = factory(
            snapshot,
            forecasts[0],
            adapter.models,
            adapter.run_config,
            native_handoff,
            independent.slo,
        )
        branch.decision_pending = True
        branch.schedule_at(snapshot.now, "decision_apply", native_decision.target)
        prediction = (branch.run if implementation == "basic" else branch._run)(
            stop_after_handoff=True
        )
        from areal.elastic.layout import ExecutionLayout, ReconfigurationPlan

        if implementation == "basic":
            expected = branch.native_plan
        else:
            emitted = next(
                e["plan"] for e in branch.events if e["kind"] == "cold_handoff_start"
            )
            expected = ReconfigurationPlan(
                ExecutionLayout.decode(emitted["source"]),
                ExecutionLayout.decode(emitted["target"]),
                emitted["actor_running"],
                tuple(emitted["steps"]),
            )
        assert prediction["peak_gpus"] <= len(manager.resource_reservation)
        assert adapter.environment.request_handoff(native_decision.target)
        assert manager.last_reconfiguration_plan == expected
        assert adapter.placements.current() == native_decision.target
        adapter.observation_provider = None
        _ready_native_episodes(manager, 4, 1)
        manager.next_batch()
        manager.actor.updates += 1
        _complete_native_update(manager, 2)
        assert manager.applied_version == manager.published_version == 2
        (tmp_path / "same-state-cold.json").write_text(
            json.dumps(
                dict(
                    observation=asdict(snapshot),
                    handoff=asdict(native_handoff),
                    decision=asdict(native_decision),
                    plan=asdict(expected),
                    receipt=adapter.last_receipt,
                    simulated_peak=prediction["peak_gpus"],
                ),
                indent=2,
            )
        )
    finally:
        manager.close()
