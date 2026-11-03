# SPDX-License-Identifier: Apache-2.0

"""Normal-path independent oracles for Basic/Reference prediction boundaries."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from areal.elastic.contracts import (
    ActorView,
    EpisodeSample,
    EpisodeView,
    Forecast,
    HandoffConfig,
    OnlineLayout,
    OnlineObservation,
    RequestView,
    SimulationConfig,
    ToolView,
    Turn,
)
from areal.elastic.layout import ReconfigurationPlanner
from areal.elastic.reconfigurator import build_reconfigurator
from areal.elastic.reconfigurator.basic.simulator import JointSimulator
from areal.elastic.reconfigurator.reference.evaluator import Evaluator
from areal.elastic.serving import ServingSLO
from areal.elastic.simulation.simulation import OnlineSimulation


class Seconds:
    source_ids = ("independent-joint-window-oracle",)

    def rollout_config(self, layout):
        from areal.elastic.reconfigurator.reference.evaluator.cost_model.contracts import (
            RolloutConfig,
        )

        return RolloutConfig(
            layout.rollout_replicas, layout.rollout_tp, 32, 512, 100000
        )

    def service(self, layout, kv, slices):
        return sum(s.query_tokens / 8 if s.kind == "prefill" else 0.1 for s in slices)

    def actor(self, layout, batch):
        return (("train", 2.0),)

    def publication(self, layout, recipients):
        return 0.5


def inputs():
    layout = OnlineLayout(2, 1, 2, 1)
    view = OnlineObservation(
        0,
        layout,
        (),
        (),
        (),
        ActorView(),
        0,
        0,
        0,
        False,
        False,
        leases=((0, "actor", ((0, 0), (0, 1))), (1, "rollout", ((0, 2), (0, 3)))),
    )
    workload = tuple(
        EpisodeSample(f"e{i}", "oracle", 0, (Turn(8, 2),)) for i in range(2)
    )
    config = SimulationConfig(2, 2, 8, (("python", 8),), 128, 4096, 100000, 100)
    handoff = HandoffConfig(
        (8,), 2, 3, 0, 0, 1, 1, 1, 0, 0, "Explicit CPU formula priors"
    )
    return view, Forecast(0, workload, 0), config, handoff, ServingSLO(5, 0.2, 0.8)


def window(implementation, *, view=None, forecast=None, handoff=None):
    v, f, config, h, slo = inputs()
    cls = JointSimulator if implementation == "basic" else Evaluator
    factory = cls.from_forecast if implementation == "basic" else cls._from_forecast
    return factory(view or v, forecast or f, Seconds(), config, handoff or h, slo)


def advance(window, **kwargs):
    call = window._run if isinstance(window, Evaluator) else window.run
    return call(**kwargs)


def test_original_joint_aggregate_formula_is_retained():
    from areal.elastic.reconfigurator.basic.cost_model.aggregate_rollout import (
        RolloutConfig,
        RolloutCostModel,
        RolloutProfile,
        RolloutStage,
        RolloutWorkload,
    )

    cfg = RolloutConfig("m", "h", "r", "s", 2, 1, 8, 1024, 10000, 2)
    profile = RolloutProfile(cfg, 100, 20, 1, 4, 1, 1000, 1000, "closed-form")
    result = RolloutCostModel((profile,)).predict(
        cfg,
        RolloutWorkload((RolloutStage(100, 20, 1, 4),), 4, 200),
    )
    # (100/100 + 20/20)/min(2,4) = 1 GPU second, then 1 tool second.
    assert result.supported and result.stages == ((1, 1),)


def test_original_joint_batch_evaluator_is_still_executable():
    from areal.elastic.contracts import BatchWorkload, TrainingConfig, WorkloadStats
    from areal.elastic.reconfigurator.basic.cost_model.aggregate_rollout import (
        RolloutConfig,
        RolloutCostModel,
        RolloutProfile,
        RolloutStage,
        RolloutWorkload,
    )
    from areal.elastic.reconfigurator.basic.cost_model.joint_transition import (
        JointTransitionModel,
        TransitionPhase,
        TransitionProfile,
    )
    from areal.elastic.reconfigurator.basic.joint import (
        JointBatch,
        JointConfig,
        JointScenario,
    )

    rollout = RolloutConfig("m", "h", "r", "s", 2, 1, 8, 1024, 10000, 2)
    target = JointConfig(TrainingConfig(1, "oracle"), rollout)
    stats = WorkloadStats(4, 80, loss_tokens=16, sequence_lengths=(20,) * 4)
    batch = JointBatch(
        BatchWorkload(4, 16, 80, actor_stats=stats),
        RolloutWorkload((RolloutStage(100, 20, 1, 4),), 4, 200),
        4,
    )
    actors = SimpleNamespace(
        actor_config=lambda c: None,
        predict=lambda c, w: SimpleNamespace(
            seconds=2, static_memory_exceeds_capacity=False, assumptions=()
        ),
    )
    rates = RolloutCostModel(
        (RolloutProfile(rollout, 100, 20, 1, 4, 1, 1000, 1000, "oracle"),)
    )
    phases = tuple(
        TransitionProfile(
            action,
            target,
            target if action == "publish" else None,
            (TransitionPhase(seconds, 1, 2),),
            "oracle",
        )
        for action, seconds in (("publish", 0.5), ("shutdown", 0.25))
    )
    scene = JointScenario((batch,), target, 1, 2, 1, 2, 3)
    result = JointSimulator(actors, rates, JointTransitionModel(phases)).evaluate(
        scene, target
    )
    # 1 aggregate GPU + 1 tool + 2 Actor + .5 publish + .25 cleanup.
    assert result.status == "complete" and result.completed_batches == 1
    assert result.seconds == pytest.approx(4.75)
    assert result.total_gpu_seconds == pytest.approx(3 * 4.75)


def test_basic_window_obeys_independent_aggregate_timing_and_gpu_integral():
    run = window("basic")
    result = advance(run)
    # Two 8-token prompts / (8 tokens/s * 2 replicas) = 1s;
    # four output tokens * .1s / 2 = .2s; Actor2s + publication.5s.
    assert result["status"] == "complete"
    assert result["seconds"] == pytest.approx(3.7)
    assert result["total_gpu_seconds"] == pytest.approx(4 * 3.7)
    assert run.provider_version == 1 and run.consumed == 2
    assert [r.view.first_token_at for r in run.requests.values()] == pytest.approx(
        [1.1, 1.1]
    )


@pytest.mark.parametrize("implementation", ["basic", "reference"])
def test_prediction_routes_through_owned_simulator_without_experiment_kernel(
    implementation,
):
    view, forecast, config, handoff, slo = inputs()
    own = JointSimulator if implementation == "basic" else Evaluator
    evaluator = build_reconfigurator(
        implementation, candidates=(view.layout,)
    ).make_evaluator(Seconds(), config, handoff, slo)
    if implementation == "reference":
        assert isinstance(evaluator, Evaluator)
    entry = "from_forecast" if implementation == "basic" else "_from_forecast"
    with patch.object(own, entry, wraps=getattr(own, entry)) as own_call:
        with patch.object(
            OnlineSimulation, "from_forecast", wraps=OnlineSimulation.from_forecast
        ) as experiment_call:
            prediction = evaluator.evaluate(view, (forecast,), view.layout)
    assert prediction.supported and own_call.call_count == 1
    experiment_call.assert_not_called()
    candidate = window(implementation)
    assert not isinstance(candidate, OnlineSimulation)


@pytest.mark.parametrize("implementation", ["basic", "reference"])
def test_candidate_queries_leave_observed_experiment_unchanged(implementation):
    _, forecast, config, handoff, slo = inputs()
    experiment = OnlineSimulation.create(
        forecast.episodes,
        OnlineLayout(2, 1, 2, 1),
        Seconds(),
        config,
        handoff,
        slo,
        input_closes_at=10,
    )
    experiment.run(until=0.5)
    before = experiment.observe()
    evaluator = build_reconfigurator(
        implementation, candidates=(before.layout,)
    ).make_evaluator(Seconds(), config, handoff, slo)
    prediction = evaluator.evaluate(
        before, (replace(forecast, input_closes_at=10),), before.layout
    )
    assert prediction.supported
    assert experiment.observe() == before and experiment.now == 0.5
    assert prediction.completed_updates is None  # sampled/complete-work scoring


@pytest.mark.parametrize("implementation", ["basic", "reference"])
def test_existing_actor_phase_publishes_once_and_preserves_consumption(implementation):
    view, forecast, _, _, _ = inputs()
    rows = tuple(
        EpisodeView(e.episode_id, e.family, 0, True, 0.3, e.turns, None, None)
        for e in forecast.episodes
    )
    view = replace(view, now=1, episodes=rows, actor=ActorView(("e0", "e1"), 0, 0.4))
    run = window(
        implementation, view=view, forecast=replace(forecast, input_closes_at=10)
    )
    advance(run, until=3)
    assert run.training_version == run.provider_version == 1 and run.consumed == 2
    assert len(run.training_records) == 1
    assert run.training_records[0]["committed_at"] == pytest.approx(2.4)
    assert next(
        e["time"] for e in run.events if e["kind"] == "weights_published"
    ) == pytest.approx(2.9)


@pytest.mark.parametrize("implementation", ["basic", "reference"])
def test_running_tool_keeps_elapsed_time_and_next_turn_dependency(implementation):
    view, _, _, _, _ = inputs()
    episode = EpisodeSample("tool", "oracle", 0, (Turn(8, 1, 2), Turn(8, 1)))
    request = RequestView("tool:0", "tool", 0, 8, 0, 8, 1, 0.1, 0.1, 0, 0.2, 1, 0)
    observed = EpisodeView(
        "tool", "oracle", 0, True, None, (), None, ToolView("python", 0.2, 0.2)
    )
    view = replace(view, now=1, episodes=(observed,), requests=(request,))
    run = window(implementation, view=view, forecast=Forecast(0, (episode,), 10))
    advance(run, until=5)
    assert next(
        r.view.ready_at for r in run.requests.values() if r.view.turn == 1
    ) == pytest.approx(2.2)
    assert run.requests["tool:0"].view.first_token_at == 0.1
    assert len(run.episodes["tool"].past) == 2


@pytest.mark.parametrize("implementation", ["basic", "reference"])
def test_cold_candidates_use_native_plan_and_exact_residency(implementation):
    view, _, _, handoff, _ = inputs()
    handoff = replace(
        handoff,
        mode="cold",
        node_gpus=(4,),
        native_checkpoint_version=0,
        phase_seconds=tuple(
            zip(
                (
                    "checkpoint",
                    "release_actor",
                    "disconnect_weights",
                    "stop_rollout",
                    "rebuild_rollout",
                    "restore_actor",
                    "connect_weights",
                ),
                (1, 2, 1, 1, 3, 4, 1),
            )
        ),
    )
    run = window(implementation, forecast=Forecast(0, (), 100), handoff=handoff)
    target = OnlineLayout(1, 1, 3, 1)
    source = run.execution_layout if implementation == "basic" else None
    run.decision_pending = True
    run.schedule_at(0, "decision_apply", target)
    result = advance(run, stop_after_handoff=True)
    assert result["peak_gpus"] <= 4
    assert run.layout == target and result["seconds"] == pytest.approx(11.5)
    assert result["total_gpu_seconds"] == pytest.approx(41)
    if source is not None:
        assert run.native_plan == ReconfigurationPlanner().plan(
            source, run.execution_layout, actor_running=True
        )


@pytest.mark.parametrize("implementation", ["basic", "reference"])
def test_hot_candidates_count_preparation_overlap(implementation):
    view, _, _, handoff, _ = inputs()
    run = window(implementation, forecast=Forecast(0, (), 100))
    run.decision_pending = True
    run.schedule_at(0, "decision_apply", OnlineLayout(1, 1, 3, 1))
    result = advance(run, stop_after_handoff=True)
    assert 4 < result["peak_gpus"] <= 8
    assert run.switching_gpu_seconds() > 0


def test_basic_fixed_window_does_not_report_future_tokens_or_completion():
    view, _, _, _, _ = inputs()
    work = tuple(EpisodeSample(f"e{i}", "oracle", 0, (Turn(8, 100),)) for i in range(2))
    run = window("basic", forecast=Forecast(0, work, 10))
    advance(run, until=1.15)
    assert all(
        r.view.generated == 1 and r.view.completed_at is None
        for r in run.requests.values()
    )
    assert run.training_version == 0 and run.consumed == 0


@pytest.mark.parametrize("implementation", ["basic", "reference"])
def test_changing_prediction_model_does_not_change_experiment_timing(implementation):
    class Biased(Seconds):
        source_ids = Seconds.source_ids + ("biased-actor",)

        def actor(self, layout, batch):
            return (("train", 20.0),)

    view, forecast, config, handoff, slo = inputs()
    truth, predicted = Seconds(), Biased()
    reconfigurator = build_reconfigurator(implementation, candidates=(view.layout,))
    experiment = OnlineSimulation.create(
        forecast.episodes,
        view.layout,
        truth,
        config,
        handoff,
        slo,
        input_closes_at=0,
        reconfigurator=reconfigurator,
        prediction_cost_models=predicted,
    )
    evaluator = reconfigurator.bind(experiment).evaluator(0)
    assert evaluator.cost_models is predicted and experiment.cost_models is truth
    result = evaluator.evaluate(experiment.observe(), (forecast,), view.layout)
    actual = experiment.run()
    assert actual["status"] == "complete" and actual["seconds"] == pytest.approx(3.6)
    assert result.supported and result.mean_gpu_seconds > actual["total_gpu_seconds"]
