# SPDX-License-Identifier: Apache-2.0

"""Normal hot/cold execution with explicit transition timing and a hard pool."""

from dataclasses import replace

import pytest

from tests.test_elastic_online_simulation import (
    _DecisionClock,
    _KnownCosts,
    config,
    episodes,
    handoff,
)

from areal.elastic.contracts import EpisodeSample, OnlineLayout, Turn
from areal.elastic.layout import ReconfigurationPlanner
from areal.elastic.reconfigurator import build_reconfigurator
from areal.elastic.serving import ServingSLO
from areal.elastic.simulation.simulation import OnlineSimulation


@pytest.mark.parametrize("measured_search_time", [False, True])
def test_cold_control_observes_rebuilding_rollout_and_finishes_same_work(
    monkeypatch, measured_search_time
):
    """Periodic control continues while old engines have released their GPUs."""
    initial, target = OnlineLayout(2, 1, 2, 1), OnlineLayout(1, 1, 1, 1)
    slo = ServingSLO(5, 0.05, 0.8)
    reconfigurator = build_reconfigurator(
        "reference",
        candidates=(initial, target),
        replay={
            "history_seconds": 5,
            "feature_window_seconds": 5,
            "initial_regime_seconds": 5,
        },
        settings={
            "check_interval_seconds": 2,
            "slo_window_seconds": 5,
            "decision_delay_seconds": 0.05,
            "minimum_completed_requests": 2,
        },
        slo=slo,
    )
    sim = OnlineSimulation.create(
        episodes(32),
        initial,
        _KnownCosts(),
        config(),
        cold_handoff(pool=8),
        slo,
        input_closes_at=31,
        reconfigurator=reconfigurator,
        measured_search_time=measured_search_time,
        search_clock=_DecisionClock(0.001),
    )
    observe, rebuilding = sim.observe, []

    def observed():
        view = observe()
        if sim.rollout_id not in sim.deployments:
            assert view.busy and not view.services
            rebuilding.append(view)
        return view

    monkeypatch.setattr(sim, "observe", observed)
    result = sim.run()
    assert sim.control is reconfigurator and rebuilding
    assert result["status"] == "complete"
    assert result["completed_episodes"] == result["trained_episodes"] == 32
    assert result["completed_updates"] == result["published_version"] == 8
    assert result["peak_gpus"] <= 8
    assert sim.control.stats["applied"] >= 1


class _Seconds(_KnownCosts):
    def service(self, layout, kv, slices):
        return 1.0

    def publication(self, layout, recipients):
        return 0.5


def cold_handoff(pool=4):
    return replace(
        handoff(),
        mode="cold",
        node_gpus=(pool,),
        native_checkpoint_version=1,
        phase_seconds=(
            ("checkpoint", 1.0),
            ("release_actor", 2.0),
            ("disconnect_weights", 1.0),
            ("stop_rollout", 1.0),
            ("rebuild_rollout", 3.0),
            ("restore_actor", 4.0),
            ("connect_weights", 1.0),
        ),
    )


def make(workload=(), *, initial=OnlineLayout(2, 1, 2, 1), pool=4):
    sim = OnlineSimulation.create(
        workload,
        initial,
        _Seconds(),
        config(),
        cold_handoff(pool),
        ServingSLO(5, 0.1, 0.8),
        input_closes_at=100,
    )
    sim.training_version = sim.provider_version = 1
    for d in sim.deployments.values():
        d.version = 1
    return sim


def test_cold_plan_releases_before_deploy_and_integrates_actual_gpu_leases():
    sim = make()
    target = OnlineLayout(1, 1, 3, 1)
    source = sim.execution_layout
    assert sim.request_handoff(target)
    plan = sim.pending_handoff["plan"]
    assert plan == ReconfigurationPlanner().plan(
        source, plan.target, actor_running=True
    )
    result = sim.run(stop_after_handoff=True)
    assert sim.layout == target and result["peak_gpus"] == 4
    assert result["seconds"] == pytest.approx(11.5)
    # 4*2 release + 2*1 stop + 3*3 rollout boot + 4*4 actor restore + 4*1 connect + 4*.5 publish.
    assert result["total_gpu_seconds"] == pytest.approx(41)
    starts = [e["phase"] for e in sim.events if e["kind"] == "cold_phase_start"]
    assert starts == list(plan.steps)
    assert (
        starts.index("release_actor")
        < starts.index("stop_rollout")
        < starts.index("rebuild_rollout")
        < starts.index("restore_actor")
    )
    assert sim.switching_gpu_seconds() == 0


def test_cold_drain_finishes_whole_model_call_and_keeps_tool_continuation():
    workload = (EpisodeSample("multi", "tools", 0, (Turn(8, 2, 2, 8), Turn(16, 2))),)
    sim = make(workload)
    sim.run(until=0.5)
    assert sim.request_handoff(OnlineLayout(1, 1, 3, 1))
    sim.run(stop_after_handoff=True)
    first = (
        sim.requests["multi:0"].view
        if "multi:0" in sim.requests
        else next(r.view for r in sim.requests.values() if r.view.turn == 0)
    )
    second = next(r.view for r in sim.requests.values() if r.view.turn == 1)
    phases = [e for e in sim.events if e["kind"] == "cold_phase_start"]
    assert next(
        e["time"] for e in phases if e["phase"] == "release_actor"
    ) == pytest.approx(first.completed_at)
    assert first.completed_at == pytest.approx(2)
    assert second.ready_at == pytest.approx(4)
    assert (
        second.first_token_at is None
    )  # Tool completed while the model gate remained closed.
    result = sim.run()
    assert result["status"] == "complete"
    assert sim.requests[second.request_id].view.completed_at > 13.5
    assert result["peak_gpus"] == 4 and result["completed_episodes"] == 1


def test_cold_rollout_only_keeps_actor_generation():
    sim = make(pool=5)
    actor_id = sim.actor_id
    assert sim.request_handoff(OnlineLayout(2, 1, 3, 1))
    result = sim.run(stop_after_handoff=True)
    assert sim.actor_id == actor_id and sim.deployments[actor_id].status == "active"
    plan = next(e["plan"] for e in sim.events if e["kind"] == "cold_handoff_start")
    assert (
        "release_actor" not in plan["steps"] and "disconnect_weights" in plan["steps"]
    )
    assert result["peak_gpus"] == 5


def test_explicit_hot_keeps_the_existing_overlap_execution():
    initial = OnlineLayout(2, 1, 2, 1)
    sim = OnlineSimulation.create(
        (),
        initial,
        _Seconds(),
        config(),
        handoff(),
        ServingSLO(5, 0.1, 0.8),
        input_closes_at=100,
    )
    assert type(sim) is OnlineSimulation
    assert sim.request_handoff(OnlineLayout(1, 1, 3, 1))
    result = sim.run(stop_after_handoff=True)
    assert result["peak_gpus"] > 4 and sim.switching_gpu_seconds() > 0


def test_cold_native_plan_rebuilds_actor_when_pool_changes_at_same_dp():
    from areal.elastic.contracts import NativePlacement
    from areal.elastic.layout import ExecutionLayout

    initial = OnlineLayout(1, 1, 1, 1)
    target = OnlineLayout(1, 1, 2, 1)
    source = ExecutionLayout(1, (0, 1, 2, 3), (4,))
    physical = ExecutionLayout(1, (0,), (4, 1))
    h = replace(
        cold_handoff(5),
        native_source=source,
        native_devices=(0, 1, 2, 3, 4),
        native_placements=(
            NativePlacement(initial, source),
            NativePlacement(target, physical),
        ),
    )
    sim = OnlineSimulation.create(
        (),
        initial,
        _Seconds(),
        config(),
        h,
        ServingSLO(5, 0.1, 0.8),
        input_closes_at=100,
    )
    sim.training_version = sim.provider_version = 1
    assert sim.pool.leases["1"][1] == ((0, 4),)
    assert sim.request_handoff(target)
    assert "release_actor" in sim.pending_handoff["plan"].steps
    result = sim.run(stop_after_handoff=True)
    assert sim.execution_layout == physical and sim.actor_id != 0
    assert sim.pool.leases[str(sim.rollout_id)][1] == ((0, 4), (0, 1))
    assert result["peak_gpus"] == 3 and result["total_gpu_seconds"] == pytest.approx(
        27.5
    )
