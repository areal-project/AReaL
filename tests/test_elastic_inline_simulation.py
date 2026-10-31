# SPDX-License-Identifier: Apache-2.0

"""Normal inline execution against an independent, closed-form timing oracle."""

import pytest

from areal.elastic.policy.basic.observations import (
    EpisodeSample,
    Forecast,
    OnlineLayout,
    Turn,
)
from areal.elastic.policy.reference.cost_model.rollout import RolloutConfig
from areal.elastic.simulation.inline_simulation import InlineSimulation
from areal.elastic.simulation.online_transitions import HandoffConfig
from areal.elastic.simulation.simulation_core import SimulationConfig

INITIAL = OnlineLayout(2, 2, 4, 1)
TARGET = OnlineLayout(1, 2, 6, 1)


class Costs:
    """Each token iteration takes 0.1s, each update 0.6s, each publish 0.1s."""

    source_ids = ("inline-independent-timing-oracle",)

    def rollout_config(self, layout):
        return RolloutConfig(
            layout.rollout_replicas, layout.rollout_tp, 16, 2048, 100000
        )

    def service(self, layout, kv, slices):
        return 0.1

    def actor(self, layout, batch):
        return (("logp", 0.2), ("train", 0.4))

    def publication(self, layout, recipients):
        return 0.1


def tasks(count=16):
    return tuple(
        EpisodeSample(f"task-{i}", "math", 0, (Turn(16, 4),)) for i in range(count)
    )


def simulation(workload=None, **kwargs):
    return InlineSimulation(
        workload or tasks(),
        INITIAL,
        Costs(),
        SimulationConfig(4, 0, 1, (("python", 2),), 16, 2048, 100000, 10000),
        HandoffConfig(
            (8,),
            0.2,
            0.2,
            0.1,
            0.1,
            1000,
            100,
            0.00001,
            0.1,
            0.1,
            "Independent test-only save/load/startup timings",
        ),
        **kwargs,
    )


def test_fixed_inline_matches_closed_form_and_replenishes_only_with_capacity():
    """Four serial rollouts plus one update/publication cost exactly 2.3s per batch."""
    sim = simulation()
    sim.run(until=0.15)
    assert sim.task_pool.cursor == len(sim.arrival_order) == 1
    result = sim.run()
    assert result["status"] == "complete"
    assert result["seconds"] == pytest.approx(4 * (4 * 4 * 0.1 + 0.6 + 0.1))
    assert result["trained_episodes"] == result["completed_episodes"] == 16
    assert result["completed_updates"] == result["published_version"] == 4
    assert not result["remaining_leases"]
    assert all(r["maximum_version_lag"] == 0 for r in sim.training_records)


def test_inline_snapshot_restores_inflight_work_and_the_task_supply_cursor():
    """An explicit oracle suffix reproduces physical restoration, with no resubmission."""
    main = simulation()
    main.run(until=0.15)
    view = main.observe()
    forecast = Forecast(1, tuple(main._truth.values()), view.now)
    branch = InlineSimulation.from_forecast(
        view, forecast, main.models, main.config, main.handoff, 8
    )
    spent = sum(main.pool.gpu_seconds.values())
    actual, expected = branch.run(), main.run()
    assert actual["status"] == expected["status"] == "complete"
    assert actual["seconds"] == pytest.approx(expected["seconds"])
    assert actual["total_gpu_seconds"] == pytest.approx(
        expected["total_gpu_seconds"] - spent
    )
    assert branch.arrival_order == main.arrival_order
    assert branch.task_pool.cursor == 16


def test_inline_and_online_execution_import_without_loading_autopilot():
    """Standalone execution must stay usable independently of the revision package."""
    import subprocess
    import sys
    from pathlib import Path

    script = """
import runpy, sys
runpy.run_path('tests/run_autopilot_cpu.py', run_name='bootstrap')
from areal.elastic.simulation.inline_simulation import InlineSimulation
from areal.elastic.simulation.online_candidates import OnlineForecastModel
from areal.elastic.simulation.online_parallel import evaluate_chunk
assert not any(n == 'areal.autopilot' or n.startswith('areal.autopilot.') for n in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )
