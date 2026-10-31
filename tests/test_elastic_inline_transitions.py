# SPDX-License-Identifier: Apache-2.0

"""Whole-run budget, task conservation and state boundaries during inline handoff."""

from dataclasses import replace

import pytest

from tests.test_elastic_inline_simulation import INITIAL, TARGET, simulation, tasks

from areal.elastic.policy.basic.observations import Turn


@pytest.mark.parametrize("target", [TARGET, replace(INITIAL, rollout_replicas=2)])
def test_hard_budget_handoff_drains_tools_releases_then_allocates_and_preserves_work(
    target,
):
    """Both two-sided migration and Rollout-only replacement preserve ordered training."""
    workload = tuple(
        replace(e, turns=(Turn(16, 2, 0.3, 4, "python", 2), Turn(24, 2)))
        for e in tasks()
    )
    sim = simulation(workload)
    sim.schedule_at(0.15, "decision_apply", target)
    result = sim.run()
    assert result["status"] == "complete", result
    start = next(e for e in sim.events if e["kind"] == "handoff_start")
    complete = next(e for e in sim.events if e["kind"] == "handoff_complete")
    save = next(e for e in sim.events if e["kind"] == "inline_save_start")
    assert complete["snapshot"] == save["snapshot"]
    assert complete["time"] > save["time"] > start["time"]
    assert not any(
        start["time"] < e["time"] < complete["time"]
        for e in sim.events
        if e["kind"] == "episode_admitted"
    )
    assert [e for r in sim.training_records for e in r["episodes"]] == [
        e.episode_id for e in workload
    ]
    allocations = [e for e in sim.events if "leases" in e]
    assert all(sum(len(g) for _, g in e["leases"].values()) <= 8 for e in allocations)
    assert result["peak_gpus"] == 8
    changed_roles = {e["role"] for e in sim.events if e["kind"] == "inline_load_start"}
    releases = [
        i
        for i, e in enumerate(sim.events)
        if e["kind"] == "deployment_release"
        and e["deployment"] in {0, 1}
        and e["role"] in changed_roles
    ]
    new_allocations = [
        i
        for i, e in enumerate(sim.events)
        if e["kind"] == "allocation" and "deployment" in e
    ]
    assert max(releases) < min(new_allocations)
    if changed_roles == {"rollout"}:
        assert sim.actor_id == 0
    assert result["completed_updates"] == result["published_version"] == 4


def test_inline_handoff_at_an_actor_update_preserves_the_single_writer():
    """The current update publishes before save; the next update uses restored state."""
    sim = simulation()
    sim.schedule_at(1.8, "decision_apply", TARGET)
    result = sim.run()
    assert result["status"] == "complete"
    save = next(e for e in sim.events if e["kind"] == "inline_save_start")
    assert save["snapshot"][:3] == (1, 1, 4)
    assert [r["preupdate_version"] for r in sim.training_records] == list(range(4))
