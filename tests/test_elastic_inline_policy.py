# SPDX-License-Identifier: Apache-2.0

"""Equal-work selection, causal forecasting and actual closed-loop speedup."""

from dataclasses import asdict, replace

import pytest

from tests.test_elastic_inline_simulation import INITIAL, TARGET, simulation, tasks

from areal.elastic.policy.basic.inline import InlinePolicy, InlinePolicyConfig
from areal.elastic.policy.basic.inline_forecast import (
    InlineForecastConfig,
    InlineForecaster,
)
from areal.elastic.policy.basic.inline_trigger import (
    InlineSearchTrigger,
    InlineWorkloadFeatures,
)
from areal.elastic.policy.basic.observations import Turn
from areal.elastic.simulation.inline_simulation import InlineCandidateModel


def policy():
    return InlinePolicy(
        (INITIAL, TARGET),
        InlineForecaster(InlineForecastConfig(42, 16, 1)),
        InlinePolicyConfig(1, 8, 4, 0.02, 8, 0.01),
    )


@pytest.mark.parametrize("hidden_kind", ["output", "tool"])
def test_inline_forecast_and_decision_ignore_unknown_outputs_and_tools(hidden_kind):
    """Changing the true unsubmitted suffix leaves the same public decision unchanged."""
    forecasts, decisions = [], []
    for output in (4, 12):
        workload = tuple(
            replace(
                e,
                turns=(
                    (Turn(16, output),)
                    if hidden_kind == "output"
                    else (Turn(16, 4, output, 4, "python", 0), Turn(24, 2))
                ),
            )
            if i >= 4
            else e
            for i, e in enumerate(tasks(32))
        )
        sim = simulation(workload)
        sim.run(until=2.4)
        view = sim.observe()
        forecaster = InlineForecaster(InlineForecastConfig(42, 16, 1))
        forecasts.append(asdict(forecaster.predict(view, sim.task_pool.pending(), 1)))
        p = policy()
        model = InlineCandidateModel(sim.models, sim.config, sim.handoff, 8)
        decisions.append(
            asdict(p.decide(view, sim.task_pool.pending(), model, sim.total_updates))
        )
    assert forecasts[0] == forecasts[1]
    assert decisions[0] == decisions[1]


def test_candidate_ranking_counts_switch_cost_and_exact_equal_work():
    """A superficially faster layout loses once its finite-horizon handoff is charged."""
    sim = simulation(tasks(32))
    sim.run(until=2.4)
    decision = policy().decide(
        sim.observe(),
        sim.task_pool.pending(),
        InlineCandidateModel(sim.models, sim.config, sim.handoff, 8),
        sim.total_updates,
    )
    assert decision.target == INITIAL
    assert all(p.supported for p in decision.predictions)
    assert len({p.target_update for p in decision.predictions}) == 1
    assert all(p.published_update == p.target_update for p in decision.predictions)
    assert decision.predictions[1].seconds > decision.predictions[0].seconds


def test_equal_work_forecast_preserves_pipeline_contention_beyond_the_target_batch():
    """With perfectly repeated shapes, the current candidate matches the live prefix."""
    sim = simulation(tasks(64))
    sim.config = replace(sim.config, staleness=2, client_concurrency=16)
    sim.run(until=1.15)
    view = sim.observe()
    p = policy()
    p.config = replace(p.config, lookahead_batches=2)
    decision = p.decide(
        view,
        sim.task_pool.pending(),
        InlineCandidateModel(sim.models, sim.config, sim.handoff, 8),
        sim.total_updates,
    )
    current = decision.predictions[0]
    assert current.supported
    sim.training_limit = decision.target_update
    expected = sim.run(until_update=decision.target_update)
    assert current.seconds == pytest.approx(expected["seconds"] - view.now)
    assert current.published_update == decision.target_update


def test_automatic_inline_switch_improves_time_with_measured_search_and_verification():
    """Search/verification elapse while work runs; an actual change beats the initial layout."""
    from tests.test_elastic_inline_simulation import Costs

    class FasterTarget(Costs):
        def service(self, layout, kv, slices):
            return 0.01 if layout == TARGET else 0.1

    calls = []

    def clock():
        calls.append(None)
        return len(calls) * 0.01

    fixed = simulation(tasks(64))
    fixed.models = FasterTarget()
    dynamic = simulation(
        tasks(64), policy=policy(), measured_search_time=True, search_clock=clock
    )
    dynamic.models = FasterTarget()
    baseline, adaptive = fixed.run(), dynamic.run()
    assert baseline["status"] == adaptive["status"] == "complete"
    assert adaptive["seconds"] < baseline["seconds"]
    assert adaptive["trained_episodes"] == baseline["trained_episodes"] == 64
    assert adaptive["search_seconds"] > 0 and adaptive["verification_seconds"] > 0
    assert any(d["stage"] == "verification" for d in dynamic.decisions)
    assert any(e["kind"] == "handoff_complete" for e in dynamic.events)
    assert all(r["maximum_version_lag"] == 0 for r in dynamic.training_records)


def test_task_count_generation_is_seeded_and_phases_follow_task_index():
    """Generation uses the existing episode sampler, independent of arrival scheduling."""
    import json
    from pathlib import Path

    from areal.elastic.workload import TaskCountSpec, generate_count

    raw = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "examples/autopilot/elastic/inline-workloads/shape-shift.json"
        ).read_text()
    )
    spec = TaskCountSpec.decode(raw)
    first, second = generate_count(spec), generate_count(spec)
    assert first == second
    boundary = raw["composition"][0]["until_task"]
    assert all(e["family"] == "short" for e in first[:boundary])
    assert all(e["family"] == "prefill" for e in first[boundary:])
    assert [e["task_id"] for e in first] == list(range(spec.count))
    assert all(e["available_at"] == 0 for e in first)


def test_workload_trigger_detects_variance_with_equal_mean_and_p90():
    """CV detects a new shape even when the usual mean/tail summaries agree."""
    first = tuple((Turn(n, 4),) for n in [50] * 48 + [100] * 24 + [200] * 24)
    second = tuple((Turn(n, 4),) for n in [10] * 48 + [180] * 24 + [200] * 24)
    before = InlineWorkloadFeatures.from_turns(first)
    after = InlineWorkloadFeatures.from_turns(second)
    assert before.initial_input.mean == after.initial_input.mean == 100
    assert before.initial_input.p90 == after.initial_input.p90 == 200
    assert after.initial_input.cv > before.initial_input.cv
    trigger = InlineSearchTrigger(change_ratio=0.3, change_checks=2, maximum_gap=16)
    assert trigger.check(2, before).reason == "initial"
    assert not trigger.check(4, after).search
    confirmed = trigger.check(6, after)
    assert confirmed.search and confirmed.reason == "workload_change"


def test_stable_workload_skips_full_searches_without_changing_training():
    """Light checks advance independently, so skipping does not poll every GPU event."""
    base = replace(
        policy().config,
        every_updates=2,
        minimum_completed_episodes=8,
        max_search_interval_updates=8,
        decision_delay_seconds=0,
        forecast_seed_by_update=True,
    )
    results, simulations = [], []
    for adaptive in (False, True):
        p = policy()
        p.config = replace(base, workload_trigger_enabled=adaptive)
        p.search_trigger = InlineSearchTrigger(
            change_ratio=0.3, change_checks=2, maximum_gap=8
        )
        sim = simulation(tasks(128), policy=p, measured_search_time=False)
        results.append(sim.run())
        simulations.append(sim)
    assert all(r["status"] == "complete" for r in results)
    assert results[0]["seconds"] == pytest.approx(results[1]["seconds"])
    assert simulations[0].training_records == simulations[1].training_records
    assert results[0]["full_searches"] == 15
    assert results[1]["full_searches"] == 4
    assert results[1]["search_checks"] == 15
    assert [r["update"] for r in simulations[1].trigger_checks if r["search"]] == [
        2,
        10,
        18,
        26,
    ]


def test_published_shape_change_triggers_search_before_refresh():
    """A causal change in completed episodes is detected between refresh searches."""
    p = policy()
    p.config = replace(
        p.config,
        every_updates=2,
        workload_trigger_enabled=True,
        max_search_interval_updates=16,
        decision_delay_seconds=0,
        forecast_seed_by_update=True,
    )
    workload = tuple(
        replace(e, turns=(Turn(16, 12),)) if i >= 32 else e
        for i, e in enumerate(tasks(128))
    )
    sim = simulation(workload, policy=p, measured_search_time=False)
    assert sim.run()["status"] == "complete"
    changes = [r for r in sim.trigger_checks if r["reason"] == "workload_change"]
    assert [r["update"] for r in changes] == [12]
    assert changes[0]["features"]["generated"]["mean"] == 12


def test_update_keyed_forecast_is_independent_of_search_and_recheck_count():
    """Suppressing redundant searches must not itself change the sampled workload."""
    sim = simulation(tasks(32))
    sim.run(until=2.4)
    p = policy()
    p.config = replace(p.config, forecast_seed_by_update=True)
    model = InlineCandidateModel(sim.models, sim.config, sim.handoff, 8)
    view = sim.observe()
    one = p.decide(view, sim.task_pool.pending(), model, sim.total_updates)
    two = p.decide(view, sim.task_pool.pending(), model, sim.total_updates)
    check = p.decide(
        view, sim.task_pool.pending(), model, sim.total_updates, targets=(one.target,)
    )
    assert one.forecast_seed == two.forecast_seed == check.forecast_seed
    assert one.predictions == two.predictions
