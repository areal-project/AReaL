# SPDX-License-Identifier: Apache-2.0

"""Normal online paths: causal forecasts, hot handoff and conserved work/cost."""

from dataclasses import asdict, replace

import pytest

from areal.autopilot.domains.elastic.online_control import OnlineControl, _Search
from areal.autopilot.domains.elastic.online_decode import _Run
from areal.autopilot.domains.elastic.online_simulation import (
    OnlineForecastModel,
    OnlineRunConfig,
    OnlineSimulation,
)
from areal.autopilot.domains.elastic.online_transitions import (
    ActorMemoryRejected,
    GPUAllocator,
    HandoffConfig,
)
from areal.elastic.policy.basic.forecast import (
    ActorWaitView,
    EpisodeSample,
    Forecast,
    ForecastConfig,
    HistoricalForecaster,
    OnlineLayout,
    Turn,
)
from areal.elastic.policy.basic.history import HistoricalReplay, HistoryReplayConfig
from areal.elastic.policy.basic.online import (
    ActorSupply,
    CandidatePrediction,
    OnlineDecision,
    OnlinePolicy,
    OnlinePolicyConfig,
    rolling_slo,
)
from areal.elastic.policy.basic.search import SearchBound, select_prediction
from areal.elastic.policy.basic.serving import ServingSLO
from areal.elastic.policy.reference.core import fingerprint
from areal.elastic.policy.reference.cost_model.rollout import RolloutConfig


class _KnownCosts:
    """Closed-form independent oracle, in seconds; not a GPU calibration."""

    source_ids = ("normal-path-closed-form-oracle",)

    def rollout_config(self, layout):
        return RolloutConfig(
            layout.rollout_replicas, layout.rollout_tp, 32, 512, 1000000
        )

    def service(self, layout, kv, slices):
        return (
            0.005 + sum(s.query_tokens for s in slices) / 100000 + kv / 10000000
        ) / layout.rollout_tp

    def actor(self, layout, batch):
        return (
            ("logp", 0.4 / layout.actor_gpus),
            ("advantage", 0.2),
            ("train", 0.8 / layout.actor_gpus),
        )

    def publication(self, layout, recipients):
        return 0.001 * sum(r.rollout_gpus for r in recipients)


class _MemoryLimitedCosts(_KnownCosts):
    """A declared capacity boundary, rather than an injected runtime fault."""

    rejected = 0

    def actor(self, layout, batch):
        if layout.actor_gpus == 1:
            self.rejected += 1
            raise ActorMemoryRejected("Declared batch footprint exceeds one GPU")
        return super().actor(layout, batch)


class _AffineCosts(_KnownCosts):
    def decode_affine(self, layout, kv, slices):
        return self.service(layout, kv, slices), len(
            slices
        ) / 10000000 / layout.rollout_tp


@pytest.mark.parametrize("increment", [0.0, 0.00001])
def test_decode_interval_root_matches_sequential_token_completion(increment):
    """Closed-form step selection preserves completion before external events."""
    finished = []
    now = 3.0
    for step in range(128):
        now += 0.005 + increment * step
        finished.append(now)
    run = _Run((), 3.005, 0.005, increment, 128)
    for boundary in (3.003, 3.041, 3.171, 3.52, 4.1):
        expected = sum(when < boundary - 1e-9 for when in finished[:-1])
        assert run.steps_before(boundary) == expected


INITIAL = OnlineLayout(4, 2, 8, 1)
SMALL = OnlineLayout(1, 2, 2, 1)
SLO = ServingSLO(5, 0.1, 0.95)


def config():
    return OnlineRunConfig(4, 2, 32, (("python", 8),), 4096, 65536, 1000000, 1000)


def handoff():
    return HandoffConfig(
        (8, 8, 8, 8),
        2,
        2,
        1,
        1,
        7000,
        1000,
        0.000001,
        0.1,
        0.1,
        "Explicit test-only startup/link/state oracle",
    )


def episodes(n=16, spacing=1.0, output=4):
    return tuple(
        EpisodeSample(f"episode-{i}", "math", i * spacing, (Turn(16, output),))
        for i in range(n)
    )


def simulation(workload, *, policy=None, event_sink=None):
    return OnlineSimulation(
        workload,
        INITIAL,
        _KnownCosts(),
        config(),
        handoff(),
        SLO,
        input_closes_at=max(e.available_at for e in workload),
        policy=policy,
        event_sink=event_sink,
    )


@pytest.mark.parametrize("replicas", [1, 2, 8])
@pytest.mark.parametrize("spacing", [0.0, 0.13])
def test_decode_intervals_preserve_requests_training_and_handoff(replicas, spacing):
    """Coalesced GPU steps retain arrivals, tools, publication and hot cutovers."""
    workload = tuple(
        EpisodeSample(
            f"episode-{i}",
            "math",
            i * spacing,
            (Turn(900, 180 + i, 0.17 + i / 1000, 12, "python", 2), Turn(1100, 90 + i)),
        )
        for i in range(16)
    )
    simulations = []
    for accelerated in (False, True):
        sim = OnlineSimulation(
            workload,
            OnlineLayout(4, 2, replicas, 1),
            _AffineCosts(),
            config(),
            handoff(),
            SLO,
            input_closes_at=workload[-1].available_at,
            coalesce_decode=accelerated,
        )
        sim.schedule_at(0.7, "decision_apply", OnlineLayout(1, 2, 4, 1))
        sim.run(until=0.88123)  # Pause inside a GPU iteration, then resume it.
        assert all(
            r.view.last_token_at is None or r.view.last_token_at <= sim.now
            for r in sim._live_requests.values()
        )
        for deployment in sim.deployments.values():
            for replica, group in deployment.replicas.items():
                expected = {
                    key: r
                    for key, r in sim._live_requests.items()
                    if r.view.deployment == deployment.identity
                    and r.view.replica == replica
                }
                assert group.rows == expected
                assert group.kv_tokens == sum(r.kv_tokens for r in expected.values())
                assert group.reserved_tokens == sum(
                    r.view.input_tokens + r.reservation - 1 for r in expected.values()
                )
        sim.run()
        simulations.append(sim)
    reference, fast = simulations
    assert reference.status == fast.status == "complete"
    assert fast.decode_steps_skipped > 100
    assert fast.processed_events == reference.processed_events
    assert fast.counters == reference.counters
    for key, value in reference.result().items():
        if isinstance(value, float):
            assert fast.result()[key] == pytest.approx(value, rel=1e-11, abs=1e-9)
        else:
            assert fast.result()[key] == value
    for key in reference.requests:
        expected = asdict(reference.requests[key].view)
        actual = asdict(fast.requests[key].view)
        for field, value in expected.items():
            if isinstance(value, float):
                assert actual[field] == pytest.approx(value, rel=1e-11, abs=1e-9)
            else:
                assert actual[field] == value
    assert len(fast.events) == len(reference.events)
    for actual, expected in zip(fast.events, reference.events, strict=True):
        assert actual["kind"] == expected["kind"]
        assert actual["time"] == pytest.approx(expected["time"], abs=1e-9)
    assert [r["episodes"] for r in fast.training_records] == [
        r["episodes"] for r in reference.training_records
    ]
    assert [r["behavior_versions"] for r in fast.training_records] == [
        r["behavior_versions"] for r in reference.training_records
    ]


def test_mutable_requests_keep_observations_and_candidate_forks_independent():
    """Later token updates in either branch cannot mutate a captured observation."""
    workload = episodes(8, spacing=0, output=300)
    sim = simulation(workload)
    sim.models = _AffineCosts()
    sim.coalesce_decode = True
    sim.run(until=0.2)
    view = sim.observe()
    identity = fingerprint(view)
    forecast = Forecast(0, workload, view.now)
    fork = OnlineSimulation.from_forecast(
        view,
        forecast,
        sim.models,
        sim.config,
        sim.handoff,
        sim.slo,
        coalesce_decode=True,
    )
    assert fork.run()["status"] == "complete"
    assert fingerprint(view) == identity
    assert sim.run()["status"] == "complete"
    assert fingerprint(view) == identity
    assert all(r.view is not fork.requests[key].view for key, r in sim.requests.items())
    assert [asdict(r.view) for r in sim.requests.values()] == pytest.approx(
        [asdict(r.view) for r in fork.requests.values()]
    )


@pytest.mark.parametrize("replicas", [2, 8])
def test_replica_intervals_preserve_visible_progress_at_repeated_pauses(replicas):
    """Independent GPU progress stays causal at observations between completions."""
    workload = tuple(
        EpisodeSample(
            f"episode-{i}",
            "math",
            i * 0.031,
            (Turn(32 + i, 220 + 7 * i, 0.13, 8, "python"), Turn(48, 100 + i)),
        )
        for i in range(16)
    )
    runs = tuple(
        OnlineSimulation(
            workload,
            OnlineLayout(4, 2, replicas, 1),
            _AffineCosts(),
            config(),
            handoff(),
            SLO,
            input_closes_at=workload[-1].available_at,
            coalesce_decode=fast,
        )
        for fast in (False, True)
    )
    for when in (0.113, 0.317, 0.661, 1.127, 1.731):
        for run in runs:
            run.run(until=when)
        expected, actual = (run.observe() for run in runs)
        assert expected.now == actual.now == when
        assert actual.consumed_episodes == expected.consumed_episodes
        assert actual.provider_version == expected.provider_version
        assert len(actual.requests) == len(expected.requests)
        for want, got in zip(expected.requests, actual.requests, strict=True):
            for name, value in asdict(want).items():
                observed = getattr(got, name)
                if isinstance(value, float):
                    assert observed == pytest.approx(value, rel=1e-11, abs=1e-9)
                else:
                    assert observed == value
            assert got.last_token_at is None or got.last_token_at <= when
        assert runs[1].pool.gpu_seconds == pytest.approx(runs[0].pool.gpu_seconds)
    for run in runs:
        assert run.run()["status"] == "complete"
    assert runs[0].processed_events == runs[1].processed_events
    assert runs[0].counters == runs[1].counters


def test_shared_forecast_templates_keep_candidate_state_and_observations_independent():
    """Reuse static inputs while rebuilding all mutable state and view bindings."""
    workload = episodes(8, spacing=0, output=300)
    main = simulation(workload)
    main.models = _AffineCosts()
    main.run(until=0.12)
    view = main.observe()
    forecasts = (Forecast(42, workload, 5),)
    identity = fingerprint(view, forecasts)
    evaluator = OnlineForecastModel(
        main.models, main.config, main.handoff, SLO, prune_irreversible_slo=False
    )
    for layout in (view.layout, SMALL):
        assert evaluator.evaluate(view, forecasts, layout).supported
    assert evaluator.input_preparations == 1
    prepared = evaluator._prepared[0]
    branches = tuple(
        OnlineSimulation.from_forecast(
            view,
            forecasts[0],
            main.models,
            main.config,
            main.handoff,
            SLO,
            coalesce_decode=True,
            _prepared=prepared,
        )
        for _ in range(2)
    )
    first, second = branches
    assert first._truth is second._truth
    assert first.heap is not second.heap
    assert all(
        first.requests[key].view is not second.requests[key].view
        for key in first.requests
    )
    untouched = fingerprint(second.observe())
    assert first.run()["status"] == "complete"
    assert fingerprint(second.observe()) == untouched
    assert fingerprint(view, forecasts) == identity
    main.run(until=0.3)
    later = main.observe()
    actual = evaluator.evaluate(later, forecasts, later.layout)
    expected = OnlineForecastModel(
        main.models, main.config, main.handoff, SLO, prune_irreversible_slo=False
    ).evaluate(later, forecasts, later.layout)
    assert actual == expected
    assert evaluator.input_preparations == 2


def test_deferred_decode_preserves_token_versions_and_publication_gaps():
    """Long responses cross another batch's update and weight publication."""
    workload = tuple(
        EpisodeSample(
            f"episode-{i}",
            "math",
            0 if i < 4 else 0.07,
            (Turn(16, 3 if i < 4 else 350 + i),),
        )
        for i in range(8)
    )
    runs = []
    for fast in (False, True):
        run = simulation(workload)
        run.models = _AffineCosts()
        run.coalesce_decode = fast
        assert run.run()["status"] == "complete"
        runs.append(run)
    reference, deferred = runs
    for key in reference.requests:
        expected, actual = reference.requests[key].view, deferred.requests[key].view
        assert actual.generated == expected.generated
        assert actual.version_counts == expected.version_counts
        assert actual.last_token_at == pytest.approx(expected.last_token_at, abs=1e-9)
        assert actual.maximum_tbt_seconds == pytest.approx(
            expected.maximum_tbt_seconds, abs=1e-9
        )
    assert len(deferred.requests["episode-4/request-0"].view.version_counts) >= 2
    assert deferred.counters == reference.counters
    assert deferred.processed_events == reference.processed_events
    for actual, expected in zip(
        deferred.training_records, reference.training_records, strict=True
    ):
        for key, value in expected.items():
            if isinstance(value, float):
                assert actual[key] == pytest.approx(value, abs=1e-9)
            else:
                assert actual[key] == value


def test_decode_intervals_preserve_detailed_gpu_trace():
    """An attached per-batch trace keeps the reference token execution path."""
    events = []
    sim = simulation(episodes(output=100), event_sink=events.append)
    sim.models = _AffineCosts()
    sim.coalesce_decode = True
    assert sim.run()["status"] == "complete"
    assert sim.decode_steps_skipped == 0
    assert (
        sum(e["kind"] == "service_batch" for e in events)
        == sim.counters["service_batch"]
    )


def test_hot_rollout_finishes_old_requests_and_counts_overlap():
    """New routing does not migrate old decode/KV or release its GPUs early."""
    workload = episodes(16, spacing=0.5)
    workload = (replace(workload[0], turns=(Turn(16, 4000),)),) + workload[1:]
    events = []
    sim = simulation(workload, event_sink=events.append)
    sim.schedule_at(0.1, "decision_apply", OnlineLayout(4, 2, 2, 1))
    result = sim.run()
    assert result["status"] == "complete", result
    assert result["completed_episodes"] == 16
    assert result["completed_updates"] == 4
    assert result["trained_episodes"] == 16
    cut = next(e for e in events if e["kind"] == "rollout_cutover")
    old = cut["source"]
    release = next(
        e
        for e in events
        if e["kind"] == "deployment_release" and e["deployment"] == old
    )
    long_request = sim.requests["episode-0/request-0"].view
    assert long_request.deployment == old
    assert long_request.generated == 4000
    assert long_request.completed_at > cut["time"]
    assert release["time"] >= long_request.completed_at
    assert result["peak_gpus"] == 18
    assert not result["remaining_leases"]
    assert (
        len(
            [
                e
                for e in events
                if e["kind"] == "request_assigned"
                and e["request"] == long_request.request_id
            ]
        )
        == 1
    )
    total = previous = 0.0
    leases = {}
    for event in events:
        if "leases" in event:
            total += (event["time"] - previous) * sum(
                len(devices) for _, devices in leases.values()
            )
            previous, leases = event["time"], event["leases"]
    assert total == pytest.approx(result["total_gpu_seconds"], abs=1e-9)


def test_actor_handoff_keeps_one_writer_and_the_exact_training_cursor():
    """Preparation overlaps updates; final handoff preserves every batch once."""
    sim = simulation(episodes(32, spacing=0.5))
    sim.schedule_at(1, "decision_apply", OnlineLayout(1, 2, 8, 1))
    result = sim.run()
    assert result["status"] == "complete", result
    assert result["completed_updates"] == 8
    assert result["published_version"] == 8
    cut = next(e for e in sim.events if e["kind"] == "actor_cutover")
    prepared = next(e for e in sim.events if e["kind"] == "deployment_prepared")
    old_updates = [
        e
        for e in sim.events
        if e["kind"] == "actor_complete" and e["time"] < prepared["time"]
    ]
    assert old_updates
    start = next(e for e in sim.events if e["kind"] == "actor_handoff_start")
    assert cut["cursor"] == start["cursor"]
    assert cut["version"] == start["version"]
    records = sim.training_records
    assert [identity for record in records for identity in record["episodes"]] == list(
        sim.arrival_order
    )
    assert [record["preupdate_version"] for record in records] == list(range(8))
    assert all(
        not (start["time"] < e["time"] < cut["time"])
        for e in sim.events
        if e["kind"] == "actor_start"
    )


def test_actor_wait_keeps_serving_then_trains_each_batch_once():
    """An unsafe old Actor waits while GPU/tool work and replacement proceed."""
    workload = tuple(
        EpisodeSample(
            f"episode-{i}",
            "math",
            i * 0.1,
            (Turn(16, 4, 0.3, 8, "python", 2), Turn(30, 4))
            if i % 2 == 0
            else (Turn(16, 4),),
        )
        for i in range(15)
    )
    costs, events = _MemoryLimitedCosts(), []
    sim = OnlineSimulation(
        workload,
        OnlineLayout(1, 1, 8, 1),
        costs,
        config(),
        handoff(),
        SLO,
        input_closes_at=workload[-1].available_at,
        event_sink=events.append,
    )
    sim.schedule_at(0.1, "decision_apply", INITIAL)
    result = sim.run()
    assert result["status"] == "complete", result
    assert result["completed_updates"] == result["published_version"] == 3
    assert result["trained_episodes"] == 12 and result["retained_episodes"] == 3
    wait = next(e for e in events if e["kind"] == "actor_wait")
    cut = next(e for e in events if e["kind"] == "actor_cutover")
    resumed = next(e for e in events if e["kind"] == "actor_wait_complete")
    assert resumed["episodes"] == wait["episodes"]
    assert wait["cursor"] == cut["cursor"] == resumed["cursor"] == 0
    assert wait["version"] == cut["version"] == resumed["version"] == 0
    assert resumed["time"] >= cut["time"] > wait["time"]
    assert result["actor_wait_seconds"] == pytest.approx(resumed["time"] - wait["time"])
    assert costs.rejected == result["counters"]["actor_wait"] == 1
    assert any(
        e["kind"] == "service_batch" and wait["time"] < e["time"] < cut["time"]
        for e in events
    )
    assert any(
        e["kind"] == "tool_complete" and wait["time"] < e["time"] < cut["time"]
        for e in events
    )
    starts = [e for e in events if e["kind"] == "actor_start"]
    assert all(e["deployment"] == cut["target"] for e in starts)
    assert [e for row in starts for e in row["episodes"]] == sim.arrival_order[:12]
    assert not result["remaining_leases"] and result["actor_wait"] is None


def test_actor_wait_restores_and_allows_a_delayed_repair_decision():
    """Waiting stays observable/searchable and survives a delayed Actor handoff."""
    workload = episodes(12, spacing=0.1)
    sim = OnlineSimulation(
        workload,
        OnlineLayout(1, 1, 8, 1),
        _MemoryLimitedCosts(),
        config(),
        handoff(),
        SLO,
        input_closes_at=workload[-1].available_at,
    )
    sim.run(until=1.0)
    view = sim.observe()
    assert view.actor.waiting is not None and not view.actor.batch and not view.busy
    assert view.consumed_episodes == view.training_version == view.provider_version == 0
    assert not sim.training_records
    forecasts = (Forecast(42, workload, 2),)
    evaluator = OnlineForecastModel(sim.models, sim.config, sim.handoff, SLO, 1)
    keep = evaluator.evaluate(view, forecasts, view.layout)
    repair = evaluator.evaluate(view, forecasts, INITIAL)
    assert not keep.supported and "no planned Actor migration" in keep.reason
    assert repair.supported and repair.worst_request_fraction == 1
    branch = OnlineSimulation.from_forecast(
        view, forecasts[0], sim.models, sim.config, sim.handoff, SLO
    )
    assert branch.actor_wait == view.actor.waiting
    for run in (sim, branch):
        run.schedule_at(2.0, "decision_apply", INITIAL)
    expected, actual = sim.run(), branch.run()
    assert expected["status"] == actual["status"] == "complete"
    assert expected["completed_updates"] == actual["completed_updates"] == 3
    assert actual["seconds"] == pytest.approx(expected["seconds"], abs=1e-9)
    assert actual["actor_wait"] is None


def test_forecast_restore_continues_current_gpu_tool_and_actor_work():
    """Restoring a visible physical state reproduces the known-cost continuation."""
    workload = tuple(
        EpisodeSample(
            f"episode-{i}",
            "math",
            i * 0.05,
            (Turn(16, 4, 1, 8, "python", 2), Turn(30, 20)),
        )
        for i in range(12)
    )
    main = simulation(workload)
    main.run(until=1.35)
    view = main.observe()
    assert not view.busy
    assert view.services or any(e.tool for e in view.episodes) or view.actor.batch
    # This explicit oracle fixture checks physical restoration. The policy's real
    # forecast is separately checked below and never receives this true suffix.
    forecast = Forecast(1, workload, 2)
    branch = OnlineSimulation.from_forecast(
        view, forecast, main.models, main.config, main.handoff, SLO
    )
    spent = sum(main.pool.gpu_seconds.values())
    expected, actual = main.run(), branch.run()
    assert actual["status"] == expected["status"] == "complete"
    assert actual["seconds"] == pytest.approx(expected["seconds"], abs=1e-9)
    assert actual["total_gpu_seconds"] == pytest.approx(
        expected["total_gpu_seconds"] - spent, abs=1e-8
    )
    assert actual["completed_updates"] == expected["completed_updates"]
    assert actual["completed_requests"] == expected["completed_requests"]


def policy():
    return OnlinePolicy(
        (INITIAL, SMALL),
        HistoricalForecaster(ForecastConfig(20, 60, 1, 1, 42, 4096, 1)),
        OnlinePolicyConfig(0.2, 20, 1, 0.01, 1, 2),
        SLO,
    )


def test_online_policy_reconfigures_from_the_fixed_initial_deployment():
    """A real empirical forecast/search selects a cheaper serving-feasible layout."""
    workload = episodes(12, spacing=3)
    fixed, dynamic = simulation(workload), simulation(workload, policy=policy())
    baseline, adaptive = fixed.run(), dynamic.run()
    assert baseline["status"] == adaptive["status"] == "complete"
    assert adaptive["completed_updates"] == baseline["completed_updates"] == 3
    assert adaptive["joint_request_fraction"] == 1
    assert any(e["kind"] == "handoff_complete" for e in dynamic.events)
    assert adaptive["total_gpu_seconds"] < baseline["total_gpu_seconds"]
    searches = [d for d in dynamic.decisions if d["predictions"]]
    assert searches
    assert searches[0]["trigger"] == "slack"
    assert len({tuple(p["work_ids"]) for p in searches[0]["predictions"]}) == 1


def test_online_decisions_do_not_change_when_hidden_future_changes():
    """Unknown active output, tool duration and later arrivals cannot leak in."""
    seed = EpisodeSample("history", "math", 0, (Turn(16, 1),))
    views = []
    decisions = []
    for hidden_output, hidden_tool, future_at in ((200, 7, 4), (400, 70, 40)):
        workload = (
            seed,
            EpisodeSample(
                "active",
                "math",
                0.1,
                (Turn(16, hidden_output, hidden_tool, 8, "python"), Turn(1000, 1)),
            ),
            EpisodeSample("future", "math", future_at, (Turn(16, 1),)),
        )
        sim = simulation(workload, policy=policy())
        sim.run(until=0.8)
        views.append(sim.observe())
        decisions.append(
            [
                (
                    d["observed_at"],
                    d["observation_id"],
                    d["target"],
                    d["forecast_seeds"],
                    d["predictions"],
                )
                for d in sim.decisions
            ]
        )
    assert views[0] == views[1]
    assert decisions[0] == decisions[1]


def test_waiting_before_admission_remains_visible_to_rolling_slo():
    """Expired pending requests remain failures even outside the history window."""
    sim = simulation(episodes(20, spacing=0, output=100))
    sim.run(until=1)
    view = sim.observe()
    waiting = replace(
        view.requests[-1],
        deployment=None,
        replica=None,
        first_token_at=None,
        last_token_at=None,
        generated=0,
        completed_at=None,
        ready_at=0,
    )
    view = replace(view, now=10, requests=(waiting,))
    stats = rolling_slo(view, SLO, seconds=1)
    assert stats.completed == 0
    assert stats.pending == stats.decided == 1
    assert stats.fraction == 0


def test_irreversible_slo_bound_matches_complete_sampled_prediction():
    """Observed admission delays can rule out a sampled window before search."""
    workload = episodes(4, spacing=0, output=1)
    workload = (replace(workload[0], turns=(Turn(16, 2000),)),) + workload[1:]
    sim = OnlineSimulation(
        workload,
        INITIAL,
        _KnownCosts(),
        replace(config(), client_concurrency=1),
        handoff(),
        SLO,
        input_closes_at=0,
    )
    sim.run(until=6)
    view = sim.observe()
    forecasts = (Forecast(42, workload, 6),)
    pruned = OnlineForecastModel(sim.models, sim.config, sim.handoff, SLO).evaluate(
        view, forecasts, INITIAL
    )
    full = OnlineForecastModel(
        sim.models, sim.config, sim.handoff, SLO, prune_irreversible_slo=False
    ).evaluate(view, forecasts, INITIAL)
    assert not pruned.supported and pruned.mean_gpu_seconds is None
    assert "joint_fraction_upper_bound=0.25" in pruned.reason
    assert full.supported and full.worst_request_fraction == 0.25
    assert full.work_ids == pruned.work_ids


def test_sampled_slo_branch_stops_after_infeasibility_is_known():
    """A predicted overload may be rejected before all sampled tokens finish."""

    class CountedCosts(_KnownCosts):
        calls = 0

        def service(self, layout, kv, slices):
            self.calls += 1
            return super().service(layout, kv, slices)

    workload = episodes(4, spacing=0, output=1)
    workload = (replace(workload[0], turns=(Turn(16, 2000),)),) + workload[1:]
    sim = OnlineSimulation(
        workload,
        INITIAL,
        _KnownCosts(),
        replace(config(), client_concurrency=1),
        handoff(),
        SLO,
        input_closes_at=0,
    )
    sim.run(until=0)
    view, forecasts = sim.observe(), (Forecast(42, workload, 0),)
    pruned_costs, full_costs = CountedCosts(), CountedCosts()
    pruned = OnlineForecastModel(pruned_costs, sim.config, sim.handoff, SLO).evaluate(
        view, forecasts, INITIAL
    )
    full = OnlineForecastModel(
        full_costs, sim.config, sim.handoff, SLO, prune_irreversible_slo=False
    ).evaluate(view, forecasts, INITIAL)
    assert not pruned.supported and "Sampled branch" in pruned.reason
    assert full.supported and full.worst_request_fraction == 0.25
    assert pruned_costs.calls < full_costs.calls


def test_rollout_final_sync_has_a_version_boundary_under_continuous_updates():
    """A prepared server catches up before the ongoing training stream ends."""
    workload = episodes(192, spacing=0.05)
    sim = simulation(workload)
    sim.schedule_at(0.1, "decision_apply", OnlineLayout(4, 2, 2, 1))
    result = sim.run()
    assert result["status"] == "complete", result
    cut = next(e for e in sim.events if e["kind"] == "rollout_cutover")
    assert cut["time"] < workload[-1].available_at
    assert 1 <= result["counters"]["rollout_catchup_start"] <= 2
    assert result["completed_updates"] == 48


def test_cross_node_rollout_catchup_transfers_one_model_image_per_node():
    """Updates during server preparation require two destination images, too."""
    work = episodes(192, spacing=0.05)
    sim = OnlineSimulation(
        work,
        OnlineLayout(2, 2, 4, 1),
        _KnownCosts(),
        config(),
        handoff(),
        SLO,
        input_closes_at=work[-1].available_at,
    )
    sim.schedule_at(0.1, "decision_apply", OnlineLayout(2, 2, 12, 1))
    result = sim.run()
    assert result["status"] == "complete" and result["completed_updates"] == 48
    catchups = [e for e in sim.events if e["kind"] == "rollout_catchup_start"]
    assert catchups and all(e["nodes"] == 2 for e in catchups)
    assert all(e["bytes"] == 2 * sim.handoff.weight_bytes for e in catchups)
    assert all(
        e["ends_at"] - e["transfer_start"]
        == pytest.approx(
            2 * sim.handoff.weight_bytes / (sim.handoff.effective_bandwidth_GBps * 1e9)
            + sim.handoff.rollout_sync_setup_seconds
        )
        for e in catchups
    )


def test_forecast_keeps_actual_node_placement_after_two_hot_changes():
    """A fragmented pool does not become two empty nodes in a prediction."""
    workload = episodes(40, spacing=2.5)
    sim = OnlineSimulation(
        workload,
        INITIAL,
        _KnownCosts(),
        config(),
        replace(handoff(), node_gpus=(8, 8, 8)),
        SLO,
        input_closes_at=workload[-1].available_at,
    )
    sim.schedule_at(0.1, "decision_apply", OnlineLayout(1, 2, 8, 1))
    sim.schedule_at(20, "decision_apply", OnlineLayout(1, 2, 6, 1))
    sim.run(until=30)
    view = sim.observe()
    assert not view.busy
    assert view.layout == OnlineLayout(1, 2, 6, 1)
    assert len({devices[0][0] for _, _, devices in view.leases}) == 2
    forecasts = HistoricalForecaster(
        ForecastConfig(120, 60, 1, 1, 42, 4096, 1)
    ).predict(view, 0)
    prediction = OnlineForecastModel(sim.models, sim.config, sim.handoff, SLO).evaluate(
        view, forecasts, INITIAL
    )
    assert not prediction.supported
    assert "spare" in prediction.reason
    assert not sim.request_handoff(INITIAL)
    assert sim.run()["status"] == "complete"


def test_tool_forecast_conditions_on_elapsed_time_with_positive_residual():
    """An ongoing tool beyond historical support is not assumed to finish now."""
    history = EpisodeSample(
        "history", "math", 0, (Turn(16, 1, 1, 8, "python", 2), Turn(27, 1))
    )
    live = EpisodeSample(
        "live", "math", 0.1, (Turn(16, 1, 100, 8, "python", 2), Turn(27, 1))
    )
    sim = simulation((history, live))
    sim.run(until=20)
    view = sim.observe()
    tool = next(e.tool for e in view.episodes if e.episode_id == "live")
    elapsed = view.now - tool.started_at
    forecast = HistoricalForecaster(ForecastConfig(120, 60, 1, 1, 42, 4096, 5)).predict(
        view, 0
    )[0]
    sampled = next(e for e in forecast.episodes if e.episode_id == "live")
    assert sampled.turns[0].tool_seconds == pytest.approx(elapsed + 1)
    assert sampled.turns[0].tool_seconds != live.turns[0].tool_seconds


def history_policy(candidates=(INITIAL, SMALL), *, history_seconds=20):
    return OnlinePolicy(
        candidates,
        HistoricalReplay(HistoryReplayConfig(history_seconds, 2, 0.5, 2, 60, 4096, 1)),
        OnlinePolicyConfig(0.2, 20, 1, 0.01, 1, len(candidates)),
        ServingSLO(5, 0.1, 0.8),
    )


def test_history_replay_preserves_real_arrival_slots_and_turn_shapes():
    """Observed history repeats without random arrivals or deployment timings."""
    sim = simulation(episodes(8, spacing=1))
    sim.run(until=5)
    view = replace(sim.observe(), input_closed=False)
    history = HistoricalReplay(HistoryReplayConfig(20, 2, 0.5, 2, 15, 4096, 1))
    first = history.predict(view, 0)[0]
    repeated = [e for e in first.episodes if e.episode_id.startswith("history-replay")]
    assert first.seed == 0 and first.replay.expected_remaining_seconds == 20
    assert [e.available_at for e in repeated[:5]] == [5, 6, 7, 8, 9]
    assert [e.available_at for e in repeated[5:]] == list(range(10, 25))
    assert all(e.turns == (Turn(16, 4),) for e in repeated)
    assert first == history.predict(view, 0)[0]


def test_regime_duration_updates_while_search_is_busy():
    """Previous lifetime comes from observed load changes, including busy polls."""
    sim = simulation(episodes(41, spacing=1))
    sim.run(until=31)
    base = sim.observe()
    rows = tuple(
        replace(e, family="terminal" if e.available_at >= 11 else "math")
        for e in base.episodes
    )
    policy = history_policy()
    policy.forecaster = HistoricalReplay(
        HistoryReplayConfig(30, 10, 0.5, 2, 100, 4096, 1)
    )
    for now in (10, 20, 30):
        view = replace(
            base,
            now=now,
            busy=True,
            episodes=tuple(e for e in rows if e.available_at < now),
        )
        policy.decide(view, None)
    replay = policy.forecaster.predict(
        replace(view, busy=False, input_closed=False), 0
    )[0]
    assert replay.replay.previous_regime_seconds == 20
    assert replay.replay.regime_started_at == 20
    assert replay.replay.expected_remaining_seconds == 30
    assert replay.replay.duration_source == "fixed_history_window"


class _ScoredReplay:
    """Independent candidate scores isolate the controller's ranking contract."""

    def __init__(self, scores):
        self.scores, self.calls = scores, []

    def evaluate(self, view, forecasts, layout):
        self.calls.append(layout)
        fraction, window, total = self.scores[layout]
        return CandidatePrediction(
            layout,
            True,
            fraction,
            total,
            ("same-history",),
            window_gpu_seconds=window,
            tail_gpu_seconds=total - window,
            actor_coverage="known_complete_batch_capacity_checked",
        )


def _service_trigger_snapshot(passed, *, now=120):
    sim = simulation(episodes(int(now) + 40))
    sim.run(until=now)
    view = sim.observe()
    requests = tuple(r for r in view.requests if r.completed_at is not None)[:20]
    assert len(requests) == 20
    return replace(
        view,
        requests=tuple(
            replace(
                request,
                ready_at=view.now - (4.5 if index < passed else 6) - 1,
                first_token_at=view.now - 1,
                last_token_at=view.now - 0.5,
                completed_at=view.now - 0.5,
                maximum_tbt_seconds=0.09,
            )
            for index, request in enumerate(requests)
        ),
    )


@pytest.mark.parametrize(
    "passed,trigger",
    [
        (13, "pressure"),
        (14, "pressure"),
        (15, "pressure"),
        (16, "none"),
        (18, "none"),
        (19, "slack"),
    ],
)
def test_service_conditions_search_on_the_first_check(passed, trigger):
    """Only fractions outside [80%, 90%] search after a full history window."""
    view = _service_trigger_snapshot(passed)
    controller = history_policy(history_seconds=120)
    controller.config = replace(
        controller.config,
        slo_window_seconds=120,
        pressure_checks=2,
        slack_checks=3,
        minimum_completed_requests=8,
    )
    scores = _ScoredReplay({INITIAL: (0.9, 100, 200), SMALL: (0.9, 90, 150)})
    decision = controller.decide(view, scores)
    assert decision.statistics.fraction == passed / 20
    assert decision.trigger == trigger
    assert decision.target == (INITIAL if trigger == "none" else SMALL)
    assert scores.calls == ([] if trigger == "none" else [INITIAL, SMALL])


@pytest.mark.parametrize("passed", [15, 19])
def test_service_judgment_waits_for_the_full_history_window(passed):
    """Neither pressure nor slack can search on a short startup history."""
    controller = history_policy(history_seconds=120)
    controller.config = replace(controller.config, deduplicate_searches=False)
    scores = _ScoredReplay({INITIAL: (0.9, 100, 200), SMALL: (0.9, 90, 150)})
    for now in (110, 119, 120):
        decision = controller.decide(_service_trigger_snapshot(passed, now=now), scores)
        if now < 120:
            assert decision.trigger == "none" and not decision.predictions
            assert decision.target == INITIAL and not scores.calls
            assert controller.decision_index == 0
            assert "full history window" in decision.reason
        else:
            assert decision.trigger == ("pressure" if passed == 15 else "slack")
            assert scores.calls == [INITIAL, SMALL]
            assert decision.target == SMALL


@pytest.mark.parametrize("measured_search_time", [False, True])
def test_simulation_first_search_follows_full_120_second_history(measured_search_time):
    """Both timeline control paths keep serving before the first eligible poll."""
    controller = history_policy(candidates=(INITIAL,), history_seconds=120)
    controller.config = replace(controller.config, check_interval_seconds=10)
    sim = OnlineSimulation(
        episodes(160),
        INITIAL,
        _KnownCosts(),
        config(),
        handoff(),
        SLO,
        input_closes_at=159,
        policy=controller,
        measured_search_time=measured_search_time,
        search_clock=_DecisionClock(0),
    )
    sim.run(until=125)
    searches = [d for d in sim.decisions if d["predictions"]]
    assert searches and searches[0]["observed_at"] == 120
    assert all(not d["predictions"] for d in sim.decisions if d["observed_at"] < 120)
    assert any(e["kind"] == "request_complete" and e["time"] < 120 for e in sim.events)


@pytest.mark.parametrize("passed", [13, 19])
def test_example_policy_waits_after_handoff_then_searches_each_matching_poll(passed):
    """Every completed switch needs fresh history, then ordinary polling resumes."""
    import json
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "examples/autopilot/elastic/online-example.json"
    )
    controller = history_policy(history_seconds=120)
    controller.config = OnlinePolicyConfig(**json.loads(path.read_text())["policy"])
    scores = _ScoredReplay({INITIAL: (0.9, 100, 200), SMALL: (0.9, 110, 220)})
    for completed_at in (120, 300):
        controller.switched(completed_at)
        before = len(scores.calls)
        for elapsed in (0, 10, 119, 120, 130):
            view = _service_trigger_snapshot(passed, now=completed_at + elapsed)
            decision = controller.decide(view, scores)
            if elapsed < 120:
                assert decision.trigger == "none" and not decision.predictions
                assert len(scores.calls) == before
                assert "completed handoff" in decision.reason
            else:
                assert decision.predictions and decision.target == (
                    SMALL if passed == 13 else INITIAL
                )
        assert scores.calls[before:] == [INITIAL, SMALL, INITIAL, SMALL]


@pytest.mark.parametrize("measured_search_time", [False, True])
def test_post_handoff_history_starts_after_old_requests_drain(measured_search_time):
    """Readiness/cutover cannot start the timer while old GPU work is still alive."""
    workload = episodes(200)
    workload = (replace(workload[0], turns=(Turn(16, 4000),)),) + workload[1:]
    target = OnlineLayout(4, 2, 2, 1)
    controller = history_policy(candidates=(target,), history_seconds=120)
    controller.config = replace(controller.config, check_interval_seconds=10)
    sim = OnlineSimulation(
        workload,
        INITIAL,
        _KnownCosts(),
        config(),
        handoff(),
        SLO,
        input_closes_at=199,
        policy=controller,
        measured_search_time=measured_search_time,
        search_clock=_DecisionClock(0),
    )
    sim.schedule_at(0.1, "decision_apply", target)
    sim.run(until=180)
    cutover = next(e["time"] for e in sim.events if e["kind"] == "rollout_cutover")
    completed = next(e["time"] for e in sim.events if e["kind"] == "handoff_complete")
    assert cutover < completed == controller.last_switch
    assert completed >= sim.requests["episode-0/request-0"].view.completed_at
    first_eligible = next(
        d for d in sim.decisions if d["observed_at"] >= completed + 120
    )
    assert first_eligible["predictions"]
    assert all(
        not d["predictions"]
        for d in sim.decisions
        if d["observed_at"] < completed + 120
    )
    assert any(
        e["kind"] == "request_complete" and completed < e["time"] < completed + 120
        for e in sim.events
    )


@pytest.mark.parametrize("actor_waiting", [False, True])
def test_initialization_and_load_changes_do_not_search_without_service_trigger(
    actor_waiting,
):
    """The middle SLO band only updates history, including an observed Actor wait."""
    view = _service_trigger_snapshot(16)
    if actor_waiting:
        view = replace(
            view,
            actor=replace(
                view.actor,
                batch=(),
                waiting=ActorWaitView(
                    tuple(e.episode_id for e in view.episodes[:4]),
                    view.actor_deployment,
                    view.now,
                    "Observed Actor batch waiting for replacement",
                ),
            ),
        )
    controller = history_policy()
    controller.config = replace(controller.config, slo_window_seconds=120)
    scores = _ScoredReplay({INITIAL: (0.9, 100, 200), SMALL: (0.9, 90, 150)})
    for now in (120, 130, 140):
        decision = controller.decide(replace(view, now=now), scores)
        assert decision.trigger == "none" and not decision.predictions
    assert controller.forecaster.anchor is not None
    assert controller.forecaster.regime_index > 0
    assert not scores.calls


@pytest.mark.parametrize(
    "trigger,current_fraction,new_fraction,new_window,new_total,should_switch",
    [
        ("slack", 0.9, 0.9, 110, 150, True),  # old window/payback gate is gone
        ("slack", 0.9, 0.9, 90, 150, True),
        ("slack", 0.9, 0.9, 90, 200, False),
        ("slack", 0.85, 0.75, 90, 150, False),
        (
            "pressure",
            0.9,
            0.85,
            120,
            220,
            True,
        ),  # feasible replacement excludes current
        ("pressure", 0.4, 0.85, 120, 220, True),
        ("pressure", 0.4, 0.7, 120, 220, True),
        ("pressure", 0.7, 0.4, 90, 150, False),
        ("slack", 0.4, 0.7, 90, 150, False),  # slack cannot pick a below-target winner
    ],
)
def test_history_selection_uses_window_score_and_trigger_filter(
    trigger, current_fraction, new_fraction, new_window, new_total, should_switch
):
    """Pressure and slack share the new cost objective, with different SLO filters."""
    view = _service_trigger_snapshot(13 if trigger == "pressure" else 19)
    policy = history_policy()
    scores = _ScoredReplay(
        {
            INITIAL: (current_fraction, 100, 200),
            SMALL: (new_fraction, new_window, new_total),
        }
    )
    decision = policy.decide(view, scores)
    assert scores.calls == [INITIAL, SMALL]
    assert decision.trigger == trigger
    assert decision.target == (SMALL if should_switch else INITIAL)
    assert not decision.forecast_seeds and decision.replay_window is not None
    if should_switch:
        assert decision.net_total_gpu_seconds == 200 - new_total
    if trigger == "pressure" and max(current_fraction, new_fraction) < 0.8:
        assert "target remains unmet" in decision.reason


def test_window_cost_ranks_gpu_count_and_switching_without_a_card_count_shortcut():
    """A two-GPU saving can lose to more than 240 GPU-seconds of extra switching."""
    few = OnlineLayout(1, 2, 2, 1)  # four GPUs
    more = OnlineLayout(1, 2, 4, 1)  # six GPUs
    same_size = OnlineLayout(1, 2, 2, 2)  # six GPUs, different TP

    def score(layout, switching):
        cost = switching + 120 * (layout.actor_gpus + layout.rollout_gpus)
        return CandidatePrediction(
            layout,
            True,
            0.9,
            cost,
            ("same-history",),
            switching_gpu_seconds=switching,
            running_gpu_seconds_per_second=layout.actor_gpus + layout.rollout_gpus,
            evaluation_seconds=120,
            running_gpu_seconds=cost - switching,
            score_gpu_seconds=cost,
        )

    assert (
        select_prediction((score(few, 300), score(more, 20)), INITIAL, 0.8).layout
        == more
    )
    assert (
        select_prediction((score(few, 200), score(more, 20)), INITIAL, 0.8).layout
        == few
    )
    assert (
        select_prediction((score(more, 30), score(same_size, 20)), INITIAL, 0.8).layout
        == same_size
    )


class _CreditCosts(_AffineCosts):
    """A 40-episode cycle costs 149 seconds on a two-GPU Actor."""

    def actor(self, layout, batch):
        return (("train", 149 * 2 / layout.actor_gpus),)


def credit_simulation(initial=INITIAL, *, accelerated=False):
    return OnlineSimulation(
        episodes(300, spacing=1.5),
        initial,
        _CreditCosts(),
        replace(config(), batch_size=40),
        replace(
            handoff(),
            actor_startup_seconds=30,
            rollout_startup_seconds=30,
            actor_warmup_seconds=5,
            rollout_warmup_seconds=5,
        ),
        ServingSLO(5, 0.1, 0.8),
        input_closes_at=448.5,
        coalesce_decode=accelerated,
    )


def test_actor_supply_blocks_buffered_short_window_scale_down():
    """Old work/credit can pass window SLO without validating the new Actor rate."""
    sim = credit_simulation()
    sim.run(until=120)
    view = sim.observe()
    forecasts = history_policy(history_seconds=120).forecaster.predict(view, 0)
    model = OnlineForecastModel(sim.models, sim.config, sim.handoff, sim.slo)
    keep, shrink = model.evaluate_many(view, forecasts, (INITIAL, SMALL))
    assert shrink.supported and shrink.worst_request_fraction >= 0.8
    assert shrink.mean_gpu_seconds < keep.mean_gpu_seconds
    supply = shrink.actor_supply[0]
    assert supply.offered_episodes == 80
    assert supply.arrival_rate == pytest.approx(80 / 120)
    assert supply.actor_seconds_per_batch == 149
    assert supply.capacity_rate == pytest.approx(40 / (149 + 0.002))
    assert supply.credit_balance_rate < 0 and supply.sustainable is False
    assert keep.actor_supply[0].sustainable is True
    assert select_prediction((keep, shrink), INITIAL, 0.8, trigger="slack") == keep
    legacy = tuple(replace(p, actor_supply=()) for p in (keep, shrink))
    assert select_prediction(legacy, INITIAL, 0.8, trigger="slack").layout == SMALL
    bounded = model.search(view, forecasts, (INITIAL, SMALL), trigger="slack")
    assert select_prediction(bounded, INITIAL, 0.8, trigger="slack") == keep


class _HeadroomCosts(_AffineCosts):
    """Declared physical footprints; all shapes fit, some lack policy headroom."""

    def actor_memory(self, layout, batch):
        long = max(t.input_tokens for e in batch for t in e) >= 2048
        return (94.0 if long and layout.actor_tp == 1 else 60.0), 100.0, False


def test_memory_reserve_uses_completed_history_beyond_the_serving_window():
    """Old observed shapes constrain new layouts without reading future episodes."""
    workload = tuple(
        EpisodeSample(
            f"episode-{i:02}",
            "math",
            float(i if i < 8 else 180 + i),
            (Turn(2048 if i < 4 else 16, 4),),
        )
        for i in range(16)
    ) + (EpisodeSample("future", "math", 400, (Turn(60000, 4),)),)
    sim = OnlineSimulation(
        workload,
        INITIAL,
        _HeadroomCosts(),
        config(),
        handoff(),
        SLO,
        input_closes_at=400,
    )
    sim.run(until=200)
    view = sim.observe()
    forecasts = history_policy(history_seconds=120).forecaster.predict(view, 0)
    target = OnlineLayout(2, 1, 2, 1)
    model = OnlineForecastModel(
        sim.models,
        sim.config,
        sim.handoff,
        SLO,
        actor_memory_reserve_fraction=0.1,
    )
    keep, shrink = model.evaluate_many(view, forecasts, (INITIAL, target))
    guard = shrink.actor_memory
    assert keep.supported and keep.actor_memory.admissible
    assert not shrink.supported and not guard.admissible
    assert guard.peak_gib == 94 and guard.allowed_gib == 90
    assert guard.checked_batches == 4
    assert guard.worst_batch == tuple(f"episode-{i:02}" for i in range(4))
    assert guard.observed_through == 200
    assert all(e.completed_at < 80 for e in view.episodes[:4])
    # Same physical simulation remains executable when only the policy margin is off.
    unguarded = OnlineForecastModel(sim.models, sim.config, sim.handoff, SLO)
    assert unguarded.evaluate(view, forecasts, target).supported
    fallback = replace(keep, worst_request_fraction=0.7)
    assert select_prediction((fallback, shrink), INITIAL, 0.8) == fallback
    assert select_prediction((keep, shrink), INITIAL, 0.8, trigger="slack") == keep
    bounded = model.search(view, forecasts, (INITIAL, target), trigger="slack")
    assert select_prediction(bounded, INITIAL, 0.8, trigger="slack") == keep
    # The ablation checks only the original evaluator's workload, so the old
    # long batch is never requested and cannot affect this candidate.
    local = OnlineForecastModel(
        sim.models,
        sim.config,
        sim.handoff,
        SLO,
        actor_memory_reserve_fraction=0.1,
        actor_memory_scope="evaluation",
    )
    local_prediction = local.evaluate(view, forecasts, target)
    assert local_prediction.supported and local_prediction.actor_memory.admissible
    assert local_prediction.actor_memory.scope == "evaluation"
    assert local_prediction.actor_memory.peak_gib == 60
    assert local_prediction.actor_memory.allowed_gib == 90
    assert local._memory_batches == {}


def test_evaluation_reserve_checks_target_calls_without_changing_physical_model():
    """Ordinary evaluated shapes apply headroom to the target Actor only."""
    from areal.autopilot.domains.elastic.online_memory import EvaluationMemoryModels

    physical = _HeadroomCosts()
    target = OnlineLayout(2, 1, 2, 1)
    small = ((Turn(16, 4),),) * 4
    large = ((Turn(2048, 4),),) * 4
    guarded = EvaluationMemoryModels(physical, target, 0.1, 200)
    assert guarded.actor(INITIAL, large) == physical.actor(INITIAL, large)
    assert guarded.report.checked_batches == 0
    assert guarded.actor(target, small) == physical.actor(target, small)
    assert guarded.report.admissible and guarded.report.checked_batches == 1
    with pytest.raises(ValueError, match="memory safety budget"):
        guarded.actor(target, large)
    assert guarded.report.peak_gib == 94 and not guarded.report.admissible
    assert physical.actor(target, large)  # The outer physical model still fits.


def test_memory_reserve_is_carried_into_latest_state_verification():
    """The measured decision clock uses the same configured guard for revalidation."""
    controller = history_policy(history_seconds=120)
    controller.config = replace(
        controller.config,
        actor_memory_reserve_fraction=0.1,
        actor_memory_scope="evaluation",
    )
    sim = OnlineSimulation(
        episodes(16),
        INITIAL,
        _HeadroomCosts(),
        config(),
        handoff(),
        SLO,
        input_closes_at=15,
        policy=controller,
        measured_search_time=True,
    )
    assert sim.control.model(1).actor_memory_reserve_fraction == 0.1
    assert sim.control.model(1).actor_memory_scope == "evaluation"


def test_actor_supply_counts_waiting_arrivals_and_preserves_credit_clock():
    """Admission throttling cannot lower measured demand; decode coalescing agrees."""
    sims = [credit_simulation(SMALL, accelerated=fast) for fast in (False, True)]
    for sim in sims:
        sim.run(until=300)
        view = sim.observe()
        recent = [e for e in view.episodes if 180 <= e.available_at < 300]
        assert any(not e.admitted for e in recent)
        forecasts = history_policy(history_seconds=120).forecaster.predict(view, 0)
        model = OnlineForecastModel(sim.models, sim.config, sim.handoff, sim.slo)
        score = model.evaluate(view, forecasts, SMALL)
        assert score.actor_supply[0].arrival_rate == pytest.approx(len(recent) / 120)
        assert score.actor_supply[0].offered_episodes == len(recent)
        assert score.admission_credit_blocked_seconds > 0
        assert score.admission_credits_start == sim.admission_credits()
        assert sim.admission_credit_blocked_seconds > 0
    assert sims[0].admission_credit_blocked_seconds == pytest.approx(
        sims[1].admission_credit_blocked_seconds, abs=1e-8
    )


@pytest.mark.parametrize("fraction", [0.7, 0.9])
def test_supply_deficit_breaks_pressure_ties_before_cost_and_cannot_prune(fraction):
    """A cheap buffer-depleting incumbent cannot prune a sustainable replacement."""
    deficit = ActorSupply(
        80,
        120,
        2 / 3,
        149,
        1,
        40 / 150,
        40 / 150 - 2 / 3,
        False,
        2,
        0,
        0,
        False,
        "Independent capacity oracle",
    )
    cheap = CandidatePrediction(
        SMALL, True, fraction, 100, ("same-history",), actor_supply=(deficit,)
    )
    enough = replace(
        deficit,
        actor_seconds_per_batch=39,
        capacity_rate=1,
        credit_balance_rate=1 - 2 / 3,
        sustainable=True,
    )
    current = replace(
        cheap, layout=INITIAL, mean_gpu_seconds=200, actor_supply=(enough,)
    )
    assert select_prediction((current, cheap), INITIAL, 0.8) == current
    assert not SearchBound(("view",), 0.8, cheap, fraction_pruning=False).rejection(
        1.0, 201
    )
    assert select_prediction((current, cheap), INITIAL, 0.8, trigger="slack") == current
    # If both deficit and SLO tie, the original GPU cost rule still applies.
    equally_slow = replace(current, actor_supply=(deficit,))
    assert select_prediction((equally_slow, cheap), INITIAL, 0.8) == cheap


@pytest.mark.parametrize("horizon", [2, 20, 120])
def test_window_cost_separates_old_rollout_overlap_from_target_residency(horizon):
    """Only the changed old role is extra; unchanged Actor residency is counted once."""
    sim = OnlineSimulation(
        episodes(8),
        INITIAL,
        _KnownCosts(),
        replace(config(), batch_size=40),
        handoff(),
        SLO,
        input_closes_at=7,
    )
    sim.run(until=5)
    view = sim.observe()
    forecasts = HistoricalReplay(
        HistoryReplayConfig(history_seconds=horizon, max_new_tokens=4096)
    ).predict(view, 0)
    target = OnlineLayout(4, 2, 4, 1)
    model = OnlineForecastModel(sim.models, sim.config, sim.handoff, SLO)
    keep, change = model.evaluate_many(view, forecasts, (INITIAL, target))
    assert keep.supported and change.supported
    assert keep.switching_gpu_seconds == 0
    assert keep.score_gpu_seconds == horizon * 16
    # Startup2 + load1000/1000 + warmup1, no publication/version catch-up.
    assert change.switching_gpu_seconds == pytest.approx(8 * 4)
    assert change.running_gpu_seconds_per_second == 12
    assert change.evaluation_seconds == horizon
    assert change.running_gpu_seconds == horizon * 12
    assert (
        change.score_gpu_seconds
        == change.mean_gpu_seconds
        == pytest.approx(32 + horizon * 12)
    )
    if horizon >= 4:
        assert change.window_gpu_seconds == pytest.approx(change.score_gpu_seconds)
    else:
        assert change.window_gpu_seconds == pytest.approx(horizon * 20)
        assert change.score_gpu_seconds > change.window_gpu_seconds
    assert change.tail_gpu_seconds == 0


@pytest.mark.parametrize("tail_output", [2000, 4000])
def test_fixed_serving_window_does_not_drain_late_requests(tail_output):
    """A long late request remains pending, without extending cost or scoring time."""
    sim = OnlineSimulation(
        episodes(8),
        INITIAL,
        _KnownCosts(),
        replace(config(), batch_size=40),
        handoff(),
        SLO,
        input_closes_at=7,
    )
    sim.run(until=5)
    view = sim.observe()
    original = history_policy().forecaster.predict(view, 0)[0]
    forecast = replace(
        original,
        episodes=tuple(
            e
            for e in original.episodes
            if not e.episode_id.startswith("history-replay")
        )
        + (EpisodeSample("late", "math", 24.999, (Turn(16, tail_output),)),),
    )
    score = OnlineForecastModel(sim.models, sim.config, sim.handoff, SLO).evaluate(
        view, (forecast,), INITIAL
    )
    assert score.supported
    assert score.evaluation_seconds == 20
    assert score.pending_requests == 1
    assert score.failed_pending_requests == 0
    assert score.completed_requests > 0
    assert score.worst_request_fraction == 1
    assert score.mean_gpu_seconds == score.running_gpu_seconds == 320
    assert score.tail_gpu_seconds == 0


def test_fixed_serving_window_counts_overdue_pending_admission_as_failure():
    """A slow candidate cannot hide known TTFT failures behind an unfinished queue."""
    sim = OnlineSimulation(
        episodes(8),
        INITIAL,
        _KnownCosts(),
        replace(config(), batch_size=40, client_concurrency=1),
        handoff(),
        SLO,
        input_closes_at=7,
    )
    sim.run(until=5)
    view = sim.observe()
    original = history_policy().forecaster.predict(view, 0)[0]
    forecast = replace(
        original,
        episodes=tuple(
            e
            for e in original.episodes
            if not e.episode_id.startswith("history-replay")
        )
        + (
            EpisodeSample("long", "math", 5.1, (Turn(16, 4096),)),
            EpisodeSample("queued", "math", 5.2, (Turn(16, 1),)),
        ),
    )
    score = OnlineForecastModel(sim.models, sim.config, sim.handoff, SLO).evaluate(
        view, (forecast,), INITIAL
    )
    assert score.supported and score.pending_requests == 2
    assert score.failed_pending_requests == 1
    assert score.worst_request_fraction == pytest.approx(
        score.completed_requests / (score.completed_requests + 1)
    )
    assert score.score_gpu_seconds == 320


def test_history_evaluation_retains_slo_failures_and_checks_actor_capacity():
    """A late request remains scored, while an unsafe Actor is still excluded."""
    workload = (EpisodeSample("history", "math", 0, (Turn(16, 1),)),) + tuple(
        EpisodeSample(f"live-{i}", "math", 0.1, (Turn(16, 2000),)) for i in range(4)
    )
    sim = OnlineSimulation(
        workload,
        INITIAL,
        _KnownCosts(),
        replace(config(), client_concurrency=1),
        handoff(),
        SLO,
        input_closes_at=0.1,
    )
    sim.run(until=6)
    replay = HistoricalReplay(HistoryReplayConfig(max_new_tokens=4096)).predict(
        sim.observe(), 0
    )
    score = OnlineForecastModel(
        sim.models, sim.config, sim.handoff, ServingSLO(5, 0.1, 0.8)
    ).evaluate(sim.observe(), replay, INITIAL)
    assert score.supported and score.worst_request_fraction < 0.8
    assert score.mean_gpu_seconds == pytest.approx(
        score.window_gpu_seconds + score.tail_gpu_seconds
    )

    blocked = OnlineSimulation(
        episodes(12, 0.1),
        OnlineLayout(1, 1, 8, 1),
        _MemoryLimitedCosts(),
        config(),
        handoff(),
        SLO,
        input_closes_at=1.1,
    )
    blocked.run(until=20)
    view = blocked.observe()
    replay = HistoricalReplay(
        HistoryReplayConfig(
            history_seconds=1, initial_regime_seconds=1, max_new_tokens=4096
        )
    ).predict(view, 0)
    evaluator = OnlineForecastModel(
        blocked.models, blocked.config, blocked.handoff, ServingSLO(5, 0.1, 0.8)
    )
    old, repair = (
        evaluator.evaluate(view, replay, target) for target in (view.layout, INITIAL)
    )
    assert not old.supported and old.actor_coverage == "known_batch_rejected"
    assert (
        repair.supported
        and repair.actor_coverage == "known_complete_batch_capacity_checked"
    )
    assert replay[0].replay.expected_remaining_seconds == 1


def test_history_policy_completes_equal_work_from_eight_plus_eight():
    """The real replay evaluator preserves ordered training and conservative costs."""
    workload = episodes(32, spacing=1)
    baseline = simulation(workload).run()
    dynamic = simulation(workload, policy=history_policy())
    result = dynamic.run()
    assert baseline["status"] == result["status"] == "complete"
    assert result["completed_updates"] == 8 and result["trained_episodes"] == 32
    assert [
        e for record in dynamic.training_records for e in record["episodes"]
    ] == list(dynamic.arrival_order)
    assert any(d["predictions"] for d in dynamic.decisions)
    assert all(not d["forecast_seeds"] for d in dynamic.decisions)


def test_history_decisions_are_independent_of_unseen_future():
    """Changing unseen output/tool/arrival facts leaves the new controller unchanged."""
    prefixes = []
    for output, tool, future_at in ((200, 7, 4), (400, 70, 40)):
        workload = (
            EpisodeSample("history", "math", 0, (Turn(16, 1),)),
            EpisodeSample(
                "active",
                "math",
                0.1,
                (Turn(16, output, tool, 8, "python", 2), Turn(1000, 1)),
            ),
            EpisodeSample("future", "math", future_at, (Turn(16, 1),)),
        )
        sim = simulation(workload, policy=history_policy(history_seconds=0.2))
        sim.run(until=0.8)
        prefixes.append(
            [
                (
                    d["observed_at"],
                    d["observation_id"],
                    d["target"],
                    d["predictions"],
                    d["replay_window"],
                )
                for d in sim.decisions
            ]
        )
    assert prefixes[0] == prefixes[1]


@pytest.mark.parametrize("fraction", [0.7, 0.9])
def test_search_bounds_keep_ties_and_potential_service_improvement(fraction):
    """Only an irreversible, strictly worse score can terminate a candidate."""
    best = CandidatePrediction(INITIAL, True, fraction, 100, ("history",))
    bound = SearchBound(("snapshot",), 0.8, best)
    assert not bound.rejection(fraction, 100)
    assert not bound.rejection(1.0, 1.0)
    assert bound.rejection(0.6, 0.0)
    assert bool(bound.rejection(1.0, 101)) == (fraction >= 0.8)


@pytest.mark.parametrize("ttft", [5.0, 0.00001])
@pytest.mark.parametrize("trigger", ["pressure", "slack"])
def test_bounded_search_matches_exhaustive_historical_selection(ttft, trigger):
    """Both feasible and below-target workloads retain the exhaustive winner."""
    sim = simulation(episodes(16))
    sim.run(until=5)
    view = sim.observe()
    forecasts = history_policy().forecaster.predict(view, 0)
    # Two equal replay samples check mean-cost and worst-fraction aggregation.
    forecasts = forecasts + (replace(forecasts[0], seed=1),)
    layouts = (INITIAL, SMALL, OnlineLayout(2, 2, 4, 1), OnlineLayout(1, 1, 1, 1))
    slo = ServingSLO(ttft, 0.1, 0.8)
    model = OnlineForecastModel(sim.models, sim.config, sim.handoff, slo)
    exhaustive = model.evaluate_many(view, forecasts, layouts)
    bounded = model.search(view, forecasts, layouts, trigger=trigger)
    assert select_prediction(
        bounded, INITIAL, 0.8, trigger=trigger
    ) == select_prediction(exhaustive, INITIAL, 0.8, trigger=trigger)
    assert model.execution_stats["considered"] == len(layouts)
    assert tuple(p.layout for p in bounded) == layouts
    for complete, partial in zip(exhaustive, bounded, strict=True):
        if partial.pruned:
            assert partial.mean_gpu_seconds is None
            assert partial.worst_request_fraction is None
            assert complete.mean_gpu_seconds >= partial.mean_cost_lower_bound - 1e-8
            assert (
                complete.worst_request_fraction <= partial.fraction_upper_bound + 1e-10
            )
        else:
            assert complete == partial


def test_unchanged_search_evidence_is_reused_until_refresh_or_resource_change():
    """Repeated SLO pressure does not repeat the same search at every poll."""
    sim = simulation(episodes(16))
    sim.run(until=5)
    view = sim.observe()
    controller = history_policy(history_seconds=5)
    scores = _ScoredReplay({INITIAL: (0.9, 100, 200), SMALL: (0.9, 110, 150)})
    first = controller.decide(view, scores)
    repeated = controller.decide(replace(view, now=5.2), scores)
    assert first.predictions and not repeated.predictions
    assert len(scores.calls) == 2
    assert "Unchanged recent search evidence" in repeated.reason
    assert not controller.redundant_search(replace(view, now=26), scores)
    controller.search_invalidated()
    assert not controller.redundant_search(view, scores)
    controller.decide(view, scores)
    assert not controller.redundant_search(replace(view, layout=SMALL), scores)


def test_observed_burst_or_changed_cpu_delay_bypasses_search_deduplication():
    """Material backlog changes are checked before any unchanged-search wait."""
    work = episodes(4, spacing=0.1) + tuple(
        EpisodeSample(f"burst-{i}", "math", 6, (Turn(1000, 1000),)) for i in range(16)
    )
    sim = simulation(work)
    sim.run(until=5)
    controller = history_policy(history_seconds=5)
    controller.config = replace(controller.config, cooldown_seconds=60)
    scores = _ScoredReplay({INITIAL: (0.9, 100, 200), SMALL: (0.9, 110, 150)})
    original = sim.observe()
    assert controller.decide(original, scores).predictions
    scores.delay = 2.0
    assert not controller.redundant_search(original, scores)
    scores.delay = 0.0
    sim.run(until=6.1)
    assert not controller.redundant_search(sim.observe(), scores)


def test_cpu_completion_refreshes_features_without_extra_regime_confirmation():
    """Two periodic checks remain two checks even with CPU completions between."""
    sim = simulation(episodes(32))
    sim.run(until=31)
    base = sim.observe()
    history = HistoricalReplay(HistoryReplayConfig(30, 10, 0.5, 2, 100, 4096, 1))

    def at(now):
        return replace(
            base,
            now=now,
            episodes=tuple(
                replace(e, family="terminal" if e.available_at >= 11 else "math")
                for e in base.episodes
                if e.available_at < now
            ),
        )

    history.observe(at(10))
    history.observe(at(20))
    history.predict(at(20.1), 0, confirm_change=False)
    history.observe(at(30), confirm_change=False)
    assert history.regime_index == 0 and history._change_count == 1
    history.observe(at(30))
    history.observe(at(30))
    assert history.regime_index == 1 and history.regime_started_at == 20
    assert history.previous_regime_seconds == 20


class _DecisionClock:
    """Explicit CPU duration oracle isolates online timing from host load."""

    def __init__(self, seconds):
        self.seconds, self.value, self.started = seconds, 0.0, False

    def __call__(self):
        if self.started:
            self.value += self.seconds
        self.started = not self.started
        return self.value


@pytest.mark.parametrize(
    "stage,available_at",
    [("search", 3.6), ("application_wait", 3.6), ("application_wait", 4.8)],
)
def test_normal_publication_keeps_candidate_and_hands_off_latest_state(
    stage, available_at
):
    """Real update/publication events advance while a scored decision arrives."""

    class Costs(_KnownCosts):
        def publication(self, layout, recipients):
            return 1.0

    class Scores(_ScoredReplay):
        execution_stats = {}

        def evaluate_many(self, view, forecasts, layouts):
            return tuple(self.evaluate(view, forecasts, layout) for layout in layouts)

    sim = OnlineSimulation(
        episodes(16), INITIAL, Costs(), config(), handoff(), SLO, input_closes_at=15
    )
    sim.run(until=3.2)
    view = sim.observe()
    assert view.actor.batch and view.training_version == 0
    sim.policy = history_policy()
    sim.policy.config = replace(
        sim.policy.config, check_interval_seconds=10, decision_delay_seconds=0.05
    )
    control = sim.control = OnlineControl(sim, clock=_DecisionClock(0))
    regime = control.observe_history(view)
    scores = Scores({INITIAL: (1.0, 100, 100), SMALL: (1.0, 50, 50)})
    predictions = scores.evaluate_many(view, (), (INITIAL, SMALL))
    scores.calls.clear()
    control.model = lambda delay: scores
    decision = OnlineDecision(
        view.now, "slack", SMALL, predictions, (), rolling_slo(view, SLO, 20), 0, "test"
    )
    control.pending = _Search(
        1,
        view,
        decision,
        dict(asdict(decision), search_wall_seconds=available_at - view.now),
        regime,
        stage=stage,
        validation=predictions if stage == "application_wait" else (),
        validation_view=view if stage == "application_wait" else None,
    )
    sim.schedule_at(
        available_at,
        "search_complete" if stage == "search" else "decision_ready",
        1,
        priority=9,
    )
    result = sim.run()
    assert result["status"] == "complete" and result["completed_updates"] == 4
    assert result["completed_episodes"] == result["trained_episodes"] == 16
    assert len(sim.decisions) == 1
    applied = sim.decisions[0]
    assert applied["application_status"] == "applied"
    assert applied["target"] == asdict(SMALL)
    assert applied["application_state"] == dict(
        training_version=1, provider_version=1, consumed_episodes=4
    )
    assert scores.calls == ([INITIAL, SMALL] if stage == "search" else [])
    assert control.stats["publication_waits"] == (available_at == 3.6)
    published = next(e for e in sim.events if e["kind"] == "weights_published")
    expected = max(available_at, published["time"])
    if stage == "search":
        expected += 0.05
    assert applied["applied_at"] == pytest.approx(expected)
    assert control.stats["discarded"] == 0
    cutover = next(e for e in sim.events if e["kind"] == "actor_cutover")
    assert cutover["version"] >= 1 and cutover["cursor"] >= 4


@pytest.mark.parametrize("interval", [1, 2.5])
@pytest.mark.parametrize("measured_search_time", [False, True])
def test_policy_observes_at_the_configured_seconds_interval(
    interval, measured_search_time
):
    """Both control paths check at x, 2x, ... without hidden extra checks."""
    controller = history_policy()
    controller.config = replace(
        controller.config,
        check_interval_seconds=interval,
        minimum_completed_requests=10000,
    )
    controller.slo = ServingSLO(100, 100, 0.8)
    sim = OnlineSimulation(
        episodes(64, spacing=1),
        INITIAL,
        _KnownCosts(),
        config(),
        handoff(),
        controller.slo,
        input_closes_at=63,
        policy=controller,
        measured_search_time=measured_search_time,
        search_clock=_DecisionClock(0),
    )
    sim.run(until=6.1)
    assert [d["observed_at"] for d in sim.decisions] == pytest.approx(
        [interval * n for n in range(1, int(6.1 / interval) + 1)]
    )


def test_measured_decisions_allow_serving_and_apply_only_after_latest_verification():
    """Old GPUs keep working while CPU search and revalidation consume time."""
    workload = episodes(16, spacing=3, output=220)
    controller = history_policy()
    controller.config = replace(controller.config, check_interval_seconds=1)
    controller.forecaster = HistoricalReplay(
        replace(controller.forecaster.config, feature_window_seconds=10)
    )
    serving_events = []
    sim = OnlineSimulation(
        workload,
        INITIAL,
        _KnownCosts(),
        config(),
        handoff(),
        SLO,
        input_closes_at=45,
        policy=controller,
        measured_search_time=True,
        search_clock=_DecisionClock(0.3),
        event_sink=serving_events.append,
    )
    result = sim.run()
    assert result["status"] == "complete" and result["completed_updates"] == 4
    records = [d for d in sim.decisions if d["predictions"]]
    assert records and result["control"]["verifications"] > 0
    assert any(d["application_status"] == "applied" for d in records)
    assert any("exceeded its modeled CPU delay" in d["reason"] for d in records)
    for decision in records:
        assert (
            decision["decision_available_at"] >= decision["observed_at"] + 0.3 - 1e-10
        )
        if decision["application_status"] == "applied":
            assert (
                decision["applied_at"]
                >= decision["verification_started_at"] + 0.3 - 1e-10
            )
            assert decision["applied_at"] == pytest.approx(
                decision["verification_started_at"]
                + decision["verification_delay_estimate_seconds"]
            )
    assert any(
        d["observed_at"] < e["time"] < d["search_available_at"]
        for d in records
        if "search_available_at" in d
        for e in serving_events
        if e["kind"] in {"service_batch", "request_complete"}
    )
    assert (
        sum(e["kind"] == "search_start" for e in sim.events)
        == result["control"]["searches"]
    )


def test_finite_work_does_not_wait_for_irrelevant_cpu_search():
    """A pending decision cannot extend GPU leases after the work is drained."""
    workload = episodes(8, spacing=1)
    sim = OnlineSimulation(
        workload,
        INITIAL,
        _KnownCosts(),
        config(),
        handoff(),
        SLO,
        input_closes_at=7,
        policy=history_policy(history_seconds=1),
        measured_search_time=True,
        search_clock=_DecisionClock(100),
    )
    result = sim.run()
    assert result["status"] == "complete" and result["completed_updates"] == 2
    assert result["control"]["cancelled"] == 1
    assert sim.now < 10 and sim.control.pending is None


def test_measured_decision_prefix_cannot_read_hidden_future_work():
    """CPU availability events use the same causal snapshot under unseen changes."""
    prefixes = []
    for output, tool, future_at in ((200, 7, 4), (400, 70, 40)):
        work = (
            EpisodeSample("history", "math", 0, (Turn(16, 1),)),
            EpisodeSample(
                "active",
                "math",
                0.1,
                (Turn(16, output, tool, 8, "python", 2), Turn(1000, 1)),
            ),
            EpisodeSample("future", "math", future_at, (Turn(16, 1),)),
        )
        sim = OnlineSimulation(
            work,
            INITIAL,
            _KnownCosts(),
            config(),
            handoff(),
            SLO,
            input_closes_at=future_at,
            policy=history_policy(history_seconds=0.2),
            measured_search_time=True,
            search_clock=_DecisionClock(0.1),
        )
        sim.run(until=0.8)
        prefixes.append(
            [
                (e["kind"], e["time"], e.get("observation_id"), e.get("target"))
                for e in sim.events
                if e["kind"]
                in {"search_start", "search_complete", "verification_start", "decision"}
            ]
        )
    assert prefixes[0] == prefixes[1]


def test_shared_sixteen_gpu_candidates_include_cross_node_data_parallel_layouts():
    """Batch40 and TP<=8 remain fixed while the two roles can share16 GPUs."""
    from examples.autopilot.elastic.simulate_online import candidates

    original = candidates(40)
    expanded = candidates(40, role_gpu_limit=16)
    assert len(original) == 165 and len(expanded) == 241
    assert set(original) <= set(expanded)
    assert OnlineLayout(10, 1, 6, 1) in expanded
    assert OnlineLayout(5, 2, 6, 1) in expanded
    assert OnlineLayout(1, 2, 14, 1) in expanded
    assert all(c.actor_gpus + c.rollout_gpus <= 16 for c in expanded)
    assert all(40 % c.actor_dp == 0 for c in expanded)


def test_cross_node_leases_preserve_tp_groups_and_all_gpu_charges():
    """DP may span two nodes; every TP group stays within its own node."""
    pool = GPUAllocator((8, 8, 8, 8))
    devices = pool.acquire("actor", "actor", 10, tp=2)
    assert len({node for node, _ in devices}) == 2
    assert all(
        len({node for node, _ in devices[i : i + 2]}) == 1 for i in range(0, 10, 2)
    )
    assert pool.acquire("rollout", "rollout", 6, tp=1) is not None
    pool.advance(3)
    assert pool.gpu_seconds == {"actor": 30, "rollout": 18}
    assert pool.peak_gpus == 16


def test_cross_node_snapshot_restore_retains_placement_and_complete_work():
    """The causal fork keeps the observed leases instead of repacking nodes."""
    work = episodes(80, spacing=0.2, output=10)
    cfg = replace(config(), batch_size=40)
    layout = OnlineLayout(5, 2, 6, 1)
    sim = OnlineSimulation(
        work,
        layout,
        _KnownCosts(),
        cfg,
        handoff(),
        SLO,
        input_closes_at=work[-1].available_at,
    )
    sim.run(until=0.5)
    view = sim.observe()
    forecast = Forecast(0, work, work[-1].available_at)
    clone = OnlineSimulation.from_forecast(
        view, forecast, sim.models, cfg, sim.handoff, SLO
    )
    assert sorted(
        (role, devices) for role, devices in clone.pool.leases.values()
    ) == sorted((role, devices) for _, role, devices in view.leases)
    original, restored = sim.run(), clone.run()
    assert original["status"] == restored["status"] == "complete"
    assert original["completed_updates"] == restored["completed_updates"] == 2
    assert original["total_gpu_seconds"] - view.now * 16 == pytest.approx(
        restored["total_gpu_seconds"]
    )


def test_cross_node_hot_handoff_keeps_work_and_charges_both_model_images():
    """A10+6 deployment changes to4+12 without overlapping TP groups or lost work."""
    work = episodes(80, spacing=0.2, output=100)
    cfg = replace(config(), batch_size=40)
    sim = OnlineSimulation(
        work,
        OnlineLayout(5, 2, 6, 1),
        _KnownCosts(),
        cfg,
        handoff(),
        SLO,
        input_closes_at=work[-1].available_at,
    )
    target = OnlineLayout(2, 2, 12, 1)
    sim.schedule_at(1, "decision_apply", target)
    result = sim.run()
    assert result["status"] == "complete" and result["completed_updates"] == 2
    assert result["final_layout"] == asdict(target)
    assert result["peak_gpus"] == 32 and not result["remaining_leases"]
    loads = [
        e
        for e in sim.events
        if e["kind"] == "deployment_load" and e["role"] == "rollout"
    ]
    assert len(loads) == 1 and loads[0]["bytes"] == 2 * sim.handoff.weight_bytes
    assert [e for t in sim.training_records for e in t["episodes"]] == list(
        sim.arrival_order
    )


def test_handoff_may_temporarily_activate_more_than_sixteen_gpus():
    """New Rollout may cut over before the old Actor reaches its update boundary."""
    work = episodes(80, spacing=0, output=4)
    cfg = replace(config(), batch_size=40)
    transition = replace(
        handoff(),
        actor_startup_seconds=0,
        rollout_startup_seconds=0,
        actor_warmup_seconds=0,
        rollout_warmup_seconds=0,
        effective_bandwidth_GBps=1,
    )
    sim = OnlineSimulation(
        work,
        OnlineLayout(5, 2, 6, 1),
        _KnownCosts(),
        cfg,
        transition,
        SLO,
        input_closes_at=0,
    )
    sim.schedule_at(0.1, "decision_apply", OnlineLayout(2, 2, 12, 1))
    sim.run(until=0.2)
    assert sim.layout.actor_gpus + sim.layout.rollout_gpus > 16
    assert sim.pending_handoff is not None
    result = sim.run()
    assert result["status"] == "complete" and result["completed_updates"] == 2
    assert result["peak_gpus"] <= 32 and not result["remaining_leases"]
