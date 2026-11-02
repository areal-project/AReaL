# SPDX-License-Identifier: Apache-2.0

"""Native PPO reconfigurator integration with real durable transactions; CPU protocol only."""

import json
from dataclasses import replace

import pytest

from tests.test_elastic_checkpoint import (
    _complete_native_update,
    _ready_native_episodes,
    native_lifecycle,  # noqa: F401 -- shared native controller fixture
)

from areal.autopilot.domains.elastic.native import RevisionAdapter
from areal.elastic.infra.activation import (
    ReconfiguratorBoundaryBusy,
    ReconfiguratorRequest,
)


@pytest.fixture
def native_reconfigurator(request):
    fixture = request.getfixturevalue("native_lifecycle")
    fixture.manager.close()
    runtime = revision_adapter()
    manager = fixture.create(runtime)
    yield manager, runtime, fixture
    manager.close()


def update(manager):
    _ready_native_episodes(
        manager, manager.applied_version * 4, manager.applied_version
    )
    assert len(manager.next_batch()) == 4
    manager.actor.updates += 1
    _complete_native_update(manager, manager.applied_version + 1)


def revision_adapter():
    """Explicit synthetic component/transition context; no native timing labels."""
    from examples.autopilot.elastic.revision_demo import demo_inputs

    from areal.autopilot.domains.elastic.contracts import decode
    from areal.elastic.simulation.online_transitions import HandoffConfig
    from areal.elastic.simulation.simulation_core import SimulationConfig

    _, _, spec = demo_inputs()
    return RevisionAdapter(
        run_config=decode(SimulationConfig, spec["run"]),
        handoff=replace(decode(HandoffConfig, spec["handoff"]), mode="cold"),
    )


def accepted_native_bundle(runtime, root, *, native=True):
    """Real acceptance on declared formula cases, bound to the native contract."""
    from examples.autopilot.elastic.revision_demo import (
        GOOD_MODEL_SOURCE,
        demo_evaluator,
        demo_inputs,
    )
    from tests.test_elastic_online_revision import runner

    from areal.autopilot.domains.elastic.contracts import BASELINE_SOURCE
    from areal.autopilot.domains.elastic.cycle import ReconfiguratorRevisionCycle
    from areal.autopilot.revision import Proposal
    from areal.autopilot.revision.execution import load_source
    from areal.elastic.simulation.online_simulation import OnlineSimulation

    _, problems, _ = demo_inputs()
    models = load_source(GOOD_MODEL_SOURCE).build_model((), {})
    problem = problems[0]
    observed = OnlineSimulation.create(
        problem.forecasts[0].episodes,
        runtime.placements.current(),
        models,
        runtime.run_config,
        runtime.handoff,
        runtime.reconfigurator.slo,
        input_closes_at=20,
    ).observe()
    problems = (
        replace(
            problem, observation=observed, candidates=runtime.reconfigurator.candidates
        ),
    )

    class Agent:
        def propose(self, request):
            return Proposal(
                GOOD_MODEL_SOURCE
                if request.target == "model"
                else BASELINE_SOURCE["optimizer"]
            )

    assembly = runtime.assembly()
    if not native:
        assembly.pop("deployment")
    result = ReconfiguratorRevisionCycle(
        demo_evaluator("model", runner.__wrapped__()),
        problems,
        root,
        assembly=assembly,
    ).run(Agent(), Agent(), max_model_rounds=1, max_optimizer_rounds=1)
    assert result["model_accepted"] and result["bundle"], result
    return result["bundle"]


def test_native_reconfigurator_records_published_updates_and_restores_history(
    native_reconfigurator,
):
    manager, runtime, fixture = native_reconfigurator
    for _ in range(3):
        update(manager)
    history = runtime.runtime_history.snapshot(3)
    assert len(history["updates"]) == 3
    assert [row["weights_version"] for row in history["updates"]] == [1, 2, 3]
    assert all(row["workload"]["samples"] == 4 for row in history["updates"])
    assert runtime.observe().consumed_episodes == 12
    assert manager.monitor.completed_tokens == 24
    assert all(not row["missing_phases"] for row in history["updates"])
    assert {row["phase"] for row in history["updates"][0]["measurements"]} >= {
        "start",
        "train",
        "checkpoint",
        "publish",
    }
    assert manager.rollout is fixture.rollout
    manager.close()
    history = runtime.runtime_history.snapshot(3)
    recovered_runtime = revision_adapter()
    recovered = fixture.create(recovered_runtime)
    try:
        assert recovered.applied_version == 3 and recovered.actor is None
        assert recovered_runtime.runtime_history.snapshot(3) == history
        assert recovered.monitor.completed_tokens == 24
        update(recovered)
        assert recovered.actor.updates == 4
    finally:
        recovered.close()


def test_native_reconfigurator_tracks_updates_between_periodic_checkpoints(
    native_reconfigurator,
):
    """Measured skipped saves are distinct from missing checkpoint measurements."""
    manager, runtime, _ = native_reconfigurator
    manager.checkpoint_frequency.step_ctl.frequency_steps = 3
    for _ in range(3):
        update(manager)
    rows = runtime.runtime_history.snapshot(3)["updates"]
    assert [row["checkpointed"] for row in rows] == [False, False, True]
    assert [row["checkpoint_id"] for row in rows[:2]] == [None, None]
    assert all(not row["missing_phases"] for row in rows)
    assert manager.monitor.completed_tokens == 24
    assert manager.inbox.status()["checkpoint_version"] == 3


def test_native_reconfigurator_rollback_removes_unsaved_history_before_replay(
    native_reconfigurator,
):
    """Abandoned weight versions cannot collide with replayed runtime records."""
    manager, runtime, _ = native_reconfigurator
    update(manager)
    manager.checkpoint_frequency.step_ctl.frequency_steps = 100
    update(manager)
    update(manager)
    assert len(runtime.runtime_history.snapshot(3)["updates"]) == 3
    manager.recover_uncertain(RuntimeError("lost unsaved GPU state"))
    assert manager.applied_version == 1 and manager.monitor.completed_tokens == 8
    assert len(runtime.runtime_history.snapshot(3)["updates"]) == 1
    for version in (2, 3):
        manager.next_batch()
        manager.actor.updates += 1
        _complete_native_update(manager, version)
    assert len(runtime.runtime_history.snapshot(3)["updates"]) == 3
    assert manager.monitor.completed_tokens == 24


def test_native_publication_failure_replays_unsaved_update_and_records_history_once(
    native_reconfigurator,
):
    manager, runtime, _ = native_reconfigurator
    _ready_native_episodes(manager, 0, 0)
    manager.next_batch()
    manager.actor.updates += 1
    manager.actor.fail_publication = True
    with pytest.raises(TimeoutError):
        manager.publish_update(1)
    assert runtime.runtime_history.snapshot(1)["updates"] == []
    manager.recover_uncertain(TimeoutError("ack lost"))
    assert manager.applied_version == manager.actor.updates == 0
    manager.next_batch()
    manager.actor.updates += 1
    _complete_native_update(manager, 1)
    assert manager.applied_version == manager.actor.updates == 1
    assert len(runtime.runtime_history.snapshot(1)["updates"]) == 1
    manager.monitor.complete_update()
    assert len(runtime.runtime_history.snapshot(1)["updates"]) == 1
    assert manager.inbox.status()["active_credits"] == 0


def test_native_reconfigurator_rejects_legacy_bundle_and_activates_native_acceptance(
    native_reconfigurator, tmp_path
):
    manager, runtime, fixture = native_reconfigurator
    legacy = accepted_native_bundle(runtime, tmp_path / "legacy", native=False)
    with pytest.raises(ValueError, match="native execution contract"):
        runtime.prepare_reconfigurator(legacy)
    bundle = accepted_native_bundle(runtime, tmp_path / "native")
    assert (
        json.loads(open(bundle).read())["kind"] == "online-reference-reconfigurator-v1"
    )
    prepared = runtime.prepare_reconfigurator(bundle)
    _ready_native_episodes(manager, 0, 0)
    manager.next_batch()
    request = ReconfiguratorRequest("activate", prepared.bundle_id, 0, 0)
    with pytest.raises(ReconfiguratorBoundaryBusy):
        runtime.apply_reconfigurator(request, prepared)
    manager.actor.updates += 1
    _complete_native_update(manager, 1)
    request = replace(request, expected_weights_version=1)
    before = manager.applied_version, manager.actor, fixture.actors[:]
    receipt = runtime.apply_reconfigurator(request, prepared)
    assert runtime.apply_reconfigurator(request, prepared) == receipt
    assert runtime.reconfigurator_generation == 1
    assert (manager.applied_version, manager.actor, fixture.actors) == before
    update(manager)
    assert manager.actor.updates == 2
    manager.close()
    restored_runtime = revision_adapter()
    restored = fixture.create(restored_runtime)
    try:
        assert restored_runtime.reconfigurator_generation == 1
        assert restored.applied_version == 2 and restored.actor is None
        assert len(restored_runtime.history.snapshot().workloads) == 2
    finally:
        restored.close()


def test_native_empty_wait_releases_lock_and_can_be_closed(
    native_reconfigurator, monkeypatch
):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    manager, runtime, _ = native_reconfigurator
    polled = threading.Event()
    original = runtime.poll

    def poll():
        polled.set()
        original()

    monkeypatch.setattr(runtime, "poll", poll)
    with ThreadPoolExecutor(max_workers=1) as executor:
        waiting = executor.submit(manager.next_batch)
        assert polled.wait(2)
        manager.close()
        with pytest.raises(RuntimeError, match="closed"):
            waiting.result(timeout=2)
    assert runtime.closed and manager.actor is None


def test_native_cleanup_closes_reconfigurator_and_consumer_after_admission_disk_failure(
    native_reconfigurator, monkeypatch
):
    manager, runtime, fixture = native_reconfigurator
    update(manager)
    with monkeypatch.context() as patch:
        patch.setattr(
            manager.inbox,
            "set_admission_open",
            lambda value: (_ for _ in ()).throw(OSError("disk full")),
        )
        with pytest.raises(ExceptionGroup, match="cleanup"):
            manager.close()
    assert runtime.closed and manager._closed and manager.actor is None
    recovered = fixture.create(revision_adapter())
    recovered.close()


def test_native_revision_attaches_to_ordinary_checkpoint_with_shared_monitor(request):
    """An ordinary run already has all measurement state required by revision."""
    fixture = request.getfixturevalue("native_lifecycle")
    update(fixture.manager)
    fixture.manager.close()
    context = fixture.manager.monitor.runtime_history.context
    resumed = fixture.create(revision_adapter())
    try:
        assert resumed.applied_version == 1
        assert resumed.monitor.completed_tokens == 8
        assert (
            resumed.reconfigurator_adapter.runtime_history
            is resumed.monitor.runtime_history
        )
        assert resumed.monitor.runtime_history.context == context
        assert len(resumed.reconfigurator_adapter.history.snapshot().workloads) == 1
    finally:
        resumed.close()


def test_native_uncertain_reconfigurator_pointer_recovers_without_closed_admission_deadlock(
    native_reconfigurator, tmp_path, monkeypatch
):
    import areal.elastic.infra.activation as store

    manager, runtime, _ = native_reconfigurator
    bundle = accepted_native_bundle(runtime, tmp_path / "revision")
    prepared = runtime.prepare_reconfigurator(bundle)
    original = store.atomic_json

    def lost_ack(path, value):
        original(path, value)
        if path.name == "active.json" and value["generation"] == 1:
            raise RuntimeError("lost reconfigurator acknowledgement")

    with monkeypatch.context() as patch:
        patch.setattr(store, "atomic_json", lost_ack)
        with pytest.raises(RuntimeError, match="acknowledgement"):
            runtime.apply_reconfigurator(
                ReconfiguratorRequest("activate", prepared.bundle_id, 0, 0), prepared
            )
    assert runtime.failed and not manager.inbox.status()["admission_open"]
    with pytest.raises(RuntimeError, match="Recover.*reconfigurator"):
        manager._poll_batch()
    manager.recover_uncertain(RuntimeError("activation outcome unknown"))
    assert runtime.reconfigurator_generation == 1 and manager.applied_version == 0
    assert manager.inbox.status()["admission_open"]
    update(manager)
    assert manager.actor.updates == 1


def test_native_history_disk_failure_still_closes_reconfigurator_and_releases_consumer(
    native_reconfigurator, monkeypatch
):
    manager, runtime, fixture = native_reconfigurator
    update(manager)
    with monkeypatch.context() as patch:
        patch.setattr(
            runtime.runtime_history,
            "append_boundary",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("history disk full")),
        )
        manager.close()
        assert not runtime.measurement_ready
    assert runtime.closed and manager._closed
    assert manager.actor is None and manager.owner is None
    recovered = fixture.create(revision_adapter())
    recovered.close()


def test_native_background_revision_uses_committed_host_and_durable_window_budget(
    native_reconfigurator, tmp_path
):
    from examples.autopilot.elastic.revision_demo import demo_evaluator, demo_inputs
    from tests.test_elastic_online_revision import runner

    from areal.autopilot.domains.elastic.contracts import BASELINE_SOURCE
    from areal.autopilot.domains.elastic.online import RuntimeRevisionLoop
    from areal.autopilot.revision import Proposal

    manager, runtime, _ = native_reconfigurator
    captured = []

    def builder(snapshot, assembly, directory):
        captured.append((snapshot, assembly))
        from examples.autopilot.elastic.revision_demo import GOOD_MODEL_SOURCE

        from areal.autopilot.domains.elastic.contracts import decode
        from areal.autopilot.revision.execution import load_source
        from areal.elastic.contracts import OnlineLayout
        from areal.elastic.simulation.online_simulation import OnlineSimulation

        catalog = tuple(decode(OnlineLayout, c) for c in assembly["candidates"])
        problem = demo_inputs()[1][0]
        models = load_source(GOOD_MODEL_SOURCE).build_model((), {})
        observed = OnlineSimulation.create(
            problem.forecasts[0].episodes,
            catalog[0],
            models,
            runtime.run_config,
            runtime.handoff,
            runtime.reconfigurator.slo,
            input_closes_at=20,
        ).observe()
        problems = (replace(problem, observation=observed, candidates=catalog),)
        return demo_evaluator("model", runner.__wrapped__()), problems

    class Unchanged:
        def propose(self, request):
            return Proposal(BASELINE_SOURCE[request.target])

    runtime.revision = RuntimeRevisionLoop(
        tmp_path / "revision",
        problem_builder=builder,
        model_agent=Unchanged(),
        optimizer_agent=Unchanged(),
        max_windows=1,
        max_model_rounds=1,
        max_optimizer_rounds=1,
    )
    try:
        for _ in range(4):
            update(manager)
        manager.revision_hooks.poll()
        assert runtime.revision.future is not None
        runtime.revision.future.result(timeout=30)
        manager.revision_hooks.poll()
        outcome = json.loads(
            (tmp_path / "revision/window-0001/outcome.json").read_text()
        )
        assert outcome["status"] == "unchanged", outcome
        assert len(captured) == 1
        assert len(captured[0][0]["updates"]) == 4
        assert captured[0][1]["handoff"]["mode"] == "cold"
        assert runtime.reconfigurator_generation == 0 and manager.applied_version == 4
        for _ in range(4):
            update(manager)
        manager.revision_hooks.poll()
        assert len(list((tmp_path / "revision").glob("window-*"))) == 1
    finally:
        manager.close()
    assert runtime.closed


@pytest.mark.parametrize("enabled", [False, True])
def test_hermes_entry_preserves_config_arguments_and_optionally_attaches_native_reconfigurator(
    monkeypatch, enabled
):
    from examples.hermes import train

    config = object()
    seen = {}

    def load(args, config_type):
        seen["args"] = args
        return config, None

    class Trainer:
        def __init__(self, actual_config, **kwargs):
            assert actual_config is config
            seen.update(kwargs)

        def __enter__(self):
            return self

        def train(self):
            seen["trained"] = True

        def __exit__(self, *args):
            if "elastic_reconfigurator" in seen:
                seen["elastic_reconfigurator"].close()

    monkeypatch.setattr(train, "load_expr_config", load)
    monkeypatch.setattr(train, "PPOTrainer", Trainer)
    config_args = [
        "--config",
        "examples/hermes/config.yaml",
        "actor_lifecycle.mode=on_demand",
    ]
    train.main([*(["--elastic-revision"] if enabled else []), *config_args])
    assert seen["args"] == config_args and seen["trained"]
    assert ("elastic_reconfigurator" in seen) == enabled
    if enabled:
        assert isinstance(seen["elastic_reconfigurator"], RevisionAdapter)
        assert seen["elastic_reconfigurator"].closed


def test_hermes_measured_profile_selects_areal_adapter_without_autopilot(
    monkeypatch, tmp_path
):
    """Profile-only runs use the ordinary adapter and default Monitor."""
    import builtins

    from examples.hermes import train

    from areal.elastic.infra.adapter import ReconfiguratorAdapter

    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"reconfigurator": {}}))
    seen = {}

    class Trainer:
        def __init__(self, config, **kwargs):
            seen.update(kwargs)

        def __enter__(self):
            return self

        def train(self):
            seen["trained"] = True

        def __exit__(self, *args):
            seen["elastic_reconfigurator"].close()

    original_import = builtins.__import__

    def without_autopilot(name, *args, **kwargs):
        if name == "areal.autopilot" or name.startswith("areal.autopilot."):
            raise AssertionError("Ordinary Online entry must not import AutoPilot")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(train, "load_expr_config", lambda *args: (object(), None))
    monkeypatch.setattr(train, "PPOTrainer", Trainer)
    monkeypatch.setattr(builtins, "__import__", without_autopilot)
    train.main(["--elastic-profile", str(profile), "actor_lifecycle.mode=on_demand"])
    assert type(seen["elastic_reconfigurator"]) is ReconfiguratorAdapter
    assert seen["elastic_reconfigurator"].hooks is None and seen["trained"]


def test_duplicate_reconfigurator_owner_rejection_does_not_close_original_trainer(
    native_reconfigurator,
):
    from areal import PPOTrainer

    manager, runtime, fixture = native_reconfigurator
    with pytest.raises(ValueError, match="already bound"):
        PPOTrainer(fixture.config, elastic_reconfigurator=runtime)
    with pytest.raises(ValueError, match="already bound"):
        fixture.create(runtime)
    assert not runtime.closed and runtime.owner is manager
    update(manager)
    assert manager.applied_version == 1


def test_history_append_failure_preserves_update_and_pauses_revision(
    native_reconfigurator, monkeypatch
):
    """An optional history failure never replays a successfully published PPO."""
    from unittest.mock import MagicMock

    from tests.test_elastic_batch_queue import _receive_episode

    manager, runtime, fixture = native_reconfigurator
    update(manager)
    for number in range(4, 8):
        _receive_episode(manager.inbox, number, version=1, rows=2)
    manager.next_batch()
    manager.actor.updates += 1
    with monkeypatch.context() as patch:
        patch.setattr(
            runtime.runtime_history,
            "append",
            lambda *args: (_ for _ in ()).throw(OSError("history unavailable")),
        )
        _complete_native_update(manager, 2)
    assert manager.applied_version == manager.published_version == 2
    assert manager.actor.updates == 2 and manager.active is None
    assert not runtime.measurement_ready and not runtime.reconfigurator_boundary_ready
    revision = MagicMock()
    monkeypatch.setattr(runtime, "revision", revision)
    manager.revision_hooks.poll()
    revision.tick.assert_not_called()
    expected = manager.data_progress.copy()
    assert manager.monitor.completed_tokens == expected["effective_tokens"]
    manager.save_checkpoint(reason="reconfiguration")
    assert manager.latest_manifest["data_progress"] == expected
    assert manager.consumer.cursor["applied_batches"] == 2
    manager.close()
    restored = fixture.create(revision_adapter())
    try:
        assert restored.applied_version == 2 and restored.data_progress == expected
        assert restored.monitor.runtime_history.snapshot(2)["updates"] == []
        restored.ensure_actor()
        assert restored.actor.updates == 2
    finally:
        restored.close()


def test_native_public_control_serializes_reconfigurator_activation_and_reports_boundary(
    native_reconfigurator,
):
    import threading

    from areal.elastic.infra.activation import ReconfiguratorBoundaryBusy

    manager, runtime, _ = native_reconfigurator
    entered, finish = threading.Event(), threading.Event()
    before = manager.control_snapshot()
    assert before.boundary_ready and not before.closed

    def lifecycle_operation():
        with manager.control_operation(blocking=True):
            entered.set()
            assert finish.wait(5)

    worker = threading.Thread(target=lifecycle_operation)
    worker.start()
    try:
        assert entered.wait(5)
        with pytest.raises(ReconfiguratorBoundaryBusy, match="busy"):
            runtime.apply_reconfigurator(None, None)
    finally:
        finish.set()
        worker.join(timeout=5)
    assert not worker.is_alive()
    assert manager.control_snapshot() == before
    assert runtime.reconfigurator_generation == 0 and not runtime.failed
    manager.fence_admission()
    assert not manager.inbox.status()["admission_open"]


def test_accepted_bundle_still_rejects_protected_comment_changes(
    native_reconfigurator, tmp_path, monkeypatch
):
    """Measurement compatibility never relaxes accepted-bundle source proofs."""
    from pathlib import Path

    from areal.autopilot.domains.elastic.bundle import validate_bundle
    from areal.elastic.infra.monitor.identity import measurement_identity

    manager, runtime, _ = native_reconfigurator
    bundle = json.loads(
        Path(accepted_native_bundle(runtime, tmp_path / "proof")).read_text()
    )
    validate_bundle(bundle)
    original = Path.read_text
    identity = measurement_identity(manager)

    def changed_trace(path, *args, **kwargs):
        source = original(path, *args, **kwargs)
        if path.as_posix().endswith("elastic/infra/monitor/trace.py"):
            source += "\n# a harmless offline parser comment\n"
        return source

    monkeypatch.setattr(Path, "read_text", changed_trace)
    assert measurement_identity(manager) == identity
    with pytest.raises(ValueError, match="code changed"):
        validate_bundle(bundle)


@pytest.mark.parametrize("fault", ["snapshot_io", "missing_metrics", "changed_series"])
def test_native_revision_pauses_on_unavailable_measurement_window(
    native_reconfigurator, tmp_path, monkeypatch, fault
):
    """Missing observations cannot launch a revision or fail the native PPO loop."""
    from unittest.mock import MagicMock

    from areal.autopilot.domains.elastic.online import RuntimeRevisionLoop
    from areal.utils.atomic import atomic_json

    initial, adapter, fixture = native_reconfigurator
    initial.close()
    fixture.config.actor_lifecycle.state_root = str(tmp_path / "measured-fault")
    host = revision_adapter()
    manager = fixture.create(host)
    loop = host.revision = RuntimeRevisionLoop(
        tmp_path / "revision-fault",
        problem_builder=MagicMock(),
        model_agent=MagicMock(),
        optimizer_agent=MagicMock(),
        max_windows=1,
        max_model_rounds=1,
        max_optimizer_rounds=1,
    )
    try:
        for _ in range(4):
            update(manager)
        snapshot = MagicMock(side_effect=OSError("measurement window unavailable"))
        if fault == "changed_series":
            atomic_json(
                loop.root / "execution.json",
                {
                    "application": host.reconfigurator_application.context,
                    "history": "previous-measurement-series",
                },
            )
        else:
            monkeypatch.setattr(host.runtime_history, "snapshot", snapshot)
            if fault == "missing_metrics":
                manager.monitor.mark_unavailable(OSError("earlier metrics failure"))
        loop.tick(host)
        assert loop.future is None and not list(loop.root.glob("window-*"))
        assert not host.measurement_ready
        assert manager.applied_version == manager.published_version == 4
        assert manager.actor.updates == 4
        if fault == "missing_metrics":
            snapshot.assert_not_called()
    finally:
        manager.close()
