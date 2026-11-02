# SPDX-License-Identifier: Apache-2.0

"""Default Monitor with real durable stores and CPU protocol workers."""

import json

import pytest

from tests.test_elastic_checkpoint import (
    _complete_native_update,
    _ready_native_episodes,
    native_lifecycle,  # noqa: F401 -- shared native controller fixture
)


def _update(manager):
    version = manager.applied_version
    _ready_native_episodes(manager, version * 4, version)
    manager.next_batch()
    manager.actor.updates += 1
    _complete_native_update(manager, version + 1)


def test_default_monitor_records_causal_inputs_only_after_update_finishes(
    request,
):
    """Publication and periodic saving precede the completed-history boundary."""
    manager = request.getfixturevalue("native_lifecycle").manager
    monitor = manager.monitor
    _ready_native_episodes(manager, 0, 0)
    manager.next_batch()
    assert monitor.pending["history"]["workloads"] == ()
    assert monitor.pending["observation"]["committed_tokens"] == 0
    manager.actor.updates += 1
    manager.publish_update(1)
    manager.maybe_checkpoint(epoch_step=0)
    assert monitor.runtime_history.snapshot(1)["updates"] == []
    with pytest.raises(RuntimeError, match="end of the native update"):
        monitor.complete_update()
    manager.finish_update()
    first = monitor.runtime_history.snapshot(1)["updates"][0]
    assert first["workload"]["effective_tokens"] == monitor.completed_tokens == 8
    assert not first["missing_phases"] and first["checkpointed"]
    assert first["reconfigurator_name"] == "OnlineReconfigurator"
    _update(manager)
    second = monitor.runtime_history.snapshot(2)["updates"][1]
    assert len(second["history"]["workloads"]) == 1
    assert second["observation"]["committed_tokens"] == 8
    assert second["committed_tokens"] == monitor.completed_tokens == 16
    assert not (manager.root / "reconfigurator-application").exists()


def test_default_monitor_rolls_back_unsaved_history_and_replay_counts_once(
    request,
):
    """Fault replay restores the same causal window and completed token count."""
    manager = request.getfixturevalue("native_lifecycle").manager
    _update(manager)
    manager.checkpoint_frequency.step_ctl.frequency_steps = 100
    _update(manager)
    _update(manager)
    assert manager.monitor.completed_tokens == 24
    manager.recover_uncertain(RuntimeError("lost unsaved updates"))
    assert manager.applied_version == 1 and manager.monitor.completed_tokens == 8
    assert [
        row["weights_version"]
        for row in manager.monitor.runtime_history.snapshot(3)["updates"]
    ] == [1]
    for version in (2, 3):
        manager.next_batch()
        manager.actor.updates += 1
        _complete_native_update(manager, version)
    rows = manager.monitor.runtime_history.snapshot(3)["updates"]
    assert [row["weights_version"] for row in rows] == [1, 2, 3]
    assert manager.monitor.completed_tokens == 24


def test_default_monitor_restart_preserves_history_order_and_ignores_old_pending(
    request,
):
    """Completed release snapshots supersede an earlier same-version sidecar."""
    fixture = request.getfixturevalue("native_lifecycle")
    manager = fixture.manager
    for _ in range(2):
        _update(manager)
        manager.save_checkpoint(reason="reconfiguration")
        manager._release()
    expected = manager.monitor.history.snapshot()
    pending = json.loads(manager.monitor.pending_path.read_text())
    state = json.loads((manager.latest_path / "monitor-state.json").read_text())
    assert pending["weights_version"] == state["weights_version"] == 2
    assert state["completed"] and pending["pending"] != state["pending"]
    manager.close()
    resumed = fixture.create()
    try:
        assert resumed.monitor.completed_tokens == 16
        assert resumed.monitor.history.snapshot() == expected
    finally:
        resumed.close()


def test_monitor_missing_measurement_state_preserves_training_progress(
    request, monkeypatch
):
    """A valid full checkpoint restores without inventing a measured history."""
    from areal.autopilot.domains.elastic.native import RevisionAdapter

    fixture = request.getfixturevalue("native_lifecycle")
    with monkeypatch.context() as patch:
        patch.setattr(fixture.manager.monitor, "checkpoint_state", lambda path: None)
        _update(fixture.manager)
        fixture.manager.close()
    restored = fixture.create(RevisionAdapter())
    try:
        assert restored.applied_version == restored.checkpoint_version == 1
        assert restored.data_progress == {"samples": 4, "effective_tokens": 8}
        assert restored.monitor.completed_tokens == 8
        assert restored.monitor.metrics_available
        assert restored.monitor.runtime_history.snapshot(1)["updates"] == []
        assert restored.monitor.history.snapshot().workloads == ()
        restored.ensure_actor()
        assert restored.actor.updates == 1
        _update(restored)
        row = restored.monitor.runtime_history.snapshot(2)["updates"][0]
        assert row["weights_version"] == 2 and row["committed_tokens"] == 16
        assert row["history"]["workloads"] == []
    finally:
        restored.close()


@pytest.mark.parametrize("failed_file", ["monitor-pending.json", "monitor-state.json"])
def test_optional_monitor_write_failure_does_not_rollback_or_block_release(
    request, monkeypatch, failed_file
):
    """Metric I/O is optional; the saved optimizer and data cursor remain exact."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from areal.elastic.infra.monitor import collector

    fixture = request.getfixturevalue("native_lifecycle")
    manager = fixture.manager
    original = collector.atomic_json

    def fail_optional(path, value):
        if path.name == failed_file:
            if failed_file == "monitor-state.json":
                assert manager.latest_path == path.parent
                assert manager.store.latest(manager.compatibility)[1][
                    "data_progress"
                ] == {
                    "samples": 4,
                    "effective_tokens": 8,
                }
            raise OSError("injected metric write failure")
        return original(path, value)

    with monkeypatch.context() as patch:
        patch.setattr(collector, "atomic_json", fail_optional)
        _update(manager)
        assert manager.applied_version == manager.published_version == 1
        assert manager.active is None and manager.actor.updates == 1
        assert not manager.monitor.metrics_available
        assert manager.monitor.completed_tokens == 8
        decide = MagicMock(
            side_effect=AssertionError("measured reconfigurator must pause")
        )
        patch.setattr(
            manager.reconfigurator_adapter,
            "reconfigurator",
            SimpleNamespace(decide=decide),
        )
        manager.save_checkpoint(reason="reconfiguration")
        manager._release()
        decide.assert_not_called()
        assert manager.actor is None
        assert manager.latest_manifest["data_progress"] == {
            "samples": 4,
            "effective_tokens": 8,
        }
        assert not (manager.latest_path / "monitor-state.json").exists()
        assert manager.consumer.cursor["applied_batches"] == 1
    manager.close()
    restored = fixture.create()
    try:
        assert restored.monitor.metrics_available
        assert restored.data_progress == {"samples": 4, "effective_tokens": 8}
        assert restored.monitor.runtime_history.snapshot(1)["updates"] == []
    finally:
        restored.close()


def test_legacy_checkpoint_rebuilds_counts_from_validated_queue(request):
    """Missing optional observations never turn prior trained tokens into zero."""
    from areal.utils.atomic import atomic_json

    fixture = request.getfixturevalue("native_lifecycle")
    manager = fixture.manager
    _update(manager)
    manager.close()
    path = manager.latest_path / "manifest.json"
    state = json.loads(path.read_text())
    state.pop("data_progress")
    atomic_json(path, state)
    (manager.latest_path / "monitor-state.json").unlink()
    restored = fixture.create()
    try:
        assert restored.data_progress == {"samples": 4, "effective_tokens": 8}
        assert restored.monitor.completed_tokens == 8
    finally:
        restored.close()


def test_mismatched_logical_progress_still_rejects_recovery(request):
    """Critical checkpoint progress remains strict even if metrics are optional."""
    from areal.utils.atomic import atomic_json

    fixture = request.getfixturevalue("native_lifecycle")
    manager = fixture.manager
    _update(manager)
    manager.close()
    path = manager.latest_path / "manifest.json"
    state = json.loads(path.read_text())
    state["data_progress"]["samples"] = 5
    atomic_json(path, state)
    with pytest.raises(ValueError, match="Logical data progress"):
        fixture.create()


def test_critical_checkpoint_failure_still_blocks_release_without_metrics(
    request, monkeypatch
):
    """Degrading observations never relaxes optimizer/cursor durability."""
    fixture = request.getfixturevalue("native_lifecycle")
    manager = fixture.manager
    _update(manager)
    manager.checkpoint_frequency.step_ctl.frequency_steps = 100
    manager.monitor.mark_unavailable(OSError("metrics unavailable"))
    _update(manager)
    actor, owner = manager.actor, manager.owner
    with monkeypatch.context() as patch:
        patch.setattr(
            manager.store,
            "commit",
            lambda **kwargs: (_ for _ in ()).throw(OSError("full-state save failed")),
        )
        with pytest.raises(OSError, match="full-state save failed"):
            manager.save_checkpoint(reason="reconfiguration")
    assert manager.actor is actor and manager.owner == owner
    assert manager.applied_version == manager.published_version == 2
    assert manager.checkpoint_version == 1
    assert manager.data_progress == {"samples": 8, "effective_tokens": 16}


def test_corrupt_monitor_snapshot_degrades_without_rejecting_model_recovery(request):
    """Malformed optional JSON cannot reject a valid model and queue checkpoint."""
    fixture = request.getfixturevalue("native_lifecycle")
    manager = fixture.manager
    _update(manager)
    manager.close()
    (manager.latest_path / "monitor-state.json").write_text("[null]")
    restored = fixture.create()
    try:
        assert restored.applied_version == 1 and restored.monitor.completed_tokens == 8
        assert not restored.monitor.metrics_available
        assert "Invalid Monitor checkpoint shape" in restored.monitor.last_error
        restored.ensure_actor()
        assert restored.actor.updates == 1
        _update(restored)
        assert restored.applied_version == restored.published_version == 2
        assert restored.monitor.completed_tokens == 16
    finally:
        restored.close()
