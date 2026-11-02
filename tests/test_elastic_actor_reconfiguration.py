# SPDX-License-Identifier: Apache-2.0

"""CPU coordinator acceptance; fake Actors do not establish tensor resharding."""

import json
from dataclasses import replace

import pytest

from tests.test_elastic_checkpoint import (
    _complete_native_update,
    _ready_native_episodes,
    native_lifecycle,  # noqa: F401 -- shared protocol fixture
)

from areal.elastic.infra.reconfiguration import (
    ExecutionLayout,
    poll_request,
    submit_request,
)

pytestmark = pytest.mark.usefixtures("native_lifecycle")


def update(manager, first):
    _ready_native_episodes(manager, first, manager.applied_version)
    manager.next_batch()
    manager.actor.updates += 1
    _complete_native_update(manager, manager.applied_version + 1)


@pytest.mark.parametrize("suspend_before_rollout", [False, True])
@pytest.mark.parametrize("native_lifecycle", [(2, 4)], indirect=True)
def test_one_journal_plans_only_changed_engines_and_continues_training(
    request, suspend_before_rollout
):
    """Actor-only, Rollout-only, joint and unchanged layouts use the same executor."""
    from unittest.mock import MagicMock

    fixture = request.getfixturevalue("native_lifecycle")
    manager, rollout, events = fixture.manager, fixture.rollout, fixture.events
    manager._actor_pool_setter = MagicMock(
        side_effect=lambda devices: events.append("placement")
    )
    rollout.resource_reconfiguration_enabled = True
    rollout.drain_model_calls = MagicMock(side_effect=lambda: events.append("drain"))
    rollout.open_model_calls = MagicMock(side_effect=lambda: events.append("open"))
    rollout.stop_inference_replicas = MagicMock(
        side_effect=lambda: events.append("stop-rollout")
    )
    rollout.rebuild_inference_replicas = MagicMock(
        side_effect=lambda devices, **kwargs: events.append("start-rollout")
    )
    update(manager, 0)

    target = ExecutionLayout(2, (0, 1), (4,))
    actor = manager.actor
    journal = manager.reconfiguration_application.root / "state.json"
    before, history = journal.read_bytes(), list(events)
    plan = manager.plan_reconfiguration(target)
    assert (
        journal.read_bytes() == before and events == history and manager.actor is actor
    )
    first = manager.reconfigure(
        target=target,
        request_id="actor-only",
        expected_generation=0,
        expected_weights_version=1,
    )
    assert manager.actor is not actor and manager.actor.updates == 1
    assert first["plan"]["steps"] == list(plan.steps)
    assert not plan.rebuilds_rollout
    rollout.stop_inference_replicas.assert_not_called()
    rollout.drain_model_calls.assert_not_called()
    assert not (manager.root / "actor-reconfiguration").exists()
    update(manager, 4)

    if suspend_before_rollout:
        manager.save_checkpoint(reason="release")
        manager._release()
    actor, owner = manager.actor, manager.owner
    target = ExecutionLayout(2, (0, 1), (2, 3, 4))
    events.clear()
    cursor = manager.consumer.cursor.copy()
    second = manager.reconfigure(
        target=target,
        request_id="rollout-only",
        expected_generation=1,
        expected_weights_version=2,
    )
    assert manager.consumer.cursor == cursor and manager.actor.updates == 2
    assert second["generation"] == 2
    if actor is not None:
        assert manager.actor is actor and manager.owner == owner
        actor.disconnect_engine.assert_called_once_with()
        actor.destroy.assert_not_called()
        assert events.index("disconnect-weights") < events.index("stop-rollout")
    else:
        assert "restore_actor" in second["plan"]["steps"]
    assert (
        events.index("stop-rollout")
        < events.index("start-rollout")
        < events.index("publish-2")
        < events.index("open")
    )
    update(manager, 8)

    events.clear()
    target = ExecutionLayout(4, (0, 1, 2, 3), (4,))
    third = manager.reconfigure(
        target=target,
        request_id="both",
        expected_generation=2,
        expected_weights_version=3,
    )
    assert third["generation"] == 3 and manager.actor.updates == 3
    assert (
        events.index("release")
        < events.index("stop-rollout")
        < events.index("placement")
        < events.index("start-rollout")
    )
    assert manager.actor.restore_plan["source_world_size"] == 2
    assert manager.actor.restore_plan["target_world_size"] == 4
    actor, history = manager.actor, list(events)
    assert (
        manager.reconfigure(
            target=target,
            request_id="both",
            expected_generation=2,
            expected_weights_version=3,
        )
        == third
    )
    unchanged = manager.reconfigure(
        target=target,
        request_id="unchanged",
        expected_generation=3,
        expected_weights_version=3,
    )
    assert unchanged["status"] == "unchanged" and unchanged["plan"]["steps"] == []
    assert manager.actor is actor and events == history
    update(manager, 12)
    assert manager.data_progress["samples"] == 16


@pytest.mark.parametrize("native_lifecycle", [(2, 4)], indirect=True)
def test_dp_journal_import_preserves_generation_receipts_and_source_file(request):
    """A completed DP-only run resumes through the single full-layout journal."""
    from areal.utils.atomic import atomic_json, json_digest

    fixture = request.getfixturevalue("native_lifecycle")
    manager = fixture.manager
    update(manager, 0)
    manager.reconfigure(
        target=replace(manager.execution_layout, world_size=4),
        request_id="grow",
        expected_generation=0,
        expected_weights_version=1,
    )
    state = manager.reconfiguration_application.state()
    manager._release()
    manager._checkpoint_dirty = False
    manager.close()
    state["config"] = {"world_size": 4}
    state["context_id"] = json_digest(
        {
            "scope": "native-fsdp-dp-v1",
            "run": manager.identity,
            "queue": manager.inbox.queue.queue_id,
            "actor_gpu_pool": list(manager.options.actor_devices),
            "rollout_gpu_pool": list(manager.options.rollout_devices),
        }
    )
    for receipt in state["receipts"].values():
        receipt.pop("plan", None)
        receipt["request"]["target"] = {
            "world_size": receipt["request"]["target"]["world_size"]
        }
        receipt["actual"]["config"] = {
            "world_size": receipt["actual"]["config"]["world_size"]
        }
    legacy = manager.root / "actor-reconfiguration/state.json"
    legacy.parent.mkdir()
    atomic_json(legacy, state)
    original = legacy.read_bytes()
    (manager.reconfiguration_application.root / "state.json").unlink()
    restored = fixture.create()
    try:
        assert restored.actor_world_size == 4
        assert restored.reconfiguration_application.state()["generation"] == 1
        restored.ensure_actor()
        restored.reconfigure(
            target=replace(restored.execution_layout, world_size=2),
            request_id="shrink",
            expected_generation=1,
            expected_weights_version=1,
        )
        assert restored.reconfiguration_application.state()["generation"] == 2
        assert legacy.read_bytes() == original
    finally:
        restored.close()


@pytest.mark.parametrize("native_lifecycle", [(2, 4)], indirect=True)
def test_actor_dp_roundtrip_preserves_cursor_and_continues_updates(request):
    """2→4→2 changes GPU generation only, retaining the same run and inbox."""
    fixture = request.getfixturevalue("native_lifecycle")
    manager = fixture.manager
    identity, compatibility = manager.identity, manager.compatibility.copy()
    update(manager, 0)
    cursor = manager.consumer.cursor.copy()
    serving = fixture.rollout
    old_measurement = manager.reconfigurator_adapter.execution_id
    first = manager.reconfigure(
        target=replace(manager.execution_layout, world_size=4),
        request_id="grow",
        expected_generation=0,
        expected_weights_version=1,
    )
    assert first["status"] == "applied" and first["generation"] == 1
    assert manager.consumer.cursor == cursor
    assert (
        manager.actor.updates
        == manager.applied_version
        == manager.published_version
        == 1
    )
    assert manager.actor.config.backend == "fsdp:d4"
    assert manager.config.actor.backend == "fsdp:d2"
    assert manager.allocation.parallel.world_size == 4
    assert manager.reconfigurator_adapter.profile["native_actor_world_size"] == 4
    assert manager.reconfigurator_adapter.execution_id != old_measurement
    assert manager.rollout is serving
    assert fixture.events.index("release") < fixture.events.index("load-1")
    assert (
        manager.reconfigure(
            target=replace(manager.execution_layout, world_size=4),
            request_id="grow",
            expected_generation=0,
            expected_weights_version=1,
        )
        == first
    )
    update(manager, 4)
    second = manager.reconfigure(
        target=replace(manager.execution_layout, world_size=2),
        request_id="shrink",
        expected_generation=1,
        expected_weights_version=2,
    )
    assert second["status"] == "applied" and second["generation"] == 2
    assert manager.actor.restore_plan["source_world_size"] == 4
    assert manager.actor.updates == 2
    update(manager, 8)
    assert manager.identity == identity and manager.compatibility == compatibility
    assert (
        manager.applied_version == 3 and manager.consumer.cursor["applied_batches"] == 3
    )
    assert manager.data_progress == {"samples": 12, "effective_tokens": 24}
    manager.close()
    restored = fixture.create()
    try:
        assert restored.actor_world_size == 2
        restored.ensure_actor()
        assert restored.actor.updates == 3 and restored.actor.restore_plan is None
    finally:
        restored.close()


@pytest.mark.parametrize("native_lifecycle", [(2, 4)], indirect=True)
def test_committed_target_restarts_while_checkpoint_still_has_source_layout(
    request,
):
    """A clean restart can retain the source checkpoint until the next target save."""
    fixture = request.getfixturevalue("native_lifecycle")
    manager = fixture.manager
    update(manager, 0)
    manager.reconfigure(
        target=replace(manager.execution_layout, world_size=4),
        request_id="grow",
        expected_generation=0,
        expected_weights_version=1,
    )
    assert manager.latest_manifest["world_size"] == 2
    # Release without creating a target checkpoint: the durable config receipt
    # and source DCP together must be sufficient for the next coordinator.
    manager._release()
    manager._checkpoint_dirty = False
    manager.close()
    restored = fixture.create()
    try:
        assert restored.actor_world_size == 4
        restored.ensure_actor()
        assert restored.actor.restore_plan["source_world_size"] == 2
        assert restored.actor.updates == 1
        update(restored, 4)
        assert restored.latest_manifest["world_size"] == 4
        assert (
            json.loads((restored.latest_path / "actor-layout.json").read_text())[
                "backend"
            ]
            == "fsdp:d4"
        )
        restored.close()
        again = fixture.create()
        try:
            assert again.actor_world_size == 4
            again.ensure_actor()
            assert again.actor.restore_plan is None and again.actor.updates == 2
        finally:
            again.close()
    finally:
        restored.close()


@pytest.mark.parametrize("native_lifecycle", [(2, 4)], indirect=True)
def test_manual_mailbox_changes_dp_at_completed_update(request):
    """A request file yields a durable receipt and leaves serving usable."""
    manager = request.getfixturevalue("native_lifecycle").manager
    update(manager, 0)
    path = submit_request(
        manager.root,
        target=replace(manager.execution_layout, world_size=4),
        request_id="manual-grow",
        expected_generation=0,
        expected_weights_version=1,
    )
    poll_request(manager)
    assert not path.exists()
    response = json.loads(path.with_name("response.json").read_text())
    assert response["receipt"]["status"] == "applied"
    assert manager.actor_world_size == 4 and manager.control_snapshot().boundary_ready
    update(manager, 4)
    assert manager.applied_version == 2


@pytest.mark.parametrize("native_lifecycle", [(2, 4)], indirect=True)
def test_execution_layout_roundtrip_preserves_cpu_service_and_progress(request):
    from unittest.mock import MagicMock

    from areal.elastic.infra.reconfiguration import (
        ExecutionLayout,
        poll_request,
        submit_request,
    )

    fixture = request.getfixturevalue("native_lifecycle")
    manager = fixture.manager
    events, rollout = fixture.events, fixture.rollout
    rollout.resource_reconfiguration_enabled = True
    manager._actor_pool_setter = setter = MagicMock(
        side_effect=lambda devices: events.append(f"pool-{devices}")
    )
    rollout.arm_reconfiguration_observation = MagicMock()
    rollout.reconfiguration_metrics = MagicMock(
        return_value=[{"proxy": "fixture", "marker": None, "first_call": None}]
    )
    rollout.drain_model_calls = MagicMock(side_effect=lambda: events.append("drain"))
    rollout.open_model_calls = MagicMock(side_effect=lambda: events.append("open"))
    rollout.stop_inference_replicas = MagicMock(
        side_effect=lambda: events.append("stop-rollout")
    )
    rollout.rebuild_inference_replicas = MagicMock(
        side_effect=lambda devices, **kwargs: events.append(f"rollout-{devices}")
    )
    update(manager, 0)
    original_identity = manager.identity
    source = manager.execution_layout
    cursor = manager.consumer.cursor.copy()
    # Unchanged and changed layouts share one request journal.
    manager.reconfigure(
        target=replace(manager.execution_layout, world_size=2),
        request_id="legacy",
        expected_generation=0,
        expected_weights_version=1,
    )
    unchanged = manager.reconfiguration_application.state()["receipts"]["legacy"]
    assert unchanged["status"] == "unchanged"
    events.clear()
    target = ExecutionLayout(1, (0,), (1, 2, 3, 4))
    submit_request(
        manager.root,
        target=target,
        request_id="more-rollout",
        expected_generation=0,
        expected_weights_version=1,
    )
    poll_request(manager)
    first = json.loads(
        (manager.root / "resource-reconfiguration/response.json").read_text()
    )["receipt"]
    assert first["status"] == "applied" and first["generation"] == 1
    assert manager.execution_layout == target
    assert manager.actor_world_size == 1 and manager.actor.updates == 1
    assert manager.consumer.cursor == cursor and manager.rollout is rollout
    assert (
        events.index("release")
        < events.index("stop-rollout")
        < events.index("pool-[0]")
    )
    assert (
        events.index("rollout-(1, 2, 3, 4)")
        < events.index("publish-1")
        < events.index("open")
    )
    assert (
        manager.reconfigure(
            target=target,
            request_id="more-rollout",
            expected_generation=0,
            expected_weights_version=1,
        )
        == first
    )
    update(manager, 4)
    timing = [
        json.loads(line)
        for line in (manager.root / "reconfiguration-timing.jsonl")
        .read_text()
        .splitlines()
    ]
    phases = {row["phase"] for row in timing if row["event"] == "phase_end"}
    assert {
        "checkpoint",
        "drain",
        "actor_release",
        "rollout_stop",
        "placement",
        "actor_start",
        "actor_restore",
        "awex_connect",
        "publish",
        "application_and_commit",
        "admission_open",
    } <= phases
    assert all(row["seconds"] >= 0 for row in timing if row["event"] == "phase_end")
    assert any(row["event"] == "first_inference_snapshot" for row in timing)
    assert rollout.arm_reconfiguration_observation.call_count == 1
    placement = json.loads((manager.latest_path / "actor-layout.json").read_text())
    assert placement["format_version"] == 2
    assert placement["reservation"] == [0, 1, 2, 3, 4]
    assert placement["gpu_pool"] == [0] and placement["rollout_devices"] == [1, 2, 3, 4]
    assert manager.reconfigurator_adapter.placements.current().rollout_replicas == 4
    assert (
        manager.reconfigurator_adapter.placements.current()
        in manager.reconfigurator_adapter.reconfigurator.candidates
    )
    assert (
        manager.reconfiguration_application.state()["receipts"]["legacy"] == unchanged
    )
    # Restart selects the authoritative target lease BEFORE the first Actor.
    manager.close()
    restored = fixture.create(actor_pool_setter=setter)
    try:
        restored.ensure_actor()
        assert restored.execution_layout == target and restored.actor.updates == 2
        restored.reconfigure(
            target=source,
            request_id="more-actor",
            expected_generation=1,
            expected_weights_version=2,
        )
        assert restored.execution_layout == source
        assert restored.actor.restore_plan["target_world_size"] == 2
        assert restored.identity == original_identity
        assert restored.consumer.cursor["applied_batches"] == 2
        update(restored, 8)
        assert restored.actor.updates == 3
    finally:
        restored.close()


@pytest.mark.parametrize("native_lifecycle", [(2, 4)], indirect=True)
def test_resource_device_movement_same_dp_uses_rank_state(request):
    from unittest.mock import MagicMock

    from areal.elastic.infra.reconfiguration import ExecutionLayout

    fixture = request.getfixturevalue("native_lifecycle")
    manager = fixture.manager
    fixture.rollout.resource_reconfiguration_enabled = True
    manager._actor_pool_setter = MagicMock()
    for method in (
        "drain_model_calls",
        "open_model_calls",
        "stop_inference_replicas",
        "rebuild_inference_replicas",
    ):
        setattr(fixture.rollout, method, MagicMock())
    update(manager, 0)
    manager.reconfigure(
        target=ExecutionLayout(2, (3, 4), (0, 1, 2)),
        request_id="move",
        expected_generation=0,
        expected_weights_version=1,
    )
    assert manager.actor.restore_plan is None
    assert manager.actor.updates == manager.applied_version == 1
    assert manager.gpu_uuids == ["GPU-3", "GPU-4"]
