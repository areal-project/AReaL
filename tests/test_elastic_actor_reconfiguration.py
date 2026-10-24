# SPDX-License-Identifier: Apache-2.0

"""CPU coordinator acceptance; fake Actors do not establish tensor resharding."""

import json

import pytest

from tests.test_elastic_checkpoint import (
    _complete_native_update,
    _ready_native_episodes,
    native_lifecycle,  # noqa: F401 -- shared protocol fixture
)

from areal.elastic.control.actor_reconfiguration import poll_request, submit_request

pytestmark = pytest.mark.usefixtures("native_lifecycle")


def update(manager, first):
    _ready_native_episodes(manager, first, manager.applied_version)
    manager.next_batch()
    manager.actor.updates += 1
    _complete_native_update(manager, manager.applied_version + 1)


@pytest.mark.parametrize("native_lifecycle", [(2, 4)], indirect=True)
def test_actor_dp_roundtrip_preserves_cursor_and_continues_updates(request):
    """2→4→2 changes GPU generation only, retaining the same run and inbox."""
    fixture = request.getfixturevalue("native_lifecycle")
    manager = fixture.manager
    identity, compatibility = manager.identity, manager.compatibility.copy()
    update(manager, 0)
    cursor = manager.consumer.cursor.copy()
    serving = fixture.rollout
    old_measurement = manager.policy_adapter.execution_id
    first = manager.reconfigure_actor(
        world_size=4,
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
    assert manager.policy_adapter.profile["native_actor_world_size"] == 4
    assert manager.policy_adapter.execution_id != old_measurement
    assert manager.rollout is serving
    assert fixture.events.index("release") < fixture.events.index("load-1")
    assert (
        manager.reconfigure_actor(
            world_size=4,
            request_id="grow",
            expected_generation=0,
            expected_weights_version=1,
        )
        == first
    )
    update(manager, 4)
    second = manager.reconfigure_actor(
        world_size=2,
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
    manager.reconfigure_actor(
        world_size=4,
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
        world_size=4,
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
