# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from areal.v2.training_service.controller.controller import GatewayTrainController


def _controller(
    events,
    *,
    fail_onload_tag=None,
    fail_continue=False,
    fail_actor_release=False,
):
    controller = object.__new__(GatewayTrainController)
    controller._colocate = True

    def onload(tags):
        tag = tags[0]
        events.append(("onload", tag))
        if tag == fail_onload_tag:
            raise RuntimeError(f"cannot restore {tag}")

    def continue_generation():
        events.append("continue")
        if fail_continue:
            raise RuntimeError("cannot continue")

    controller.rollout = SimpleNamespace(
        pause=lambda: events.append("pause"),
        pause_generation_sync=lambda: events.append("drain"),
        offload=lambda tags: events.append(("offload", tags[0])),
        onload=onload,
        continue_generation=continue_generation,
        resume=lambda: events.append("resume"),
        pause_generation=lambda: events.append("pause_generation"),
    )

    def actor_memory_op(op, tags):
        events.append(("actor", op, tuple(tags)))
        if fail_actor_release and op == "release_memory":
            raise RuntimeError("cannot release actor")

    controller._broadcast_awex_memory_op = actor_memory_op
    return controller


def test_failure_before_weight_update_restores_all_rollout_memory():
    events = []
    controller = _controller(events)

    with pytest.raises(RuntimeError, match="training failed"):
        with controller.train_phase():
            raise RuntimeError("training failed")

    assert ("onload", "weights") in events
    assert ("onload", "cuda_graph") in events
    assert ("onload", "kv_cache") in events
    assert events[-2:] == ["continue", "resume"]


def test_exit_attempts_remaining_cleanup_after_one_restore_fails():
    events = []
    controller = _controller(events, fail_onload_tag="cuda_graph")
    controller.enter_train_phase()
    controller._colocate_update_started = True
    controller._colocate_update_succeeded = True

    with pytest.raises(ExceptionGroup, match="cleanup"):
        controller.exit_train_phase()

    assert ("onload", "cuda_graph") in events
    assert ("onload", "kv_cache") in events
    assert "continue" not in events
    assert "resume" not in events


def test_resume_is_gated_on_generation_continue_success():
    events = []
    controller = _controller(events, fail_continue=True)
    controller.enter_train_phase()
    controller._colocate_update_started = True
    controller._colocate_transfer_succeeded = True
    controller._colocate_update_succeeded = True

    with pytest.raises(RuntimeError, match="cannot continue"):
        controller.exit_train_phase()

    assert "continue" in events
    assert "resume" not in events


def test_inference_restore_is_gated_on_actor_release_success():
    events = []
    controller = _controller(events, fail_actor_release=True)
    controller.enter_train_phase()
    controller._colocate_update_started = True
    controller._colocate_transfer_succeeded = True
    controller._colocate_update_succeeded = True

    with pytest.raises(ExceptionGroup, match="cleanup"):
        controller.exit_train_phase()

    assert ("actor", "release_memory", ("weights", "optimizer")) in events
    assert not any(event[0] == "onload" for event in events if isinstance(event, tuple))
    assert "continue" not in events
    assert "resume" not in events


def test_failed_weight_update_keeps_generation_closed():
    events = []
    controller = _controller(events)
    controller._weight_update_ctrl = SimpleNamespace(
        update_weights=lambda version: (_ for _ in ()).throw(
            RuntimeError(f"update {version} failed")
        )
    )

    with pytest.raises(RuntimeError, match="update 1 failed"):
        with controller.train_phase():
            controller.update_weights(SimpleNamespace(version=1))

    assert controller._colocate_update_started
    assert not controller._colocate_transfer_succeeded
    assert not controller._colocate_update_succeeded
    assert "continue" not in events
    assert "resume" not in events


def test_error_weight_update_result_keeps_generation_closed():
    events = []
    controller = _controller(events)
    controller._weight_update_ctrl = SimpleNamespace(
        update_weights=lambda version: SimpleNamespace(
            status="error", error=f"update {version} failed"
        )
    )

    with pytest.raises(RuntimeError, match="update 1 failed"):
        with controller.train_phase():
            controller.update_weights(SimpleNamespace(version=1))

    assert controller._colocate_update_started
    assert not controller._colocate_transfer_succeeded
    assert not controller._colocate_update_succeeded
    assert "continue" not in events
    assert "resume" not in events


def test_transfer_without_version_commit_keeps_generation_closed():
    events = []
    controller = _controller(events)
    controller._weight_update_ctrl = SimpleNamespace(
        update_weights=lambda version: SimpleNamespace(
            status="ok", version=version, duration_ms=1.0
        )
    )

    with pytest.raises(RuntimeError, match="version commit failed"):
        with controller.train_phase():
            controller.update_weights(SimpleNamespace(version=1))
            raise RuntimeError("version commit failed")

    assert controller._colocate_update_started
    assert controller._colocate_transfer_succeeded
    assert not controller._colocate_update_succeeded
    assert "continue" not in events
    assert "resume" not in events


def test_mark_committed_allows_generation_to_resume():
    events = []
    controller = _controller(events)
    controller._weight_update_ctrl = SimpleNamespace(
        update_weights=lambda version: SimpleNamespace(
            status="ok", version=version, duration_ms=1.0
        )
    )

    with controller.train_phase():
        controller.update_weights(SimpleNamespace(version=1))
        controller.mark_colocate_update_committed()

    assert controller._colocate_update_succeeded
    assert controller._colocate_transfer_succeeded
    assert "continue" in events
    assert "resume" in events
