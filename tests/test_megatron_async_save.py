"""Focused tests for ordinary and managed Megatron asynchronous saves."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from areal.engine.megatron_utils import checkpointer as checkpointer_module


@pytest.fixture
def patched_checkpointer():
    queue = MagicMock()
    queue.get_num_unfinalized_calls.return_value = 0
    queue.maybe_finalize_async_calls.return_value = []
    queue.schedule_async_request.return_value = 0
    queue.call_idx = -1

    def gather_object(output, value, **_kwargs):
        output[0] = value

    def finalize_managed(async_queue, _group, *, blocking, recovery_token, **_kwargs):
        try:
            result = async_queue.maybe_finalize_async_calls(blocking=blocking)
        except BaseException:
            recovery_token.mark_cleared()
            raise
        if result:
            recovery_token.mark_cleared()
        return result

    with (
        patch("torch.distributed.get_rank", return_value=0),
        patch("torch.distributed.get_world_size", return_value=1),
        patch("torch.distributed.get_backend", return_value="gloo"),
        patch("torch.distributed.get_process_group_ranks", return_value=[0]),
        patch("torch.distributed.broadcast_object_list"),
        patch("torch.distributed.all_gather_object", side_effect=gather_object),
        patch.object(checkpointer_module, "AsyncCallsQueue", return_value=queue),
        patch.object(
            checkpointer_module,
            "finalize_managed_async_calls",
            side_effect=finalize_managed,
        ),
        patch.object(checkpointer_module, "preflight_managed_async_finalize"),
    ):
        manager = checkpointer_module.MegatronCheckpointManager(
            model=MagicMock(),
            optimizer=MagicMock(),
            lr_scheduler=None,
            async_save=True,
        )
        yield manager, queue


def test_async_disabled_has_no_queue() -> None:
    manager = checkpointer_module.MegatronCheckpointManager(
        model=MagicMock(),
        optimizer=MagicMock(),
        lr_scheduler=None,
        async_save=False,
    )

    assert manager._async_queue is None
    manager.close()


def test_ordinary_async_save_schedules_and_reaps(
    patched_checkpointer, tmp_path
) -> None:
    manager, queue = patched_checkpointer
    request = object()
    with (
        patch.object(manager, "generate_state_dict", return_value={"model": {}}),
        patch.object(
            checkpointer_module,
            "save_dist_checkpointing",
            return_value=request,
        ),
        patch("torch.cuda.empty_cache"),
        patch("torch.distributed.barrier"),
    ):
        manager.save_checkpoint(str(tmp_path / "step0"))

    queue.maybe_finalize_async_calls.assert_called_once_with(blocking=False)
    queue.schedule_async_request.assert_called_once_with(request)


def _configure_managed_manager(manager, control_group):
    manager.managed_checkpoint_enabled = True
    manager.checkpoint_process_group = control_group
    manager._retry_managed_checkpoint_cleanup = MagicMock()
    manager._managed_optimizer_identities = MagicMock(
        return_value={(): {"leaf": "managed"}}
    )
    manager._vote_managed_phase = MagicMock(return_value=None)
    manager._require_managed_checkpoint_group = MagicMock(return_value=control_group)

    def run_phase(_phase, operation, _transaction):
        return operation(), None

    manager._run_managed_phase = run_phase


def test_managed_async_save_releases_fence_after_finalize(
    patched_checkpointer, tmp_path
) -> None:
    manager, queue = patched_checkpointer
    checkpoint_path = tmp_path / "managed-step0"
    request = object()
    leaf = MagicMock()
    released = MagicMock()
    _configure_managed_manager(manager, object())
    queue.get_num_unfinalized_calls.return_value = 1
    queue.maybe_finalize_async_calls.side_effect = [[], [], [7], []]
    queue.schedule_async_request.return_value = 7
    queue.call_idx = 6

    with (
        patch.object(manager, "generate_state_dict", return_value={"optimizer": {}}),
        patch.object(
            checkpointer_module,
            "save_dist_checkpointing",
            return_value=request,
        ),
        patch.object(
            checkpointer_module,
            "begin_managed_async_checkpoint_save",
            return_value=(leaf,),
        ),
        patch.object(
            checkpointer_module, "bind_managed_async_checkpoint_request"
        ) as bind,
        patch.object(
            checkpointer_module, "complete_managed_async_checkpoint_save"
        ) as complete,
    ):
        manager.save_checkpoint(
            str(checkpoint_path), async_completion_callback=released
        )
        released.assert_not_called()
        bind.assert_called_once_with((leaf,), request, 7)

        (checkpoint_path / "metadata.json").write_text("{}")
        manager.wait_async_saves()
        manager.wait_async_saves()

    complete.assert_called_once_with((leaf,))
    released.assert_called_once_with()
    assert manager._managed_async_save is None
    assert (checkpoint_path / checkpointer_module._MANAGED_ASYNC_COMPLETE).is_file()


def test_managed_async_background_failure_remains_incomplete(
    patched_checkpointer, tmp_path
) -> None:
    manager, queue = patched_checkpointer
    checkpoint_path = tmp_path / "managed-failed"
    leaf = MagicMock()
    released = MagicMock()
    _configure_managed_manager(manager, object())
    queue.maybe_finalize_async_calls.side_effect = [[], RuntimeError("disk exploded")]
    queue.schedule_async_request.return_value = 9
    queue.call_idx = 8
    queue.get_num_unfinalized_calls.return_value = 1

    with (
        patch.object(manager, "generate_state_dict", return_value={"optimizer": {}}),
        patch.object(
            checkpointer_module,
            "save_dist_checkpointing",
            return_value=object(),
        ),
        patch.object(
            checkpointer_module,
            "begin_managed_async_checkpoint_save",
            return_value=(leaf,),
        ),
        patch.object(checkpointer_module, "bind_managed_async_checkpoint_request"),
        patch.object(checkpointer_module, "fail_managed_async_checkpoint_save") as fail,
    ):
        manager.save_checkpoint(
            str(checkpoint_path), async_completion_callback=released
        )
        with pytest.raises(RuntimeError, match="disk exploded"):
            manager.wait_async_saves()

    fail.assert_called_once()
    released.assert_called_once_with()
    assert manager.managed_async_save_state == "FAILED"
    assert (checkpoint_path / checkpointer_module._MANAGED_ASYNC_INCOMPLETE).is_file()
    assert not (checkpoint_path / checkpointer_module._MANAGED_ASYNC_COMPLETE).exists()
