"""Unit tests for async checkpoint manager and SaverConfig.

These tests do NOT require GPU or distributed environment.
They test:
1. SaverConfig mode validation
2. AsyncCheckpointManager state machine (SYNC and ASYNC modes)
3. Background consolidation (dedicated PG, no gating)
4. Full training loop lifecycle
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import pytest
from torch.distributed.checkpoint.state_dict_saver import AsyncSaveResponse

from areal.api.cli_args import SaverConfig
from areal.utils.async_checkpoint import AsyncCheckpointManager, AsyncMode

# Module-level patch target
_DIST = "areal.utils.async_checkpoint.dist"
_DCP = "areal.utils.async_checkpoint.dcp"


# =============================================================================
# SaverConfig validation
# =============================================================================


class TestSaverConfigMode:
    def test_default_is_auto(self):
        config = SaverConfig(experiment_name="test", trial_name="test", fileroot="/tmp")
        assert config.mode == "auto"

    @pytest.mark.parametrize("mode", ["auto", "sync", "async"])
    def test_valid_modes(self, mode):
        config = SaverConfig(
            experiment_name="test",
            trial_name="test",
            fileroot="/tmp",
            mode=mode,
        )
        assert config.mode == mode

    @pytest.mark.parametrize("mode", ["invalid", "", "ASYNC", "async_pinned_mem"])
    def test_invalid_modes_raise(self, mode):
        with pytest.raises(ValueError, match="Invalid mode"):
            SaverConfig(
                experiment_name="test",
                trial_name="test",
                fileroot="/tmp",
                mode=mode,
            )


# =============================================================================
# Helpers
# =============================================================================


def _make_completed_future(result=None):
    f = Future()
    f.set_result(result)
    return f


def _make_failed_future(exc=None):
    f = Future()
    f.set_exception(exc or RuntimeError("test error"))
    return f


def _make_async_save_response():
    """Create a real AsyncSaveResponse with completed futures."""
    return AsyncSaveResponse(
        staging_completion=_make_completed_future(),
        upload_completion=_make_completed_future(),
    )


def _make_mgr(mock_dist):
    """Create an ASYNC manager with mocked dist."""
    mock_dist.is_initialized.return_value = True
    mock_dist.new_group.side_effect = [
        MagicMock(name="save_pg"),
        MagicMock(name="consolidation_pg"),
    ]
    return AsyncCheckpointManager(AsyncMode.ASYNC)


# =============================================================================
# SYNC mode
# =============================================================================


class TestSyncMode:
    def test_sync_init(self):
        with patch(_DIST):
            mgr = AsyncCheckpointManager(AsyncMode.SYNC)
        assert mgr.is_async is False
        assert mgr._pg is None
        assert mgr._consolidation_pg is None
        assert mgr._executor is None

    @patch(_DCP)
    @patch(_DIST)
    def test_save_calls_sync_dcp(self, mock_dist, mock_dcp):
        mgr = AsyncCheckpointManager(AsyncMode.SYNC)
        writer = MagicMock()
        mgr.save({"model": {}}, storage_writer=writer, checkpoint_id="/tmp/ckpt")

        mock_dcp.save.assert_called_once_with(
            {"model": {}}, storage_writer=writer, checkpoint_id="/tmp/ckpt"
        )
        mock_dcp.async_save.assert_not_called()

    @patch(_DCP)
    @patch(_DIST)
    def test_save_with_post_fn_runs_inline(self, mock_dist, mock_dcp):
        """SYNC mode: post_fn runs inline, no deadlock."""
        mgr = AsyncCheckpointManager(AsyncMode.SYNC)
        callback = MagicMock()
        mgr.save({"model": {}}, storage_writer=MagicMock(), post_fn=callback)
        mock_dcp.save.assert_called_once()
        callback.assert_called_once()


# =============================================================================
# ASYNC mode
# =============================================================================


class TestAsyncModeCheckpoint:
    @patch(_DIST)
    def test_async_init(self, mock_dist):
        mgr = _make_mgr(mock_dist)
        assert mgr.is_async is True
        assert mock_dist.new_group.call_count == 2
        assert mgr._pg is not None
        assert mgr._consolidation_pg is not None
        assert mgr._executor is not None

    @patch(_DCP)
    @patch(_DIST)
    def test_save_uses_process_type_and_stager(self, mock_dist, mock_dcp):
        mgr = _make_mgr(mock_dist)
        response = _make_async_save_response()
        mock_dcp.async_save.return_value = response

        mgr.save({"model": {}}, storage_writer=MagicMock())

        mock_dcp.async_save.assert_called_once()
        mock_dcp.save.assert_not_called()
        assert mgr._stager is not None
        assert mgr._staging_future is response.staging_completion
        assert mgr._save_future is response.upload_completion

    @patch(_DIST)
    def test__wait_for_upload_swallows_error(self, mock_dist):
        """Upload errors are logged by bg thread; _wait_for_upload doesn't raise."""
        mgr = _make_mgr(mock_dist)
        mgr._save_future = _make_failed_future()
        mgr._wait_for_upload()  # should not raise
        assert mgr._save_future is None

    def test_save_without_pg_raises(self):
        with patch(_DIST) as mock_dist:
            mock_dist.is_initialized.return_value = False
            mgr = AsyncCheckpointManager(AsyncMode.ASYNC)
        with pytest.raises(RuntimeError, match="process group"):
            mgr.save({"model": {}})

    @patch(_DIST)
    def test_staging_wait(self, mock_dist):
        mgr = _make_mgr(mock_dist)
        mgr._staging_future = _make_completed_future()
        mgr.maybe_wait_for_staging()
        assert mgr._staging_future is None

    @patch(_DIST)
    def test_staging_error_handled(self, mock_dist):
        mgr = _make_mgr(mock_dist)
        mgr._staging_future = _make_failed_future()
        mgr.maybe_wait_for_staging()  # should not raise
        assert mgr._staging_future is None

    @patch(_DCP)
    @patch(_DIST)
    def test_finalize_cleans_up(self, mock_dist, mock_dcp):
        mock_dist.is_initialized.return_value = True
        save_pg = MagicMock(name="save_pg")
        consolidation_pg = MagicMock(name="consolidation_pg")
        mock_dist.new_group.side_effect = [save_pg, consolidation_pg]
        mgr = AsyncCheckpointManager(AsyncMode.ASYNC)

        mock_stager = MagicMock()
        mgr._stager = mock_stager

        mgr.finalize()

        mock_stager.close.assert_called_once()
        assert mgr._stager is None
        assert mgr._executor is None
        mock_dist.destroy_process_group.assert_any_call(save_pg)
        mock_dist.destroy_process_group.assert_any_call(consolidation_pg)
        assert mock_dist.destroy_process_group.call_count == 2
        assert mgr._pg is None
        assert mgr._consolidation_pg is None


# =============================================================================
# Background consolidation
# =============================================================================


class TestBackgroundConsolidation:
    @patch(_DIST)
    def test_post_fn_runs_in_bg_thread(self, mock_dist):
        """Verify callback runs in a different thread."""
        mgr = _make_mgr(mock_dist)
        called_in_bg = threading.Event()
        main_tid = threading.current_thread().ident

        def _callback():
            if threading.current_thread().ident != main_tid:
                called_in_bg.set()

        mgr._save_future = _make_completed_future()
        mgr._submit_post_fn(_callback)

        mgr._bg_future.result(timeout=2.0)
        assert called_in_bg.is_set()

    @patch(_DIST)
    def test_post_fn_runs_even_on_upload_failure(self, mock_dist):
        """Consolidation runs even when upload fails (has barrier, prevents deadlock)."""
        mgr = _make_mgr(mock_dist)
        callback = MagicMock()

        mgr._save_future = _make_failed_future()
        mgr._submit_post_fn(callback)

        mgr._bg_future.result(timeout=2.0)
        callback.assert_called_once()

    @patch(_DIST)
    def test_consolidation_failure_propagated(self, mock_dist):
        """_check_bg_error() raises when bg done+failed, clears future."""
        mgr = _make_mgr(mock_dist)

        def bad_callback():
            raise RuntimeError("consolidation error")

        mgr._save_future = _make_completed_future()
        mgr._submit_post_fn(bad_callback)

        # Wait for the bg job to actually complete
        try:
            mgr._bg_future.result(timeout=2.0)
        except RuntimeError:
            pass

        with pytest.raises(RuntimeError, match="consolidation error"):
            mgr._check_bg_error()
        assert mgr._bg_future is None

    @patch(_DIST)
    def test_consecutive_consolidations_ordered(self, mock_dist):
        """Multiple saves produce ordered consolidations via sequential executor."""
        mgr = _make_mgr(mock_dist)
        call_order: list[str] = []

        def _c1():
            call_order.append("c1")

        def _c2():
            call_order.append("c2")

        mgr._save_future = _make_completed_future()
        mgr._submit_post_fn(_c1)

        mgr._save_future = _make_completed_future()
        mgr._submit_post_fn(_c2)

        mgr.finalize()
        assert call_order == ["c1", "c2"]


# =============================================================================
# Training loop lifecycle
# =============================================================================


class TestTrainingLoop:
    @patch(_DCP)
    @patch(_DIST)
    def test_full_training_loop_cycle(self, mock_dist, mock_dcp):
        """Simulate: save -> staging_wait -> train -> save -> finalize."""
        mgr = _make_mgr(mock_dist)

        # Save 1
        mock_dcp.async_save.return_value = _make_async_save_response()
        callback1 = MagicMock()
        mgr.save({"step": 1}, storage_writer=MagicMock(), post_fn=callback1)

        # Training loop iteration 2:
        mgr.maybe_wait_for_staging()
        assert mgr._staging_future is None
        # Simulate training (consolidation runs during this)
        time.sleep(0.1)

        # Save 2
        mock_dcp.async_save.return_value = _make_async_save_response()
        callback2 = MagicMock()
        mgr.save({"step": 2}, storage_writer=MagicMock(), post_fn=callback2)

        # Finalize
        mgr.finalize()
        callback1.assert_called_once()
        callback2.assert_called_once()

    @patch(_DCP)
    @patch(_DIST)
    def test_upload_failure_does_not_block_next_save(self, mock_dist, mock_dcp):
        """Upload failure doesn't prevent next save."""
        mgr = _make_mgr(mock_dist)

        mgr._save_future = _make_failed_future()
        callback = MagicMock()
        mgr._submit_post_fn(callback)

        # Wait for bg job to finish
        mgr._bg_future.result(timeout=2.0)
        callback.assert_called_once()

        # Clear the done+failed bg future before next save
        mgr._check_bg_error()

        # Next save works
        mock_dcp.async_save.return_value = _make_async_save_response()
        mgr.save({"step": 2}, storage_writer=MagicMock())
        assert mgr._save_future is not None

        mgr.finalize()

    @patch(_DIST)
    def test_finalize_propagates_error(self, mock_dist):
        """finalize() must propagate errors from the final bg job."""
        mgr = _make_mgr(mock_dist)

        def bad_callback():
            raise RuntimeError("final consolidation error")

        mgr._save_future = _make_completed_future()
        mgr._submit_post_fn(bad_callback)

        with pytest.raises(RuntimeError, match="final consolidation error"):
            mgr.finalize()

        # Resources still cleaned up despite the error
        assert mgr._executor is None
        assert mgr._stager is None
        assert mgr._pg is None
        assert mgr._consolidation_pg is None

    @patch(_DIST)
    def test_save_catches_bg_error_completed_during_training(self, mock_dist):
        """Regression: bg job that fails during training is caught by next save().

        1. Submit bg job that fails after short delay
        2. Sleep to simulate training (bg finishes and fails during "training")
        3. Call save() -> expect raise from _check_bg_error() inside save()
        """
        mgr = _make_mgr(mock_dist)

        def delayed_failure():
            time.sleep(0.05)
            raise RuntimeError("bg error during training")

        mgr._save_future = _make_completed_future()
        mgr._submit_post_fn(delayed_failure)

        # Simulate training - bg job fails during this
        time.sleep(0.2)

        # Next save should catch the error via _check_bg_error
        with pytest.raises(RuntimeError, match="bg error during training"):
            mgr.save({"step": 2}, storage_writer=MagicMock())
        assert mgr._bg_future is None
