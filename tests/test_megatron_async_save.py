"""Unit tests for the Megatron checkpoint manager's async-save state machine.

These tests do NOT exercise real Megatron dist_checkpointing or distributed
process groups. They patch the queue and `save_dist_checkpointing` to verify
that the manager correctly:

- skips queue creation when async_save is False
- routes the AsyncRequest to AsyncCallsQueue.schedule_async_request when True
- finalizes completed saves non-blockingly on each new save call
- blocks on load_checkpoint / close
- treats close as idempotent
- emits the async-only metric (queue_depth on schedule)
- emits no async metrics on the sync path
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from unittest.mock import MagicMock, patch

import pytest


def _import_checkpointer():
    """Import the checkpointer module without triggering the full areal package."""
    if "areal.engine.megatron_utils.checkpointer" in sys.modules:
        return sys.modules["areal.engine.megatron_utils.checkpointer"]

    # Stub out heavy/optional dependencies so the module loads on a CPU box
    # without Megatron or a Stager-capable torch build.
    for path in (
        "megatron",
        "megatron.core",
        "megatron.core.dist_checkpointing",
        "megatron.core.dist_checkpointing.mapping",
        "megatron.core.dist_checkpointing.serialization",
        "megatron.core.dist_checkpointing.strategies",
        "megatron.core.dist_checkpointing.strategies.async_utils",
        "megatron.core.dist_checkpointing.strategies.fully_parallel",
        "areal",
        "areal.engine",
        "areal.engine.megatron_utils",
        "areal.infra",
        "areal.infra.platforms",
        "areal.utils",
        "areal.utils.logging",
    ):
        sys.modules.setdefault(path, types.ModuleType(path))

    sys.modules["megatron.core"].dist_checkpointing = sys.modules[
        "megatron.core.dist_checkpointing"
    ]
    sys.modules["megatron.core"].mpu = MagicMock()
    sys.modules["megatron.core"].tensor_parallel = MagicMock()
    sys.modules["megatron.core.dist_checkpointing.mapping"].ShardedObject = MagicMock()
    sys.modules[
        "megatron.core.dist_checkpointing.serialization"
    ].get_default_load_sharded_strategy = MagicMock()
    sys.modules[
        "megatron.core.dist_checkpointing.serialization"
    ].get_default_save_sharded_strategy = MagicMock()
    async_utils = sys.modules["megatron.core.dist_checkpointing.strategies.async_utils"]
    async_utils.AsyncCallsQueue = MagicMock
    async_utils.AsyncRequest = MagicMock
    fp_mod = sys.modules["megatron.core.dist_checkpointing.strategies.fully_parallel"]
    fp_mod.FullyParallelLoadStrategyWrapper = MagicMock
    fp_mod.FullyParallelSaveStrategyWrapper = MagicMock
    sys.modules["areal.infra.platforms"].current_platform = MagicMock(
        device_type="cuda", is_available=lambda: False
    )
    sys.modules["areal.utils.logging"].getLogger = lambda *_a, **_k: MagicMock()

    # stats_tracker.scalar is called by the manager to report latency.
    stats_mod = types.ModuleType("areal.utils.stats_tracker")
    stats_mod.scalar = MagicMock()
    sys.modules["areal.utils.stats_tracker"] = stats_mod

    # The checkpointer imports `from areal.utils import logging, stats_tracker`.
    # Make sure the parent `areal.utils` package exposes both as attributes.
    sys.modules["areal.utils"].logging = sys.modules["areal.utils.logging"]
    sys.modules["areal.utils"].stats_tracker = stats_mod

    # Load the real checkpointer module from disk under the stubbed parents.
    import pathlib

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "areal.engine.megatron_utils.checkpointer",
        repo_root / "areal" / "engine" / "megatron_utils" / "checkpointer.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["areal.engine.megatron_utils.checkpointer"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def patched_checkpointer():
    mod = _import_checkpointer()

    queue = MagicMock()
    queue.get_num_unfinalized_calls.return_value = 0
    queue.maybe_finalize_async_calls.return_value = []
    queue.schedule_async_request.return_value = 0

    with (
        patch("torch.distributed.get_rank", return_value=0),
        patch.object(mod, "AsyncCallsQueue", return_value=queue),
    ):
        manager = mod.MegatronCheckpointManager(
            model=MagicMock(),
            optimizer=MagicMock(),
            lr_scheduler=None,
            async_save=True,
        )
        yield mod, manager, queue


def test_async_disabled_creates_no_queue():
    mod = _import_checkpointer()
    with patch("torch.distributed.get_rank", return_value=0):
        m = mod.MegatronCheckpointManager(
            model=MagicMock(),
            optimizer=MagicMock(),
            lr_scheduler=None,
            async_save=False,
        )
    assert m._async_queue is None
    m._reap_finished_async_saves()
    m.wait_async_saves()
    m.close()


def test_save_schedules_async_request(patched_checkpointer, tmp_path):
    mod, manager, queue = patched_checkpointer
    fake_request = object()

    with (
        patch.object(manager, "generate_state_dict", return_value={"model": {}}),
        patch.object(
            mod, "save_dist_checkpointing", return_value=fake_request
        ) as save_fn,
        patch("torch.cuda.empty_cache"),
        patch("torch.distributed.barrier"),
    ):
        manager.save_checkpoint(str(tmp_path / "step0"))

    save_fn.assert_called_once()
    assert save_fn.call_args.kwargs["async_save"] is True
    queue.schedule_async_request.assert_called_once_with(fake_request)


def test_save_reaps_before_scheduling_next(patched_checkpointer, tmp_path):
    mod, manager, queue = patched_checkpointer

    with (
        patch.object(manager, "generate_state_dict", return_value={"model": {}}),
        patch.object(mod, "save_dist_checkpointing", side_effect=["r1", "r2"]),
        patch("torch.cuda.empty_cache"),
        patch("torch.distributed.barrier"),
    ):
        manager.save_checkpoint(str(tmp_path / "step0"))
        manager.save_checkpoint(str(tmp_path / "step1"))

    calls = queue.maybe_finalize_async_calls.call_args_list
    assert len(calls) == 2
    assert all(call.kwargs.get("blocking", False) is False for call in calls)
    assert queue.schedule_async_request.call_count == 2


def test_load_blocks_on_pending_saves(patched_checkpointer, tmp_path):
    mod, manager, queue = patched_checkpointer
    queue.get_num_unfinalized_calls.return_value = 1

    with (
        patch("os.path.exists", return_value=True),
        patch.object(manager, "generate_state_dict", return_value={}),
        patch.object(mod, "load_dist_checkpointing", return_value={}),
    ):
        with pytest.raises((AssertionError, KeyError)):
            manager.load_checkpoint(str(tmp_path / "step0"))

    queue.maybe_finalize_async_calls.assert_called_with(blocking=True)


def test_close_is_idempotent(patched_checkpointer):
    _, manager, queue = patched_checkpointer
    queue.get_num_unfinalized_calls.return_value = 0

    manager.close()
    manager.close()

    assert manager._async_queue is None


def test_async_save_reports_queue_depth_only(patched_checkpointer, tmp_path):
    """async_save emits ckpt/async_save_queue_depth on schedule and no other metric.

    Successful finalize is observable as queue_depth returning to 0; a failing
    background save raises from wait_async_saves, so an explicit count metric
    would be redundant.
    """
    mod, manager, queue = patched_checkpointer
    stats_scalar = sys.modules["areal.utils.stats_tracker"].scalar
    stats_scalar.reset_mock()

    queue.schedule_async_request.return_value = 42
    queue.get_num_unfinalized_calls.return_value = 1

    with (
        patch.object(manager, "generate_state_dict", return_value={"model": {}}),
        patch.object(mod, "save_dist_checkpointing", side_effect=["r1", "r2"]),
        patch("torch.cuda.empty_cache"),
        patch("torch.distributed.barrier"),
    ):
        manager.save_checkpoint(str(tmp_path / "step0"))

        # On the next save, reap returns [42] -> still no extra metrics.
        queue.maybe_finalize_async_calls.return_value = [42]
        queue.schedule_async_request.return_value = 43
        manager.save_checkpoint(str(tmp_path / "step1"))

    all_keys = set()
    for c in stats_scalar.call_args_list:
        all_keys.update(c.kwargs.keys())
    assert all_keys == {"ckpt/async_save_queue_depth"}


def test_sync_save_emits_no_async_metrics(patched_checkpointer, tmp_path):
    """Sync save path stays metric-free; trainer-side `timeperf/save` is sufficient."""
    mod, manager, _ = patched_checkpointer
    manager.async_save = False
    manager._async_queue = None
    stats_scalar = sys.modules["areal.utils.stats_tracker"].scalar
    stats_scalar.reset_mock()

    with (
        patch.object(manager, "generate_state_dict", return_value={"model": {}}),
        patch.object(mod, "save_dist_checkpointing", return_value=None),
        patch("torch.cuda.empty_cache"),
        patch("torch.distributed.barrier"),
    ):
        manager.save_checkpoint(str(tmp_path / "step0"))

    stats_scalar.assert_not_called()
