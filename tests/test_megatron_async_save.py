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
import json
import os
import pathlib
import stat
import subprocess
import sys
import types
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import NamedTuple
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
        "areal.engine.megatron_utils.gpu_staged_optimizer_checkpoint",
        "areal.engine.megatron_utils.managed_async_checkpoint",
        "areal.infra",
        "areal.infra.platforms",
        "areal.utils",
        "areal.utils.logging",
    ):
        sys.modules.setdefault(path, types.ModuleType(path))

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    for module_name, relative_path in (
        (
            "areal.engine.megatron_utils.checkpoint_snapshot",
            "areal/engine/megatron_utils/checkpoint_snapshot.py",
        ),
        (
            "areal.engine.megatron_utils.managed_async_marker",
            "areal/engine/megatron_utils/managed_async_marker.py",
        ),
        (
            "areal.engine.megatron_utils.managed_async_finalize",
            "areal/engine/megatron_utils/managed_async_finalize.py",
        ),
    ):
        spec = importlib.util.spec_from_file_location(
            module_name, repo_root / relative_path
        )
        support = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = support
        spec.loader.exec_module(support)

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
    staged_checkpoint = sys.modules[
        "areal.engine.megatron_utils.gpu_staged_optimizer_checkpoint"
    ]
    staged_names = (
        "abort_managed_checkpoint_load",
        "apply_begin_managed_checkpoint_load",
        "apply_managed_optimizer_reset_from_model",
        "attach_managed_optimizer_identities",
        "begin_managed_async_checkpoint_save",
        "bind_managed_async_checkpoint_request",
        "build_managed_optimizer_identities",
        "build_managed_optimizer_outer_template",
        "build_managed_optimizer_tensor_manifest",
        "complete_managed_async_checkpoint_save",
        "configure_managed_checkpoint_snapshots",
        "fail_managed_async_checkpoint_save",
        "has_managed_mcore_outer_schema",
        "is_managed_optimizer_tensor_checkpoint_key",
        "merge_managed_optimizer_tensor_manifests",
        "poison_managed_checkpoint_transaction",
        "preflight_managed_checkpoint_snapshots",
        "prepare_managed_checkpoint_commit",
        "prepare_managed_checkpoint_load",
        "prepare_managed_checkpoint_recovery",
        "prepare_managed_checkpoint_save",
        "retry_managed_checkpoint_cleanup",
        "validate_managed_checkpoint_load_request",
        "validate_managed_optimizer_outer_state",
        "validate_managed_optimizer_source_tensor_metadata",
        "vote_managed_checkpoint_phase",
    )
    for name in staged_names:
        setattr(staged_checkpoint, name, MagicMock())
    staged_checkpoint.prepare_managed_checkpoint_save.return_value = ()
    staged_checkpoint.create_managed_checkpoint_load_transaction = MagicMock(
        return_value=types.SimpleNamespace(
            leaves=[], committed=False, cleanup_pending=[]
        )
    )
    sys.modules[
        "areal.engine.megatron_utils.checkpoint_snapshot"
    ].validate_shared_snapshot_capacity = MagicMock()
    managed_async = sys.modules["areal.engine.megatron_utils.managed_async_checkpoint"]

    class ManagedAsyncSaveState(Enum):
        IDLE = auto()
        SAVE_STAGING = auto()
        SAVE_IN_FLIGHT = auto()
        COMPLETE = auto()
        FAILED = auto()

    @dataclass
    class ManagedAsyncSaveTransaction:
        checkpoint_id: str
        path: str
        leaves: tuple
        control_group: object
        logical_call_id: int
        expected_call_idx: int
        marker_leaves: list[dict]
        marker_leaves_digest: str
        state: ManagedAsyncSaveState = ManagedAsyncSaveState.SAVE_STAGING
        request: object | None = None
        call_idx: int | None = None
        completion_callbacks: list = field(default_factory=list)
        error: BaseException | None = None
        marker_created: bool = False
        marker_authority: object | None = None
        marker_committed: bool = False
        marker_cleanup_diagnostic: str | None = None
        worker_recovery: object | None = None
        recovery_token: object | None = None

    managed_async.ManagedAsyncSaveState = ManagedAsyncSaveState
    managed_async.ManagedAsyncSaveTransaction = ManagedAsyncSaveTransaction
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
    queue.call_idx = -1

    def gather_object(output, value, **_kwargs):
        output[0] = value

    def finalize_managed(async_queue, _group, *, blocking, recovery_token, **_kwargs):
        try:
            result = async_queue.maybe_finalize_async_calls(blocking=blocking)
        except BaseException:
            # This generic manager fixture has no worker journal; its mocked
            # native queue is the cleanup authority and is fully consumed when
            # it reports the terminal error.
            recovery_token.mark_cleared()
            raise
        if result:
            recovery_token.mark_cleared()
        return result

    def abort_managed(async_queue, _group, *, recovery_token, **_kwargs):
        async_queue.async_calls.clear()
        recovery_token.mark_cleared()

    with (
        patch("torch.distributed.get_rank", return_value=0),
        patch("torch.distributed.get_world_size", return_value=1),
        patch("torch.distributed.get_backend", return_value="gloo"),
        patch("torch.distributed.get_process_group_ranks", return_value=[0]),
        patch("torch.distributed.broadcast_object_list"),
        patch("torch.distributed.all_gather_object", side_effect=gather_object),
        patch.object(mod, "AsyncCallsQueue", return_value=queue),
        patch.object(
            mod,
            "finalize_managed_async_calls",
            side_effect=finalize_managed,
        ),
        patch.object(
            mod,
            "abort_managed_async_calls",
            side_effect=abort_managed,
        ),
        patch.object(mod, "preflight_managed_async_finalize"),
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


class _FakeActiveRequest(NamedTuple):
    idx: int
    async_caller: object
    async_request: object


class _FakeProcess:
    def __init__(self, *, alive: bool = False, exitcode: int = 0):
        self.alive = alive
        self.exitcode = exitcode
        self.join_count = 0
        self.kill_count = 0
        self.close_count = 0

    def is_alive(self):
        return self.alive

    def join(self, _timeout=None):
        self.join_count += 1

    def kill(self):
        self.kill_count += 1
        self.alive = False
        self.exitcode = -9

    def close(self):
        self.close_count += 1


class _FakeCaller:
    def __init__(self, process):
        self.process = process
        self.start_time = 0.0
        self.preloaded_holder = object()


def _fake_finalize_callbacks(writer, strategy, expected_finalize):
    save_state_dict_async_finalize = expected_finalize
    save_state_dict_ret = (
        writer,
        object(),
        types.SimpleNamespace(group=None, coordinator_rank=0),
    )
    checkpoint_dir = "/checkpoint"
    sharded_strategy = strategy

    def finalize_fn():
        save_state_dict_async_finalize(*save_state_dict_ret)

    def metadata_finalize_fn():
        return checkpoint_dir, sharded_strategy

    return [finalize_fn, metadata_finalize_fn]


def _run_fake_managed_finalize(
    monkeypatch, *, failure_phase=None, extra_callback=False
):
    _import_checkpointer()
    finalize = sys.modules["areal.engine.megatron_utils.managed_async_finalize"]

    def expected_finalize(*_args):
        return None

    results_queue = object()
    writer = types.SimpleNamespace(
        checkpoint_dir="/checkpoint",
        results_queue=results_queue,
        retrieve_write_results=MagicMock(return_value=["rank0-write"]),
        finish=MagicMock(),
    )
    strategy = types.SimpleNamespace(backend="torch_dist", version=1)
    callbacks = _fake_finalize_callbacks(writer, strategy, expected_finalize)
    if extra_callback:
        callbacks.append(lambda: None)
    request = types.SimpleNamespace(
        finalize_fns=callbacks,
        async_fn=lambda: None,
        async_fn_args=(0, object(), results_queue),
        call_idx=0,
    )
    process = _FakeProcess()
    caller = _FakeCaller(process)
    active = _FakeActiveRequest(0, caller, request)
    queue = types.SimpleNamespace(async_calls=deque([active]), call_idx=0)

    def local_vote(_group, phase, error, *, details=None, **_kwargs):
        if error is not None:
            raise finalize.ManagedAsyncFinalizeError(
                phase,
                [
                    {
                        "global_rank": 0,
                        "phase": phase,
                        "error": {"type": type(error).__name__, "message": str(error)},
                        "details": details,
                    }
                ],
            ) from error
        return [
            {
                "global_rank": 0,
                "phase": phase,
                "error": None,
                "details": details,
            }
        ]

    contract = finalize._MCore017Contract(
        temporal_caller_type=_FakeCaller,
        expected_finalize_fn=expected_finalize,
        active_request_type=_FakeActiveRequest,
        dcp_callback_code=callbacks[0].__code__,
        dcp_callback_globals=callbacks[0].__globals__,
        metadata_callback_code=callbacks[1].__code__,
        metadata_callback_globals=callbacks[1].__globals__,
    )
    monkeypatch.setattr(
        finalize, "_validate_mcore_017_contract", lambda _queue: contract
    )
    monkeypatch.setattr(finalize, "_group_manifest", lambda _group: (0, (0,)))
    monkeypatch.setattr(finalize, "_vote", local_vote)
    monkeypatch.setattr(finalize.dist, "get_process_group_ranks", lambda _group: [0])

    def gather(output, value, **_kwargs):
        output[0] = value

    monkeypatch.setattr(finalize.dist, "all_gather_object", gather)
    serialization = sys.modules["megatron.core.dist_checkpointing.serialization"]
    save_config = MagicMock()
    monkeypatch.setattr(serialization, "save_config", save_config, raising=False)
    monkeypatch.setattr(
        serialization,
        "CheckpointingConfig",
        lambda backend, version: (backend, version),
        raising=False,
    )
    if failure_phase == "worker_result":
        writer.retrieve_write_results.side_effect = RuntimeError("worker result failed")
    elif failure_phase == "dcp_metadata_finish":
        writer.finish.side_effect = RuntimeError("metadata finish failed")
    elif failure_phase == "mcore_config_finish":
        save_config.side_effect = RuntimeError("config finish failed")

    return finalize, queue, process, writer, save_config


def test_managed_finalize_adapter_replaces_collective_callbacks_once(monkeypatch):
    finalize, queue, process, writer, save_config = _run_fake_managed_finalize(
        monkeypatch
    )

    assert finalize.finalize_managed_async_calls(
        queue, object(), expected_call_idx=0, bound_call_idx=0, blocking=True
    ) == [0]
    assert len(queue.async_calls) == 0
    assert finalize.get_managed_async_worker_recovery(queue) is None
    assert finalize._worker_journal(queue) is None
    writer.finish.assert_called_once_with(
        writer.finish.call_args.args[0], [["rank0-write"]]
    )
    save_config.assert_called_once()
    assert process.close_count == 1

    writer.finish.assert_called_once()
    save_config.assert_called_once()


@pytest.mark.slow
def test_managed_finalize_two_real_queue_cycles_release_all_authority():
    """Two MCore queue schedules succeed without retaining prior authority."""
    script = r"""
from types import CodeType, FunctionType, SimpleNamespace

from megatron.core.dist_checkpointing import serialization
from megatron.core.dist_checkpointing.strategies import async_utils
from megatron.core.dist_checkpointing.strategies.state_dict_saver import (
    save_state_dict_async_finalize,
)
from megatron.core.dist_checkpointing.strategies.torch import (
    TorchDistSaveShardedStrategy,
)

import areal.engine.megatron_utils.managed_async_finalize as finalize


def nested_code(producer, name, freevars):
    matches = [
        value
        for value in producer.__code__.co_consts
        if isinstance(value, CodeType)
        and value.co_name == name
        and value.co_freevars == freevars
    ]
    assert len(matches) == 1
    return matches[0]


def closure_cell(value):
    def inner():
        return value

    return inner.__closure__[0]


class Writer:
    checkpoint_dir = "/checkpoint"
    results_queue = None

    def __init__(self):
        self.finish_count = 0

    def retrieve_write_results(self):
        return ["rank0-write"]

    def finish(self, _metadata, results):
        assert results == [["rank0-write"]]
        self.finish_count += 1


writer = Writer()
global_metadata = object()
dist_wrapper = SimpleNamespace(group=None, coordinator_rank=0)
save_state_dict_ret = (writer, global_metadata, dist_wrapper)
finalize_fn = FunctionType(
    nested_code(
        TorchDistSaveShardedStrategy._get_save_and_finalize_callbacks,
        "finalize_fn",
        ("save_state_dict_async_finalize", "save_state_dict_ret"),
    ),
    TorchDistSaveShardedStrategy._get_save_and_finalize_callbacks.__globals__,
    closure=(
        closure_cell(save_state_dict_async_finalize),
        closure_cell(save_state_dict_ret),
    ),
)
checkpoint_dir = writer.checkpoint_dir
sharded_strategy = SimpleNamespace(backend="torch_dist", version=1)
metadata_finalize_fn = FunctionType(
    nested_code(
        serialization.save,
        "metadata_finalize_fn",
        ("checkpoint_dir", "sharded_strategy"),
    ),
    serialization.save.__globals__,
    closure=(closure_cell(checkpoint_dir), closure_cell(sharded_strategy)),
)


def local_vote(_group, phase, error, *, details=None, **_kwargs):
    if error is not None:
        raise finalize.ManagedAsyncFinalizeError(
            phase,
            [{"global_rank": 0, "error": {"message": str(error)}, "details": details}],
        ) from error
    return [{"global_rank": 0, "error": None, "details": details}]


finalize._group_manifest = lambda _group: (0, (0,))
finalize._vote = local_vote
serialization.save_config = lambda *_args, **_kwargs: None
serialization.CheckpointingConfig = lambda backend, version: (backend, version)
captured = []
original_ensure = finalize._ensure_worker_journal


def capture_journal(queue, active):
    journal = original_ensure(queue, active)
    if all(item is not journal for item in captured):
        captured.append(journal)
    return journal


finalize._ensure_worker_journal = capture_journal
queue = async_utils.AsyncCallsQueue()
callbacks = [finalize_fn, metadata_finalize_fn]
for expected_idx in range(2):
    request = async_utils.AsyncRequest(None, (), callbacks)
    assert queue.schedule_async_request(request) == expected_idx
    assert finalize.finalize_managed_async_calls(
        queue,
        object(),
        expected_call_idx=expected_idx,
        bound_call_idx=expected_idx,
        blocking=True,
    ) == [expected_idx]
    assert queue.get_num_unfinalized_calls() == 0
    assert finalize._worker_journal(queue) is None
    assert finalize._worker_terminal_cleanup(queue) is None
    assert finalize._worker_recovery_publication(queue) is None

assert len(captured) == 2
for journal in captured:
    assert journal.active is None
    assert journal.caller is None
    assert journal.process is None
    assert journal.diagnostics is None
    assert journal.original_failure is None
assert writer.finish_count == 2
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=90)


@pytest.mark.parametrize(
    "failure_phase",
    ("worker_result", "dcp_metadata_finish", "mcore_config_finish"),
)
def test_managed_finalize_adapter_reaps_queue_on_local_phase_failure(
    monkeypatch, failure_phase
):
    finalize, queue, process, writer, save_config = _run_fake_managed_finalize(
        monkeypatch, failure_phase=failure_phase
    )

    with pytest.raises(finalize.ManagedAsyncFinalizeError, match=failure_phase):
        finalize.finalize_managed_async_calls(
            queue, object(), expected_call_idx=0, bound_call_idx=0, blocking=True
        )

    assert len(queue.async_calls) == 0
    assert finalize._worker_journal(queue) is None
    assert finalize._worker_terminal_cleanup(queue) is None
    assert finalize._worker_recovery_publication(queue) is None
    assert process.close_count == 1
    if failure_phase != "worker_result":
        writer.finish.assert_called_once()
    if failure_phase == "mcore_config_finish":
        save_config.assert_called_once()


def test_managed_finalize_adapter_rejects_unknown_callback_without_replay(monkeypatch):
    finalize, queue, process, writer, save_config = _run_fake_managed_finalize(
        monkeypatch, extra_callback=True
    )

    with pytest.raises(finalize.ManagedAsyncFinalizeError, match="callback_audit"):
        finalize.finalize_managed_async_calls(
            queue, object(), expected_call_idx=0, bound_call_idx=0, blocking=True
        )

    assert len(queue.async_calls) == 0
    assert process.close_count == 1
    writer.finish.assert_not_called()
    save_config.assert_not_called()


def test_managed_finalize_adapter_rejects_forged_same_shape_callback(monkeypatch):
    """A callback with matching name/freevars is still not an audited callback."""
    finalize, queue, process, writer, save_config = _run_fake_managed_finalize(
        monkeypatch
    )
    checkpoint_dir = "/checkpoint"
    sharded_strategy = types.SimpleNamespace(backend="torch_dist", version=1)

    def metadata_finalize_fn():
        return checkpoint_dir, sharded_strategy

    queue.async_calls[0].async_request.finalize_fns[1] = metadata_finalize_fn

    with pytest.raises(finalize.ManagedAsyncFinalizeError, match="callback_audit"):
        finalize.finalize_managed_async_calls(
            queue, object(), expected_call_idx=0, bound_call_idx=0, blocking=True
        )

    assert len(queue.async_calls) == 0
    assert process.close_count == 1
    writer.finish.assert_not_called()
    save_config.assert_not_called()


def test_managed_finalize_adapter_rejects_queue_call_index_mismatch(monkeypatch):
    """The active record must remain bound to the queue's current call index."""
    finalize, queue, process, writer, save_config = _run_fake_managed_finalize(
        monkeypatch
    )
    queue.call_idx = 7

    with pytest.raises(finalize.ManagedAsyncFinalizeError, match="queue"):
        finalize.finalize_managed_async_calls(
            queue, object(), expected_call_idx=0, bound_call_idx=0, blocking=True
        )

    assert len(queue.async_calls) == 0
    assert process.close_count == 1
    writer.finish.assert_not_called()
    save_config.assert_not_called()


@pytest.mark.parametrize(
    ("active_idx", "expected_idx", "bound_idx"),
    ((7, 0, 0), (0, 0, 7)),
)
def test_managed_finalize_adapter_rejects_record_or_transaction_index_mismatch(
    monkeypatch, active_idx, expected_idx, bound_idx
):
    finalize, queue, process, writer, save_config = _run_fake_managed_finalize(
        monkeypatch
    )
    active = queue.async_calls[0]
    queue.async_calls[0] = active._replace(idx=active_idx)

    with pytest.raises(finalize.ManagedAsyncFinalizeError, match="queue_binding"):
        finalize.finalize_managed_async_calls(
            queue,
            object(),
            expected_call_idx=expected_idx,
            bound_call_idx=bound_idx,
            blocking=True,
        )

    assert len(queue.async_calls) == 0
    assert process.close_count == 1
    writer.finish.assert_not_called()
    save_config.assert_not_called()


def test_managed_finalize_adapter_votes_post_pop_failure_and_reconciles_queue(
    monkeypatch,
):
    finalize, queue, process, writer, save_config = _run_fake_managed_finalize(
        monkeypatch
    )
    original_remove = finalize._remove_active_call

    def remove_then_raise(async_queue, active):
        original_remove(async_queue, active)
        raise RuntimeError("injected post-pop failure")

    monkeypatch.setattr(finalize, "_remove_active_call", remove_then_raise)

    with pytest.raises(finalize.ManagedAsyncFinalizeError, match="queue_pop"):
        finalize.finalize_managed_async_calls(
            queue,
            object(),
            expected_call_idx=0,
            bound_call_idx=0,
            blocking=True,
        )

    assert len(queue.async_calls) == 0
    assert finalize._worker_journal(queue) is None
    assert finalize._worker_terminal_cleanup(queue) is None
    assert finalize._worker_recovery_publication(queue) is None
    assert process.close_count == 1
    writer.finish.assert_called_once()
    save_config.assert_called_once()


def test_failure_cleanup_holds_global_publication_until_every_rank_pops(monkeypatch):
    """A locally clean rank participates without replaying its completed pop."""

    finalize, queue, _process, _writer, _save_config = _run_fake_managed_finalize(
        monkeypatch
    )
    remove_count = 0
    original_remove = finalize._remove_active_call

    def counted_remove(async_queue, active):
        nonlocal remove_count
        remove_count += 1
        original_remove(async_queue, active)

    monkeypatch.setattr(finalize, "_remove_active_call", counted_remove)
    cleanup_round = 0
    phase_trace = []

    def two_rank_vote(
        _group,
        phase,
        error,
        *,
        phase_id=None,
        details=None,
        **_kwargs,
    ):
        nonlocal cleanup_round
        phase_trace.append((cleanup_round, phase_id, phase))
        if error is not None:
            raise finalize.ManagedAsyncFinalizeError(
                phase,
                [{"global_rank": 0, "error": {"message": str(error)}}],
            ) from error
        remote_details = details
        if phase == "failure_queue_pop":
            remote_pending = cleanup_round < 2
            remote_details = {
                "call_idx": 0,
                "record_removed": not remote_pending,
                "terminal_cleanup_pending": remote_pending,
                "worker_recovery_pending": False,
                "error": None,
            }
        return [
            {
                "global_rank": 0,
                "phase": phase,
                "phase_id": phase_id,
                "error": None,
                "details": details,
            },
            {
                "global_rank": 1,
                "phase": phase,
                "phase_id": phase_id,
                "error": None,
                "details": remote_details,
            },
        ]

    monkeypatch.setattr(finalize, "_vote", two_rank_vote)
    original = RuntimeError("checkpoint failed")
    recovery_token = finalize.ManagedAsyncRecoveryToken()
    for cleanup_round in range(3):
        runner = finalize._FinalizePhaseRunner(object())
        finalize._cleanup_after_failure(
            runner,
            queue,
            original,
            transaction_call_idx=0,
            recovery_token=recovery_token,
        )
        if cleanup_round < 2:
            assert finalize.get_managed_async_worker_recovery(queue) is not None
        else:
            assert finalize.get_managed_async_worker_recovery(queue) is None

    assert remove_count == 1
    assert len(queue.async_calls) == 0
    assert [phase for round_id, _phase_id, phase in phase_trace if round_id == 0] == [
        "failure_worker_reap",
        "failure_worker_recovery_publish_prepare",
        "failure_worker_recovery_publish",
        "failure_worker_recovery_publish_result",
        "failure_queue_pop",
        "failure_worker_recovery_hold_prepare",
        "failure_worker_recovery_hold",
        "failure_worker_recovery_hold_result",
    ]
    assert [phase for round_id, _phase_id, phase in phase_trace if round_id == 2] == [
        "failure_worker_reap",
        "failure_worker_recovery_publish_prepare",
        "failure_worker_recovery_publish",
        "failure_worker_recovery_publish_result",
        "failure_queue_pop",
        "failure_worker_recovery_clear_prepare",
        "failure_worker_recovery_clear",
        "failure_worker_recovery_clear_result",
    ]


def test_failure_cleanup_republishes_after_single_rank_clear_failure(monkeypatch):
    """A post-pop clear failure cannot release the manager fence on a peer."""

    finalize, queue, _process, _writer, _save_config = _run_fake_managed_finalize(
        monkeypatch
    )
    fail_clear = True
    original_clear = finalize._clear_worker_recovery_publication

    def fail_first_clear(async_queue):
        nonlocal fail_clear
        if fail_clear:
            fail_clear = False
            raise RuntimeError("injected clear failure")
        return original_clear(async_queue)

    monkeypatch.setattr(
        finalize, "_clear_worker_recovery_publication", fail_first_clear
    )
    original = RuntimeError("checkpoint failed")
    recovery_token = finalize.ManagedAsyncRecoveryToken()
    finalize._cleanup_after_failure(
        finalize._FinalizePhaseRunner(object()),
        queue,
        original,
        transaction_call_idx=0,
        recovery_token=recovery_token,
    )
    assert finalize.get_managed_async_worker_recovery(queue) is not None
    assert len(queue.async_calls) == 0

    finalize._cleanup_after_failure(
        finalize._FinalizePhaseRunner(object()),
        queue,
        original,
        transaction_call_idx=0,
        recovery_token=recovery_token,
    )
    assert finalize.get_managed_async_worker_recovery(queue) is None


def test_manager_finalize_call_index_mismatch_is_fail_closed(
    patched_checkpointer,
):
    mod, manager, _queue = patched_checkpointer

    class PhaseError(RuntimeError):
        def __init__(self, phase, local_error):
            super().__init__(f"{phase}: {local_error}")
            self.local_error = local_error

    transaction = mod.ManagedAsyncSaveTransaction(
        checkpoint_id="checkpoint-id",
        path="/checkpoint",
        leaves=(),
        control_group=object(),
        logical_call_id=1,
        expected_call_idx=3,
        marker_leaves=[],
        marker_leaves_digest="digest",
    )
    transaction.call_idx = 3
    transaction.state = mod.ManagedAsyncSaveState.SAVE_IN_FLIGHT
    manager._managed_async_save = transaction
    manager._vote_managed_phase = lambda phase, error, *_args, **_kwargs: (
        PhaseError(phase, error) if error is not None else None
    )

    with (
        patch.object(
            mod,
            "create_managed_checkpoint_load_transaction",
            return_value=types.SimpleNamespace(committed=False),
        ),
        patch.object(mod, "fail_managed_async_checkpoint_save") as fail,
        pytest.raises(PhaseError, match="async_finalize_binding"),
    ):
        manager._finalize_managed_async_save(4)

    fail.assert_called_once()
    assert manager.managed_async_save_state == "FAILED"
    assert manager._managed_async_save_error is not None


def test_managed_finalize_adapter_times_out_and_reaps_exact_worker(monkeypatch):
    finalize, queue, process, writer, save_config = _run_fake_managed_finalize(
        monkeypatch
    )
    process.alive = True

    with pytest.raises(finalize.ManagedAsyncFinalizeError, match="worker_reap"):
        finalize.finalize_managed_async_calls(
            queue,
            object(),
            expected_call_idx=0,
            bound_call_idx=0,
            blocking=True,
            timeout_seconds=0.001,
        )

    assert process.kill_count == 1
    assert process.close_count == 1
    assert len(queue.async_calls) == 0
    writer.finish.assert_not_called()
    save_config.assert_not_called()


def test_managed_finalize_adapter_retains_unterminated_worker_authority(monkeypatch):
    """A worker that survives kill must remain reachable for later recovery."""
    finalize, queue, process, writer, save_config = _run_fake_managed_finalize(
        monkeypatch
    )
    process.alive = True

    def ignore_kill():
        process.kill_count += 1

    process.kill = ignore_kill

    with pytest.raises(finalize.ManagedAsyncFinalizeError, match="worker_reap"):
        finalize.finalize_managed_async_calls(
            queue,
            object(),
            expected_call_idx=0,
            bound_call_idx=0,
            blocking=True,
            timeout_seconds=0.001,
        )

    assert process.is_alive()
    assert queue.async_calls[0].async_caller.process is process
    assert process.close_count == 0
    writer.finish.assert_not_called()
    save_config.assert_not_called()


def test_managed_finalize_early_failure_publishes_worker_recovery(monkeypatch):
    """A journal created during cleanup must retain the triggering failure."""
    finalize, queue, process, writer, save_config = _run_fake_managed_finalize(
        monkeypatch
    )
    process.alive = True
    process.terminate = lambda: None

    def ignore_kill():
        process.kill_count += 1

    process.kill = ignore_kill

    with pytest.raises(
        finalize.ManagedAsyncFinalizeError, match="transaction_call_indices"
    ):
        finalize.finalize_managed_async_calls(
            queue,
            object(),
            expected_call_idx=True,
            bound_call_idx=0,
            blocking=True,
            timeout_seconds=0.001,
        )

    recovery = finalize.get_managed_async_worker_recovery(queue)
    assert recovery is not None
    assert recovery.original_failure is not None
    assert recovery.process is process
    assert len(queue.async_calls) == 1
    writer.finish.assert_not_called()
    save_config.assert_not_called()


def test_managed_finalize_adapter_retains_worker_alive_after_terminate_and_kill(
    monkeypatch,
):
    """Successful signals do not consume authority until bounded join sees exit."""
    finalize, queue, process, writer, save_config = _run_fake_managed_finalize(
        monkeypatch
    )
    process.alive = True
    process.terminate_count = 0

    def ignore_terminate():
        process.terminate_count += 1

    def ignore_kill():
        process.kill_count += 1

    process.terminate = ignore_terminate
    process.kill = ignore_kill

    with pytest.raises(finalize.ManagedAsyncFinalizeError, match="worker_reap"):
        finalize.finalize_managed_async_calls(
            queue,
            object(),
            expected_call_idx=0,
            bound_call_idx=0,
            blocking=True,
            timeout_seconds=0.001,
        )

    journal = finalize.get_managed_async_worker_recovery(queue)
    assert journal is not None
    assert journal.process is process
    assert journal.active is queue.async_calls[0]
    assert journal.call_idx == 0
    assert journal.stage is finalize._WorkerReapStage.KILL_JOIN_PENDING
    assert process.terminate_count == 1
    assert process.kill_count == 1
    assert process.close_count == 0
    assert queue.async_calls[0].async_caller.process is process
    writer.finish.assert_not_called()
    save_config.assert_not_called()


def test_managed_finalize_recovery_retries_only_pending_worker_steps(monkeypatch):
    """A successful kill is not replayed while its pending join is retried."""
    finalize, queue, process, writer, save_config = _run_fake_managed_finalize(
        monkeypatch
    )
    process.alive = True

    def kill_without_exit():
        process.kill_count += 1

    process.kill = kill_without_exit
    with pytest.raises(finalize.ManagedAsyncFinalizeError, match="worker_reap"):
        finalize.finalize_managed_async_calls(
            queue,
            object(),
            expected_call_idx=0,
            bound_call_idx=0,
            blocking=True,
            timeout_seconds=0.001,
        )

    journal = finalize.get_managed_async_worker_recovery(queue)
    assert journal is not None
    assert journal.stage is finalize._WorkerReapStage.KILL_JOIN_PENDING
    assert process.kill_count == 1
    process.alive = False
    process.exitcode = -9

    with pytest.raises(finalize.ManagedAsyncFinalizeError, match="recovery_terminal"):
        finalize.finalize_managed_async_calls(
            queue,
            object(),
            expected_call_idx=0,
            bound_call_idx=0,
            blocking=True,
            timeout_seconds=0.001,
        )

    assert process.kill_count == 1
    assert process.close_count == 1
    assert len(queue.async_calls) == 0
    assert finalize.get_managed_async_worker_recovery(queue) is None
    writer.finish.assert_not_called()
    save_config.assert_not_called()


def test_managed_abort_retains_worker_before_transaction_index_is_bound(monkeypatch):
    """A schedule-abort journal remains recoverable while bound index is unset."""
    finalize, queue, process, writer, save_config = _run_fake_managed_finalize(
        monkeypatch
    )
    process.alive = True

    def ignore_kill():
        process.kill_count += 1

    process.kill = ignore_kill
    with pytest.raises(finalize.ManagedAsyncFinalizeError, match="abort_worker_reap"):
        finalize.abort_managed_async_calls(queue, object())

    journal = finalize.get_managed_async_worker_recovery(queue)
    assert journal is not None
    assert journal.process is process
    assert journal.call_idx == 0
    assert queue.async_calls[0].async_caller.process is process

    process.alive = False
    process.exitcode = -9
    with pytest.raises(finalize.ManagedAsyncFinalizeError, match="recovery_terminal"):
        finalize.finalize_managed_async_calls(
            queue,
            object(),
            expected_call_idx=0,
            bound_call_idx=None,
            blocking=True,
            timeout_seconds=0.001,
        )

    assert len(queue.async_calls) == 0
    assert process.close_count == 1
    assert finalize.get_managed_async_worker_recovery(queue) is None
    writer.finish.assert_not_called()
    save_config.assert_not_called()


def test_managed_unbound_schedule_mismatch_recovers_active_record(monkeypatch):
    """An unbound transaction reaps the exact scheduled record despite index drift."""
    finalize, queue, process, writer, save_config = _run_fake_managed_finalize(
        monkeypatch
    )
    process.alive = True

    def ignore_kill():
        process.kill_count += 1

    process.kill = ignore_kill
    with pytest.raises(finalize.ManagedAsyncFinalizeError, match="abort_worker_reap"):
        finalize.abort_managed_async_calls(queue, object())

    journal = finalize.get_managed_async_worker_recovery(queue)
    assert journal is not None
    assert journal.call_idx == 0
    process.alive = False
    process.exitcode = -9

    with pytest.raises(finalize.ManagedAsyncFinalizeError, match="recovery_terminal"):
        finalize.finalize_managed_async_calls(
            queue,
            object(),
            expected_call_idx=7,
            bound_call_idx=None,
            blocking=True,
            timeout_seconds=0.001,
        )

    assert process.kill_count == 1
    assert process.close_count == 1
    assert len(queue.async_calls) == 0
    assert finalize.get_managed_async_worker_recovery(queue) is None
    writer.finish.assert_not_called()
    save_config.assert_not_called()


@pytest.mark.parametrize("failure_step", ("terminate", "kill", "join", "close"))
def test_managed_finalize_worker_recovery_retains_each_failed_action(
    monkeypatch, failure_step
):
    finalize, queue, process, writer, save_config = _run_fake_managed_finalize(
        monkeypatch
    )
    original_join = process.join
    original_kill = process.kill
    original_close = process.close
    process.alive = failure_step in ("terminate", "kill")

    if failure_step == "terminate":
        process.terminate = MagicMock(side_effect=RuntimeError("terminate failed"))
    elif failure_step == "kill":
        process.kill = MagicMock(side_effect=RuntimeError("kill failed"))
    elif failure_step == "join":
        process.join = MagicMock(side_effect=RuntimeError("join failed"))
    else:
        process.close = MagicMock(side_effect=RuntimeError("close failed"))

    with pytest.raises(finalize.ManagedAsyncFinalizeError):
        finalize.finalize_managed_async_calls(
            queue,
            object(),
            expected_call_idx=0,
            bound_call_idx=0,
            blocking=True,
            timeout_seconds=0.001,
        )

    journal = finalize.get_managed_async_worker_recovery(queue)
    assert journal is not None
    assert queue.async_calls[0].async_caller.process is process
    if failure_step == "terminate":
        process.terminate = lambda: setattr(process, "alive", False)
        process.exitcode = -15
    elif failure_step == "kill":
        process.kill = original_kill
    elif failure_step == "join":
        process.join = original_join
    else:
        process.close = original_close

    with pytest.raises(finalize.ManagedAsyncFinalizeError, match="recovery_terminal"):
        finalize.finalize_managed_async_calls(
            queue,
            object(),
            expected_call_idx=0,
            bound_call_idx=0,
            blocking=True,
            timeout_seconds=0.001,
        )

    assert len(queue.async_calls) == 0
    assert finalize.get_managed_async_worker_recovery(queue) is None
    writer.finish.assert_not_called()
    save_config.assert_not_called()


def test_manager_retains_fence_and_lease_until_worker_recovery(
    monkeypatch, patched_checkpointer
):
    mod, manager, _manager_queue = patched_checkpointer
    finalize, queue, process, writer, save_config = _run_fake_managed_finalize(
        monkeypatch
    )
    process.alive = True

    def kill_without_exit():
        process.kill_count += 1

    process.kill = kill_without_exit
    transaction = mod.ManagedAsyncSaveTransaction(
        checkpoint_id="checkpoint-id",
        path="/checkpoint",
        leaves=(object(),),
        control_group=object(),
        logical_call_id=1,
        expected_call_idx=0,
        marker_leaves=[],
        marker_leaves_digest="digest",
    )
    transaction.call_idx = 0
    transaction.request = object()
    lease = MagicMock()
    transaction.completion_callbacks.append(lease)
    manager._managed_async_save = transaction
    manager._async_queue = queue

    with (
        patch.object(mod, "fail_managed_async_checkpoint_save") as fail,
        pytest.raises(finalize.ManagedAsyncFinalizeError) as first_failure,
    ):
        finalize.finalize_managed_async_calls(
            queue,
            object(),
            expected_call_idx=0,
            bound_call_idx=0,
            blocking=True,
            timeout_seconds=0.001,
        )
    with patch.object(mod, "fail_managed_async_checkpoint_save") as fail:
        manager._record_managed_async_failure(transaction, first_failure.value)
        fail.assert_not_called()
    lease.assert_not_called()
    assert transaction.request is not None
    assert transaction.worker_recovery is not None

    process.alive = False
    process.exitcode = -9
    with pytest.raises(finalize.ManagedAsyncFinalizeError) as recovery_failure:
        finalize.finalize_managed_async_calls(
            queue,
            object(),
            expected_call_idx=0,
            bound_call_idx=0,
            blocking=True,
            timeout_seconds=0.001,
        )
    with patch.object(mod, "fail_managed_async_checkpoint_save") as fail:
        manager._record_managed_async_failure(transaction, recovery_failure.value)
        fail.assert_called_once()

    lease.assert_called_once()
    assert transaction.request is None
    assert transaction.worker_recovery is None
    assert len(queue.async_calls) == 0
    writer.finish.assert_not_called()
    save_config.assert_not_called()


def test_manager_retains_global_recovery_without_local_worker(patched_checkpointer):
    """A peer-owned worker keeps this rank's fence, lease, and marker authority."""
    mod, manager, _manager_queue = patched_checkpointer
    finalize = sys.modules["areal.engine.megatron_utils.managed_async_finalize"]
    queue = types.SimpleNamespace(async_calls=deque(), call_idx=-1)
    original = RuntimeError("rank 1 worker remains alive")
    publication = finalize._WorkerRecoveryPublication(
        original_failure=original,
        local_authority=None,
    )
    setattr(queue, finalize._WORKER_RECOVERY_PUBLICATION_ATTR, publication)
    transaction = mod.ManagedAsyncSaveTransaction(
        checkpoint_id="checkpoint-id",
        path="/checkpoint",
        leaves=(object(),),
        control_group=object(),
        logical_call_id=1,
        expected_call_idx=0,
        marker_leaves=[],
        marker_leaves_digest="digest",
    )
    transaction.request = object()
    marker = MagicMock()
    transaction.marker_authority = marker
    lease = MagicMock()
    transaction.completion_callbacks.append(lease)
    manager._managed_async_save = transaction
    manager._async_queue = queue

    with patch.object(mod, "fail_managed_async_checkpoint_save") as fail:
        manager._record_managed_async_failure(transaction, original)

    fail.assert_not_called()
    lease.assert_not_called()
    marker.close.assert_not_called()
    assert transaction.request is not None
    assert transaction.marker_authority is marker
    assert transaction.worker_recovery is publication
    assert manager._managed_async_save is transaction


def test_manager_recovery_token_is_authoritative_without_local_publication(
    patched_checkpointer,
):
    """A missing local publication cannot release request, fence, lease, or marker."""

    mod, manager, _manager_queue = patched_checkpointer
    transaction = mod.ManagedAsyncSaveTransaction(
        checkpoint_id="checkpoint-id",
        path="/checkpoint",
        leaves=(object(),),
        control_group=object(),
        logical_call_id=1,
        expected_call_idx=0,
        marker_leaves=[],
        marker_leaves_digest="digest",
    )
    transaction.request = object()
    transaction.marker_authority = MagicMock()
    lease = MagicMock()
    transaction.completion_callbacks.append(lease)
    token = mod.ManagedAsyncRecoveryToken()
    token.require_recovery()
    transaction.recovery_token = token
    manager._managed_async_save = transaction

    with patch.object(mod, "fail_managed_async_checkpoint_save") as fail:
        manager._record_managed_async_failure(
            transaction,
            RuntimeError("publication write failed"),
        )

    fail.assert_not_called()
    lease.assert_not_called()
    assert transaction.request is not None
    assert transaction.marker_authority is not None
    assert transaction.worker_recovery is token

    token.mark_cleared()
    with patch.object(mod, "fail_managed_async_checkpoint_save") as fail:
        manager._record_managed_async_failure(
            transaction,
            RuntimeError("checkpoint remains failed after cleanup"),
        )

    fail.assert_called_once()
    lease.assert_called_once()
    assert transaction.request is None
    assert transaction.marker_authority is None
    assert transaction.worker_recovery is None


def test_managed_finalize_adapter_preserves_base_exception_failure(monkeypatch):
    class FatalFinalize(BaseException):
        pass

    finalize, queue, process, writer, save_config = _run_fake_managed_finalize(
        monkeypatch
    )
    writer.retrieve_write_results.side_effect = FatalFinalize("fatal writer result")

    with pytest.raises(finalize.ManagedAsyncFinalizeError) as raised:
        finalize.finalize_managed_async_calls(
            queue, object(), expected_call_idx=0, bound_call_idx=0, blocking=True
        )

    assert "FatalFinalize" in str(raised.value)
    assert len(queue.async_calls) == 0
    assert process.close_count == 1
    writer.finish.assert_not_called()
    save_config.assert_not_called()


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


def test_managed_async_save_keeps_fence_until_foreground_finalize(
    patched_checkpointer, tmp_path
):
    mod, manager, queue = patched_checkpointer
    checkpoint_path = tmp_path / "managed-step0"
    fake_request = object()
    leaf = MagicMock()
    released = MagicMock()
    control_group = object()
    manager.managed_checkpoint_enabled = True
    manager.checkpoint_process_group = control_group
    manager._retry_managed_checkpoint_cleanup = MagicMock()
    manager._managed_optimizer_identities = MagicMock(
        return_value={(): {"leaf": "managed"}}
    )
    manager._vote_managed_phase = MagicMock(return_value=None)

    def run_phase(_phase, operation, _transaction):
        return operation(), None

    manager._run_managed_phase = run_phase
    manager._require_managed_checkpoint_group = MagicMock(return_value=control_group)
    queue.get_num_unfinalized_calls.return_value = 1
    queue.maybe_finalize_async_calls.side_effect = [[], [], [7], []]
    queue.schedule_async_request.return_value = 7
    queue.call_idx = 6

    with (
        patch.object(manager, "generate_state_dict", return_value={"optimizer": {}}),
        patch.object(mod, "save_dist_checkpointing", return_value=fake_request),
        patch.object(
            mod,
            "begin_managed_async_checkpoint_save",
            return_value=(leaf,),
        ),
        patch.object(mod, "bind_managed_async_checkpoint_request") as bind,
        patch.object(mod, "complete_managed_async_checkpoint_save") as complete,
        patch("torch.distributed.get_rank", return_value=0),
    ):
        manager.save_checkpoint(
            str(checkpoint_path), async_completion_callback=released
        )

        assert manager._managed_async_save is not None
        assert manager._managed_async_save.state.name == "SAVE_IN_FLIGHT"
        released.assert_not_called()
        bind.assert_called_once_with((leaf,), fake_request, 7)

        with pytest.raises(RuntimeError, match="only one managed asynchronous"):
            manager.save_checkpoint(str(tmp_path / "managed-step1"))

        (checkpoint_path / "metadata.json").write_text("{}")
        manager.wait_async_saves()
        manager.wait_async_saves()

    complete.assert_called_once_with((leaf,))
    released.assert_called_once_with()
    assert manager._managed_async_save is None
    assert not (checkpoint_path / mod._MANAGED_ASYNC_INCOMPLETE).exists()
    assert (checkpoint_path / mod._MANAGED_ASYNC_COMPLETE).exists()


def test_managed_async_schedule_failure_poison_fence_and_release_lease(
    patched_checkpointer, tmp_path
):
    mod, manager, queue = patched_checkpointer
    checkpoint_path = tmp_path / "managed-failed-schedule"
    leaf = MagicMock()
    released = MagicMock()
    control_group = object()

    class PhaseError(RuntimeError):
        def __init__(self, phase, local_error):
            super().__init__(f"{phase}: {local_error}")
            self.local_error = local_error

    manager.managed_checkpoint_enabled = True
    manager.checkpoint_process_group = control_group
    manager._retry_managed_checkpoint_cleanup = MagicMock()
    manager._managed_optimizer_identities = MagicMock(return_value={(): {}})
    manager._require_managed_checkpoint_group = MagicMock(return_value=control_group)
    manager._vote_managed_phase = lambda phase, error, *_a, **_k: (
        PhaseError(phase, error) if error is not None else None
    )

    def run_phase(phase, operation, transaction):
        try:
            result, error = operation(), None
        except BaseException as caught:
            result, error = None, caught
        return result, manager._vote_managed_phase(phase, error, transaction)

    manager._run_managed_phase = run_phase
    queue.async_calls = deque()
    queue.maybe_finalize_async_calls.return_value = []
    queue.schedule_async_request.side_effect = RuntimeError("schedule exploded")

    with (
        patch.object(manager, "generate_state_dict", return_value={"optimizer": {}}),
        patch.object(mod, "save_dist_checkpointing", return_value=object()),
        patch.object(mod, "begin_managed_async_checkpoint_save", return_value=(leaf,)),
        patch.object(mod, "fail_managed_async_checkpoint_save") as fail,
        patch("torch.distributed.get_rank", return_value=0),
    ):
        with pytest.raises(RuntimeError, match="schedule exploded"):
            manager.save_checkpoint(
                str(checkpoint_path), async_completion_callback=released
            )

    fail.assert_called_once()
    released.assert_called_once_with()
    assert manager.managed_async_save_state == "FAILED"
    assert (checkpoint_path / mod._MANAGED_ASYNC_INCOMPLETE).is_file()
    assert not (checkpoint_path / mod._MANAGED_ASYNC_COMPLETE).exists()


def test_managed_async_background_failure_is_observed_and_remains_incomplete(
    patched_checkpointer, tmp_path
):
    mod, manager, queue = patched_checkpointer
    checkpoint_path = tmp_path / "managed-failed-background"
    leaf = MagicMock()
    released = MagicMock()
    control_group = object()

    class PhaseError(RuntimeError):
        def __init__(self, phase, local_error):
            super().__init__(f"{phase}: {local_error}")
            self.local_error = local_error

    manager.managed_checkpoint_enabled = True
    manager.checkpoint_process_group = control_group
    manager._retry_managed_checkpoint_cleanup = MagicMock()
    manager._managed_optimizer_identities = MagicMock(return_value={(): {}})
    manager._require_managed_checkpoint_group = MagicMock(return_value=control_group)
    manager._vote_managed_phase = lambda phase, error, *_a, **_k: (
        PhaseError(phase, error) if error is not None else None
    )

    def run_phase(phase, operation, transaction):
        try:
            result, error = operation(), None
        except BaseException as caught:
            result, error = None, caught
        return result, manager._vote_managed_phase(phase, error, transaction)

    manager._run_managed_phase = run_phase
    queue.maybe_finalize_async_calls.side_effect = [[], RuntimeError("disk exploded")]
    queue.schedule_async_request.return_value = 9
    queue.call_idx = 8
    queue.get_num_unfinalized_calls.return_value = 1

    with (
        patch.object(manager, "generate_state_dict", return_value={"optimizer": {}}),
        patch.object(mod, "save_dist_checkpointing", return_value=object()),
        patch.object(mod, "begin_managed_async_checkpoint_save", return_value=(leaf,)),
        patch.object(mod, "bind_managed_async_checkpoint_request"),
        patch.object(mod, "fail_managed_async_checkpoint_save") as fail,
        patch("torch.distributed.get_rank", return_value=0),
    ):
        manager.save_checkpoint(
            str(checkpoint_path), async_completion_callback=released
        )
        with pytest.raises(RuntimeError, match="disk exploded"):
            manager.wait_async_saves()

    fail.assert_called_once()
    released.assert_called_once_with()
    assert manager.managed_async_save_state == "FAILED"
    assert (checkpoint_path / mod._MANAGED_ASYNC_INCOMPLETE).is_file()
    assert not (checkpoint_path / mod._MANAGED_ASYNC_COMPLETE).exists()


def test_load_blocks_on_pending_saves(patched_checkpointer, tmp_path):
    mod, manager, queue = patched_checkpointer
    queue.get_num_unfinalized_calls.return_value = 1

    with (
        patch("os.path.exists", return_value=True),
        patch.object(manager, "_validate_managed_async_load_marker"),
        patch.object(manager, "generate_state_dict", return_value={}),
        patch.object(mod, "load_dist_checkpointing", return_value={}),
    ):
        with pytest.raises((AssertionError, KeyError)):
            manager.load_checkpoint(str(tmp_path / "step0"))

    queue.maybe_finalize_async_calls.assert_called_with(blocking=True)


def test_ordinary_load_rejects_managed_async_incomplete_marker(
    patched_checkpointer, tmp_path
):
    mod, manager, _ = patched_checkpointer
    checkpoint = tmp_path / "incomplete"
    checkpoint.mkdir()
    (checkpoint / mod._MANAGED_ASYNC_INCOMPLETE).write_text("{}")

    with pytest.raises(RuntimeError, match="incomplete managed async save"):
        manager.load_checkpoint(str(checkpoint))


def test_managed_async_load_rejects_symlink_complete_marker(
    patched_checkpointer, tmp_path
):
    """A copied marker outside the owned checkpoint directory is not authority."""
    mod, manager, _ = patched_checkpointer
    checkpoint = tmp_path / "replaced-marker"
    checkpoint.mkdir()
    (checkpoint / "metadata.json").write_text("{}")
    external_marker = tmp_path / "external-marker.json"
    external_marker.write_text(
        '{"schema": 1, "checkpoint_id": "attacker", "path": "/other"}'
    )
    (checkpoint / mod._MANAGED_ASYNC_COMPLETE).symlink_to(external_marker)

    with pytest.raises((OSError, RuntimeError), match="marker|symlink|ownership"):
        manager._validate_managed_async_load_marker(str(checkpoint))


def _secure_marker_fixture(tmp_path):
    _import_checkpointer()
    marker = sys.modules["areal.engine.megatron_utils.managed_async_marker"]
    path = tmp_path / "secure-managed-async"
    leaves, digest = marker.canonical_ranked_leaf_identities(
        [{"rank": 0, "leaves": [{"tree_path": [], "identity": {"v": 1}}]}]
    )
    authority = marker.create_incomplete_marker(
        path=str(path),
        checkpoint_id="1" * 32,
        logical_call_id=3,
        mcore_async_call_index=7,
        participant_ranks=(0,),
        control_group_backend="gloo",
        managed_leaves=leaves,
        managed_leaves_digest=digest,
        mcore_version="0.17.0",
    )
    return marker, path, leaves, digest, authority


@pytest.mark.parametrize("marker_kind", ["complete", "incomplete", "temporary"])
def test_managed_async_load_rejects_symlink_marker_kind(tmp_path, marker_kind):
    _import_checkpointer()
    marker = sys.modules["areal.engine.megatron_utils.managed_async_marker"]
    checkpoint = tmp_path / f"symlink-{marker_kind}"
    checkpoint.mkdir()
    external = tmp_path / f"external-{marker_kind}"
    external.write_text("{}")
    names = {
        "complete": marker.MANAGED_ASYNC_COMPLETE,
        "incomplete": marker.MANAGED_ASYNC_INCOMPLETE,
        "temporary": f"{marker.MANAGED_ASYNC_TEMP_PREFIX}foreign",
    }
    (checkpoint / names[marker_kind]).symlink_to(external)

    with pytest.raises(RuntimeError, match="regular file|marker"):
        marker.validate_load_marker(
            path=str(checkpoint),
            participant_ranks=(0,),
            managed_leaves=[],
            managed_leaves_digest=marker.canonical_ranked_leaf_identities([])[1],
        )


@pytest.mark.parametrize("kind", ["directory", "fifo"])
def test_managed_async_load_rejects_nonregular_marker(tmp_path, kind):
    _import_checkpointer()
    marker = sys.modules["areal.engine.megatron_utils.managed_async_marker"]
    checkpoint = tmp_path / f"nonregular-{kind}"
    checkpoint.mkdir()
    target = checkpoint / marker.MANAGED_ASYNC_COMPLETE
    if kind == "directory":
        target.mkdir()
    else:
        os.mkfifo(target)

    with pytest.raises(RuntimeError, match="not a regular file"):
        marker.validate_load_marker(
            path=str(checkpoint),
            participant_ranks=(0,),
            managed_leaves=[],
            managed_leaves_digest=marker.canonical_ranked_leaf_identities([])[1],
        )


def test_managed_async_complete_publish_is_noreplace(tmp_path):
    marker, path, _leaves, _digest, authority = _secure_marker_fixture(tmp_path)
    (path / "metadata.json").write_text('{"ok": true}')
    external_payload = b"external-complete"
    complete = path / marker.MANAGED_ASYNC_COMPLETE
    fd = os.open(complete, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, external_payload)
    finally:
        os.close(fd)

    try:
        with pytest.raises((FileExistsError, RuntimeError)):
            marker.publish_complete_marker(authority)
        assert complete.read_bytes() == external_payload
        assert (path / marker.MANAGED_ASYNC_INCOMPLETE).is_file()
    finally:
        authority.close()


def test_managed_async_publish_rejects_replaced_checkpoint_directory(tmp_path):
    marker, path, _leaves, _digest, authority = _secure_marker_fixture(tmp_path)
    (path / "metadata.json").write_text("{}")
    original = tmp_path / "renamed-original"
    path.rename(original)
    path.mkdir()
    sentinel = path / "sentinel"
    sentinel.write_text("replacement")

    try:
        with pytest.raises(RuntimeError, match="replaced"):
            marker.publish_complete_marker(authority)
        assert sentinel.read_text() == "replacement"
        assert not (path / marker.MANAGED_ASYNC_COMPLETE).exists()
    finally:
        authority.close()


def test_managed_async_publish_link_failure_keeps_owned_incomplete(
    tmp_path, monkeypatch
):
    marker, path, _leaves, _digest, authority = _secure_marker_fixture(tmp_path)
    (path / "metadata.json").write_text("{}")

    monkeypatch.setattr(
        marker.os,
        "link",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("link fault")),
    )
    try:
        with pytest.raises(OSError, match="link fault"):
            marker.publish_complete_marker(authority)
        assert (path / marker.MANAGED_ASYNC_INCOMPLETE).is_file()
        assert not (path / marker.MANAGED_ASYNC_COMPLETE).exists()
        assert len(list(path.glob(f"{marker.MANAGED_ASYNC_TEMP_PREFIX}*"))) == 1
    finally:
        authority.close()


def test_managed_async_load_rejects_complete_while_incomplete_exists(tmp_path):
    """A prepared complete marker remains invisible until commit."""
    marker, path, leaves, digest, authority = _secure_marker_fixture(tmp_path)
    (path / "metadata.json").write_text("{}")
    try:
        marker.prepare_complete_marker(authority)
        assert (path / marker.MANAGED_ASYNC_COMPLETE).is_file()
        assert (path / marker.MANAGED_ASYNC_INCOMPLETE).is_file()
        with pytest.raises(RuntimeError, match="incomplete managed async save"):
            marker.validate_load_marker(
                path=str(path),
                participant_ranks=(0,),
                managed_leaves=leaves,
                managed_leaves_digest=digest,
            )
    finally:
        authority.close()


def test_managed_async_publish_recovers_unlink_after_effect(tmp_path, monkeypatch):
    marker, path, _leaves, _digest, authority = _secure_marker_fixture(tmp_path)
    (path / "metadata.json").write_text("{}")
    real_unlink = marker.os.unlink
    failed = False

    def unlink_then_raise(name, *, dir_fd):
        nonlocal failed
        real_unlink(name, dir_fd=dir_fd)
        if name == marker.MANAGED_ASYNC_INCOMPLETE and not failed:
            failed = True
            raise OSError("unlink completed before injected error")

    monkeypatch.setattr(marker.os, "unlink", unlink_then_raise)
    try:
        outcome = marker.publish_complete_marker(authority)
        assert outcome.committed is True
        assert not (path / marker.MANAGED_ASYNC_INCOMPLETE).exists()
        assert (path / marker.MANAGED_ASYNC_COMPLETE).is_file()
        assert any(
            "unlink committed before error" in item for item in authority.diagnostics
        )
        assert not list(path.glob(f"{marker.MANAGED_ASYNC_TEMP_PREFIX}*"))
    finally:
        authority.close()


def test_managed_async_publish_final_directory_fsync_failure_before_commit(
    tmp_path, monkeypatch
):
    """A failed final durability barrier must not expose a loadable checkpoint."""
    marker, path, leaves, digest, authority = _secure_marker_fixture(tmp_path)
    (path / "metadata.json").write_text("{}")
    real_fsync = marker.os.fsync
    directory_fsyncs = 0

    def fail_first_directory_fsync(fd):
        nonlocal directory_fsyncs
        if fd == authority.dir_fd:
            directory_fsyncs += 1
            if directory_fsyncs == 1:
                raise OSError("injected final directory fsync failure")
        return real_fsync(fd)

    monkeypatch.setattr(marker.os, "fsync", fail_first_directory_fsync)
    try:
        with pytest.raises(OSError, match="final directory fsync failure"):
            marker.publish_complete_marker(authority)
        with pytest.raises(RuntimeError, match="incomplete|temporary|marker"):
            marker.validate_load_marker(
                path=str(path),
                participant_ranks=(0,),
                managed_leaves=leaves,
                managed_leaves_digest=digest,
            )
    finally:
        authority.close()


def test_managed_async_publish_final_directory_fsync_failure_after_commit(
    tmp_path, monkeypatch
):
    """Post-commit fsync failure is cleanup-pending, never a failed save."""
    marker, path, leaves, digest, authority = _secure_marker_fixture(tmp_path)
    (path / "metadata.json").write_text("{}")
    real_fsync = marker.os.fsync
    directory_fsyncs = 0

    def fail_second_directory_fsync(fd):
        nonlocal directory_fsyncs
        if fd == authority.dir_fd:
            directory_fsyncs += 1
            if directory_fsyncs == 2:
                raise OSError("injected post-commit directory fsync failure")
        return real_fsync(fd)

    monkeypatch.setattr(marker.os, "fsync", fail_second_directory_fsync)
    outcome = marker.publish_complete_marker(authority)
    assert outcome.committed is True
    assert outcome.cleanup_pending is True
    assert marker.validate_load_marker(
        path=str(path),
        participant_ranks=(0,),
        managed_leaves=leaves,
        managed_leaves_digest=digest,
    )
    monkeypatch.setattr(marker.os, "fsync", real_fsync)
    assert marker.retry_post_commit_cleanup(authority) is True


def test_managed_async_publish_temp_unlink_after_effect_is_committed(
    tmp_path, monkeypatch
):
    """An unacknowledged temp unlink cannot make a failed publish loadable."""
    marker, path, leaves, digest, authority = _secure_marker_fixture(tmp_path)
    (path / "metadata.json").write_text("{}")
    real_unlink = marker.os.unlink
    failed = False

    def unlink_then_raise(name, *, dir_fd):
        nonlocal failed
        real_unlink(name, dir_fd=dir_fd)
        if name.startswith(marker.MANAGED_ASYNC_TEMP_PREFIX) and not failed:
            failed = True
            raise OSError("temp unlink completed before injected error")

    monkeypatch.setattr(marker.os, "unlink", unlink_then_raise)
    try:
        outcome = marker.publish_complete_marker(authority)
        assert outcome.committed is True
        assert marker.validate_load_marker(
            path=str(path),
            participant_ranks=(0,),
            managed_leaves=leaves,
            managed_leaves_digest=digest,
        )
        assert any(
            "unlink committed before error" in item for item in authority.diagnostics
        )
    finally:
        authority.close()


def test_managed_async_marker_rejects_symlink_parent_component(tmp_path):
    """Checkpoint creation must not traverse a symlinked parent authority."""
    _import_checkpointer()
    marker = sys.modules["areal.engine.megatron_utils.managed_async_marker"]
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    leaves, digest = marker.canonical_ranked_leaf_identities([])

    with pytest.raises((OSError, RuntimeError), match="symlink|parent|path"):
        marker.create_incomplete_marker(
            path=str(linked_parent / "checkpoint"),
            checkpoint_id="2" * 32,
            logical_call_id=1,
            mcore_async_call_index=0,
            participant_ranks=(0,),
            control_group_backend="gloo",
            managed_leaves=leaves,
            managed_leaves_digest=digest,
            mcore_version="0.17.0",
        )


def test_managed_async_marker_rejects_symlink_grandparent_component(tmp_path):
    _import_checkpointer()
    marker = sys.modules["areal.engine.megatron_utils.managed_async_marker"]
    real = tmp_path / "real-grandparent"
    (real / "parent").mkdir(parents=True)
    linked = tmp_path / "linked-grandparent"
    linked.symlink_to(real, target_is_directory=True)
    leaves, digest = marker.canonical_ranked_leaf_identities([])

    with pytest.raises(RuntimeError, match="real directory"):
        marker.create_incomplete_marker(
            path=str(linked / "parent" / "checkpoint"),
            checkpoint_id="3" * 32,
            logical_call_id=1,
            mcore_async_call_index=0,
            participant_ranks=(0,),
            control_group_backend="gloo",
            managed_leaves=leaves,
            managed_leaves_digest=digest,
            mcore_version="0.17.0",
        )


def test_managed_async_marker_rejects_parent_replacement_during_traversal(
    tmp_path, monkeypatch
):
    """Replacing a checked component before open must fail closed."""
    _import_checkpointer()
    marker = sys.modules["areal.engine.megatron_utils.managed_async_marker"]
    victim = tmp_path / "replace-during-open"
    victim.mkdir()
    leaves, digest = marker.canonical_ranked_leaf_identities([])
    real_open = marker.os.open
    replaced = False

    def replace_then_open(name, flags, *args, **kwargs):
        nonlocal replaced
        dir_fd = kwargs.get("dir_fd")
        if name == victim.name and dir_fd is not None and not replaced:
            replaced = True
            os.rename(
                victim.name,
                f"{victim.name}-old",
                src_dir_fd=dir_fd,
                dst_dir_fd=dir_fd,
            )
            os.mkdir(victim.name, 0o700, dir_fd=dir_fd)
        return real_open(name, flags, *args, **kwargs)

    monkeypatch.setattr(marker.os, "open", replace_then_open)
    with pytest.raises(RuntimeError, match="changed while opening"):
        marker.create_incomplete_marker(
            path=str(victim / "checkpoint"),
            checkpoint_id="4" * 32,
            logical_call_id=1,
            mcore_async_call_index=0,
            participant_ranks=(0,),
            control_group_backend="gloo",
            managed_leaves=leaves,
            managed_leaves_digest=digest,
            mcore_version="0.17.0",
        )


def test_managed_async_marker_short_fd_preclose_retries_without_leak(
    tmp_path, monkeypatch
):
    marker, path, leaves, digest, authority = _secure_marker_fixture(tmp_path)
    snapshot = sys.modules["areal.engine.megatron_utils.checkpoint_snapshot"]
    (path / "metadata.json").write_text("{}")
    real_prepare = snapshot._prepare_fd_close
    failures = 3
    attempts = 0
    baseline_fds = len(os.listdir("/proc/self/fd"))
    authority_fds = len(authority.directory.owners)

    def fail_regular_preclose(owner):
        nonlocal attempts
        if owner.file_type == stat.S_IFREG and attempts < failures:
            attempts += 1
            raise OSError("injected marker pre-close failure")
        return real_prepare(owner)

    monkeypatch.setattr(snapshot, "_prepare_fd_close", fail_regular_preclose)
    for _ in range(failures):
        with pytest.raises(OSError, match="pre-close"):
            marker.publish_complete_marker(authority)
        assert len(authority.short_fd_owners) == 1
        assert len(os.listdir("/proc/self/fd")) == baseline_fds + 1

    outcome = marker.publish_complete_marker(authority)
    assert outcome.committed is True
    assert not authority.short_fd_owners
    assert marker.validate_load_marker(
        path=str(path),
        participant_ranks=(0,),
        managed_leaves=leaves,
        managed_leaves_digest=digest,
    )
    assert len(os.listdir("/proc/self/fd")) == baseline_fds - authority_fds


def test_managed_async_marker_authority_close_failure_is_post_commit(
    tmp_path, monkeypatch
):
    marker, path, leaves, digest, authority = _secure_marker_fixture(tmp_path)
    snapshot = sys.modules["areal.engine.megatron_utils.checkpoint_snapshot"]
    (path / "metadata.json").write_text("{}")
    real_prepare = snapshot._prepare_fd_close
    fail = True

    def fail_directory_preclose(owner):
        if fail and owner.file_type == stat.S_IFDIR:
            raise OSError("injected authority close failure")
        return real_prepare(owner)

    monkeypatch.setattr(snapshot, "_prepare_fd_close", fail_directory_preclose)
    outcome = marker.publish_complete_marker(authority)
    assert outcome.committed is True
    assert outcome.cleanup_pending is True
    fail = False
    assert marker.validate_load_marker(
        path=str(path),
        participant_ranks=(0,),
        managed_leaves=leaves,
        managed_leaves_digest=digest,
    )

    assert marker.retry_post_commit_cleanup(authority) is True


def test_ordinary_marker_probe_does_not_query_default_world(
    patched_checkpointer, tmp_path
):
    _, manager, _ = patched_checkpointer
    checkpoint = tmp_path / "ordinary-no-marker"
    checkpoint.mkdir()
    with patch(
        "torch.distributed.get_world_size",
        side_effect=AssertionError("implicit WORLD access"),
    ):
        manager._validate_managed_async_load_marker(str(checkpoint))


def test_managed_async_marker_payload_binds_path_leaf_and_metadata(tmp_path):
    marker, path, leaves, digest, authority = _secure_marker_fixture(tmp_path)
    (path / "metadata.json").write_text('{"metadata": 1}')
    marker.publish_complete_marker(authority)
    authority.close()

    payload = marker.validate_load_marker(
        path=str(path),
        participant_ranks=(0,),
        managed_leaves=leaves,
        managed_leaves_digest=digest,
    )
    assert payload is not None
    assert payload["checkpoint_id"] == "1" * 32
    assert payload["logical_call_id"] == 3
    assert payload["mcore_async_call_index"] == 7
    assert payload["participant_ranks"] == [0]
    assert payload["managed_leaves"] == leaves
    assert stat.S_IMODE((path / marker.MANAGED_ASYNC_COMPLETE).stat().st_mode) == 0o600

    with pytest.raises(RuntimeError, match="leaf identity"):
        marker.validate_load_marker(
            path=str(path),
            participant_ranks=(0,),
            managed_leaves=[],
            managed_leaves_digest=marker.canonical_ranked_leaf_identities([])[1],
        )


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("checkpoint_id", "0" * 32),
        ("logical_call_id", 99),
        ("mcore_async_call_index", 99),
        ("checkpoint_path", "/wrong"),
        ("checkpoint_directory", {"device": 0, "inode": 0}),
        ("participant_world_size", 2),
        ("participant_ranks", [1]),
        ("control_group", {"backend": "nccl", "ranks": [0]}),
        ("managed_leaves", []),
        ("managed_leaves_digest", "0" * 64),
        ("mcore_version", "0.18.0"),
        ("backend", "other"),
        ("metadata", {"name": "metadata.json"}),
        ("request_digest", "0" * 64),
        ("version", 1),
    ],
)
def test_managed_async_marker_payload_v2_rejects_tampered_field(
    tmp_path, field, bad_value
):
    marker, path, leaves, digest, authority = _secure_marker_fixture(tmp_path)
    (path / "metadata.json").write_text('{"metadata": 1}')
    marker.publish_complete_marker(authority)
    authority.close()
    complete = path / marker.MANAGED_ASYNC_COMPLETE
    payload = json.loads(complete.read_text())
    payload[field] = bad_value
    complete.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))

    with pytest.raises(RuntimeError, match="marker|checkpoint|payload|metadata"):
        marker.validate_load_marker(
            path=str(path),
            participant_ranks=(0,),
            managed_leaves=leaves,
            managed_leaves_digest=digest,
        )


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_managed_async_marker_payload_v2_rejects_field_set_change(tmp_path, mutation):
    marker, path, leaves, digest, authority = _secure_marker_fixture(tmp_path)
    (path / "metadata.json").write_text('{"metadata": 1}')
    marker.publish_complete_marker(authority)
    authority.close()
    complete = path / marker.MANAGED_ASYNC_COMPLETE
    payload = json.loads(complete.read_text())
    if mutation == "missing":
        payload.pop("checkpoint_id")
    else:
        payload["unexpected"] = True
    complete.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))

    with pytest.raises(RuntimeError, match="fields mismatch"):
        marker.validate_load_marker(
            path=str(path),
            participant_ranks=(0,),
            managed_leaves=leaves,
            managed_leaves_digest=digest,
        )


def test_managed_async_load_rejects_checkpoint_directory_symlink(tmp_path):
    _import_checkpointer()
    marker = sys.modules["areal.engine.megatron_utils.managed_async_marker"]
    real = tmp_path / "real-checkpoint"
    real.mkdir()
    linked = tmp_path / "linked-checkpoint"
    linked.symlink_to(real, target_is_directory=True)

    with pytest.raises(RuntimeError, match="real directory"):
        marker.validate_load_marker(
            path=str(linked),
            participant_ranks=(0,),
            managed_leaves=[],
            managed_leaves_digest=marker.canonical_ranked_leaf_identities([])[1],
        )


def test_close_is_idempotent(patched_checkpointer):
    _, manager, queue = patched_checkpointer
    queue.get_num_unfinalized_calls.return_value = 0

    manager.close()
    manager.close()

    assert manager._async_queue is None


@pytest.mark.skip(
    reason="Fixture is not isolated across test files: if test_megatron_engine "
    "(or any test that imports MegatronEngine) runs first, "
    "areal.engine.megatron_utils.checkpointer is already cached in sys.modules, "
    "so _import_checkpointer's stub-installation branch (which mocks "
    "areal.utils.stats_tracker.scalar) is skipped. Tracked in a follow-up issue."
)
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


@pytest.mark.skip(
    reason="Fixture is not isolated across test files: if test_megatron_engine "
    "(or any test that imports MegatronEngine) runs first, "
    "areal.engine.megatron_utils.checkpointer is already cached in sys.modules, "
    "so _import_checkpointer's stub-installation branch (which mocks "
    "areal.utils.stats_tracker.scalar) is skipped. Tracked in a follow-up issue."
)
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


def test_generate_state_dict_requests_dp_reshardable_sharding(patched_checkpointer):
    _, manager, _ = patched_checkpointer

    with patch("torch.distributed.barrier"):
        state_dict = manager.generate_state_dict(
            with_model=False, with_optimizer=True, with_rng=False
        )

    kwargs = manager.optimizer.sharded_state_dict.call_args.kwargs
    assert kwargs["metadata"] == {"distrib_optim_sharding_type": "dp_reshardable"}
    assert kwargs["is_loading"] is False
    assert "optimizer" in state_dict


def test_load_checkpoint_builds_optimizer_template_with_is_loading(
    patched_checkpointer, tmp_path
):
    mod, manager, _ = patched_checkpointer

    with (
        patch("os.path.exists", return_value=True),
        patch.object(manager, "_validate_managed_async_load_marker"),
        patch("torch.distributed.barrier"),
        patch.object(
            mod, "load_dist_checkpointing", return_value={"optimizer": {"step": 1}}
        ),
    ):
        manager.load_checkpoint(
            str(tmp_path / "step0"),
            with_model=False,
            with_optimizer=True,
            with_rng=False,
        )

    kwargs = manager.optimizer.sharded_state_dict.call_args.kwargs
    assert kwargs["is_loading"] is True
    assert kwargs["metadata"] == {"distrib_optim_sharding_type": "dp_reshardable"}
    manager.optimizer.load_state_dict.assert_called_once_with({"step": 1})
