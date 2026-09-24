"""Unit tests for the Megatron checkpoint manager's async-save state machine.

These tests do NOT exercise real Megatron dist_checkpointing or distributed
process groups. They patch the queue and `save_dist_checkpointing` to verify
that the manager correctly:

- skips queue creation when async_save is False
- routes the AsyncRequest to AsyncCallsQueue.schedule_async_request when True
- finalizes completed saves non-blockingly on each new save call
- appends recovery publication after MCore's existing finalize callbacks
- blocks on load_checkpoint / close
- treats close as idempotent
- emits the async-only metric (queue_depth on schedule)
- emits no async metrics on the sync path
"""

from __future__ import annotations

import gc
import importlib
import importlib.util
import sys
import types
import weakref
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
    # load_checkpoint imports its lightweight optimizer-state helper lazily.
    sys.modules["areal.engine.megatron_utils"].__path__ = [
        str(repo_root / "areal" / "engine" / "megatron_utils")
    ]
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
    tensor_lists = [[]]

    with (
        patch.object(manager, "generate_state_dict", return_value={"model": {}}),
        patch.object(
            mod, "save_dist_checkpointing", return_value=fake_request
        ) as save_fn,
        patch.object(
            mod, "_inspect_retained_payload", return_value=tensor_lists
        ) as inspect_fn,
        patch.object(mod, "_release_retained_payload") as release_fn,
        patch("torch.cuda.empty_cache"),
        patch("torch.distributed.barrier"),
    ):
        manager.save_checkpoint(str(tmp_path / "step0"))

    save_fn.assert_called_once()
    assert save_fn.call_args.kwargs["async_save"] is True
    inspect_fn.assert_called_once_with(fake_request)
    queue.schedule_async_request.assert_called_once_with(fake_request)
    release_fn.assert_called_once_with(tensor_lists)


def test_async_save_publishes_only_from_finalize(patched_checkpointer, tmp_path):
    mod, manager, queue = patched_checkpointer
    fake_request = MagicMock()
    publish = MagicMock()

    with (
        patch.object(manager, "generate_state_dict", return_value={"model": {}}),
        patch.object(mod, "save_dist_checkpointing", return_value=fake_request),
        patch.object(mod, "_inspect_retained_payload", return_value=[]),
        patch("torch.cuda.empty_cache"),
        patch("torch.distributed.barrier"),
    ):
        manager.save_checkpoint(str(tmp_path / "step0"), finalize_fn=publish)

    publish.assert_not_called()
    fake_request.add_finalize_fn.assert_called_once()
    queue.schedule_async_request.assert_called_once_with(fake_request)

    appended_finalize = fake_request.add_finalize_fn.call_args.args[0]
    appended_finalize()

    publish.assert_called_once_with()


def test_publication_failure_is_broadcast_before_all_ranks_raise(
    patched_checkpointer,
):
    mod, _, _ = patched_checkpointer
    publish = MagicMock(side_effect=OSError("disk unavailable"))

    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_world_size", return_value=2),
        patch("torch.distributed.broadcast_object_list") as broadcast,
        pytest.raises(RuntimeError, match="disk unavailable"),
    ):
        mod._run_checkpoint_publication(0, publish)

    status = broadcast.call_args.args[0]
    assert status == [("OSError", "disk unavailable")]


def test_nonzero_rank_raises_publication_failure_received_from_rank0(
    patched_checkpointer,
):
    mod, _, _ = patched_checkpointer
    publish = MagicMock()

    def broadcast_rank0_failure(status, src):
        assert src == 0
        status[0] = ("OSError", "disk unavailable")

    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_world_size", return_value=2),
        patch(
            "torch.distributed.broadcast_object_list",
            side_effect=broadcast_rank0_failure,
        ),
        pytest.raises(RuntimeError, match="disk unavailable"),
    ):
        mod._run_checkpoint_publication(1, publish)

    publish.assert_not_called()


def test_save_reaps_before_scheduling_next(patched_checkpointer, tmp_path):
    mod, manager, queue = patched_checkpointer

    with (
        patch.object(manager, "generate_state_dict", return_value={"model": {}}),
        patch.object(mod, "save_dist_checkpointing", side_effect=["r1", "r2"]),
        patch.object(mod, "_inspect_retained_payload", return_value=[]),
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
        patch.object(mod, "_inspect_retained_payload", return_value=[]),
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


class _FakeRequest:
    def __init__(self, args, preload_fn, finalize_fns):
        self.async_fn_args = args
        self.preload_fn = preload_fn
        self.finalize_fns = finalize_fns


def _request_with_pending_payload(buckets, finalize_fns=None):
    return _FakeRequest(
        args=(0, buckets, "results_queue"),
        preload_fn=lambda: buckets,
        finalize_fns=finalize_fns or [lambda: None],
    )


def test_release_retained_payload_clears_tensor_data_and_preserves_finalize():
    mod = _import_checkpointer()

    class Payload:
        pass

    tensor = Payload()
    tensor_ref = weakref.ref(tensor)
    writer = types.SimpleNamespace(
        write_buckets=[
            (
                "file",
                "key",
                ([("byte-item", b"payload")], [("tensor-item", tensor)]),
            )
        ]
    )
    buckets = writer.write_buckets

    def finalize_fn():
        return len(writer.write_buckets), writer.write_buckets[0][2][0]

    request = _request_with_pending_payload(buckets, [finalize_fn])
    tensor_lists = mod._inspect_retained_payload(request)
    del tensor

    mod._release_retained_payload(tensor_lists)
    gc.collect()

    assert tensor_ref() is None
    assert writer.write_buckets is buckets
    assert writer.write_buckets == [("file", "key", ([("byte-item", b"payload")], []))]
    assert request.async_fn_args == (0, buckets, "results_queue")
    assert request.preload_fn() is buckets
    assert request.finalize_fns == [finalize_fn]
    assert finalize_fn() == (1, [("byte-item", b"payload")])


def test_save_unknown_payload_layout_fails_before_scheduling(
    patched_checkpointer, tmp_path
):
    mod, manager, queue = patched_checkpointer
    buckets = ["opaque-bucket"]
    request = _request_with_pending_payload(buckets)

    with (
        patch.object(manager, "generate_state_dict", return_value={"model": {}}),
        patch.object(mod, "save_dist_checkpointing", return_value=request),
        patch.object(mod, "_mcore_version", return_value="0.18.0"),
        pytest.raises(
            mod._UnsupportedMCoreAsyncLayout,
            match=r"megatron-core=0\.18\.0.*write_buckets\[0\]",
        ),
    ):
        manager.save_checkpoint(str(tmp_path / "step0"))

    queue.schedule_async_request.assert_not_called()
    assert buckets == ["opaque-bucket"]


def test_sync_save_releases_state_before_host_cleanup(
    patched_checkpointer, monkeypatch, tmp_path
):
    mod, manager, _ = patched_checkpointer
    manager.async_save = False
    manager._async_queue = None
    references = []
    events = []

    class Payload:
        pass

    def generate(*args):
        payload = Payload()
        references.append(weakref.ref(payload))
        return {"model": payload}

    def save(**kwargs):
        assert kwargs["sharded_state_dict"]["model"] is references[0]()
        events.append("saved")

    def cleanup(rank):
        assert rank == manager.rank
        assert references[0]() is None
        events.append("cleaned")

    monkeypatch.setattr(manager, "generate_state_dict", generate)
    monkeypatch.setattr(mod, "save_dist_checkpointing", save)
    monkeypatch.setattr(mod, "_release_cached_host_memory", cleanup)
    monkeypatch.setattr(mod.torch.distributed, "barrier", lambda: None)

    manager.save_checkpoint(
        str(tmp_path / "step0"), finalize_fn=lambda: events.append("published")
    )

    assert events == ["saved", "published", "cleaned"]


@pytest.mark.parametrize("blocking", [False, True])
def test_async_host_cleanup_preserves_pending_and_follows_completed_payload(
    patched_checkpointer, monkeypatch, tmp_path, blocking
):
    mod, manager, queue = patched_checkpointer
    holder = []
    references = []
    cleanup_liveness = []
    completed = False

    class Payload:
        pass

    def schedule(request):
        payload = Payload()
        holder.append(payload)
        references.append(weakref.ref(payload))
        return 0

    def finalize(*, blocking):
        if not completed:
            return []
        holder.clear()
        return [0]

    monkeypatch.setattr(manager, "generate_state_dict", lambda *args: {})
    monkeypatch.setattr(
        mod,
        "save_dist_checkpointing",
        lambda **kwargs: _request_with_pending_payload([]),
    )
    monkeypatch.setattr(queue, "schedule_async_request", schedule)
    monkeypatch.setattr(queue, "maybe_finalize_async_calls", finalize)
    monkeypatch.setattr(
        mod,
        "_release_cached_host_memory",
        lambda rank: cleanup_liveness.append(references[0]() is not None),
    )

    manager.save_checkpoint(str(tmp_path / "step0"))
    # Scheduling must not wait for the writer or clear its CPU holder.
    assert cleanup_liveness == [True]
    manager._reap_finished_async_saves()
    assert cleanup_liveness == [True]
    assert len(holder) == 1

    completed = True
    if blocking:
        manager.wait_async_saves()
    else:
        manager._reap_finished_async_saves()

    assert cleanup_liveness == [True, False]
    assert references[0]() is None


@pytest.mark.parametrize("device", ["cpu", "npu", "cuda"])
def test_host_cleanup_requires_cuda_and_callable_binding(monkeypatch, device):
    mod = _import_checkpointer()
    empty_cache = MagicMock()
    monkeypatch.setattr(mod, "get_device_name", lambda: device)
    monkeypatch.setattr(mod.torch._C, "_host_emptyCache", empty_cache, raising=False)
    mod._release_cached_host_memory(0)
    assert empty_cache.call_count == int(device == "cuda")
    monkeypatch.delattr(mod.torch._C, "_host_emptyCache")
    mod._release_cached_host_memory(0)
    monkeypatch.setattr(mod.torch._C, "_host_emptyCache", None, raising=False)
    mod._release_cached_host_memory(0)


def test_host_cleanup_stats_failure_does_not_prevent_reclamation(monkeypatch):
    mod = _import_checkpointer()
    empty_cache = MagicMock()
    monkeypatch.setattr(mod, "get_device_name", lambda: "cuda")
    monkeypatch.setattr(mod.torch._C, "_host_emptyCache", empty_cache, raising=False)
    monkeypatch.setattr(
        mod.torch.cuda,
        "host_memory_stats",
        MagicMock(side_effect=RuntimeError("stats unavailable")),
        raising=False,
    )

    mod._release_cached_host_memory(0)

    empty_cache.assert_called_once_with()


def test_host_cleanup_failure_preserves_published_save(
    patched_checkpointer, monkeypatch, tmp_path
):
    mod, manager, _ = patched_checkpointer
    manager.async_save = False
    manager._async_queue = None
    publish = MagicMock()
    empty_cache = MagicMock(side_effect=RuntimeError("cleanup unavailable"))
    monkeypatch.setattr(mod, "get_device_name", lambda: "cuda")
    monkeypatch.setattr(mod.torch._C, "_host_emptyCache", empty_cache, raising=False)
    monkeypatch.setattr(manager, "generate_state_dict", lambda *args: {})
    monkeypatch.setattr(mod, "save_dist_checkpointing", lambda **kwargs: None)
    monkeypatch.setattr(mod.torch.distributed, "barrier", lambda: None)

    manager.save_checkpoint(str(tmp_path / "step0"), finalize_fn=publish)

    publish.assert_called_once_with()
    empty_cache.assert_called_once_with()
