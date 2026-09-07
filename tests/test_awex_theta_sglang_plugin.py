"""CPU fakes for the Theta SGLang AWEX integration points."""

import sys
from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace

import pytest

from areal.engine.awex.sglang_plugin import (
    AwexSchedulerPlugin,
    _is_retract_paused,
    _patch_flush_cache_for_retract_pause,
    _patch_pause_mode_tracking,
    _patch_release_memory_for_retract_pause,
    _patch_theta_memory_transitions,
)


class _RunningBatch:
    def __init__(self, empty: bool = True) -> None:
        self._empty = empty

    def is_empty(self) -> bool:
        return self._empty


def test_modern_theta_hook_preserves_scheduler_event_loops():
    """The modern hook must not replace Theta's evolving scheduler loops."""
    calls = []

    def process_input_requests(requests):
        calls.append(("input", requests))

    overlap_loop = object()
    normal_loop = object()
    scheduler = SimpleNamespace(
        ps=SimpleNamespace(tp_rank=0, tp_size=1, gpu_id=0),
        request_receiver=object(),
        process_input_requests=process_input_requests,
        event_loop_overlap=overlap_loop,
        event_loop_normal=normal_loop,
        _engine_paused=True,
    )
    plugin = AwexSchedulerPlugin(scheduler)
    plugin.process_awex_queue = lambda: calls.append(("awex", None))
    plugin._paused_poll_interval_s = 0

    plugin._patch_event_loop()
    scheduler.process_input_requests(["pause"])

    assert calls == [("input", ["pause"]), ("awex", None)]
    assert scheduler.event_loop_overlap is overlap_loop
    assert scheduler.event_loop_normal is normal_loop


def test_retract_pause_flush_temporarily_relaxes_idle_gate(monkeypatch):
    """Retracted waiters permit one cache flush and retain their queue state."""
    scheduler_module = ModuleType("sglang.srt.managers.scheduler")

    class Scheduler:
        _engine_paused = True
        _areal_pause_mode = "retract"

        def __init__(self) -> None:
            self.running_batch = _RunningBatch()
            self.waiting_queue = [object()]

        def is_fully_idle(self) -> bool:
            return not self.waiting_queue

        def flush_cache(self) -> bool:
            return self.is_fully_idle()

    scheduler_module.Scheduler = Scheduler
    monkeypatch.setitem(sys.modules, "sglang.srt.managers.scheduler", scheduler_module)

    _patch_flush_cache_for_retract_pause()

    scheduler = Scheduler()
    assert scheduler.flush_cache() is True
    assert scheduler.is_fully_idle() is False


def test_retract_pause_release_temporarily_relaxes_idle_gate(monkeypatch):
    """The weight updater may release memory while requests are retracted."""
    updater_module = ModuleType(
        "sglang.srt.managers.scheduler_components.weight_updater"
    )

    class SchedulerWeightUpdaterManager:
        def __init__(self, scheduler) -> None:
            self.scheduler = scheduler
            self.is_fully_idle = scheduler.is_fully_idle

        def release_memory_occupation(self, request):
            assert self.is_fully_idle()
            return request

    updater_module.SchedulerWeightUpdaterManager = SchedulerWeightUpdaterManager
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.managers.scheduler_components.weight_updater",
        updater_module,
    )
    scheduler = SimpleNamespace(
        _engine_paused=True,
        _areal_pause_mode="retract",
        running_batch=_RunningBatch(),
        waiting_queue=[object()],
    )
    scheduler.is_fully_idle = lambda: not scheduler.waiting_queue
    manager = SchedulerWeightUpdaterManager(scheduler)
    request = object()

    _patch_release_memory_for_retract_pause()

    assert manager.release_memory_occupation(request) is request
    assert manager.is_fully_idle() is False
    assert scheduler.is_fully_idle() is False


def test_retract_pause_release_keeps_nonpaused_assert(monkeypatch):
    """The compatibility patch does not weaken the normal idle assertion."""
    updater_module = ModuleType(
        "sglang.srt.managers.scheduler_components.weight_updater"
    )

    class SchedulerWeightUpdaterManager:
        def __init__(self, scheduler) -> None:
            self.scheduler = scheduler
            self.is_fully_idle = scheduler.is_fully_idle

        def release_memory_occupation(self, request):
            assert self.is_fully_idle()
            return request

    updater_module.SchedulerWeightUpdaterManager = SchedulerWeightUpdaterManager
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.managers.scheduler_components.weight_updater",
        updater_module,
    )
    scheduler = SimpleNamespace(
        _engine_paused=False,
        _areal_pause_mode="retract",
        running_batch=_RunningBatch(),
        waiting_queue=[object()],
        is_fully_idle=lambda: False,
    )
    manager = SchedulerWeightUpdaterManager(scheduler)

    _patch_release_memory_for_retract_pause()

    with pytest.raises(AssertionError):
        manager.release_memory_occupation(object())


@pytest.mark.parametrize("mode", ["in_place", None])
def test_idle_gate_requires_successful_retract_pause(mode):
    """An in-place pause must retain all native cache-release protections."""
    scheduler = SimpleNamespace(
        _engine_paused=True,
        _areal_pause_mode=mode,
        waiting_queue=[object()],
        is_fully_idle=lambda: True,
    )
    assert not _is_retract_paused(scheduler)


def test_idle_gate_preserves_pending_gpu_work_and_queue_on_error():
    """Hiding waiters must not hide a chunked request or overlap result."""
    waiters = [object()]
    scheduler = SimpleNamespace(
        _engine_paused=True,
        _areal_pause_mode="retract",
        waiting_queue=waiters,
        pending_gpu_work=True,
    )
    scheduler.is_fully_idle = lambda: (
        not scheduler.waiting_queue and not scheduler.pending_gpu_work
    )
    assert not _is_retract_paused(scheduler)
    assert scheduler.waiting_queue is waiters
    scheduler.pending_gpu_work = False
    assert _is_retract_paused(scheduler)
    assert scheduler.waiting_queue is waiters

    def broken_idle_check():
        raise RuntimeError("native idle check failed")

    scheduler.is_fully_idle = broken_idle_check
    with pytest.raises(RuntimeError, match="native idle check failed"):
        _is_retract_paused(scheduler)
    assert scheduler.waiting_queue is waiters


def test_pause_mode_tracking_is_idempotent_and_clears_failed_pause(monkeypatch):
    """Record mode only after native pause succeeds; do not stack wrappers."""
    scheduler_module = ModuleType("sglang.srt.managers.scheduler")
    calls = []

    class Scheduler:
        def pause_generation(self, request):
            calls.append(request.mode)
            if request.mode == "invalid":
                raise ValueError("invalid mode")

    scheduler_module.Scheduler = Scheduler
    monkeypatch.setitem(sys.modules, "sglang.srt.managers.scheduler", scheduler_module)
    _patch_pause_mode_tracking()
    _patch_pause_mode_tracking()
    scheduler = Scheduler()
    scheduler.pause_generation(SimpleNamespace(mode="retract"))
    assert scheduler._areal_pause_mode == "retract"
    with pytest.raises(ValueError):
        scheduler.pause_generation(SimpleNamespace(mode="invalid"))
    assert scheduler._areal_pause_mode is None
    assert calls == ["retract", "invalid"]


@pytest.mark.parametrize("modern", [False, True])
def test_reader_memory_calls_support_both_scheduler_layouts(monkeypatch, modern):
    """Actual reader release/resume routes to the owner and tracks tags."""
    from areal.engine.awex.colocate_reader import AwexColocateReader

    io_module = ModuleType("sglang.srt.managers.io_struct")
    io_module.ReleaseMemoryOccupationReqInput = SimpleNamespace
    io_module.ResumeMemoryOccupationReqInput = SimpleNamespace
    monkeypatch.setitem(sys.modules, "sglang.srt.managers.io_struct", io_module)
    calls = []
    owner = SimpleNamespace(
        release_memory_occupation=lambda req: calls.append(("release", req.tags)),
        resume_memory_occupation=lambda req: calls.append(("resume", req.tags)),
    )
    scheduler = SimpleNamespace(weight_updater=owner) if modern else owner
    reader = AwexColocateReader(scheduler)
    reader.release_memory(["kv_cache"])
    reader.release_memory(["kv_cache"])
    reader.resume_memory(["kv_cache"])
    reader.resume_memory(["kv_cache"])
    assert calls == [("release", ["kv_cache"]), ("resume", ["kv_cache"])]


def test_queue_update_resumes_modern_manager_before_receiving(monkeypatch):
    """The real queue consumer calls manager.resume, then receives and signals."""
    io_module = ModuleType("sglang.srt.managers.io_struct")
    io_module.ResumeMemoryOccupationReqInput = SimpleNamespace
    monkeypatch.setitem(sys.modules, "sglang.srt.managers.io_struct", io_module)
    calls = []
    scheduler = SimpleNamespace(
        ps=SimpleNamespace(tp_rank=0, tp_size=1, gpu_id=0),
        tp_cpu_group=object(),
        weight_updater=SimpleNamespace(
            resume_memory_occupation=lambda req: calls.append(("resume", req.tags)),
        ),
    )
    plugin = AwexSchedulerPlugin(scheduler)
    plugin._receiver = SimpleNamespace(
        wait_for_training_offloaded=lambda version: calls.append(("wait", version)),
        update_weights=lambda version: calls.append(("receive", version)),
        signal_finished_weights_update=lambda: calls.append(("done", None)),
    )
    plugin._weight_queue.put({"version": 1})
    plugin.process_awex_queue()
    assert calls == [
        ("wait", 1),
        ("resume", ["weights"]),
        ("receive", 1),
        ("done", None),
    ]
    assert plugin._version == 1


@pytest.fixture
def modern_memory_manager(monkeypatch):
    """Match the slots manager, all-memory tags and typed dispatcher replies."""
    updater_module = ModuleType(
        "sglang.srt.managers.scheduler_components.weight_updater"
    )
    io_module = ModuleType("sglang.srt.managers.io_struct")
    constants_module = ModuleType("sglang.srt.constants")
    all_tags = ["kv_cache", "weights", "cuda_graph"]
    constants_module.GPU_MEMORY_ALL_TYPES = all_tags

    class ReleaseOutput:
        pass

    class ResumeOutput:
        pass

    @dataclass(slots=True)
    class Manager:
        scheduler: object = None
        is_fully_idle: object = lambda: True
        offload_tags: set = field(default_factory=set)
        calls: list = field(default_factory=list)
        fail: bool = False

        def release_memory_occupation(self, request):
            assert self.is_fully_idle()
            if self.fail:
                raise RuntimeError("native release failed")
            tags = request.tags or all_tags
            self.calls.append(("release", list(tags)))
            self.offload_tags.update(tags)
            return ReleaseOutput()

        def resume_memory_occupation(self, request):
            if self.fail:
                raise RuntimeError("native resume failed")
            tags = request.tags or all_tags
            self.calls.append(("resume", list(tags)))
            for tag in tags:
                self.offload_tags.remove(tag)
            return ResumeOutput()

    updater_module.SchedulerWeightUpdaterManager = Manager
    io_module.ReleaseMemoryOccupationReqOutput = ReleaseOutput
    io_module.ResumeMemoryOccupationReqOutput = ResumeOutput
    monkeypatch.setitem(sys.modules, updater_module.__name__, updater_module)
    monkeypatch.setitem(sys.modules, io_module.__name__, io_module)
    monkeypatch.setitem(sys.modules, constants_module.__name__, constants_module)
    monkeypatch.setenv("AREAL_SGLANG_FORK", "theta")
    return SimpleNamespace(
        cls=Manager,
        all_tags=all_tags,
        release_output=ReleaseOutput,
        resume_output=ResumeOutput,
    )


def test_modern_memory_retries_use_captured_slots_callbacks(modern_memory_manager):
    runtime = modern_memory_manager
    _patch_theta_memory_transitions()
    first_wrapper = runtime.cls.resume_memory_occupation
    _patch_theta_memory_transitions()
    assert runtime.cls.resume_memory_occupation is first_wrapper
    manager = runtime.cls()
    assert not hasattr(manager, "__dict__")
    dispatch = {
        "release": manager.release_memory_occupation,
        "resume": manager.resume_memory_occupation,
    }
    request = SimpleNamespace(tags=["weights"])
    assert isinstance(dispatch["release"](request), runtime.release_output)
    assert isinstance(dispatch["release"](request), runtime.release_output)
    assert isinstance(dispatch["resume"](request), runtime.resume_output)
    assert isinstance(dispatch["resume"](request), runtime.resume_output)
    assert manager.calls == [("release", ["weights"]), ("resume", ["weights"])]
    assert manager.offload_tags == set()


@pytest.mark.parametrize("tags", [None, []])
def test_modern_memory_empty_tags_preserve_all_regions(modern_memory_manager, tags):
    runtime = modern_memory_manager
    _patch_theta_memory_transitions()
    manager = runtime.cls(offload_tags={"weights"})
    request = SimpleNamespace(tags=tags)
    manager.release_memory_occupation(request)
    assert manager.offload_tags == set(runtime.all_tags)
    assert manager.calls == [("release", ["kv_cache", "cuda_graph"])]
    assert request.tags is tags
    assert isinstance(
        manager.release_memory_occupation(request), runtime.release_output
    )
    manager.resume_memory_occupation(request)
    assert manager.offload_tags == set()
    assert manager.calls[-1] == ("resume", runtime.all_tags)
    assert isinstance(manager.resume_memory_occupation(request), runtime.resume_output)
    assert len(manager.calls) == 2


def test_modern_memory_mixed_tags_do_not_mutate_request(modern_memory_manager):
    runtime = modern_memory_manager
    _patch_theta_memory_transitions()
    manager = runtime.cls(offload_tags={"weights"})
    request = SimpleNamespace(tags=["weights", "kv_cache"], rid="request-id")
    manager.release_memory_occupation(request)
    manager.offload_tags.remove("weights")
    manager.resume_memory_occupation(request)
    assert manager.calls == [("release", ["kv_cache"]), ("resume", ["kv_cache"])]
    assert request.tags == ["weights", "kv_cache"]
    assert request.rid == "request-id"


@pytest.mark.parametrize("release", [False, True])
def test_modern_memory_preserves_native_errors(modern_memory_manager, release):
    runtime = modern_memory_manager
    _patch_theta_memory_transitions()
    manager = runtime.cls(offload_tags=set() if release else {"weights"}, fail=True)
    method = (
        manager.release_memory_occupation
        if release
        else manager.resume_memory_occupation
    )
    with pytest.raises(RuntimeError, match="native .* failed"):
        method(SimpleNamespace(tags=["weights"]))


def test_modern_memory_retries_compose_with_retract_release(modern_memory_manager):
    runtime = modern_memory_manager
    scheduler = SimpleNamespace(
        _engine_paused=True,
        _areal_pause_mode="retract",
        waiting_queue=[object()],
    )
    scheduler.is_fully_idle = lambda: not scheduler.waiting_queue
    _patch_release_memory_for_retract_pause()
    _patch_theta_memory_transitions()
    manager = runtime.cls(scheduler=scheduler, is_fully_idle=scheduler.is_fully_idle)
    request = SimpleNamespace(tags=["kv_cache"])
    assert isinstance(
        manager.release_memory_occupation(request), runtime.release_output
    )
    assert isinstance(
        manager.release_memory_occupation(request), runtime.release_output
    )
    assert manager.calls == [("release", ["kv_cache"])]
    assert len(scheduler.waiting_queue) == 1
    assert not manager.is_fully_idle()
    assert not scheduler.is_fully_idle()


def test_modern_memory_patch_requires_theta(monkeypatch, modern_memory_manager):
    runtime = modern_memory_manager
    original = runtime.cls.resume_memory_occupation
    monkeypatch.delenv("AREAL_SGLANG_FORK")
    _patch_theta_memory_transitions()
    assert runtime.cls.resume_memory_occupation is original


def test_modern_memory_patch_accepts_legacy_scheduler(monkeypatch):
    monkeypatch.setenv("AREAL_SGLANG_FORK", "theta")
    monkeypatch.setitem(
        sys.modules, "sglang.srt.managers.scheduler_components.weight_updater", None
    )
    _patch_theta_memory_transitions()


def test_scheduler_entry_patches_memory_before_dispatcher_capture(
    monkeypatch, modern_memory_manager
):
    import areal.engine.awex.sglang_plugin as plugin_module

    runtime = modern_memory_manager
    monkeypatch.setenv("AWEX_META_SERVER_ADDR", "localhost:1234")
    for name in (
        "register_awex_plugin",
        "_patch_pause_mode_tracking",
        "_patch_release_memory_for_retract_pause",
        "_patch_flush_cache_for_retract_pause",
    ):
        monkeypatch.setattr(plugin_module, name, lambda: None)

    scheduler_module = ModuleType("sglang.srt.managers.scheduler")

    def run_scheduler_process():
        manager = runtime.cls(offload_tags={"weights"})
        # Modern Scheduler stores this bound method in TypeBasedDispatcher.
        captured_resume = manager.resume_memory_occupation
        request = SimpleNamespace(tags=["weights"])
        assert isinstance(captured_resume(request), runtime.resume_output)
        assert isinstance(captured_resume(request), runtime.resume_output)
        assert manager.calls == [("resume", ["weights"])]
        return "scheduler completed"

    scheduler_module.run_scheduler_process = run_scheduler_process
    monkeypatch.setitem(sys.modules, scheduler_module.__name__, scheduler_module)
    assert plugin_module.awex_run_scheduler_process() == "scheduler completed"
