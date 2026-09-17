from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from areal.engine.awex.colocate_reader import (
    AwexColocateReader,
    NCCLWorkerWeightsReader,
    _DeviceBoundWeightsReader,
)
from areal.engine.awex.sglang_plugin import AwexSchedulerPlugin


@pytest.mark.parametrize(
    "visible,logical", [(None, 7), ("0,1,2,3,4,5,6,7", 6), ("7", 0), ("7,4,6,5", 1)]
)
def test_reader_device_with_visible_remapping_matches_model(
    monkeypatch, visible, logical
):
    """Device selection must not fall back to LOCAL_RANK or physical ids."""
    import torch

    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    monkeypatch.setenv("LOCAL_RANK", "0")
    device = torch.device("cuda", logical)
    model = NS(parameters=lambda: iter([NS(device=device)]))
    monkeypatch.setattr(NCCLWorkerWeightsReader, "__init__", lambda *a, **kw: None)
    set_device, tensor = Mock(), Mock()
    monkeypatch.setattr(torch.cuda, "set_device", set_device)
    monkeypatch.setattr(torch, "tensor", tensor)
    reader = _DeviceBoundWeightsReader(model=model)
    reader.transfer_rank = 23

    reader._set_device()

    set_device.assert_called_once_with(device)
    tensor.assert_called_once_with(1, device=device)
    assert reader.barrier_device == logical
    assert reader.backend == "nccl"
    assert reader.ready_tensor is tensor.return_value


def test_reader_device_with_cpu_weights_fails_before_initialization(monkeypatch):
    """Do not silently select GPU zero when weights have not been resumed."""
    import torch

    model = NS(parameters=lambda: iter([NS(device=torch.device("cpu"))]))
    initialize = Mock()
    monkeypatch.setattr(NCCLWorkerWeightsReader, "__init__", initialize)

    with pytest.raises(RuntimeError, match="weights resumed on CUDA"):
        _DeviceBoundWeightsReader(model=model)

    initialize.assert_not_called()


@pytest.mark.parametrize("location", ["scheduler", "ps", "tp_worker", "instance"])
def test_model_context_resolves_nonzero_tp_rank(location):
    scheduler = NS(server_args=NS(tp_size=2, pp_size=1, dp_size=1))
    if location == "scheduler":
        scheduler.tp_rank = 1
    elif location != "instance":
        setattr(scheduler, location, NS(tp_rank=1))
    reader = AwexColocateReader(scheduler)
    if location == "instance":
        reader._instance_local_rank = 1
    context = reader._build_model_context()
    assert context["tp_rank"] == context["attn_tp_rank"] == 1
    assert context["global_rank"] == 1


def test_model_context_rejects_missing_multi_tp_rank():
    reader = AwexColocateReader(NS(server_args=NS(tp_size=2, pp_size=1)))
    with pytest.raises(RuntimeError, match="TP rank"):
        reader._build_model_context()


class Manager:
    __slots__ = (
        "is_fully_idle",
        "flush_cache",
        "offload_tags",
        "calls",
        "fail_release",
    )

    def __init__(self, scheduler):
        self.is_fully_idle = scheduler.is_fully_idle
        self.flush_cache = scheduler.flush_cache
        self.offload_tags = set()
        self.calls = []
        self.fail_release = False

    def release_memory_occupation(self, request):
        assert self.is_fully_idle(), "busy"
        if self.fail_release:
            raise ValueError("release failed")
        assert self.flush_cache()
        self.calls.append("release")
        self.offload_tags.update(request.tags)

    def resume_memory_occupation(self, request):
        self.calls.append("resume")
        for tag in request.tags:
            self.offload_tags.remove(tag)


class Scheduler:
    def __init__(self):
        self._engine_paused = True
        self.waiting_queue = [object()]
        self.active = False
        self.flushes = 0
        self.weight_updater = Manager(self)
        self._request_dispatcher = NS(
            _mapping={
                "release": self.weight_updater.release_memory_occupation,
                "resume": self.weight_updater.resume_memory_occupation,
                "flush": self.flush_cache,
            }
        )

    def is_fully_idle(self):
        return not self.active and not self.waiting_queue

    def flush_cache(self):
        if not self.is_fully_idle():
            return False
        self.flushes += 1
        return True


def test_manager_dispatch_preserves_parked_requests_and_idempotency():
    from sglang.srt.managers.io_struct import (
        ReleaseMemoryOccupationReqOutput,
        ResumeMemoryOccupationReqOutput,
    )

    scheduler = Scheduler()
    parked = scheduler.waiting_queue
    original_idle = scheduler.weight_updater.is_fully_idle
    original_release = scheduler.weight_updater.release_memory_occupation
    AwexSchedulerPlugin(scheduler)._patch_memory_transitions()
    request = NS(tags=["kv_cache"])
    callbacks = scheduler._request_dispatcher._mapping
    callbacks["release"](request)
    assert isinstance(callbacks["release"](request), ReleaseMemoryOccupationReqOutput)
    assert scheduler.flush_cache()
    callbacks["resume"](request)
    assert isinstance(callbacks["resume"](request), ResumeMemoryOccupationReqOutput)
    assert scheduler.weight_updater.calls == ["release", "resume"]
    assert scheduler.flushes == 2
    assert scheduler.waiting_queue is parked
    assert scheduler.weight_updater.is_fully_idle == original_idle
    assert scheduler.weight_updater.release_memory_occupation == original_release
    assert not scheduler.is_fully_idle()
    assert "is_fully_idle" not in vars(scheduler)


@pytest.mark.parametrize("paused,active", [(False, False), (True, True)])
def test_memory_release_does_not_bypass_active_work(paused, active):
    scheduler = Scheduler()
    scheduler._engine_paused = paused
    scheduler.active = active
    parked = scheduler.waiting_queue
    AwexSchedulerPlugin(scheduler)._patch_memory_transitions()
    with pytest.raises(AssertionError, match="busy"):
        scheduler.release_memory_occupation(NS(tags=["weights"]))
    assert not scheduler.flush_cache()
    assert scheduler.waiting_queue is parked
    assert not scheduler.weight_updater.offload_tags


def test_idle_gate_restored_when_release_raises():
    scheduler = Scheduler()
    original_idle = scheduler.weight_updater.is_fully_idle

    scheduler.weight_updater.fail_release = True
    AwexSchedulerPlugin(scheduler)._patch_memory_transitions()
    with pytest.raises(ValueError, match="release failed"):
        scheduler.release_memory_occupation(NS(tags=["weights"]))
    assert scheduler.weight_updater.is_fully_idle == original_idle
    assert not scheduler.is_fully_idle()
    assert "is_fully_idle" not in vars(scheduler)


def test_native_slotted_weight_manager_release_and_resume(monkeypatch):
    import torch

    module = pytest.importorskip(
        "sglang.srt.managers.scheduler_components.weight_updater"
    )
    native = object.__new__(module.SchedulerWeightUpdaterManager)
    scheduler = Scheduler()
    native.scheduler = None
    native.is_fully_idle = scheduler.is_fully_idle
    native.flush_cache = scheduler.flush_cache
    native.offload_tags = set()
    native.memory_saver_adapter = Mock()
    scheduler.weight_updater = native
    scheduler._request_dispatcher = NS(
        _mapping={
            "release": native.release_memory_occupation,
            "resume": native.resume_memory_occupation,
        }
    )
    monkeypatch.setattr(
        torch, "get_device_module", lambda: NS(synchronize=lambda: None)
    )
    parked = scheduler.waiting_queue
    AwexSchedulerPlugin(scheduler)._patch_memory_transitions()
    for _ in range(2):
        assert (
            scheduler._request_dispatcher._mapping["release"](NS(tags=["kv_cache"]))
            is not None
        )
    for _ in range(2):
        assert (
            scheduler._request_dispatcher._mapping["resume"](NS(tags=["kv_cache"]))
            is not None
        )
    native.memory_saver_adapter.pause.assert_called_once_with("kv_cache")
    native.memory_saver_adapter.resume.assert_called_once_with("kv_cache")
    assert scheduler.waiting_queue is parked
    assert not scheduler.is_fully_idle()


@pytest.mark.parametrize("tags", [None, []])
def test_unspecified_memory_tags_delegate_to_native_semantics(tags):
    release = Mock(return_value="native-result")
    scheduler = NS(
        offload_tags=set(),
        release_memory_occupation=release,
        resume_memory_occupation=Mock(),
    )
    AwexSchedulerPlugin(scheduler)._patch_memory_transitions()
    request = NS(tags=tags)
    assert scheduler.release_memory_occupation(request) == "native-result"
    release.assert_called_once_with(request)
