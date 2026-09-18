# SPDX-License-Identifier: Apache-2.0
"""KV cache residency and preservation of frozen visual parameters."""

from dataclasses import dataclass
from types import (
    ModuleType,
    SimpleNamespace,
)
from typing import Any

import pytest
import torch
from torch import nn

from areal.models.mcore.qwen4_exp_awex_memory import install_kv_residency_hooks
from areal.models.mcore.qwen4_exp_frozen_state import (
    restore_visual_parameters,
    snapshot_visual_parameters,
)


@pytest.fixture
def lifecycle(monkeypatch):
    events = []
    memory = SimpleNamespace(resident=True, enabled=True)
    monkeypatch.setattr(
        "areal.models.mcore.qwen4_exp_awex_memory.torch.get_device_module",
        lambda: SimpleNamespace(synchronize=lambda: events.append("sync")),
    )

    class Scheduler:
        def __init__(self):
            self._engine_paused = True
            self.running_batch = SimpleNamespace(is_empty=lambda: True)
            self.idle = True
            self.flush_success = True

        def flush_cache(self):
            if not memory.resident:
                raise RuntimeError("write to unmapped KV")
            events.append("clear")
            return self.idle and self.flush_success

    @dataclass(slots=True)
    class Manager:
        scheduler: Any
        tp_worker: Any
        memory_saver_adapter: Any
        is_fully_idle: Any
        flush_cache: Any
        fail_resume: bool = False

        def release_memory_occupation(self, request):
            if not request.tags or "kv_cache" in request.tags:
                assert self.is_fully_idle()
                events.append("unmap")
                memory.resident = False
                self.flush_cache()
            return "released"

        def resume_memory_occupation(self, request):
            if self.fail_resume:
                raise RuntimeError("mapping failed")
            if not request.tags or "kv_cache" in request.tags:
                events.append("map")
                memory.resident = True
            return "resumed"

    module = ModuleType("fake_weight_updater")
    module.SchedulerWeightUpdaterManager = Manager
    module.GPU_MEMORY_TYPE_KV_CACHE = "kv_cache"
    install_kv_residency_hooks(module, Scheduler)
    scheduler = Scheduler()
    model = type("Qwen4ExpForConditionalGeneration", (), {})()
    manager = Manager(
        scheduler,
        SimpleNamespace(model_runner=SimpleNamespace(model=model)),
        memory,
        lambda: scheduler.idle,
        scheduler.flush_cache,
    )
    return SimpleNamespace(
        scheduler=scheduler,
        manager=manager,
        memory=memory,
        events=events,
        module=module,
        scheduler_type=Scheduler,
    )


@pytest.mark.parametrize("tags", [["kv_cache"], ["weights", "kv_cache"], None, []])
def test_kv_multiple_cycles_reset_only_resident_memory(lifecycle, tags):
    case = lifecycle
    request = SimpleNamespace(tags=tags)
    bound_flush = case.manager.flush_cache
    for _ in range(2):
        assert case.manager.release_memory_occupation(request) == "released"
        assert not case.memory.resident
        assert case.manager.flush_cache is bound_flush
        assert case.scheduler.flush_cache() is True
        assert case.manager.resume_memory_occupation(request) == "resumed"
        assert case.memory.resident
    assert case.events == ["clear", "sync", "unmap", "map", "clear", "sync"] * 2


def test_busy_release_does_not_clear_or_unmap(lifecycle):
    lifecycle.scheduler.idle = False
    with pytest.raises(RuntimeError, match="idle or retract-paused"):
        lifecycle.manager.release_memory_occupation(SimpleNamespace(tags=["kv_cache"]))
    assert lifecycle.memory.resident
    assert lifecycle.events == []


def test_failed_flush_does_not_unmap(lifecycle):
    lifecycle.scheduler.flush_success = False
    with pytest.raises(RuntimeError, match="before KV release"):
        lifecycle.manager.release_memory_occupation(SimpleNamespace(tags=["kv_cache"]))
    assert lifecycle.memory.resident
    assert lifecycle.events == ["clear"]


def test_weights_only_does_not_change_kv(lifecycle):
    request = SimpleNamespace(tags=["weights"])
    lifecycle.manager.release_memory_occupation(request)
    lifecycle.manager.resume_memory_occupation(request)
    assert lifecycle.events == []
    assert lifecycle.memory.resident


@pytest.mark.parametrize("unsafe_state", ["unpaused", "running", "not_invalidated"])
def test_nonresident_flush_rejects_unsafe_state(lifecycle, unsafe_state):
    case = lifecycle
    case.manager.release_memory_occupation(SimpleNamespace(tags=["kv_cache"]))
    if unsafe_state == "unpaused":
        case.scheduler._engine_paused = False
    elif unsafe_state == "running":
        case.scheduler.running_batch.is_empty = lambda: False
    else:
        case.scheduler._areal_qwen4_exp_kv_state.invalidated = False
    with pytest.raises(RuntimeError, match="nonresident"):
        case.scheduler.flush_cache()
    assert case.events == ["clear", "sync", "unmap"]


def test_failed_resume_does_not_touch_unmapped_cache(lifecycle):
    case = lifecycle
    request = SimpleNamespace(tags=["kv_cache"])
    case.manager.release_memory_occupation(request)
    case.manager.fail_resume = True
    with pytest.raises(RuntimeError, match="mapping failed"):
        case.manager.resume_memory_occupation(request)
    assert case.scheduler._areal_qwen4_exp_kv_state.released
    assert case.events == ["clear", "sync", "unmap"]


def test_hooks_are_idempotent(lifecycle):
    case = lifecycle
    release = case.module.SchedulerWeightUpdaterManager.release_memory_occupation
    install_kv_residency_hooks(case.module, case.scheduler_type)
    assert (
        case.module.SchedulerWeightUpdaterManager.release_memory_occupation is release
    )


def test_other_model_keeps_native_release_behavior(lifecycle):
    case = lifecycle
    case.manager.tp_worker.model_runner.model = SimpleNamespace()
    with pytest.raises(RuntimeError, match="write to unmapped"):
        case.manager.release_memory_occupation(SimpleNamespace(tags=["kv_cache"]))
    assert case.events == ["unmap"]


class Qwen4ExpForConditionalGeneration(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Linear(3, 2)
        self.model = nn.Linear(3, 2)


def test_visual_snapshot_restores_parameters_without_touching_language_weights():
    model = Qwen4ExpForConditionalGeneration()
    saved = snapshot_visual_parameters(model)
    original_ids = {name: id(p) for name, p in model.named_parameters()}
    for name, parameter in model.named_parameters():
        if name in saved:
            assert saved[name].data_ptr() != parameter.data_ptr()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(42)
    restore_visual_parameters(model, saved)
    for name, parameter in model.named_parameters():
        assert id(parameter) == original_ids[name]
        expected = saved[name] if name in saved else torch.full_like(parameter, 42)
        torch.testing.assert_close(parameter, expected, rtol=0, atol=0)


@pytest.mark.parametrize("fault", ["missing", "extra", "shape", "dtype"])
def test_invalid_visual_snapshot_rejected_before_any_parameter_changes(fault):
    model = Qwen4ExpForConditionalGeneration()
    saved = snapshot_visual_parameters(model)
    if fault == "missing":
        del saved["visual.bias"]
    elif fault == "extra":
        saved["model.bias"] = model.model.bias.detach().clone()
    elif fault == "shape":
        saved["visual.bias"] = torch.ones(1)
    else:
        saved["visual.bias"] = saved["visual.bias"].double()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(42)
    with pytest.raises(ValueError):
        restore_visual_parameters(model, saved)
    for parameter in model.parameters():
        torch.testing.assert_close(
            parameter, torch.full_like(parameter, 42), rtol=0, atol=0
        )


def test_visual_snapshot_rejects_other_model_architectures():
    with pytest.raises(TypeError):
        snapshot_visual_parameters(nn.Linear(3, 2))


def test_static_hooks_preserve_visual_and_native_buffer_lifecycle():
    from types import SimpleNamespace

    from areal.models.mcore.qwen4_exp_frozen_state import install_static_state_hooks

    events = []
    updater = SimpleNamespace(
        _export_static_state=lambda model: {"native": "buffer-state"},
        _import_static_state=lambda model, state: events.append(state["native"]),
    )
    install_static_state_hooks(updater)
    exporter = updater._export_static_state
    install_static_state_hooks(updater)
    assert updater._export_static_state is exporter
    model = Qwen4ExpForConditionalGeneration()
    initial = snapshot_visual_parameters(model)
    for _ in range(2):
        state = updater._export_static_state(model)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(42)
        updater._import_static_state(model, state)
        for name, parameter in model.named_parameters():
            if name in initial:
                torch.testing.assert_close(parameter, initial[name], rtol=0, atol=0)
    assert events == ["buffer-state", "buffer-state"]
    with pytest.raises(ValueError, match="not saved"):
        updater._import_static_state(model, {"native": "buffer-state"})


def test_static_hooks_leave_other_architectures_with_native_state_only():
    from types import SimpleNamespace

    from areal.models.mcore.qwen4_exp_frozen_state import install_static_state_hooks

    state = {"native": "buffer-state"}
    received = []
    updater = SimpleNamespace(
        _export_static_state=lambda model: state,
        _import_static_state=lambda model, value: received.append(value),
    )
    install_static_state_hooks(updater)
    model = nn.Linear(2, 2)
    assert updater._export_static_state(model) is state
    updater._import_static_state(model, state)
    assert received == [state]
