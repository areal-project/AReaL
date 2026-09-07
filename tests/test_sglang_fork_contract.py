"""CPU-only tests for the SGLang fork contract."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from areal.engine.sglang_fork_contract import (
    check_scheduler_contract,
    check_static_contract,
    resolve_scheduler_memory_method,
    resolve_scheduler_parallel_value,
)


class _RunningBatch:
    def is_empty(self) -> bool:
        return True


def _scheduler(**overrides):
    scheduler = SimpleNamespace(
        _engine_paused=False,
        ps=SimpleNamespace(tp_rank=3, tp_size=8, gpu_id=3),
        tp_worker=SimpleNamespace(
            model_runner=SimpleNamespace(model=object()),
        ),
        tp_cpu_group=object(),
        running_batch=_RunningBatch(),
        waiting_queue=[],
        weight_updater=SimpleNamespace(
            release_memory_occupation=lambda request: request,
            resume_memory_occupation=lambda request: request,
        ),
        process_input_requests=lambda requests: requests,
    )
    for name, value in overrides.items():
        setattr(scheduler, name, value)
    return scheduler


def test_parallel_value_prefers_theta_parallel_state():
    """Theta's scheduler.ps layout is accepted without a rank-zero fallback."""
    scheduler = _scheduler()

    assert resolve_scheduler_parallel_value(scheduler, "tp_rank") == 3


def test_parallel_value_accepts_legacy_tp_worker_layout():
    """The production Theta tp_worker layout remains supported."""
    scheduler = _scheduler(ps=None)
    scheduler.tp_worker.tp_rank = 5

    assert resolve_scheduler_parallel_value(scheduler, "tp_rank") == 5


def test_parallel_value_rejects_missing_rank():
    """A missing rank fails instead of silently selecting rank zero."""
    scheduler = _scheduler(ps=None)

    with pytest.raises(RuntimeError, match="implicit rank-zero"):
        resolve_scheduler_parallel_value(scheduler, "tp_rank")


def test_parallel_value_rejects_conflicting_sources():
    """Conflicting layouts fail before they can corrupt a transfer plan."""
    scheduler = _scheduler(tp_rank=4)

    with pytest.raises(RuntimeError, match="Conflicting scheduler tp_rank"):
        resolve_scheduler_parallel_value(scheduler, "tp_rank")


def test_theta_scheduler_contract_passes():
    """The reviewed Theta scheduler surface satisfies the instance contract."""
    check_scheduler_contract(_scheduler())


def test_theta_static_contract_passes(monkeypatch):
    """The reviewed Theta class-level scheduler surface passes preflight."""
    modules = {
        name: ModuleType(name)
        for name in (
            "sglang",
            "sglang.srt",
            "sglang.srt.managers",
            "sglang.srt.managers.scheduler_components",
            "sglang.srt.managers.io_struct",
            "sglang.srt.managers.scheduler",
            "sglang.srt.managers.scheduler_components.weight_updater",
        )
    }

    class PauseGenerationReqInput:
        def __init__(self, mode: str) -> None:
            if mode != "retract":
                raise ValueError(mode)

    class Scheduler:
        def process_input_requests(self, requests):
            return requests

        def pause_generation(self, request):
            return request

        def continue_generation(self, request):
            return request

        def flush_cache(self):
            return True

        def is_fully_idle(self):
            return True

    class SchedulerWeightUpdaterManager:
        def release_memory_occupation(self, request):
            return request

    modules[
        "sglang.srt.managers.io_struct"
    ].PauseGenerationReqInput = PauseGenerationReqInput
    modules["sglang.srt.managers.scheduler"].Scheduler = Scheduler
    modules[
        "sglang.srt.managers.scheduler_components.weight_updater"
    ].SchedulerWeightUpdaterManager = SchedulerWeightUpdaterManager
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    check_static_contract()


def test_scheduler_contract_reports_all_missing_requirements():
    """Contract failures report every missing integration surface together."""
    with pytest.raises(RuntimeError) as exc_info:
        check_scheduler_contract(SimpleNamespace())

    message = str(exc_info.value)
    assert "tp_rank" in message
    assert "tp_cpu_group" in message
    assert "running_batch.is_empty" in message
    assert "model_runner.model" in message


@pytest.mark.parametrize("modern", [False, True])
def test_memory_method_resolves_actual_api_owner(modern):
    """Modern managers and legacy schedulers use the same native request."""
    request = object()
    calls = []
    owner = SimpleNamespace(resume_memory_occupation=lambda req: calls.append(req))
    scheduler = SimpleNamespace(weight_updater=owner) if modern else owner

    resolve_scheduler_memory_method(scheduler, "resume_memory_occupation")(request)

    assert calls == [request]


def test_contract_rejects_manager_without_memory_methods():
    """A manager object alone does not prove that the required API exists."""
    with pytest.raises(RuntimeError, match="resume_memory_occupation"):
        check_scheduler_contract(_scheduler(weight_updater=object()))
