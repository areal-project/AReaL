# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from areal.trainer.ppo import actor as actor_module
from areal.utils import memory

GIB = 1024**3


@pytest.fixture
def peak_memory_platform(monkeypatch):
    events = []
    platform = SimpleNamespace(
        reset_peak_memory_stats=Mock(side_effect=lambda: events.append("reset")),
        max_memory_allocated=Mock(
            side_effect=lambda: events.append("allocated") or 25 * GIB
        ),
        max_memory_reserved=Mock(
            side_effect=lambda: events.append("reserved") or 30 * GIB
        ),
    )
    monkeypatch.setattr(memory, "current_platform", platform)
    monkeypatch.setattr(memory.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(memory.dist, "get_rank", lambda: 0)
    log_info = Mock()
    monkeypatch.setattr(memory.logger, "info", log_info)
    return platform, events, log_info


def test_peak_memory_report_resets_before_phase_and_logs_high_water_mark(
    peak_memory_platform,
):
    platform, events, log_info = peak_memory_platform

    with memory.report_peak_memory("ref logp"):
        events.append("phase")

    assert events == ["reset", "phase", "allocated", "reserved"]
    log_info.assert_called_once_with(
        "[PeakMemory Rank 0] ref logp: "
        "max allocated (GB): 25.00, max reserved (GB): 30.00"
    )
    platform.reset_peak_memory_stats.assert_called_once_with()


def test_peak_memory_report_runs_when_phase_raises(peak_memory_platform):
    _, events, log_info = peak_memory_platform
    error = RuntimeError("device out of memory")

    with pytest.raises(RuntimeError) as exc_info:
        with memory.report_peak_memory("ppo update"):
            events.append("phase")
            raise error

    assert exc_info.value is error
    assert events == ["reset", "phase", "allocated", "reserved"]
    assert log_info.call_args.args[0].startswith("[PeakMemory Rank 0] ppo update:")


def test_peak_memory_report_skips_unsupported_platform(monkeypatch):
    events = []
    platform = SimpleNamespace(
        reset_peak_memory_stats=Mock(),
        max_memory_allocated=Mock(),
    )
    monkeypatch.setattr(memory, "current_platform", platform)
    log_info = Mock()
    monkeypatch.setattr(memory.logger, "info", log_info)

    with memory.report_peak_memory("ppo update"):
        events.append("phase")

    assert events == ["phase"]
    platform.reset_peak_memory_stats.assert_not_called()
    log_info.assert_not_called()


def test_peak_memory_reset_failure_does_not_skip_phase(peak_memory_platform):
    platform, events, _ = peak_memory_platform
    platform.reset_peak_memory_stats.side_effect = RuntimeError("counter unavailable")

    with memory.report_peak_memory("ppo update"):
        events.append("phase")

    assert events == ["phase"]


def test_peak_memory_reporting_failure_does_not_mask_phase_error(
    peak_memory_platform,
):
    platform, _, _ = peak_memory_platform
    platform.max_memory_allocated.side_effect = RuntimeError("counter unavailable")
    phase_error = RuntimeError("device out of memory")

    with pytest.raises(RuntimeError) as exc_info:
        with memory.report_peak_memory("ppo update"):
            raise phase_error

    assert exc_info.value is phase_error


def test_ppo_actor_reports_one_peak_for_each_outer_phase(monkeypatch):
    events = []

    @contextmanager
    def record_peak(phase):
        events.append(("enter", phase))
        try:
            yield
        finally:
            events.append(("exit", phase))

    def record_batch(_fn, _data, **kwargs):
        events.append(("batch", kwargs))
        return []

    monkeypatch.setattr(actor_module, "report_peak_memory", record_peak)
    monkeypatch.setattr(actor_module, "batched_call", record_batch)
    actor = object.__new__(actor_module.PPOActor)

    assert actor.compute_logp([], peak_memory_phase="ref logp") == []
    actor.ppo_update([], peak_memory_phase="actor ppo update")

    assert events == [
        ("enter", "ref logp"),
        ("batch", {}),
        ("exit", "ref logp"),
        ("enter", "actor ppo update"),
        ("batch", {"unpack": False}),
        ("exit", "actor ppo update"),
    ]
