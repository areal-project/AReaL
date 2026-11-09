# SPDX-License-Identifier: Apache-2.0

"""CPU-only checks for v1 rollout timing attribution."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from areal.experimental.openai.proxy.session_trace import ProxySessionTrace
from areal.utils.rollout_timeperf import RolloutTraceObserver, SessionTiming


def _observer() -> RolloutTraceObserver:
    config = SimpleNamespace(
        collect_interval_seconds=60.0,
        log_interval_seconds=600.0,
        long_tail_seconds=0.0,
        top_n=20,
    )
    return RolloutTraceObserver([], "test-key", config, Mock())


def test_attribute_step_concurrent_sessions_splits_wall_time_by_active_count():
    """Two generating sessions and one reward session split 60s as 40/20."""
    observer = _observer()
    for session_id, phase in (("a", "generate"), ("b", "generate"), ("c", "reward")):
        session = SessionTiming(session_id)
        session.apply({"kind": "session_start", "at": 0.0})
        if phase == "generate":
            session.apply({"kind": "generate_start", "at": 0.0})
        else:
            session.apply(
                {
                    "kind": "phase_interval",
                    "at": 0.0,
                    "phase": "reward",
                    "started_at": 0.0,
                    "ended_at": 60.0,
                }
            )
        observer._sessions[("proxy", session_id)] = session

    result = observer.attribute_step(0.0, 60.0, 60.0)

    assert result["generate"] == pytest.approx(40.0)
    assert result["reward"] == pytest.approx(20.0)
    assert result["other"] == pytest.approx(0.0)
    assert sum(result.values()) == pytest.approx(60.0)


def test_attribute_step_clips_sessions_and_assigns_unobserved_time_to_other():
    """Only explicit intervals inside the trainer's rollout scope are counted."""
    observer = _observer()
    session = SessionTiming("a")
    for event in (
        {"kind": "session_start", "at": -20.0},
        {"kind": "generate_start", "at": -10.0},
        {"kind": "generate_end", "at": 20.0},
        {
            "kind": "phase_interval",
            "at": 20.0,
            "phase": "reward",
            "started_at": 20.0,
            "ended_at": 40.0,
        },
        {"kind": "session_end", "at": 50.0},
    ):
        session.apply(event)
    observer._sessions[("proxy", "a")] = session

    result = observer.attribute_step(0.0, 60.0, 61.0)

    assert result["generate"] == pytest.approx(61.0 / 3.0)
    assert result["reward"] == pytest.approx(61.0 / 3.0)
    assert result["other"] == pytest.approx(61.0 / 3.0)
    assert sum(result.values()) == pytest.approx(61.0)


def test_session_timing_active_generate_includes_completed_and_ongoing():
    """Live display uses completed generation plus the current open interval."""
    session = SessionTiming("a")
    for event in (
        {"kind": "session_start", "at": 0.0},
        {"kind": "generate_start", "at": 0.0},
        {"kind": "generate_end", "at": 360.0},
        {"kind": "generate_start", "at": 600.0},
    ):
        session.apply(event)

    assert session.current_phase(780.0) == ("generate", 600.0)
    assert session.cumulative("generate", 780.0) == pytest.approx(540.0)


def test_long_tail_log_reports_current_phase_and_total_generate(monkeypatch):
    """A 3m open request is included in the 9m generation total."""
    observer = _observer()
    session = SessionTiming("a", arena_task_id="arena-task", attempt_id="1")
    for event in (
        {"kind": "session_start", "at": 1.0},
        {"kind": "generate_start", "at": 1.0},
        {"kind": "generate_end", "at": 361.0},
        {"kind": "generate_start", "at": 601.0},
    ):
        session.apply(event)
    observer._sessions[("proxy", "a")] = session
    monkeypatch.setattr("areal.utils.rollout_timeperf.time.time", lambda: 781.0)

    observer.log_long_tails()

    line = observer.logger.info.call_args.args[-1]
    assert "arena-task / 1" in line
    assert "phase=generate (3.0m)" in line
    assert "generate=9.0m" in line


def test_proxy_trace_deduplicates_arena_phase_and_tracks_request_count():
    """Repeated Arena polling does not create duplicate reward intervals."""
    trace = ProxySessionTrace()
    trace.start_session("session")
    trace.begin_generate("session")
    trace.end_generate("session")
    assert trace.bind_arena_task("session", "task", "1")
    assert trace.record_phase("session", "reward", 10.0, None, "1")
    assert trace.record_phase("session", "reward", 10.0, None, "1")
    assert trace.record_phase("session", "reward", 10.0, 20.0, "1")
    events = trace.events_after(0)["events"]

    assert sum(event["kind"] == "request_start" for event in events) == 1
    assert sum(event["kind"] == "phase_interval" for event in events) == 2
    assert events[-1]["ended_at"] == 20.0
