import asyncio
import getpass
import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from areal.api.cli_args import PerfTracerConfig, SessionTracerConfig
from areal.infra.platforms import current_platform
from areal.utils import perf_tracer
from areal.utils.network import find_free_ports
from areal.utils.perf_tracer import Category, SessionTraceEvent


def _make_config(
    root: Path,
    *,
    enabled: bool = True,
    experiment: str = "test-exp",
    trial: str = "trial",
) -> PerfTracerConfig:
    return PerfTracerConfig(
        experiment_name=experiment,
        trial_name=trial,
        fileroot=str(root),
        enabled=enabled,
    )


def _expected_trace_path(
    config: PerfTracerConfig,
    *,
    rank: int,
    role: str | None = None,
) -> Path:
    base_dir = Path(os.path.expanduser(config.fileroot))
    filename = f"traces-r{rank}.jsonl"
    path = (
        base_dir
        / "logs"
        / getpass.getuser()
        / config.experiment_name
        / config.trial_name
        / "perf_tracer"
    )
    if role:
        path = path / role
    return path / filename


def _expected_session_trace_path(
    config: PerfTracerConfig,
    *,
    rank: int,
    role: str | None = None,
) -> Path:
    base_dir = Path(os.path.expanduser(config.fileroot))
    filename = f"sessions-r{rank}.jsonl"
    path = (
        base_dir
        / "logs"
        / getpass.getuser()
        / config.experiment_name
        / config.trial_name
        / "session_tracer"
    )
    if role:
        path = path / role
    return path / filename


def _load_trace_events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _install_fake_profiler(monkeypatch) -> dict[str, int]:
    counters = {"enter": 0, "exit": 0}

    class _FakeProfiler:
        def __enter__(self):
            counters["enter"] += 1
            return self

        def __exit__(self, exc_type, exc, exc_tb):
            counters["exit"] += 1
            return False

        def export_chrome_trace(self, path: str) -> None:
            payload = {
                "traceEvents": [
                    {
                        "ts": 0,
                        "pid": 0,
                        "tid": 0,
                        "ph": "X",
                        "name": "fake",
                        "args": {},
                    }
                ]
            }
            Path(path).write_text(json.dumps(payload))

    monkeypatch.setattr(
        perf_tracer.profiler, "profile", lambda *a, **k: _FakeProfiler()
    )
    return counters


@pytest.fixture(autouse=True)
def clean_global_tracer():
    perf_tracer.reset()
    yield
    perf_tracer.reset()


def test_module_level_helpers_require_configuration():
    with pytest.raises(RuntimeError):
        perf_tracer.get_tracer()


def test_perf_tracer_records_events_and_save(tmp_path):
    config = _make_config(tmp_path, experiment="unit", trial="scope")
    tracer = perf_tracer.PerfTracer(config, rank=0)
    assert tracer._rank == 0  # noqa: SLF001

    with tracer.trace_scope(
        "unit-block",
        category=Category.INSTR,
        args={"step": 1},
    ):
        tracer.instant("inner-mark", args={"value": 42})
    tracer.instant("outer-mark")

    tracer.save()
    saved_path = _expected_trace_path(config, rank=0)
    assert saved_path.exists()

    events = _load_trace_events(saved_path)
    event_names = {evt["name"] for evt in events if evt["ph"] != "M"}
    assert {"unit-block", "inner-mark", "outer-mark"}.issubset(event_names)


def test_perf_tracer_with_role(tmp_path):
    """Test that role parameter is properly handled in tracer."""
    config = _make_config(tmp_path, experiment="unit", trial="role_test")
    tracer = perf_tracer.PerfTracer(config, rank=0, role="actor")
    assert tracer._rank == 0  # noqa: SLF001
    assert tracer._role == "actor"  # noqa: SLF001

    with tracer.trace_scope("role-block", category=Category.INSTR):
        tracer.instant("role-mark")

    tracer.save()
    saved_path = _expected_trace_path(config, rank=0, role="actor")
    assert saved_path.exists()

    events = _load_trace_events(saved_path)
    non_meta_events = [evt for evt in events if evt.get("ph") != "M"]

    # Verify role is in event args
    for evt in non_meta_events:
        assert evt["args"].get("role") == "actor"
        assert evt["args"].get("rank") == 0

    # Verify process metadata includes role in label
    process_name_events = [
        evt
        for evt in events
        if evt.get("ph") == "M" and evt.get("name") == "process_name"
    ]
    assert len(process_name_events) > 0
    for evt in process_name_events:
        process_name = evt["args"].get("name", "")
        assert "[actor]" in process_name
        assert "Rank 0" in process_name


def test_perf_tracer_emits_separate_rank_logs(tmp_path):
    config0 = _make_config(
        tmp_path,
        enabled=True,
        experiment="agg",
        trial="shared",
    )
    tracer0 = perf_tracer.PerfTracer(config0, rank=0)
    with tracer0.trace_scope("rank0-step", args={"rank": 0}):
        pass
    tracer0.instant("rank0-mark", args={"rank": 0})
    tracer0.save()
    saved_path_rank0 = _expected_trace_path(config0, rank=0)
    assert saved_path_rank0.exists()

    config1 = _make_config(
        tmp_path,
        enabled=True,
        experiment="agg",
        trial="shared",
    )
    tracer1 = perf_tracer.PerfTracer(config1, rank=1)
    tracer1._pid = tracer0._pid + 1  # noqa: SLF001 - simulate distinct process id
    tracer1_thread = getattr(tracer1, "_thread_meta_emitted", set())
    tracer1_thread.clear()
    tracer1.instant("rank1-mark", args={"rank": 1})
    tracer1.save()
    saved_path_rank1 = _expected_trace_path(config1, rank=1)
    assert saved_path_rank1.exists()

    events_rank0 = _load_trace_events(saved_path_rank0)
    events_rank1 = _load_trace_events(saved_path_rank1)

    def _non_meta(events: list[dict]) -> list[dict]:
        return [evt for evt in events if evt.get("ph") != "M"]

    def _meta(events: list[dict], name: str) -> list[dict]:
        return [
            evt for evt in events if evt.get("ph") == "M" and evt.get("name") == name
        ]

    event_names_rank0 = {evt["name"] for evt in _non_meta(events_rank0)}
    event_names_rank1 = {evt["name"] for evt in _non_meta(events_rank1)}
    assert {"rank0-step", "rank0-mark"}.issubset(event_names_rank0)
    assert {"rank1-mark"}.issubset(event_names_rank1)

    ranks_rank0 = {evt["args"].get("rank") for evt in _non_meta(events_rank0)}
    ranks_rank1 = {evt["args"].get("rank") for evt in _non_meta(events_rank1)}
    assert ranks_rank0 == {0}
    assert ranks_rank1 == {1}

    pid_rank0 = {evt["pid"] for evt in _non_meta(events_rank0)}
    pid_rank1 = {evt["pid"] for evt in _non_meta(events_rank1)}
    assert pid_rank0 == {tracer0._pid}  # noqa: SLF001
    assert pid_rank1 == {tracer1._pid}  # noqa: SLF001

    process_name_rank0 = _meta(events_rank0, "process_name")
    process_name_rank1 = _meta(events_rank1, "process_name")
    assert any(
        evt["args"].get("name") == "Rank 0, Process" for evt in process_name_rank0
    )
    assert any(
        evt["args"].get("name") == "Rank 1, Process" for evt in process_name_rank1
    )

    sort_meta_rank0 = _meta(events_rank0, "process_sort_index")
    sort_meta_rank1 = _meta(events_rank1, "process_sort_index")
    assert any(evt["args"].get("sort_index") == 0 for evt in sort_meta_rank0)
    assert any(evt["args"].get("sort_index") == 1 for evt in sort_meta_rank1)


@pytest.mark.asyncio
async def test_global_tracer_configure_roundtrip(tmp_path):
    config = _make_config(tmp_path, experiment="global", trial="roundtrip")
    tracer = perf_tracer.configure(
        config,
        rank=0,
    )

    async with perf_tracer.atrace_scope(
        "async-step",
        category=Category.INSTR,
        args={"phase": "enter"},
    ):
        perf_tracer.instant("inside-async", args={"flag": True})

    with perf_tracer.trace_scope(
        "sync-step",
        category=Category.INSTR,
    ):
        pass

    tracer.save()
    saved_path = _expected_trace_path(config, rank=0)
    assert saved_path.exists()
    events = _load_trace_events(saved_path)
    event_names = {evt["name"] for evt in events if evt["ph"] != "M"}
    assert {"async-step", "inside-async", "sync-step"}.issubset(event_names)


@pytest.mark.asyncio
async def test_configure_with_role(tmp_path):
    """Test that configure() function accepts role parameter."""
    config = _make_config(tmp_path, experiment="global", trial="role_roundtrip")
    tracer = perf_tracer.configure(
        config,
        rank=1,
        role="learner",
    )

    # Verify tracer has role stored
    assert tracer._role == "learner"  # noqa: SLF001

    with perf_tracer.trace_scope("role-sync-step", category=Category.INSTR):
        perf_tracer.instant("role-inside", args={"task": "train"})

    tracer.save()
    saved_path = _expected_trace_path(config, rank=1, role="learner")
    assert saved_path.exists()
    # Session tracer is not enabled in this config, so session_path doesn't exist

    events = _load_trace_events(saved_path)

    # Verify role in events
    non_meta_events = [evt for evt in events if evt.get("ph") != "M"]
    for evt in non_meta_events:
        assert evt["args"].get("role") == "learner"

    # Verify process metadata
    process_events = [
        evt
        for evt in events
        if evt.get("ph") == "M" and evt.get("name") == "process_name"
    ]
    assert len(process_events) > 0
    for evt in process_events:
        pname = evt["args"].get("name", "")
        assert "[learner]" in pname
        assert "Rank 1" in pname


@pytest.mark.asyncio
async def test_async_multi_session_cross_phase_trace(tmp_path):
    config = _make_config(tmp_path, experiment="async", trial="sessions")
    tracer = perf_tracer.configure(
        config,
        rank=0,
    )

    phase_schedules = {
        "sess-0": [
            ("rollout", 0.1),
            ("train", 0.2),
            ("reward", 0.15),
        ],
        "sess-1": [
            ("rollout", 0.12),
            ("train", 0.11),
            ("reward", 0.18),
        ],
    }

    async def run_session(
        session_id: str, phases: list[tuple[str, float]], offset: float
    ) -> None:
        if offset:
            await asyncio.sleep(offset)
        loop = asyncio.get_running_loop()
        async with perf_tracer.atrace_scope(
            "session", category=f"session.{session_id}", args={"session_id": session_id}
        ):
            for phase, duration in phases:
                with perf_tracer.trace_scope(
                    phase,
                    category=f"{phase}.{session_id}",
                    args={"session_id": session_id},
                ):
                    await loop.run_in_executor(None, time.sleep, duration)

    await asyncio.gather(
        run_session("sess-0", phase_schedules["sess-0"], 0.0),
        run_session("sess-1", phase_schedules["sess-1"], 0.05),
    )

    tracer.save()
    saved_path = _expected_trace_path(config, rank=0)
    assert saved_path.exists()
    events = [evt for evt in _load_trace_events(saved_path) if evt.get("ph") != "M"]

    observed = {
        (evt["args"].get("session_id"), evt["name"])
        for evt in events
        if evt["ph"] == "X"
    }
    expected_phases = {
        ("sess-0", "session"),
        ("sess-0", "rollout"),
        ("sess-0", "train"),
        ("sess-0", "reward"),
        ("sess-1", "session"),
        ("sess-1", "rollout"),
        ("sess-1", "train"),
        ("sess-1", "reward"),
    }
    assert expected_phases.issubset(observed)

    session_timestamps = {
        sess_id: [
            evt["ts"] for evt in events if evt["args"].get("session_id") == sess_id
        ]
        for sess_id in phase_schedules
    }
    overlap = min(session_timestamps["sess-1"]) < max(
        session_timestamps["sess-0"]
    ) and min(session_timestamps["sess-0"]) < max(session_timestamps["sess-1"])
    assert overlap


def test_configure_rejects_repeated_calls(tmp_path):
    config = _make_config(tmp_path, experiment="ranked", trial="zero")
    perf_tracer.configure(
        config,
        rank=0,
    )
    with pytest.raises(RuntimeError):
        perf_tracer.configure(
            config,
            rank=1,
        )


def test_session_tracer_with_role(tmp_path):
    """Test SessionTracer with role parameter produces correct output paths."""
    config = _make_config(tmp_path, experiment="session", trial="role_enabled")
    config.session_tracer = SessionTracerConfig(enabled=True, flush_threshold=1)
    tracer = perf_tracer.PerfTracer(config, rank=2, role="critic")

    session_tracer = tracer.session_tracer
    assert session_tracer is not None
    assert session_tracer._role == "critic"  # noqa: SLF001

    task_id = 456
    session_tracer.register_task(task_id)
    session_id = session_tracer.register_session(task_id)
    session_tracer.record_event(
        session_id,
        SessionTraceEvent.GENERATE_START,
    )
    session_tracer.record_event(
        session_id,
        SessionTraceEvent.GENERATE_END,
    )
    session_tracer.record_event(
        session_id, SessionTraceEvent.FINALIZED, status="accepted"
    )
    tracer.save(force=True)

    session_path = _expected_session_trace_path(config, rank=2, role="critic")
    assert session_path.exists()
    payload = [json.loads(line) for line in session_path.read_text().splitlines()]
    assert any(entry["status"] == "accepted" for entry in payload)

    # Verify role field in session record
    record = payload[0]
    assert record.get("role") == "critic"


def test_module_level_save_helper(tmp_path):
    config = _make_config(tmp_path, experiment="module", trial="helper")
    perf_tracer.configure(
        config,
        rank=0,
    )
    perf_tracer.instant("module-level-mark", args={"flag": True})

    perf_tracer.save()
    saved_path = _expected_trace_path(config, rank=0)
    assert saved_path.exists()
    assert saved_path == _expected_trace_path(config, rank=0)
    events = _load_trace_events(saved_path)
    event_names = {evt["name"] for evt in events if evt.get("ph") != "M"}
    assert "module-level-mark" in event_names


def test_perf_tracer_respects_save_interval(tmp_path):
    config = _make_config(tmp_path, experiment="interval", trial="steps")
    config.save_interval = 3
    tracer = perf_tracer.PerfTracer(config, rank=0)
    trace_path = _expected_trace_path(config, rank=0)

    for step in (0, 1):
        tracer.instant(f"mark-{step}", args={"step": step})
        tracer.save(step=step)
        assert not trace_path.exists()

    tracer.instant("mark-2", args={"step": 2})
    tracer.save(step=2)
    assert trace_path.exists()
    events = _load_trace_events(trace_path)
    names = {evt["name"] for evt in events if evt.get("ph") != "M"}
    assert {"mark-0", "mark-1", "mark-2"}.issubset(names)

    tracer.instant("mark-3", args={"step": 3})
    tracer.save(step=3)
    tracer.instant("mark-4", args={"step": 4})
    tracer.save(step=4)
    tracer.save(force=True)

    events = _load_trace_events(trace_path)
    names = {evt["name"] for evt in events if evt.get("ph") != "M"}
    assert {"mark-3", "mark-4"}.issubset(names)


def test_profiler_only_runs_on_configured_steps(tmp_path, monkeypatch):
    counters = _install_fake_profiler(monkeypatch)
    config = _make_config(tmp_path, experiment="profile", trial="steps")
    config.profile_steps = [3]
    tracer = perf_tracer.PerfTracer(config, rank=0)

    with tracer.trace_scope("skip", enable_profiler=True, args={"global_step": 2}):
        pass
    assert counters["enter"] == 0

    with tracer.trace_scope("hit", enable_profiler=True, args={"global_step": 3}):
        pass
    assert counters["enter"] == 1


def test_profiler_request_is_skipped_when_active(tmp_path, monkeypatch):
    counters = _install_fake_profiler(monkeypatch)
    config = _make_config(tmp_path, experiment="profile", trial="nesting")
    config.profile_steps = [0]
    tracer = perf_tracer.PerfTracer(config, rank=0)

    with tracer.trace_scope("outer", enable_profiler=True, args={"global_step": 0}):
        with tracer.trace_scope("inner", enable_profiler=True, args={"global_step": 0}):
            pass
    # Nested scope should not start a second profiler session
    assert counters["enter"] == 1

    with tracer.trace_scope("after", enable_profiler=True, args={"global_step": 0}):
        pass
    assert counters["enter"] == 2


def test_session_tracer_configuration(tmp_path):
    config = _make_config(tmp_path, experiment="session", trial="enabled")
    config.session_tracer = SessionTracerConfig(enabled=True, flush_threshold=1)
    tracer = perf_tracer.PerfTracer(config, rank=0)

    session_tracer = tracer.session_tracer
    assert session_tracer is not None

    task_id = 123
    session_tracer.register_task(task_id)
    session_id = session_tracer.register_session(task_id)
    session_tracer.record_event(
        session_id,
        SessionTraceEvent.GENERATE_START,
    )
    session_tracer.record_event(
        session_id,
        SessionTraceEvent.GENERATE_END,
    )
    session_tracer.record_event(
        session_id, SessionTraceEvent.FINALIZED, status="accepted"
    )
    tracer.save(force=True)

    session_path = _expected_session_trace_path(config, rank=0)
    assert session_path.exists()
    payload = [json.loads(line) for line in session_path.read_text().splitlines()]
    assert any(entry["status"] == "accepted" for entry in payload)

    updated = _make_config(tmp_path, experiment="session", trial="enabled")
    updated.session_tracer = SessionTracerConfig(enabled=False)
    tracer_disabled = perf_tracer.PerfTracer(updated, rank=1)
    assert tracer_disabled.session_tracer is None


def _run_perf_tracer_torchrun(tmp_path: Path, world_size: int) -> None:
    port = find_free_ports(1)[0]
    env = {
        **os.environ,
        "AREAL_PERF_TRACE_BASE": str(tmp_path),
    }
    subprocess.run(
        [
            "torchrun",
            f"--nproc_per_node={world_size}",
            "--nnodes=1",
            "--master-addr=localhost",
            f"--master_port={port}",
            "tests/torchrun/run_perf_tracer.py",
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.multi_gpu
@pytest.mark.parametrize("world_size", [2])
def test_perf_tracer_torchrun_multi_rank(tmp_path, world_size):
    device_count_fn = getattr(current_platform, "device_count", None)
    available_devices = device_count_fn() if callable(device_count_fn) else 0
    if available_devices < world_size:
        pytest.skip(f"This test requires {world_size} gpus")

    _run_perf_tracer_torchrun(tmp_path, world_size)

    config = PerfTracerConfig(
        experiment_name="torchrun",
        trial_name=f"world-{world_size}",
        fileroot=str(tmp_path),
        enabled=True,
    )
    trace_paths = [
        _expected_trace_path(config, rank=rank) for rank in range(world_size)
    ]
    for path in trace_paths:
        assert path.exists()

    payload: list[dict] = []
    for path in trace_paths:
        payload.extend(_load_trace_events(path))
    ranks_seen = {
        evt["args"].get("rank") for evt in payload if evt["name"] == "torchrun-step"
    }
    assert ranks_seen == set(range(world_size))
    mark_ranks = {
        evt["args"].get("rank") for evt in payload if evt["name"] == "torchrun-mark"
    }
    assert mark_ranks == set(range(world_size))


@pytest.mark.asyncio
async def test_trace_session_phase(tmp_path):
    """Test that atrace_session_phase context manager works correctly."""
    config = PerfTracerConfig(
        experiment_name="test-phase",
        trial_name="trial",
        fileroot=str(tmp_path),
        enabled=True,
        session_tracer=SessionTracerConfig(enabled=True, flush_threshold=1),
    )
    perf_tracer.configure(config, rank=0)
    try:
        tracer = perf_tracer.get_session_tracer()
        assert tracer is not None

        # Register a task and session
        task_id = 234
        tracer.register_task(task_id)
        session_id = tracer.register_session(task_id)

        perf_tracer.set_session_id(session_id)

        # Use context manager for generate phase
        async with perf_tracer.atrace_session_phase("generate"):
            await asyncio.sleep(0.01)  # Simulate work

        # Use context manager for reward phase
        async with perf_tracer.atrace_session_phase("reward"):
            await asyncio.sleep(0.01)  # Simulate work

        # Mark as completed using SessionTracer API
        tracer.record_event(session_id, SessionTraceEvent.FINALIZED, status="accepted")

        # Force flush
        tracer.flush(force=True)

        # Read the trace file
        session_trace_path = (
            Path(tmp_path)
            / "logs"
            / getpass.getuser()
            / "test-phase"
            / "trial"
            / "session_tracer"
            / "sessions-r0.jsonl"
        )
        assert session_trace_path.exists()

        with open(session_trace_path) as f:
            records = [json.loads(line) for line in f]

        assert len(records) == 1
        record = records[0]

        # Verify phases exist
        assert "phases" in record
        assert "generate" in record["phases"]
        assert "reward" in record["phases"]

        # Verify each phase has start and end timestamps
        generate_phase = record["phases"]["generate"]
        assert len(generate_phase) == 1
        assert generate_phase[0]["start_ts"] is not None
        assert generate_phase[0]["end_ts"] is not None

        reward_phase = record["phases"]["reward"]
        assert len(reward_phase) == 1
        assert reward_phase[0]["start_ts"] is not None
        assert reward_phase[0]["end_ts"] is not None

        # Verify computed times
        assert "generate_s" in record
        assert record["generate_s"] > 0
        assert "reward_s" in record
        assert record["reward_s"] > 0

    finally:
        perf_tracer.reset()


@pytest.mark.asyncio
async def test_trace_session_phase_with_exception(tmp_path):
    """Test that atrace_session_phase handles exceptions correctly."""
    config = PerfTracerConfig(
        experiment_name="test-phase-exc",
        trial_name="trial",
        fileroot=str(tmp_path),
        enabled=True,
        session_tracer=SessionTracerConfig(enabled=True, flush_threshold=1),
    )
    perf_tracer.configure(config, rank=0)
    try:
        tracer = perf_tracer.get_session_tracer()
        assert tracer is not None

        task_id = 345
        tracer.register_task(task_id)
        session_id = tracer.register_session(task_id)

        perf_tracer.set_session_id(session_id)

        # Even if exception occurs, end event should be recorded
        with pytest.raises(ValueError):
            async with perf_tracer.atrace_session_phase("generate"):
                raise ValueError("Simulated error")

        # Complete the session
        tracer.record_event(session_id, SessionTraceEvent.FINALIZED, status="failed")
        tracer.flush(force=True)

        # Read the trace
        session_trace_path = (
            Path(tmp_path)
            / "logs"
            / getpass.getuser()
            / "test-phase-exc"
            / "trial"
            / "session_tracer"
            / "sessions-r0.jsonl"
        )
        assert session_trace_path.exists()

        with open(session_trace_path) as f:
            records = [json.loads(line) for line in f]

        assert len(records) == 1
        record = records[0]

        # Verify generate phase was recorded with both start and end
        assert "phases" in record
        assert "generate" in record["phases"]
        generate_phase = record["phases"]["generate"]
        assert len(generate_phase) == 1
        assert generate_phase[0]["start_ts"] is not None
        assert generate_phase[0]["end_ts"] is not None

    finally:
        perf_tracer.reset()
