# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import os
import subprocess
import sys
import time
import uuid
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from areal.elastic.control.processes import (
    OWNER_ENV,
    owned_processes,
    stop_owned_processes,
)


def test_orphaned_session_child_is_stopped_without_touching_other_owner(tmp_path):
    """Leader exit and setsid must not leave a compute process behind."""
    owner = uuid.uuid4().hex
    code = (
        "import subprocess,sys; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)'], "
        "start_new_session=True, stdout=subprocess.DEVNULL); "
        "print(p.pid, flush=True)"
    )
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        leader = subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ, OWNER_ENV: owner},
            stdout=subprocess.PIPE,
            text=True,
            check=True,
            timeout=10,
        )
        child = int(leader.stdout)
        deadline = time.monotonic() + 5
        while child not in [p.pid for p in owned_processes(owner)]:
            if time.monotonic() > deadline:
                pytest.fail("Owned child did not start")
            time.sleep(0.05)
        stop_owned_processes(owner, grace=1, force_timeout=2)
        assert not owned_processes(owner)
        assert unrelated.poll() is None
    finally:
        stop_owned_processes(owner, grace=0.1, force_timeout=2)
        unrelated.terminate()
        unrelated.wait(timeout=5)


@pytest.fixture(scope="module")
def batch_observer():
    """Load the actual observer without optional inference-service dependencies."""
    path = (
        Path(__file__).resolve().parents[1]
        / "areal/v2/inference_service/sglang/batch_observer.py"
    )
    spec = importlib.util.spec_from_file_location("batch_observer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "mode,lengths,expected",
    [
        ("EXTEND", [4, 8], {"prefill": (2, 18, 0, 18, 7), "decode": (0, 0, 0, 0, 0)}),
        ("DECODE", [11, 9], {"prefill": (0, 0, 0, 0, 0), "decode": (2, 18, 2, 20, 2)}),
        ("MIXED", [4, 9], {"prefill": (1, 10, 0, 10, 4), "decode": (1, 8, 1, 9, 1)}),
    ],
)
def test_batch_observer_counts_chunked_prefill_and_overlap_decode(
    batch_observer, mode, lengths, expected
):
    """Full prompt totals differ from prefill chunks; overlap lengths beat stale outputs."""
    reqs = [
        SimpleNamespace(rid=f"r-{i}", origin_input_ids=list(range(size)), output_ids=[])
        for i, size in enumerate([10, 8])
    ]
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(name=mode),
        reqs=reqs,
        seq_lens_cpu=lengths,
        extend_lens=[4, 3],
        decoding_reqs=[reqs[1]],
    )
    result = batch_observer.snapshot_batch(batch)
    original_lengths = list(lengths)
    lengths[:] = [100, 200]
    reqs[0].origin_input_ids.append(999)
    for phase, values in expected.items():
        assert result[phase] == dict(
            zip(
                (
                    "requests",
                    "prompt_tokens",
                    "generated_tokens",
                    "context_tokens",
                    "input_tokens",
                ),
                values,
            )
        )
    assert [req["request_id"] for req in result["requests"]] == ["r-0", "r-1"]
    for req, length in zip(result["requests"], original_lengths):
        assert req["sequence_tokens"] == length
        q = req["query_tokens"]
        assert req["past_tokens"] == length - q
        assert req["attention_pairs"] == q * (length - q) + q * (q + 1) // 2


def test_batch_observer_drains_last_gpu_event_during_idle(
    batch_observer, monkeypatch, tmp_path
):
    """Native contexts/results stay intact; the last batch needs no later forward."""
    events = []
    stamps = iter([0, 2, 5])

    class Event:
        def __init__(self, enable_timing):
            assert enable_timing
            self.ready = False
            events.append(self)

        def record(self):
            self.stamp = next(stamps)

        def query(self):
            return self.ready

        def elapsed_time(self, other):
            assert self.ready and other.ready
            return other.stamp - self.stamp

    native_calls = []
    scheduler = SimpleNamespace(
        tp_rank=0,
        pp_rank=0,
        dp_rank=None,
        attn_cp_rank=0,
        gpu_id=4,
        device_module=SimpleNamespace(Event=Event),
        server_args=SimpleNamespace(
            enable_dp_attention=False, speculative_algorithm=None, enable_pdmux=False
        ),
        record_forward_metrics=lambda batch: nullcontext(native_calls.append(batch)),
        recv_requests=lambda: ["next request"],
        process_batch_result=lambda value: value,
        max_total_num_tokens=128,
        token_to_kv_pool_allocator=SimpleNamespace(available_size=lambda: 64),
    )
    monkeypatch.setenv(batch_observer.TRACE_DIR_ENV, str(tmp_path))
    observer = batch_observer.bind_batch_observer(scheduler)
    now = [100.0]
    observer._wall = lambda: now[0]
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(name="DECODE"),
        reqs=[SimpleNamespace(origin_input_ids=[1, 2, 3], output_ids=[])],
        seq_lens_cpu=[7],
        decoding_reqs=None,
    )
    with scheduler.record_forward_metrics(batch):
        now[0] = 100.006
    assert native_calls == [batch]
    assert len(observer.path.read_text().splitlines()) == 1
    assert scheduler.process_batch_result("original result") == "original result"
    for event in events:
        event.ready = True
    now[0] = 100.020
    assert scheduler.recv_requests() == ["next request"]
    observer.close()
    rows = [json.loads(line) for line in observer.path.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["gpu_id"] == 4
    assert rows[1]["timing_status"] == "complete"
    assert rows[1]["decode"]["context_tokens"] == 7
    assert rows[1]["kv_allocated_tokens_at_forward_entry"] == 64
    assert rows[1]["gpu_start_ms"] == 2
    assert rows[1]["gpu_end_ms"] == 5
    assert rows[1]["gpu_seconds"] == pytest.approx(0.003)
    assert rows[1]["gpu_origin_wall_lower"] <= rows[1]["gpu_origin_wall_upper"]
    assert rows[1]["host_finished"] - rows[1]["host_started"] == pytest.approx(0.006)


def test_batch_observer_is_opt_in_and_avoids_tp_duplicates(
    batch_observer, monkeypatch, tmp_path
):
    """Disabled collection does not touch the GPU, native methods, or filesystem."""
    monkeypatch.delenv(batch_observer.TRACE_DIR_ENV, raising=False)
    assert batch_observer.bind_batch_observer(SimpleNamespace()) is None
    monkeypatch.setenv(batch_observer.TRACE_DIR_ENV, str(tmp_path / "trace"))
    assert (
        batch_observer.bind_batch_observer(SimpleNamespace(tp_rank=1, attn_cp_rank=0))
        is None
    )
    assert not (tmp_path / "trace").exists()
