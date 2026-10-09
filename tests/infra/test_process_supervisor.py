# SPDX-License-Identifier: Apache-2.0

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import psutil
import pytest

SUPERVISOR = (
    Path(__file__).resolve().parents[2] / "areal/infra/utils/process_supervisor.py"
)
pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper")

# Double-fork and setsid reproduce workers that escape process-group cleanup.
WORKER = """
import json, os, signal, sys, time
from pathlib import Path
path, mode = sys.argv[1:]
if os.fork() == 0:
    os.setsid()
    if os.fork() != 0:
        os._exit(0)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    Path(path).write_text(json.dumps({'worker': int(os.environ['WORKER_PID']), 'orphan': os.getpid()}))
    while True:
        signal.pause()
os.wait()
while not Path(path).exists():
    time.sleep(0.01)
if mode == 'exit':
    sys.exit(7)
while True:
    signal.pause()
"""


def _wait_for_file(path):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if path.exists():
            try:
                return json.loads(path.read_text())
            except json.JSONDecodeError:
                pass
        time.sleep(0.01)
    raise AssertionError("worker did not publish its PID")


@pytest.mark.parametrize("shutdown", ["exit", "worker_kill", "term", "int", "hup"])
def test_supervisor_cleans_daemonized_descendants_and_preserves_status(
    tmp_path, shutdown
):
    """Detached TERM-resistant descendants are killed and reaped on every exit."""
    pid_file = tmp_path / "pids.json"
    worker = "import os; os.environ['WORKER_PID'] = str(os.getpid());\n" + WORKER
    mode = "exit" if shutdown == "exit" else "wait"
    # Matching Slurm environments must not cause an unrelated process to be killed.
    env = dict(os.environ, SLURM_JOB_ID="supervisor-test")
    outsider = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], env=env
    )
    process = subprocess.Popen(
        [
            sys.executable,
            str(SUPERVISOR),
            "--shutdown-timeout",
            "0.2",
            "--",
            sys.executable,
            "-c",
            worker,
            str(pid_file),
            mode,
        ],
        env=env,
    )
    pids = {}
    try:
        pids = _wait_for_file(pid_file)
        if shutdown == "worker_kill":
            os.kill(pids["worker"], signal.SIGKILL)
            expected = 137
        elif shutdown == "exit":
            expected = 7
        else:
            signum = {
                "term": signal.SIGTERM,
                "int": signal.SIGINT,
                "hup": signal.SIGHUP,
            }[shutdown]
            process.send_signal(signum)
            expected = 128 + signum
        assert process.wait(timeout=5) == expected
        assert not psutil.pid_exists(pids["orphan"])
        assert not psutil.pid_exists(pids["worker"])
        assert outsider.poll() is None
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
        for pid in pids.values():
            try:
                psutil.Process(pid).kill()
            except psutil.NoSuchProcess:
                pass
        outsider.terminate()
        outsider.wait(timeout=5)


def test_supervisor_cleans_children_created_during_shutdown(tmp_path):
    """A TERM handler cannot leave a newly created, detached child behind."""
    pid_file = tmp_path / "pids.json"
    ready_file = tmp_path / "ready.json"
    worker = """
import json, os, signal, sys
from pathlib import Path
def on_term(signum, frame):
    if os.fork() == 0:
        os.setsid()
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        Path(sys.argv[1]).write_text(json.dumps({'orphan': os.getpid()}))
        while True:
            signal.pause()
    sys.exit(0)
signal.signal(signal.SIGTERM, on_term)
Path(sys.argv[2]).write_text('{}')
while True:
    signal.pause()
"""
    process = subprocess.Popen(
        [
            sys.executable,
            str(SUPERVISOR),
            "--shutdown-timeout",
            "0.3",
            "--",
            sys.executable,
            "-c",
            worker,
            str(pid_file),
            str(ready_file),
        ]
    )
    pids = {}
    try:
        _wait_for_file(ready_file)
        process.terminate()
        pids = _wait_for_file(pid_file)
        assert process.wait(timeout=5) == 143
        assert not psutil.pid_exists(pids["orphan"])
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
        for pid in pids.values():
            try:
                psutil.Process(pid).kill()
            except psutil.NoSuchProcess:
                pass


def test_supervisor_preserves_output_environment_and_success():
    result = subprocess.run(
        [
            sys.executable,
            str(SUPERVISOR),
            "--",
            sys.executable,
            "-c",
            "import os,sys; sys.stdout.write(os.environ['SUPERVISOR_VALUE']); sys.stderr.write('error-stream')",
        ],
        env=dict(os.environ, SUPERVISOR_VALUE="space ' quote"),
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0
    assert result.stdout == "space ' quote"
    assert result.stderr == "error-stream"


def test_scheduler_cancel_allows_default_supervisor_cleanup(tmp_path, monkeypatch):
    """Scheduler escalation must not kill the subreaper before its default grace."""
    from areal.infra.scheduler.slurm import SlurmScheduler
    from areal.infra.utils.launcher import JobState

    pid_file = tmp_path / "pids.json"
    worker = "import os; os.environ['WORKER_PID'] = str(os.getpid());\n" + WORKER
    process = subprocess.Popen(
        [
            sys.executable,
            str(SUPERVISOR),
            "--",
            sys.executable,
            "-c",
            worker,
            str(pid_file),
            "wait",
        ]
    )
    scheduler = object.__new__(SlurmScheduler)
    scheduler._colocated_roles = {}
    scheduler._workers = {"actor": []}
    scheduler._jobs = {"actor": 1}
    scheduler._job_status_cache = {}
    scheduler._destroy_engines_on_workers = Mock()
    cancel = Mock(
        side_effect=lambda **kwargs: process.send_signal(
            signal.SIGTERM if kwargs["signal"] == "SIGTERM" else signal.SIGKILL
        )
    )
    monkeypatch.setattr("areal.infra.scheduler.slurm.cancel_jobs", cancel)
    monkeypatch.setattr(
        "areal.infra.scheduler.slurm.query_jobs",
        lambda **kwargs: []
        if process.poll() is not None
        else [SimpleNamespace(state=JobState.RUNNING)],
    )
    pids = {}
    try:
        pids = _wait_for_file(pid_file)
        scheduler.delete_workers("actor")
        assert process.wait(timeout=2) == 143
        assert not psutil.pid_exists(pids["orphan"])
        cancel.assert_called_once_with(slurm_ids=[1], signal="SIGTERM")
        assert scheduler._jobs == {}
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=8)
        for pid in pids.values():
            try:
                psutil.Process(pid).kill()
            except psutil.NoSuchProcess:
                pass
