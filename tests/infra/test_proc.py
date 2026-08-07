from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import psutil
import pytest

from areal.infra.utils.proc import kill_process_group, run_with_streaming_logs

MODULE = "areal.infra.utils.proc"


def _run_with_mocked_popen(tmp_path: Path, *, isolate: bool):
    with patch(f"{MODULE}.subprocess.Popen") as popen:
        run_with_streaming_logs(
            ["python", "-c", "print('ok')"],
            tmp_path / "role.log",
            tmp_path / "merged.log",
            "worker",
            isolate_process_group=isolate,
        )
    return popen.call_args.kwargs


def test_run_with_streaming_logs_preserves_default_process_group(tmp_path: Path):
    kwargs = _run_with_mocked_popen(tmp_path, isolate=False)
    assert "process_group" not in kwargs


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups required")
def test_run_with_streaming_logs_can_create_isolated_process_group(tmp_path: Path):
    kwargs = _run_with_mocked_popen(tmp_path, isolate=True)
    assert kwargs["process_group"] == 0


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups required")
def test_kill_process_group_escalates_to_sigkill():
    member = MagicMock(pid=123)
    with (
        patch(f"{MODULE}.os.getpgrp", return_value=999),
        patch(f"{MODULE}._get_process_group_members", return_value=[member]),
        patch(
            f"{MODULE}._wait_for_process_group_exit",
            side_effect=[[123], []],
        ),
        patch(f"{MODULE}.os.killpg") as killpg,
    ):
        kill_process_group(123, timeout=0.1, graceful=True)

    assert killpg.call_args_list == [
        call(123, signal.SIGTERM),
        call(123, signal.SIGKILL),
    ]


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups required")
def test_kill_process_group_rejects_current_group():
    with (
        patch(f"{MODULE}.os.getpgrp", return_value=123),
        pytest.raises(RuntimeError, match="current process group"),
    ):
        kill_process_group(123)


def _is_live_non_zombie(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups required")
def test_kill_process_group_survives_shell_leader_exit():
    code = (
        "import subprocess, sys; "
        "child = subprocess.Popen([sys.executable, '-c', "
        "'import time; time.sleep(60)'], "
        "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); "
        "print(child.pid, flush=True)"
    )
    leader = subprocess.Popen(
        [sys.executable, "-c", code],
        process_group=0,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = leader.communicate(timeout=5)
    assert leader.returncode == 0, stderr
    child_pid = int(stdout.strip())
    assert _is_live_non_zombie(child_pid)
    assert os.getpgid(child_pid) == leader.pid

    try:
        kill_process_group(leader.pid, timeout=1, graceful=True)
        assert not _is_live_non_zombie(child_pid)
    finally:
        if _is_live_non_zombie(child_pid):
            os.kill(child_pid, signal.SIGKILL)


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups required")
def test_kill_process_group_force_kills_sigterm_ignoring_member():
    code = (
        "import signal, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "print('ready', flush=True); "
        "time.sleep(60)"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", code],
        process_group=0,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "ready"
        kill_process_group(process.pid, timeout=0.1, graceful=True)
        process.wait(timeout=2)
        assert process.returncode == -signal.SIGKILL
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=2)
