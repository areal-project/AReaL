# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import signal
import socket
import subprocess
import time
from pathlib import Path


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def spawn_process(
    cmd: list[str], log_file: Path, env: dict[str, str] | None = None
) -> int:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(log_file, "ab", buffering=0)
    final_env = os.environ.copy()
    final_env.setdefault("PYTHONUNBUFFERED", "1")
    if env:
        final_env.update(env)
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=final_env,
    )
    return proc.pid


def signal_pid(pid: int, sig: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.killpg(os.getpgid(pid), sig)
        return True
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        pass
    try:
        os.kill(pid, sig)
        return True
    except ProcessLookupError:
        return False


def kill_pids(pids: list[int], grace_s: float) -> None:
    pids = [p for p in pids if p > 0]
    if not pids:
        return
    for pid in pids:
        if pid_alive(pid):
            signal_pid(pid, signal.SIGTERM)
    deadline = time.time() + grace_s
    while time.time() < deadline:
        if not any(pid_alive(pid) for pid in pids):
            return
        time.sleep(0.2)
    for pid in pids:
        if pid_alive(pid):
            signal_pid(pid, signal.SIGKILL)
