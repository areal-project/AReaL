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
    # If ``pid`` is one of our own children we may catch it mid-zombie:
    # ``os.kill(pid, 0)`` still succeeds for zombies, so an unreaped child
    # would look alive forever and kill_pids would burn its full grace
    # window before sending a redundant SIGKILL. Reap it first via WNOHANG
    # so a zombie is reported dead immediately.
    try:
        reaped, _ = os.waitpid(pid, os.WNOHANG)
        if reaped == pid:
            return False
    except (ChildProcessError, OSError):
        # Not our child, or already reaped — fall through to the kill probe.
        pass
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
    """Spawn a detached subprocess that survives parent exit.

    ``start_new_session=True`` puts the child in its own session so the
    parent receiving SIGHUP (terminal close) does not propagate to the
    child. stdout / stderr are appended to ``log_file``. Extra env vars
    in ``env`` are merged on top of the parent environment.
    """

    log_file.parent.mkdir(parents=True, exist_ok=True)
    final_env = os.environ.copy()
    final_env.setdefault("PYTHONUNBUFFERED", "1")
    if env:
        final_env.update(env)
    # Popen dup()s the fd for the child, so closing our copy here does not
    # affect the child's stdout/stderr — and it stops us from leaking a fd
    # in the parent every time spawn_process is called.
    with open(log_file, "ab", buffering=0) as log_handle:
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
    """Best-effort signal delivery, first to the process group then the bare pid."""

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
    """SIGTERM all pids, wait up to ``grace_s`` for them to exit, then SIGKILL stragglers."""

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
