# SPDX-License-Identifier: Apache-2.0

"""Local-only process primitives for service-style CLIs.

These helpers operate on local PIDs and process trees. Remote backends
(K8s / Slurm) should not consume them directly — route teardown through
``areal.v2.cli.scheduler.terminate`` instead, which dispatches
per backend.

For port allocation use ``areal.utils.network.find_free_ports``; it
draws from a non-ephemeral range and supports excluding already-handed-
out ports, avoiding the TOCTOU collisions a naive ``bind(0)`` walks
into.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from areal.infra.utils.proc import kill_process_tree


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


def kill_pids(pids: list[int], grace_s: float) -> None:
    """SIGTERM → wait ``grace_s`` → SIGKILL across local pids and their
    descendant process trees.

    Delegates per-pid to ``areal.infra.utils.proc.kill_process_tree``,
    which walks ``psutil.Process(pid).children(recursive=True)`` so
    grandchildren that escaped the original process group are still
    caught. Local-only — for remote backends use
    ``areal.v2.cli.scheduler.terminate``.
    """

    for pid in pids:
        if pid > 0:
            kill_process_tree(pid, timeout=int(grace_s), graceful=True)
