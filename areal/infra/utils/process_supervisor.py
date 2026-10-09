# SPDX-License-Identifier: Apache-2.0

"""Supervise one Slurm task, including descendants that daemonize.

Run this file directly to keep the supervisor independent of AReaL's heavy
package imports. Linux reparents orphaned descendants to this subreaper, so
they remain discoverable even after their worker or intermediate parent dies.
SIGKILL of the supervisor itself requires Slurm's cgroup process tracking.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import signal
import subprocess
import sys
import time

import psutil

_PR_SET_CHILD_SUBREAPER = 36
DEFAULT_SHUTDOWN_TIMEOUT = 5
KILL_TIMEOUT = 2


def _enable_subreaper() -> None:
    if sys.platform != "linux":
        raise RuntimeError("Slurm process supervision requires Linux")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int] + [ctypes.c_ulong] * 4
    libc.prctl.restype = ctypes.c_int
    if libc.prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def _reap_children() -> dict[int, int]:
    statuses = {}
    while True:
        try:
            pid, status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            break
        if pid == 0:
            break
        statuses[pid] = os.waitstatus_to_exitcode(status)
    return statuses


def _cleanup_descendants(shutdown_timeout: float) -> dict[int, int]:
    parent = psutil.Process()
    statuses = {}
    # Refresh the tree throughout shutdown: TERM handlers may spawn children,
    # and descendants may be reparented as intermediate processes exit.
    for signum, timeout in (
        (signal.SIGTERM, shutdown_timeout),
        (signal.SIGKILL, KILL_TIMEOUT),
    ):
        deadline = time.monotonic() + timeout
        signalled = set()
        while True:
            statuses.update(_reap_children())
            children = parent.children(recursive=True)
            if not children:
                return statuses
            for child in children:
                try:
                    identity = (child.pid, child.create_time())
                    if identity not in signalled:
                        # psutil verifies process identity before sending a signal.
                        child.send_signal(signum)
                        signalled.add(identity)
                except psutil.NoSuchProcess:
                    pass
            if time.monotonic() >= deadline:
                break
            time.sleep(0.05)
    statuses.update(_reap_children())
    if parent.children(recursive=True):
        raise RuntimeError("Slurm task descendants survived shutdown")
    return statuses


def supervise(
    command: list[str], shutdown_timeout: float = DEFAULT_SHUTDOWN_TIMEOUT
) -> int:
    """Run a command and clean only this task's descendants on exit or signal."""
    _enable_subreaper()
    received_signal = None

    def handle_signal(signum, frame):
        nonlocal received_signal
        if received_signal is None:
            received_signal = signum

    handled_signals = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
    previous_handlers = {s: signal.signal(s, handle_signal) for s in handled_signals}
    previous_sigchld = signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    process = None
    returncode = None
    try:
        if received_signal is None:
            process = subprocess.Popen(command)
            while returncode is None and received_signal is None:
                returncode = _reap_children().get(process.pid)
                if returncode is None:
                    time.sleep(0.05)
    finally:
        try:
            statuses = _cleanup_descendants(shutdown_timeout)
            if process is not None:
                if returncode is None:
                    returncode = statuses.get(process.pid)
                process.returncode = returncode
        finally:
            signal.signal(signal.SIGCHLD, previous_sigchld)
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)

    if received_signal is not None:
        return 128 + received_signal
    if returncode is None:
        raise RuntimeError("Could not obtain Slurm worker exit status")
    return returncode if returncode >= 0 else 128 - returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shutdown-timeout", type=float, default=DEFAULT_SHUTDOWN_TIMEOUT
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command or args.shutdown_timeout < 0:
        parser.error("a command and a nonnegative shutdown timeout are required")
    sys.exit(supervise(command, args.shutdown_timeout))


if __name__ == "__main__":
    main()
