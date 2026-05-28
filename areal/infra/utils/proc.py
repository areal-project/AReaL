# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import psutil

from areal.utils import logging
from areal.utils.logging import LOG_PREFIX_WIDTH

logger = logging.getLogger("ProcUtils")

if TYPE_CHECKING:
    from typing import IO


def build_streaming_log_cmd(
    cmd: str | list[str],
    log_file: str | Path,
    merged_log: str | Path,
    role: str,
    *,
    env_vars: dict[str, str] | None = None,
) -> str:
    """Build a shell command that streams output to stdout and log files.

    The command uses tee/sed pattern:
    - stdout streams to terminal in real-time
    - Output appends to role-specific log file (no prefix)
    - Output appends to merged log with [role] prefix

    Parameters
    ----------
    cmd : str | list[str]
        Command to execute. If list, will be shell-escaped and joined.
    log_file : str | Path
        Path to role-specific log file (appended with -a)
    merged_log : str | Path
        Path to merged log file (appended with prefix)
    role : str
        Role name for log prefix (e.g., "actor", "master")
    env_vars : dict[str, str] | None
        Optional environment variables to prefix the command with KEY=VALUE

    Returns
    -------
    str
        Shell command string ready for execution with bash
    """
    # Escape command if it's a list
    if isinstance(cmd, list):
        cmd_str = " ".join(shlex.quote(str(c)) for c in cmd)
    else:
        cmd_str = cmd

    # Check if stdbuf is available (not present on macOS by default)
    _has_stdbuf = shutil.which("stdbuf") is not None

    # Build prefix with env vars if provided
    prefix_parts = []
    if env_vars:
        prefix_parts.append(
            " ".join(f"{k}={shlex.quote(str(v))}" for k, v in env_vars.items())
        )
    if _has_stdbuf:
        prefix_parts.append(f"stdbuf -oL {cmd_str}")
    else:
        prefix_parts.append(cmd_str)
    full_cmd = " ".join(prefix_parts)

    # Build log prefix for merged log
    log_prefix = f"[{role}]".ljust(LOG_PREFIX_WIDTH)

    # Construct tee/sed pipeline
    if _has_stdbuf:
        sed_prefix = f"stdbuf -oL sed 's/^/{log_prefix}/'"
    else:
        sed_prefix = f"sed 's/^/{log_prefix}/'"
    shell_cmd = f"{full_cmd} 2>&1 | tee -a {log_file} >({sed_prefix} >> {merged_log})"
    return shell_cmd


def run_with_streaming_logs(
    cmd: str | list[str],
    log_file: str | Path,
    merged_log: str | Path,
    role: str,
    *,
    env: dict[str, str] | None = None,
    env_vars_in_cmd: dict[str, str] | None = None,
    stdout: IO | None = None,
) -> subprocess.Popen:
    """Run a command with streaming output to stdout and log files.

    Parameters
    ----------
    cmd : str | list[str]
        Command to execute
    log_file : str | Path
        Path to role-specific log file
    merged_log : str | Path
        Path to merged log file
    role : str
        Role name for log prefix
    env : dict[str, str] | None
        Environment dict for Popen (passed to subprocess)
    env_vars_in_cmd : dict[str, str] | None
        Environment variables to prefix in the shell command (KEY=VALUE format)
    stdout : IO | None
        File object for stdout. Defaults to sys.stdout

    Returns
    -------
    subprocess.Popen
        The spawned process
    """
    shell_cmd = build_streaming_log_cmd(
        cmd, str(log_file), str(merged_log), role, env_vars=env_vars_in_cmd
    )

    return subprocess.Popen(
        shell_cmd,
        shell=True,
        executable="/bin/bash",
        env=env,
        stdout=stdout or sys.stdout,
        stderr=stdout or sys.stdout,
    )


def kill_process_tree(
    parent_pid: int | None = None,
    timeout: int = 5,
    include_parent: bool = True,
    skip_pid: int | None = None,
    graceful: bool = True,
) -> None:
    # Remove sigchld handler to avoid spammy logs (only in main thread)
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)

    # Handle None parent_pid - defaults to current process
    if parent_pid is None:
        parent_pid = os.getpid()
        include_parent = False

    # Get process tree
    try:
        parent = psutil.Process(parent_pid)
        children = parent.children(recursive=True)
    except psutil.NoSuchProcess:
        logger.info(f"Process {parent_pid} already terminated")
        return

    # Filter skip_pid from children
    if skip_pid is not None:
        children = [c for c in children if c.pid != skip_pid]

    # Terminate based on mode
    if graceful:
        # Graceful mode: SIGTERM → wait → SIGKILL
        logger.info(
            f"Sending SIGTERM to process {parent_pid} and {len(children)} children"
        )

        # Send SIGTERM to all children
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass

        # Send SIGTERM to parent if requested
        if include_parent:
            try:
                parent.terminate()
            except psutil.NoSuchProcess:
                # Parent is already gone, but we still need to wait for children.
                pass

        # Wait for graceful shutdown
        procs_to_wait = children + ([parent] if include_parent else [])
        gone, alive = psutil.wait_procs(procs_to_wait, timeout=timeout)

        # Force kill any remaining processes
        if alive:
            logger.warning(
                f"Force killing {len(alive)} processes that didn't terminate gracefully"
            )
            for proc in alive:
                try:
                    proc.kill()
                except psutil.NoSuchProcess:
                    pass

            # Final wait to ensure they're gone
            psutil.wait_procs(alive, timeout=1)

        logger.info(f"Successfully cleaned up process tree for PID {parent_pid}")
    else:
        # Aggressive mode: immediate SIGKILL
        for child in children:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass

        if include_parent:
            try:
                if parent_pid == os.getpid():
                    parent.kill()
                    sys.exit(0)

                parent.kill()

                # Sometime processes cannot be killed with SIGKILL (e.g, PID=1 launched by kubernetes),
                # so we send an additional signal to kill them.
                parent.send_signal(signal.SIGQUIT)
            except psutil.NoSuchProcess:
                pass
