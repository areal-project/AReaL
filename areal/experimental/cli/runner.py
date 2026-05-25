# SPDX-License-Identifier: Apache-2.0

"""Driver resolution and process management for AReaL CLI subcommands.

This module is scheduler-agnostic: the CLI's responsibility is to (1) resolve
which driver function to invoke and (2) manage the driver process lifecycle.
Scheduler routing (local / slurm / ray) is performed *inside* the driver
itself based on ``config.scheduler.type``, exactly as today's hand-written
driver scripts do (see ``examples/experimental/inference_service/online_rollout.py``).
"""

from __future__ import annotations

import importlib
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from areal.experimental.cli.state import (
    RunState,
    log_path,
    pid_alive,
    state_path,
)

# The CLI must work in light environments where the full AReaL runtime
# (colorlog, torch, ...) isn't installed. Fall back to stdlib logging.
try:
    from areal.utils.logging import getLogger

    logger = getLogger("ArealCLI")
except Exception:  # pragma: no cover - fallback for thin installs
    import logging

    logger = logging.getLogger("ArealCLI")

DriverFn = Callable[[list[str]], Any]


# ---------------------------------------------------------------------------
# YAML peek helpers
#
# We need a few top-level fields (`driver`, `experiment_name`, `trial_name`,
# `scheduler.type`) before the driver itself loads the config via Hydra. Read
# the raw YAML for these without going through OmegaConf so we don't trigger
# the structured-config validation early.
# ---------------------------------------------------------------------------


def _raw_yaml(config_path: Path) -> dict[str, Any]:
    with open(config_path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"Top-level of {config_path} must be a YAML mapping.")
    return data


def _peek_driver(config_path: Path) -> str | None:
    return _raw_yaml(config_path).get("driver")


def _peek_scheduler_type(config_path: Path) -> str | None:
    sched = _raw_yaml(config_path).get("scheduler") or {}
    if isinstance(sched, dict):
        return sched.get("type")
    return None


def _peek_name(config_path: Path) -> str | None:
    raw = _raw_yaml(config_path)
    exp = raw.get("experiment_name")
    trial = raw.get("trial_name")
    if exp and trial:
        return f"{exp}/{trial}"
    return None


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def _import_driver(spec: str) -> DriverFn:
    """Resolve 'module.path:func' to a callable."""
    if ":" not in spec:
        raise SystemExit(
            f"Invalid --driver value {spec!r}; expected 'module.path:func'."
        )
    mod_path, func_name = spec.split(":", 1)
    try:
        mod = importlib.import_module(mod_path)
    except ImportError as e:
        raise SystemExit(f"Cannot import driver module {mod_path!r}: {e}") from e
    fn = getattr(mod, func_name, None)
    if fn is None:
        raise SystemExit(f"Module {mod_path!r} has no attribute {func_name!r}.")
    if not callable(fn):
        raise SystemExit(f"{spec!r} is not callable.")
    return fn


def resolve_driver(
    config_path: Path,
    cli_driver: str | None,
    fallback: str | None = None,
    command_hint: str = "run",
) -> str:
    """Pick the driver spec by priority: --driver > yaml `driver:` > fallback."""
    if cli_driver:
        return cli_driver
    yaml_driver = _peek_driver(config_path)
    if yaml_driver:
        return yaml_driver
    if fallback:
        return fallback
    raise SystemExit(
        f"No driver specified for `areal {command_hint}`. "
        f"Either pass --driver MOD:FUNC, or add a top-level "
        f"`driver: module.path:func` field to {config_path}."
    )


def resolve_name(config_path: Path, cli_name: str | None) -> str:
    if cli_name:
        return cli_name
    yaml_name = _peek_name(config_path)
    if yaml_name:
        return yaml_name
    raise SystemExit(
        f"No --name given and `experiment_name`/`trial_name` not both present "
        f"in {config_path}."
    )


# ---------------------------------------------------------------------------
# Foreground & background execution
# ---------------------------------------------------------------------------


def _refuse_if_active(name: str, command: str) -> None:
    p = state_path(name)
    if not p.exists():
        return
    try:
        existing = RunState.load(name)
    except (FileNotFoundError, ValueError):
        return
    if pid_alive(existing.pid):
        raise SystemExit(
            f"Run {name!r} already active "
            f"(pid={existing.pid}, command={existing.command}). "
            f"Use `areal {existing.command} stop {name}` first."
        )


def run_foreground(
    name: str,
    command: str,
    driver_spec: str,
    config_path: Path,
    overrides: list[str],
) -> int:
    """Resolve the driver and invoke it in this process. Returns exit code."""
    _refuse_if_active(name, command)

    sched_type = _peek_scheduler_type(config_path)
    argv = ["--config", str(config_path)] + list(overrides)

    state = RunState(
        name=name,
        command=command,
        driver=driver_spec,
        config_path=str(config_path),
        pid=os.getpid(),
        started_at=time.time(),
        status="running",
        scheduler_type=sched_type,
        argv=argv,
    )
    state.save()

    logger.info(
        "[%s] starting (driver=%s, scheduler=%s, name=%s, pid=%d)",
        command,
        driver_spec,
        sched_type,
        name,
        os.getpid(),
    )

    rc = 0
    try:
        fn = _import_driver(driver_spec)
        result = fn(argv)
        if isinstance(result, int):
            rc = result
    except SystemExit as e:
        rc = int(e.code) if isinstance(e.code, int) else 1
    except BaseException:
        state.status = "failed"
        state.save()
        raise

    state.status = "completed" if rc == 0 else "failed"
    state.save()
    return rc


def start_background(
    name: str,
    command: str,
    driver_spec: str,
    config_path: Path,
    overrides: list[str],
) -> None:
    """Spawn a detached subprocess that runs the driver via ``areal-exec``."""
    _refuse_if_active(name, command)

    sched_type = _peek_scheduler_type(config_path)
    argv = ["--config", str(config_path)] + list(overrides)

    log_file = log_path(name)
    cmd = [
        sys.executable,
        "-m",
        "areal.experimental.cli._exec",
        "--driver",
        driver_spec,
        "--config",
        str(config_path),
        "--name",
        name,
        "--command",
        command,
    ]
    if overrides:
        cmd.append("--")
        cmd.extend(overrides)

    with open(log_file, "ab", buffering=0) as lf:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=lf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=os.environ.copy(),
        )

    state = RunState(
        name=name,
        command=command,
        driver=driver_spec,
        config_path=str(config_path),
        pid=proc.pid,
        started_at=time.time(),
        status="running",
        log_path=str(log_file),
        scheduler_type=sched_type,
        argv=argv,
    )
    state.save()

    logger.info(
        "[%s] started in background (pid=%d, name=%s, log=%s)",
        command,
        proc.pid,
        name,
        log_file,
    )
    print(f"Started {name} (pid {proc.pid})")
    print(f"  log:   {log_file}")
    print(f"  state: {state_path(name)}")
    print(f"  stop:  areal {command} stop {name}")


def stop_run(name: str, command_hint: str, timeout: float = 15.0) -> int:
    """Send SIGTERM (then SIGKILL after timeout) to a recorded run's pgroup."""
    try:
        state = RunState.load(name)
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from e

    if state.command != command_hint and command_hint != "run":
        raise SystemExit(
            f"{name!r} was started via `areal {state.command} start`, "
            f"not `areal {command_hint}`. "
            f"Use `areal {state.command} stop {name}` instead."
        )

    if not pid_alive(state.pid):
        state.status = "stopped"
        state.save()
        print(f"{name} not running (pid {state.pid} dead).")
        return 0

    _signal(state.pid, signal.SIGTERM)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not pid_alive(state.pid):
            break
        time.sleep(0.5)
    else:
        logger.warning(
            "[stop] SIGTERM timed out after %.1fs; sending SIGKILL to pid %d.",
            timeout,
            state.pid,
        )
        _signal(state.pid, signal.SIGKILL)

    state.status = "stopped"
    state.save()
    print(f"Stopped {name} (pid {state.pid}).")
    return 0


def _signal(pid: int, sig: int) -> None:
    """Signal the whole process group; fall back to direct pid on failure."""
    try:
        os.killpg(os.getpgid(pid), sig)
        return
    except ProcessLookupError:
        return
    except (PermissionError, OSError):
        pass
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        pass
