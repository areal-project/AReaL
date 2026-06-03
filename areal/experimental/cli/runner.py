# SPDX-License-Identifier: Apache-2.0

"""Driver resolution and foreground execution for ``areal run``."""

from __future__ import annotations

import importlib
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from areal.experimental.cli.state import RunState, pid_alive, run_state_path


DriverFn = Callable[[list[str]], Any]


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
    return sched.get("type") if isinstance(sched, dict) else None


def _peek_name(config_path: Path) -> str | None:
    raw = _raw_yaml(config_path)
    exp = raw.get("experiment_name")
    trial = raw.get("trial_name")
    return f"{exp}/{trial}" if exp and trial else None


def _import_driver(spec: str) -> DriverFn:
    if ":" not in spec:
        raise SystemExit(f"Invalid driver {spec!r}; expected 'module.path:func'.")
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


def resolve_driver(config_path: Path, cli_driver: str | None) -> str:
    if cli_driver:
        return cli_driver
    yaml_driver = _peek_driver(config_path)
    if yaml_driver:
        return yaml_driver
    raise SystemExit(
        f"No driver specified.\n"
        f"  Either add a `driver:` field to {config_path}:\n"
        f"      driver: examples.math.gsm8k_rl:main\n"
        f"  Or pass --driver on the command line:\n"
        f"      areal run --config {config_path} --driver examples.math.gsm8k_rl:main"
    )


def resolve_name(config_path: Path, cli_name: str | None) -> str:
    if cli_name:
        return cli_name
    n = _peek_name(config_path)
    if n:
        return n
    raise SystemExit(
        f"No --name given and `experiment_name`/`trial_name` not both present in {config_path}."
    )


def _refuse_if_active(name: str) -> None:
    p = run_state_path(name)
    if not p.exists():
        return
    try:
        existing = RunState.load(name)
    except (FileNotFoundError, ValueError):
        return
    if pid_alive(existing.pid):
        raise SystemExit(
            f"Run {name!r} already active (pid={existing.pid}). "
            f"Use `areal stop {name}` first."
        )


def run_foreground(
    *, name: str, driver_spec: str, config_path: Path, overrides: list[str]
) -> int:
    _refuse_if_active(name)

    argv = ["--config", str(config_path)] + list(overrides)
    state = RunState(
        name=name,
        driver=driver_spec,
        config_path=str(config_path),
        pid=os.getpid(),
        started_at=time.time(),
        scheduler_type=_peek_scheduler_type(config_path),
        overrides=list(overrides),
        argv=argv,
    )
    state.save()

    rc = 0
    try:
        fn = _import_driver(driver_spec)
        result = fn(argv)
        if isinstance(result, int):
            rc = result
    except SystemExit as e:
        if isinstance(e.code, int):
            rc = e.code
        elif e.code is not None:
            print(str(e.code), file=sys.stderr)
            rc = 1
    except BaseException:
        state.status = "failed"
        state.save()
        raise

    state.status = "completed" if rc == 0 else "failed"
    state.save()
    return rc
