# SPDX-License-Identifier: Apache-2.0

"""``areal train status`` — show one training run's recorded state + PID liveness."""

from __future__ import annotations

import argparse
import json
import time

from areal.experimental.cli.state import RunState, pid_alive


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "status",
        help="Show a single training run's recorded state.",
        description=(
            "Load the state file for <run_name> from ~/.areal/runs/, probe "
            "PID liveness, and print a compact summary. PID-liveness can "
            "differ from the recorded status field if the driver crashed "
            "without updating its state."
        ),
        aliases=["health"],
    )
    p.add_argument("run_name", help="Run name (typically experiment_name/trial_name).")
    p.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    p.set_defaults(func=_handle)


def _handle(args: argparse.Namespace) -> int:
    try:
        state = RunState.load(args.run_name)
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from e

    alive = pid_alive(state.pid)
    payload = {
        "name": state.name,
        "command": state.command,
        "status": state.status,
        "pid": state.pid,
        "pid_alive": alive,
        "driver": state.driver,
        "config_path": state.config_path,
        "scheduler_type": state.scheduler_type,
        "log_path": state.log_path,
        "started_at": state.started_at,
        "argv": state.argv,
    }

    if args.json:
        print(json.dumps(payload, indent=2, default=str))
        return 0 if alive or state.status in ("completed",) else 1

    print(f"Run:       {state.name}")
    print(f"  command:    {state.command}")
    print(f"  status:     {state.status}{'' if alive else '  (pid dead)'}")
    print(f"  pid:        {state.pid}")
    print(f"  driver:     {state.driver}")
    print(f"  config:     {state.config_path}")
    print(f"  scheduler:  {state.scheduler_type or '-'}")
    print(f"  log:        {state.log_path or '-'}")
    print(
        f"  started:    "
        f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(state.started_at))}"
    )
    if state.argv:
        print(f"  argv:       {' '.join(state.argv)}")
    return 0 if alive or state.status == "completed" else 1
