# SPDX-License-Identifier: Apache-2.0

"""``areal train ps`` — list training runs recorded under ~/.areal/runs/.

Filters the global run registry by ``command=train``. Use ``--all-commands``
to show runs from every CLI subcommand (``run``, ``inf``, ``agent``, ...).
"""

from __future__ import annotations

import argparse
import json
import time

from areal.experimental.cli.commands.inf._common import print_table
from areal.experimental.cli.state import load_all_runs, pid_alive


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "ps",
        help="List training runs tracked in ~/.areal/runs/.",
        description=(
            "Read every state file under ~/.areal/runs/, probe PID liveness, "
            "and print a one-line summary per run. Filtered to "
            "command=train unless --all-commands is given."
        ),
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output.",
    )
    p.add_argument(
        "--all-commands",
        action="store_true",
        help="Show runs from every command, not just `train`.",
    )
    p.add_argument(
        "--status",
        default=None,
        choices=["running", "stopped", "completed", "failed"],
        help="Filter by recorded status field.",
    )
    p.set_defaults(func=_handle)


def _row(state) -> dict:
    alive = pid_alive(state.pid)
    return {
        "name": state.name,
        "command": state.command,
        "status": state.status,
        "alive": alive,
        "pid": state.pid,
        "driver": state.driver,
        "scheduler": state.scheduler_type or "-",
        "log_path": state.log_path or "-",
        "started_at": state.started_at,
    }


def _handle(args: argparse.Namespace) -> int:
    rows = [
        _row(s)
        for s in load_all_runs()
        if (args.all_commands or s.command == "train")
        and (args.status is None or s.status == args.status)
    ]

    if args.json:
        print(json.dumps(rows, indent=2, default=str))
        return 0

    if not rows:
        scope = "any command" if args.all_commands else "command=train"
        print(f"No runs ({scope}). Start one with `areal train start --config ...`.")
        return 0

    table = [
        [
            r["name"],
            r["command"],
            r["status"] + ("" if r["alive"] else " (dead)"),
            str(r["pid"]),
            r["scheduler"],
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["started_at"])),
            r["driver"],
        ]
        for r in rows
    ]
    print_table(
        ["NAME", "COMMAND", "STATUS", "PID", "SCHEDULER", "STARTED", "DRIVER"], table
    )
    return 0
