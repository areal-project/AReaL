# SPDX-License-Identifier: Apache-2.0

"""``areal inf logs`` — tail the supervisor's main.log for a service."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


_DESCRIPTION = __doc__


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "logs",
        help="Tail the main log for a service.",
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("name", nargs="?", help="Service name (defaults to current).")
    p.add_argument(
        "--component", default="main",
        help="Log file component (default: main). Examples: main, merged, "
             "inf-server, router, gateway. Becomes `<component>.log` in the log dir.",
    )
    p.add_argument(
        "-f", "--follow", action="store_true",
        help="Follow the log (tail -F).",
    )
    p.add_argument(
        "-n", "--lines", type=int, default=200,
        help="Number of trailing lines to print (default 200).",
    )
    p.set_defaults(func=_handle)


def _handle(args: argparse.Namespace) -> int:
    from areal.experimental.cli.inf_state import (
        ServiceState,
        get_current_service,
    )

    name = args.name or get_current_service()
    if not name:
        print(
            "No service name given and no current service set. "
            "Use `areal inf ps` to list services.",
            file=sys.stderr,
        )
        return 2

    try:
        state = ServiceState.load(name)
    except FileNotFoundError:
        print(f"No service named {name!r}.", file=sys.stderr)
        return 1

    if not state.log_dir:
        print(f"Service {name!r} has no log_dir recorded.", file=sys.stderr)
        return 1
    log_file = Path(state.log_dir) / f"{args.component}.log"
    if not log_file.exists():
        print(f"No {args.component}.log at {log_file}.", file=sys.stderr)
        return 1

    cmd = ["tail", f"-n{args.lines}"]
    if args.follow:
        cmd.append("-F")
    cmd.append(str(log_file))
    os.execvp(cmd[0], cmd)
