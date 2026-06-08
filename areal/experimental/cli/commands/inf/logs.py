# SPDX-License-Identifier: Apache-2.0

"""``areal inf logs`` — tail the supervisor log for a service."""

from __future__ import annotations

import argparse
import os
import sys


_DESCRIPTION = __doc__


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "logs",
        help="Tail the supervisor log for a service.",
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("name", nargs="?", help="Service name (defaults to current).")
    p.add_argument(
        "-f", "--follow", action="store_true",
        help="Follow the log (tail -f).",
    )
    p.add_argument(
        "-n", "--lines", type=int, default=200,
        help="Number of trailing lines to print (default 200).",
    )
    p.set_defaults(func=_handle)


def _handle(args: argparse.Namespace) -> int:
    from areal.experimental.cli.inf_state import (
        get_current_service,
        service_logs_dir,
    )

    name = args.name or get_current_service()
    if not name:
        print(
            "No service name given and no current service set. "
            "Use `areal inf ps` to list services.",
            file=sys.stderr,
        )
        return 2

    log_file = service_logs_dir(name) / "supervisor.log"
    if not log_file.exists():
        print(f"No supervisor log at {log_file}.", file=sys.stderr)
        return 1

    cmd = ["tail", f"-n{args.lines}"]
    if args.follow:
        cmd.append("-F")
    cmd.append(str(log_file))
    os.execvp(cmd[0], cmd)
