# SPDX-License-Identifier: Apache-2.0

"""``areal inf ps`` — list locally tracked inference services."""

from __future__ import annotations

import argparse
import sys
import time


_DESCRIPTION = __doc__


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "ps",
        help="List locally tracked inference services.",
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.set_defaults(func=_handle)


def _handle(args: argparse.Namespace) -> int:
    from areal.experimental.cli.inf_state import (
        ServiceState,
        get_current_service,
        services_dir,
        supervisor_alive,
    )

    current = get_current_service()
    states: list[ServiceState] = []
    for f in sorted(services_dir().glob("*.json")):
        name = f.stem.replace("__", "/")
        try:
            states.append(ServiceState.load(name))
        except (ValueError, TypeError, KeyError):
            continue
    if not states:
        print("No inference services.", file=sys.stderr)
        return 0

    cols = ("CURRENT", "NAME", "STATE", "AGE", "SUPERVISOR_PID", "GATEWAY")
    rows = []
    now = time.time()
    for s in states:
        alive = supervisor_alive(s)
        age = int(max(0, now - s.created_at))
        rows.append(
            (
                "*" if s.name == current else "",
                s.name,
                "running" if alive else "dead",
                f"{age}s",
                str(s.supervisor_pid),
                s.gateway_addr or "-",
            )
        )

    widths = [max(len(r[i]) for r in (cols, *rows)) for i in range(len(cols))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*cols))
    for r in rows:
        print(fmt.format(*r))
    return 0
