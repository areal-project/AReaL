# SPDX-License-Identifier: Apache-2.0

"""``areal inf status`` — detail for a single inference service."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict


_DESCRIPTION = __doc__


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "status",
        help="Show status for one inference service.",
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("name", nargs="?", help="Service name (defaults to current).")
    p.add_argument(
        "--json", action="store_true", dest="as_json",
        help="Emit raw state as JSON.",
    )
    p.set_defaults(func=_handle)


def _handle(args: argparse.Namespace) -> int:
    from areal.experimental.cli.inf_state import (
        ServiceState,
        get_current_service,
        service_logs_dir,
        supervisor_alive,
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

    if args.as_json:
        print(json.dumps(asdict(state), indent=2))
        return 0

    alive = supervisor_alive(state)
    age = int(max(0, time.time() - state.created_at))
    print(f"Service:    {state.name}")
    print(f"State:      {'running' if alive else 'dead'}")
    print(f"Age:        {age}s")
    print(f"Supervisor: pid={state.supervisor_pid}")
    print(f"Config:     {state.config_path}")
    if state.overrides:
        print(f"Overrides:  {' '.join(state.overrides)}")
    print(f"Gateway:    {state.gateway_addr or '-'}")
    print(f"Router:     {state.router_addr or '-'}")
    if state.server_addrs:
        print(f"Servers:    {len(state.server_addrs)}")
        for addr in state.server_addrs:
            print(f"  - {addr}")
    else:
        print("Servers:    (none)")
    log_dir = service_logs_dir(state.name)
    print(f"Logs:       {log_dir}")
    return 0
