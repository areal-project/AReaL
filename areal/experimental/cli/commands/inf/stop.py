# SPDX-License-Identifier: Apache-2.0

"""``areal inf stop`` — destroy an inference service instance."""

from __future__ import annotations

import argparse

from areal.experimental.cli.inf_launcher import stop_service
from areal.experimental.cli.inf_state import (
    get_current_service,
    resolve_service_name,
    set_current_service,
)


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "stop",
        help="Stop a running inference service.",
        description=(
            "Send SIGTERM to gateway + router (then SIGKILL after the grace "
            "period) and remove the local state file. Per-model backend "
            "processes (data proxies, inference servers) are handled by "
            "`areal inf deregister`, not this command."
        ),
        aliases=["destroy"],
    )
    p.add_argument(
        "--service",
        default=None,
        help="Service instance name (default: current-service file, or sole known).",
    )
    p.add_argument(
        "--grace-period",
        type=float,
        default=10.0,
        help="Seconds to wait for graceful shutdown before SIGKILL.",
    )
    p.add_argument(
        "--keep-state",
        action="store_true",
        help="Keep the state file on disk for debugging.",
    )
    p.set_defaults(func=_handle)


def _handle(args: argparse.Namespace) -> int:
    name = resolve_service_name(args.service)
    rc = stop_service(name, grace_period=args.grace_period, keep_state=args.keep_state)
    if get_current_service() == name and not args.keep_state:
        set_current_service(None)
    print(f"Stopped service {name!r}.")
    return rc
