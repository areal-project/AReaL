# SPDX-License-Identifier: Apache-2.0

"""``areal agent stop`` — destroy an agent service instance."""

from __future__ import annotations

import argparse

from areal.experimental.cli.agent_launcher import stop_agent_service
from areal.experimental.cli.agent_state import (
    get_current_agent_service,
    resolve_agent_service_name,
    set_current_agent_service,
)


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "stop",
        help="Stop a running agent service.",
        description=(
            "Unregister each data_proxy from the router, then SIGTERM all "
            "tracked PIDs (router, gateway, every worker+proxy) and SIGKILL "
            "stragglers after the grace period. Removes the local state file."
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
    name = resolve_agent_service_name(args.service)
    rc = stop_agent_service(name, grace_period=args.grace_period, keep_state=args.keep_state)
    if get_current_agent_service() == name and not args.keep_state:
        set_current_agent_service(None)
    print(f"Stopped agent service {name!r}.")
    return rc
