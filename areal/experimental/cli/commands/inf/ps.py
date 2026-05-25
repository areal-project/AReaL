# SPDX-License-Identifier: Apache-2.0

"""``areal inf ps`` — list all locally known inference services."""

from __future__ import annotations

import argparse
import json

from areal.experimental.cli.commands.inf._common import print_table
from areal.experimental.cli.gateway_client import (
    GatewayClient,
    GatewayError,
)
from areal.experimental.cli.inf_state import (
    get_current_service,
    liveness_summary,
    load_all_services,
)


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "ps",
        help="List services tracked in ~/.areal/inf/services/.",
        description=(
            "Read every state file under ~/.areal/inf/services/, probe PIDs "
            "(and optionally the gateway /health), and print a one-line "
            "summary per service."
        ),
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output.",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Include stale/unhealthy instances (default already includes them).",
    )
    p.add_argument(
        "--no-probe",
        action="store_true",
        help="Skip live HTTP probes (faster; show PID-liveness only).",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=1.0,
        help="Per-service health probe timeout (s).",
    )
    p.set_defaults(func=_handle)


def _service_row(state, probe: bool, timeout: float, current: str | None) -> dict:
    live = liveness_summary(state)
    status = "alive" if live["gateway_pid_alive"] and live["router_pid_alive"] else "stale"
    health: str = "?"
    if probe:
        try:
            GatewayClient(state.gateway_url, timeout=timeout).health()
            health = "ok"
            status = "running"
        except GatewayError:
            health = "down"
    return {
        "service": state.name,
        "default": state.name == current,
        "status": status,
        "health": health,
        "gateway": state.gateway_url,
        "router": state.router_url,
        "gateway_pid": state.gateway_pid,
        "router_pid": state.router_pid,
        "mode": state.mode,
        "created_at": state.created_at,
    }


def _handle(args: argparse.Namespace) -> int:
    services = load_all_services()
    current = get_current_service()
    rows = [
        _service_row(s, probe=not args.no_probe, timeout=args.timeout, current=current)
        for s in services
    ]

    if args.json:
        print(json.dumps(rows, indent=2, default=str))
        return 0

    if not rows:
        print("No inference services. Start one with `areal inf run`.")
        return 0

    table = [
        [
            "*" + r["service"] if r["default"] else " " + r["service"],
            r["status"],
            r["health"],
            r["gateway"],
            f"gw={r['gateway_pid']} rt={r['router_pid']}",
        ]
        for r in rows
    ]
    print_table(["SERVICE", "STATUS", "HEALTH", "GATEWAY", "PIDS"], table)
    return 0
