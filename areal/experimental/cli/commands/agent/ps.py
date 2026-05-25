# SPDX-License-Identifier: Apache-2.0

"""``areal agent ps`` — list locally known agent services."""

from __future__ import annotations

import argparse
import json

from areal.experimental.cli.agent_gateway_client import (
    AgentGatewayClient,
    AgentGatewayError,
)
from areal.experimental.cli.agent_state import (
    get_current_agent_service,
    liveness_summary,
    load_all_agent_services,
)
from areal.experimental.cli.commands.agent._common import print_table


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "ps",
        help="List agent services tracked in ~/.areal/agent/services/.",
        description=(
            "Read every state file under ~/.areal/agent/services/, probe PIDs "
            "(and optionally the agent gateway /health), and print a one-line "
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
    workers_alive = sum(1 for x in live["worker_pids_alive"] if x)
    proxies_alive = sum(1 for x in live["proxy_pids_alive"] if x)
    overall_alive = (
        live["gateway_pid_alive"]
        and live["router_pid_alive"]
        and workers_alive == state.num_pairs
        and proxies_alive == state.num_pairs
    )
    status = "alive" if overall_alive else "stale"
    health = "?"
    if probe:
        try:
            AgentGatewayClient(state.gateway_url, timeout=timeout).health()
            health = "ok"
            if overall_alive:
                status = "running"
        except AgentGatewayError:
            health = "down"
    return {
        "service": state.name,
        "default": state.name == current,
        "status": status,
        "health": health,
        "gateway": state.gateway_url,
        "router": state.router_url,
        "pairs": f"{workers_alive}/{state.num_pairs}",
        "gateway_pid": state.gateway_pid,
        "router_pid": state.router_pid,
        "agent_class": state.agent_class,
        "mode": state.mode,
        "created_at": state.created_at,
    }


def _handle(args: argparse.Namespace) -> int:
    services = load_all_agent_services()
    current = get_current_agent_service()
    rows = [
        _service_row(s, probe=not args.no_probe, timeout=args.timeout, current=current)
        for s in services
    ]

    if args.json:
        print(json.dumps(rows, indent=2, default=str))
        return 0

    if not rows:
        print("No agent services. Start one with `areal agent run --agent ...`.")
        return 0

    table = [
        [
            "*" + r["service"] if r["default"] else " " + r["service"],
            r["status"],
            r["health"],
            r["gateway"],
            f"pairs={r['pairs']}",
            r["agent_class"],
        ]
        for r in rows
    ]
    print_table(
        ["SERVICE", "STATUS", "HEALTH", "GATEWAY", "PAIRS", "AGENT"], table
    )
    return 0
