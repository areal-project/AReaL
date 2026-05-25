# SPDX-License-Identifier: Apache-2.0

"""``areal agent status`` — agent-gateway / router / pair health.

Composed client-side from agent ``GET /health`` + local state + PID liveness
checks (design §11.4).
"""

from __future__ import annotations

import argparse
import json
import time

from areal.experimental.cli.agent_gateway_client import (
    AgentGatewayError,
    AgentGatewayUnreachable,
)
from areal.experimental.cli.agent_sessions import SessionRegistry
from areal.experimental.cli.agent_state import liveness_summary
from areal.experimental.cli.commands.agent._common import (
    add_targeting_flags,
    print_table,
    resolve_agent_target,
)


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "status",
        help="Show gateway / router / pair / session health.",
        description=(
            "Combine the agent gateway's /health with local PID liveness "
            "and the session registry to produce a status table."
        ),
        aliases=["health"],
    )
    add_targeting_flags(p)
    p.add_argument(
        "--watch",
        action="store_true",
        help="Refresh continuously until interrupted.",
    )
    p.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Refresh interval (s) for --watch.",
    )
    p.add_argument("--timeout", type=float, default=3.0, help="Per-request timeout (s).")
    p.set_defaults(func=_handle)


def _collect(args: argparse.Namespace) -> dict:
    target = resolve_agent_target(args)
    client = target.client(timeout=args.timeout)

    out: dict = {
        "service": target.service,
        "gateway_url": target.gateway_url,
        "gateway": {"status": "unknown"},
        "sessions": {},
    }

    if target.state is not None:
        out["liveness"] = liveness_summary(target.state)
        out["router_url"] = target.state.router_url
        out["agent_class"] = target.state.agent_class
        out["num_pairs"] = target.state.num_pairs
        out["pairs"] = [
            {
                "index": p.index,
                "worker_addr": f"http://{p.worker_host}:{p.worker_port}",
                "proxy_addr": f"http://{p.proxy_host}:{p.proxy_port}",
            }
            for p in target.state.pairs
        ]

    try:
        h = client.health()
        out["gateway"] = {"status": "ok", "raw": h}
    except AgentGatewayUnreachable as e:
        out["gateway"] = {"status": "unreachable", "error": str(e)}
    except AgentGatewayError as e:
        out["gateway"] = {"status": "error", "error": str(e)}

    if target.service:
        try:
            reg = SessionRegistry.load(target.service)
            out["sessions"] = {
                "default": reg.default_session,
                "total": len(reg.sessions),
                "active": len(reg.active_sessions()),
            }
        except Exception:
            pass

    return out


def _render(data: dict) -> None:
    service = data["service"] or "-"
    rows = [
        [
            service,
            "gateway",
            data["gateway"]["status"],
            data["gateway_url"],
            f"sessions={data.get('sessions', {}).get('active', 0)}",
        ],
        [
            service,
            "router",
            "ok" if data.get("liveness", {}).get("router_pid_alive") else "?",
            data.get("router_url", "-"),
            f"pairs={data.get('num_pairs', 0)}",
        ],
    ]
    for i, pair in enumerate(data.get("pairs", [])):
        worker_alive = data.get("liveness", {}).get("worker_pids_alive", [])
        proxy_alive = data.get("liveness", {}).get("proxy_pids_alive", [])
        wstat = "ok" if i < len(worker_alive) and worker_alive[i] else "down"
        pstat = "ok" if i < len(proxy_alive) and proxy_alive[i] else "down"
        rows.append([service, f"worker-{pair['index']}", wstat, pair["worker_addr"], ""])
        rows.append([service, f"proxy-{pair['index']}", pstat, pair["proxy_addr"], ""])
    print_table(["SERVICE", "COMPONENT", "STATUS", "ADDR", "DETAILS"], rows)


def _handle(args: argparse.Namespace) -> int:
    if not args.watch:
        data = _collect(args)
        if args.json:
            print(json.dumps(data, indent=2, default=str))
        else:
            _render(data)
        return 0 if data["gateway"]["status"] == "ok" else 1

    while True:
        data = _collect(args)
        if args.json:
            print(json.dumps(data, default=str))
        else:
            print("\033[2J\033[H", end="")
            _render(data)
            print(f"\n(refresh every {args.interval}s; Ctrl-C to stop)")
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            return 0
