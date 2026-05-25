# SPDX-License-Identifier: Apache-2.0

"""``areal inf status`` — gateway/router/model health for one service.

Composed client-side from ``GET /health`` + ``GET /models`` + local state +
PID liveness checks (design §10.3).
"""

from __future__ import annotations

import argparse
import json
import time

from areal.experimental.cli.commands.inf._common import (
    add_targeting_flags,
    print_table,
    resolve_target,
)
from areal.experimental.cli.gateway_client import (
    GatewayAuthError,
    GatewayError,
    GatewayUnreachable,
)
from areal.experimental.cli.inf_state import liveness_summary


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "status",
        help="Show gateway / router / model component health.",
        description=(
            "Combine the gateway's /health and /models responses with local "
            "PID liveness to produce a compact status table for one service."
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
    target = resolve_target(args)
    client = target.client(timeout=args.timeout)

    out: dict = {
        "service": target.service,
        "gateway_url": target.gateway_url,
        "gateway": {"status": "unknown"},
        "router": {"status": "unknown"},
        "models": [],
    }

    if target.state is not None:
        out["pids"] = {
            "gateway": target.state.gateway_pid,
            "router": target.state.router_pid,
        }
        out["liveness"] = liveness_summary(target.state)
        out["router_url"] = target.state.router_url

    try:
        h = client.health()
        out["gateway"] = {"status": "ok", "raw": h}
        if isinstance(h, dict) and h.get("router_addr"):
            out["router_url"] = h["router_addr"]
            out["router"] = {"status": "ok"}
    except GatewayUnreachable as e:
        out["gateway"] = {"status": "unreachable", "error": str(e)}
    except GatewayError as e:
        out["gateway"] = {"status": "error", "error": str(e)}

    if out["gateway"]["status"] == "ok":
        try:
            out["models"] = client.models()
        except GatewayAuthError as e:
            out["models_error"] = str(e)
        except GatewayError as e:
            out["models_error"] = str(e)

    return out


def _render(data: dict) -> None:
    rows = []
    service = data["service"] or "-"
    rows.append([
        service,
        "gateway",
        data["gateway"]["status"],
        data["gateway_url"],
        f"models={len(data.get('models', []))}",
    ])
    router_status = data["router"]["status"]
    rows.append([
        service,
        "router",
        router_status,
        data.get("router_url", "-"),
        "",
    ])
    for m in data.get("models", []):
        rows.append([service, m, "registered", "-", ""])
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
            print("\033[2J\033[H", end="")  # clear screen
            _render(data)
            print(f"\n(refresh every {args.interval}s; Ctrl-C to stop)")
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            return 0
