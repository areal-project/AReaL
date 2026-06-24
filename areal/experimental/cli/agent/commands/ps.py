# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import click

from areal.experimental.cli.agent.process import pid_alive
from areal.experimental.cli.agent.state import ServiceState, list_service_names


@click.command(name="ps", help="List locally known agent services.")
@click.option("--json", "as_json", is_flag=True)
@click.option("--all", "include_all", is_flag=True, help="Include stale services.")
def ps_cmd(as_json: bool, include_all: bool) -> None:
    raise SystemExit(do_ps(as_json=as_json, include_all=include_all) or 0)


def do_ps(*, as_json: bool, include_all: bool) -> int:
    rows = []
    for service in list_service_names():
        try:
            state = ServiceState.load(service)
        except Exception:
            if include_all:
                rows.append({"service": service, "status": "stale"})
            continue
        running = any(pid_alive(pid) for pid in state.all_pids())
        if running or include_all:
            rows.append(
                {
                    "service": service,
                    "status": "running" if running else "stale",
                    "gateway_url": state.gateway.url,
                    "agent": state.agent,
                }
            )

    if as_json:
        click.echo(json.dumps(rows, indent=2))
        return 0
    if not rows:
        click.echo("no agent services")
        return 0
    cols = ("SERVICE", "STATUS", "GATEWAY", "AGENT")
    table = [
        (
            row["service"],
            row["status"],
            row.get("gateway_url", ""),
            row.get("agent", ""),
        )
        for row in rows
    ]
    widths = [max(len(str(r[i])) for r in (cols, *table)) for i in range(4)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    click.echo(fmt.format(*cols))
    for row in table:
        click.echo(fmt.format(*row))
    return 0
