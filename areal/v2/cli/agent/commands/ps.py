# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import click

from areal.v2.cli.agent.lifecycle import agent_lifecycle
from areal.v2.cli.process import pid_alive
from areal.v2.cli.utils import json_or_table


@click.command(name="ps", help="List locally known agent services.")
@click.option("--json", "as_json", is_flag=True)
@click.option("--all", "include_all", is_flag=True, help="Include stale services.")
def ps_cmd(as_json: bool, include_all: bool) -> None:
    raise SystemExit(do_ps(as_json=as_json, include_all=include_all) or 0)


def do_ps(*, as_json: bool, include_all: bool) -> int:
    rows: list[dict] = []
    for service in agent_lifecycle.list_services():
        try:
            state = agent_lifecycle.load_state(service)
        except Exception:
            if include_all:
                rows.append({"service": service, "status": "stale"})
            continue
        running = pid_alive(state.gateway.pid)
        if running or include_all:
            rows.append(
                {
                    "service": service,
                    "status": "running" if running else "stale",
                    "gateway_url": state.gateway.url,
                    "agent": state.agent,
                }
            )

    json_or_table(rows, as_json=as_json, table_renderer=_print_table)
    return 0


def _print_table(rows: list[dict]) -> None:
    if not rows:
        click.echo("no agent services")
        return
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
