# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import click

from areal.v2.cli.inference.lifecycle import inf_lifecycle
from areal.v2.cli.utils import json_or_table


@click.command(name="ps", help="List locally known inference services.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON.")
@click.option("--all", "include_all", is_flag=True, help="Include stale services.")
def ps_cmd(as_json: bool, include_all: bool) -> None:
    raise SystemExit(do_ps(as_json, include_all) or 0)


def do_ps(as_json: bool, include_all: bool) -> int:
    rows: list[dict] = []
    for service in inf_lifecycle.list_services():
        try:
            state = inf_lifecycle.load_state(service)
        except Exception:
            if include_all:
                rows.append({"service": service, "status": "stale"})
            continue
        running = state.gateway_alive()
        if running or include_all:
            rows.append(
                {
                    "service": service,
                    "status": "running" if running else "stale",
                    "backend": state.backend,
                    "gateway_url": state.gateway_url,
                    "gateway_pid": state.gateway_pid,
                    "router_url": state.router_url,
                    "models": len(state.models),
                }
            )

    json_or_table(rows, as_json=as_json, table_renderer=_print_table)
    return 0


def _print_table(rows: list[dict]) -> None:
    if not rows:
        click.echo("no inference services")
        return

    def _pid_cell(row: dict) -> str:
        pid = row.get("gateway_pid", 0)
        return str(pid) if pid else "-"

    table = [
        (
            row["service"],
            row["status"],
            row.get("backend", "-"),
            str(row.get("models", "")),
            row.get("gateway_url", ""),
            _pid_cell(row),
        )
        for row in rows
    ]
    cols = ("SERVICE", "STATUS", "BACKEND", "MODELS", "GATEWAY", "PID")
    widths = [max(len(str(r[i])) for r in (cols, *table)) for i in range(len(cols))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    click.echo(fmt.format(*cols))
    for row in table:
        click.echo(fmt.format(*row))
