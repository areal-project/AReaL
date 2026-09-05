# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import click

from areal.v2.cli.utils import json_or_table
from areal.v2.cli.weight_update.lifecycle import wu_lifecycle


@click.command(name="ps", help="List weight-update services known to this machine.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON.")
@click.option("--all", "include_all", is_flag=True, help="Include stale services.")
def ps_cmd(as_json: bool, include_all: bool) -> None:
    raise SystemExit(do_ps(as_json, include_all) or 0)


def do_ps(as_json: bool, include_all: bool = False) -> int:
    rows = []
    for name in wu_lifecycle.list_services():
        try:
            state = wu_lifecycle.load_state(name)
        except Exception:
            # A half-written file should not hide the services that do parse;
            # report it as a row instead of aborting the listing.
            if include_all:
                rows.append({"service": name, "gateway": "", "status": "unreadable"})
            continue
        alive = state.gateway_alive()
        if not (alive or include_all):
            continue
        rows.append(
            {
                "service": state.service,
                "gateway": state.gateway.url,
                "launch_mode": state.launch_mode,
                "started_at": state.started_at,
                "status": "up" if alive else "unreachable",
            }
        )
    json_or_table(rows, as_json=as_json, table_renderer=_print_ps)
    return 0


def _print_ps(rows: list[dict]) -> None:
    if not rows:
        click.echo("no weight-update services on this machine")
        return
    cols = ("SERVICE", "GATEWAY", "LAUNCH", "STATUS")
    table = [
        (r["service"], r.get("gateway", ""), r.get("launch_mode", ""), r["status"])
        for r in rows
    ]
    widths = [max(len(r[i]) for r in (cols, *table)) for i in range(len(cols))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    click.echo(fmt.format(*cols))
    for row in table:
        click.echo(fmt.format(*row))
