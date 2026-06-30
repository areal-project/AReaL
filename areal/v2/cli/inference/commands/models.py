# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import asdict

import click

from areal.v2.cli.inference.lifecycle import inf_lifecycle
from areal.v2.cli.utils import json_or_table


@click.command(name="models", help="List registered models.")
@click.option("--service", default=None, help="Target service instance.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON.")
def models_cmd(service: str | None, as_json: bool) -> None:
    raise SystemExit(do_models(as_json, service=service) or 0)


def do_models(as_json: bool, *, service: str | None = None) -> int:
    state = inf_lifecycle.running_state(service)
    if state is None:
        click.echo("[]" if as_json else "service not running")
        return 0

    payload = [{"name": name, **asdict(entry)} for name, entry in state.models.items()]
    json_or_table(payload, as_json=as_json, table_renderer=_print_models_table)
    return 0


def _print_models_table(rows: list[dict]) -> None:
    if not rows:
        click.echo("no models registered")
        return
    cols = ("NAME", "BACKEND", "WORKERS")
    table = [(row["name"], row["backend"], str(len(row["replicas"]))) for row in rows]
    widths = [max(len(r[i]) for r in (cols, *table)) for i in range(len(cols))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    click.echo(fmt.format(*cols))
    for row in table:
        click.echo(fmt.format(*row))
