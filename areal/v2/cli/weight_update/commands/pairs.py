# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import click

from areal.v2.cli.utils import json_or_table
from areal.v2.cli.weight_update.common import resolve_target


@click.command(
    name="pairs", help="List (train, inference) pairs connected to a gateway."
)
@click.option("--service", default=None, help="Target service instance.")
@click.option(
    "--gateway", default=None, help="Gateway base URL, e.g. http://host:7080."
)
@click.option("--admin-api-key", "admin_api_key", default=None, help="Admin API key.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON.")
def pairs_cmd(
    service: str | None, gateway: str | None, admin_api_key: str | None, as_json: bool
) -> None:
    raise SystemExit(
        do_pairs(as_json, service=service, gateway=gateway, admin_api_key=admin_api_key)
    )


def do_pairs(
    as_json: bool,
    *,
    service: str | None = None,
    gateway: str | None = None,
    admin_api_key: str | None = None,
) -> int:
    client, _ = resolve_target(service, gateway, admin_api_key)
    payload = client.pairs()
    json_or_table(payload, as_json=as_json, table_renderer=_print_pairs)
    return 0


def _print_pairs(rows: list[dict]) -> None:
    if not rows:
        click.echo("no pairs connected")
        return
    cols = ("NAME", "MODE", "TRAIN", "INFER", "VERSION", "COLOCATE")
    table = [
        (
            row.get("pair_name", ""),
            row.get("mode", ""),
            str(row.get("train_world_size", 0)),
            str(row.get("inference_world_size", 0)),
            str(row.get("last_version", 0)),
            "yes" if row.get("colocate") else "no",
        )
        for row in rows
    ]
    widths = [max(len(r[i]) for r in (cols, *table)) for i in range(len(cols))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    click.echo(fmt.format(*cols))
    for row in table:
        click.echo(fmt.format(*row))
