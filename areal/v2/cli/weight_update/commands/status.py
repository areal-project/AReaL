# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import click

from areal.v2.cli.client import ServiceHTTPError, ServiceUnreachable
from areal.v2.cli.utils import json_or_table
from areal.v2.cli.weight_update.common import resolve_target


@click.command(name="status", help="Show whether a weight-update gateway is reachable.")
@click.option("--service", default=None, help="Target service instance.")
@click.option(
    "--gateway", default=None, help="Gateway base URL, e.g. http://host:7080."
)
@click.option("--admin-api-key", "admin_api_key", default=None, help="Admin API key.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON.")
def status_cmd(
    service: str | None, gateway: str | None, admin_api_key: str | None, as_json: bool
) -> None:
    raise SystemExit(
        do_status(
            as_json, service=service, gateway=gateway, admin_api_key=admin_api_key
        )
    )


def do_status(
    as_json: bool,
    *,
    service: str | None = None,
    gateway: str | None = None,
    admin_api_key: str | None = None,
) -> int:
    client, state = resolve_target(service, gateway, admin_api_key)

    payload: dict = {
        "service": state.service if state else None,
        "gateway": client.base,
        "launch_mode": state.launch_mode if state else "external",
        "started_at": state.started_at if state else None,
    }
    try:
        client.health()
        payload["reachable"] = True
    except (ServiceUnreachable, ServiceHTTPError) as exc:
        payload["reachable"] = False
        payload["error"] = str(exc)

    try:
        payload["pairs"] = len(client.pairs())
    except (ServiceUnreachable, ServiceHTTPError):
        payload["pairs"] = None

    json_or_table(payload, as_json=as_json, table_renderer=_print_status)
    # Exit non-zero when unreachable so `areal weight-update status` is usable
    # as a health gate in a script.
    return 0 if payload["reachable"] else 1


def _print_status(row: dict) -> None:
    click.echo(f"gateway      {row['gateway']}")
    click.echo(f"service      {row['service'] or '(external)'}")
    click.echo(f"launch mode  {row['launch_mode']}")
    click.echo(f"reachable    {'yes' if row['reachable'] else 'no'}")
    if not row["reachable"]:
        click.echo(f"error        {row.get('error', '')}")
    elif row["pairs"] is not None:
        click.echo(f"pairs        {row['pairs']}")
