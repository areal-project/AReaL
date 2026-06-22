# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import asdict

import click

from areal.experimental.cli.inference.client import (
    GatewayClient,
    GatewayHTTPError,
    GatewayUnreachable,
)
from areal.experimental.cli.inference.state import (
    gateway_alive,
    load_runtime_state,
    resolve_service_name,
    service_state_path,
)
from areal.experimental.cli.process import pid_alive


@click.command(name="status", help="Show inference service status.")
@click.option("--service", default=None, help="Target service instance.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON.")
def status_cmd(service: str | None, as_json: bool) -> None:
    raise SystemExit(do_status(as_json, service=service) or 0)


def do_status(as_json: bool, *, service: str | None = None) -> int:
    service_name = resolve_service_name(service)
    if not service_state_path(service_name).exists():
        if as_json:
            click.echo(
                json.dumps({"service": service_name, "running": False}, indent=2)
            )
        else:
            click.echo(f"service {service_name!r} not running")
        return 0

    try:
        state = load_runtime_state(service_name)
    except Exception as exc:
        raise click.ClickException(f"failed to load state: {exc}") from exc

    pid_ok = gateway_alive(state)
    gateway_http = "down"
    if pid_ok:
        try:
            GatewayClient(state.gateway_url, state.admin_api_key).health(timeout=2.0)
            gateway_http = "ok"
        except GatewayUnreachable:
            gateway_http = "unreachable"
        except GatewayHTTPError:
            gateway_http = "error"

    if as_json:
        snap = {
            "service": state.service,
            "running": pid_ok,
            "gateway_url": state.gateway_url,
            "gateway_pid": state.gateway_pid,
            "gateway_http": gateway_http,
            "router_url": state.router_url,
            "started_at": state.started_at,
            "models": {name: asdict(entry) for name, entry in state.models.items()},
        }
        click.echo(json.dumps(snap, indent=2))
        return 0

    rows: list[tuple[str, str, str, str]] = [
        (
            f"{state.service}/gateway",
            gateway_http,
            state.gateway_url,
            f"models={len(state.models)}",
        ),
        (
            f"{state.service}/router",
            "ok" if pid_alive(state.router_pid) else "down",
            state.router_url,
            "",
        ),
    ]
    for name, entry in state.models.items():
        if entry.kind == "internal":
            detail = f"backend={entry.backend} workers={len(entry.pids) // 2}"
            addr = "internal"
        else:
            detail = f"api_url={entry.api_url}"
            addr = "external"
        rows.append((f"{state.service}/{name}", "registered", addr, detail))

    cols = ("COMPONENT", "STATUS", "ADDR", "DETAILS")
    widths = [max(len(row[i]) for row in (cols, *rows)) for i in range(len(cols))]
    fmt = "  ".join(f"{{:<{width}}}" for width in widths)
    click.echo(fmt.format(*cols))
    for row in rows:
        click.echo(fmt.format(*row))
    return 0
