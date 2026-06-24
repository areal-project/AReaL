# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import time

import click

from areal.experimental.cli.agent.http import (
    AgentHTTPError,
    AgentUnreachable,
    DataProxyClient,
    GatewayClient,
    RouterClient,
)
from areal.experimental.cli.agent.process import pid_alive
from areal.experimental.cli.agent.state import (
    ServiceState,
    resolve_service_name,
    service_state_path,
)


@click.command(name="status", help="Show agent service health.")
@click.option("--service", default=None)
@click.option("--watch", is_flag=True)
@click.option("--interval", type=float, default=2.0, show_default=True)
@click.option("--json", "as_json", is_flag=True)
def status_cmd(
    service: str | None, watch: bool, interval: float, as_json: bool
) -> None:
    raise SystemExit(
        do_status(service=service, watch=watch, interval=interval, as_json=as_json) or 0
    )


def do_status(
    *, service: str | None, watch: bool, interval: float, as_json: bool
) -> int:
    name = resolve_service_name(service)
    while True:
        snapshot = _snapshot(name)
        if as_json:
            click.echo(json.dumps(snapshot, indent=2))
        else:
            _print_table(snapshot)
        if not watch:
            return 0
        time.sleep(interval)


def _snapshot(service: str) -> dict:
    if not service_state_path(service).exists():
        return {"service": service, "running": False, "components": []}
    try:
        state = ServiceState.load(service)
    except Exception as exc:
        return {
            "service": service,
            "running": False,
            "error": f"failed to read state: {exc}",
            "components": [],
        }

    components = [
        _component_health(
            service,
            "gateway",
            state.gateway.url,
            state.gateway.pid,
            lambda: GatewayClient(state.gateway.url, state.admin_api_key).health(),
        ),
        _component_health(
            service,
            "router",
            state.router.url,
            state.router.pid,
            lambda: RouterClient(state.router.url, state.admin_api_key).health(),
        ),
    ]
    for pair in state.pairs:
        components.append(
            _component_health(
                service,
                pair.worker.component,
                pair.worker.url,
                pair.worker.pid,
                lambda url=pair.worker.url: DataProxyClient(url).health(),
            )
        )
        components.append(
            _component_health(
                service,
                pair.data_proxy.component,
                pair.data_proxy.url,
                pair.data_proxy.pid,
                lambda url=pair.data_proxy.url: DataProxyClient(url).health(),
            )
        )

    return {
        "service": service,
        "running": any(c["pid_alive"] for c in components),
        "gateway_url": state.gateway.url,
        "router_url": state.router.url,
        "components": components,
    }


def _component_health(service: str, component: str, url: str, pid: int, fn) -> dict:
    http_status = "down"
    detail = ""
    try:
        data = fn()
        http_status = "ok"
        detail = json.dumps(data, sort_keys=True)
    except AgentHTTPError as exc:
        http_status = f"http-{exc.status}"
        detail = exc.body
    except AgentUnreachable as exc:
        detail = str(exc)
    return {
        "service": service,
        "component": component,
        "status": http_status,
        "addr": url,
        "pid": pid,
        "pid_alive": pid_alive(pid),
        "details": detail,
    }


def _print_table(snapshot: dict) -> None:
    components = snapshot.get("components") or []
    if not components:
        click.echo(f"service {snapshot['service']!r} is not running")
        return
    rows = [
        (row["service"], row["component"], row["status"], row["addr"], row["details"])
        for row in components
    ]
    cols = ("SERVICE", "COMPONENT", "STATUS", "ADDR", "DETAILS")
    widths = [max(len(str(r[i])) for r in (cols, *rows)) for i in range(len(cols))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    click.echo(fmt.format(*cols))
    for row in rows:
        click.echo(fmt.format(*row))
