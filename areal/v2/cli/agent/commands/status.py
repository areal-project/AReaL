# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time

import click

from areal.v2.cli.agent.lifecycle import agent_lifecycle
from areal.v2.cli.process import pid_alive
from areal.v2.cli.state import SupportsComponentProbe
from areal.v2.cli.status import ColumnSpec, StatusReporter
from areal.v2.cli.utils import json_or_table


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
    name = agent_lifecycle.resolve_service_name(service)
    while True:
        _emit_once(name, as_json=as_json)
        if not watch:
            return 0
        time.sleep(interval)


def _emit_once(service: str, *, as_json: bool) -> None:
    path = agent_lifecycle.state_path(service)
    if not path.exists():
        payload = {"service": service, "running": False, "components": []}
        json_or_table(
            payload,
            as_json=as_json,
            table_renderer=lambda p: click.echo(
                f"service {p['service']!r} is not running"
            ),
        )
        return
    try:
        state = agent_lifecycle.load_state(service)
    except Exception as exc:
        payload = {
            "service": service,
            "running": False,
            "error": f"failed to read state: {exc}",
            "components": [],
        }
        json_or_table(
            payload,
            as_json=as_json,
            table_renderer=lambda p: click.echo(
                f"service {p['service']!r}: {p['error']}"
            ),
        )
        return

    components = list(state.components())
    reporter = StatusReporter(components, _columns(service))
    alive = reporter.probe_all()

    if as_json:
        payload = {
            "service": service,
            "running": pid_alive(state.gateway.pid),
            "gateway_url": state.gateway.url,
            "router_url": state.router.url,
            "components": reporter.json_snapshot(alive),
        }
        click.echo(_json_indent(payload))
        return
    reporter.print_table(
        reporter.render_rows(alive),
        header_line=f"service: {service}  gateway: {state.gateway.url}",
    )


def _columns(service: str) -> list[ColumnSpec]:
    def _service(label: str, _: SupportsComponentProbe, __: bool) -> str:
        del label
        return service

    def _component(label: str, _: SupportsComponentProbe, __: bool) -> str:
        return label

    def _status(_: str, __: SupportsComponentProbe, alive: bool) -> str:
        return "ok" if alive else "down"

    def _addr(_: str, handle: SupportsComponentProbe, __: bool) -> str:
        return handle.addr or "-"

    def _pid(_: str, handle: SupportsComponentProbe, __: bool) -> str:
        return str(handle.pid) if handle.pid > 0 else "-"

    return [
        ColumnSpec("SERVICE", _service),
        ColumnSpec("COMPONENT", _component),
        ColumnSpec("STATUS", _status),
        ColumnSpec("ADDR", _addr),
        ColumnSpec("PID", _pid),
    ]


def _json_indent(payload: dict) -> str:
    import json

    return json.dumps(payload, indent=2, default=str)
