# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import click

from areal.experimental.cli.inference.common import (
    format_gpu_count,
    format_placement,
    format_ref,
    probe_http_health,
)
from areal.experimental.cli.inference.scheduler import TaskHandle
from areal.experimental.cli.inference.state import (
    RuntimeState,
    gateway_alive,
    load_runtime_state,
    resolve_service_name,
    service_state_path,
)


@click.command(
    name="status",
    help="Show inference service status — per-component drill-down for one service.",
)
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

    rows = _collect_rows(state)

    if as_json:
        click.echo(json.dumps(_json_snapshot(state, rows), indent=2))
        return 0

    _print_table(state, rows)
    return 0


def _collect_rows(state: RuntimeState) -> list[dict]:
    components: list[tuple[str, TaskHandle]] = [
        ("gateway", state.gateway_handle),
        ("router", state.router_handle),
    ]
    for name, entry in state.model_state.models.items():
        for i, replica in enumerate(entry.replicas):
            components.append((f"data_proxy[{name}/{i}]", replica.data_proxy))
            components.append((f"worker[{name}/{i}]", replica.worker))

    addrs = [handle.addr or "-" for _, handle in components]
    alive_flags = _probe_concurrently(addrs)
    return [
        _component_row(label, handle, state.backend, alive=alive)
        for (label, handle), alive in zip(components, alive_flags, strict=True)
    ]


def _probe_concurrently(addrs: list[str]) -> list[bool]:
    if not addrs:
        return []
    with ThreadPoolExecutor(max_workers=max(4, len(addrs))) as pool:
        return list(pool.map(probe_http_health, addrs))


def _component_row(
    label: str, handle: TaskHandle, backend: str, *, alive: bool
) -> dict:
    addr = handle.addr or "-"
    return {
        "component": label,
        "placement": format_placement(backend, handle) or "-",
        "gpus": format_gpu_count(handle),
        "addr": addr,
        "ref": format_ref(backend, handle),
        "alive": "yes" if alive else "no",
    }


def _print_table(state: RuntimeState, rows: list[dict]) -> None:
    gateway_addr = state.gateway_handle.addr or "-"
    click.echo(
        f"service: {state.service}   "
        f"backend: {state.backend}   "
        f"gateway: {gateway_addr}   "
        f"models: {len(state.models)}   "
        f"running: {'yes' if gateway_alive(state) else 'no'}"
    )
    click.echo()

    cols = ("COMPONENT", "PLACEMENT", "GPUS", "ADDR", "REF", "ALIVE")
    keys = ("component", "placement", "gpus", "addr", "ref", "alive")
    widths = [max(len(str(r[k])) for r in (dict(zip(keys, cols)), *rows)) for k in keys]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    click.echo(fmt.format(*cols))
    for row in rows:
        click.echo(fmt.format(*(str(row[k]) for k in keys)))


def _json_snapshot(state: RuntimeState, rows: list[dict]) -> dict:
    return {
        "service": state.service,
        "backend": state.backend,
        "running": gateway_alive(state),
        "gateway_handle": asdict(state.gateway_handle),
        "router_handle": asdict(state.router_handle),
        "started_at": state.started_at,
        "models": {name: asdict(entry) for name, entry in state.models.items()},
        "components": rows,
    }
