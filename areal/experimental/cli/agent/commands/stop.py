# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import click

from areal.experimental.cli.agent.process import kill_pids
from areal.experimental.cli.agent.state import (
    ServiceState,
    resolve_service_name,
    service_state_path,
)


@click.command(name="stop", help="Stop an agent service.")
@click.option("--service", default=None)
@click.option("--grace-period", type=float, default=10.0, show_default=True)
@click.option("--keep-state", is_flag=True)
@click.option("--force", is_flag=True, help="SIGKILL immediately.")
def stop_cmd(
    service: str | None, grace_period: float, keep_state: bool, force: bool
) -> None:
    raise SystemExit(
        do_stop(
            service=service,
            grace_period=grace_period,
            keep_state=keep_state,
            force=force,
        )
        or 0
    )


def do_stop(
    *,
    service: str | None,
    grace_period: float,
    keep_state: bool,
    force: bool,
) -> int:
    name = resolve_service_name(service)
    if not service_state_path(name).exists():
        click.echo(f"service {name!r} is not running")
        return 0
    try:
        state = ServiceState.load(name)
    except Exception:
        if not keep_state:
            ServiceState.remove(name)
        click.echo(f"removed stale state for {name!r}")
        return 0

    kill_pids(state.all_pids(), grace_s=0.0 if force else grace_period)
    if not keep_state:
        ServiceState.remove(name)
    click.echo(f"service {name!r} stopped")
    return 0
