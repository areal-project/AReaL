# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import click

from areal.v2.cli.agent.lifecycle import agent_lifecycle
from areal.v2.cli.agent.state import ServiceState
from areal.v2.cli.process import kill_pids


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
    name = agent_lifecycle.resolve_service_name(service)
    path = agent_lifecycle.state_path(name)
    if not path.exists():
        click.echo(f"service {name!r} is not running")
        return 0

    try:
        state = agent_lifecycle.load_state(name)
    except Exception:
        if not keep_state:
            ServiceState.remove(name)
        click.echo(f"removed stale state for {name!r}")
        return 0

    pids = [pid for _, h in state.components() if (pid := h.pid) > 0]
    kill_pids(pids, grace_s=0.0 if force else grace_period)
    if not keep_state:
        ServiceState.remove(name)
    click.echo(f"service {name!r} stopped")
    return 0
