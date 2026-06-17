# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import signal

import click

from areal.experimental.cli.inference.common import logger
from areal.experimental.cli.inference.state import DaemonState, state_path
from areal.experimental.cli.process import kill_pids, signal_pid


@click.command(name="stop", help="Stop the inference daemon.")
@click.option("--grace", type=float, default=10.0, show_default=True)
@click.option("--force", is_flag=True, help="SIGKILL immediately.")
def stop_cmd(grace: float, force: bool) -> None:
    raise SystemExit(do_stop(grace, force) or 0)


def do_stop(grace: float, force: bool) -> int:
    if not state_path().exists():
        click.echo("daemon not running")
        return 0

    try:
        state = DaemonState.load()
    except Exception:
        logger.warning("stale state; removing")
        DaemonState.remove()
        return 0

    pids = [
        pid
        for pid in (state.gateway_pid, state.router_pid, *state.all_worker_pids())
        if pid > 0
    ]
    if force:
        for pid in pids:
            signal_pid(pid, signal.SIGKILL)
    else:
        kill_pids(pids, grace_s=grace)

    DaemonState.remove()
    click.echo("daemon stopped")
    return 0
