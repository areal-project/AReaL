# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import click

from areal.experimental.cli.inference.common import logger, terminate_runtime_state
from areal.experimental.cli.inference.state import (
    ModelState,
    ServiceState,
    load_runtime_state,
    resolve_service_name,
    service_state_path,
)


@click.command(name="stop", help="Stop an inference service.")
@click.option("--service", default=None, help="Target service instance.")
@click.option("--grace", type=float, default=10.0, show_default=True)
@click.option("--force", is_flag=True, help="SIGKILL immediately.")
@click.option("--keep-state", is_flag=True, help="Keep service state after stopping.")
def stop_cmd(service: str | None, grace: float, force: bool, keep_state: bool) -> None:
    raise SystemExit(do_stop(grace, force, service=service, keep_state=keep_state) or 0)


def do_stop(
    grace: float,
    force: bool,
    *,
    service: str | None = None,
    keep_state: bool = False,
) -> int:
    service_name = resolve_service_name(service)
    if not service_state_path(service_name).exists():
        click.echo(f"service {service_name!r} not running")
        return 0

    try:
        state = load_runtime_state(service_name)
    except Exception:
        logger.warning("stale state; removing")
        if not keep_state:
            ServiceState.remove(service_name)
            ModelState.remove(service_name)
        return 0

    terminate_runtime_state(state, grace_s=grace, force=force)

    if not keep_state:
        ServiceState.remove(service_name)
        ModelState.remove(service_name)
    click.echo(f"service {service_name!r} stopped")
    return 0
