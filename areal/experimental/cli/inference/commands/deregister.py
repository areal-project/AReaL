# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import signal

import click

from areal.experimental.cli.inference.client import (
    GatewayHTTPError,
    GatewayUnreachable,
    RouterClient,
)
from areal.experimental.cli.inference.common import load_running_state, logger
from areal.experimental.cli.process import kill_pids, signal_pid


@click.command(name="deregister", help="Deregister a model and tear down its workers.")
@click.argument("name")
@click.option("--grace", type=float, default=10.0, show_default=True)
@click.option("--force", is_flag=True, help="SIGKILL workers immediately.")
def deregister_cmd(name: str, grace: float, force: bool) -> None:
    raise SystemExit(do_deregister(name, grace, force) or 0)


def do_deregister(name: str, grace: float, force: bool) -> int:
    state = load_running_state()
    if name not in state.models:
        raise click.ClickException(f"model {name!r} is not registered")
    entry = state.models[name]
    router = RouterClient(state.router_url, state.admin_api_key)

    try:
        router.remove_model(name)
    except GatewayHTTPError as exc:
        if exc.status != 404:
            logger.warning("router remove_model %s returned %d", name, exc.status)
    except GatewayUnreachable as exc:
        logger.warning("router unreachable while removing %s: %s", name, exc)

    for addr in entry.proxy_addrs:
        try:
            router.unregister_worker(addr)
        except (GatewayHTTPError, GatewayUnreachable) as exc:
            logger.warning("router unregister %s failed: %s", addr, exc)

    if entry.pids:
        if force:
            for pid in entry.pids:
                signal_pid(pid, signal.SIGKILL)
        else:
            kill_pids(entry.pids, grace_s=grace)

    if entry.gpu_count > 0 and entry.base_gpu_id + entry.gpu_count == state.next_gpu_id:
        state.next_gpu_id = entry.base_gpu_id
    del state.models[name]
    state.save()
    logger.info("deregistered model %r", name)
    return 0
