# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import click

from areal.experimental.cli.client import ServiceHTTPError, ServiceUnreachable
from areal.experimental.cli.inference.client import RouterClient
from areal.experimental.cli.inference.common import logger
from areal.experimental.cli.inference.lifecycle import inf_lifecycle
from areal.experimental.cli.process import kill_pids


@click.command(name="deregister", help="Deregister a model and tear down its workers.")
@click.option("--model-name", required=True, help="Model name to deregister.")
@click.option("--service", default=None, help="Target service instance.")
@click.option("--grace", type=float, default=10.0, show_default=True)
@click.option("--force", is_flag=True, help="SIGKILL workers immediately.")
def deregister_cmd(
    model_name: str, service: str | None, grace: float, force: bool
) -> None:
    raise SystemExit(do_deregister(model_name, grace, force, service=service) or 0)


def do_deregister(
    model_name: str, grace: float, force: bool, *, service: str | None = None
) -> int:
    state = inf_lifecycle.load_running_state(service)
    if model_name not in state.models:
        raise click.ClickException(
            f"model {model_name!r} is not registered in service {state.service!r}"
        )
    entry = state.models[model_name]
    router = RouterClient(state.router_url, state.admin_api_key)

    try:
        router.remove_model(model_name)
    except ServiceHTTPError as exc:
        if exc.status != 404:
            logger.warning("router remove_model %s returned %d", model_name, exc.status)
    except ServiceUnreachable as exc:
        logger.warning("router unreachable while removing %s: %s", model_name, exc)

    # Router unregister → kill data-proxies → kill workers (same data-flow
    # order as terminate_runtime_state).
    for r in entry.replicas:
        try:
            router.unregister_worker(r.data_proxy.addr)
        except (ServiceHTTPError, ServiceUnreachable) as exc:
            logger.warning("router unregister %s failed: %s", r.data_proxy.addr, exc)

    proxy_pids = [r.data_proxy.pid for r in entry.replicas if r.data_proxy.pid > 0]
    worker_pids = [r.worker.pid for r in entry.replicas if r.worker.pid > 0]
    effective_grace = 0.0 if force else grace
    for pids in (proxy_pids, worker_pids):
        if pids:
            kill_pids(pids, grace_s=effective_grace)

    del state.model_state.models[model_name]
    state.model_state.save()
    logger.info("deregistered model %r from service %r", model_name, state.service)
    return 0
