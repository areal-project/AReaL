# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time

import click

from areal.experimental.cli.inference.client import GatewayClient, RouterClient
from areal.experimental.cli.inference.common import (
    ENGINE_ARGS_HELP,
    PROXY_ARGS_HELP,
    logger,
    register_model,
    terminate_runtime_state,
    validate_register_opts,
)
from areal.experimental.cli.inference.launcher import spawn_gateway, spawn_router
from areal.experimental.cli.inference.lifecycle import inf_lifecycle
from areal.experimental.cli.inference.scheduler import TaskHandle
from areal.experimental.cli.inference.state import (
    DEFAULT_SERVICE,
    INF_NAMESPACE,
    ModelState,
    RuntimeState,
    ServiceState,
    locked_model_state,
)
from areal.experimental.cli.process import kill_pids
from areal.experimental.cli.state import logs_dir
from areal.experimental.cli.utils import wait_client_health
from areal.experimental.cli.watcher import ForegroundWatcher


@click.command(name="run", help="Start an inference service (gateway + router).")
@click.option("--service", default=DEFAULT_SERVICE, show_default=True)
@click.option("--port", type=int, default=8080, show_default=True, help="Gateway port.")
@click.option(
    "--host", default="127.0.0.1", show_default=True, help="Gateway bind host."
)
@click.option("--admin-api-key", default="areal-admin-key", show_default=True)
@click.option(
    "--routing-strategy",
    type=click.Choice(["round_robin", "least_busy"]),
    default="round_robin",
    show_default=True,
)
@click.option(
    "--log-level",
    type=click.Choice(["debug", "info", "warning", "error"]),
    default="info",
    show_default=True,
)
@click.option(
    "--launch-timeout",
    type=float,
    default=30.0,
    show_default=True,
    help="Seconds to wait for gateway /health.",
)
@click.option("-d", "--detach", is_flag=True, help="Fork the daemon and exit.")
@click.option("--model", default=None, help="Register this model at startup.")
@click.option("--backend", default=None, help="Backend spec, e.g. 'sglang:tp=2,dp=2'.")
@click.option("--model-path", default=None)
@click.option("--tokenizer-path", default=None)
@click.option("--engine-args", default="", show_default=False, help=ENGINE_ARGS_HELP)
@click.option("--proxy-args", default="", show_default=False, help=PROXY_ARGS_HELP)
@click.option(
    "--model-health-timeout",
    type=float,
    default=600.0,
    show_default=True,
    help="Seconds to wait for the model server to come up.",
)
@click.option(
    "--scheduler",
    type=click.Choice(["local"]),
    default=None,
    help=(
        "Scheduler backend used to place workers and data-proxies. Pinned "
        "for the service's lifetime — register / stop / status will read it "
        "from saved state. Defaults to [scheduler].type in config.toml, "
        "falling back to 'local'."
    ),
)
@click.option("--force", is_flag=True, help="Replace stale or running service state.")
def run_cmd(**opts) -> None:
    raise SystemExit(do_run(opts) or 0)


def do_run(opts: dict) -> int:
    service = opts["service"]
    if opts["force"]:
        inf_lifecycle.force_replace_slot(service, grace_s=5.0)
    else:
        inf_lifecycle.refuse_if_running(service)

    if opts["model"]:
        validate_register_opts(opts)
    elif opts["backend"]:
        raise click.UsageError("model registration flags require --model.")

    log_dir = logs_dir(INF_NAMESPACE, service)
    logger.info("starting inference service %r (logs: %s)", service, log_dir)

    router_pid, router_port = spawn_router(
        host="127.0.0.1",
        admin_api_key=opts["admin_api_key"],
        routing_strategy=opts["routing_strategy"],
        log_level=opts["log_level"],
        log_file=log_dir / "router.log",
    )
    router_url = f"http://127.0.0.1:{router_port}"
    logger.info("router pid=%d %s", router_pid, router_url)

    time.sleep(0.3)
    gateway_pid = spawn_gateway(
        host=opts["host"],
        port=opts["port"],
        admin_api_key=opts["admin_api_key"],
        router_url=router_url,
        log_level=opts["log_level"],
        log_file=log_dir / "gateway.log",
    )
    host_for_url = "127.0.0.1" if opts["host"] in ("0.0.0.0", "::") else opts["host"]
    gateway_url = f"http://{host_for_url}:{opts['port']}"
    logger.info("gateway pid=%d %s", gateway_pid, gateway_url)

    gateway_client = GatewayClient(gateway_url, opts["admin_api_key"])
    router_client = RouterClient(router_url, opts["admin_api_key"])
    try:
        wait_client_health(
            gateway_client, timeout=opts["launch_timeout"], label="gateway"
        )
    except BaseException:
        kill_pids([gateway_pid, router_pid], grace_s=5.0)
        raise

    backend = opts.get("scheduler") or "local"

    service_state = ServiceState(
        service=service,
        backend=backend,
        gateway_handle=TaskHandle(
            host=host_for_url,
            ports=[opts["port"]],
            gpu_devices=[],
            ref={"pid": gateway_pid},
        ),
        router_handle=TaskHandle(
            host="127.0.0.1",
            ports=[router_port],
            gpu_devices=[],
            ref={"pid": router_pid},
        ),
        admin_api_key=opts["admin_api_key"],
        started_at=time.time(),
    )
    model_state = ModelState(service=service)
    with locked_model_state(service):
        service_state.save()
        model_state.save()

        if opts["model"]:
            try:
                entry = register_model(
                    model_name=opts["model"],
                    opts=opts,
                    gateway=gateway_client,
                    router=router_client,
                    log_dir=log_dir,
                    admin_api_key=opts["admin_api_key"],
                    scheduler_backend=backend,
                    occupied_gpus=model_state.occupied_gpus(),
                )
                model_state.models[opts["model"]] = entry
                model_state.save()
            except BaseException:
                _cleanup_runtime(service, service_state, model_state, grace_s=5.0)
                raise

    logger.info("service %r ready pid=%d url=%s", service, gateway_pid, gateway_url)
    if opts["model"]:
        logger.info("default model: %s (%s)", opts["model"], opts["backend"])

    if opts["detach"]:
        return 0

    logger.info(
        "foreground watcher running. Ctrl+C / SIGTERM tears down the service; "
        "closing the terminal (SIGHUP) detaches the watcher and leaves it "
        "running — use `areal inf stop --service %s` to terminate.",
        service,
    )
    watcher = ForegroundWatcher(
        is_alive=lambda: inf_lifecycle.gateway_alive(
            RuntimeState(service_state=service_state, model_state=model_state)
        ),
        teardown=lambda: _cleanup_runtime(
            service, service_state, model_state, grace_s=10.0
        ),
        service_name=service,
    )
    rc = watcher.watch()
    # gateway died externally (is_alive flipped to False on its own) — the
    # children are gone, just clean the state files.
    if rc == 0 and not inf_lifecycle.gateway_alive(
        RuntimeState(service_state=service_state, model_state=model_state)
    ):
        ServiceState.remove(service)
        ModelState.remove(service)
    return rc


def _cleanup_runtime(
    service: str,
    service_state: ServiceState,
    model_state: ModelState,
    *,
    grace_s: float,
) -> None:
    model_state = _latest_model_state_for_cleanup(service, model_state)
    try:
        terminate_runtime_state(
            RuntimeState(service_state=service_state, model_state=model_state),
            grace_s=grace_s,
        )
    finally:
        ServiceState.remove(service)
        ModelState.remove(service)


def _latest_model_state_for_cleanup(
    service: str,
    fallback: ModelState,
) -> ModelState:
    try:
        latest = ModelState.load(service)
    except Exception:
        return fallback
    latest.models.update(fallback.models)
    return latest
