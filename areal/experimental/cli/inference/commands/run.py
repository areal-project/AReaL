# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import signal
import time

import click

from areal.experimental.cli.inference.client import GatewayClient, RouterClient
from areal.experimental.cli.inference.common import (
    ENGINE_ARGS_HELP,
    PROXY_ARGS_HELP,
    logger,
    refuse_if_running,
    register_external,
    register_internal,
    resolve_provider_api_key,
    split_args,
    terminate_runtime_state,
    wait_client_health,
)
from areal.experimental.cli.inference.launcher import spawn_gateway, spawn_router
from areal.experimental.cli.inference.state import (
    DEFAULT_SERVICE,
    ModelEntry,
    ModelState,
    RuntimeState,
    ServiceState,
    gateway_alive,
    locked_model_state,
    logs_dir,
    recover_pids_from_raw_state,
    service_state_path,
)
from areal.experimental.cli.process import kill_pids


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
@click.option("--api-url", default=None, help="External provider URL.")
@click.option("--provider-api-key", default=None)
@click.option(
    "--provider-api-key-env", default=None, help="Env var holding the provider API key."
)
@click.option("--provider-model", default=None, help="Upstream model name.")
@click.option(
    "--backend", default=None, help="Internal backend spec, e.g. 'sglang:tp=2,dp=2'."
)
@click.option("--model-path", default=None, help="HF / local model path.")
@click.option("--tokenizer-path", default=None, help="Tokenizer path.")
@click.option("--engine-args", default="", show_default=False, help=ENGINE_ARGS_HELP)
@click.option("--proxy-args", default="", show_default=False, help=PROXY_ARGS_HELP)
@click.option(
    "--model-health-timeout",
    type=float,
    default=600.0,
    show_default=True,
    help="Seconds to wait for the model server to come up.",
)
@click.option("--force", is_flag=True, help="Replace stale or running service state.")
def run_cmd(**opts) -> None:
    raise SystemExit(do_run(opts) or 0)


def do_run(opts: dict) -> int:
    service = opts["service"]
    _prepare_service_slot(service=service, force=opts["force"])

    if opts["model"]:
        if opts["api_url"] and opts["backend"]:
            raise click.UsageError(
                "Use --api-url (external) OR --backend (internal), not both."
            )
        if not opts["api_url"] and not opts["backend"]:
            raise click.UsageError(
                "--model requires --api-url <url> or --backend <spec>."
            )
        if opts["backend"] and not opts["model_path"]:
            raise click.UsageError("--backend requires --model-path <path>.")
    elif opts["api_url"] or opts["backend"]:
        raise click.UsageError("model registration flags require --model.")

    log_dir = logs_dir(service)
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

    service_state = ServiceState(
        service=service,
        gateway_pid=gateway_pid,
        gateway_url=gateway_url,
        router_pid=router_pid,
        router_url=router_url,
        admin_api_key=opts["admin_api_key"],
        started_at=time.time(),
    )
    model_state = ModelState(service=service)
    with locked_model_state(service):
        service_state.save()
        model_state.save()

        if opts["model"]:
            try:
                if opts["api_url"]:
                    api_key = resolve_provider_api_key(
                        provider_api_key=opts["provider_api_key"],
                        provider_api_key_env=opts["provider_api_key_env"],
                    )
                    register_external(
                        model=opts["model"],
                        api_url=opts["api_url"],
                        api_key=api_key,
                        provider_model=opts["provider_model"],
                        gateway=gateway_client,
                    )
                    model_state.models[opts["model"]] = ModelEntry(
                        kind="external", api_url=opts["api_url"]
                    )
                    model_state.set_default_if_empty(opts["model"])
                else:
                    (
                        pids,
                        engine_pids,
                        proxy_pids,
                        proxy_addrs,
                        inf_addrs,
                        base_gpu,
                        n_gpu,
                    ) = register_internal(
                        model=opts["model"],
                        backend=opts["backend"],
                        model_path=opts["model_path"],
                        tokenizer_path=opts["tokenizer_path"] or opts["model_path"],
                        engine_extra=split_args(opts["engine_args"]),
                        proxy_extra=split_args(opts["proxy_args"]),
                        model_health_timeout=opts["model_health_timeout"],
                        log_level=opts["log_level"],
                        admin_api_key=opts["admin_api_key"],
                        gateway=gateway_client,
                        router=router_client,
                        log_dir=log_dir,
                        base_gpu_id=model_state.next_gpu_id,
                    )
                    model_state.models[opts["model"]] = ModelEntry(
                        kind="internal",
                        backend=opts["backend"],
                        base_gpu_id=base_gpu,
                        gpu_count=n_gpu,
                        pids=pids,
                        engine_pids=engine_pids,
                        proxy_pids=proxy_pids,
                        proxy_addrs=proxy_addrs,
                        inference_server_addrs=inf_addrs,
                    )
                    model_state.next_gpu_id = base_gpu + n_gpu
                    model_state.set_default_if_empty(opts["model"])
                model_state.save()
            except BaseException:
                _cleanup_runtime(service, service_state, model_state, grace_s=5.0)
                raise

    logger.info("service %r ready pid=%d url=%s", service, gateway_pid, gateway_url)
    if opts["model"]:
        kind = "external" if opts["api_url"] else f"internal ({opts['backend']})"
        logger.info("default model: %s (%s)", opts["model"], kind)

    if opts["detach"]:
        return 0

    logger.info("foreground (Ctrl-C to stop) ...")
    previous_handlers = _install_shutdown_signal_handlers()
    try:
        while gateway_alive(service_state):
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("shutting down ...")
        _cleanup_runtime(service, service_state, model_state, grace_s=10.0)
        return 0
    except BaseException:
        _cleanup_runtime(service, service_state, model_state, grace_s=10.0)
        raise
    finally:
        _restore_signal_handlers(previous_handlers)

    logger.warning("gateway exited")
    ServiceState.remove(service)
    ModelState.remove(service)
    return 0


def _prepare_service_slot(*, service: str, force: bool) -> None:
    if not service_state_path(service).exists():
        return
    if force:
        try:
            from areal.experimental.cli.inference.state import load_runtime_state

            state = load_runtime_state(service)
            terminate_runtime_state(state, grace_s=5.0)
        except Exception as exc:
            try:
                pids = recover_pids_from_raw_state(service)
            except Exception as recover_exc:
                raise click.ClickException(
                    f"failed to load state for service {service!r}: {exc}; "
                    "refusing --force because no reliable PID list could be recovered"
                ) from recover_exc
            if not pids:
                raise click.ClickException(
                    f"failed to load state for service {service!r}: {exc}; "
                    "refusing --force because no PIDs could be recovered"
                ) from exc
            kill_pids(pids, grace_s=5.0)
        ServiceState.remove(service)
        ModelState.remove(service)
        return
    refuse_if_running(service)
    raise click.ClickException(
        f"stale state exists for service {service!r}; "
        f"use `areal inf run --service {service} --force`"
    )


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


def _install_shutdown_signal_handlers():
    def handle_shutdown(signum, frame):
        del signum, frame
        raise KeyboardInterrupt

    signals = [signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        signals.append(signal.SIGHUP)

    previous = {}
    for sig in signals:
        previous[sig] = signal.getsignal(sig)
        signal.signal(sig, handle_shutdown)
    return previous


def _restore_signal_handlers(previous_handlers) -> None:
    for sig, handler in previous_handlers.items():
        signal.signal(sig, handler)
