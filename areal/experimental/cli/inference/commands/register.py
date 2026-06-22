# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import click

from areal.experimental.cli.inference.client import GatewayClient, RouterClient
from areal.experimental.cli.inference.common import (
    ENGINE_ARGS_HELP,
    PROXY_ARGS_HELP,
    load_running_state,
    logger,
    register_external,
    register_internal,
    resolve_provider_api_key,
    split_args,
)
from areal.experimental.cli.inference.state import ModelEntry, logs_dir


@click.command(name="register", help="Register a model against a running service.")
@click.argument("name")
@click.option("--service", default=None, help="Target service instance.")
@click.option("--api-url", default=None, help="External provider URL.")
@click.option("--provider-api-key", default=None)
@click.option("--provider-api-key-env", default=None)
@click.option("--provider-model", default=None)
@click.option("--backend", default=None, help="Internal backend spec.")
@click.option("--model-path", default=None)
@click.option("--tokenizer-path", default=None)
@click.option("--engine-args", default="", help=ENGINE_ARGS_HELP)
@click.option("--proxy-args", default="", help=PROXY_ARGS_HELP)
@click.option("--model-health-timeout", type=float, default=600.0, show_default=True)
@click.option(
    "--log-level",
    type=click.Choice(["debug", "info", "warning", "error"]),
    default="info",
    show_default=True,
)
def register_cmd(name: str, service: str | None, **opts) -> None:
    raise SystemExit(do_register(name, opts, service=service) or 0)


def do_register(name: str, opts: dict, *, service: str | None = None) -> int:
    state = load_running_state(service)
    if opts["api_url"] and opts["backend"]:
        raise click.UsageError("Use --api-url OR --backend, not both.")
    if not opts["api_url"] and not opts["backend"]:
        raise click.UsageError("Provide --api-url <url> or --backend <spec>.")
    if opts["backend"] and not opts["model_path"]:
        raise click.UsageError("--backend requires --model-path.")
    if name in state.models:
        raise click.ClickException(
            f"model {name!r} already registered in service {state.service!r}"
        )

    gateway = GatewayClient(state.gateway_url, state.admin_api_key)
    router = RouterClient(state.router_url, state.admin_api_key)
    if opts["api_url"]:
        api_key = resolve_provider_api_key(
            provider_api_key=opts["provider_api_key"],
            provider_api_key_env=opts["provider_api_key_env"],
        )
        register_external(
            model=name,
            api_url=opts["api_url"],
            api_key=api_key,
            provider_model=opts["provider_model"],
            gateway=gateway,
        )
        state.model_state.models[name] = ModelEntry(
            kind="external", api_url=opts["api_url"]
        )
        state.model_state.set_default_if_empty(name)
        state.model_state.save()
        logger.info("registered external model %r in service %r", name, state.service)
        return 0

    pids, proxy_addrs, inf_addrs, base_gpu, n_gpu = register_internal(
        model=name,
        backend=opts["backend"],
        model_path=opts["model_path"],
        tokenizer_path=opts["tokenizer_path"] or opts["model_path"],
        engine_extra=split_args(opts["engine_args"]),
        proxy_extra=split_args(opts["proxy_args"]),
        model_health_timeout=opts["model_health_timeout"],
        log_level=opts["log_level"],
        admin_api_key=state.admin_api_key,
        gateway=gateway,
        router=router,
        log_dir=logs_dir(state.service),
        base_gpu_id=state.model_state.next_gpu_id,
    )
    state.model_state.models[name] = ModelEntry(
        kind="internal",
        backend=opts["backend"],
        base_gpu_id=base_gpu,
        gpu_count=n_gpu,
        pids=pids,
        proxy_addrs=proxy_addrs,
        inference_server_addrs=inf_addrs,
    )
    state.model_state.next_gpu_id = base_gpu + n_gpu
    state.model_state.set_default_if_empty(name)
    state.model_state.save()
    logger.info(
        "registered internal model %r in service %r (%d worker(s), GPU %d-%d)",
        name,
        state.service,
        len(pids),
        base_gpu,
        base_gpu + n_gpu - 1,
    )
    return 0
