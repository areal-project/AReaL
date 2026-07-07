# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import click

from areal.v2.cli.inference.client import GatewayClient, RouterClient
from areal.v2.cli.inference.common import (
    ENGINE_ARGS_HELP,
    PROXY_ARGS_HELP,
    logger,
    register_model,
)
from areal.v2.cli.inference.lifecycle import inf_lifecycle
from areal.v2.cli.inference.state import store


@click.command(name="register", help="Register a model against a running service.")
@click.option("--model-name", required=True, help="Model name to register.")
@click.option("--service", default=None, help="Target service instance.")
@click.option(
    "--backend",
    default=None,
    help="Backend spec; same grammar as InferenceEngineConfig.backend, "
    "e.g. 'sglang:d4', 'vllm:d2t4'.",
)
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
def register_cmd(model_name: str, service: str | None, **opts) -> None:
    raise SystemExit(do_register(model_name, opts, service=service) or 0)


def do_register(model_name: str, opts: dict, *, service: str | None = None) -> int:
    service_name = inf_lifecycle.resolve_service_name(service)
    with store.lock_model_state(service_name):
        state = inf_lifecycle.load_running_state(service_name)
        if model_name in state.models:
            raise click.ClickException(
                f"model {model_name!r} already registered in service {state.service!r}"
            )

        entry = register_model(
            model_name=model_name,
            opts=opts,
            gateway=GatewayClient(state.gateway_url, state.admin_api_key),
            router=RouterClient(state.router_url, state.admin_api_key),
            log_dir=store.logs_dir(state.service),
            admin_api_key=state.admin_api_key,
            scheduler_backend=state.backend,
            occupied_gpus=state.model_state.occupied_gpus(),
        )
        state.model_state.models[model_name] = entry
        state.model_state.save()
        gpus_used = sorted({g for r in entry.replicas for g in r.worker.gpu_devices})
        logger.info(
            "registered model %r in service %r (%d replica(s), GPUs=%s)",
            model_name,
            state.service,
            len(entry.replicas),
            gpus_used,
        )
    return 0
