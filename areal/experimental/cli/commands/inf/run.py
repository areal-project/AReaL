# SPDX-License-Identifier: Apache-2.0

"""``areal inf run`` — launch the inference service (detached).

Spawns a gateway and a router as detached subprocesses, polls the gateway's
``/health`` until it answers, optionally registers a model inline (external
API or local sglang/vllm), persists ``ServiceState`` (and ``ModelState`` if
a model was registered) under ``~/.areal/inf/``, and exits.

There is **no supervisor process**.  Subsequent commands (``stop``,
``status``, ``ps``, ``logs``) reconcile via ``pid_alive`` + gateway
``/health``.

Examples::

    # empty service
    areal inf run

    # custom ports
    areal inf run --service demo --gateway-port 18080 --router-port 18081

    # inline external model
    areal inf run --model gpt-4o --api-url https://api.openai.com/v1 \\
                  --provider-api-key-env OPENAI_API_KEY

    # inline internal model (spawns sglang locally)
    areal inf run --model qwen3 --backend sglang:tp=2 \\
                  --model-path Qwen/Qwen3-8B
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from areal.utils.logging import getLogger

logger = getLogger("InfCli")


_DESCRIPTION = __doc__


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "run",
        help="Launch the inference service (detached).",
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Service arguments (design 11.2)
    p.add_argument("--service", default="default", help="Service instance name.")
    p.add_argument("--gateway-host", default="127.0.0.1")
    p.add_argument("--gateway-port", type=int, default=8080)
    p.add_argument("--router-host", default="127.0.0.1")
    p.add_argument("--router-port", type=int, default=8081)
    p.add_argument("--admin-api-key", default="areal-admin-key")
    p.add_argument(
        "--routing-strategy", default="round_robin",
        choices=["round_robin", "least_busy"],
    )
    p.add_argument("--poll-interval", type=float, default=5.0)
    p.add_argument("--router-timeout", type=float, default=2.0)
    p.add_argument("--forward-timeout", type=float, default=120.0)
    p.add_argument(
        "--log-level", default="info",
        choices=["debug", "info", "warning", "error"],
    )
    p.add_argument("--launch-timeout", type=float, default=30.0)
    p.add_argument(
        "--config", type=Path, default=None,
        help="Optional TOML override file merged over ~/.areal/inf/config.toml.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Stop an existing healthy service with the same name first.",
    )

    # Inline model registration (external if --api-url, else internal).
    p.add_argument(
        "--model", default=None,
        help="Model name to register at startup. Triggers inline registration.",
    )

    # External-model flags
    p.add_argument(
        "--api-url", default=None,
        help="External provider URL.  Presence marks the model as external.",
    )
    p.add_argument("--provider-api-key", default=None)
    p.add_argument(
        "--provider-api-key-env", default=None,
        help="Name of an environment variable holding the provider API key.",
    )
    p.add_argument(
        "--provider-model", default=None,
        help="Upstream model name to send to the provider (defaults to --model).",
    )

    # Internal-model flags
    p.add_argument(
        "--backend", default=None,
        help="Backend spec for internal model, e.g. 'sglang', 'sglang:tp=2', "
             "'vllm:tp=2,dp=2'.  Required for internal models.",
    )
    p.add_argument(
        "--model-path", default=None,
        help="HuggingFace or local path to weights (internal models).",
    )
    p.add_argument(
        "--tokenizer-path", default=None,
        help="Tokenizer path for the data-proxy.  Defaults to --model-path.",
    )
    p.add_argument(
        "--model-health-timeout", type=float, default=600.0,
        help="Seconds to wait for an internal model server to become healthy.",
    )
    p.add_argument(
        "--engine-args", default="",
        help="Extra args forwarded verbatim to the sglang / vllm process. "
             "Pass as a single shell-style string, e.g. "
             "'--mem-fraction-static 0.85 --chunked-prefill-size 4096'. "
             "See the sglang or vllm CLI docs for the full flag set.",
    )
    p.add_argument(
        "--proxy-args", default="",
        help="Extra args forwarded verbatim to the data-proxy process. "
             "Pass as a single shell-style string, e.g. "
             "'--tool-call-parser qwen --reasoning-parser qwen3 "
             "--chat-template-type hf'. See "
             "`python -m areal.experimental.inference_service.data_proxy --help` "
             "for the full flag set.",
    )

    p.set_defaults(func=_handle)


# ---- helpers --------------------------------------------------------------


def _resolve_provider_api_key(args: argparse.Namespace) -> str:
    if args.provider_api_key:
        return args.provider_api_key
    if args.provider_api_key_env:
        v = os.environ.get(args.provider_api_key_env)
        if not v:
            raise SystemExit(
                f"--provider-api-key-env={args.provider_api_key_env!r} "
                f"is not set in the environment."
            )
        return v
    raise SystemExit(
        "External model registration requires either --provider-api-key or "
        "--provider-api-key-env."
    )


def _refuse_or_replace(name: str, force: bool) -> None:
    from areal.experimental.cli.commands.inf.gateway_client import (
        GatewayClient,
        GatewayUnreachable,
    )
    from areal.experimental.cli.commands.inf.launcher import kill_pids
    from areal.experimental.cli.commands.inf.state import (
        ServiceModels,
        ServiceState,
        gateway_alive,
        models_state_path,
        router_alive,
        service_state_path,
    )

    p = service_state_path(name)
    if not p.exists():
        return
    try:
        existing = ServiceState.load(name)
    except (FileNotFoundError, ValueError, TypeError):
        p.unlink()
        return

    pid_says_alive = gateway_alive(existing) or router_alive(existing)
    healthy = False
    if pid_says_alive:
        # Confirm via HTTP — PIDs can be reused after a crash; only a real
        # /health response means the gateway is actually serving.
        try:
            GatewayClient(
                existing.gateway_url, admin_api_key=existing.admin_api_key,
                timeout=1.0,
            ).health()
            healthy = True
        except GatewayUnreachable:
            healthy = False

    if healthy and not force:
        raise SystemExit(
            f"Service {name!r} is already running "
            f"(gateway pid={existing.gateway_pid}, router pid={existing.router_pid}). "
            f"Use --force to replace it, or `areal inf stop {name}` first."
        )
    if not healthy and pid_says_alive:
        logger.warning(
            "Service %r has live pids (gateway=%d, router=%d) but gateway "
            "is unreachable; treating as stale and reclaiming.",
            name, existing.gateway_pid, existing.router_pid,
        )
    if healthy or pid_says_alive:
        # Also kill any tracked model workers from the previous incarnation.
        worker_pids: list[int] = []
        mp = models_state_path(name)
        if mp.exists():
            sm = ServiceModels.load(name)
            for m in sm.list_all():
                worker_pids.extend(m.worker_pids)
        kill_pids(
            [existing.gateway_pid, existing.router_pid, *worker_pids],
            grace_s=10.0,
        )

    existing.remove()
    mp = models_state_path(name)
    if mp.exists():
        mp.unlink()


def _wait_health(client, supervisor_pids: list[int], deadline: float) -> None:
    from areal.experimental.cli.commands.inf.gateway_client import GatewayUnreachable
    from areal.experimental.cli.state import pid_alive

    last_err: Exception | None = None
    while time.time() < deadline:
        if not all(pid_alive(p) for p in supervisor_pids):
            raise SystemExit(
                "Gateway or router subprocess died during startup."
            )
        try:
            client.health()
            return
        except GatewayUnreachable as e:
            last_err = e
            time.sleep(0.5)
    raise SystemExit(
        f"Service did not become healthy within timeout. Last error: {last_err}"
    )


def _register_external_inline(
    *, args: argparse.Namespace, service_name: str, gateway_url: str
) -> None:
    from areal.experimental.cli.commands.inf.gateway_client import (
        GatewayClient,
        GatewayHTTPError,
        GatewayUnreachable,
    )
    from areal.experimental.cli.commands.inf.state import (
        ModelState,
        ServiceModels,
    )

    api_key = _resolve_provider_api_key(args)
    payload = {
        "model": args.model,
        "url": args.api_url,
        "api_key": api_key,
        "data_proxy_addrs": [],
    }
    if args.provider_model:
        payload["provider_model"] = args.provider_model

    client = GatewayClient(
        gateway_url, admin_api_key=args.admin_api_key, timeout=10.0
    )
    try:
        client.register_model(payload)
    except (GatewayUnreachable, GatewayHTTPError) as e:
        raise SystemExit(
            f"Inline register of model {args.model!r} failed: {e}"
        ) from e

    models = ServiceModels.load(service_name)
    models.add(
        ModelState(
            name=args.model,
            kind="external",
            api_url=args.api_url,
            provider_model=args.provider_model or args.model,
            registered_at=time.time(),
        )
    )
    models.save()


def _register_internal_inline(
    *, args: argparse.Namespace, service_name: str,
    gateway_url: str, router_url: str, log_dir: Path,
) -> None:
    from areal.experimental.cli.commands.inf.register_helper import (
        InternalRegisterArgs,
        register_internal_model,
    )
    from areal.experimental.cli.commands.inf.state import ModelState, ServiceModels

    import shlex

    if not args.model_path:
        raise SystemExit(
            "--model-path is required for internal model registration."
        )

    result = register_internal_model(
        InternalRegisterArgs(
            model_name=args.model,
            backend_spec=args.backend,
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path or args.model_path,
            log_dir=log_dir,
            admin_api_key=args.admin_api_key,
            log_level=args.log_level,
            health_timeout=args.model_health_timeout,
            engine_extra_args=shlex.split(args.engine_args) if args.engine_args else [],
            proxy_extra_args=shlex.split(args.proxy_args) if args.proxy_args else [],
        ),
        gateway_url=gateway_url,
        router_url=router_url,
    )

    models = ServiceModels.load(service_name)
    models.add(
        ModelState(
            name=args.model,
            kind="internal",
            backend_spec=args.backend,
            data_proxy_addrs=result.data_proxy_addrs,
            inference_server_addrs=result.inference_server_addrs,
            worker_pids=result.worker_pids,
            registered_at=time.time(),
        )
    )
    models.save()


# ---- main entry ----------------------------------------------------------


def _handle(args: argparse.Namespace) -> int:
    # Sanity-check model flags up front.
    if args.api_url and not args.model:
        raise SystemExit("--api-url requires --model.")
    if args.backend and not args.model:
        raise SystemExit("--backend requires --model.")
    if args.model and args.api_url and args.backend:
        raise SystemExit(
            "Specify either --api-url (external) OR --backend (internal), not both."
        )
    if args.model and not (args.api_url or args.backend):
        raise SystemExit(
            "--model requires either --api-url <url> (external) or "
            "--backend <spec> --model-path <path> (internal)."
        )

    from areal.experimental.cli.commands.inf.gateway_client import GatewayClient
    from areal.experimental.cli.commands.inf.launcher import (
        kill_pids,
        spawn_gateway,
        spawn_router,
    )
    from areal.experimental.cli.commands.inf.state import (
        ServiceState,
        get_current_service,
        service_logs_dir,
        set_current_service,
    )

    service = args.service
    _refuse_or_replace(service, force=args.force)

    logs = service_logs_dir(service)
    logger.info("Starting service %r (logs: %s)", service, logs)

    router_pid = spawn_router(
        host=args.router_host,
        port=args.router_port,
        admin_api_key=args.admin_api_key,
        poll_interval=args.poll_interval,
        routing_strategy=args.routing_strategy,
        log_level=args.log_level,
        log_file=logs / "router.log",
    )
    logger.info("Spawned router pid=%d", router_pid)

    time.sleep(0.3)

    gateway_pid = spawn_gateway(
        host=args.gateway_host,
        port=args.gateway_port,
        admin_api_key=args.admin_api_key,
        router_host=args.router_host,
        router_port=args.router_port,
        router_timeout=args.router_timeout,
        forward_timeout=args.forward_timeout,
        log_level=args.log_level,
        log_file=logs / "gateway.log",
    )
    logger.info("Spawned gateway pid=%d", gateway_pid)

    state = ServiceState(
        name=service,
        gateway_host=args.gateway_host,
        gateway_port=args.gateway_port,
        router_host=args.router_host,
        router_port=args.router_port,
        gateway_pid=gateway_pid,
        router_pid=router_pid,
        admin_api_key=args.admin_api_key,
        routing_strategy=args.routing_strategy,
        log_level=args.log_level,
        created_at=time.time(),
    )

    client = GatewayClient(
        state.gateway_url, admin_api_key=args.admin_api_key, timeout=2.0
    )
    try:
        _wait_health(client, [router_pid, gateway_pid],
                     deadline=time.time() + args.launch_timeout)
    except SystemExit:
        kill_pids([gateway_pid, router_pid], grace_s=5.0)
        raise

    state.save()

    if args.model:
        try:
            if args.api_url:
                _register_external_inline(
                    args=args, service_name=service, gateway_url=state.gateway_url
                )
            else:
                _register_internal_inline(
                    args=args, service_name=service,
                    gateway_url=state.gateway_url,
                    router_url=state.router_url,
                    log_dir=logs,
                )
        except SystemExit:
            kill_pids([gateway_pid, router_pid], grace_s=5.0)
            state.remove()
            raise

    if get_current_service() is None:
        set_current_service(service)

    logger.info("Service %r ready.", service)
    logger.info("  gateway: %s", state.gateway_url)
    logger.info("  router:  %s", state.router_url)
    logger.info("  pids:    gateway=%d, router=%d", gateway_pid, router_pid)
    if args.model:
        kind = "external" if args.api_url else f"internal ({args.backend})"
        logger.info("  default model: %s (%s)", args.model, kind)
    logger.info("  log dir: %s", logs)
    return 0
