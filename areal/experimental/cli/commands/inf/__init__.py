# SPDX-License-Identifier: Apache-2.0

"""``areal inf`` — Ollama-style inference service operator console.

This namespace owns its OWN service lifecycle (gateway + router +
optional model backends).  Services are standalone — they don't need a
training yaml, an experiment_name, or any cluster scheduler integration.
A user starts a serving stack, registers external or internal models,
and chats / collects against them.

A separate training surface (``areal train ...``) will land in a later
PR; the two namespaces are intentionally decoupled — training experiments
wrap ``RolloutControllerV2`` in-process, while ``inf`` spawns
gateway/router/data-proxy as detached subprocesses owned by nothing but
the kernel.

All five Phase-1 verbs live in this file so the surface stays scannable
at a glance.  Non-verb logic (process spawning, HTTP client, on-disk
state, internal-model spawn pipeline) lives in sibling modules:

    state.py            ServiceState / ModelState / ServiceModels + paths
    launcher.py         spawn_router / spawn_gateway / kill_pids
    gateway_client.py   urllib HTTP client (gateway + router endpoints)
    register_helper.py  internal-model spawn pipeline
    config.py           ~/.areal/inf/config.toml loader

State lives under ``~/.areal/inf/``; see ``state.py`` for layout.
"""

from __future__ import annotations

import json
import os
import shlex
import sys
import time
from pathlib import Path

import click

from areal.utils.logging import getLogger

logger = getLogger("InfCli")


@click.group(help="Manage inference services and models.")
def inf() -> None:
    pass


# =========================================================================
# `areal inf run` — launch the inference service (detached)
# =========================================================================


@inf.command(name="run", help="Launch the inference service (detached).")
# Service flags
@click.option("--service", default="default", help="Service instance name.")
@click.option("--gateway-host", default="127.0.0.1")
@click.option("--gateway-port", type=int, default=8080)
@click.option("--router-host", default="127.0.0.1")
@click.option("--router-port", type=int, default=8081)
@click.option("--admin-api-key", default="areal-admin-key")
@click.option(
    "--routing-strategy",
    type=click.Choice(["round_robin", "least_busy"]),
    default="round_robin",
)
@click.option("--poll-interval", type=float, default=5.0)
@click.option("--router-timeout", type=float, default=2.0)
@click.option("--forward-timeout", type=float, default=120.0)
@click.option(
    "--log-level",
    type=click.Choice(["debug", "info", "warning", "error"]),
    default="info",
)
@click.option("--launch-timeout", type=float, default=30.0)
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional TOML override file merged over ~/.areal/inf/config.toml.",
)
@click.option(
    "--force/--no-force",
    default=False,
    help="Stop an existing healthy service with the same name first.",
)
# Inline model registration
@click.option(
    "--model",
    default=None,
    help="Model name to register at startup. Triggers inline registration.",
)
# External-model flags
@click.option(
    "--api-url",
    default=None,
    help="External provider URL. Presence marks the model as external.",
)
@click.option("--provider-api-key", default=None)
@click.option(
    "--provider-api-key-env",
    default=None,
    help="Name of an environment variable holding the provider API key.",
)
@click.option(
    "--provider-model",
    default=None,
    help="Upstream model name to send to the provider (defaults to --model).",
)
# Internal-model flags
@click.option(
    "--backend",
    default=None,
    help="Backend spec for internal model, e.g. 'sglang', 'sglang:tp=2', "
    "'vllm:tp=2,dp=2'.",
)
@click.option(
    "--model-path",
    default=None,
    help="HuggingFace or local path to weights (internal models).",
)
@click.option(
    "--tokenizer-path",
    default=None,
    help="Tokenizer path for the data-proxy. Defaults to --model-path.",
)
@click.option(
    "--model-health-timeout",
    type=float,
    default=600.0,
    help="Seconds to wait for an internal model server to become healthy.",
)
@click.option(
    "--engine-args",
    default="",
    help="Extra args forwarded verbatim to the sglang / vllm process. "
    "Shell-style string, e.g. '--mem-fraction-static 0.85'.",
)
@click.option(
    "--proxy-args",
    default="",
    help="Extra args forwarded verbatim to the data-proxy process. "
    "Shell-style string, e.g. '--tool-call-parser qwen'.",
)
def _run(**opts) -> None:
    raise SystemExit(_do_run(opts) or 0)


def _resolve_provider_api_key(opts: dict) -> str:
    if opts["provider_api_key"]:
        return opts["provider_api_key"]
    env_name = opts["provider_api_key_env"]
    if env_name:
        v = os.environ.get(env_name)
        if not v:
            raise SystemExit(
                f"--provider-api-key-env={env_name!r} is not set in the environment."
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
        try:
            GatewayClient(
                existing.gateway_url,
                admin_api_key=existing.admin_api_key,
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
            name,
            existing.gateway_pid,
            existing.router_pid,
        )
    if healthy or pid_says_alive:
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


def _wait_gateway_health(client, supervisor_pids: list[int], deadline: float) -> None:
    from areal.experimental.cli.commands.inf.gateway_client import GatewayUnreachable
    from areal.experimental.cli.state import pid_alive

    last_err: Exception | None = None
    while time.time() < deadline:
        if not all(pid_alive(p) for p in supervisor_pids):
            raise SystemExit("Gateway or router subprocess died during startup.")
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
    *, opts: dict, service_name: str, gateway_url: str
) -> None:
    from areal.experimental.cli.commands.inf.gateway_client import (
        GatewayClient,
        GatewayHTTPError,
        GatewayUnreachable,
    )
    from areal.experimental.cli.commands.inf.state import ModelState, ServiceModels

    api_key = _resolve_provider_api_key(opts)
    payload = {
        "model": opts["model"],
        "url": opts["api_url"],
        "api_key": api_key,
        "data_proxy_addrs": [],
    }
    if opts["provider_model"]:
        payload["provider_model"] = opts["provider_model"]

    client = GatewayClient(
        gateway_url, admin_api_key=opts["admin_api_key"], timeout=10.0
    )
    try:
        client.register_model(payload)
    except (GatewayUnreachable, GatewayHTTPError) as e:
        raise SystemExit(
            f"Inline register of model {opts['model']!r} failed: {e}"
        ) from e

    models = ServiceModels.load(service_name)
    models.add(
        ModelState(
            name=opts["model"],
            kind="external",
            api_url=opts["api_url"],
            provider_model=opts["provider_model"] or opts["model"],
            registered_at=time.time(),
        )
    )
    models.save()


def _register_internal_inline(
    *, opts: dict, service_name: str,
    gateway_url: str, router_url: str, log_dir: Path,
) -> None:
    from areal.experimental.cli.commands.inf.register_helper import (
        InternalRegisterArgs,
        register_internal_model,
    )
    from areal.experimental.cli.commands.inf.state import ModelState, ServiceModels

    if not opts["model_path"]:
        raise SystemExit("--model-path is required for internal model registration.")

    result = register_internal_model(
        InternalRegisterArgs(
            model_name=opts["model"],
            backend_spec=opts["backend"],
            model_path=opts["model_path"],
            tokenizer_path=opts["tokenizer_path"] or opts["model_path"],
            log_dir=log_dir,
            admin_api_key=opts["admin_api_key"],
            log_level=opts["log_level"],
            health_timeout=opts["model_health_timeout"],
            engine_extra_args=shlex.split(opts["engine_args"]) if opts["engine_args"] else [],
            proxy_extra_args=shlex.split(opts["proxy_args"]) if opts["proxy_args"] else [],
        ),
        gateway_url=gateway_url,
        router_url=router_url,
    )

    models = ServiceModels.load(service_name)
    models.add(
        ModelState(
            name=opts["model"],
            kind="internal",
            backend_spec=opts["backend"],
            data_proxy_addrs=result.data_proxy_addrs,
            inference_server_addrs=result.inference_server_addrs,
            worker_pids=result.worker_pids,
            registered_at=time.time(),
        )
    )
    models.save()


def _do_run(opts: dict) -> int:
    # Sanity-check model flags up front.
    if opts["api_url"] and not opts["model"]:
        raise SystemExit("--api-url requires --model.")
    if opts["backend"] and not opts["model"]:
        raise SystemExit("--backend requires --model.")
    if opts["model"] and opts["api_url"] and opts["backend"]:
        raise SystemExit(
            "Specify either --api-url (external) OR --backend (internal), not both."
        )
    if opts["model"] and not (opts["api_url"] or opts["backend"]):
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

    service = opts["service"]
    _refuse_or_replace(service, force=opts["force"])

    logs = service_logs_dir(service)
    logger.info("Starting service %r (logs: %s)", service, logs)

    router_pid = spawn_router(
        host=opts["router_host"],
        port=opts["router_port"],
        admin_api_key=opts["admin_api_key"],
        poll_interval=opts["poll_interval"],
        routing_strategy=opts["routing_strategy"],
        log_level=opts["log_level"],
        log_file=logs / "router.log",
    )
    logger.info("Spawned router pid=%d", router_pid)

    time.sleep(0.3)

    gateway_pid = spawn_gateway(
        host=opts["gateway_host"],
        port=opts["gateway_port"],
        admin_api_key=opts["admin_api_key"],
        router_host=opts["router_host"],
        router_port=opts["router_port"],
        router_timeout=opts["router_timeout"],
        forward_timeout=opts["forward_timeout"],
        log_level=opts["log_level"],
        log_file=logs / "gateway.log",
    )
    logger.info("Spawned gateway pid=%d", gateway_pid)

    state = ServiceState(
        name=service,
        gateway_host=opts["gateway_host"],
        gateway_port=opts["gateway_port"],
        router_host=opts["router_host"],
        router_port=opts["router_port"],
        gateway_pid=gateway_pid,
        router_pid=router_pid,
        admin_api_key=opts["admin_api_key"],
        routing_strategy=opts["routing_strategy"],
        log_level=opts["log_level"],
        created_at=time.time(),
    )

    client = GatewayClient(
        state.gateway_url, admin_api_key=opts["admin_api_key"], timeout=2.0
    )
    try:
        _wait_gateway_health(
            client,
            [router_pid, gateway_pid],
            deadline=time.time() + opts["launch_timeout"],
        )
    except SystemExit:
        kill_pids([gateway_pid, router_pid], grace_s=5.0)
        raise

    state.save()

    if opts["model"]:
        try:
            if opts["api_url"]:
                _register_external_inline(
                    opts=opts, service_name=service, gateway_url=state.gateway_url
                )
            else:
                _register_internal_inline(
                    opts=opts, service_name=service,
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
    if opts["model"]:
        kind = "external" if opts["api_url"] else f"internal ({opts['backend']})"
        logger.info("  default model: %s (%s)", opts["model"], kind)
    logger.info("  log dir: %s", logs)
    return 0


# =========================================================================
# `areal inf stop` — stop a running inference service
# =========================================================================


@inf.command(name="stop", help="Stop a running inference service.")
@click.argument("name", required=False)
@click.option(
    "--grace-period", type=float, default=10.0,
    help="Seconds to wait before escalating to SIGKILL.",
)
@click.option(
    "--keep-state", is_flag=True,
    help="Keep state files after shutdown (debugging).",
)
@click.option(
    "--force", is_flag=True,
    help="Skip confirmations; reserved for future interactive prompts.",
)
def _stop(name: str | None, grace_period: float, keep_state: bool, force: bool) -> None:
    raise SystemExit(_do_stop(name, grace_period, keep_state, force) or 0)


def _do_stop(
    name_arg: str | None, grace_period: float, keep_state: bool, force: bool
) -> int:
    from areal.experimental.cli.commands.inf.gateway_client import (
        GatewayClient,
        GatewayUnreachable,
    )
    from areal.experimental.cli.commands.inf.launcher import kill_pids
    from areal.experimental.cli.commands.inf.state import (
        ServiceModels,
        ServiceState,
        gateway_alive,
        get_current_service,
        models_state_path,
        resolve_service,
        router_alive,
        set_current_service,
    )

    name = resolve_service(name_arg)

    try:
        state = ServiceState.load(name)
    except FileNotFoundError:
        logger.error("No service named %r.", name)
        return 1

    pids: list[int] = [state.gateway_pid, state.router_pid]
    models_path = models_state_path(name)
    if models_path.exists():
        sm = ServiceModels.load(name)
        for m in sm.list_all():
            for pid in m.worker_pids:
                if pid > 0:
                    pids.append(pid)

    alive = gateway_alive(state) or router_alive(state)
    if not alive:
        logger.warning(
            "Service %r is already down (no live gateway/router pid); "
            "cleaning up state.", name,
        )
    else:
        logger.info(
            "Stopping service %r: gateway=%d, router=%d ...",
            name, state.gateway_pid, state.router_pid,
        )
        kill_pids(pids, grace_s=grace_period)

        client = GatewayClient(state.gateway_url, timeout=1.0)
        deadline = time.time() + min(5.0, grace_period)
        while time.time() < deadline:
            try:
                client.health()
                time.sleep(0.3)
            except GatewayUnreachable:
                break

    if not keep_state:
        state.remove()
        if models_path.exists():
            models_path.unlink()
        if get_current_service() == name:
            set_current_service(None)

    logger.info("Service %r stopped.", name)
    return 0


# =========================================================================
# `areal inf status` — show health for one service
# =========================================================================


@inf.command(name="status", help="Show service / component health.")
@click.argument("name", required=False)
@click.option("--watch", is_flag=True, help="Refresh until interrupted.")
@click.option("--interval", type=float, default=2.0)
@click.option("--json", "as_json", is_flag=True, help="Emit JSON.")
def _status(name: str | None, watch: bool, interval: float, as_json: bool) -> None:
    raise SystemExit(_do_status(name, watch, interval, as_json) or 0)


def _collect_status(name: str) -> dict:
    from areal.experimental.cli.commands.inf.gateway_client import (
        GatewayClient,
        GatewayUnreachable,
    )
    from areal.experimental.cli.commands.inf.state import (
        ServiceModels,
        ServiceState,
        gateway_alive,
        router_alive,
    )

    state = ServiceState.load(name)
    rows: list[dict] = []

    g_alive = gateway_alive(state)
    r_alive = router_alive(state)

    client = GatewayClient(
        state.gateway_url, admin_api_key=state.admin_api_key, timeout=2.0
    )
    gateway_status = "down"
    gateway_models_count = 0
    if g_alive:
        try:
            client.health()
            gateway_status = "ok"
            try:
                gw_models = client.list_models()
                if isinstance(gw_models, dict):
                    items = gw_models.get("data") or gw_models.get("models") or []
                    gateway_models_count = len(items) if isinstance(items, list) else 0
                elif isinstance(gw_models, list):
                    gateway_models_count = len(gw_models)
            except GatewayUnreachable:
                pass
        except GatewayUnreachable:
            gateway_status = "unreachable"

    router_status = "ok" if r_alive else "down"

    rows.append({
        "service": name, "component": "gateway", "status": gateway_status,
        "addr": f"{state.gateway_host}:{state.gateway_port}",
        "details": f"models={gateway_models_count}",
    })
    rows.append({
        "service": name, "component": "router", "status": router_status,
        "addr": f"{state.router_host}:{state.router_port}", "details": "",
    })

    sm = ServiceModels.load(name)
    for m in sm.list_all():
        details_parts = [f"kind={m.kind}"]
        if m.kind == "internal" and m.backend_spec:
            details_parts.append(f"backend={m.backend_spec}")
        if m.kind == "external" and m.api_url:
            details_parts.append(f"upstream={m.api_url}")
        if sm.default_model == m.name:
            details_parts.append("default")
        rows.append({
            "service": name, "component": m.name, "status": "registered",
            "addr": "internal" if m.kind == "internal" else "external",
            "details": " ".join(details_parts),
        })

    return {"service": name, "rows": rows, "default_model": sm.default_model}


def _print_status_table(snap: dict) -> None:
    cols = ("SERVICE", "COMPONENT", "STATUS", "ADDR", "DETAILS")
    rows = [
        (r["service"], r["component"], r["status"], r["addr"], r["details"])
        for r in snap["rows"]
    ]
    if not rows:
        return
    widths = [max(len(r[i]) for r in (cols, *rows)) for i in range(len(cols))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*cols))
    for r in rows:
        print(fmt.format(*r))


def _do_status(
    name_arg: str | None, watch: bool, interval: float, as_json: bool
) -> int:
    from areal.experimental.cli.commands.inf.state import resolve_service

    name = resolve_service(name_arg)
    try:
        if not watch:
            snap = _collect_status(name)
            if as_json:
                print(json.dumps(snap, indent=2))
            else:
                _print_status_table(snap)
            return 0
        while True:
            snap = _collect_status(name)
            if as_json:
                print(json.dumps(snap, indent=2))
            else:
                sys.stdout.write("\033[2J\033[H")
                _print_status_table(snap)
                sys.stdout.flush()
            time.sleep(interval)
    except FileNotFoundError:
        logger.error("No service named %r.", name)
        return 1
    except KeyboardInterrupt:
        return 0


# =========================================================================
# `areal inf ps` — list locally tracked services
# =========================================================================


@inf.command(name="ps", help="List locally tracked services.")
@click.option("--all", "show_all", is_flag=True, help="Include stale/dead services.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON.")
def _ps(show_all: bool, as_json: bool) -> None:
    raise SystemExit(_do_ps(show_all, as_json) or 0)


def _do_ps(show_all: bool, as_json: bool) -> int:
    from areal.experimental.cli.commands.inf.state import (
        ServiceModels,
        ServiceState,
        gateway_alive,
        get_current_service,
        router_alive,
        services_dir,
    )

    current = get_current_service()
    entries: list[dict] = []
    now = time.time()

    for f in sorted(services_dir().glob("*.json")):
        try:
            name = f.stem
            state = ServiceState.load(name)
        except (FileNotFoundError, ValueError, TypeError, KeyError):
            continue

        alive = gateway_alive(state) or router_alive(state)
        if not alive and not show_all:
            continue

        sm = ServiceModels.load(name)
        entries.append({
            "name": name,
            "current": name == current,
            "state": "running" if alive else "dead",
            "gateway": state.gateway_url,
            "router": state.router_url,
            "models": len(sm.models),
            "default_model": sm.default_model,
            "age_s": int(max(0, now - state.created_at)),
        })

    if as_json:
        print(json.dumps(entries, indent=2))
        return 0

    if not entries:
        msg = "No services."
        if not show_all:
            msg += "  (Add --all to include dead ones.)"
        logger.info("%s", msg)
        return 0

    cols = ("CURRENT", "NAME", "STATE", "GATEWAY", "ROUTER", "MODELS", "AGE")
    rows = [
        (
            "*" if e["current"] else "",
            e["name"], e["state"], e["gateway"], e["router"],
            f"{e['models']}"
            + (f" (default={e['default_model']})" if e["default_model"] else ""),
            f"{e['age_s']}s",
        )
        for e in entries
    ]
    widths = [max(len(r[i]) for r in (cols, *rows)) for i in range(len(cols))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*cols))
    for r in rows:
        print(fmt.format(*r))
    return 0


# =========================================================================
# `areal inf logs` — tail gateway / router / model logs
# =========================================================================


@inf.command(name="logs", help="Tail gateway / router / model logs.")
@click.argument("name", required=False)
@click.option(
    "--component", default="gateway",
    help="One of `gateway`, `router`, or a model name. "
    "Becomes `<component>.log` under the service log dir.",
)
@click.option("-f", "--follow", is_flag=True, help="Stream new lines.")
@click.option("-n", "--lines", type=int, default=200)
def _logs(name: str | None, component: str, follow: bool, lines: int) -> None:
    raise SystemExit(_do_logs(name, component, follow, lines) or 0)


def _do_logs(
    name_arg: str | None, component: str, follow: bool, lines: int
) -> int:
    from areal.experimental.cli.commands.inf.state import (
        resolve_service,
        service_logs_dir,
    )

    name = resolve_service(name_arg)
    log_dir = service_logs_dir(name)
    log_file = log_dir / f"{component}.log"
    if not log_file.exists():
        logger.error("No %s.log at %s.", component, log_file)
        return 1

    cmd = ["tail", f"-n{lines}"]
    if follow:
        cmd.append("-F")
    cmd.append(str(log_file))
    os.execvp(cmd[0], cmd)
