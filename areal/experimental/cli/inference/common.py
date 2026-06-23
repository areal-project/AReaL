# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import shlex
import signal
import time
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path

import click

from areal.experimental.cli.inference.client import (
    GatewayClient,
    GatewayHTTPError,
    GatewayUnreachable,
    RouterClient,
)
from areal.experimental.cli.inference.launcher import (
    spawn_data_proxy,
    spawn_sglang,
    spawn_vllm,
)
from areal.experimental.cli.inference.state import (
    DEFAULT_SERVICE,
    ModelState,
    RuntimeState,
    gateway_alive,
    list_service_names,
    load_runtime_state,
    resolve_service_name,
    service_state_path,
)
from areal.experimental.cli.process import (
    kill_pids,
    pick_free_port,
    pid_alive,
    signal_pid,
)
from areal.utils.logging import getLogger

logger = getLogger("InfCli")

ENGINE_ARGS_HELP = (
    "Shell-style string forwarded verbatim to the sglang / vllm process. "
    "Common sglang knobs: --mem-fraction-static 0.85, "
    "--max-running-requests 256, --chunked-prefill-size 4096, "
    "--disable-radix-cache, --enable-torch-compile. "
    "See the sglang / vllm CLI docs for the full surface."
)

PROXY_ARGS_HELP = (
    "Shell-style string forwarded verbatim to the data-proxy process. "
    "Available flags: --request-timeout, --set-reward-finish-timeout, "
    "--tool-call-parser, --reasoning-parser, --engine-max-tokens, "
    "--chat-template-type {hf|concat}."
)


def running_state(service: str | None = None) -> RuntimeState | None:
    service_name = resolve_service_name(service)
    if not service_state_path(service_name).exists():
        return None
    try:
        state = load_runtime_state(service_name)
    except Exception:
        return None
    if not gateway_alive(state):
        return None
    return state


def load_running_state(service: str | None = None) -> RuntimeState:
    service_name = resolve_service_name(service)
    if not service_state_path(service_name).exists():
        raise click.ClickException(f"service {service_name!r} is not running")
    try:
        state = load_runtime_state(service_name)
    except Exception as exc:
        raise click.ClickException(f"failed to load state: {exc}") from exc
    if not gateway_alive(state):
        raise click.ClickException(f"service {service_name!r} gateway pid not alive")
    return state


def refuse_if_running(service: str | None = None) -> None:
    service_name = service or DEFAULT_SERVICE
    state = running_state(service_name)
    if state is None:
        return
    raise click.ClickException(
        f"service {service_name!r} already running "
        f"(pid={state.gateway_pid}, url={state.gateway_url}). "
        f"Run `areal inf stop --service {service_name}` first."
    )


def resolve_model_name(state: RuntimeState, model: str | None) -> str:
    if model:
        return model
    if state.model_state.default_model:
        return state.model_state.default_model
    raise click.ClickException(
        f"service {state.service!r} has no default model; pass a model name"
    )


def wait_client_health(client, *, timeout: float, label: str) -> None:
    deadline = time.time() + timeout
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            client.health(timeout=1.5)
            return
        except (GatewayUnreachable, GatewayHTTPError) as exc:
            last_err = exc
            time.sleep(0.3)
    raise click.ClickException(
        f"{label} did not become healthy within {timeout:.0f}s (last error: {last_err})"
    )


def wait_http_health(addr: str, *, deadline: float, pid: int, label: str) -> None:
    last_err: Exception | None = None
    url = f"{addr}/health"
    while time.time() < deadline:
        if not pid_alive(pid):
            raise click.ClickException(f"{label} subprocess died during startup")
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                if resp.status < 500:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
            last_err = exc
            time.sleep(1.0)
    raise click.ClickException(
        f"{label} did not become healthy (last error: {last_err})"
    )


def parse_backend_spec(spec: str) -> tuple[str, int, int, int]:
    if ":" not in spec:
        engine, rest = spec, ""
    else:
        engine, rest = spec.split(":", 1)
    engine = engine.strip().lower()
    if engine not in ("sglang", "vllm"):
        raise click.BadParameter(
            f"backend must be one of: sglang, vllm; got {engine!r}"
        )
    tp = dp = pp = 1
    if rest:
        for pair in rest.split(","):
            pair = pair.strip()
            if not pair:
                continue
            if "=" not in pair:
                raise click.BadParameter(
                    f"backend spec part {pair!r}: expected key=value"
                )
            key, value = pair.split("=", 1)
            try:
                value_int = int(value)
            except ValueError as exc:
                raise click.BadParameter(
                    f"backend spec value for {key!r} must be int, got {value!r}"
                ) from exc
            if key == "tp":
                tp = value_int
            elif key == "dp":
                dp = value_int
            elif key == "pp":
                pp = value_int
            else:
                raise click.BadParameter(f"unknown backend key: {key!r}")
    return engine, tp, dp, pp


def resolve_provider_api_key(
    *, provider_api_key: str | None, provider_api_key_env: str | None
) -> str | None:
    if provider_api_key:
        return provider_api_key
    if provider_api_key_env:
        value = os.environ.get(provider_api_key_env)
        if not value:
            raise click.ClickException(
                f"env var {provider_api_key_env!r} is unset or empty"
            )
        return value
    return None


def split_args(value: str) -> list[str]:
    return shlex.split(value) if value else []


def terminate_runtime_state(
    state: RuntimeState, *, grace_s: float, force: bool = False
) -> None:
    phases = (
        state.model_state.all_engine_pids(),
        state.model_state.all_proxy_pids(),
        [state.gateway_pid],
        [state.router_pid],
    )
    for pids in phases:
        pids = [pid for pid in pids if pid > 0]
        if not pids:
            continue
        if force:
            for pid in pids:
                signal_pid(pid, signal.SIGKILL)
        else:
            kill_pids(pids, grace_s=grace_s)


def register_external(
    *,
    model: str,
    api_url: str,
    api_key: str | None,
    provider_model: str | None,
    gateway: GatewayClient,
) -> None:
    payload = {
        "model": model,
        "url": api_url,
        "api_key": api_key,
        "data_proxy_addrs": [],
    }
    if provider_model:
        payload["provider_model"] = provider_model
    try:
        gateway.register_model(payload)
    except (GatewayUnreachable, GatewayHTTPError) as exc:
        raise click.ClickException(f"register_model failed: {exc}") from exc


def register_internal(
    *,
    model: str,
    backend: str,
    model_path: str,
    tokenizer_path: str,
    engine_extra: list[str],
    proxy_extra: list[str],
    model_health_timeout: float,
    log_level: str,
    admin_api_key: str,
    gateway: GatewayClient,
    router: RouterClient,
    log_dir: Path,
    base_gpu_id: int = 0,
) -> tuple[list[int], list[int], list[int], list[str], list[str], int, int]:
    engine, tp, dp, pp = parse_backend_spec(backend)
    if pp > 1:
        raise click.ClickException(
            "pp > 1 is not supported by `areal inf` (single-node only)."
        )

    spawned: list[int] = []
    engine_pids: list[int] = []
    proxy_pids: list[int] = []
    proxy_addrs: list[str] = []
    inf_addrs: list[str] = []
    gpu_count = dp * tp

    try:
        for rank in range(dp):
            inf_port = pick_free_port()
            inf_log = log_dir / f"{model}-inf-{rank}.log"
            if engine == "sglang":
                pid = spawn_sglang(
                    model_path=model_path,
                    host="127.0.0.1",
                    port=inf_port,
                    tp=tp,
                    base_gpu_id=base_gpu_id + rank * tp,
                    extra_args=engine_extra,
                    log_file=inf_log,
                )
            else:
                pid = spawn_vllm(
                    model_path=model_path,
                    host="127.0.0.1",
                    port=inf_port,
                    tp=tp,
                    pp=pp,
                    extra_args=engine_extra,
                    log_file=inf_log,
                )
            spawned.append(pid)
            engine_pids.append(pid)
            inf_addr = f"http://127.0.0.1:{inf_port}"
            inf_addrs.append(inf_addr)
            logger.info(
                "spawned %s replica %d/%d pid=%d port=%d",
                engine,
                rank,
                dp,
                pid,
                inf_port,
            )
            wait_http_health(
                inf_addr,
                deadline=time.time() + model_health_timeout,
                pid=pid,
                label=f"{engine} replica {rank}",
            )

            proxy_port = pick_free_port()
            proxy_log = log_dir / f"{model}-data-proxy-{rank}.log"
            proxy_pid = spawn_data_proxy(
                host="127.0.0.1",
                port=proxy_port,
                backend_addr=inf_addr,
                backend_type=engine,
                tokenizer_path=tokenizer_path,
                admin_api_key=admin_api_key,
                log_level=log_level,
                extra_args=proxy_extra,
                log_file=proxy_log,
            )
            spawned.append(proxy_pid)
            proxy_pids.append(proxy_pid)
            proxy_addr = f"http://127.0.0.1:{proxy_port}"
            proxy_addrs.append(proxy_addr)
            logger.info(
                "spawned data-proxy %d/%d pid=%d port=%d",
                rank,
                dp,
                proxy_pid,
                proxy_port,
            )
            wait_http_health(
                proxy_addr,
                deadline=time.time() + 30.0,
                pid=proxy_pid,
                label=f"data-proxy {rank}",
            )

        for addr in proxy_addrs:
            try:
                router.register_worker(addr)
            except (GatewayUnreachable, GatewayHTTPError) as exc:
                raise click.ClickException(
                    f"router register_worker {addr} failed: {exc}"
                ) from exc

        try:
            gateway.register_model(
                {
                    "model": model,
                    "url": "",
                    "api_key": "",
                    "data_proxy_addrs": proxy_addrs,
                }
            )
        except (GatewayUnreachable, GatewayHTTPError) as exc:
            raise click.ClickException(f"gateway register_model failed: {exc}") from exc

    except BaseException:
        if spawned:
            logger.error(
                "internal register failed; killing %d spawned worker(s)",
                len(spawned),
            )
            kill_pids(spawned, grace_s=10.0)
        raise

    return (
        spawned,
        engine_pids,
        proxy_pids,
        proxy_addrs,
        inf_addrs,
        base_gpu_id,
        gpu_count,
    )


def print_models(state: RuntimeState | ModelState, as_json: bool) -> int:
    model_state = state.model_state if isinstance(state, RuntimeState) else state
    if as_json:
        out = [
            {
                "name": name,
                "default": name == model_state.default_model,
                **asdict(entry),
            }
            for name, entry in model_state.models.items()
        ]
        click.echo(json.dumps(out, indent=2))
        return 0

    if not model_state.models:
        click.echo("no models registered")
        return 0

    rows = []
    for name, entry in model_state.models.items():
        backend = entry.backend if entry.kind == "internal" else "-"
        workers = str(len(entry.pids) // 2) if entry.kind == "internal" else "-"
        default = "*" if name == model_state.default_model else ""
        rows.append((name, default, entry.kind, backend, workers))
    cols = ("NAME", "DEFAULT", "KIND", "BACKEND", "WORKERS")
    widths = [max(len(row[i]) for row in (cols, *rows)) for i in range(len(cols))]
    fmt = "  ".join(f"{{:<{width}}}" for width in widths)
    click.echo(fmt.format(*cols))
    for row in rows:
        click.echo(fmt.format(*row))
    return 0


def print_services(*, as_json: bool, include_all: bool) -> int:
    rows = []
    for service in list_service_names():
        try:
            state = load_runtime_state(service)
        except Exception:
            if include_all:
                rows.append({"service": service, "status": "stale"})
            continue
        running = gateway_alive(state)
        if running or include_all:
            rows.append(
                {
                    "service": service,
                    "status": "running" if running else "stale",
                    "gateway_url": state.gateway_url,
                    "gateway_pid": state.gateway_pid,
                    "router_url": state.router_url,
                    "models": len(state.models),
                }
            )

    if as_json:
        click.echo(json.dumps(rows, indent=2))
        return 0
    if not rows:
        click.echo("no inference services")
        return 0

    table = [
        (
            row["service"],
            row["status"],
            str(row.get("models", "")),
            row.get("gateway_url", ""),
            str(row.get("gateway_pid", "")),
        )
        for row in rows
    ]
    cols = ("SERVICE", "STATUS", "MODELS", "GATEWAY", "PID")
    widths = [max(len(str(row[i])) for row in (cols, *table)) for i in range(5)]
    fmt = "  ".join(f"{{:<{width}}}" for width in widths)
    click.echo(fmt.format(*cols))
    for row in table:
        click.echo(fmt.format(*row))
    return 0
