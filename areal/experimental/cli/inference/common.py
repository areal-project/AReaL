# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
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
    build_data_proxy_task_spec,
    build_sglang_task_spec,
    build_vllm_task_spec,
)
from areal.experimental.cli.inference.scheduler import (
    Scheduler,
    TaskHandle,
    build_scheduler,
)
from areal.experimental.cli.inference.state import (
    DEFAULT_SERVICE,
    ModelEntry,
    ModelReplica,
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


def split_args(value: str) -> list[str]:
    return shlex.split(value) if value else []


def _pids(handles: list[TaskHandle]) -> list[int]:
    return [h.pid for h in handles if h.pid > 0]


def terminate_runtime_state(
    state: RuntimeState, *, grace_s: float, force: bool = False
) -> None:
    # Data-flow order: kill upstream (data-proxies stop accepting requests)
    # before the worker behind it goes away, so in-flight requests fail at
    # the proxy boundary instead of mid-worker. Control plane last.
    phases = (
        _pids(state.model_state.all_data_proxies()),
        _pids(state.model_state.all_workers()),
        _pids([state.gateway_handle]),
        _pids([state.router_handle]),
    )
    for pids in phases:
        if not pids:
            continue
        if force:
            for pid in pids:
                signal_pid(pid, signal.SIGKILL)
        else:
            kill_pids(pids, grace_s=grace_s)


def format_placement(backend: str, handle: TaskHandle) -> str:
    if backend == "local":
        import socket

        return socket.gethostname()
    if backend == "k8s":
        node = handle.ref.get("node", "?")
        pod = handle.ref.get("pod_name", "?")
        return f"{node}/{pod}"
    if backend == "slurm":
        node = handle.ref.get("node", "?")
        job = handle.ref.get("job_id", "?")
        return f"{node}/job={job}"
    return "-"


def format_ref(backend: str, handle: TaskHandle) -> str:
    if backend == "local":
        return f"pid={handle.pid}" if handle.pid > 0 else "-"
    if backend == "k8s":
        pod = handle.ref.get("pod_name", "")
        return f"pod={pod}" if pod else "-"
    if backend == "slurm":
        job = handle.ref.get("job_id", "")
        return f"job={job}" if job else "-"
    return "-"


def format_gpu_count(handle: TaskHandle) -> str:
    n = len(handle.gpu_devices)
    return f"×{n}" if n > 0 else "-"


def probe_http_health(addr: str, *, timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(f"{addr}/health", timeout=timeout) as resp:
            return resp.status < 500
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
        return False


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
    scheduler: Scheduler,
) -> list[ModelReplica]:
    engine, tp, dp, pp = parse_backend_spec(backend)
    if pp > 1:
        raise click.ClickException(
            "pp > 1 is not supported by `areal inf` (single-node only)."
        )

    replicas: list[ModelReplica] = []
    spawned_handles: list[TaskHandle] = []

    try:
        for rank in range(dp):
            worker_log = log_dir / f"{model}-worker-{rank}.log"
            if engine == "sglang":
                worker_spec = build_sglang_task_spec(
                    name=f"worker/{model}/{rank}",
                    model_path=model_path,
                    tp=tp,
                    extra_args=engine_extra,
                    log_file=worker_log,
                )
            else:
                worker_spec = build_vllm_task_spec(
                    name=f"worker/{model}/{rank}",
                    model_path=model_path,
                    tp=tp,
                    pp=pp,
                    extra_args=engine_extra,
                    log_file=worker_log,
                )
            worker_handle = scheduler.submit(worker_spec)
            spawned_handles.append(worker_handle)
            logger.info(
                "spawned %s worker %d/%d pid=%d port=%d gpus=%s",
                engine,
                rank,
                dp,
                worker_handle.pid,
                worker_handle.ports[0],
                worker_handle.gpu_devices,
            )
            wait_http_health(
                worker_handle.addr,
                deadline=time.time() + model_health_timeout,
                pid=worker_handle.pid,
                label=f"{engine} worker {rank}",
            )

            proxy_log = log_dir / f"{model}-data-proxy-{rank}.log"
            proxy_spec = build_data_proxy_task_spec(
                name=f"data_proxy/{model}/{rank}",
                backend_addr=worker_handle.addr,
                backend_type=engine,
                tokenizer_path=tokenizer_path,
                admin_api_key=admin_api_key,
                log_level=log_level,
                extra_args=proxy_extra,
                log_file=proxy_log,
            )
            proxy_handle = scheduler.submit(proxy_spec)
            spawned_handles.append(proxy_handle)
            logger.info(
                "spawned data-proxy %d/%d pid=%d port=%d",
                rank,
                dp,
                proxy_handle.pid,
                proxy_handle.ports[0],
            )
            wait_http_health(
                proxy_handle.addr,
                deadline=time.time() + 30.0,
                pid=proxy_handle.pid,
                label=f"data-proxy {rank}",
            )

            replicas.append(ModelReplica(data_proxy=proxy_handle, worker=worker_handle))

        proxy_addrs = [r.data_proxy.addr for r in replicas]
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
        if spawned_handles:
            logger.error(
                "internal register failed; killing %d spawned worker(s)",
                len(spawned_handles),
            )
            pids = [h.pid for h in spawned_handles if h.pid > 0]
            kill_pids(pids, grace_s=10.0)
        raise

    return replicas


def validate_register_opts(opts: dict) -> None:
    if not opts.get("backend"):
        raise click.UsageError("--backend <spec> is required.")
    if not opts.get("model_path"):
        raise click.UsageError("--backend requires --model-path <path>.")


def register_model(
    *,
    model_name: str,
    opts: dict,
    gateway: GatewayClient,
    router: RouterClient,
    log_dir: Path,
    admin_api_key: str,
    scheduler_backend: str,
    occupied_gpus: set[int],
) -> ModelEntry:
    """Validate registration opts and spawn the model's worker + data-proxy
    fleet. The returned ModelEntry is meant to be slotted into model_state
    by the caller — this helper does not touch model_state itself so the
    caller can keep the file lock scope explicit."""

    validate_register_opts(opts)
    scheduler = build_scheduler(scheduler_backend, occupied_gpus=occupied_gpus)
    replicas = register_internal(
        model=model_name,
        backend=opts["backend"],
        model_path=opts["model_path"],
        tokenizer_path=opts.get("tokenizer_path") or opts["model_path"],
        engine_extra=split_args(opts.get("engine_args", "")),
        proxy_extra=split_args(opts.get("proxy_args", "")),
        model_health_timeout=opts.get("model_health_timeout", 600.0),
        log_level=opts.get("log_level", "info"),
        admin_api_key=admin_api_key,
        gateway=gateway,
        router=router,
        log_dir=log_dir,
        scheduler=scheduler,
    )
    return ModelEntry(backend=opts["backend"], replicas=replicas)


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
        default = "*" if name == model_state.default_model else ""
        rows.append((name, default, entry.backend, str(len(entry.replicas))))
    cols = ("NAME", "DEFAULT", "BACKEND", "WORKERS")
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
                    "backend": state.backend,
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

    def _pid_cell(row: dict) -> str:
        pid = row.get("gateway_pid", 0)
        return str(pid) if pid else "-"

    table = [
        (
            row["service"],
            row["status"],
            row.get("backend", "-"),
            str(row.get("models", "")),
            row.get("gateway_url", ""),
            _pid_cell(row),
        )
        for row in rows
    ]
    cols = ("SERVICE", "STATUS", "BACKEND", "MODELS", "GATEWAY", "PID")
    widths = [max(len(str(row[i])) for row in (cols, *table)) for i in range(len(cols))]
    fmt = "  ".join(f"{{:<{width}}}" for width in widths)
    click.echo(fmt.format(*cols))
    for row in table:
        click.echo(fmt.format(*row))
    return 0
