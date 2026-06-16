# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import shlex
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path

import click

from areal.experimental.cli.commands.inf.client import (
    GatewayClient,
    GatewayHTTPError,
    GatewayUnreachable,
    RouterClient,
)
from areal.utils.logging import getLogger

logger = getLogger("InfCli")
from areal.experimental.cli.commands.inf.launcher import (
    kill_pids,
    pick_free_port,
    signal_pid,
    spawn_data_proxy,
    spawn_gateway,
    spawn_router,
    spawn_sglang,
    spawn_vllm,
)
from areal.experimental.cli.commands.inf.state import (
    DaemonState,
    ModelEntry,
    gateway_alive,
    inf_root,
    logs_dir,
    state_path,
)
from areal.experimental.cli.state import pid_alive


@click.group(help="Manage the local AReaL inference service.")
def inf() -> None:
    pass


# =============================================================================
# Internal helpers
# =============================================================================


def _running_state() -> DaemonState | None:
    if not state_path().exists():
        return None
    try:
        s = DaemonState.load()
    except Exception:
        return None
    if not gateway_alive(s):
        return None
    return s


def _refuse_if_running() -> None:
    s = _running_state()
    if s is None:
        return
    raise click.ClickException(
        f"daemon already running (pid={s.gateway_pid}, url={s.gateway_url}). "
        "Run `areal inf stop` first."
    )


def _wait_health(client, *, timeout: float, label: str) -> None:
    deadline = time.time() + timeout
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            client.health(timeout=1.5)
            return
        except (GatewayUnreachable, GatewayHTTPError) as e:
            last_err = e
            time.sleep(0.3)
    raise click.ClickException(
        f"{label} did not become healthy within {timeout:.0f}s "
        f"(last error: {last_err})"
    )


def _parse_backend_spec(spec: str) -> tuple[str, int, int, int]:
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
            k, v = pair.split("=", 1)
            try:
                v_i = int(v)
            except ValueError as e:
                raise click.BadParameter(
                    f"backend spec value for {k!r} must be int, got {v!r}"
                ) from e
            if k == "tp":
                tp = v_i
            elif k == "dp":
                dp = v_i
            elif k == "pp":
                pp = v_i
            else:
                raise click.BadParameter(f"unknown backend key: {k!r}")
    return engine, tp, dp, pp


def _resolve_provider_api_key(
    *, provider_api_key: str | None, provider_api_key_env: str | None
) -> str | None:
    if provider_api_key:
        return provider_api_key
    if provider_api_key_env:
        v = os.environ.get(provider_api_key_env)
        if not v:
            raise click.ClickException(
                f"env var {provider_api_key_env!r} is unset or empty"
            )
        return v
    return None


def _split_args(s: str) -> list[str]:
    return shlex.split(s) if s else []


# =============================================================================
# Model registration helpers (shared between `inf run --model ...` and
# the standalone `inf register` verb)
# =============================================================================


def _register_external(
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
    except (GatewayUnreachable, GatewayHTTPError) as e:
        raise click.ClickException(f"register_model failed: {e}") from e


def _register_internal(
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
) -> tuple[list[int], list[str], list[str]]:
    engine, tp, dp, pp = _parse_backend_spec(backend)
    if pp > 1:
        raise click.ClickException(
            "pp > 1 is not supported by `areal inf` (single-node only)."
        )

    spawned: list[int] = []
    proxy_addrs: list[str] = []
    inf_addrs: list[str] = []

    try:
        for r in range(dp):
            inf_port = pick_free_port()
            inf_log = log_dir / f"{model}-inf-{r}.log"
            if engine == "sglang":
                pid = spawn_sglang(
                    model_path=model_path,
                    host="127.0.0.1",
                    port=inf_port,
                    tp=tp,
                    base_gpu_id=r * tp,
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
            inf_addr = f"http://127.0.0.1:{inf_port}"
            inf_addrs.append(inf_addr)
            logger.info(
                "spawned %s replica %d/%d pid=%d port=%d",
                engine, r, dp, pid, inf_port,
            )

            _wait_inf_health(inf_addr, deadline=time.time() + model_health_timeout,
                             pid=pid, label=f"{engine} replica {r}")

            proxy_port = pick_free_port()
            proxy_log = log_dir / f"{model}-data-proxy-{r}.log"
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
            proxy_addr = f"http://127.0.0.1:{proxy_port}"
            proxy_addrs.append(proxy_addr)
            logger.info(
                "spawned data-proxy %d/%d pid=%d port=%d",
                r, dp, proxy_pid, proxy_port,
            )

            _wait_inf_health(proxy_addr, deadline=time.time() + 30.0,
                             pid=proxy_pid, label=f"data-proxy {r}")

        # Each proxy must self-register with the router's worker pool
        for addr in proxy_addrs:
            try:
                router.register_worker(addr)
            except (GatewayUnreachable, GatewayHTTPError) as e:
                raise click.ClickException(
                    f"router register_worker {addr} failed: {e}"
                ) from e

        try:
            gateway.register_model({
                "model": model,
                "url": "",
                "api_key": "",
                "data_proxy_addrs": proxy_addrs,
            })
        except (GatewayUnreachable, GatewayHTTPError) as e:
            raise click.ClickException(
                f"gateway register_model failed: {e}"
            ) from e

    except BaseException:
        if spawned:
            logger.error(
                "internal register failed; killing %d spawned worker(s)",
                len(spawned),
            )
            kill_pids(spawned, grace_s=10.0)
        raise

    return spawned, proxy_addrs, inf_addrs


def _wait_inf_health(addr: str, *, deadline: float, pid: int, label: str) -> None:
    import urllib.error
    import urllib.request

    last_err: Exception | None = None
    url = f"{addr}/health"
    while time.time() < deadline:
        if not pid_alive(pid):
            raise click.ClickException(f"{label} subprocess died during startup")
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                if resp.status < 500:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            last_err = e
            time.sleep(1.0)
    raise click.ClickException(
        f"{label} did not become healthy (last error: {last_err})"
    )


# =============================================================================
# inf run
# =============================================================================


@inf.command(name="run", help="Start the inference daemon (gateway + router).")
@click.option("--port", type=int, default=8080, show_default=True,
              help="Gateway port.")
@click.option("--host", default="127.0.0.1", show_default=True,
              help="Gateway bind host.")
@click.option("--admin-api-key", default="admin-api-key", show_default=True)
@click.option("--routing-strategy",
              type=click.Choice(["round_robin", "least_busy"]),
              default="round_robin", show_default=True)
@click.option("--log-level",
              type=click.Choice(["debug", "info", "warning", "error"]),
              default="info", show_default=True)
@click.option("--launch-timeout", type=float, default=30.0, show_default=True,
              help="Seconds to wait for gateway /health.")
@click.option("-d", "--detach", is_flag=True,
              help="Fork the daemon and exit; default is foreground.")
# Inline registration
@click.option("--model", default=None,
              help="Register this model at startup.")
@click.option("--api-url", default=None,
              help="External provider URL (presence => external model).")
@click.option("--provider-api-key", default=None)
@click.option("--provider-api-key-env", default=None,
              help="Env var holding the provider API key.")
@click.option("--provider-model", default=None,
              help="Upstream model name (defaults to --model).")
@click.option("--backend", default=None,
              help="Internal backend spec, e.g. 'sglang:tp=2,dp=2'.")
@click.option("--model-path", default=None,
              help="HF / local model path (internal only).")
@click.option("--tokenizer-path", default=None,
              help="Tokenizer path (defaults to --model-path).")
@click.option("--engine-args", default="", show_default=False,
              help="Extra shell-style args passed to sglang/vllm.")
@click.option("--proxy-args", default="", show_default=False,
              help="Extra shell-style args passed to the data-proxy.")
@click.option("--model-health-timeout", type=float, default=600.0,
              show_default=True,
              help="Seconds to wait for the model server to come up.")
def _run_cmd(**opts) -> None:
    raise SystemExit(_do_run(opts) or 0)


def _do_run(opts: dict) -> int:
    _refuse_if_running()

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
            raise click.UsageError(
                "--backend requires --model-path <path>."
            )
    elif opts["api_url"] or opts["backend"]:
        raise click.UsageError(
            "model registration flags require --model."
        )

    log_dir = logs_dir()
    router_log = log_dir / "router.log"
    gateway_log = log_dir / "gateway.log"

    logger.info("starting inference daemon (logs: %s)", log_dir)

    router_pid, router_port = spawn_router(
        host="127.0.0.1",
        admin_api_key=opts["admin_api_key"],
        routing_strategy=opts["routing_strategy"],
        log_level=opts["log_level"],
        log_file=router_log,
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
        log_file=gateway_log,
    )
    host_for_url = "127.0.0.1" if opts["host"] in ("0.0.0.0", "::") else opts["host"]
    gateway_url = f"http://{host_for_url}:{opts['port']}"
    logger.info("gateway pid=%d %s", gateway_pid, gateway_url)

    gateway_client = GatewayClient(gateway_url, opts["admin_api_key"])
    router_client = RouterClient(router_url, opts["admin_api_key"])

    try:
        _wait_health(gateway_client, timeout=opts["launch_timeout"], label="gateway")
    except BaseException:
        kill_pids([gateway_pid, router_pid], grace_s=5.0)
        raise

    state = DaemonState(
        gateway_pid=gateway_pid,
        gateway_url=gateway_url,
        router_pid=router_pid,
        router_url=router_url,
        admin_api_key=opts["admin_api_key"],
        started_at=time.time(),
    )
    state.save()

    if opts["model"]:
        try:
            if opts["api_url"]:
                api_key = _resolve_provider_api_key(
                    provider_api_key=opts["provider_api_key"],
                    provider_api_key_env=opts["provider_api_key_env"],
                )
                _register_external(
                    model=opts["model"],
                    api_url=opts["api_url"],
                    api_key=api_key,
                    provider_model=opts["provider_model"],
                    gateway=gateway_client,
                )
                state.models[opts["model"]] = ModelEntry(
                    kind="external",
                    api_url=opts["api_url"],
                )
                state.save()
            else:
                pids, proxy_addrs, inf_addrs = _register_internal(
                    model=opts["model"],
                    backend=opts["backend"],
                    model_path=opts["model_path"],
                    tokenizer_path=opts["tokenizer_path"] or opts["model_path"],
                    engine_extra=_split_args(opts["engine_args"]),
                    proxy_extra=_split_args(opts["proxy_args"]),
                    model_health_timeout=opts["model_health_timeout"],
                    log_level=opts["log_level"],
                    admin_api_key=opts["admin_api_key"],
                    gateway=gateway_client,
                    router=router_client,
                    log_dir=log_dir,
                )
                state.models[opts["model"]] = ModelEntry(
                    kind="internal",
                    backend=opts["backend"],
                    pids=pids,
                    proxy_addrs=proxy_addrs,
                    inference_server_addrs=inf_addrs,
                )
                state.save()
        except BaseException:
            kill_pids(
                [gateway_pid, router_pid, *state.all_worker_pids()],
                grace_s=5.0,
            )
            DaemonState.remove()
            raise

    logger.info("daemon ready pid=%d url=%s", gateway_pid, gateway_url)
    if opts["model"]:
        kind = "external" if opts["api_url"] else f"internal ({opts['backend']})"
        logger.info("default model: %s (%s)", opts["model"], kind)

    if opts["detach"]:
        return 0

    logger.info("foreground (Ctrl-C to stop) ...")
    try:
        while gateway_alive(state):
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("shutting down ...")
        kill_pids(
            [gateway_pid, router_pid, *state.all_worker_pids()], grace_s=10.0
        )
        DaemonState.remove()
        return 0

    logger.warning("gateway exited")
    DaemonState.remove()
    return 0


# =============================================================================
# inf ps
# =============================================================================


@inf.command(name="ps", help="List registered models.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON.")
def _ps_cmd(as_json: bool) -> None:
    raise SystemExit(_do_ps(as_json) or 0)


def _do_ps(as_json: bool) -> int:
    s = _running_state()
    if s is None:
        if as_json:
            click.echo("[]")
        else:
            click.echo("daemon not running")
        return 0

    if as_json:
        out = [
            {"name": name, **asdict(entry)}
            for name, entry in s.models.items()
        ]
        click.echo(json.dumps(out, indent=2))
        return 0

    if not s.models:
        click.echo("no models registered")
        return 0

    rows = []
    for name, entry in s.models.items():
        backend = entry.backend if entry.kind == "internal" else "-"
        workers = str(len(entry.pids) // 2) if entry.kind == "internal" else "-"
        rows.append((name, entry.kind, backend, workers))
    cols = ("NAME", "KIND", "BACKEND", "WORKERS")
    widths = [max(len(r[i]) for r in (cols, *rows)) for i in range(len(cols))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    click.echo(fmt.format(*cols))
    for r in rows:
        click.echo(fmt.format(*r))
    return 0


# =============================================================================
# inf status
# =============================================================================


@inf.command(name="status", help="Show daemon status.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON.")
def _status_cmd(as_json: bool) -> None:
    raise SystemExit(_do_status(as_json) or 0)


def _do_status(as_json: bool) -> int:
    if not state_path().exists():
        snap = {"running": False}
        if as_json:
            click.echo(json.dumps(snap, indent=2))
        else:
            click.echo("daemon not running")
        return 0

    try:
        s = DaemonState.load()
    except Exception as e:
        raise click.ClickException(f"failed to load state: {e}") from e

    pid_ok = gateway_alive(s)
    gateway_http = "down"
    if pid_ok:
        try:
            GatewayClient(s.gateway_url, s.admin_api_key).health(timeout=2.0)
            gateway_http = "ok"
        except GatewayUnreachable:
            gateway_http = "unreachable"
        except GatewayHTTPError:
            gateway_http = "error"

    rows: list[tuple[str, str, str, str]] = []
    rows.append((
        "gateway", gateway_http, s.gateway_url, f"models={len(s.models)}",
    ))
    rows.append((
        "router", "ok" if pid_alive(s.router_pid) else "down",
        s.router_url, "",
    ))
    for name, entry in s.models.items():
        if entry.kind == "internal":
            detail = (
                f"backend={entry.backend} workers={len(entry.pids) // 2}"
            )
            addr = "internal"
        else:
            detail = f"api_url={entry.api_url}"
            addr = "external"
        rows.append((name, "registered", addr, detail))

    if as_json:
        snap = {
            "running": pid_ok,
            "gateway_url": s.gateway_url,
            "gateway_pid": s.gateway_pid,
            "gateway_http": gateway_http,
            "router_url": s.router_url,
            "started_at": s.started_at,
            "models": {n: asdict(e) for n, e in s.models.items()},
        }
        click.echo(json.dumps(snap, indent=2))
        return 0

    cols = ("COMPONENT", "STATUS", "ADDR", "DETAILS")
    widths = [max(len(r[i]) for r in (cols, *rows)) for i in range(len(cols))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    click.echo(fmt.format(*cols))
    for r in rows:
        click.echo(fmt.format(*r))
    return 0


# =============================================================================
# inf stop
# =============================================================================


@inf.command(name="stop", help="Stop the inference daemon.")
@click.option("--grace", type=float, default=10.0, show_default=True,
              help="Seconds to wait before SIGKILL.")
@click.option("--force", is_flag=True, help="SIGKILL immediately.")
def _stop_cmd(grace: float, force: bool) -> None:
    raise SystemExit(_do_stop(grace, force) or 0)


def _do_stop(grace: float, force: bool) -> int:
    if not state_path().exists():
        click.echo("daemon not running")
        return 0

    try:
        s = DaemonState.load()
    except Exception:
        logger.warning("stale state; removing")
        DaemonState.remove()
        return 0

    pids = [
        p for p in (s.gateway_pid, s.router_pid, *s.all_worker_pids()) if p > 0
    ]

    if force:
        for p in pids:
            signal_pid(p, signal.SIGKILL)
    else:
        kill_pids(pids, grace_s=grace)

    DaemonState.remove()
    click.echo("daemon stopped")
    return 0


# =============================================================================
# inf register
# =============================================================================


@inf.command(name="register", help="Register a model against the running daemon.")
@click.argument("name")
@click.option("--api-url", default=None,
              help="External provider URL (presence => external model).")
@click.option("--provider-api-key", default=None)
@click.option("--provider-api-key-env", default=None)
@click.option("--provider-model", default=None)
@click.option("--backend", default=None,
              help="Internal backend spec, e.g. 'sglang:tp=2,dp=2'.")
@click.option("--model-path", default=None)
@click.option("--tokenizer-path", default=None)
@click.option("--engine-args", default="")
@click.option("--proxy-args", default="")
@click.option("--model-health-timeout", type=float, default=600.0,
              show_default=True)
@click.option("--log-level",
              type=click.Choice(["debug", "info", "warning", "error"]),
              default="info", show_default=True)
def _register_cmd(name: str, **opts) -> None:
    raise SystemExit(_do_register(name, opts) or 0)


def _do_register(name: str, opts: dict) -> int:
    s = _load_running_state()

    if opts["api_url"] and opts["backend"]:
        raise click.UsageError("Use --api-url OR --backend, not both.")
    if not opts["api_url"] and not opts["backend"]:
        raise click.UsageError(
            "Provide --api-url <url> (external) or --backend <spec> (internal)."
        )
    if opts["backend"] and not opts["model_path"]:
        raise click.UsageError("--backend requires --model-path.")

    if name in s.models:
        raise click.ClickException(f"model {name!r} already registered")

    gateway = GatewayClient(s.gateway_url, s.admin_api_key)
    router = RouterClient(s.router_url, s.admin_api_key)

    if opts["api_url"]:
        api_key = _resolve_provider_api_key(
            provider_api_key=opts["provider_api_key"],
            provider_api_key_env=opts["provider_api_key_env"],
        )
        _register_external(
            model=name,
            api_url=opts["api_url"],
            api_key=api_key,
            provider_model=opts["provider_model"],
            gateway=gateway,
        )
        s.models[name] = ModelEntry(
            kind="external",
            api_url=opts["api_url"],
        )
        s.save()
        logger.info("registered external model %r", name)
        return 0

    pids, proxy_addrs, inf_addrs = _register_internal(
        model=name,
        backend=opts["backend"],
        model_path=opts["model_path"],
        tokenizer_path=opts["tokenizer_path"] or opts["model_path"],
        engine_extra=_split_args(opts["engine_args"]),
        proxy_extra=_split_args(opts["proxy_args"]),
        model_health_timeout=opts["model_health_timeout"],
        log_level=opts["log_level"],
        admin_api_key=s.admin_api_key,
        gateway=gateway,
        router=router,
        log_dir=logs_dir(),
    )
    s.models[name] = ModelEntry(
        kind="internal",
        backend=opts["backend"],
        pids=pids,
        proxy_addrs=proxy_addrs,
        inference_server_addrs=inf_addrs,
    )
    s.save()
    logger.info(
        "registered internal model %r (%d worker(s))", name, len(pids)
    )
    return 0


# =============================================================================
# inf deregister
# =============================================================================


@inf.command(name="deregister", help="Deregister a model and tear down its workers.")
@click.argument("name")
@click.option("--grace", type=float, default=10.0, show_default=True,
              help="Seconds to wait before SIGKILL on the model's workers.")
@click.option("--force", is_flag=True, help="SIGKILL workers immediately.")
def _deregister_cmd(name: str, grace: float, force: bool) -> None:
    raise SystemExit(_do_deregister(name, grace, force) or 0)


def _do_deregister(name: str, grace: float, force: bool) -> int:
    s = _load_running_state()
    if name not in s.models:
        raise click.ClickException(f"model {name!r} is not registered")
    entry = s.models[name]

    router = RouterClient(s.router_url, s.admin_api_key)

    try:
        router.remove_model(name)
    except GatewayHTTPError as e:
        if e.status != 404:
            logger.warning("router remove_model %s returned %d", name, e.status)
    except GatewayUnreachable as e:
        logger.warning("router unreachable while removing %s: %s", name, e)

    for addr in entry.proxy_addrs:
        try:
            router.unregister_worker(addr)
        except (GatewayHTTPError, GatewayUnreachable) as e:
            logger.warning("router unregister %s failed: %s", addr, e)

    if entry.pids:
        if force:
            for p in entry.pids:
                signal_pid(p, signal.SIGKILL)
        else:
            kill_pids(entry.pids, grace_s=grace)

    del s.models[name]
    s.save()
    logger.info("deregistered model %r", name)
    return 0


def _load_running_state() -> DaemonState:
    if not state_path().exists():
        raise click.ClickException("daemon not running")
    try:
        s = DaemonState.load()
    except Exception as e:
        raise click.ClickException(f"failed to load state: {e}") from e
    if not gateway_alive(s):
        raise click.ClickException("daemon pid not alive")
    return s


# =============================================================================
# inf reward
# =============================================================================


@inf.command(name="reward", help="Set reward on a session (closes the active trajectory).")
@click.argument("session_api_key")
@click.argument("reward_value", type=float)
@click.option("--model", default=None, help="Model name (optional, used for routing).")
def _reward_cmd(session_api_key: str, reward_value: float, model: str | None) -> None:
    raise SystemExit(_do_reward(session_api_key, reward_value, model) or 0)


def _do_reward(skey: str, reward: float, model: str | None) -> int:
    s = _load_running_state()
    gateway = GatewayClient(s.gateway_url, s.admin_api_key)
    try:
        gateway.set_reward(session_api_key=skey, reward=reward, model=model)
    except (GatewayUnreachable, GatewayHTTPError) as e:
        raise click.ClickException(f"set_reward failed: {e}") from e
    return 0


# =============================================================================
# inf collect — client-side batch orchestrator
# =============================================================================


@inf.command(
    name="collect",
    help="Start N sessions, wait for trajectories to be ready, export and dump them.",
)
@click.argument("model")
@click.option("--batch-size", type=int, required=True, help="Number of sessions / trajectories to collect.")
@click.option("--task-id", default="cli-collect", help="Logical task identifier passed to start_session.")
@click.option("--sessions-out", "sessions_out", type=click.Path(path_type=Path), default=None,
              help="Write the (session_id, session_api_key) list here so an external agent can pick it up.")
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None,
              help="Write collected trajectories as JSONL here.  Defaults to stdout.")
@click.option("--timeout", type=float, default=1800.0, show_default=True,
              help="Max seconds to wait for the batch.")
@click.option("--poll-interval", type=float, default=2.0, show_default=True,
              help="Seconds between /export_trajectories polls.")
@click.option("--discount", type=float, default=1.0, show_default=True,
              help="Reward discount passed to export_trajectories.")
@click.option("--style",
              type=click.Choice(["individual", "concat"]),
              default="individual", show_default=True)
def _collect_cmd(
    model: str, batch_size: int, task_id: str,
    sessions_out: Path | None, output: Path | None,
    timeout: float, poll_interval: float,
    discount: float, style: str,
) -> None:
    raise SystemExit(_do_collect(
        model=model, batch_size=batch_size, task_id=task_id,
        sessions_out=sessions_out, output=output,
        timeout=timeout, poll_interval=poll_interval,
        discount=discount, style=style,
    ) or 0)


def _do_collect(
    *, model: str, batch_size: int, task_id: str,
    sessions_out: Path | None, output: Path | None,
    timeout: float, poll_interval: float,
    discount: float, style: str,
) -> int:
    s = _load_running_state()
    gateway = GatewayClient(s.gateway_url, s.admin_api_key)

    logger.info("starting %d session(s) for model %r ...", batch_size, model)
    try:
        resp = gateway.start_session(model=model, task_id=task_id, group_size=batch_size)
    except (GatewayUnreachable, GatewayHTTPError) as e:
        raise click.ClickException(f"start_session failed: {e}") from e

    group_id = resp.get("group_id", "")
    sessions = resp.get("sessions") or []
    sids = [sess["session_id"] for sess in sessions]
    if len(sids) != batch_size:
        logger.warning(
            "server returned %d sessions but %d were requested",
            len(sids), batch_size,
        )

    if sessions_out:
        sessions_out.parent.mkdir(parents=True, exist_ok=True)
        with open(sessions_out, "w") as f:
            json.dump({"group_id": group_id, "sessions": sessions}, f, indent=2)
        logger.info("wrote %d session(s) to %s", len(sessions), sessions_out)
    else:
        click.echo(json.dumps({"group_id": group_id, "sessions": sessions}, indent=2))

    logger.info(
        "polling /export_trajectories every %.1fs (timeout=%.0fs) ...",
        poll_interval, timeout,
    )
    collected: dict[str, dict] = {}
    deadline = time.time() + timeout
    while len(collected) < batch_size and time.time() < deadline:
        try:
            r = gateway.export_trajectories(
                session_ids=sids, group_id=group_id,
                remove_session=False, discount=discount, style=style,
            )
        except GatewayHTTPError as e:
            raise click.ClickException(f"export_trajectories failed: {e}") from e
        except GatewayUnreachable as e:
            logger.warning("gateway unreachable mid-poll: %s", e)
            time.sleep(poll_interval)
            continue
        traj = r.get("traj") or {}
        for tid, interaction in traj.items():
            if tid not in collected:
                collected[tid] = interaction
        if len(collected) < batch_size:
            logger.info("collected %d/%d, waiting ...", len(collected), batch_size)
            time.sleep(poll_interval)

    try:
        r = gateway.export_trajectories(
            session_ids=sids, group_id=group_id,
            remove_session=True, discount=discount, style=style,
        )
        for tid, interaction in (r.get("traj") or {}).items():
            if tid not in collected:
                collected[tid] = interaction
    except (GatewayHTTPError, GatewayUnreachable) as e:
        logger.warning("final cleanup export failed: %s", e)

    if len(collected) < batch_size:
        logger.warning(
            "collected %d/%d trajectories before timeout (%.0fs)",
            len(collected), batch_size, timeout,
        )
    else:
        logger.info("collected %d trajectories", len(collected))

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            for tid, interaction in collected.items():
                f.write(json.dumps({"trajectory_id": tid, **interaction}) + "\n")
        click.echo(f"wrote {len(collected)} trajectories to {output}")
    else:
        for tid, interaction in collected.items():
            click.echo(json.dumps({"trajectory_id": tid, **interaction}))

    return 0 if len(collected) >= batch_size else 1
