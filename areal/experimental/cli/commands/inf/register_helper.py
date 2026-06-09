# SPDX-License-Identifier: Apache-2.0

"""Internal-model registration: spawn sglang/vllm + data-proxy locally, then
``POST /register_model`` on the gateway.

Both ``inf run --model X --backend sglang:tp=2`` (inline) and the future
standalone ``inf register`` verb call into ``register_internal_model`` here.

This is **not** the v2 controller's RPCGuard / scheduler-based path.  The
``inf`` CLI is a single-node operator console: it spawns sglang and the
data-proxy as detached subprocesses on whatever node the CLI happens to
run on.  Multi-node serving is out of scope for the CLI for now; users
who need that should script it directly with the scheduler.
"""

from __future__ import annotations

import socket
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from areal.experimental.cli.commands.inf.launcher import _spawn_detached, kill_pids
from areal.experimental.cli.state import pid_alive
from areal.utils.logging import getLogger

logger = getLogger("InfCli")


# ---- backend spec parsing ------------------------------------------------


@dataclass
class BackendSpec:
    engine: str  # "sglang" | "vllm"
    tp: int = 1
    pp: int = 1
    dp: int = 1


def parse_backend_spec(spec: str) -> BackendSpec:
    """Parse ``"sglang"`` or ``"sglang:tp=2,pp=1,dp=2"`` into ``BackendSpec``.

    Raises ``SystemExit`` with an actionable error on malformed input.
    """
    if not spec:
        raise SystemExit("--backend is required for internal model registration.")

    if ":" not in spec:
        engine = spec.strip()
        return BackendSpec(engine=_validate_engine(engine))

    engine_part, args_part = spec.split(":", 1)
    engine = _validate_engine(engine_part.strip())
    out = BackendSpec(engine=engine)
    for tok in args_part.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "=" not in tok:
            raise SystemExit(
                f"Bad --backend token {tok!r}. Expected key=value (e.g. tp=2)."
            )
        k, v = tok.split("=", 1)
        k = k.strip().lower()
        try:
            iv = int(v.strip())
        except ValueError:
            raise SystemExit(f"Bad --backend value for {k}={v!r}: not an int.")
        if iv < 1:
            raise SystemExit(f"--backend {k} must be >= 1, got {iv}.")
        if k == "tp":
            out.tp = iv
        elif k == "pp":
            out.pp = iv
        elif k == "dp":
            out.dp = iv
        else:
            raise SystemExit(
                f"Unknown --backend key {k!r}. Supported: tp, pp, dp."
            )
    return out


def _validate_engine(name: str) -> str:
    if name not in ("sglang", "vllm"):
        raise SystemExit(
            f"Unsupported --backend engine {name!r}. Supported: sglang, vllm."
        )
    return name


# ---- port allocation -----------------------------------------------------


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# ---- inference server spawn ----------------------------------------------


def _build_sglang_cmd(
    *, model_path: str, host: str, port: int, tp: int,
    extra_args: list[str],
) -> list[str]:
    from areal.api.cli_args import SGLangConfig

    cfg = SGLangConfig(model_path=model_path)
    cmd = list(
        SGLangConfig.build_cmd(
            sglang_config=cfg,
            tp_size=tp,
            base_gpu_id=0,
            host=host,
            port=port,
            n_nodes=1,
            node_rank=0,
            pp_size=1,
        )
    )
    cmd.extend(extra_args)
    return cmd


def _build_vllm_cmd(
    *, model_path: str, host: str, port: int, tp: int, pp: int,
    extra_args: list[str],
) -> list[str]:
    from areal.api.cli_args import vLLMConfig

    cfg = vLLMConfig(model=model_path)
    cmd = list(
        vLLMConfig.build_cmd(
            vllm_config=cfg,
            tp_size=tp,
            pp_size=pp,
            host=host,
            port=port,
        )
    )
    cmd.extend(extra_args)
    return cmd


def _build_data_proxy_cmd(
    *,
    host: str,
    port: int,
    backend_addr: str,
    backend_type: str,
    tokenizer_path: str,
    admin_api_key: str,
    log_level: str,
    extra_args: list[str],
) -> list[str]:
    cmd = [
        sys.executable, "-m", "areal.experimental.inference_service.data_proxy",
        "--host", host,
        "--port", str(port),
        "--backend-addr", backend_addr,
        "--backend-type", backend_type,
        "--tokenizer-path", tokenizer_path,
        "--admin-api-key", admin_api_key,
        "--log-level", log_level,
    ]
    cmd.extend(extra_args)
    return cmd


# ---- health polling ------------------------------------------------------


def _wait_health(
    url: str, deadline: float, supervisor_pids: list[int], what: str
) -> None:
    import urllib.error
    import urllib.request

    last_err: Exception | None = None
    while time.time() < deadline:
        if not all(pid_alive(p) for p in supervisor_pids):
            raise SystemExit(
                f"{what} subprocess died before becoming healthy."
            )
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                if resp.status < 500:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            last_err = e
            time.sleep(1.0)
    raise SystemExit(
        f"{what} did not become healthy within timeout. Last error: {last_err}"
    )


# ---- public entry --------------------------------------------------------


@dataclass
class InternalRegisterArgs:
    model_name: str
    backend_spec: str  # raw "sglang:tp=2,dp=1"
    model_path: str
    tokenizer_path: str = ""
    log_dir: Path = Path()
    admin_api_key: str = "areal-admin-key"
    log_level: str = "info"
    health_timeout: float = 600.0  # sglang load can be slow
    # Passthrough flag strings, shlex.split into argv tokens.  See
    # `areal inf run --help` for guidance; the engine flag set is whatever
    # sglang / vllm accept, the proxy flag set is whatever
    # `python -m areal.experimental.inference_service.data_proxy --help`
    # accepts.
    engine_extra_args: list[str] = field(default_factory=list)
    proxy_extra_args: list[str] = field(default_factory=list)


@dataclass
class InternalRegisterResult:
    backend_spec: BackendSpec
    inference_server_addrs: list[str]
    data_proxy_addrs: list[str]
    worker_pids: list[int]


def register_internal_model(
    args: InternalRegisterArgs, *, gateway_url: str, router_url: str
) -> InternalRegisterResult:
    """Spawn N sglang/vllm + N data-proxy + register with gateway.

    Each data-proxy is also POSTed to the router's ``/register`` endpoint so
    the worker pool is non-empty when chat traffic arrives — otherwise the
    router answers with ``503 No registered workers``.

    On any failure, kills everything spawned so far and re-raises.
    """
    from areal.experimental.cli.commands.inf.gateway_client import (
        GatewayClient,
        GatewayHTTPError,
        GatewayUnreachable,
        RouterClient,
    )

    spec = parse_backend_spec(args.backend_spec)
    if spec.pp > 1:
        # Multi-rank pipeline-parallel needs distributed bootstrap; v1 of
        # this CLI keeps to single-node, single-rank-per-instance.
        raise SystemExit(
            "pp > 1 is not yet supported by `areal inf` (single-node only). "
            "Use the v2 trainer's controller for distributed pipeline parallel."
        )

    tokenizer_path = args.tokenizer_path or args.model_path

    spawned: list[int] = []
    inf_addrs: list[str] = []
    proxy_addrs: list[str] = []

    try:
        for replica in range(spec.dp):
            inf_port = _free_port()
            inf_log = args.log_dir / f"{args.model_name}-inf-{replica}.log"
            if spec.engine == "sglang":
                cmd = _build_sglang_cmd(
                    model_path=args.model_path,
                    host="127.0.0.1",
                    port=inf_port,
                    tp=spec.tp,
                    extra_args=args.engine_extra_args,
                )
            else:
                cmd = _build_vllm_cmd(
                    model_path=args.model_path,
                    host="127.0.0.1",
                    port=inf_port,
                    tp=spec.tp,
                    pp=spec.pp,
                    extra_args=args.engine_extra_args,
                )
            logger.info(
                "Spawning %s server (replica %d/%d, tp=%d, port=%d) ...",
                spec.engine, replica, spec.dp, spec.tp, inf_port,
            )
            inf_pid = _spawn_detached(cmd, inf_log)
            spawned.append(inf_pid)
            inf_addr = f"http://127.0.0.1:{inf_port}"
            inf_addrs.append(inf_addr)
            logger.info("  -> pid=%d, log=%s", inf_pid, inf_log)

            # Wait for this server to come up before spawning its data-proxy.
            _wait_health(
                f"{inf_addr}/health",
                deadline=time.time() + args.health_timeout,
                supervisor_pids=[inf_pid],
                what=f"{spec.engine} server {replica}",
            )

            proxy_port = _free_port()
            proxy_log = args.log_dir / f"{args.model_name}-data-proxy-{replica}.log"
            proxy_cmd = _build_data_proxy_cmd(
                host="127.0.0.1",
                port=proxy_port,
                backend_addr=inf_addr,
                backend_type=spec.engine,
                tokenizer_path=tokenizer_path,
                admin_api_key=args.admin_api_key,
                log_level=args.log_level,
                extra_args=args.proxy_extra_args,
            )
            logger.info(
                "Spawning data-proxy (replica %d/%d, port=%d) ...",
                replica, spec.dp, proxy_port,
            )
            proxy_pid = _spawn_detached(proxy_cmd, proxy_log)
            spawned.append(proxy_pid)
            proxy_addr = f"http://127.0.0.1:{proxy_port}"
            proxy_addrs.append(proxy_addr)
            logger.info("  -> pid=%d, log=%s", proxy_pid, proxy_log)

            _wait_health(
                f"{proxy_addr}/health",
                deadline=time.time() + 30.0,
                supervisor_pids=[proxy_pid],
                what=f"data-proxy {replica}",
            )

        # Each data-proxy must self-register into the router's worker pool
        # (the router doesn't auto-discover them and `/register_model` only
        # touches the model registry).  Without this, /v1/chat/completions
        # returns 503 "No registered workers".
        router_client = RouterClient(
            router_url, admin_api_key=args.admin_api_key, timeout=10.0
        )
        for addr in proxy_addrs:
            try:
                router_client.register_worker(addr)
                logger.info("Registered worker %s with router", addr)
            except (GatewayUnreachable, GatewayHTTPError) as e:
                raise SystemExit(
                    f"Failed to register data-proxy {addr} with router at "
                    f"{router_url}: {e}"
                ) from e

        # Register with gateway (model -> data_proxy_addrs mapping)
        client = GatewayClient(gateway_url, admin_api_key=args.admin_api_key, timeout=15.0)
        payload = {
            "model": args.model_name,
            "url": "",  # internal — gateway routes via data_proxy_addrs
            "api_key": "",
            "data_proxy_addrs": proxy_addrs,
        }
        try:
            client.register_model(payload)
        except (GatewayUnreachable, GatewayHTTPError) as e:
            raise SystemExit(
                f"Gateway register_model failed for internal model "
                f"{args.model_name!r}: {e}"
            ) from e

        return InternalRegisterResult(
            backend_spec=spec,
            inference_server_addrs=inf_addrs,
            data_proxy_addrs=proxy_addrs,
            worker_pids=spawned,
        )
    except BaseException:
        # Roll back: kill anything we spawned.
        if spawned:
            logger.error(
                "Internal registration failed; killing %d spawned worker(s).",
                len(spawned),
            )
            kill_pids(spawned, grace_s=10.0)
        raise
