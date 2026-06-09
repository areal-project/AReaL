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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from areal.experimental.cli.commands.inf.launcher import _spawn_detached, kill_pids
from areal.experimental.cli.state import pid_alive


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
    *, model_path: str, tokenizer_path: str, host: str, port: int, tp: int
) -> list[str]:
    from areal.api.cli_args import SGLangConfig

    cfg = SGLangConfig(model_path=model_path)
    cmd_str = SGLangConfig.build_cmd(
        sglang_config=cfg,
        tp_size=tp,
        base_gpu_id=0,
        host=host,
        port=port,
        n_nodes=1,
        node_rank=0,
        pp_size=1,
    )
    # build_cmd returns a single shell-style string; split into argv.
    return _shlex_split(cmd_str)


def _build_vllm_cmd(
    *, model_path: str, tokenizer_path: str, host: str, port: int, tp: int, pp: int
) -> list[str]:
    from areal.api.cli_args import vLLMConfig

    cfg = vLLMConfig(model_path=model_path)
    cmd_str = vLLMConfig.build_cmd(
        vllm_config=cfg,
        tp_size=tp,
        pp_size=pp,
    )
    parts = _shlex_split(cmd_str)
    # vLLMConfig.build_cmd does not embed --host/--port; append.
    parts += ["--host", host, "--port", str(port)]
    return parts


def _shlex_split(cmd: str) -> list[str]:
    import shlex
    return shlex.split(cmd)


def _build_data_proxy_cmd(
    *,
    host: str,
    port: int,
    backend_addr: str,
    backend_type: str,
    tokenizer_path: str,
    admin_api_key: str,
    log_level: str,
    request_timeout: float,
    set_reward_finish_timeout: float,
    tool_call_parser: str,
    reasoning_parser: str,
    chat_template_type: str,
    engine_max_tokens: int | None,
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
        "--request-timeout", str(request_timeout),
        "--set-reward-finish-timeout", str(set_reward_finish_timeout),
        "--tool-call-parser", tool_call_parser,
        "--reasoning-parser", reasoning_parser,
        "--chat-template-type", chat_template_type,
    ]
    if engine_max_tokens is not None:
        cmd += ["--engine-max-tokens", str(engine_max_tokens)]
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
    # data-proxy knobs
    request_timeout: float = 120.0
    set_reward_finish_timeout: float = 0.0
    tool_call_parser: str = "qwen"
    reasoning_parser: str = "qwen3"
    chat_template_type: str = "hf"
    engine_max_tokens: int | None = None


@dataclass
class InternalRegisterResult:
    backend_spec: BackendSpec
    inference_server_addrs: list[str]
    data_proxy_addrs: list[str]
    worker_pids: list[int]


def register_internal_model(
    args: InternalRegisterArgs, *, gateway_url: str
) -> InternalRegisterResult:
    """Spawn N sglang/vllm + N data-proxy + register with gateway.

    On any failure, kills everything spawned so far and re-raises.
    """
    from areal.experimental.cli.commands.inf.gateway_client import (
        GatewayClient,
        GatewayHTTPError,
        GatewayUnreachable,
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
                    tokenizer_path=tokenizer_path,
                    host="127.0.0.1",
                    port=inf_port,
                    tp=spec.tp,
                )
            else:
                cmd = _build_vllm_cmd(
                    model_path=args.model_path,
                    tokenizer_path=tokenizer_path,
                    host="127.0.0.1",
                    port=inf_port,
                    tp=spec.tp,
                    pp=spec.pp,
                )
            print(
                f"  spawning {spec.engine} server "
                f"(replica {replica}/{spec.dp}, tp={spec.tp}, port={inf_port}) ...",
                file=sys.stderr,
            )
            inf_pid = _spawn_detached(cmd, inf_log)
            spawned.append(inf_pid)
            inf_addr = f"http://127.0.0.1:{inf_port}"
            inf_addrs.append(inf_addr)
            print(f"    pid={inf_pid}, log={inf_log}", file=sys.stderr)

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
                request_timeout=args.request_timeout,
                set_reward_finish_timeout=args.set_reward_finish_timeout,
                tool_call_parser=args.tool_call_parser,
                reasoning_parser=args.reasoning_parser,
                chat_template_type=args.chat_template_type,
                engine_max_tokens=args.engine_max_tokens,
            )
            print(
                f"  spawning data-proxy "
                f"(replica {replica}/{spec.dp}, port={proxy_port}) ...",
                file=sys.stderr,
            )
            proxy_pid = _spawn_detached(proxy_cmd, proxy_log)
            spawned.append(proxy_pid)
            proxy_addr = f"http://127.0.0.1:{proxy_port}"
            proxy_addrs.append(proxy_addr)
            print(f"    pid={proxy_pid}, log={proxy_log}", file=sys.stderr)

            _wait_health(
                f"{proxy_addr}/health",
                deadline=time.time() + 30.0,
                supervisor_pids=[proxy_pid],
                what=f"data-proxy {replica}",
            )

        # Register with gateway
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
            print(
                "  internal registration failed -- killing spawned workers.",
                file=sys.stderr,
            )
            kill_pids(spawned, grace_s=10.0)
        raise
