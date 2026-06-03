# SPDX-License-Identifier: Apache-2.0

"""Process management for ``areal inf`` services.

Spawns the v2 inference gateway and router as detached subprocesses, polls
the gateway's ``/health`` until ready, and writes ``ServiceState``. No
hidden supervisor — later commands reconcile via state + PID + HTTP.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from areal.experimental.cli.gateway_client import GatewayClient, GatewayUnreachable
from areal.experimental.cli.inf_state import (
    ServiceState,
    liveness_summary,
    service_logs_dir,
    service_state_path,
)
from areal.experimental.cli.state import pid_alive


HEALTH_POLL_INTERVAL_S = 0.5


def _spawn(cmd: list[str], log_file: Path) -> subprocess.Popen:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    lf = open(log_file, "wb", buffering=0)
    return subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=lf,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=os.environ.copy(),
    )


def _gateway_cmd(
    *,
    host: str,
    port: int,
    admin_api_key: str,
    router_host: str,
    router_port: int,
    router_timeout: float,
    forward_timeout: float,
    log_level: str,
) -> list[str]:
    return [
        sys.executable, "-m", "areal.experimental.inference_service.gateway",
        "--host", host,
        "--port", str(port),
        "--admin-api-key", admin_api_key,
        "--router-addr", f"http://{router_host}:{router_port}",
        "--router-timeout", str(router_timeout),
        "--forward-timeout", str(forward_timeout),
        "--log-level", log_level,
    ]


def _router_cmd(
    *,
    host: str,
    port: int,
    admin_api_key: str,
    poll_interval: float,
    routing_strategy: str,
    log_level: str,
) -> list[str]:
    return [
        sys.executable, "-m", "areal.experimental.inference_service.router",
        "--host", host,
        "--port", str(port),
        "--admin-api-key", admin_api_key,
        "--poll-interval", str(poll_interval),
        "--routing-strategy", routing_strategy,
        "--log-level", log_level,
    ]


def _signal_pid(pid: int, sig: int) -> None:
    try:
        os.killpg(os.getpgid(pid), sig)
        return
    except ProcessLookupError:
        return
    except (PermissionError, OSError):
        pass
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        pass


def _wait_dead(pids: list[int], deadline: float) -> bool:
    while time.time() < deadline:
        if not any(pid_alive(p) for p in pids):
            return True
        time.sleep(0.2)
    return False


def _kill_state(state: ServiceState, grace: float) -> None:
    pids = [state.gateway_pid, state.router_pid]
    for p in pids:
        if pid_alive(p):
            _signal_pid(p, signal.SIGTERM)
    if not _wait_dead(pids, time.time() + grace):
        for p in pids:
            if pid_alive(p):
                _signal_pid(p, signal.SIGKILL)


def _refuse_if_active(name: str, force: bool) -> None:
    p = service_state_path(name)
    if not p.exists():
        return
    try:
        existing = ServiceState.load(name)
    except (ValueError, FileNotFoundError, TypeError):
        return
    live = liveness_summary(existing)
    healthy = False
    if live["gateway_pid_alive"]:
        try:
            GatewayClient(existing.gateway_url, timeout=1.0).health()
            healthy = True
        except GatewayUnreachable:
            healthy = False
    if healthy and not force:
        raise SystemExit(
            f"Service {name!r} is already healthy (gateway={existing.gateway_url}). "
            f"Use --force to replace it."
        )
    if force and (live["gateway_pid_alive"] or live["router_pid_alive"]):
        _kill_state(existing, grace=5.0)
    existing.remove()


def start_service(
    *,
    name: str,
    gateway_host: str,
    gateway_port: int,
    router_host: str,
    router_port: int,
    admin_api_key: str,
    routing_strategy: str = "round_robin",
    poll_interval: float = 5.0,
    router_timeout: float = 2.0,
    forward_timeout: float = 120.0,
    log_level: str = "info",
    force: bool = False,
    launch_timeout: float = 30.0,
) -> ServiceState:
    _refuse_if_active(name, force=force)

    logs = service_logs_dir(name)

    router_proc = _spawn(
        _router_cmd(
            host=router_host,
            port=router_port,
            admin_api_key=admin_api_key,
            poll_interval=poll_interval,
            routing_strategy=routing_strategy,
            log_level=log_level,
        ),
        logs / "router.log",
    )

    time.sleep(0.3)

    gateway_proc = _spawn(
        _gateway_cmd(
            host=gateway_host,
            port=gateway_port,
            admin_api_key=admin_api_key,
            router_host=router_host if router_host not in ("0.0.0.0", "::") else "127.0.0.1",
            router_port=router_port,
            router_timeout=router_timeout,
            forward_timeout=forward_timeout,
            log_level=log_level,
        ),
        logs / "gateway.log",
    )

    state = ServiceState(
        name=name,
        gateway_host=gateway_host,
        gateway_port=gateway_port,
        router_host=router_host,
        router_port=router_port,
        gateway_pid=gateway_proc.pid,
        router_pid=router_proc.pid,
        admin_api_key=admin_api_key,
        log_level=log_level,
        routing_strategy=routing_strategy,
        created_at=time.time(),
    )

    client = GatewayClient(state.gateway_url, admin_api_key=admin_api_key, timeout=1.5)
    deadline = time.time() + launch_timeout
    last_err: Exception | None = None
    while time.time() < deadline:
        if not pid_alive(gateway_proc.pid) or not pid_alive(router_proc.pid):
            _kill_state(state, grace=2.0)
            raise SystemExit(
                f"Service {name!r} died during launch. Check logs under {logs}."
            )
        try:
            client.health()
            break
        except GatewayUnreachable as e:
            last_err = e
            time.sleep(HEALTH_POLL_INTERVAL_S)
    else:
        _kill_state(state, grace=2.0)
        raise SystemExit(
            f"Service {name!r} did not become healthy within {launch_timeout:.0f}s "
            f"(last error: {last_err}). Logs: {logs}"
        )

    state.save()
    return state
