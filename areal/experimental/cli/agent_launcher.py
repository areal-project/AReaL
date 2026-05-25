# SPDX-License-Identifier: Apache-2.0

"""Process management for ``areal agent`` services.

Spawns one router + N × (worker + data_proxy) + one gateway as detached
subprocesses (their existing ``python -m areal.experimental.agent_service.{...}``
entry points), registers each proxy with the router, polls the gateway's
``/health`` until ready, and writes ``AgentServiceState`` to
``~/.areal/agent/services/<name>.json``.

Non-daemon design mirrors ``inf_launcher.py``: the CLI exits after writing
state; later commands reconcile via state file + live PIDs + HTTP health.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from areal.experimental.cli.agent_gateway_client import (
    AgentGatewayClient,
    AgentGatewayUnreachable,
)
from areal.experimental.cli.agent_state import (
    AgentServiceState,
    PairProcess,
    agent_logs_dir,
    agent_service_state_path,
    liveness_summary,
)
from areal.experimental.cli.state import pid_alive

try:
    from areal.utils.logging import getLogger

    logger = getLogger("ArealCLI")
except Exception:  # pragma: no cover - thin install fallback
    import logging

    logger = logging.getLogger("ArealCLI")


HEALTH_POLL_INTERVAL_S = 0.5


def _spawn(cmd: list[str], log_file: Path) -> subprocess.Popen:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    lf = open(log_file, "ab", buffering=0)
    return subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=lf,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=os.environ.copy(),
    )


def _router_cmd(
    *,
    host: str,
    port: int,
    admin_api_key: str,
    poll_interval: float,
    worker_health_timeout: float,
    log_level: str,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "areal.experimental.agent_service.router",
        "--host",
        host,
        "--port",
        str(port),
        "--admin-api-key",
        admin_api_key,
        "--poll-interval",
        str(poll_interval),
        "--worker-health-timeout",
        str(worker_health_timeout),
        "--log-level",
        log_level,
    ]


def _worker_cmd(
    *,
    agent_class: str,
    host: str,
    port: int,
    log_level: str,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "areal.experimental.agent_service.worker",
        "--agent",
        agent_class,
        "--host",
        host,
        "--port",
        str(port),
        "--log-level",
        log_level,
    ]


def _proxy_cmd(
    *,
    worker_addr: str,
    host: str,
    port: int,
    request_timeout: float,
    session_timeout: int,
    log_level: str,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "areal.experimental.agent_service.data_proxy",
        "--worker-addr",
        worker_addr,
        "--host",
        host,
        "--port",
        str(port),
        "--request-timeout",
        str(request_timeout),
        "--session-timeout",
        str(session_timeout),
        "--log-level",
        log_level,
    ]


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
        sys.executable,
        "-m",
        "areal.experimental.agent_service.gateway",
        "--host",
        host,
        "--port",
        str(port),
        "--admin-api-key",
        admin_api_key,
        "--router-addr",
        f"http://{router_host}:{router_port}",
        "--router-timeout",
        str(router_timeout),
        "--forward-timeout",
        str(forward_timeout),
        "--log-level",
        log_level,
    ]


def _refuse_if_active(name: str, force: bool) -> None:
    p = agent_service_state_path(name)
    if not p.exists():
        return
    try:
        existing = AgentServiceState.load(name)
    except (ValueError, FileNotFoundError, TypeError):
        return
    live = liveness_summary(existing)
    healthy = False
    if live["gateway_pid_alive"]:
        try:
            AgentGatewayClient(existing.gateway_url, timeout=1.0).health()
            healthy = True
        except AgentGatewayUnreachable:
            healthy = False
    if healthy and not force:
        raise SystemExit(
            f"Agent service {name!r} is already healthy "
            f"(gateway={existing.gateway_url}). "
            f"Use --force to replace it."
        )
    if force and (
        live["gateway_pid_alive"]
        or live["router_pid_alive"]
        or any(live["worker_pids_alive"])
        or any(live["proxy_pids_alive"])
    ):
        _kill_state(existing, grace=5.0)
    existing.remove()


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


def _all_pids(state: AgentServiceState) -> list[int]:
    pids = [state.gateway_pid, state.router_pid]
    for pair in state.pairs:
        pids.extend([pair.worker_pid, pair.proxy_pid])
    return pids


def _kill_state(state: AgentServiceState, grace: float) -> None:
    pids = _all_pids(state)
    for p in pids:
        if pid_alive(p):
            _signal_pid(p, signal.SIGTERM)
    if not _wait_dead(pids, time.time() + grace):
        for p in pids:
            if pid_alive(p):
                logger.warning(
                    "[agent stop] SIGTERM timed out after %.1fs; "
                    "sending SIGKILL to pid %d.",
                    grace,
                    p,
                )
                _signal_pid(p, signal.SIGKILL)


def _localish(host: str) -> str:
    return "127.0.0.1" if host in ("0.0.0.0", "::") else host


def _post_router(
    *,
    router_addr: str,
    path: str,
    body: dict,
    admin_api_key: str,
    timeout: float,
) -> tuple[int, bytes]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        router_addr.rstrip("/") + path,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {admin_api_key}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        raw = e.read() if hasattr(e, "read") else b""
        return e.code, raw


def _wait_for_http(
    url: str,
    *,
    deadline: float,
    interval: float = HEALTH_POLL_INTERVAL_S,
) -> bool:
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if 200 <= resp.status < 300:
                    return True
        except (urllib.error.URLError, urllib.error.HTTPError, socket.timeout, ConnectionError):
            pass
        time.sleep(interval)
    return False


def _register_proxy(
    *,
    router_addr: str,
    proxy_addr: str,
    admin_api_key: str,
    timeout: float = 10.0,
) -> None:
    status, raw = _post_router(
        router_addr=router_addr,
        path="/register",
        body={"addr": proxy_addr},
        admin_api_key=admin_api_key,
        timeout=timeout,
    )
    if not (200 <= status < 300):
        raise RuntimeError(
            f"router /register rejected proxy {proxy_addr} "
            f"(HTTP {status}: {raw[:200]!r})"
        )


def _unregister_proxy(
    *,
    router_addr: str,
    proxy_addr: str,
    admin_api_key: str,
    timeout: float = 5.0,
) -> None:
    try:
        _post_router(
            router_addr=router_addr,
            path="/unregister",
            body={"addr": proxy_addr},
            admin_api_key=admin_api_key,
            timeout=timeout,
        )
    except Exception as exc:  # pragma: no cover - best-effort
        logger.debug("[agent stop] unregister %s failed: %s", proxy_addr, exc)


def start_agent_service(
    *,
    name: str,
    agent_class: str,
    num_pairs: int,
    gateway_host: str,
    gateway_port: int,
    router_host: str,
    router_port: int,
    worker_host: str,
    worker_base_port: int,
    proxy_host: str,
    proxy_base_port: int,
    admin_api_key: str,
    inf_addr: str = "",
    inf_model: str = "",
    inf_api_key: str | None = None,
    router_poll_interval: float = 5.0,
    worker_health_timeout: float = 2.0,
    proxy_request_timeout: float = 600.0,
    proxy_session_timeout: int = 3600,
    gateway_router_timeout: float = 2.0,
    gateway_forward_timeout: float = 120.0,
    log_level: str = "info",
    force: bool = False,
    launch_timeout: float = 60.0,
    mode: str = "detached",
    extra: dict | None = None,
) -> AgentServiceState:
    """Spawn router + N pairs + gateway, register pairs, poll health, persist state.

    Inference-side coordinates (``inf_addr``, ``inf_model``, ``inf_api_key``)
    are propagated to *all* child processes via environment variables
    (``AREAL_INF_ADDR``, ``AREAL_INF_MODEL``, ``AREAL_INF_API_KEY``); the
    agent class reads them at startup. The CLI never persists the key in
    plaintext — only ``inf_api_key_present`` is saved.
    """
    if num_pairs < 1:
        raise SystemExit(f"num_pairs must be >= 1 (got {num_pairs}).")
    _refuse_if_active(name, force=force)

    logs = agent_logs_dir(name)

    env_overrides: dict[str, str] = {}
    if inf_addr:
        env_overrides["AREAL_INF_ADDR"] = inf_addr
    if inf_model:
        env_overrides["AREAL_INF_MODEL"] = inf_model
    if inf_api_key:
        env_overrides["AREAL_INF_API_KEY"] = inf_api_key
    if env_overrides:
        for k, v in env_overrides.items():
            os.environ[k] = v

    router_proc = _spawn(
        _router_cmd(
            host=router_host,
            port=router_port,
            admin_api_key=admin_api_key,
            poll_interval=router_poll_interval,
            worker_health_timeout=worker_health_timeout,
            log_level=log_level,
        ),
        logs / "router.log",
    )

    router_local = _localish(router_host)
    router_addr = f"http://{router_local}:{router_port}"
    if not _wait_for_http(router_addr + "/health", deadline=time.time() + 10.0):
        _signal_pid(router_proc.pid, signal.SIGTERM)
        raise SystemExit(
            f"Router for agent service {name!r} did not become healthy. "
            f"Logs: {logs / 'router.log'}"
        )

    pairs: list[PairProcess] = []
    pair_procs: list[subprocess.Popen] = [router_proc]
    try:
        for i in range(num_pairs):
            w_port = worker_base_port + i
            p_port = proxy_base_port + i
            worker_proc = _spawn(
                _worker_cmd(
                    agent_class=agent_class,
                    host=worker_host,
                    port=w_port,
                    log_level=log_level,
                ),
                logs / f"worker-{i}.log",
            )
            pair_procs.append(worker_proc)
            worker_addr = f"http://{_localish(worker_host)}:{w_port}"
            if not _wait_for_http(worker_addr + "/health", deadline=time.time() + 15.0):
                raise SystemExit(
                    f"Worker pair {i} did not become healthy at {worker_addr}. "
                    f"Logs: {logs / f'worker-{i}.log'}"
                )

            proxy_proc = _spawn(
                _proxy_cmd(
                    worker_addr=worker_addr,
                    host=proxy_host,
                    port=p_port,
                    request_timeout=proxy_request_timeout,
                    session_timeout=proxy_session_timeout,
                    log_level=log_level,
                ),
                logs / f"proxy-{i}.log",
            )
            pair_procs.append(proxy_proc)
            proxy_addr = f"http://{_localish(proxy_host)}:{p_port}"
            if not _wait_for_http(proxy_addr + "/health", deadline=time.time() + 15.0):
                raise SystemExit(
                    f"Proxy pair {i} did not become healthy at {proxy_addr}. "
                    f"Logs: {logs / f'proxy-{i}.log'}"
                )

            _register_proxy(
                router_addr=router_addr,
                proxy_addr=proxy_addr,
                admin_api_key=admin_api_key,
            )

            pairs.append(
                PairProcess(
                    index=i,
                    worker_host=worker_host,
                    worker_port=w_port,
                    worker_pid=worker_proc.pid,
                    proxy_host=proxy_host,
                    proxy_port=p_port,
                    proxy_pid=proxy_proc.pid,
                )
            )

        # Brief grace so the router fully observes registrations before
        # gateway probes /route.
        time.sleep(0.3)

        gateway_proc = _spawn(
            _gateway_cmd(
                host=gateway_host,
                port=gateway_port,
                admin_api_key=admin_api_key,
                router_host=router_local,
                router_port=router_port,
                router_timeout=gateway_router_timeout,
                forward_timeout=gateway_forward_timeout,
                log_level=log_level,
            ),
            logs / "gateway.log",
        )
        pair_procs.append(gateway_proc)

        state = AgentServiceState(
            name=name,
            agent_class=agent_class,
            num_pairs=num_pairs,
            gateway_host=gateway_host,
            gateway_port=gateway_port,
            router_host=router_host,
            router_port=router_port,
            gateway_pid=gateway_proc.pid,
            router_pid=router_proc.pid,
            admin_api_key=admin_api_key,
            pairs=pairs,
            inf_addr=inf_addr,
            inf_model=inf_model,
            inf_api_key_present=bool(inf_api_key),
            mode=mode,
            log_level=log_level,
            created_at=time.time(),
            extra=extra or {},
        )

        client = AgentGatewayClient(
            state.gateway_url, admin_api_key=admin_api_key, timeout=1.5
        )
        deadline = time.time() + launch_timeout
        last_err: Exception | None = None
        while time.time() < deadline:
            if not all(pid_alive(p.pid) for p in pair_procs):
                _kill_state(state, grace=2.0)
                raise SystemExit(
                    f"Agent service {name!r} died during launch. "
                    f"Check logs under {logs}."
                )
            try:
                client.health()
                break
            except AgentGatewayUnreachable as e:
                last_err = e
                time.sleep(HEALTH_POLL_INTERVAL_S)
        else:
            _kill_state(state, grace=2.0)
            raise SystemExit(
                f"Agent service {name!r} did not become healthy within "
                f"{launch_timeout:.0f}s (last error: {last_err}). Logs: {logs}"
            )

        state.save()
        logger.info(
            "[agent run] service=%s gateway=%s router=%s pairs=%d "
            "(gw_pid=%d, rt_pid=%d)",
            name,
            state.gateway_url,
            state.router_url,
            num_pairs,
            gateway_proc.pid,
            router_proc.pid,
        )
        return state
    except BaseException:
        for proc in pair_procs:
            try:
                if proc.poll() is None:
                    _signal_pid(proc.pid, signal.SIGTERM)
            except Exception:
                pass
        raise


def stop_agent_service(
    name: str,
    grace_period: float = 10.0,
    keep_state: bool = False,
) -> int:
    try:
        state = AgentServiceState.load(name)
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from e

    router_addr = state.router_url
    for pair in state.pairs:
        proxy_addr = f"http://{_localish(pair.proxy_host)}:{pair.proxy_port}"
        _unregister_proxy(
            router_addr=router_addr,
            proxy_addr=proxy_addr,
            admin_api_key=state.admin_api_key,
        )

    _kill_state(state, grace=grace_period)
    if not keep_state:
        state.remove()
    return 0
