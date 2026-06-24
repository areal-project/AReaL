# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

from areal.experimental.cli.agent.common import wait_http_health
from areal.experimental.cli.agent.http import RouterClient
from areal.experimental.cli.agent.process import pick_free_port, spawn_process
from areal.experimental.cli.agent.state import (
    PairState,
    ProcessState,
    ServiceState,
    service_logs_dir,
)


def launch_agent_stack(
    *,
    service: str,
    agent: str,
    admin_api_key: str,
    num_pairs: int,
    setup_timeout: float,
    session_timeout: float,
    health_poll_interval: float,
    drain_timeout: float,
    log_level: str,
    inf_addr: str,
    inf_api_key: str,
    inf_model: str,
) -> ServiceState:
    log_dir = service_logs_dir(service)

    router_port = pick_free_port()
    router_log = log_dir / "router.log"
    router_pid = _spawn_router(
        host="127.0.0.1",
        port=router_port,
        admin_api_key=admin_api_key,
        log_level=log_level,
        log_file=router_log,
    )
    router_url = f"http://127.0.0.1:{router_port}"
    wait_http_health(router_url, pid=router_pid, timeout=setup_timeout, label="router")

    pairs: list[PairState] = []
    router = RouterClient(router_url, admin_api_key)
    for idx in range(num_pairs):
        pair = _spawn_pair(
            index=idx,
            agent=agent,
            session_timeout=session_timeout,
            log_level=log_level,
            log_dir=log_dir,
            setup_timeout=setup_timeout,
        )
        router.register_proxy(pair.data_proxy.url)
        pairs.append(pair)

    gateway_port = pick_free_port()
    gateway_log = log_dir / "gateway.log"
    gateway_pid = _spawn_gateway(
        host="127.0.0.1",
        port=gateway_port,
        router_url=router_url,
        admin_api_key=admin_api_key,
        log_level=log_level,
        log_file=gateway_log,
    )
    gateway_url = f"http://127.0.0.1:{gateway_port}"
    wait_http_health(
        gateway_url, pid=gateway_pid, timeout=setup_timeout, label="gateway"
    )

    return ServiceState(
        service=service,
        launch_mode="detached",
        agent=agent,
        admin_api_key=admin_api_key,
        gateway=ProcessState(
            component="gateway",
            pid=gateway_pid,
            url=gateway_url,
            log_file=str(gateway_log),
        ),
        router=ProcessState(
            component="router",
            pid=router_pid,
            url=router_url,
            log_file=str(router_log),
        ),
        pairs=pairs,
        inf_addr=inf_addr,
        inf_api_key=inf_api_key,
        inf_model=inf_model,
        session_timeout=session_timeout,
        health_poll_interval=health_poll_interval,
        drain_timeout=drain_timeout,
    )


def _spawn_router(
    *,
    host: str,
    port: int,
    admin_api_key: str,
    log_level: str,
    log_file: Path,
) -> int:
    return spawn_process(
        [
            sys.executable,
            "-m",
            "areal.experimental.agent_service.router",
            "--host",
            host,
            "--port",
            str(port),
            "--admin-api-key",
            admin_api_key,
            "--log-level",
            log_level,
        ],
        log_file,
    )


def _spawn_gateway(
    *,
    host: str,
    port: int,
    router_url: str,
    admin_api_key: str,
    log_level: str,
    log_file: Path,
) -> int:
    return spawn_process(
        [
            sys.executable,
            "-m",
            "areal.experimental.agent_service.gateway",
            "--host",
            host,
            "--port",
            str(port),
            "--router-addr",
            router_url,
            "--admin-api-key",
            admin_api_key,
            "--log-level",
            log_level,
        ],
        log_file,
    )


def _spawn_pair(
    *,
    index: int,
    agent: str,
    session_timeout: float,
    log_level: str,
    log_dir: Path,
    setup_timeout: float,
) -> PairState:
    worker_port = pick_free_port()
    worker_log = log_dir / f"worker-{index}.log"
    worker_pid = spawn_process(
        [
            sys.executable,
            "-m",
            "areal.experimental.agent_service.worker",
            "--agent",
            agent,
            "--host",
            "127.0.0.1",
            "--port",
            str(worker_port),
            "--log-level",
            log_level,
        ],
        worker_log,
    )
    worker_url = f"http://127.0.0.1:{worker_port}"
    wait_http_health(
        worker_url, pid=worker_pid, timeout=setup_timeout, label=f"worker-{index}"
    )

    proxy_port = pick_free_port()
    proxy_log = log_dir / f"proxy-{index}.log"
    proxy_pid = spawn_process(
        [
            sys.executable,
            "-m",
            "areal.experimental.agent_service.data_proxy",
            "--worker-addr",
            worker_url,
            "--host",
            "127.0.0.1",
            "--port",
            str(proxy_port),
            "--session-timeout",
            str(int(session_timeout)),
            "--log-level",
            log_level,
        ],
        proxy_log,
    )
    proxy_url = f"http://127.0.0.1:{proxy_port}"
    wait_http_health(
        proxy_url, pid=proxy_pid, timeout=setup_timeout, label=f"proxy-{index}"
    )

    return PairState(
        index=index,
        worker=ProcessState(
            component=f"worker-{index}",
            pid=worker_pid,
            url=worker_url,
            log_file=str(worker_log),
        ),
        data_proxy=ProcessState(
            component=f"proxy-{index}",
            pid=proxy_pid,
            url=proxy_url,
            log_file=str(proxy_log),
        ),
    )
