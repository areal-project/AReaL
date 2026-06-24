# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
import urllib.error
import urllib.request

import click

from areal.experimental.cli.agent.process import pid_alive
from areal.experimental.cli.agent.state import (
    ServiceState,
    resolve_service_name,
    service_state_path,
)
from areal.utils.logging import getLogger

logger = getLogger("AgentCli")


def running_state(service: str | None = None) -> ServiceState | None:
    name = resolve_service_name(service)
    if not service_state_path(name).exists():
        return None
    try:
        state = ServiceState.load(name)
    except Exception:
        return None
    return state


def load_running_state(service: str | None = None) -> ServiceState:
    name = resolve_service_name(service)
    if not service_state_path(name).exists():
        raise click.ClickException(f"service {name!r} is not running")
    try:
        return ServiceState.load(name)
    except Exception as exc:
        raise click.ClickException(f"failed to load state: {exc}") from exc


def wait_http_health(url: str, *, pid: int, timeout: float, label: str) -> None:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        if not pid_alive(pid):
            raise click.ClickException(f"{label} subprocess died during startup")
        try:
            with urllib.request.urlopen(
                f"{url.rstrip('/')}/health", timeout=2.0
            ) as resp:
                if resp.status < 500:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
            last_error = exc
            time.sleep(0.3)
    raise click.ClickException(f"{label} did not become healthy: {last_error}")
