# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import click

from areal.experimental.cli.agent.client import AgentHTTPError, AgentUnreachable
from areal.experimental.cli.agent.common import logger
from areal.experimental.cli.agent.launcher import launch_agent_stack
from areal.experimental.cli.agent.process import kill_pids, pid_alive
from areal.experimental.cli.agent.state import (
    DEFAULT_SERVICE,
    ServiceState,
    service_state_path,
)


@click.command(name="run", help="Launch an agent service.")
@click.option("--service", default=DEFAULT_SERVICE, show_default=True)
@click.option("--agent", default=None, help="Agent import path.")
@click.option("--num-pairs", type=int, default=1, show_default=True)
@click.option("--admin-api-key", default="areal-agent-admin", show_default=True)
@click.option("--setup-timeout", type=float, default=120.0, show_default=True)
@click.option("--health-poll-interval", type=float, default=5.0, show_default=True)
@click.option("--drain-timeout", type=float, default=30.0, show_default=True)
@click.option("--session-timeout", type=float, default=1800.0, show_default=True)
@click.option(
    "--log-level",
    type=click.Choice(["debug", "info", "warning", "error"]),
    default="info",
    show_default=True,
)
@click.option("--inf-addr", default="", help="Optional inference service base URL.")
@click.option("--inf-api-key", default="", help="Inference admin API key.")
@click.option("--inf-model", default="", help="Inference model name.")
@click.option("--force", is_flag=True, help="Replace stale or running service state.")
def run_cmd(**opts) -> None:
    raise SystemExit(do_run(**opts) or 0)


def do_run(
    *,
    service: str,
    agent: str | None,
    num_pairs: int,
    admin_api_key: str,
    setup_timeout: float,
    health_poll_interval: float,
    drain_timeout: float,
    session_timeout: float,
    log_level: str,
    inf_addr: str,
    inf_api_key: str,
    inf_model: str,
    force: bool,
) -> int:
    if not agent:
        raise click.UsageError("--agent is required")

    existing = _load_existing(service)
    if existing is not None and any(pid_alive(pid) for pid in existing.all_pids()):
        if not force:
            raise click.ClickException(
                f"service {service!r} already has live processes; "
                f"run `areal agent stop` first or use --force"
            )
        kill_pids(existing.all_pids(), grace_s=5.0)
    elif service_state_path(service).exists() and not force:
        raise click.ClickException(
            f"stale state exists for {service!r}; "
            f"use `areal agent run --service {service} --force`"
        )

    launched: ServiceState | None = None
    try:
        launched = launch_agent_stack(
            service=service,
            agent=agent,
            admin_api_key=admin_api_key,
            num_pairs=num_pairs,
            setup_timeout=setup_timeout,
            session_timeout=session_timeout,
            health_poll_interval=health_poll_interval,
            drain_timeout=drain_timeout,
            log_level=log_level,
            inf_addr=inf_addr,
            inf_api_key=inf_api_key,
            inf_model=inf_model,
        )
        launched.save()
    except (AgentHTTPError, AgentUnreachable, RuntimeError, ValueError) as exc:
        if launched is not None:
            kill_pids(launched.all_pids(), grace_s=5.0)
        raise click.ClickException(f"failed to launch agent service: {exc}") from exc

    logger.info("service=%s gateway=%s", service, launched.gateway.url)
    return 0


def _load_existing(service: str) -> ServiceState | None:
    if not service_state_path(service).exists():
        return None
    try:
        return ServiceState.load(service)
    except Exception:
        return None
