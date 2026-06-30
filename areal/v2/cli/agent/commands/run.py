# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import click

from areal.v2.cli.agent.launcher import launch_agent_stack
from areal.v2.cli.agent.lifecycle import agent_lifecycle
from areal.v2.cli.agent.state import ServiceState
from areal.v2.cli.client import ServiceHTTPError, ServiceUnreachable
from areal.v2.cli.process import kill_pids
from areal.v2.cli.state import DEFAULT_SERVICE
from areal.v2.cli.utils import register_cli_logger

logger = register_cli_logger("AgentCli")


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
    force: bool,
) -> int:
    if not agent:
        raise click.UsageError("--agent is required")

    if force:
        agent_lifecycle.force_replace_slot(service, grace_s=5.0)
    else:
        agent_lifecycle.refuse_if_running(service)

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
        )
        launched.save()
    except (ServiceHTTPError, ServiceUnreachable, RuntimeError, ValueError) as exc:
        if launched is not None:
            kill_pids(
                [pid for _, handle in launched.components() if (pid := handle.pid) > 0],
                grace_s=5.0,
            )
        raise click.ClickException(f"failed to launch agent service: {exc}") from exc

    logger.info("service=%s gateway=%s", service, launched.gateway.url)
    return 0
