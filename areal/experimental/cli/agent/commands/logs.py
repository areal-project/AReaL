# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import click

from areal.experimental.cli.agent.state import resolve_service_name, service_logs_dir


@click.command(name="logs", help="Tail a log file under ~/.areal/agent/logs/.")
@click.option("--service", default=None)
@click.option("--component", default="gateway", show_default=True)
@click.option("-f", "--follow", is_flag=True)
@click.option("-n", "--lines", type=int, default=200, show_default=True)
def logs_cmd(service: str | None, component: str, follow: bool, lines: int) -> None:
    raise SystemExit(
        do_logs(service=service, component=component, follow=follow, lines=lines) or 0
    )


def do_logs(*, service: str | None, component: str, follow: bool, lines: int) -> int:
    name = resolve_service_name(service)
    log_dir = service_logs_dir(name)
    target = log_dir / f"{component}.log"
    if not target.exists():
        available = sorted(p.stem for p in log_dir.glob("*.log"))
        if not available:
            raise click.ClickException(f"no logs found under {log_dir}")
        raise click.ClickException(
            f"no log named {component!r}; available: {', '.join(available)}"
        )

    cmd = ["tail", f"-n{lines}"]
    if follow:
        cmd.append("-F")
    cmd.append(str(target))
    os.execvp(cmd[0], cmd)
    return 0
