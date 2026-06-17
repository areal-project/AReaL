# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import click

from areal.experimental.cli.inference.state import logs_dir


@click.command(name="logs", help="Tail a log file under ~/.areal/inf/logs/.")
@click.option("--component", default="gateway", show_default=True)
@click.option("-f", "--follow", is_flag=True, help="Stream appended lines.")
@click.option("-n", "--lines", type=int, default=200, show_default=True)
def logs_cmd(component: str, follow: bool, lines: int) -> None:
    raise SystemExit(do_logs(component, follow, lines) or 0)


def do_logs(component: str, follow: bool, lines: int) -> int:
    log_dir = logs_dir()
    target = log_dir / f"{component}.log"
    if not target.exists():
        available = sorted(path.stem for path in log_dir.glob("*.log"))
        if not available:
            raise click.ClickException(f"no logs found under {log_dir}")
        raise click.ClickException(
            f"no log named {component!r} under {log_dir}; "
            f"available: {', '.join(available)}"
        )

    cmd = ["tail", f"-n{lines}"]
    if follow:
        cmd.append("-F")
    cmd.append(str(target))
    os.execvp(cmd[0], cmd)
