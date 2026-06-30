# SPDX-License-Identifier: Apache-2.0

"""``<group> logs`` verb factory.

Every service-style CLI has a ``logs`` command that does the same
thing: resolve the service, find the log directory, tail one component
file. This class packages that pattern. Subcommand CLIs do::

    from areal.v2.cli.commands.logs import LogsCommand
    from areal.v2.cli.inference.lifecycle import inf_lifecycle

    logs_cmd = LogsCommand(lifecycle=inf_lifecycle).build()
    inf.add_command(logs_cmd)
"""

from __future__ import annotations

import os
from pathlib import Path

import click

from areal.v2.cli.lifecycle import ServiceLifecycle


class LogsCommand:
    default_component: str = "gateway"
    default_lines: int = 200

    def __init__(self, *, lifecycle: ServiceLifecycle) -> None:
        self.lifecycle = lifecycle

    def build(self) -> click.Command:
        @click.command(name="logs", help="Tail a service log file.")
        @click.option("--service", default=None)
        @click.option(
            "--component",
            default=self.default_component,
            show_default=True,
            help="Component log to tail (gateway / router / worker-N / etc.).",
        )
        @click.option("-f", "--follow", is_flag=True, help="Stream appended lines.")
        @click.option(
            "-n",
            "--lines",
            type=int,
            default=self.default_lines,
            show_default=True,
            help="Number of trailing lines to show before optional follow.",
        )
        def logs_cmd(
            service: str | None, component: str, follow: bool, lines: int
        ) -> None:
            self.execute(service, component, follow, lines)

        return logs_cmd

    def execute(
        self,
        service: str | None,
        component: str,
        follow: bool,
        lines: int,
    ) -> None:
        name = self.lifecycle.resolve_service_name(service)
        target = self.log_target(name, component)
        if not target.exists():
            available = sorted(p.stem for p in target.parent.glob("*.log"))
            if not available:
                raise click.ClickException(f"no logs found under {target.parent}")
            raise click.ClickException(
                f"no log named {component!r}; available: {', '.join(available)}"
            )
        self._exec_tail(target, lines=lines, follow=follow)

    def log_target(self, service: str, component: str) -> Path:
        """Resolve the log file path. Subclasses may override to add a
        component name alias / translation layer (e.g. accepting
        ``qwen/0/worker`` as shorthand for ``qwen-worker-0``)."""

        return self.lifecycle.store.logs_dir(service) / f"{component}.log"

    def _exec_tail(self, target: Path, *, lines: int, follow: bool) -> None:
        cmd = ["tail", f"-n{lines}"]
        if follow:
            cmd.append("-F")
        cmd.append(str(target))
        os.execvp(cmd[0], cmd)
