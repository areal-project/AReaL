# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import click

from areal.v2.cli.agent.commands.ps import ps_cmd
from areal.v2.cli.agent.commands.run import run_cmd
from areal.v2.cli.agent.commands.status import status_cmd
from areal.v2.cli.agent.commands.stop import stop_cmd
from areal.v2.cli.agent.config import load_click_default_map
from areal.v2.cli.agent.lifecycle import agent_lifecycle
from areal.v2.cli.commands.logs import LogsCommand


@click.group(help="Manage agent services.")
@click.option(
    "--config",
    "config_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Extra TOML file merged on top of ~/.areal/agent/config.toml.",
)
@click.pass_context
def agent(ctx: click.Context, config_file: Path | None) -> None:
    ctx.default_map = load_click_default_map(extra=config_file)


agent.add_command(run_cmd)
agent.add_command(stop_cmd)
agent.add_command(status_cmd)
agent.add_command(ps_cmd)
agent.add_command(LogsCommand(lifecycle=agent_lifecycle).build())
