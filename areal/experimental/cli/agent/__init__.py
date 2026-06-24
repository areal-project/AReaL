# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import click

from areal.experimental.cli.agent.commands.logs import logs_cmd
from areal.experimental.cli.agent.commands.ps import ps_cmd
from areal.experimental.cli.agent.commands.run import run_cmd
from areal.experimental.cli.agent.commands.status import status_cmd
from areal.experimental.cli.agent.commands.stop import stop_cmd


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
    from areal.experimental.cli.agent.config import load_click_default_map

    ctx.default_map = load_click_default_map(extra=config_file)


agent.add_command(run_cmd)
agent.add_command(stop_cmd)
agent.add_command(status_cmd)
agent.add_command(ps_cmd)
agent.add_command(logs_cmd)
