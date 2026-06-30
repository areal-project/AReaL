# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import click

from areal.v2.cli.commands.logs import LogsCommand
from areal.v2.cli.inference.commands.deregister import deregister_cmd
from areal.v2.cli.inference.commands.models import models_cmd
from areal.v2.cli.inference.commands.ps import ps_cmd
from areal.v2.cli.inference.commands.register import register_cmd
from areal.v2.cli.inference.commands.run import run_cmd
from areal.v2.cli.inference.commands.status import status_cmd
from areal.v2.cli.inference.commands.stop import stop_cmd
from areal.v2.cli.inference.config import load_click_default_map
from areal.v2.cli.inference.lifecycle import inf_lifecycle


@click.group(help="Manage local AReaL inference services.")
@click.option(
    "--config",
    "config_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Extra TOML file merged on top of ~/.areal/inf/config.toml.",
)
@click.pass_context
def inf(ctx: click.Context, config_file: Path | None) -> None:
    ctx.default_map = load_click_default_map(extra=config_file)


inf.add_command(run_cmd)
inf.add_command(stop_cmd)
inf.add_command(status_cmd)
inf.add_command(ps_cmd)
inf.add_command(register_cmd)
inf.add_command(deregister_cmd)
inf.add_command(models_cmd)
inf.add_command(LogsCommand(lifecycle=inf_lifecycle).build())
