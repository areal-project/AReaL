# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import click

from areal.experimental.cli.inference.commands.collect import collect_cmd
from areal.experimental.cli.inference.commands.deregister import deregister_cmd
from areal.experimental.cli.inference.commands.logs import logs_cmd
from areal.experimental.cli.inference.commands.models import models_cmd
from areal.experimental.cli.inference.commands.ps import ps_cmd
from areal.experimental.cli.inference.commands.register import register_cmd
from areal.experimental.cli.inference.commands.reward import reward_cmd
from areal.experimental.cli.inference.commands.run import run_cmd
from areal.experimental.cli.inference.commands.status import status_cmd
from areal.experimental.cli.inference.commands.stop import stop_cmd


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
    from areal.experimental.cli.inference.config import load_click_default_map

    ctx.default_map = load_click_default_map(extra=config_file)


inf.add_command(run_cmd)
inf.add_command(stop_cmd)
inf.add_command(status_cmd)
inf.add_command(ps_cmd)
inf.add_command(register_cmd)
inf.add_command(deregister_cmd)
inf.add_command(models_cmd)
inf.add_command(collect_cmd)
inf.add_command(reward_cmd)
inf.add_command(logs_cmd)
