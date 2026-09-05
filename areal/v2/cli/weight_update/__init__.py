# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import click

from areal.v2.cli.weight_update.commands.pairs import pairs_cmd
from areal.v2.cli.weight_update.commands.ps import ps_cmd
from areal.v2.cli.weight_update.commands.status import status_cmd
from areal.v2.cli.weight_update.config import load_click_default_map


@click.group(
    name="weight-update",
    help="Inspect weight-update services. Read-only: the gateway belongs to "
    "the training controller that started it.",
)
@click.option(
    "--config",
    "config_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Extra TOML file merged on top of ~/.areal/weight-update/config.toml.",
)
@click.pass_context
def weight_update(ctx: click.Context, config_file: Path | None) -> None:
    ctx.default_map = load_click_default_map(extra=config_file)


weight_update.add_command(status_cmd)
weight_update.add_command(pairs_cmd)
weight_update.add_command(ps_cmd)
