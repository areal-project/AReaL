# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import click

from areal.v2.cli.agent import agent
from areal.v2.cli.inference import inf
from areal.v2.cli.training import train
from areal.version import __version__


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    help="AReaL operator CLI.",
)
@click.version_option(__version__, prog_name="areal")
def cli() -> None:
    pass


cli.add_command(agent)
cli.add_command(inf)
cli.add_command(train)
