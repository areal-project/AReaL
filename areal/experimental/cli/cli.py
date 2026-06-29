# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import click

from areal.version import __version__


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    help="AReaL operator CLI.",
)
@click.version_option(__version__, prog_name="areal")
def cli() -> None:
    pass


# Subcommand groups (inf / train / agent / ...) attach themselves to ``cli``
# from their own modules — see e.g. ``areal.experimental.cli.inference``,
# ``areal.experimental.cli.training``, ``areal.experimental.cli.agent``.
