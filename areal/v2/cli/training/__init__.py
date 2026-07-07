# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import click

from areal.v2.cli.training.commands.run import run_cmd


@click.group(help="Run AReaL training experiments.")
def train() -> None:
    pass


train.add_command(run_cmd)
