# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import click

from areal.experimental.cli.inference.common import print_models, running_state


@click.command(name="models", help="List registered models.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON.")
def models_cmd(as_json: bool) -> None:
    raise SystemExit(do_models(as_json) or 0)


def do_models(as_json: bool) -> int:
    state = running_state()
    if state is None:
        click.echo("[]" if as_json else "daemon not running")
        return 0
    return print_models(state, as_json)
