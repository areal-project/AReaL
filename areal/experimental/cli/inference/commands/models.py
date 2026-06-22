# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import click

from areal.experimental.cli.inference.common import print_models, running_state


@click.command(name="models", help="List registered models.")
@click.option("--service", default=None, help="Target service instance.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON.")
def models_cmd(service: str | None, as_json: bool) -> None:
    raise SystemExit(do_models(as_json, service=service) or 0)


def do_models(as_json: bool, *, service: str | None = None) -> int:
    state = running_state(service)
    if state is None:
        click.echo("[]" if as_json else "service not running")
        return 0
    return print_models(state, as_json)
