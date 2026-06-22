# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import click

from areal.experimental.cli.inference.common import print_services


@click.command(name="ps", help="List locally known inference services.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON.")
@click.option("--all", "include_all", is_flag=True, help="Include stale services.")
def ps_cmd(as_json: bool, include_all: bool) -> None:
    raise SystemExit(do_ps(as_json, include_all) or 0)


def do_ps(as_json: bool, include_all: bool) -> int:
    return print_services(as_json=as_json, include_all=include_all)
