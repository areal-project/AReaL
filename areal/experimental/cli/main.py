# SPDX-License-Identifier: Apache-2.0

"""``areal`` CLI top-level entry point."""

from __future__ import annotations

import argparse
import sys

# Import the version module directly rather than `import areal`, which pulls
# in the full infra package (aiohttp, torch, ...). The CLI should stay light.
from areal.version import __version__
from areal.experimental.cli.commands import agent as cmd_agent
from areal.experimental.cli.commands import inf as cmd_inf
from areal.experimental.cli.commands import run as cmd_run
from areal.experimental.cli.commands import train as cmd_train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="areal",
        description="AReaL command-line interface.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"areal {__version__}",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        metavar="COMMAND",
    )
    cmd_run.add_parser(subparsers)
    cmd_train.add_parser(subparsers)
    cmd_inf.add_parser(subparsers)
    cmd_agent.add_parser(subparsers)
    return parser


def cli(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 2
    result = func(args)
    return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":
    sys.exit(cli())
