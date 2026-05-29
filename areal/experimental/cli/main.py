# SPDX-License-Identifier: Apache-2.0

"""Top-level entry point for the ``areal`` console-script.

This module wires the four sub-CLI namespaces (`inf`, `agent`, `train`,
`weight-update`) into a single argparse tree. Each namespace lives under
``areal/experimental/cli/commands/<namespace>/`` and exports an
``add_parser(subparsers)`` function; this file imports them and registers
them. No verb behavior is implemented at this level.

The import path is kept deliberately light: only stdlib and the namespace
``__init__`` modules are touched here. Heavy dependencies (torch, ray,
megatron, sglang, vllm, fastapi, …) must never appear on the import path
that ``areal --help`` triggers — the invariant is locked by
``tests/experimental/test_cli_lightness.py``.
"""

from __future__ import annotations

import argparse
import sys

from areal.experimental.cli.commands import agent as cmd_agent
from areal.experimental.cli.commands import inf as cmd_inf
from areal.experimental.cli.commands import train as cmd_train
from areal.experimental.cli.commands import weight_update as cmd_weight_update
from areal.version import __version__

_DESCRIPTION = """\
AReaL operator CLI for the v2 microservice architecture.

Each namespace drives one of the v2 services. Verbs land incrementally;
no verbs are implemented in this scaffold release — each namespace's
--help describes its planned surface and points at the design discussion.

Namespaces:
  inf            Operate an inference service (gateway + router + models)
  agent          Operate an agent service (gateway + router + sessions)
  train          Submit and observe training jobs
  weight-update  Diagnose weight-sync state between train and inference

Run `areal <namespace> --help` for what's planned (and what's available
today). State files for each namespace live under ~/.areal/<namespace>/.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="areal",
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"areal {__version__}",
    )
    subparsers = parser.add_subparsers(
        dest="namespace",
        required=True,
        metavar="NAMESPACE",
    )
    cmd_inf.add_parser(subparsers)
    cmd_agent.add_parser(subparsers)
    cmd_train.add_parser(subparsers)
    cmd_weight_update.add_parser(subparsers)
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
