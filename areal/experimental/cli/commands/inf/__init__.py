# SPDX-License-Identifier: Apache-2.0

"""``areal inf`` — inference service operator console."""

from __future__ import annotations

import argparse


_DESCRIPTION = """\
Operate an inference service: gateway + router + optional model backends.

Implemented verbs:
  run    Launch the gateway+router stack (detached).

Planned (not yet implemented):
  stop / status / ps / register / deregister / models / logs

State lives under ~/.areal/inf/.
"""


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "inf",
        help="Operate an inference service.",
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="verb", required=True, metavar="VERB")

    from areal.experimental.cli.commands.inf import run as cmd_run

    cmd_run.add_parser(sub)
