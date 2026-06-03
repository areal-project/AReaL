# SPDX-License-Identifier: Apache-2.0

"""``areal logs`` — tail a training job's stdout/stderr (scaffold)."""

from __future__ import annotations

import argparse

_DESCRIPTION = """\
Tail a training job's combined stdout/stderr.

NO BEHAVIOR YET. Reserves the `areal logs` command name.

Planned flags:
  run_name               Name of the run (positional, required).
  -f, --follow           Stream new log lines as they arrive.
  -n, --lines N          Number of recent lines to print initially (default: 200).

Log files live under ~/.areal/runs/<name>/.
"""


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "logs",
        help="Tail a training job's stdout/stderr (scaffold).",
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.set_defaults(func=_handle)


def _handle(_: argparse.Namespace) -> int:
    return 0
