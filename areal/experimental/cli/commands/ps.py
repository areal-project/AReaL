# SPDX-License-Identifier: Apache-2.0

"""``areal ps`` — list locally tracked training jobs (scaffold)."""

from __future__ import annotations

import argparse

_DESCRIPTION = """\
List locally tracked training jobs.

NO BEHAVIOR YET. Reserves the `areal ps` command name.

Planned flags:
  --json                 Emit machine-readable JSON.
  --all                  Include completed / failed runs (default: running only).

State lives under ~/.areal/runs/.
"""


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "ps",
        help="List locally tracked training jobs (scaffold).",
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.set_defaults(func=_handle)


def _handle(_: argparse.Namespace) -> int:
    return 0
