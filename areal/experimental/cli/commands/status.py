# SPDX-License-Identifier: Apache-2.0

"""``areal status`` — show status of one training job (scaffold)."""

from __future__ import annotations

import argparse

_DESCRIPTION = """\
Show status of one training job.

NO BEHAVIOR YET. Reserves the `areal status` command name.

Planned flags:
  run_name               Name of the run to inspect (positional, required).
  --json                 Emit machine-readable JSON.

State lives under ~/.areal/runs/.
"""


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "status",
        help="Show status of one training job (scaffold).",
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.set_defaults(func=_handle)


def _handle(_: argparse.Namespace) -> int:
    return 0
