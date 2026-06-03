# SPDX-License-Identifier: Apache-2.0

"""``areal stop`` — signal a running training job by name (scaffold)."""

from __future__ import annotations

import argparse

_DESCRIPTION = """\
Signal a running training job by name (SIGTERM, escalating to SIGKILL).

NO BEHAVIOR YET. Reserves the `areal stop` command name.

Planned flags:
  run_name               Name of the run to stop (positional, required).
  --timeout SECONDS      Grace period before SIGKILL (default: 15).

State lives under ~/.areal/runs/.
"""


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "stop",
        help="Signal a running training job by name (scaffold).",
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.set_defaults(func=_handle)


def _handle(_: argparse.Namespace) -> int:
    return 0
