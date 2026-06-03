# SPDX-License-Identifier: Apache-2.0

"""``areal start`` — spawn a detached training driver process (scaffold)."""

from __future__ import annotations

import argparse

_DESCRIPTION = """\
Spawn a detached training driver process (background).

NO BEHAVIOR YET. Reserves the `areal start` command name.

Planned flags:
  --config PATH          Training YAML config (required).
  --name NAME            Override run name.
  --driver MOD:FUNC      Override driver entry.
  overrides...           Hydra-style overrides forwarded to the driver.

Use `areal run` for foreground execution. Use `areal ps`, `areal status`,
`areal logs`, and `areal stop` to manage the resulting background job.

State lives under ~/.areal/runs/.
"""


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "start",
        help="Spawn a detached training driver process (scaffold).",
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.set_defaults(func=_handle)


def _handle(_: argparse.Namespace) -> int:
    return 0
