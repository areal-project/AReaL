# SPDX-License-Identifier: Apache-2.0

"""``areal inf`` — inference service operator console (scaffold).

Drives an inference service (gateway + router + optional model backends)
for day-to-day operator and debugging work. No verbs are implemented in
this scaffold release; this module only reserves the ``areal inf``
command name and tells the user what is coming.
"""

from __future__ import annotations

import argparse


_DESCRIPTION = """\
Operate an inference service: gateway + router + optional model backends.

NO VERBS IMPLEMENTED YET. This namespace currently only reserves the
`areal inf ...` command surface.

Planned verb surface (flag matrices live in the design discussion issue):
  run          launch gateway + router (optionally with --model inline)
  stop         tear them down
  status       health for one service
  ps           list locally known services
  logs         show gateway / router / model logs

State lives under ~/.areal/inf/.
"""


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "inf",
        help="Operate an inference service (scaffold — no verbs yet).",
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.set_defaults(func=_handle)


def _handle(_: argparse.Namespace) -> int:
    return 0
