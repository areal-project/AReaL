# SPDX-License-Identifier: Apache-2.0

"""``areal agent`` — agent service operator console (scaffold).

Drives an agent service (gateway + router + N (worker, data-proxy) pairs)
for session-centric operator and debugging work. No verbs are implemented
in this scaffold release; this module only reserves the ``areal agent``
command name and tells the user what is coming.

The agent CLI is session-centric (not model-centric like ``areal inf``).
Sessions can negotiate an RL session key with a configured inference
service when they start, enabling online RL trajectory tracking.
"""

from __future__ import annotations

import argparse


_DESCRIPTION = """\
Operate an agent service: gateway + router + (worker, data-proxy) pairs.
Session-centric: the primary unit of interaction is an agent session,
not a model.

NO VERBS IMPLEMENTED YET. This namespace currently only reserves the
`areal agent ...` command surface.

Planned verb surface (flag matrices live in the design discussion issue):
  run             launch router + N pairs + gateway
  stop            tear them down
  status          health for one service
  ps              list locally known services
  logs            show gateway / router / worker / data-proxy logs

State lives under ~/.areal/agent/.
"""


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "agent",
        help="Operate an agent service (scaffold — no verbs yet).",
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.set_defaults(func=_handle)


def _handle(_: argparse.Namespace) -> int:
    return 0
