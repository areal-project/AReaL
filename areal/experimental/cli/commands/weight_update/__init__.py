# SPDX-License-Identifier: Apache-2.0

"""``areal weight-update`` — weight-sync diagnostic console (scaffold).

Drives the weight-update service that sits between training and
inference. The operator-facing surface is small and diagnostic-only:
humans don't invoke `/connect` / `/update_weights` / `/disconnect`
directly during normal use — those are called by adapter code inside
the training and inference engines. The CLI's job is to show whether
the sync is healthy, which (train, inference) pairs are connected, and
where the logs are.

No verbs are implemented in this scaffold release.

The CLI-surface namespace is ``weight-update`` (hyphenated, matching
the v2 service naming). The Python module is ``weight_update`` because
identifiers can't contain hyphens.
"""

from __future__ import annotations

import argparse

_DESCRIPTION = """\
Diagnose the weight-update service that bridges training and inference.

NO VERBS IMPLEMENTED YET. This namespace currently only reserves the
`areal weight-update ...` command surface.

Planned verb surface (flag matrices live in the design discussion issue):
  status      is the gateway alive? how many pairs are connected?
  ps          list locally known weight-update services
  logs        tail the gateway log

Note: there is no `run` verb in the first cut — in the v2 flow the
weight-update gateway is brought up by the training-side controller,
not by the operator.

State lives under ~/.areal/weight-update/.
"""


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "weight-update",
        help="Diagnose weight-sync state (scaffold — no verbs yet).",
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.set_defaults(func=_handle)


def _handle(_: argparse.Namespace) -> int:
    return 0
