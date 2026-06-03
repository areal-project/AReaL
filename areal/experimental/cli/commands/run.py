# SPDX-License-Identifier: Apache-2.0

"""``areal run`` — launch a training driver in the foreground (scaffold).

Training is AReaL's primary use case, so the verbs that drive a training
job (`run`, `start`, `stop`, `ps`, `status`, `logs`) sit at the top
level instead of being nested under a `train` namespace. The service-side
operator surfaces (inference / agent / weight-update) stay namespaced.

No verb behavior is implemented in this scaffold release; this module
only reserves the ``areal run`` command name.
"""

from __future__ import annotations

import argparse

_DESCRIPTION = """\
Launch a training driver in the foreground.

NO BEHAVIOR YET. Reserves the `areal run` command name.

Planned flags:
  --config PATH          Training YAML config (required).
  --name NAME            Override run name (default: derived from yaml).
  --driver MOD:FUNC      Override driver entry (default: yaml `driver:` field).
  overrides...           Hydra-style overrides forwarded to the driver.

Companion verbs (also top-level):
  areal start    Spawn a detached driver process (background).
  areal stop     Signal a running job by name.
  areal ps       List locally tracked jobs.
  areal status   Status of one job.
  areal logs     Tail a job's combined stdout/stderr.

State lives under ~/.areal/runs/.
"""


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "run",
        help="Launch a training driver in the foreground (scaffold).",
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.set_defaults(func=_handle)


def _handle(_: argparse.Namespace) -> int:
    return 0
