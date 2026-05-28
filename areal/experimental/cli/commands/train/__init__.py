# SPDX-License-Identifier: Apache-2.0

"""``areal train`` — training job submitter (scaffold).

Wraps the launch lifecycle of a training driver. Unlike ``areal inf`` /
``areal agent`` (which manage long-running services), ``areal train``
treats each run as a job: it terminates, and the CLI's job is purely
lifecycle wrapping. The scheduling decision (local / slurm / ray) stays
inside the driver, decided by ``config.scheduler.type`` — the CLI does
not pick a scheduler.

No verbs are implemented in this scaffold release.
"""

from __future__ import annotations

import argparse


_DESCRIPTION = """\
Submit and observe training jobs. Job-shaped (terminates), not
service-shaped — the CLI wraps lifecycle only and does not choose the
scheduler (that decision lives in the driver via config.scheduler.type).

NO VERBS IMPLEMENTED YET. This namespace currently only reserves the
`areal train ...` command surface.

Planned verb surface (flag matrices live in the design discussion issue):
  run      run a driver in the foreground (small jobs, debugging)
  start    spawn a detached driver process (cluster jobs)
  stop     signal a running job by name
  ps       list locally tracked jobs
  status   status of one job
  logs     tail a job's combined stdout/stderr

State lives under ~/.areal/train/.
"""


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "train",
        help="Submit and observe training jobs (scaffold — no verbs yet).",
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.set_defaults(func=_handle)


def _handle(_: argparse.Namespace) -> int:
    return 0
