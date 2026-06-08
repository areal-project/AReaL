# SPDX-License-Identifier: Apache-2.0

"""``areal inf`` — inference service operator console.

`run` spawns a detached supervisor that wraps a v2 ``RolloutControllerV2``
(which itself manages workers + router + gateway + data-proxies). `stop`
sends SIGTERM to the supervisor — its handler runs ``controller.destroy()``,
the same teardown path PPOTrainer uses on training completion.

Verbs:
  run     Launch the inference service (detached).
  stop    Stop a running service via supervisor SIGTERM.
  ps      List locally tracked services.
  status  Detail for a single service.
  logs    Tail the supervisor log.

State lives under ~/.areal/inf/.
"""

from __future__ import annotations

import argparse


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "inf",
        help="Operate an inference service.",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="verb", required=True, metavar="VERB")

    from areal.experimental.cli.commands.inf import (
        logs as cmd_logs,
        ps as cmd_ps,
        run as cmd_run,
        status as cmd_status,
        stop as cmd_stop,
    )

    cmd_run.add_parser(sub)
    cmd_stop.add_parser(sub)
    cmd_ps.add_parser(sub)
    cmd_status.add_parser(sub)
    cmd_logs.add_parser(sub)
