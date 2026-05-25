# SPDX-License-Identifier: Apache-2.0

"""``areal train`` — training driver service.

Thin wrapper around :mod:`areal.experimental.cli.runner`: ``train`` is just
``run`` with state-file tracking enabled and a 6-action surface that mirrors
``inf`` / ``agent`` shape (``run / start / stop / ps / status / logs``).

The CLI itself stays scheduler-agnostic. The driver decides how to dispatch
to LocalScheduler / SlurmScheduler / RayScheduler based on
``config.scheduler.type``.
"""

from __future__ import annotations

import argparse


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "train",
        help="Manage training driver runs (PPO / SFT / DPO / RW).",
        description=(
            "Launch a training driver (foreground or background), stop one, "
            "list locally tracked runs, inspect a single run's status, or "
            "tail its log."
        ),
    )
    sub = p.add_subparsers(dest="action", required=True, metavar="ACTION")

    from areal.experimental.cli.commands.train import logs as cmd_logs
    from areal.experimental.cli.commands.train import ps as cmd_ps
    from areal.experimental.cli.commands.train import run as cmd_run
    from areal.experimental.cli.commands.train import start as cmd_start
    from areal.experimental.cli.commands.train import status as cmd_status
    from areal.experimental.cli.commands.train import stop as cmd_stop

    cmd_run.add_parser(sub)
    cmd_start.add_parser(sub)
    cmd_stop.add_parser(sub)
    cmd_ps.add_parser(sub)
    cmd_status.add_parser(sub)
    cmd_logs.add_parser(sub)
