# SPDX-License-Identifier: Apache-2.0

"""``areal train stop`` — stop a training driver by name.

SIGTERMs the driver's process group, escalates to SIGKILL after the timeout,
and marks the state file ``status=stopped``. Refuses to stop runs started
under a different command (e.g. ``areal inf start``).
"""

from __future__ import annotations

import argparse

from areal.experimental.cli.runner import stop_run


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "stop",
        help="Stop a running training driver by name.",
        description=(
            "Send SIGTERM to the recorded process group, then SIGKILL after "
            "the timeout. Updates the state file's status field."
        ),
    )
    p.add_argument(
        "run_name",
        help="Run name (typically experiment_name/trial_name).",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="Seconds to wait for SIGTERM before sending SIGKILL (default: 15).",
    )
    p.set_defaults(func=_handle)


def _handle(args: argparse.Namespace) -> int:
    return stop_run(args.run_name, command_hint="train", timeout=args.timeout)
