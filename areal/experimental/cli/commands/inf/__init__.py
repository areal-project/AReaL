# SPDX-License-Identifier: Apache-2.0

"""``areal inf`` — inference / rollout service."""

from __future__ import annotations

import argparse


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "inf",
        help="Manage inference services and models.",
        description=(
            "Manage an inference / rollout service: launch a gateway+router pair, "
            "stop it, inspect health, list services and models, or tail logs."
        ),
    )
    sub = p.add_subparsers(dest="action", required=True, metavar="ACTION")

    from areal.experimental.cli.commands.inf import chat as cmd_chat
    from areal.experimental.cli.commands.inf import collect as cmd_collect
    from areal.experimental.cli.commands.inf import deregister as cmd_deregister
    from areal.experimental.cli.commands.inf import logs as cmd_logs
    from areal.experimental.cli.commands.inf import models as cmd_models
    from areal.experimental.cli.commands.inf import ps as cmd_ps
    from areal.experimental.cli.commands.inf import register as cmd_register
    from areal.experimental.cli.commands.inf import run as cmd_run
    from areal.experimental.cli.commands.inf import status as cmd_status
    from areal.experimental.cli.commands.inf import stop as cmd_stop

    cmd_run.add_parser(sub)
    cmd_stop.add_parser(sub)
    cmd_status.add_parser(sub)
    cmd_ps.add_parser(sub)
    cmd_register.add_parser(sub)
    cmd_deregister.add_parser(sub)
    cmd_models.add_parser(sub)
    cmd_chat.add_parser(sub)
    cmd_collect.add_parser(sub)
    cmd_logs.add_parser(sub)
