# SPDX-License-Identifier: Apache-2.0

"""``areal agent`` — agent service and session management."""

from __future__ import annotations

import argparse


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "agent",
        help="Manage agent services and sessions.",
        description=(
            "Manage an agent service: launch its gateway/router/N pairs, "
            "stop it, inspect health, list services, manage sessions, "
            "send rewards, chat with the agent, or tail logs."
        ),
    )
    sub = p.add_subparsers(dest="action", required=True, metavar="ACTION")

    from areal.experimental.cli.commands.agent import chat as cmd_chat
    from areal.experimental.cli.commands.agent import logs as cmd_logs
    from areal.experimental.cli.commands.agent import new_session as cmd_new_session
    from areal.experimental.cli.commands.agent import ps as cmd_ps
    from areal.experimental.cli.commands.agent import reward as cmd_reward
    from areal.experimental.cli.commands.agent import run as cmd_run
    from areal.experimental.cli.commands.agent import status as cmd_status
    from areal.experimental.cli.commands.agent import stop as cmd_stop
    from areal.experimental.cli.commands.agent import switch_session as cmd_switch_session

    cmd_run.add_parser(sub)
    cmd_stop.add_parser(sub)
    cmd_status.add_parser(sub)
    cmd_ps.add_parser(sub)
    cmd_new_session.add_parser(sub)
    cmd_switch_session.add_parser(sub)
    cmd_reward.add_parser(sub)
    cmd_chat.add_parser(sub)
    cmd_logs.add_parser(sub)
