# SPDX-License-Identifier: Apache-2.0

"""``areal agent switch_session`` — change the current default session."""

from __future__ import annotations

import argparse
import json

from areal.experimental.cli.agent_sessions import SessionRegistry
from areal.experimental.cli.agent_state import resolve_agent_service_name


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "switch_session",
        help="Switch the current default session.",
        description=(
            "Change the current default session used by `areal agent chat` "
            "and `areal agent reward` when no explicit session key is given. "
            "Local-only operation; does not touch the agent gateway."
        ),
    )
    p.add_argument("session_key", help="Existing active session key to promote.")
    p.add_argument(
        "--service",
        default=None,
        help="Target service instance.",
    )
    p.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    p.set_defaults(func=_handle)


def _handle(args: argparse.Namespace) -> int:
    name = resolve_agent_service_name(args.service)
    reg = SessionRegistry.load(name)
    entry = reg.get(args.session_key)
    if entry is None:
        raise SystemExit(
            f"Session {args.session_key!r} unknown on service {name!r}. "
            f"Create one with `areal agent new_session {args.session_key}`."
        )
    if not entry.active:
        raise SystemExit(
            f"Session {args.session_key!r} on service {name!r} is not active."
        )
    reg.default_session = args.session_key
    reg.save()
    payload = {"service": name, "default_session": args.session_key}
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Default session for {name!r} -> {args.session_key!r}.")
    return 0
