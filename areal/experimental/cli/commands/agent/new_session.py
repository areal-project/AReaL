# SPDX-License-Identifier: Apache-2.0

"""``areal agent new_session`` — create a new agent session.

Steps (design §11.6):

1. Generate or accept the session key.
2. If the service has an inference backend configured, attempt
   ``POST /rl/start_session`` on the inference gateway to negotiate an RL
   session API key. The actual key is never persisted in plaintext — only an
   ``rl_session_key_present`` flag is stored.
3. Record the session locally; promote it to default unless ``--no-switch``.
"""

from __future__ import annotations

import argparse
import json
import os

from areal.experimental.cli.agent_sessions import (
    SessionEntry,
    SessionRegistry,
    generate_session_key,
)
from areal.experimental.cli.agent_state import (
    AgentServiceState,
    resolve_agent_service_name,
)
from areal.experimental.cli.gateway_client import (
    GatewayClient,
    GatewayError,
)


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "new_session",
        help="Start a new agent session and negotiate /rl/start_session.",
        description=(
            "Create a local session record and (if the inference backend is "
            "configured) ask it to start a new RL session. The session key "
            "is used as the `user` field for subsequent /v1/responses calls."
        ),
    )
    p.add_argument(
        "session_key",
        nargs="?",
        default=None,
        help="Optional explicit session key (auto-generated if omitted).",
    )
    p.add_argument(
        "--service",
        default=None,
        help="Target service instance.",
    )
    p.add_argument(
        "--no-switch",
        action="store_true",
        help="Do not promote the new session to default after creation.",
    )
    p.add_argument(
        "--session-timeout",
        type=float,
        default=1800.0,
        help="Inactivity timeout for this session (seconds).",
    )
    p.add_argument(
        "--inf-api-key",
        default=os.environ.get("AREAL_INF_API_KEY", None),
        help=(
            "Inference admin API key (env: AREAL_INF_API_KEY). "
            "Required for /rl/start_session if the service has an inf-addr."
        ),
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Inference request timeout (s).",
    )
    p.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    p.set_defaults(func=_handle)


def _handle(args: argparse.Namespace) -> int:
    name = resolve_agent_service_name(args.service)
    try:
        state = AgentServiceState.load(name)
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from e

    key = args.session_key or generate_session_key()
    reg = SessionRegistry.load(name)
    if reg.get(key) is not None:
        raise SystemExit(f"Session {key!r} already exists for service {name!r}.")

    rl_present = False
    warning: str | None = None
    if state.inf_addr:
        try:
            client = GatewayClient(
                state.inf_addr, admin_api_key=args.inf_api_key, timeout=args.timeout
            )
            resp = client.start_rl_session(model=state.inf_model)
            sessions = resp.get("sessions") if isinstance(resp, dict) else None
            if sessions and isinstance(sessions, list):
                first = sessions[0] if sessions else {}
                rl_present = bool(first.get("api_key"))
            if not rl_present:
                warning = (
                    f"Inference at {state.inf_addr} accepted /rl/start_session "
                    f"but did not return a session api_key."
                )
        except GatewayError as e:
            warning = (
                f"Inference at {state.inf_addr} did not accept /rl/start_session: "
                f"{e}. Using original API key."
            )

    entry = SessionEntry(
        key=key,
        active=True,
        rl_session_key_present=rl_present,
        session_timeout=args.session_timeout,
    )
    reg.add(entry)
    if not args.no_switch:
        reg.default_session = key
    reg.save()

    payload = {
        "service": name,
        "session": key,
        "rl_session_key_present": rl_present,
        "default": reg.default_session == key,
        "warning": warning,
    }

    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(f"Created session {key!r} on service {name!r}.")
        print(f"  rl-key: {'yes' if rl_present else 'no'}")
        print(f"  default: {'yes' if reg.default_session == key else 'no'}")
        if warning:
            print(f"  WARNING: {warning}")
    return 0
