# SPDX-License-Identifier: Apache-2.0

"""``areal agent reward`` — send /rl/set_reward to the inference backend.

The agent service is a *client* of the inference service for RL coordination
(design §10.2). This command does not require the agent service itself to
be running — only that the local agent service state knows where the
inference gateway is.
"""

from __future__ import annotations

import argparse
import json
import os
import time

from areal.experimental.cli.agent_sessions import (
    SessionRegistry,
    resolve_session_key,
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
        "reward",
        help="Send /rl/set_reward to the inference service for a session.",
        description=(
            "POST /rl/set_reward to the inference gateway recorded for the "
            "target agent service. Session resolution: explicit > default > "
            "sole-active > error."
        ),
    )
    p.add_argument("value", type=float, help="Reward value to send.")
    p.add_argument(
        "session_key",
        nargs="?",
        default=None,
        help="Target session key (defaults to current default session).",
    )
    p.add_argument(
        "--service",
        default=None,
        help="Target service instance.",
    )
    p.add_argument(
        "--inf-api-key",
        default=os.environ.get("AREAL_INF_API_KEY", None),
        help="Inference admin API key (env: AREAL_INF_API_KEY).",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=10.0,
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

    if not state.inf_addr:
        raise SystemExit(
            f"Agent service {name!r} has no inference backend configured "
            f"(launch with --inf-addr to enable RL coordination)."
        )

    session_key = resolve_session_key(name, args.session_key)

    client = GatewayClient(
        state.inf_addr, admin_api_key=args.inf_api_key, timeout=args.timeout
    )
    try:
        resp = client.set_rl_reward(
            session_id=session_key,
            reward=args.value,
            model=state.inf_model or None,
        )
    except GatewayError as e:
        warn = (
            f"Inference at {state.inf_addr} did not accept /rl/set_reward: {e}. "
            f"The agent session remains active, but no reward was recorded."
        )
        if args.json:
            print(json.dumps({
                "ok": False,
                "service": name,
                "session": session_key,
                "reward": args.value,
                "error": str(e),
            }, indent=2))
        else:
            print(f"Warning: {warn}")
        return 1

    reg = SessionRegistry.load(name)
    entry = reg.get(session_key)
    if entry is not None:
        entry.last_reward = args.value
        entry.last_active_at = time.time()
        reg.save()

    payload = {
        "ok": True,
        "service": name,
        "session": session_key,
        "reward": args.value,
        "response": resp,
    }
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(f"Recorded reward {args.value} for session {session_key!r} on {name!r}.")
    return 0
