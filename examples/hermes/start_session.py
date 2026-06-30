#!/usr/bin/env python3
"""Start an RL session on AReaL's inference gateway.

Mints a per-session ``sk-sess-*`` key on the inference gateway that
``train.py`` embeds, then prints it. You forward that key to the Hermes
Agent Service (``hermes_loop.py``) so the agent's LLM calls flow through the
inference service under it and get captured as a training trajectory; you then
score it with ``set_reward.py``.

The inference gateway's admin key is whatever you passed as
``rollout.admin_api_key`` to ``train.py``.

Pass ``--api-key`` with a previously issued key to **refresh** an existing
session: the old session is ended (default reward 0 if none was set), its
trajectory is exported, and a new session is started reusing the same key.

Usage:
    python start_session.py http://host:port --admin-key sk-xxx
    python start_session.py http://host:port --admin-key sk-xxx --task-id my-task
    python start_session.py http://host:port --admin-key sk-xxx --api-key <key>
"""

from __future__ import annotations

import argparse
import os
import sys

import requests
from _fmt import (
    BOLD,
    RESET,
    arrow,
    die,
    header,
    info,
    show_request,
    show_response,
    success,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Start an AReaL RL session")
    parser.add_argument("gateway_url", help="Inference gateway URL")
    parser.add_argument(
        "--admin-key",
        default=os.getenv("ADMIN_KEY", "areal-admin-key"),
        help="Inference gateway admin key (== rollout.admin_api_key; env: ADMIN_KEY)",
    )
    parser.add_argument(
        "--task-id",
        default=os.getenv("TASK_ID", "demo-task"),
        help="Task identifier (env: TASK_ID)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("SESSION_API_KEY"),
        help="Reuse a previously issued API key (refresh). (env: SESSION_API_KEY)",
    )
    args = parser.parse_args()

    is_refresh = args.api_key is not None
    header("Refresh Session" if is_refresh else "Start Session")
    if is_refresh:
        info(
            "Refreshing: end old session → export trajectory → start new session (same key)"
        )
    else:
        info("Requesting a new RL session (admin auth → gateway routes to a worker)")
    show_request("POST", "rl/start_session", "Bearer ***", args.gateway_url)

    try:
        resp = requests.post(
            f"{args.gateway_url}/rl/start_session",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {args.admin_key}",
            },
            json={"task_id": args.task_id, "api_key": args.api_key},
            timeout=130 if is_refresh else 10,
        )
    except requests.RequestException as e:
        die(f"Failed to reach gateway: {e}")

    show_response(resp.status_code, resp.text)

    if resp.status_code != 201:
        if resp.status_code == 429 and is_refresh:
            die(
                "Refresh timed out — the training pipeline hasn't cycled yet. "
                "Retry in a few seconds."
            )
        die(
            "start_session failed. "
            "If HTTP 429, no capacity — the RL trainer hasn't granted capacity yet."
        )

    # v2 inference gateway returns 201 with a flat list of session credentials:
    #   {"group_id": ..., "sessions": [{"session_id": ..., "session_api_key": ...}]}
    try:
        data = resp.json()
        session = data["sessions"][0]
        session_api_key = session["session_api_key"]
        session_id = session["session_id"]
    except (ValueError, KeyError, IndexError) as e:
        die(f"Failed to parse response: {e}")

    success("Session started!")
    arrow(f"Session ID : {BOLD}{session_id}{RESET}")
    arrow(f"API Key    : {BOLD}{session_api_key}{RESET}")
    print()
    info("Forward this key to the Hermes Agent Service to capture the trajectory:")
    print()
    print(
        f"  python hermes_loop.py <agent-gateway>"
        f" --admin-api-key <agent-admin-key>"
        f" --inf-base-url {args.gateway_url}"
        f" --session-api-key {session_api_key}"
    )
    print()
    info("Then score the episode with the same key:")
    print()
    print(
        f"  python set_reward.py {args.gateway_url}"
        f" --api-key {session_api_key} --reward 1.0"
    )
    print()
    info("To start the next episode reusing the same key:")
    print()
    print(
        f"  python start_session.py {args.gateway_url}"
        f" --admin-key {args.admin_key} --api-key {session_api_key}"
    )
    print()

    # Machine-readable output on stderr for scripting
    print(f"SESSION_API_KEY={session_api_key}", file=sys.stderr)
    print(f"SESSION_ID={session_id}", file=sys.stderr)


if __name__ == "__main__":
    main()
