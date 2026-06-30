#!/usr/bin/env python3
"""Start an RL session on the proxy gateway.

Allocates a backend worker and prints the session API key that the user's
application needs for subsequent requests (chat/completions, set_reward, etc.).

Pass ``--api-key`` with a previously issued key to **refresh** an existing
session: the old session is automatically ended (with default reward 0 if no
reward was explicitly set), the trajectory is exported, and a new session is
started with the same key.

Usage:
    python start_session.py http://host:port --admin-key sk-test123456
    python start_session.py http://host:port --admin-key sk-test123456 --task-id my-task
    python start_session.py http://host:port --admin-key sk-test123456 --api-key <key>
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
    parser.add_argument("gateway_url", help="Proxy gateway URL")
    parser.add_argument(
        "--admin-key",
        default=os.getenv("ADMIN_KEY", "areal-admin-key"),
        help="Admin API key (env: ADMIN_KEY)",
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

    if resp.status_code != 200:
        if resp.status_code == 429 and is_refresh:
            die(
                "Refresh timed out — the training pipeline hasn't cycled yet. "
                "Retry in a few seconds."
            )
        die(
            "start_session failed. "
            "If HTTP 429, no capacity — the RL trainer hasn't granted capacity yet."
        )

    try:
        data = resp.json()
        session_api_key = data["api_key"]
        session_id = data["session_id"]
    except (ValueError, KeyError) as e:
        die(f"Failed to parse response: {e}")

    success("Session started!")
    arrow(f"Session ID : {BOLD}{session_id}{RESET}")
    arrow(f"API Key    : {BOLD}{session_api_key}{RESET}")
    print()
    info("Use this API key as your Bearer token for all subsequent requests.")
    if not is_refresh:
        info("Example with OpenAI SDK:")
        print()
        print(f"  export OPENAI_API_KEY={session_api_key}")
        print(f"  export OPENAI_BASE_URL={args.gateway_url}")
        print()
    info("To start the next episode with the same key:")
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
