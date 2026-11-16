#!/usr/bin/env python3
"""Set reward for the last interaction in an active session.

Requires the per-session ``sk-sess-*`` key minted by ``start_session.py``.

Usage:
    python set_reward.py http://host:port --api-key <KEY> --reward 1.0
    python set_reward.py http://host:port --api-key <KEY> --reward 0.0 --interaction-id cmpl_abc
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import requests

# =============================================================================
# CLI formatting helpers (ANSI colors + output helpers)
# =============================================================================
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
GREEN = "\033[0;32m"
CYAN = "\033[0;36m"
RED = "\033[0;31m"
BLUE = "\033[0;34m"


def info(msg: str) -> None:
    print(f"  {CYAN}ℹ{RESET}  {msg}")


def success(msg: str) -> None:
    print(f"  {GREEN}✔{RESET}  {msg}")


def error(msg: str) -> None:
    print(f"  {RED}✘{RESET}  {msg}")


def dim(msg: str) -> None:
    print(f"  {DIM}{msg}{RESET}")


def die(msg: str) -> None:
    error(msg)
    sys.exit(1)


def show_request(method: str, path: str, auth_label: str, gateway_url: str) -> None:
    print(f"  {DIM}{method} {gateway_url}/{path}{RESET}")
    print(f"  {DIM}Auth: {auth_label}{RESET}")


def show_response(status_code: int, body: str) -> None:
    if 200 <= status_code < 300:
        print(f"  {GREEN}HTTP {status_code}{RESET}")
    else:
        print(f"  {RED}HTTP {status_code}{RESET}")
    if body:
        try:
            formatted = json.dumps(json.loads(body), indent=2)
            for line in formatted.split("\n"):
                print(f"  {DIM}{line}{RESET}")
        except (json.JSONDecodeError, ValueError):
            for line in body.split("\n"):
                print(f"  {DIM}{line}{RESET}")


def header(title: str) -> None:
    """Print a boxed header."""
    print()
    print(
        f"{BOLD}{BLUE}══════════════════════════════════════════════════════════════{RESET}"
    )
    print(f"{BOLD}{BLUE}  {title}{RESET}")
    print(
        f"{BOLD}{BLUE}══════════════════════════════════════════════════════════════{RESET}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Set reward for an interaction in an AReaL RL session"
    )
    parser.add_argument("gateway_url", help="Proxy gateway URL")
    parser.add_argument(
        "--api-key",
        required=True,
        help="Per-session sk-sess-* key from start_session.py",
    )
    parser.add_argument(
        "--reward",
        type=float,
        default=float(os.getenv("REWARD", "1.0")),
        help="Reward value (default: 1.0, env: REWARD)",
    )
    parser.add_argument(
        "--interaction-id",
        default=None,
        help="Specific interaction ID (default: last interaction)",
    )
    args = parser.parse_args()

    header("Set Reward")
    info(f"Assigning reward={args.reward} to the last interaction.")
    show_request("POST", "rl/set_reward", "Bearer ***", args.gateway_url)

    reward_body: dict = {"reward": args.reward}
    if args.interaction_id is not None:
        reward_body["interaction_id"] = args.interaction_id

    dim(f"Request body: {json.dumps(reward_body)}")

    try:
        resp = requests.post(
            f"{args.gateway_url}/rl/set_reward",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {args.api_key}",
            },
            json=reward_body,
            timeout=10,
        )
    except requests.RequestException as e:
        die(f"Failed to reach gateway: {e}")

    show_response(resp.status_code, resp.text)

    if resp.status_code != 200:
        die(f"set_reward failed (HTTP {resp.status_code}).")

    success(f"Reward {args.reward} applied")


if __name__ == "__main__":
    main()
