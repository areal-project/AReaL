#!/usr/bin/env python3
"""Set reward for the last interaction in an active session.

Requires the session API key obtained from ``start_session.py``.

Usage:
    python set_reward.py http://host:port --api-key <KEY> --reward 1.0
    python set_reward.py http://host:port --api-key <KEY> --reward 0.0 --interaction-id cmpl_abc
"""

from __future__ import annotations

import argparse
import json
import os

import requests
from _fmt import die, dim, header, info, show_request, show_response, success


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Set reward for an interaction in an AReaL RL session"
    )
    parser.add_argument("gateway_url", help="Proxy gateway URL")
    parser.add_argument(
        "--api-key",
        required=True,
        help="Session API key from start_session.py",
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
