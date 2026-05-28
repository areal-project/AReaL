#!/usr/bin/env python3
"""AReaL Online Training — Multi-Episode Demo with Key Reuse.

Demonstrates the single-key refresh mechanism:
  Episode 1: start_session (no api_key) → chat → set_reward
  Episode 2+: start_session(api_key=same_key) → chat → set_reward

The gateway auto-ends the previous session on refresh, so no explicit
end_session call is needed between episodes.

Usage:
    python demo_lifecycle.py http://host:port --admin-key sk-test123456
    python demo_lifecycle.py http://host:port --admin-key sk-test123456 --episodes 3
    ADMIN_KEY=my-key python demo_lifecycle.py http://host:port
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import requests
from _fmt import (
    BOLD,
    CYAN,
    DIM,
    GREEN,
    MAGENTA,
    RESET,
    arrow,
    die,
    error,
    header,
    info,
    show_request,
    show_response,
    success,
)

_step_counter = 0


def _step(title: str) -> None:
    global _step_counter
    _step_counter += 1
    print()
    print(
        f"{BOLD}{MAGENTA}──────────────────────────────────────────────────────────────{RESET}"
    )
    print(f"{BOLD}{MAGENTA}  Step {_step_counter}: {title}{RESET}")
    print(
        f"{BOLD}{MAGENTA}──────────────────────────────────────────────────────────────{RESET}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AReaL Multi-Episode Demo (single-key with refresh)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "gateway_url",
        nargs="?",
        default="http://localhost:8090",
        help="Gateway URL (default: http://localhost:8090)",
    )
    parser.add_argument(
        "--admin-key",
        default=os.getenv("ADMIN_KEY", "areal-admin-key"),
        help="Admin API key (env: ADMIN_KEY)",
    )
    parser.add_argument(
        "--task-id",
        default=os.getenv("TASK_ID", "demo-task"),
        help="Task ID (env: TASK_ID)",
    )
    parser.add_argument(
        "--reward",
        default=os.getenv("REWARD", "1.0"),
        help="Reward value (env: REWARD)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("MODEL", "default"),
        help="Model name (env: MODEL)",
    )
    parser.add_argument(
        "--prompt",
        default=os.getenv("PROMPT", "Solve step by step: What is 12 * 15 + 3?"),
        help="Chat prompt (env: PROMPT)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=2,
        help="Number of episodes to run (default: 2)",
    )
    args = parser.parse_args()

    gateway_url = args.gateway_url
    admin_key = args.admin_key
    task_id = args.task_id
    reward_val = args.reward
    model = args.model
    prompt = args.prompt
    num_episodes = args.episodes

    # Banner
    header("AReaL Online Training — Multi-Episode Demo")
    print(f"  Gateway  : {CYAN}{gateway_url}{RESET}")
    print(f"  Task ID  : {CYAN}{task_id}{RESET}")
    print(f"  Model    : {CYAN}{model}{RESET}")
    print(f"  Prompt   : {CYAN}{prompt}{RESET}")
    print(f"  Reward   : {CYAN}{reward_val}{RESET}")
    print(f"  Episodes : {CYAN}{num_episodes}{RESET}")
    print()

    # ---- Health check --------------------------------------------------------
    _step("Health Check")
    show_request("GET", "health", "none", gateway_url)

    try:
        resp = requests.get(f"{gateway_url}/health", timeout=10)
    except requests.RequestException:
        error(f"Cannot reach gateway at {gateway_url}")
        info("Make sure the proxy gateway is running.")
        info(
            "Start online training first:  python examples/openclaw/train.py --config ..."
        )
        sys.exit(1)

    show_response(resp.status_code, resp.text)

    if resp.status_code != 200:
        die(f"Health check failed (HTTP {resp.status_code}). Is the gateway running?")

    try:
        worker_count = resp.json().get("workers", "?")
    except (ValueError, KeyError):
        worker_count = "?"
    success(f"Gateway healthy — {worker_count} backend worker(s)")

    # Build chat body once (reused for all episodes)
    chat_body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful math tutor. Show your work step by step.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 512,
    }

    # ---- Episodes Loop -------------------------------------------------------
    session_api_key: str | None = None
    session_id: str = ""
    episode_summaries = []
    for episode_num in range(1, num_episodes + 1):
        print()
        print()
        print(
            f"{BOLD}{CYAN}═══════════════════════════════════════════════════════════════{RESET}"
        )
        print(f"{BOLD}{CYAN}  EPISODE {episode_num} of {num_episodes}{RESET}")
        print(
            f"{BOLD}{CYAN}═══════════════════════════════════════════════════════════════{RESET}"
        )
        print()

        # ---- Start / Refresh Session ---
        is_refresh = session_api_key is not None
        if is_refresh:
            _step(f"Start Session (Refresh) — Episode {episode_num}")
            info(
                "Refreshing session with existing API key "
                "(auto-ends old session, waits for pipeline)."
            )
        else:
            _step("Start Session (New)")
            info(
                "Requesting a new RL session (admin auth → gateway routes to a worker)"
            )

        start_body: dict = {"task_id": task_id}
        if session_api_key is not None:
            start_body["api_key"] = session_api_key

        show_request("POST", "rl/start_session", "Bearer ***", gateway_url)
        print(f"  {DIM}Request body:{RESET}")
        for line in json.dumps(start_body, indent=2).split("\n"):
            print(f"  {DIM}{line}{RESET}")
        print()

        try:
            resp = requests.post(
                f"{gateway_url}/rl/start_session",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {admin_key}",
                },
                json=start_body,
                timeout=130 if is_refresh else 10,
            )
        except requests.RequestException as e:
            die(f"Failed to reach start_session endpoint: {e}")

        show_response(resp.status_code, resp.text)

        if resp.status_code != 200:
            die(
                "start_session failed. "
                "If HTTP 429, no capacity — the RL trainer hasn't granted capacity yet."
            )

        try:
            session_data = resp.json()
            session_api_key = session_data["api_key"]
            session_id = session_data["session_id"]
        except (ValueError, KeyError) as e:
            die(f"Failed to parse start_session response: {e}")

        if is_refresh:
            success(f"Session refreshed for Episode {episode_num}!")
            arrow(f"Session ID : {BOLD}{session_id}{RESET}")
            arrow(
                f"API Key    : {BOLD}{session_api_key[:12]}...{RESET} (same key, refreshed)"
            )
        else:
            success("Session started!")
            arrow(f"Session ID : {BOLD}{session_id}{RESET}")
            arrow(f"API Key    : {BOLD}{session_api_key[:12]}...{RESET}")
            print()
            info(
                f"This key will be reused across all {num_episodes} episodes. "
                "No reconfiguration needed."
            )
        print()

        # ---- Chat Completion ---
        _step(f"Chat Completion — Episode {episode_num}")
        info("Sending an OpenAI-compatible chat completion request")
        info("The model generates a response; tokens and logprobs are recorded for RL.")
        show_request("POST", "chat/completions", "Bearer ***", gateway_url)
        print()

        print(f"  {DIM}Request body:{RESET}")
        for line in json.dumps(chat_body, indent=2).split("\n"):
            print(f"  {DIM}{line}{RESET}")
        print()

        try:
            resp = requests.post(
                f"{gateway_url}/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {session_api_key}",
                },
                json=chat_body,
                timeout=30,
            )
        except requests.RequestException as e:
            die(f"Failed to reach chat/completions endpoint: {e}")

        show_response(resp.status_code, resp.text)

        if resp.status_code != 200:
            die(f"Chat completion failed (HTTP {resp.status_code}).")

        try:
            completion_data = resp.json()
            completion_id = completion_data.get("id", "")
            choices = completion_data.get("choices", [])
            assistant_msg = (
                choices[0].get("message", {}).get("content", "(empty)")
                if choices
                else "(no choices)"
            )
        except (ValueError, KeyError):
            completion_id = ""
            assistant_msg = "(parse error)"

        success("Completion received!")
        arrow(f"Completion ID: {BOLD}{completion_id}{RESET}")
        print()
        print(f"  {BOLD}{GREEN}Assistant response:{RESET}")
        for line in assistant_msg.split("\n"):
            print(f"  {CYAN}│{RESET} {line}")
        print()

        # ---- Set Reward ---
        _step(f"Set Reward — Episode {episode_num}")
        info(f"Assigning reward={BOLD}{reward_val}{RESET} to the last interaction.")
        info("This is how external evaluators provide training signal back to AReaL.")
        show_request("POST", "rl/set_reward", "Bearer ***", gateway_url)
        print()

        reward_body = {"reward": float(reward_val)}
        print(f"  {DIM}Request body: {json.dumps(reward_body)}{RESET}")
        print()

        try:
            resp = requests.post(
                f"{gateway_url}/rl/set_reward",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {session_api_key}",
                },
                json=reward_body,
                timeout=10,
            )
        except requests.RequestException as e:
            die(f"Failed to reach set_reward endpoint: {e}")

        show_response(resp.status_code, resp.text)

        if resp.status_code != 200:
            die(f"set_reward failed (HTTP {resp.status_code}).")

        success(f"Reward {reward_val} applied to interaction")
        print()

        episode_summaries.append(
            f"Episode {episode_num}: completion_id={completion_id}, reward={reward_val}"
        )

    # ---- Summary ----------------------------------------------------------------
    header(f"Demo Complete ✔ ({num_episodes} episodes)")
    print()
    print(
        f"  {BOLD}API Key{RESET}: {session_api_key[:12]}... (reused across all episodes)"
    )
    print()
    for summary in episode_summaries:
        print(f"    {GREEN}✔{RESET} {summary}")
    print()
    print(f"  {BOLD}Each episode:{RESET}")
    print(
        f"    {GREEN}1.{RESET} {BOLD}start_session{RESET}    → New session (ep 1) or refresh (ep 2+)"
    )
    print(
        f"    {GREEN}2.{RESET} {BOLD}chat/completions{RESET} → Model response (tokens recorded for RL)"
    )
    print(
        f"    {GREEN}3.{RESET} {BOLD}set_reward{RESET}       → External evaluator assigned reward"
    )
    print()
    info(
        "The last session's trajectory is exported on the next refresh "
        "or when training shuts down."
    )
    print()


if __name__ == "__main__":
    main()
