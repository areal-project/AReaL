# SPDX-License-Identifier: Apache-2.0

"""Interactive conversation loop against a running Hermes Agent Service.

This is the standalone chat window: it assumes the Agent Service is **already
running** (started separately with ``areal agent run``) and a per-session
``sk-sess-*`` key has **already been minted** on the inference gateway (with
``start_session.py``). It only opens a ``You:`` prompt and drives the agent.

Each user message becomes one turn of the Hermes conversation, posted to the
agent gateway's ``/v1/responses``. When ``--inf-base-url`` and
``--session-api-key`` are supplied, every turn forwards the inference-routing
fields so the agent's LLM calls flow through AReaL's inference service under
that key and the trajectory is captured for training; score it afterwards with
``set_reward.py``.

Usage::

    # plain chat (env upstream; agent admin key authorizes the gateway)
    python hermes_loop.py http://<agent-gateway> --admin-api-key <agent-admin-key>

    # self-evolution: capture the trajectory under a session key
    python hermes_loop.py http://<agent-gateway> --admin-api-key <agent-admin-key> \\
        --inf-base-url http://<inference-gateway> --session-api-key sk-sess-xxxx
"""

from __future__ import annotations

import argparse
import asyncio
import time

import httpx


async def interactive_loop(
    gateway_addr: str,
    admin_key: str,
    inference: dict[str, str] | None = None,
) -> None:
    session_key = f"hermes-{int(time.time())}"
    print(f"Session: {session_key}")
    if inference:
        print(f"Self-evolution: routing LLM calls via {inference['inf_base_url']}")
    print("Type your message (or 'quit' to exit):\n")

    async with httpx.AsyncClient(timeout=120.0) as client:
        while True:
            try:
                user_input = input("You: ")
            except (EOFError, KeyboardInterrupt):
                break
            if user_input.strip().lower() in ("quit", "exit", "q"):
                break
            if not user_input.strip():
                continue

            payload: dict[str, object] = {
                "input": [{"type": "message", "content": user_input}],
                "model": "hermes-agent",
                "user": session_key,
            }
            if inference:
                payload.update(inference)

            resp = await client.post(
                f"{gateway_addr}/v1/responses",
                json=payload,
                headers={"Authorization": f"Bearer {admin_key}"},
            )
            data = resp.json()

            if data.get("status") == "completed":
                for item in data.get("output", []):
                    if item.get("type") == "message":
                        for block in item.get("content", []):
                            if block.get("type") == "output_text":
                                print(f"Agent: {block['text']}")
                    elif item.get("type") == "function_call":
                        print(f"[tool] {item.get('name', '')}")
                print()
            elif data.get("error"):
                print(f"Error: {data['error'].get('message', '')[:200]}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive chat against a running Hermes Agent Service"
    )
    parser.add_argument(
        "gateway_addr",
        help="Agent Service gateway URL (from 'areal agent status')",
    )
    parser.add_argument(
        "--admin-api-key",
        required=True,
        help="Agent Service admin key (the --admin-api-key passed to 'areal agent run')",
    )
    parser.add_argument(
        "--inf-base-url",
        default="",
        help="Inference gateway base URL (enables self-evolution capture)",
    )
    parser.add_argument(
        "--inf-model",
        default="",
        help="Model id the agent requests from the inference service",
    )
    parser.add_argument(
        "--session-api-key",
        default="",
        help="Per-session sk-sess-* key from start_session.py (required for capture)",
    )
    args = parser.parse_args()

    gateway_addr = args.gateway_addr.rstrip("/")

    inference: dict[str, str] | None = None
    if args.inf_base_url or args.session_api_key:
        if not (args.inf_base_url and args.session_api_key):
            raise SystemExit(
                "self-evolution requires BOTH --inf-base-url and --session-api-key"
            )
        inference = {
            "inf_base_url": args.inf_base_url.rstrip("/"),
            "inf_model": args.inf_model,
            "session_api_key": args.session_api_key,
        }

    asyncio.run(
        interactive_loop(
            gateway_addr, admin_key=args.admin_api_key, inference=inference
        )
    )


if __name__ == "__main__":
    main()
