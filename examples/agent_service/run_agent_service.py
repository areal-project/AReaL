# SPDX-License-Identifier: Apache-2.0

"""Launch the Agent Service with Claude Agent SDK.

Usage::

    python examples/agent_service/run_agent_service.py
    python examples/agent_service/run_agent_service.py

Requires::

    uv pip install claude-agent-sdk
    export ANTHROPIC_API_KEY=sk-...
"""

from __future__ import annotations

import argparse
import asyncio
import time

import httpx

from areal.api.cli_args import AgentConfig
from areal.experimental.agent_service.controller import AgentController


async def _wait_healthy(url: str, timeout: float = 60.0) -> None:
    async with httpx.AsyncClient() as client:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    return
            except httpx.ConnectError:
                pass
            await asyncio.sleep(0.5)
    raise TimeoutError(f"Service at {url} did not become healthy")


async def interactive_loop(gateway_addr: str, admin_key: str) -> None:
    session_key = f"session-{int(time.time())}"
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

            resp = await client.post(
                f"{gateway_addr}/v1/responses",
                json={
                    "input": [{"type": "message", "content": user_input}],
                    "model": "claude-agent",
                    "user": session_key,
                },
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
    parser = argparse.ArgumentParser(description="Agent Service — Claude Agent SDK")
    parser.add_argument(
        "--admin-api-key",
        default="areal-agent-admin",
        help="Admin API key for inter-service auth",
    )
    args = parser.parse_args()

    from areal.infra.scheduler.local import LocalScheduler

    scheduler = LocalScheduler(
        experiment_name="agent-service-demo",
        trial_name="run0",
        gpu_devices=[],
    )

    ctrl_config = AgentConfig(
        agent_cls_path="examples.agent_service.agent.ClaudeAgent",
        admin_api_key=args.admin_api_key,
    )
    ctrl = AgentController(config=ctrl_config, scheduler=scheduler)

    try:
        print("Initializing with 1 pair ...")
        ctrl.initialize()
        print(f"  Router:  {ctrl.router_addr}")
        print(f"  Gateway: {ctrl.gateway_addr}")
        print(f"  Pairs:   {len(ctrl.pairs)}")

        asyncio.run(_wait_healthy(f"{ctrl.gateway_addr}/health"))
        print("All services ready.\n")

        asyncio.run(interactive_loop(ctrl.gateway_addr, admin_key=args.admin_api_key))
    finally:
        print("\nShutting down ...")
        ctrl.destroy()
        print("Done.")


if __name__ == "__main__":
    main()
