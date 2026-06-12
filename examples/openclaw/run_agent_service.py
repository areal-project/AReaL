# SPDX-License-Identifier: Apache-2.0

"""Launch the Agent Service with the OpenClaw Agent.

Mirrors examples/agent_service/run_agent_service.py but loads the
OpenClaw-backed AgentRunnable.  The worker spawns one OpenClaw Gateway
subprocess per session, so only the ``openclaw`` CLI on ``PATH`` is
required (``npm i -g openclaw``); no externally-running gateway.

Usage::

    OPENCLAW_UPSTREAM_BASE_URL=https://your-llm/v1 \\
    OPENCLAW_UPSTREAM_API_KEY=sk-... \\
    OPENCLAW_UPSTREAM_MODEL=claude-sonnet-4-6 \\
        python examples/openclaw/run_agent_service.py

The launcher boots one Worker+DataProxy pair behind a Gateway, then
drops into an interactive prompt.  Each user message becomes one turn
of the OpenClaw conversation; the per-session OpenClaw subprocess drives
its configured upstream LLM internally.

Training integration (Layer 2, not wired here yet)
--------------------------------------------------
Once the controller learns to mint per-session ``sk-sess-*`` keys via
the AReaL ProxyGateway, this script will additionally call
``/session/{key}/episode/start`` with a :class:`TrainingContext` before
the first turn, ``/session/{key}/reward`` at the end, and
``/session/{key}/episode/end`` to flush the trajectory.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import secrets
import tempfile
import time

import httpx

from areal.api.cli_args import AgentConfig, SchedulingSpec
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
    session_key = f"openclaw-{int(time.time())}"
    print(f"Session: {session_key}")
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
                    "model": "openclaw-agent",
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
    parser = argparse.ArgumentParser(description="Agent Service — OpenClaw")
    parser.add_argument(
        "--admin-api-key",
        default="",
        help="Admin API key for inter-service auth (random if omitted)",
    )
    parser.add_argument(
        "--fileroot",
        default=os.path.join(tempfile.gettempdir(), "areal-openclaw"),
        help="Working root for logs and name-resolve records",
    )
    parser.add_argument(
        "--upstream-url",
        default=os.environ.get("OPENCLAW_UPSTREAM_BASE_URL", ""),
        help="Upstream LLM base URL the OpenClaw subprocess routes to",
    )
    parser.add_argument(
        "--upstream-model",
        default=os.environ.get("OPENCLAW_UPSTREAM_MODEL", "default"),
        help="Upstream model id",
    )
    args = parser.parse_args()

    # A random admin key keeps the service usable on non-loopback binds
    # without the publicly-known default key (see validate_admin_api_key).
    admin_key = args.admin_api_key or secrets.token_hex(16)

    upstream_url = args.upstream_url
    upstream_key = os.environ.get("OPENCLAW_UPSTREAM_API_KEY", "")
    if not upstream_url or not upstream_key:
        raise SystemExit(
            "OPENCLAW_UPSTREAM_BASE_URL and OPENCLAW_UPSTREAM_API_KEY must be "
            "set (export them before running)."
        )

    from areal.infra.scheduler.local import LocalScheduler

    # LocalScheduler validates these paths exist before starting workers.
    os.makedirs(os.path.join(args.fileroot, "name_resolve"), exist_ok=True)
    scheduler = LocalScheduler(
        experiment_name="openclaw-agent-service",
        trial_name="run0",
        gpu_devices=[],
        fileroot=args.fileroot,
    )

    env_vars = {
        "OPENCLAW_UPSTREAM_BASE_URL": upstream_url,
        "OPENCLAW_UPSTREAM_API_KEY": upstream_key,
        "OPENCLAW_UPSTREAM_MODEL": args.upstream_model,
    }
    for passthrough in (
        "OPENCLAW_BIN",
        "OPENCLAW_UPSTREAM_API",
        "OPENCLAW_NODE_EXTRA_CA_CERTS",
        "OPENCLAW_TLS_INSECURE",
    ):
        if passthrough in os.environ:
            env_vars[passthrough] = os.environ[passthrough]

    ctrl_config = AgentConfig(
        agent_cls_path="areal.experimental.agent_service.runtimes.openclaw.OpenClawAgent",
        admin_api_key=admin_key,
        scheduling_spec=(SchedulingSpec(env_vars=env_vars),),
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

        asyncio.run(interactive_loop(ctrl.gateway_addr, admin_key=admin_key))
    finally:
        print("\nShutting down ...")
        ctrl.destroy()
        print("Done.")


if __name__ == "__main__":
    main()
