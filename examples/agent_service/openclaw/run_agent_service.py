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
        python examples/agent_service/openclaw/run_agent_service.py

The launcher boots one Worker+DataProxy pair behind a Gateway, then
drops into an interactive prompt.  Each user message becomes one turn
of the OpenClaw conversation; the per-session OpenClaw subprocess drives
its configured upstream LLM internally.

Self-evolution (optional)
-------------------------
Pass ``--use-areal-inference --inf-base-url http://<inf-gateway>`` to route the
OpenClaw subprocess's LLM calls through AReaL's own inference service instead of
the env upstream.  The Agent Service is fully decoupled from the training side,
so **this example** mints the per-session ``sk-sess-*`` itself by POSTing to the
inference gateway's ``/rl/start_session`` (with ``--inf-admin-key``) before the
loop starts.  Each turn then carries the ``inf_base_url`` / ``inf_model`` /
``session_api_key`` fields straight through to the agent (their presence opts the
turn into self-evolution), so its LLM calls flow through the inference service
under that key and the trajectory is captured for training.  Set the reward
yourself afterwards by POSTing to the inference service's ``/rl/set_reward`` with
the same ``sk-sess-*``.  Requires a running AReaL inference gateway.
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
from areal.v2.agent_service.controller import AgentController


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


def _start_inference_session(
    inf_base_url: str, inf_admin_key: str, task_id: str
) -> str:
    """Mint a per-session ``sk-sess-*`` on the inference gateway.

    The Agent Service never talks to the training side, so the caller (this
    example) obtains the session key itself via ``/rl/start_session`` and then
    forwards it on every turn.  Returns the ``session_api_key``.
    """
    resp = httpx.post(
        f"{inf_base_url}/rl/start_session",
        headers={"Authorization": f"Bearer {inf_admin_key}"},
        json={"task_id": task_id},
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()["api_key"]


async def interactive_loop(
    gateway_addr: str,
    admin_key: str,
    inference: dict[str, str] | None = None,
) -> None:
    session_key = f"openclaw-{int(time.time())}"
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
                "model": "openclaw-agent",
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
    parser.add_argument(
        "--use-areal-inference",
        action="store_true",
        help="Route the agent's LLM calls through AReaL's inference service "
        "(self-evolution) instead of the env upstream",
    )
    parser.add_argument(
        "--inf-base-url",
        default="",
        help="AReaL inference gateway base URL (required with --use-areal-inference)",
    )
    parser.add_argument(
        "--inf-admin-key",
        default="areal-admin-key",
        help="Admin key for the inference gateway's /rl/start_session",
    )
    parser.add_argument(
        "--inf-model",
        default="",
        help="Model id the agent requests from the inference service",
    )
    args = parser.parse_args()

    # A random admin key keeps the service usable on non-loopback binds
    # without the publicly-known default key (see validate_admin_api_key).
    admin_key = args.admin_api_key or secrets.token_hex(16)

    inference: dict[str, str] | None = None
    if args.use_areal_inference:
        if not args.inf_base_url:
            raise SystemExit("--use-areal-inference requires --inf-base-url")
        inf_base_url = args.inf_base_url.rstrip("/")
        # The Agent Service is decoupled from training: this example mints the
        # per-session key itself and forwards it on every turn.
        session_api_key = _start_inference_session(
            inf_base_url, args.inf_admin_key, task_id=f"openclaw-{int(time.time())}"
        )
        inference = {
            "inf_base_url": inf_base_url,
            "inf_model": args.inf_model,
            "session_api_key": session_api_key,
        }

    upstream_url = args.upstream_url
    upstream_key = os.environ.get("OPENCLAW_UPSTREAM_API_KEY", "")
    # The env upstream is the fallback when not self-evolving; when
    # --use-areal-inference is set the per-turn inference upstream takes over,
    # so the env upstream is optional.
    if not inference and (not upstream_url or not upstream_key):
        raise SystemExit(
            "OPENCLAW_UPSTREAM_BASE_URL and OPENCLAW_UPSTREAM_API_KEY must be "
            "set (export them before running), or pass --use-areal-inference."
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
        agent_cls_path="examples.agent_service.openclaw.openclaw.OpenClawAgent",
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

        asyncio.run(
            interactive_loop(
                ctrl.gateway_addr, admin_key=admin_key, inference=inference
            )
        )
    finally:
        print("\nShutting down ...")
        ctrl.destroy()
        print("Done.")


if __name__ == "__main__":
    main()
