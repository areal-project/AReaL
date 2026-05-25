# SPDX-License-Identifier: Apache-2.0

"""``areal agent run`` — launch the agent service (detached by default).

Interactive shell (``-i``) is deferred — the flag is accepted but raises
``NotImplementedError`` when used. Default-session creation happens at the
end of the launch if the inference service is configured; otherwise only the
gateway/router/N pairs are started and the user creates sessions on demand
via ``areal agent new_session``.
"""

from __future__ import annotations

import argparse
import os
import time

from areal.experimental.cli.agent_launcher import start_agent_service
from areal.experimental.cli.agent_sessions import (
    SessionEntry,
    SessionRegistry,
    generate_session_key,
)
from areal.experimental.cli.agent_state import (
    agent_logs_dir,
    get_current_agent_service,
    set_current_agent_service,
)
from areal.experimental.cli.gateway_client import (
    GatewayClient,
    GatewayError,
)


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "run",
        help="Launch the agent service (detached by default).",
        description=(
            "Spawn router + N × (worker + data_proxy) + gateway as detached "
            "subprocesses, wait for HTTP health, optionally bootstrap a "
            "default agent session (negotiating /rl/start_session with the "
            "inference backend if configured), and persist state under "
            "~/.areal/agent/."
        ),
    )

    # Service identity / topology
    p.add_argument(
        "--agent",
        required=True,
        help="Fully-qualified import path of the AgentRunnable implementation.",
    )
    p.add_argument("--service", default="default", help="Service instance name.")
    p.add_argument(
        "--num-pairs",
        type=int,
        default=1,
        help="Number of worker + data_proxy pairs to launch (>=1).",
    )

    # Addressing
    p.add_argument("--gateway-host", default="127.0.0.1", help="Gateway bind host.")
    p.add_argument("--gateway-port", type=int, default=19080, help="Gateway bind port.")
    p.add_argument("--router-host", default="127.0.0.1", help="Router bind host.")
    p.add_argument("--router-port", type=int, default=19081, help="Router bind port.")
    p.add_argument("--worker-host", default="127.0.0.1", help="Workers bind host.")
    p.add_argument(
        "--worker-base-port",
        type=int,
        default=19082,
        help="First worker port; pair i uses base+i.",
    )
    p.add_argument("--proxy-host", default="127.0.0.1", help="DataProxies bind host.")
    p.add_argument(
        "--proxy-base-port",
        type=int,
        default=19182,
        help="First data_proxy port; pair i uses base+i.",
    )
    p.add_argument(
        "--admin-api-key",
        default="areal-agent-admin",
        help="Admin API key injected into gateway and router.",
    )

    # Timeouts
    p.add_argument("--router-poll-interval", type=float, default=5.0)
    p.add_argument("--worker-health-timeout", type=float, default=2.0)
    p.add_argument("--proxy-request-timeout", type=float, default=600.0)
    p.add_argument("--proxy-session-timeout", type=int, default=3600)
    p.add_argument("--gateway-router-timeout", type=float, default=2.0)
    p.add_argument("--gateway-forward-timeout", type=float, default=120.0)
    p.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
    )
    p.add_argument(
        "--launch-timeout",
        type=float,
        default=60.0,
        help="Seconds to wait for the gateway to become healthy.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing healthy instance with the same name.",
    )

    # Inference backend hookup
    p.add_argument(
        "--inf-addr",
        default=os.environ.get("AREAL_INF_ADDR", ""),
        help="Inference gateway base URL (env: AREAL_INF_ADDR).",
    )
    p.add_argument(
        "--inf-model",
        default=os.environ.get("AREAL_INF_MODEL", ""),
        help="Default inference model name (env: AREAL_INF_MODEL).",
    )
    p.add_argument(
        "--inf-api-key",
        default=os.environ.get("AREAL_INF_API_KEY", None),
        help="Inference admin API key (env: AREAL_INF_API_KEY). Never persisted.",
    )

    # Default-session bootstrap
    p.add_argument(
        "--session-key",
        default=None,
        help="Optional initial session key for the default session (auto-generated otherwise).",
    )
    p.add_argument(
        "--no-default-session",
        action="store_true",
        help="Skip creating a default session at launch.",
    )
    p.add_argument(
        "--session-timeout",
        type=float,
        default=1800.0,
        help="Inactivity timeout for the bootstrap session (seconds).",
    )

    # Forward-compat
    p.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Enter an interactive shell after launch (not yet implemented).",
    )
    p.add_argument(
        "--stop-on-exit",
        action="store_true",
        help="Stop the service when leaving the interactive shell (not yet implemented).",
    )

    p.set_defaults(func=_handle)


def _bootstrap_default_session(
    *,
    service: str,
    session_key: str,
    session_timeout: float,
    inf_addr: str,
    inf_model: str,
    inf_api_key: str | None,
) -> tuple[SessionEntry, str | None]:
    """Negotiate /rl/start_session with the inference service if configured.

    Returns ``(entry, warning_text)``. Warning is non-None if the negotiation
    failed (e.g., inference unreachable, no api_key returned). The session
    is *still* created locally — chat continues to work, only RL tracking is
    missing.
    """
    rl_present = False
    warning: str | None = None

    if inf_addr:
        try:
            client = GatewayClient(inf_addr, admin_api_key=inf_api_key, timeout=5.0)
            resp = client.start_rl_session(model=inf_model)
            sessions = resp.get("sessions") if isinstance(resp, dict) else None
            if sessions and isinstance(sessions, list):
                first = sessions[0] if sessions else {}
                rl_present = bool(first.get("api_key"))
            if not rl_present:
                warning = (
                    f"Inference at {inf_addr} accepted /rl/start_session but "
                    f"did not return a session api_key."
                )
        except GatewayError as e:
            warning = (
                f"Inference at {inf_addr} did not accept /rl/start_session: {e}. "
                f"Using original API key."
            )

    entry = SessionEntry(
        key=session_key,
        active=True,
        rl_session_key_present=rl_present,
        session_timeout=session_timeout,
    )
    reg = SessionRegistry.load(service)
    reg.add(entry)
    reg.save()
    return entry, warning


def _handle(args: argparse.Namespace) -> int:
    if args.interactive:
        raise NotImplementedError(
            "`areal agent run -i` (interactive shell) is not yet implemented. "
            "Run in detached mode and use `areal agent chat`."
        )

    state = start_agent_service(
        name=args.service,
        agent_class=args.agent,
        num_pairs=args.num_pairs,
        gateway_host=args.gateway_host,
        gateway_port=args.gateway_port,
        router_host=args.router_host,
        router_port=args.router_port,
        worker_host=args.worker_host,
        worker_base_port=args.worker_base_port,
        proxy_host=args.proxy_host,
        proxy_base_port=args.proxy_base_port,
        admin_api_key=args.admin_api_key,
        inf_addr=args.inf_addr,
        inf_model=args.inf_model,
        inf_api_key=args.inf_api_key,
        router_poll_interval=args.router_poll_interval,
        worker_health_timeout=args.worker_health_timeout,
        proxy_request_timeout=args.proxy_request_timeout,
        proxy_session_timeout=args.proxy_session_timeout,
        gateway_router_timeout=args.gateway_router_timeout,
        gateway_forward_timeout=args.gateway_forward_timeout,
        log_level=args.log_level,
        force=args.force,
        launch_timeout=args.launch_timeout,
        mode="detached",
    )

    if get_current_agent_service() is None:
        set_current_agent_service(state.name)

    logs = agent_logs_dir(state.name)
    print(f"Started agent service {state.name!r}.")
    print(f"  gateway: {state.gateway_url}  (pid {state.gateway_pid})")
    print(f"  router:  {state.router_url}  (pid {state.router_pid})")
    print(f"  pairs:   {len(state.pairs)}")
    for pair in state.pairs:
        print(
            f"    [{pair.index}] worker pid={pair.worker_pid} port={pair.worker_port} | "
            f"proxy pid={pair.proxy_pid} port={pair.proxy_port}"
        )
    print(f"  agent:   {state.agent_class}")
    print(f"  logs:    {logs}")
    print(
        f"  uptime:  {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(state.created_at))}"
    )

    if not args.no_default_session:
        key = args.session_key or generate_session_key()
        entry, warning = _bootstrap_default_session(
            service=state.name,
            session_key=key,
            session_timeout=args.session_timeout,
            inf_addr=state.inf_addr,
            inf_model=state.inf_model,
            inf_api_key=args.inf_api_key,
        )
        rl_str = "rl-key=yes" if entry.rl_session_key_present else "rl-key=no"
        print(f"  session: {entry.key}  ({rl_str})")
        if warning:
            print(f"  WARNING: {warning}")

    print(f"  stop:    areal agent stop --service {state.name}")
    return 0
