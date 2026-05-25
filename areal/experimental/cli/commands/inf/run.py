# SPDX-License-Identifier: Apache-2.0

"""``areal inf run`` — launch the inference service (detached mode).

Interactive mode (``-i``) and inline model registration (``--model``) are
deferred to subsequent steps; the flags are accepted for forward
compatibility but raise ``NotImplementedError`` when actually used.
"""

from __future__ import annotations

import argparse
import time

from areal.experimental.cli.inf_launcher import start_service
from areal.experimental.cli.inf_state import (
    get_current_service,
    service_logs_dir,
    set_current_service,
)


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "run",
        help="Launch the inference service (detached by default).",
        description=(
            "Spawn gateway + router as detached subprocesses, wait for HTTP "
            "health, and persist state under ~/.areal/inf/. By default the "
            "CLI exits after the service is healthy. Use --interactive / -i "
            "to enter an interactive shell (not yet implemented)."
        ),
    )
    # Service identity / addressing
    p.add_argument("--service", default="default", help="Service instance name.")
    p.add_argument("--gateway-host", default="127.0.0.1", help="Gateway bind host.")
    p.add_argument("--gateway-port", type=int, default=8080, help="Gateway bind port.")
    p.add_argument("--router-host", default="127.0.0.1", help="Router bind host.")
    p.add_argument("--router-port", type=int, default=8081, help="Router bind port.")
    p.add_argument(
        "--admin-api-key",
        default="areal-admin-key",
        help="Admin API key injected into gateway and router.",
    )
    # Routing / timeouts
    p.add_argument(
        "--routing-strategy",
        default="round_robin",
        choices=["round_robin", "least_busy"],
        help="Router strategy.",
    )
    p.add_argument("--poll-interval", type=float, default=5.0, help="Router worker poll interval (s).")
    p.add_argument("--router-timeout", type=float, default=2.0, help="Gateway→router routing timeout (s).")
    p.add_argument("--forward-timeout", type=float, default=120.0, help="Gateway forwarding timeout (s).")
    p.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Log level for launched processes.",
    )
    # Lifecycle
    p.add_argument(
        "--launch-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for the gateway to become healthy.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing healthy instance with the same name.",
    )
    # Forward-compat flags (raise NotImplementedError on use)
    p.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Enter an interactive shell after launch (not yet implemented).",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Register a model inline at startup (not yet implemented; use `areal inf register`).",
    )

    p.set_defaults(func=_handle)


def _handle(args: argparse.Namespace) -> int:
    if args.interactive:
        raise NotImplementedError(
            "`areal inf run -i` (interactive mode) is not yet implemented. "
            "Run in detached mode and use `areal inf chat` (coming soon)."
        )
    if args.model:
        raise NotImplementedError(
            "Inline `--model` registration is not yet implemented. "
            "Start the service first, then use `areal inf register` (coming soon)."
        )

    state = start_service(
        name=args.service,
        gateway_host=args.gateway_host,
        gateway_port=args.gateway_port,
        router_host=args.router_host,
        router_port=args.router_port,
        admin_api_key=args.admin_api_key,
        routing_strategy=args.routing_strategy,
        poll_interval=args.poll_interval,
        router_timeout=args.router_timeout,
        forward_timeout=args.forward_timeout,
        log_level=args.log_level,
        force=args.force,
        launch_timeout=args.launch_timeout,
        mode="detached",
    )

    # First service launched becomes the current service unless one is already set.
    if get_current_service() is None:
        set_current_service(state.name)

    logs = service_logs_dir(state.name)
    print(f"Started service {state.name!r}.")
    print(f"  gateway: {state.gateway_url}  (pid {state.gateway_pid})")
    print(f"  router:  {state.router_url}  (pid {state.router_pid})")
    print(f"  logs:    {logs}")
    print(f"  uptime:  {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(state.created_at))}")
    print(f"  stop:    areal inf stop --service {state.name}")
    return 0
