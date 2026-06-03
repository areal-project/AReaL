# SPDX-License-Identifier: Apache-2.0

"""``areal inf run`` — launch the v2 inference service (detached)."""

from __future__ import annotations

import argparse
import time


_DESCRIPTION = """\
Spawn the v2 inference gateway + router as detached subprocesses, wait
for HTTP /health, persist state under ~/.areal/inf/, and exit.

The CLI process exits after the service is healthy; the gateway and
router keep running. Later commands (stop / status / ps / logs / ...)
reconcile via state + PID + HTTP.
"""


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "run",
        help="Launch the inference service (detached).",
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--service", default="default", help="Service instance name.")
    p.add_argument("--gateway-host", default="127.0.0.1")
    p.add_argument("--gateway-port", type=int, default=8080)
    p.add_argument("--router-host", default="127.0.0.1")
    p.add_argument("--router-port", type=int, default=8081)
    p.add_argument("--admin-api-key", default="areal-admin-key")
    p.add_argument(
        "--routing-strategy", default="round_robin",
        choices=["round_robin", "least_busy"],
    )
    p.add_argument("--poll-interval", type=float, default=5.0)
    p.add_argument("--router-timeout", type=float, default=2.0)
    p.add_argument("--forward-timeout", type=float, default=120.0)
    p.add_argument(
        "--log-level", default="info",
        choices=["debug", "info", "warning", "error"],
    )
    p.add_argument("--launch-timeout", type=float, default=30.0)
    p.add_argument(
        "--force", action="store_true",
        help="Replace an existing healthy instance with the same name.",
    )
    p.set_defaults(func=_handle)


def _handle(args: argparse.Namespace) -> int:
    from areal.experimental.cli.inf_launcher import start_service
    from areal.experimental.cli.inf_state import (
        get_current_service,
        service_logs_dir,
        set_current_service,
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
    )

    if get_current_service() is None:
        set_current_service(state.name)

    logs = service_logs_dir(state.name)
    print(f"Started service {state.name!r}.")
    print(f"  gateway: {state.gateway_url}  (pid {state.gateway_pid})")
    print(f"  router:  {state.router_url}  (pid {state.router_pid})")
    print(f"  logs:    {logs}")
    print(f"  started: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(state.created_at))}")
    return 0
