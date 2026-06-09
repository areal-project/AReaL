# SPDX-License-Identifier: Apache-2.0

"""``areal inf stop`` — stop a running inference service.

Sends SIGTERM to the gateway, the router, and any tracked model worker
PIDs (data proxies + inference servers, populated in phase 3).  After the
grace period, escalates to SIGKILL.  State files are removed unless
``--keep-state`` is passed.

Alias: ``areal inf destroy`` (later — keeping the alias spec for design
reference).
"""

from __future__ import annotations

import argparse
import sys
import time

from areal.utils.logging import getLogger

logger = getLogger("InfCli")


_DESCRIPTION = __doc__


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "stop",
        help="Stop a running inference service.",
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--service", default=None,
        help="Service instance name (defaults to current).",
    )
    p.add_argument(
        "--grace-period", type=float, default=10.0,
        help="Seconds to wait before escalating to SIGKILL.",
    )
    p.add_argument(
        "--keep-state", action="store_true",
        help="Keep state files after shutdown (debugging).",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Skip confirmations; reserved for future interactive prompts.",
    )
    p.set_defaults(func=_handle)


def _handle(args: argparse.Namespace) -> int:
    from areal.experimental.cli.commands.inf.gateway_client import (
        GatewayClient,
        GatewayUnreachable,
    )
    from areal.experimental.cli.commands.inf.launcher import kill_pids
    from areal.experimental.cli.commands.inf.state import (
        ServiceModels,
        ServiceState,
        get_current_service,
        gateway_alive,
        models_state_path,
        resolve_service,
        router_alive,
        set_current_service,
    )

    name = resolve_service(args.service)

    try:
        state = ServiceState.load(name)
    except FileNotFoundError:
        print(f"No service named {name!r}.", file=sys.stderr)
        return 1

    pids: list[int] = [state.gateway_pid, state.router_pid]

    # Pull any tracked model worker pids so we kill them in one sweep.
    models_path = models_state_path(name)
    if models_path.exists():
        sm = ServiceModels.load(name)
        for m in sm.list_all():
            for pid in m.worker_pids:
                if pid > 0:
                    pids.append(pid)

    alive = gateway_alive(state) or router_alive(state)
    if not alive:
        logger.warning(
            "Service %r is already down (no live gateway/router pid); "
            "cleaning up state.", name,
        )
    else:
        logger.info(
            "Stopping service %r: gateway=%d, router=%d ...",
            name, state.gateway_pid, state.router_pid,
        )
        kill_pids(pids, grace_s=args.grace_period)

        # Best-effort: confirm gateway /health stops responding within a few seconds.
        client = GatewayClient(state.gateway_url, timeout=1.0)
        deadline = time.time() + min(5.0, args.grace_period)
        while time.time() < deadline:
            try:
                client.health()
                time.sleep(0.3)
            except GatewayUnreachable:
                break

    if not args.keep_state:
        state.remove()
        if models_path.exists():
            models_path.unlink()
        if get_current_service() == name:
            set_current_service(None)

    print(f"Service {name!r} stopped.")
    return 0
