# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for ``areal inf`` subcommands.

- Service / gateway target resolution (design §11.1).
- A tiny text-table formatter for human output.
- A common ``add_targeting_flags()`` so every subcommand accepts the same
  ``--service / --gateway-url / --admin-api-key / --json`` flags.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from areal.experimental.cli.gateway_client import GatewayClient
from areal.experimental.cli.inf_state import (
    ServiceState,
    resolve_service_name,
    service_state_path,
)


@dataclass
class ResolvedTarget:
    """A gateway endpoint the CLI can talk to.

    ``state`` is ``None`` when the user supplied ``--gateway-url`` for a
    service this host doesn't know about (remote gateway).
    """

    service: str | None
    gateway_url: str
    admin_api_key: str | None
    state: ServiceState | None

    def client(self, timeout: float = 5.0) -> GatewayClient:
        return GatewayClient(
            self.gateway_url, admin_api_key=self.admin_api_key, timeout=timeout
        )


def add_targeting_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--service",
        default=None,
        help="Service instance name (default: current-service file, or sole known).",
    )
    parser.add_argument(
        "--gateway-url",
        default=None,
        help="Override gateway URL instead of resolving from local state.",
    )
    parser.add_argument(
        "--admin-api-key",
        default=None,
        help="Admin API key for privileged gateway requests.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output.",
    )


def resolve_target(args: argparse.Namespace) -> ResolvedTarget:
    # 1. Explicit --gateway-url short-circuits everything (remote-mode).
    if args.gateway_url:
        return ResolvedTarget(
            service=args.service,
            gateway_url=args.gateway_url.rstrip("/"),
            admin_api_key=args.admin_api_key,
            state=None,
        )
    # 2-4. Resolve service name from explicit/current/sole.
    name = resolve_service_name(args.service)
    if not service_state_path(name).exists():
        raise SystemExit(
            f"No local state for service {name!r}. "
            f"Start it with `areal inf run --service {name}` "
            f"or pass --gateway-url for a remote gateway."
        )
    state = ServiceState.load(name)
    return ResolvedTarget(
        service=name,
        gateway_url=state.gateway_url,
        admin_api_key=args.admin_api_key or state.admin_api_key,
        state=state,
    )


# ---------------------------------------------------------------------------
# Minimal text-table for human-readable output (no `tabulate` dep).
# ---------------------------------------------------------------------------


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    cols = [list(map(str, col)) for col in zip(headers, *rows, strict=False)] if rows else [
        [h] for h in headers
    ]
    widths = [max(len(s) for s in col) for col in cols]
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    for row in rows:
        print("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)))
