# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for ``areal agent`` subcommands.

Mirrors ``commands/inf/_common.py``: gateway-target resolution and a tiny
text-table formatter (imported from the inf helpers to avoid duplication).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from areal.experimental.cli.agent_gateway_client import AgentGatewayClient
from areal.experimental.cli.agent_state import (
    AgentServiceState,
    agent_service_state_path,
    resolve_agent_service_name,
)
from areal.experimental.cli.commands.inf._common import print_table  # noqa: F401

__all__ = ["ResolvedAgentTarget", "add_targeting_flags", "resolve_agent_target", "print_table"]


@dataclass
class ResolvedAgentTarget:
    service: str | None
    gateway_url: str
    admin_api_key: str | None
    state: AgentServiceState | None

    def client(self, timeout: float = 5.0) -> AgentGatewayClient:
        return AgentGatewayClient(
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
        help="Override agent gateway URL instead of resolving from local state.",
    )
    parser.add_argument(
        "--admin-api-key",
        default=None,
        help="Admin API key for privileged agent-gateway requests.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output.",
    )


def resolve_agent_target(args: argparse.Namespace) -> ResolvedAgentTarget:
    if args.gateway_url:
        return ResolvedAgentTarget(
            service=args.service,
            gateway_url=args.gateway_url.rstrip("/"),
            admin_api_key=args.admin_api_key,
            state=None,
        )
    name = resolve_agent_service_name(args.service)
    if not agent_service_state_path(name).exists():
        raise SystemExit(
            f"No local state for agent service {name!r}. "
            f"Start it with `areal agent run --service {name}` "
            f"or pass --gateway-url for a remote gateway."
        )
    state = AgentServiceState.load(name)
    return ResolvedAgentTarget(
        service=name,
        gateway_url=state.gateway_url,
        admin_api_key=args.admin_api_key or state.admin_api_key,
        state=state,
    )
