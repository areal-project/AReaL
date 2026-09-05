# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import click

from areal.v2.cli.weight_update.client import WeightUpdateClient
from areal.v2.cli.weight_update.lifecycle import wu_lifecycle
from areal.v2.cli.weight_update.state import ServiceState


def resolve_target(
    service: str | None, gateway: str | None, admin_api_key: str | None
) -> tuple[WeightUpdateClient, ServiceState | None]:
    """Return a client for the gateway to talk to, plus its state if local.

    ``--gateway`` wins so an operator can reach a gateway this machine has no
    state file for -- a split-cluster or remote setup, where the controller
    that owns the process runs somewhere else. Otherwise the address comes
    from the state file the controller wrote.
    """
    if gateway:
        return WeightUpdateClient(gateway, admin_api_key), None

    name = wu_lifecycle.resolve_service_name(service)
    if not wu_lifecycle.state_path(name).exists():
        raise click.ClickException(
            f"no weight-update service named '{name}' is known to this machine. "
            "Pass --gateway URL to reach one started elsewhere."
        )
    state = wu_lifecycle.load_state(name)
    return WeightUpdateClient(
        state.gateway.url, admin_api_key or state.admin_api_key
    ), state
