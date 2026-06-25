# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import click

from areal.experimental.cli.client import ServiceHTTPError, ServiceUnreachable
from areal.experimental.cli.inference.client import GatewayClient
from areal.experimental.cli.inference.lifecycle import inf_lifecycle


@click.command(name="reward", help="Set reward on a session.")
@click.argument("reward_value", type=float)
@click.option("--session-api-key", required=True, help="Session API key.")
@click.option("--service", default=None, help="Target service instance.")
@click.option("--model", required=True, help="Model name used for routing.")
def reward_cmd(
    reward_value: float,
    session_api_key: str,
    service: str | None,
    model: str,
) -> None:
    raise SystemExit(
        do_reward(session_api_key, reward_value, model, service=service) or 0
    )


def do_reward(
    session_api_key: str,
    reward: float,
    model: str,
    *,
    service: str | None = None,
) -> int:
    state = inf_lifecycle.load_running_state(service)
    gateway = GatewayClient(state.gateway_url, state.admin_api_key)
    try:
        gateway.set_reward(session_api_key=session_api_key, reward=reward, model=model)
    except (ServiceUnreachable, ServiceHTTPError) as exc:
        raise click.ClickException(f"set_reward failed: {exc}") from exc
    return 0
