# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import click

from areal.experimental.cli.inference.client import (
    GatewayClient,
    GatewayHTTPError,
    GatewayUnreachable,
)
from areal.experimental.cli.inference.common import load_running_state


@click.command(name="reward", help="Set reward on a session.")
@click.argument("session_api_key")
@click.argument("reward_value", type=float)
@click.option("--service", default=None, help="Target service instance.")
@click.option("--model", default=None, help="Model name used for routing.")
def reward_cmd(
    session_api_key: str,
    reward_value: float,
    service: str | None,
    model: str | None,
) -> None:
    raise SystemExit(
        do_reward(session_api_key, reward_value, model, service=service) or 0
    )


def do_reward(
    session_api_key: str,
    reward: float,
    model: str | None,
    *,
    service: str | None = None,
) -> int:
    state = load_running_state(service)
    gateway = GatewayClient(state.gateway_url, state.admin_api_key)
    try:
        gateway.set_reward(session_api_key=session_api_key, reward=reward, model=model)
    except (GatewayUnreachable, GatewayHTTPError) as exc:
        raise click.ClickException(f"set_reward failed: {exc}") from exc
    return 0
