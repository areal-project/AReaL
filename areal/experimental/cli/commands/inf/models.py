# SPDX-License-Identifier: Apache-2.0

"""``areal inf models`` — list models registered on the target service."""

from __future__ import annotations

import argparse
import json

from areal.experimental.cli.commands.inf._common import (
    add_targeting_flags,
    print_table,
    resolve_target,
)
from areal.experimental.cli.gateway_client import GatewayError


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "models",
        help="List models registered on the target service.",
        description=(
            "Call the gateway's GET /models endpoint and print the registered "
            "model names. Use `areal inf register` to add a model."
        ),
    )
    add_targeting_flags(p)
    p.add_argument("--timeout", type=float, default=5.0, help="Request timeout (s).")
    p.set_defaults(func=_handle)


def _handle(args: argparse.Namespace) -> int:
    target = resolve_target(args)
    client = target.client(timeout=args.timeout)
    try:
        names = client.models()
    except GatewayError as e:
        raise SystemExit(str(e)) from e

    if args.json:
        print(json.dumps({"service": target.service, "models": names}, indent=2))
        return 0

    if not names:
        print(
            f"No models registered on service {target.service or target.gateway_url!r}. "
            f"Add one with `areal inf register` (coming soon)."
        )
        return 0
    print_table(["MODEL"], [[m] for m in names])
    return 0
