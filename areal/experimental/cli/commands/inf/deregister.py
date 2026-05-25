# SPDX-License-Identifier: Apache-2.0

"""``areal inf deregister`` — drop a model from a service.

Design §11.7.5. For external models this is a single ``POST
/deregister_model`` call to the gateway plus local-state cleanup. For
internal models we also SIGTERM (and escalate to SIGKILL after the grace
period) the data proxy and inference server processes the CLI launched at
``register`` time.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import time

from areal.experimental.cli.commands.inf._common import (
    add_targeting_flags,
    resolve_target,
)
from areal.experimental.cli.gateway_client import GatewayError
from areal.experimental.cli.inf_models import ModelRegistry
from areal.experimental.cli.state import pid_alive


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "deregister",
        help="Deregister a model and tear down its backends.",
        description=(
            "Remove a model from the routing table (POST /deregister_model) "
            "and clean up any local backend processes the CLI launched at "
            "register time. Local state under ~/.areal/inf/models/ is "
            "updated regardless of gateway success."
        ),
    )
    add_targeting_flags(p)
    p.add_argument("model_name", help="Name of the model to deregister.")
    p.add_argument(
        "--grace-period",
        type=float,
        default=10.0,
        help="Seconds to wait before SIGKILL for tracked backend processes.",
    )
    p.add_argument(
        "--keep-local",
        action="store_true",
        help="Keep the model entry in local state for debugging.",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Gateway request timeout (s).",
    )
    p.set_defaults(func=_handle)


def _kill_pids(pids: list[int], grace: float) -> dict[int, str]:
    """SIGTERM each PID, wait up to `grace` seconds, escalate to SIGKILL."""
    outcome: dict[int, str] = {}
    alive: list[int] = []
    for pid in pids:
        if pid <= 0 or not pid_alive(pid):
            outcome[pid] = "already-gone"
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            alive.append(pid)
        except OSError as e:
            outcome[pid] = f"sigterm-error: {e}"
    deadline = time.time() + max(0.0, grace)
    while alive and time.time() < deadline:
        alive = [p for p in alive if pid_alive(p)]
        if not alive:
            break
        time.sleep(0.2)
    for pid in alive:
        try:
            os.kill(pid, signal.SIGKILL)
            outcome[pid] = "sigkill"
        except OSError as e:
            outcome[pid] = f"sigkill-error: {e}"
    for pid in pids:
        outcome.setdefault(pid, "sigterm-exited")
    return outcome


def _handle(args: argparse.Namespace) -> int:
    target = resolve_target(args)
    client = target.client(timeout=args.timeout)

    if target.service is None:
        raise SystemExit(
            "deregister requires a local service. Pass --service or use a "
            "service launched with `areal inf run`."
        )

    registry = ModelRegistry.load(target.service)
    entry = registry.get(args.model_name)

    gateway_status: str
    gateway_error: str | None = None
    try:
        resp = client.deregister_model(args.model_name)
        gateway_status = resp.get("status", "deregistered") if isinstance(resp, dict) else "deregistered"
    except GatewayError as e:
        gateway_status = "error"
        gateway_error = str(e)

    backend_outcomes: dict[int, str] = {}
    if entry is not None:
        pids = list(entry.inference_pids) + list(entry.data_proxy_pids)
        if pids:
            backend_outcomes = _kill_pids(pids, args.grace_period)

    if entry is not None and not args.keep_local:
        registry.remove(args.model_name)
        registry.save()
        local_status = "removed"
    elif entry is not None:
        local_status = "kept"
    else:
        local_status = "absent"

    if args.json:
        print(json.dumps({
            "service": target.service,
            "model": args.model_name,
            "gateway_status": gateway_status,
            "gateway_error": gateway_error,
            "local_status": local_status,
            "backend_outcomes": {str(k): v for k, v in backend_outcomes.items()},
            "new_default": registry.default_model,
        }, indent=2))
    else:
        msg = f"Deregistered {args.model_name!r} from service {target.service!r}."
        if gateway_error:
            msg += f" (gateway error: {gateway_error})"
        if backend_outcomes:
            msg += f" Killed {len(backend_outcomes)} backend pid(s)."
        print(msg)
    return 0 if gateway_status == "deregistered" else 1
