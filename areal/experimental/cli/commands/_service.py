# SPDX-License-Identifier: Apache-2.0

"""Shared scaffolding for service-style commands: ``areal {train,inf,agent}``.

Each service exposes the same shape::

    areal <service> start --config <yaml> [--driver MOD:FUNC] [--name X] [overrides...]
    areal <service> stop <run-name> [--timeout SECS]

The CLI itself is scheduler-agnostic; the driver decides how to dispatch to
``LocalScheduler`` / ``SlurmScheduler`` / ``RayScheduler`` based on
``config.scheduler.type``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from areal.experimental.cli.runner import (
    resolve_driver,
    resolve_name,
    start_background,
    stop_run,
)


def add_service_parser(
    subparsers: argparse._SubParsersAction,
    name: str,
    description: str,
    fallback_driver: str | None = None,
) -> None:
    """Attach a ``<service> start|stop`` parser pair to the given subparser."""
    p = subparsers.add_parser(name, help=description, description=description)
    sub = p.add_subparsers(dest="action", required=True, metavar="ACTION")

    sp_start = sub.add_parser(
        "start",
        help=f"Start the {name} service in the background.",
        description=(
            f"Start a detached {name} driver process. Driver entry resolution: "
            f"--driver > yaml `driver:` field"
            + (f" > built-in default ({fallback_driver})" if fallback_driver else "")
            + "."
        ),
    )
    sp_start.add_argument("--config", required=True, help="Path to the yaml config.")
    sp_start.add_argument(
        "--name",
        default=None,
        help="Override run name (default: experiment_name/trial_name from yaml).",
    )
    sp_start.add_argument(
        "--driver",
        default=None,
        help="Driver entry as 'module.path:func' (overrides yaml `driver:`).",
    )
    sp_start.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Hydra-style overrides forwarded to the driver (e.g. actor.path=...).",
    )
    sp_start.set_defaults(
        func=lambda a: _do_start(a, command=name, fallback_driver=fallback_driver)
    )

    sp_stop = sub.add_parser(
        "stop",
        help=f"Stop a running {name} service by name.",
    )
    sp_stop.add_argument(
        "run_name",
        help="Run name (typically experiment_name/trial_name).",
    )
    sp_stop.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="Seconds to wait for SIGTERM before sending SIGKILL (default: 15).",
    )
    sp_stop.set_defaults(func=lambda a: _do_stop(a, command=name))


def _do_start(
    args: argparse.Namespace, command: str, fallback_driver: str | None
) -> int:
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")
    driver = resolve_driver(
        config_path,
        cli_driver=args.driver,
        fallback=fallback_driver,
        command_hint=f"{command} start",
    )
    name = resolve_name(config_path, args.name)
    start_background(
        name=name,
        command=command,
        driver_spec=driver,
        config_path=config_path,
        overrides=args.overrides or [],
    )
    return 0


def _do_stop(args: argparse.Namespace, command: str) -> int:
    return stop_run(args.run_name, command_hint=command, timeout=args.timeout)
