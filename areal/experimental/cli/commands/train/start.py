# SPDX-License-Identifier: Apache-2.0

"""``areal train start`` — start a training driver in the background.

Spawns ``python -m areal.experimental.cli._exec`` detached (start_new_session=True),
captures stdout+stderr into ``~/.areal/logs/<run-name>.log``, and writes a
``RunState`` under ``~/.areal/runs/<run-name>.json``. Refuses to start if a
recorded PID for the same name is still alive.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from areal.experimental.cli.runner import (
    resolve_driver,
    resolve_name,
    start_background,
)


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "start",
        help="Start a training driver in the background.",
        description=(
            "Detach a training driver process. Driver entry resolution: "
            "--driver > yaml `driver:` field. Use `areal train stop <name>` "
            "to tear it down."
        ),
    )
    p.add_argument("--config", required=True, help="Path to the yaml config.")
    p.add_argument(
        "--name",
        default=None,
        help="Override run name (default: experiment_name/trial_name from yaml).",
    )
    p.add_argument(
        "--driver",
        default=None,
        help="Driver entry as 'module.path:func' (overrides yaml `driver:`).",
    )
    p.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Hydra-style overrides forwarded to the driver (e.g. actor.path=...).",
    )
    p.set_defaults(func=_handle)


def _handle(args: argparse.Namespace) -> int:
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")
    driver = resolve_driver(
        config_path,
        cli_driver=args.driver,
        command_hint="train start",
    )
    name = resolve_name(config_path, args.name)
    start_background(
        name=name,
        command="train",
        driver_spec=driver,
        config_path=config_path,
        overrides=args.overrides or [],
    )
    return 0
