# SPDX-License-Identifier: Apache-2.0

"""``areal run`` — foreground driver execution."""

from __future__ import annotations

import argparse
from pathlib import Path

from areal.experimental.cli.runner import resolve_driver, resolve_name, run_foreground


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "run",
        help="Run a driver in the foreground.",
        description=(
            "Resolve the driver entry from --driver or the yaml `driver:` field, "
            "then invoke it in this process. Scheduler dispatch (local/slurm/ray) "
            "is decided inside the driver based on config.scheduler.type."
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
        command_hint="run",
    )
    name = resolve_name(config_path, args.name)
    return run_foreground(
        name=name,
        command="run",
        driver_spec=driver,
        config_path=config_path,
        overrides=args.overrides or [],
    )
