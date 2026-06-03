# SPDX-License-Identifier: Apache-2.0

"""``areal run`` — foreground training driver invoker."""

from __future__ import annotations

import argparse
from pathlib import Path


_DESCRIPTION = """\
Launch a training driver in the foreground.

Resolve the driver entry from --driver or the yaml `driver:` field, then
invoke it in this process. Scheduler dispatch (local / slurm / ray) is
decided inside the driver based on config.scheduler.type.

Examples:
  areal run --config experiments/grpo.yaml
  areal run --config experiments/grpo.yaml --driver examples.math.gsm8k_rl:main
  areal run --config experiments/grpo.yaml actor.lr=1e-5 +debug.foo=bar

Hydra overrides (key=value, +key=value, ~key) after the parsed flags are
forwarded verbatim to the driver.
"""


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "run",
        help="Launch a training driver in the foreground.",
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", required=True, type=Path)
    p.add_argument(
        "--name", default=None,
        help="Override run name (default: <experiment_name>/<trial_name> from yaml).",
    )
    p.add_argument(
        "--driver", default=None,
        help="Driver entry 'module.path:func' (overrides yaml `driver:`).",
    )
    p.add_argument(
        "overrides", nargs=argparse.REMAINDER,
        help="Hydra-style overrides forwarded to the driver.",
    )
    p.set_defaults(func=_handle)


def _handle(args: argparse.Namespace) -> int:
    from areal.experimental.cli.runner import (
        resolve_driver,
        resolve_name,
        run_foreground,
    )

    config_path = args.config.expanduser().resolve()
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")

    driver = resolve_driver(config_path, cli_driver=args.driver)
    name = resolve_name(config_path, cli_name=args.name)
    return run_foreground(
        name=name,
        driver_spec=driver,
        config_path=config_path,
        overrides=args.overrides or [],
    )
