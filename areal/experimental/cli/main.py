# SPDX-License-Identifier: Apache-2.0

"""Top-level entry point for the ``areal`` console-script.

This module wires the top-level training verbs (`run`, `start`, `stop`,
`ps`, `status`, `logs`) and the three service namespaces (`inf`, `agent`,
`weight-update`) into a single argparse tree.

The shape is asymmetric on purpose: training is AReaL's primary use case,
so its verbs sit at the top level for ergonomic invocation (`areal run`
vs `areal train run`). Service-side operator commands stay namespaced
because they are auxiliary surfaces, not the main entry.

The import path is kept deliberately light: only stdlib and the verb /
namespace modules are touched here. Verb implementations must defer
heavy imports (torch, ray, megatron, sglang, vllm, fastapi, ...) into
their ``_handle`` function bodies, never at module top level.
"""

from __future__ import annotations

import argparse
import sys

from areal.experimental.cli.commands import agent as cmd_agent
from areal.experimental.cli.commands import inf as cmd_inf
from areal.experimental.cli.commands import logs as cmd_logs
from areal.experimental.cli.commands import ps as cmd_ps
from areal.experimental.cli.commands import run as cmd_run
from areal.experimental.cli.commands import start as cmd_start
from areal.experimental.cli.commands import status as cmd_status
from areal.experimental.cli.commands import stop as cmd_stop
from areal.experimental.cli.commands import weight_update as cmd_weight_update
from areal.version import __version__

_DESCRIPTION = """\
AReaL operator CLI for the v2 microservice architecture.

Training is AReaL's primary use case and its verbs are top-level
(`areal run`, `areal start`, etc.). Service-side operator surfaces
(inference, agent, weight-update) live under their own namespaces.

Top-level training verbs:
  run            Launch a training driver in the foreground
  start          Spawn a detached driver process (background)
  stop           Signal a running job by name
  ps             List locally tracked jobs
  status         Status of one job
  logs           Tail a job's combined stdout/stderr

Service namespaces:
  inf            Operate an inference service (gateway + router + models)
  agent          Operate an agent service (gateway + router + sessions)
  weight-update  Diagnose weight-sync state between train and inference

Run `areal <command> --help` for the planned surface of each verb.
Training state lives under ~/.areal/runs/.
Service state lives under ~/.areal/<namespace>/.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="areal",
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"areal {__version__}",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        metavar="COMMAND",
    )
    # Top-level training verbs
    cmd_run.add_parser(subparsers)
    cmd_start.add_parser(subparsers)
    cmd_stop.add_parser(subparsers)
    cmd_ps.add_parser(subparsers)
    cmd_status.add_parser(subparsers)
    cmd_logs.add_parser(subparsers)
    # Service namespaces
    cmd_inf.add_parser(subparsers)
    cmd_agent.add_parser(subparsers)
    cmd_weight_update.add_parser(subparsers)
    return parser


def cli(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 2
    result = func(args)
    return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":
    sys.exit(cli())
