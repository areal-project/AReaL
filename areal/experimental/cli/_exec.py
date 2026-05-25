# SPDX-License-Identifier: Apache-2.0

"""Internal entrypoint used by ``areal <service> start`` to execute the
driver in a detached subprocess. Not intended for direct user invocation.

The parent process spawns ``python -m areal.experimental.cli._exec`` with the resolved
driver spec; this module imports the driver and forwards argv. State file
updates (status transitions on completion / failure) happen here so the
final state reflects what the subprocess actually saw.
"""

from __future__ import annotations

import argparse
import sys

from areal.experimental.cli.runner import _import_driver
from areal.experimental.cli.state import RunState


def _update_status(name: str, status: str) -> None:
    try:
        state = RunState.load(name)
    except FileNotFoundError:
        return
    state.status = status
    state.save()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="areal-exec", add_help=True)
    parser.add_argument("--driver", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--command", required=True)
    args, rest = parser.parse_known_args(argv)

    # Strip leading '--' separator added by the parent.
    if rest and rest[0] == "--":
        rest = rest[1:]

    driver_argv = ["--config", args.config, *rest]

    rc = 0
    try:
        fn = _import_driver(args.driver)
        result = fn(driver_argv)
        if isinstance(result, int):
            rc = result
    except SystemExit as e:
        rc = int(e.code) if isinstance(e.code, int) else 1
    except BaseException:
        _update_status(args.name, "failed")
        raise

    _update_status(args.name, "completed" if rc == 0 else "failed")
    return rc


if __name__ == "__main__":
    sys.exit(main())
