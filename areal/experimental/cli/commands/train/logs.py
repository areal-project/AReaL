# SPDX-License-Identifier: Apache-2.0

"""``areal train logs`` — tail the captured stdout/stderr of a training run.

Reads from the path recorded in ``RunState.log_path`` (set by
``start_background``). Foreground runs (``areal train run``) don't have a
captured log file — this command will report no log available.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from areal.experimental.cli.state import RunState, log_path


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "logs",
        help="Tail a training run's captured log.",
        description=(
            "Print the tail of ~/.areal/logs/<run-name>.log (or the path "
            "recorded in the run's state file). Use --follow to stream new "
            "lines as the driver writes them."
        ),
    )
    p.add_argument("run_name", help="Run name (typically experiment_name/trial_name).")
    p.add_argument(
        "--follow",
        "-f",
        action="store_true",
        help="Stream appended lines until interrupted.",
    )
    p.add_argument(
        "--lines",
        "-n",
        type=int,
        default=200,
        help="Number of recent lines to print initially.",
    )
    p.set_defaults(func=_handle)


def _resolve_log(run_name: str) -> Path:
    try:
        state = RunState.load(run_name)
        if state.log_path:
            return Path(state.log_path)
    except FileNotFoundError:
        pass
    return log_path(run_name)


def _tail(path: Path, n: int) -> list[str]:
    if not path.exists():
        return []
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        block = 4096
        data = b""
        pos = size
        while pos > 0 and data.count(b"\n") <= n:
            read = min(block, pos)
            pos -= read
            f.seek(pos)
            data = f.read(read) + data
    lines = data.splitlines()
    return [ln.decode("utf-8", "replace") for ln in lines[-n:]]


def _follow(path: Path) -> int:
    while not path.exists():
        time.sleep(0.5)
    with open(path, "rb") as f:
        f.seek(0, 2)
        try:
            while True:
                line = f.readline()
                if line:
                    print(line.decode("utf-8", "replace"), end="")
                else:
                    time.sleep(0.3)
        except KeyboardInterrupt:
            return 0


def _handle(args: argparse.Namespace) -> int:
    path = _resolve_log(args.run_name)
    if not path.exists() and not args.follow:
        raise SystemExit(
            f"No log file at {path}. Foreground runs (`areal train run`) "
            f"don't capture stdout/stderr — use `start` for a background "
            f"run with a captured log."
        )
    for line in _tail(path, args.lines):
        print(line)
    if args.follow:
        return _follow(path)
    return 0
