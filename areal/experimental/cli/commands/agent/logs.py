# SPDX-License-Identifier: Apache-2.0

"""``areal agent logs`` — tail component logs under ~/.areal/agent/logs/."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from areal.experimental.cli.agent_state import (
    agent_logs_dir,
    resolve_agent_service_name,
)


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "logs",
        help="Show gateway / router / worker-N / proxy-N logs.",
        description=(
            "Print the tail of a component log file from "
            "~/.areal/agent/logs/<service>/. Use --follow to stream new lines."
        ),
    )
    p.add_argument(
        "--service",
        default=None,
        help="Service instance name (default: current-service file, or sole known).",
    )
    p.add_argument(
        "--component",
        default="gateway",
        help="One of `gateway`, `router`, `worker-<idx>`, `proxy-<idx>`.",
    )
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
    name = resolve_agent_service_name(args.service)
    logs = agent_logs_dir(name)
    path = logs / f"{args.component}.log"

    if not path.exists():
        avail = sorted(p.name for p in logs.glob("*.log"))
        msg = f"No log file at {path}."
        if avail:
            msg += (
                " Available components: "
                + ", ".join(s.removesuffix(".log") for s in avail)
            )
        raise SystemExit(msg)

    for line in _tail(path, args.lines):
        print(line)
    if args.follow:
        return _follow(path)
    return 0
