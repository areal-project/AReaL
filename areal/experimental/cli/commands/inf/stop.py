# SPDX-License-Identifier: Apache-2.0

"""``areal inf stop`` — gracefully tear down a running service.

Sends SIGTERM to the supervisor.  The supervisor's SIGTERM handler runs
``controller.destroy()`` — the same teardown path ``PPOTrainer.close()``
uses on normal training shutdown — so workers, router, gateway, and
data-proxies are all released through the scheduler.

If the supervisor does not exit within --grace, sends SIGKILL.  Either
way, the state file is removed so subsequent ``areal inf run`` calls
with the same name can proceed.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time


_DESCRIPTION = __doc__


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "stop",
        help="Stop a running inference service.",
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("name", help="Service name (e.g. <experiment>/<trial>).")
    p.add_argument(
        "--grace", type=float, default=60.0,
        help="Seconds to wait for graceful teardown before SIGKILL.",
    )
    p.set_defaults(func=_handle)


def _signal(pid: int, sig: int) -> bool:
    try:
        os.kill(pid, sig)
        return True
    except ProcessLookupError:
        return False


def _handle(args: argparse.Namespace) -> int:
    from areal.experimental.cli.inf_state import (
        ServiceState,
        get_current_service,
        set_current_service,
        supervisor_alive,
    )

    try:
        state = ServiceState.load(args.name)
    except FileNotFoundError:
        print(f"No service named {args.name!r}.", file=sys.stderr)
        return 1

    if not supervisor_alive(state):
        print(
            f"Service {args.name!r} supervisor (pid {state.supervisor_pid}) is not alive; "
            f"removing stale state.",
            file=sys.stderr,
        )
        state.remove()
        if get_current_service() == state.name:
            set_current_service(None)
        return 0

    print(
        f"Sending SIGTERM to supervisor pid={state.supervisor_pid} ...",
        file=sys.stderr,
    )
    _signal(state.supervisor_pid, signal.SIGTERM)

    deadline = time.time() + args.grace
    while time.time() < deadline:
        if not supervisor_alive(state):
            break
        time.sleep(0.5)

    if supervisor_alive(state):
        print(
            f"Supervisor still alive after {args.grace:.0f}s — sending SIGKILL.",
            file=sys.stderr,
        )
        _signal(state.supervisor_pid, signal.SIGKILL)
        time.sleep(1.0)

    state.remove()
    if get_current_service() == state.name:
        set_current_service(None)
    print(f"Service {args.name!r} stopped.")
    return 0
