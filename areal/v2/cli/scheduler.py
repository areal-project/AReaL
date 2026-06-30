# SPDX-License-Identifier: Apache-2.0

"""Unified termination dispatch for service-style CLIs.

Provides a stateless top-level ``terminate`` that takes a TaskHandle's
``ref`` dict plus the backend tag, and routes to the matching kill
path. Currently only the local backend is implemented; future backends
extend this module with their own branches.
"""

from __future__ import annotations

from areal.v2.cli.process import kill_pids


def terminate(
    ref: dict,
    *,
    backend: str = "local",
    grace_s: float = 10.0,
) -> None:
    """Terminate the task identified by ``ref`` using the given backend.

    For ``backend="local"`` ``ref`` must contain a ``"pid"`` key — the
    local PID and its descendant process tree are SIGTERM'd, given
    ``grace_s`` seconds to exit, then SIGKILL'd.
    """

    if backend == "local":
        pid = int(ref.get("pid", 0) or 0)
        if pid > 0:
            kill_pids([pid], grace_s=grace_s)
        return

    raise ValueError(f"unknown scheduler backend: {backend!r}")
