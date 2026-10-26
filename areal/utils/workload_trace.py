# SPDX-License-Identifier: Apache-2.0
"""Opt-in CPU observations kept outside training tensor payloads."""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any

from areal.utils import logging

logger = logging.getLogger("WorkloadTrace")
_lock = threading.Lock()


def append_trace(name: str, row: dict[str, Any]) -> None:
    """Call from a worker thread when used by an asynchronous request handler."""
    directory = os.environ.get("AREAL_ROLLOUT_TRACE_DIR")
    if not directory:
        return
    host, pid = socket.gethostname(), os.getpid()
    record = {
        "schema_version": 1,
        "run_id": os.environ.get("AREAL_TRACE_RUN_ID"),
        "host": host,
        "pid": pid,
        "wall_time": time.time(),
        "monotonic": time.monotonic(),
        **row,
    }
    try:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        with _lock, (root / f"{name}-{host}-{pid}.jsonl").open("a") as stream:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    except OSError:
        logger.exception("Could not append %s observation", name)
