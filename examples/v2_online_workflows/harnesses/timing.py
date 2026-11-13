"""Optional, credential-free startup phase measurements for a Hermes task."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

_previous = time.monotonic()


def mark(phase: str) -> None:
    global _previous
    now = time.monotonic()
    elapsed = now - _previous
    _previous = now
    if os.environ.get("AUTOPILOT_STARTUP_TIMING") != "1":
        return
    directory = os.environ.get("ARENA_AGENT_OUTPUT_DIR")
    if not directory:
        return
    try:
        path = Path(directory) / "startup-timing.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "phase": phase,
                        "seconds": elapsed,
                        "wall_time": time.time(),
                        "pid": os.getpid(),
                    }
                )
                + "\n"
            )
    except OSError:
        # Observation must not turn a completed model trajectory into a failure.
        pass
