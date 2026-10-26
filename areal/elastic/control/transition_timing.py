# SPDX-License-Identifier: Apache-2.0

"""Sparse wall-clock observations; no GPU synchronization or timing estimates."""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from areal.utils import logging

logger = logging.getLogger("RLTrainer")


class TransitionTiming:
    def __init__(self, root: Path, request_id: str, source: dict, target: dict):
        self.path = root / "reconfiguration-timing.jsonl"
        self.transition_id = uuid.uuid4().hex
        self.request_id = request_id
        self.source, self.target = source, target
        self.available = True

    def record(self, event: str, **details) -> None:
        row = {
            "schema_version": 1,
            "transition_id": self.transition_id,
            "request_id": self.request_id,
            "event": event,
            "time": time.time(),
            "monotonic": time.monotonic(),
            **details,
        }
        if not self.available:
            return
        try:
            with self.path.open("a") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        except Exception:
            self.available = False
            logger.warning(
                "Reconfiguration timing unavailable; native execution continues",
                exc_info=True,
            )

    @contextmanager
    def phase(self, name: str):
        self.record("phase_start", phase=name)
        started = time.monotonic()
        status = "completed"
        try:
            yield
        except BaseException:
            status = "failed"
            raise
        finally:
            ended = time.monotonic()
            self.record(
                "phase_end",
                phase=name,
                status=status,
                started_monotonic=started,
                ended_monotonic=ended,
                seconds=ended - started,
            )
