# SPDX-License-Identifier: Apache-2.0

"""Observe the first admitted model call after a reconfiguration marker."""

import time
from contextvars import ContextVar

current_observation: ContextVar[dict | None] = ContextVar(
    "resume_observation", default=None
)


class ResumeObservation:
    def __init__(self):
        self.marker = None
        self.first_call = None

    def arm(self, transition_id: str) -> None:
        self.marker = {
            "transition_id": transition_id,
            "armed_time": time.time(),
            "armed_monotonic": time.monotonic(),
        }
        self.first_call = None

    def claim(self, received: float) -> dict | None:
        if self.marker is None or self.first_call is not None:
            return None
        admitted = time.monotonic()
        self.first_call = {
            **self.marker,
            "received_monotonic": received,
            "admitted_monotonic": admitted,
            "admission_wait_seconds": admitted - received,
            "status": "running",
            "backend_attempts": [],
        }
        return self.first_call

    def snapshot(self) -> dict:
        return {"marker": self.marker, "first_call": self.first_call}
