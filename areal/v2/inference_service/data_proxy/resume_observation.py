# SPDX-License-Identifier: Apache-2.0

"""Track request observations and the first call after reconfiguration."""

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

    def claim(self, received: float, *, collect: bool = False) -> dict | None:
        first = self.marker is not None and self.first_call is None
        if not first and not collect:
            return None
        admitted = time.monotonic()
        observation = {
            "received_monotonic": received,
            "admitted_monotonic": admitted,
            "admission_wait_seconds": admitted - received,
            "status": "running",
            "backend_attempts": [],
        }
        if first:
            observation.update(self.marker)
            self.first_call = observation
        return observation

    def snapshot(self) -> dict:
        return {"marker": self.marker, "first_call": self.first_call}
