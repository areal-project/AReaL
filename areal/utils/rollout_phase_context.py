# SPDX-License-Identifier: Apache-2.0

"""Optional controller-thread sink for v1 rollout phase intervals."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

step_phase_recorder: ContextVar[Callable[[str, float, float], None] | None] = (
    ContextVar("step_phase_recorder", default=None)
)


@contextmanager
def track_step_phase(phase: str) -> Iterator[None]:
    recorder = step_phase_recorder.get()
    if recorder is None:
        yield
        return
    started_at = time.time()
    try:
        yield
    finally:
        recorder(phase, started_at, time.time())
