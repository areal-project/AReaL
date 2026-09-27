# SPDX-License-Identifier: Apache-2.0
"""Wait for a file to appear with a bounded budget, with the clock and sleep injectable for tests."""

import os
import time
from collections.abc import Callable

POLL_S = 0.5
SLACK_S = 1e-6  # a remainder below this is rounding, not time left to wait


def wait_for_path(
    path: str,
    *,
    timeout_s: float,
    exists: Callable[[str], bool] = os.path.exists,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[float, bool]:
    """Poll until ``path`` exists or ``timeout_s`` passes; return (seconds waited, timed_out)."""
    if timeout_s <= 0:
        raise ValueError(f"timeout_s must be positive, got {timeout_s}")
    start = clock()
    while not exists(path):
        elapsed = clock() - start
        remaining = timeout_s - elapsed
        if remaining <= SLACK_S:
            return elapsed, True
        sleep(min(POLL_S, remaining))
    return clock() - start, False
