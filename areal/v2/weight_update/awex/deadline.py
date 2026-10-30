# SPDX-License-Identifier: Apache-2.0
"""Bound a dedicated NCCL transfer without aborting model process groups."""

from __future__ import annotations

import math
import threading
import time
from types import TracebackType

import torch.distributed as dist
from torch.distributed import distributed_c10d


class TransferDeadline:
    """Abort only the captured AWEX group; never detach a running watchdog.

    NCCL abort can make a blocked transfer return normally. Success therefore
    requires an explicit deadline check, including the final context exit.
    This uses Torch's private group-specific abort API; errors retain the group
    and prevent teardown/reuse rather than pretending cancellation succeeded.
    """

    def __init__(self, group: dist.ProcessGroup, timeout_s: float):
        if group is None or group is dist.group.WORLD:
            raise ValueError("A dedicated AWEX process group is required")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("AWEX transfer timeout must be finite and positive")
        self.group = group
        self.timeout_s = timeout_s
        self.aborted = False
        self._abort = getattr(distributed_c10d, "_abort_process_group", None)
        if self._abort is None:
            raise RuntimeError("This Torch version cannot abort a dedicated NCCL group")
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._succeeded = False
        self._expired = False
        self._abort_error: BaseException | None = None
        self._deadline = 0.0
        self._thread: threading.Thread | None = None

    @property
    def stopped(self) -> bool:
        return self._thread is not None and not self._thread.is_alive()

    def check(self) -> float:
        """Return the remaining budget or reject an expired operation."""
        with self._lock:
            remaining = self._deadline - time.monotonic()
            if self._expired or remaining <= 0:
                self._expired = True
                self._wake.set()
                raise TimeoutError("AWEX transfer deadline exceeded")
            return remaining

    def _watch(self) -> None:
        self._wake.wait(self.timeout_s)
        with self._lock:
            if self._succeeded:
                return
            self._expired = self._expired or time.monotonic() >= self._deadline
        try:
            # Capture the explicit old group, never the adapter's current group.
            self._abort(self.group)
            self.aborted = True
        except BaseException as error:
            self._abort_error = error

    def __enter__(self) -> TransferDeadline:
        self._deadline = time.monotonic() + self.timeout_s
        self._thread = threading.Thread(
            target=self._watch, name="AWEXTransferDeadline", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        with self._lock:
            self._expired = self._expired or time.monotonic() >= self._deadline
            self._succeeded = exc_type is None and not self._expired
        self._wake.set()
        self.wait()
        if self._expired:
            raise TimeoutError("AWEX transfer deadline exceeded") from exc

    def wait(self) -> None:
        """Require abort completion before clearing handles or reusing the adapter."""
        if self._thread is None:
            return
        self._thread.join(timeout=self.timeout_s + 30.0)
        if self._thread.is_alive():
            raise RuntimeError("AWEX group abort is still running; retain its fence")
        if self._abort_error is not None:
            raise RuntimeError(
                "AWEX group abort failed; retain its fence"
            ) from self._abort_error
