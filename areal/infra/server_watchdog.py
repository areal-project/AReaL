# SPDX-License-Identifier: Apache-2.0

"""Liveness watchdog for inference servers that are not behind a backend router."""

from __future__ import annotations

import socket
import threading

from areal.utils import logging
from areal.utils.network import format_hostport

logger = logging.getLogger("ServerWatchdog")


class InferenceServerWatchdog:
    """Detect inference servers that stopped listening after a successful launch.

    A server that exited (crash, OOM, kill) refuses TCP connections on its port.
    Rollouts sent to it only fail and get resubmitted, so without this check the
    trainer waits for a batch that can never complete. Only refused connections
    count: a busy or paused server still accepts them, and timeouts are ignored
    so that a slow network cannot abort a healthy run.
    """

    def __init__(
        self,
        addrs: list[tuple[str, int]],
        interval: float = 5.0,
        max_failures: int = 3,
        connect_timeout: float = 2.0,
    ):
        self._addrs = list(addrs)
        self._interval = interval
        self._max_failures = max_failures
        self._connect_timeout = connect_timeout
        self._failures = dict.fromkeys(self._addrs, 0)
        self._error: RuntimeError | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None or not self._addrs:
            return
        self._thread = threading.Thread(
            target=self._watch, name="inference-server-watchdog", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._connect_timeout + 1.0)
            self._thread = None

    def check_health(self) -> None:
        if self._error is not None:
            raise self._error

    def _probe(self) -> None:
        for host, port in self._addrs:
            try:
                socket.create_connection(
                    (host, port), timeout=self._connect_timeout
                ).close()
            except ConnectionRefusedError:
                self._failures[(host, port)] += 1
            except OSError:
                continue
            else:
                self._failures[(host, port)] = 0
            if self._failures[(host, port)] >= self._max_failures:
                self._error = RuntimeError(
                    f"Inference server at {format_hostport(host, port)} refused "
                    f"{self._max_failures} consecutive connections; the server "
                    "process has most likely exited. Check its log for the "
                    "traceback (e.g. CUDA out of memory)."
                )
                return

    def _watch(self) -> None:
        while self._error is None and not self._stop.wait(self._interval):
            self._probe()
        if self._error is not None:
            logger.error(str(self._error))
