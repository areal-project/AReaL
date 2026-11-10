# SPDX-License-Identifier: Apache-2.0

"""Correlate physical SGLang requests with a v1 rollout session."""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_current_request_ids: ContextVar[SessionRequestIds | None] = ContextVar(
    "session_request_ids", default=None
)


class SessionRequestIds:
    """Allocate unique request IDs under one stable session UUID."""

    def __init__(self, session_uuid: str | None = None) -> None:
        self.session_uuid = session_uuid or uuid.uuid4().hex
        self._next_call_index = 0
        self._lock = threading.Lock()
        self._server_timing_callback: Callable[[str, float, float], None] | None = None

    @classmethod
    def current(cls) -> SessionRequestIds | None:
        return _current_request_ids.get()

    @contextmanager
    def activate(self) -> Iterator[SessionRequestIds]:
        token = _current_request_ids.set(self)
        try:
            yield self
        finally:
            _current_request_ids.reset(token)

    def next_call_index(self) -> int:
        with self._lock:
            self._next_call_index += 1
            return self._next_call_index

    def request_rid(self, call_index: int, send_index: int) -> str:
        """Name one physical send within a session and logical generation call."""
        if call_index < 1 or send_index < 1:
            raise ValueError("call_index and send_index must be positive")
        return f"{self.session_uuid}:{call_index}.{send_index}"

    def set_server_timing_callback(
        self, callback: Callable[[str, float, float], None]
    ) -> None:
        """Receive SGLang's reported latency for each completed physical request."""
        self._server_timing_callback = callback

    def record_server_timing(
        self, rid: str, duration_seconds: float, client_finished_at: float
    ) -> None:
        callback = self._server_timing_callback
        if callback is not None:
            callback(rid, duration_seconds, client_finished_at)
