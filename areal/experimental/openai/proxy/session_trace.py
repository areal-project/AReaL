# SPDX-License-Identifier: Apache-2.0

"""Bounded, opt-in event stream for v1 proxy session timing."""

from __future__ import annotations

import time
import uuid
from collections import deque
from collections.abc import AsyncGenerator
from threading import Lock
from typing import Any

RL_SESSION_TRACE_EVENTS_PATHNAME = "rl/session_trace/events"
RL_SESSION_TRACE_TASK_PATHNAME = "rl/session_trace/task"
RL_SESSION_TRACE_PHASE_PATHNAME = "rl/session_trace/phase"


class ProxySessionTrace:
    """Record timestamped session transitions without logging every request.

    The controller reads events by cursor. A bounded buffer prevents tracing
    from growing without limit if the controller stops collecting. A lost
    cursor is reported explicitly so missing time remains unattributed.
    """

    def __init__(self, max_events: int = 250_000) -> None:
        self._lock = Lock()
        self._events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self._next_seq = 1
        self._generating: dict[str, int] = {}
        self._active_sessions: set[str] = set()
        self._phase_values: dict[
            tuple[str, str, str | None, str | None, float], float | None
        ] = {}
        self._task_values: dict[str, tuple[str, str | None]] = {}

    def _append(
        self,
        session_id: str,
        kind: str,
        *,
        at: float | None = None,
        **details: Any,
    ) -> None:
        event = {
            "seq": self._next_seq,
            "session_id": session_id,
            "kind": kind,
            "at": time.time() if at is None else at,
            **details,
        }
        self._events.append(event)
        self._next_seq += 1

    def start_session(
        self,
        session_id: str,
        *,
        task_id: str | None = None,
        group_size: int = 1,
        session_uuid: str | None = None,
    ) -> str:
        with self._lock:
            self._active_sessions.add(session_id)
            session_uuid = session_uuid or uuid.uuid4().hex
            group_id = task_id
            sample_idx = None
            if task_id is not None and group_size > 1:
                # GroupedRolloutWorkflow passes "<group task id>:<sample index>".
                prefix, separator, suffix = task_id.rpartition(":")
                if separator and suffix.isdecimal():
                    group_id = prefix
                    sample_idx = int(suffix)
            self._append(
                session_id,
                "session_start",
                session_uuid=session_uuid,
                group_id=group_id,
                group_size=group_size,
                sample_idx=sample_idx,
            )
            return session_uuid

    def end_session(self, session_id: str) -> None:
        with self._lock:
            if session_id not in self._active_sessions:
                return
            at = time.time()
            if self._generating.pop(session_id, 0):
                self._append(session_id, "generate_end", at=at)
            self._append(session_id, "session_end", at=at)
            self._active_sessions.remove(session_id)
            self._task_values.pop(session_id, None)
            for key in tuple(self._phase_values):
                if key[0] == session_id:
                    del self._phase_values[key]

    def record_request(self, session_id: str) -> None:
        with self._lock:
            if session_id in self._active_sessions:
                self._append(session_id, "request_start")

    def begin_generate(self, session_id: str, *, record_request: bool = True) -> None:
        with self._lock:
            if session_id not in self._active_sessions:
                return
            count = self._generating.get(session_id, 0)
            self._generating[session_id] = count + 1
            if record_request:
                self._append(session_id, "request_start")
            if count == 0:
                self._append(session_id, "generate_start")

    def end_generate(self, session_id: str) -> None:
        with self._lock:
            count = self._generating.get(session_id, 0)
            if count <= 0:
                return
            if count == 1:
                del self._generating[session_id]
                self._append(session_id, "generate_end")
            else:
                self._generating[session_id] = count - 1

    async def wrap_stream(
        self, session_id: str, stream: AsyncGenerator[Any, None]
    ) -> AsyncGenerator[Any, None]:
        try:
            async for item in stream:
                yield item
        finally:
            self.end_generate(session_id)

    def bind_arena_task(
        self, session_id: str, task_id: str, attempt_id: str | None = None
    ) -> bool:
        with self._lock:
            if session_id not in self._active_sessions:
                return False
            previous = self._task_values.get(session_id)
            if attempt_id is None and previous is not None and previous[0] == task_id:
                attempt_id = previous[1]
            value = (task_id, attempt_id)
            if self._task_values.get(session_id) != value:
                self._task_values[session_id] = value
                self._append(
                    session_id, "arena_task", task_id=task_id, attempt_id=attempt_id
                )
            return True

    def record_phase(
        self,
        session_id: str,
        phase: str,
        started_at: float,
        ended_at: float | None,
        attempt_id: str | None = None,
        source_id: str | None = None,
    ) -> bool:
        """Record an explicit Arena interval or local proxy processing span."""
        if phase not in {
            "reward",
            "pause_wait",
            "toolcall",
            "pre_process",
            "post_process",
            "server_generate",
        }:
            return False
        with self._lock:
            if session_id not in self._active_sessions:
                return False
            key = (session_id, phase, attempt_id, source_id, started_at)
            if key in self._phase_values and self._phase_values[key] == ended_at:
                return True
            self._phase_values[key] = ended_at
            self._append(
                session_id,
                "phase_interval",
                at=ended_at if ended_at is not None else started_at,
                phase=phase,
                started_at=started_at,
                ended_at=ended_at,
                attempt_id=attempt_id,
                source_id=source_id,
            )
            return True

    def events_after(self, cursor: int, limit: int = 10_000) -> dict[str, Any]:
        with self._lock:
            oldest = self._events[0]["seq"] if self._events else self._next_seq
            lost = cursor < oldest - 1
            events = [event for event in self._events if event["seq"] > cursor][:limit]
            next_cursor = events[-1]["seq"] if events else max(cursor, oldest - 1)
            return {
                "events": events,
                "next_cursor": next_cursor,
                "lost": lost,
                "latest_seq": self._next_seq - 1,
            }
