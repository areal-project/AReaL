# SPDX-License-Identifier: Apache-2.0

"""Controller-side live v1 session state and additive rollout attribution."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import httpx

from areal.experimental.openai.proxy.session_trace import (
    RL_SESSION_TRACE_EVENTS_PATHNAME,
)


@dataclass
class SessionTiming:
    session_id: str
    started_at: float | None = None
    ended_at: float | None = None
    arena_task_id: str | None = None
    attempt_id: str | None = None
    request_count: int = 0
    last_progress_at: float | None = None
    generate_started_at: float | None = None
    generate_intervals: list[tuple[float, float]] = field(default_factory=list)
    phase_intervals: dict[tuple[str, str | None, float], float | None] = field(
        default_factory=dict
    )

    def apply(self, event: dict[str, Any]) -> None:
        kind = event["kind"]
        at = float(event["at"])
        self.last_progress_at = max(self.last_progress_at or at, at)
        if kind == "session_start":
            self.started_at = at
        elif kind == "session_end":
            self.ended_at = at
            if self.generate_started_at is not None:
                self.generate_intervals.append((self.generate_started_at, at))
                self.generate_started_at = None
        elif kind == "request_start":
            self.request_count += 1
        elif kind == "generate_start":
            if self.generate_started_at is None:
                self.generate_started_at = at
        elif kind == "generate_end":
            if self.generate_started_at is not None:
                self.generate_intervals.append((self.generate_started_at, at))
                self.generate_started_at = None
        elif kind == "arena_task":
            self.arena_task_id = str(event["task_id"])
            if event.get("attempt_id") is not None:
                self.attempt_id = str(event["attempt_id"])
        elif kind == "phase_interval":
            phase = str(event["phase"])
            start = float(event["started_at"])
            end_value = event.get("ended_at")
            attempt_id = event.get("attempt_id")
            self.phase_intervals[(phase, attempt_id, start)] = (
                float(end_value) if end_value is not None else None
            )
            if attempt_id is not None:
                self.attempt_id = str(attempt_id)

    def current_phase(self, now: float) -> tuple[str, float]:
        if self.ended_at is not None:
            return "done", self.ended_at
        for phase in ("pause_wait", "generate", "reward", "toolcall"):
            if phase == "generate" and self.generate_started_at is not None:
                return phase, self.generate_started_at
            active_starts = [
                start
                for (name, _, start), end in self.phase_intervals.items()
                if name == phase and start <= now and (end is None or end > now)
            ]
            if active_starts:
                return phase, min(active_starts)
        last_end = max(
            (
                end
                for end in (
                    [finish for _, finish in self.generate_intervals]
                    + [
                        finish
                        for finish in self.phase_intervals.values()
                        if finish is not None
                    ]
                )
                if end <= now
            ),
            default=self.started_at or now,
        )
        return "other", last_end

    def cumulative(self, phase: str, now: float) -> float:
        if phase == "generate":
            total = sum(end - start for start, end in self.generate_intervals)
            if self.generate_started_at is not None:
                total += max(0.0, now - self.generate_started_at)
            return total
        return sum(
            max(0.0, (end if end is not None else now) - start)
            for (name, _, start), end in self.phase_intervals.items()
            if name == phase
        )

    def boundaries(self) -> list[tuple[float, str, int]]:
        if self.started_at is None:
            return []
        boundaries = [(self.started_at, "session", 1)]
        if self.ended_at is not None:
            boundaries.append((self.ended_at, "session", -1))
        for start, end in self.generate_intervals:
            boundaries.extend(((start, "generate", 1), (end, "generate", -1)))
        if self.generate_started_at is not None:
            boundaries.append((self.generate_started_at, "generate", 1))
        for (phase, _, start), end in self.phase_intervals.items():
            boundaries.append((start, phase, 1))
            if end is not None:
                boundaries.append((end, phase, -1))
        return boundaries


class RolloutTraceObserver:
    """Poll one event stream per proxy and log bounded long-tail snapshots."""

    PHASES = ("generate", "reward", "pause_wait", "toolcall", "other")

    def __init__(
        self,
        proxy_addrs: list[str],
        admin_api_key: str,
        config: Any,
        logger: Any,
    ) -> None:
        self.proxy_addrs = list(proxy_addrs)
        self.admin_api_key = admin_api_key
        self.config = config
        self.logger = logger
        self._cursors = dict.fromkeys(self.proxy_addrs, 0)
        self._sessions: dict[tuple[str, str], SessionTiming] = {}
        self._lock = threading.Lock()
        self._poll_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._unreliable = False
        self._last_log = time.monotonic()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="rollout-trace-observer", daemon=True
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll_once()
                if (
                    time.monotonic() - self._last_log
                    >= self.config.log_interval_seconds
                ):
                    self.log_long_tails()
                    self._last_log = time.monotonic()
            except Exception:
                self.logger.exception("Rollout trace observer failed; will retry")
            self._stop.wait(self.config.collect_interval_seconds)

    def poll_once(self) -> bool:
        if not self.proxy_addrs:
            return True
        with (
            self._poll_lock,
            ThreadPoolExecutor(max_workers=min(16, len(self.proxy_addrs))) as pool,
        ):
            return all(list(pool.map(self._poll_proxy, self.proxy_addrs)))

    def _poll_proxy(self, addr: str) -> bool:
        with httpx.Client(timeout=5.0) as client:
            for _ in range(32):
                try:
                    response = client.get(
                        f"{addr.rstrip('/')}/{RL_SESSION_TRACE_EVENTS_PATHNAME}",
                        params={"after": self._cursors[addr]},
                        headers={"Authorization": f"Bearer {self.admin_api_key}"},
                    )
                    response.raise_for_status()
                    batch = response.json()
                    events = batch["events"]
                    next_cursor = int(batch["next_cursor"])
                    latest_seq = int(batch["latest_seq"])
                except (httpx.HTTPError, ValueError, KeyError, TypeError):
                    self.logger.warning(
                        "Rollout trace collection failed for proxy %s", addr
                    )
                    return False
                if batch.get("lost"):
                    self._unreliable = True
                    self.logger.warning(
                        "Rollout trace event buffer overflowed on proxy %s; "
                        "step attribution will remain in other",
                        addr,
                    )
                with self._lock:
                    for event in events:
                        key = (addr, str(event["session_id"]))
                        session = self._sessions.setdefault(
                            key, SessionTiming(session_id=key[1])
                        )
                        session.apply(event)
                self._cursors[addr] = next_cursor
                if next_cursor >= latest_seq:
                    return True
        self.logger.warning("Rollout trace did not catch up with proxy %s", addr)
        return False

    def log_long_tails(self) -> None:
        now = time.time()
        with self._lock:
            long_tails = [
                session
                for session in self._sessions.values()
                if session.started_at is not None
                and session.ended_at is None
                and now - session.started_at >= self.config.long_tail_seconds
            ]
            long_tails.sort(key=lambda session: session.started_at or now)
            lines = []
            for session in long_tails[: self.config.top_n]:
                phase, phase_start = session.current_phase(now)
                elapsed = now - session.started_at
                generate = session.cumulative("generate", now)
                reward = session.cumulative("reward", now)
                pause = session.cumulative("pause_wait", now)
                toolcall = session.cumulative("toolcall", now)
                other = max(0.0, elapsed - generate - reward - pause - toolcall)
                retry_count = (
                    str(max(0, int(session.attempt_id) - 1))
                    if session.attempt_id is not None and session.attempt_id.isdecimal()
                    else "?"
                )
                recent = (
                    datetime.fromtimestamp(session.last_progress_at)
                    .astimezone()
                    .strftime("%H:%M:%S")
                    if session.last_progress_at is not None
                    else "?"
                )
                lines.append(
                    f"{session.arena_task_id or '?'} / {session.attempt_id or '?'} "
                    f"| session={session.session_id} "
                    f"| elapsed={elapsed / 60:.1f}m "
                    f"| phase={phase} ({(now - phase_start) / 60:.1f}m) "
                    f"| generate={generate / 60:.1f}m "
                    f"| reward={reward / 60:.1f}m "
                    f"| pause={pause / 60:.1f}m | other={other / 60:.1f}m "
                    f"| last_progress={recent} | requests={session.request_count} "
                    f"| arena_retries={retry_count}"
                )
        if long_tails:
            self.logger.info(
                "Rollout long tails: %d active, showing %d\n%s",
                len(long_tails),
                len(lines),
                "\n".join(lines),
            )

    def attribute_step(
        self, started_at: float, ended_at: float, parent_seconds: float
    ) -> dict[str, float]:
        """Divide each wall-clock slice equally among active session phases."""
        if ended_at <= started_at or self._unreliable:
            return {
                "generate": 0.0,
                "reward": 0.0,
                "pause_wait": 0.0,
                "toolcall": 0.0,
                "other": parent_seconds,
            }
        with self._lock:
            boundaries = [
                (at, key, phase, delta)
                for key, session in self._sessions.items()
                for at, phase, delta in session.boundaries()
                if at <= ended_at
            ]
        boundaries.sort(key=lambda item: item[0])
        state: dict[tuple[str, str], dict[str, int]] = {}
        counts = dict.fromkeys(self.PHASES, 0)
        attributed = dict.fromkeys(self.PHASES, 0.0)
        previous = started_at

        def category(values: dict[str, int]) -> str | None:
            if values.get("session", 0) <= 0:
                return None
            for phase in ("pause_wait", "generate", "reward", "toolcall"):
                if values.get(phase, 0) > 0:
                    return phase
            return "other"

        for at, key, phase, delta in boundaries + [(ended_at, ("", ""), "", 0)]:
            if at > started_at:
                duration = max(0.0, min(at, ended_at) - previous)
                active_count = sum(counts.values())
                if active_count:
                    for name, count in counts.items():
                        attributed[name] += duration * count / active_count
                else:
                    attributed["other"] += duration
                previous = min(at, ended_at)
            if not phase:
                break
            values = state.setdefault(key, {})
            before = category(values)
            values[phase] = values.get(phase, 0) + delta
            after = category(values)
            if before is not None:
                counts[before] -= 1
            if after is not None:
                counts[after] += 1

        wall_seconds = ended_at - started_at
        result = {
            name: max(0.0, seconds * parent_seconds / wall_seconds)
            for name, seconds in attributed.items()
        }
        known = sum(value for name, value in result.items() if name != "other")
        if known > parent_seconds:
            scale = parent_seconds / known
            for name in result:
                if name != "other":
                    result[name] *= scale
            known = parent_seconds
        result["other"] = parent_seconds - known
        return result

    def prune_completed(self, before: float) -> None:
        """Drop sessions that cannot affect a future rollout window."""
        with self._lock:
            for key, session in tuple(self._sessions.items()):
                if session.ended_at is not None and session.ended_at <= before:
                    del self._sessions[key]
