# SPDX-License-Identifier: Apache-2.0

"""Controller-side live v1 session state and additive rollout attribution."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import httpx

from areal.experimental.openai.proxy.session_trace import (
    RL_SESSION_TRACE_EVENTS_PATHNAME,
)
from areal.utils.rollout_phase_context import step_phase_recorder


@dataclass
class SessionTiming:
    session_id: str
    session_uuid: str | None = None
    started_at: float | None = None
    ended_at: float | None = None
    group_id: str | None = None
    group_size: int = 1
    sample_idx: int | None = None
    arena_task_id: str | None = None
    attempt_id: str | None = None
    request_count: int = 0
    last_progress_at: float | None = None
    generate_started_at: float | None = None
    generate_intervals: list[tuple[float, float]] = field(default_factory=list)
    phase_intervals: dict[tuple[str, str | None, str | None, float], float | None] = (
        field(default_factory=dict)
    )

    def apply(self, event: dict[str, Any]) -> None:
        kind = event["kind"]
        at = float(event["at"])
        self.last_progress_at = max(self.last_progress_at or at, at)
        if kind == "session_start":
            self.started_at = at
            self.session_uuid = event.get("session_uuid")
            if event.get("group_id") is not None:
                self.group_id = str(event["group_id"])
            self.group_size = max(1, int(event.get("group_size") or 1))
            if event.get("sample_idx") is not None:
                self.sample_idx = int(event["sample_idx"])
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
            source_id = event.get("source_id")
            self.phase_intervals[(phase, attempt_id, source_id, start)] = (
                float(end_value) if end_value is not None else None
            )
            if attempt_id is not None:
                self.attempt_id = str(attempt_id)

    def current_phase(self, now: float) -> tuple[str, float]:
        if self.ended_at is not None:
            return "done", self.ended_at
        for phase in (
            "pre_process",
            "post_process",
            "pause_wait",
            "reward",
            "toolcall",
            "generate",
        ):
            if phase == "generate" and self.generate_started_at is not None:
                return phase, self.generate_started_at
            active_starts = [
                start
                for (name, _, _, start), end in self.phase_intervals.items()
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

    def cumulative(
        self, phase: str, now: float, *, server_generate: bool = False
    ) -> float:
        if phase == "generate":
            if server_generate:
                intervals = sorted(
                    (start, end)
                    for (name, _, _, start), end in self.phase_intervals.items()
                    if name == "server_generate" and end is not None
                )
                total = 0.0
                latest_end = 0.0
                for start, end in intervals:
                    total += max(0.0, end - max(start, latest_end))
                    latest_end = max(latest_end, end)
                return total
            total = sum(end - start for start, end in self.generate_intervals)
            if self.generate_started_at is not None:
                total += max(0.0, now - self.generate_started_at)
            return total
        return sum(
            max(0.0, (end if end is not None else now) - start)
            for (name, _, _, start), end in self.phase_intervals.items()
            if name == phase
        )

    def boundaries(
        self, *, server_generate: bool = False
    ) -> list[tuple[float, str, int]]:
        if self.started_at is None:
            return []
        boundaries = [(self.started_at, "session", 1)]
        if self.ended_at is not None:
            boundaries.append((self.ended_at, "session", -1))
        if not server_generate:
            for start, end in self.generate_intervals:
                boundaries.extend(((start, "generate", 1), (end, "generate", -1)))
            if self.generate_started_at is not None:
                boundaries.append((self.generate_started_at, "generate", 1))
        for (phase, _, _, start), end in self.phase_intervals.items():
            if phase == "server_generate":
                if not server_generate or end is None:
                    continue
                phase = "generate"
            boundaries.append((start, phase, 1))
            if end is not None:
                boundaries.append((end, phase, -1))
        return boundaries

    @staticmethod
    def category(
        values: dict[str, int], *, prefer_server_generate: bool = False
    ) -> str | None:
        if values.get("session", 0) <= 0:
            return None
        if prefer_server_generate and values.get("generate", 0) > 0:
            return "generate"
        for phase in (
            "pre_process",
            "post_process",
            "pause_wait",
            "reward",
            "toolcall",
            "schedule_requests",
            "data_collect",
            "generate",
        ):
            if values.get(phase, 0) > 0:
                return phase
        return "other"

    def phase_durations(
        self,
        started_at: float,
        ended_at: float,
        *,
        server_generate: bool = False,
    ) -> dict[str, float]:
        """Return exclusive session wall time clipped to a requested window."""
        totals = {
            name: 0.0
            for name in (
                "pre_process",
                "generate",
                "post_process",
                "reward",
                "pause_wait",
                "toolcall",
                "other",
            )
        }
        if self.started_at is None or ended_at <= started_at:
            return totals
        values: dict[str, int] = {}
        previous = started_at
        boundaries = sorted(
            (
                item
                for item in self.boundaries(server_generate=server_generate)
                if item[0] <= ended_at
            ),
            key=lambda item: item[0],
        )
        for at, phase, delta in boundaries + [(ended_at, "", 0)]:
            if at > started_at:
                duration = max(0.0, min(at, ended_at) - previous)
                current = self.category(values, prefer_server_generate=server_generate)
                if current is not None:
                    totals[current] += duration
                previous = min(at, ended_at)
            if not phase:
                break
            values[phase] = values.get(phase, 0) + delta
        return totals


@dataclass
class GroupProgress:
    size: int
    finished: int = 0


class RolloutTraceObserver:
    """Poll proxy events for step attribution and optional progress snapshots."""

    PHASES = (
        "schedule_requests",
        "pre_process",
        "generate",
        "post_process",
        "reward",
        "pause_wait",
        "data_collect",
        "toolcall",
        "other",
    )

    def __init__(
        self,
        proxy_addrs: list[str],
        admin_api_key: str,
        config: Any,
        logger: Any,
        use_sglang_server_timing: bool = False,
    ) -> None:
        self.proxy_addrs = list(proxy_addrs)
        self.admin_api_key = admin_api_key
        self.config = config
        self.logger = logger
        self.use_sglang_server_timing = use_sglang_server_timing
        self._cursors = dict.fromkeys(self.proxy_addrs, 0)
        self._sessions: dict[tuple[str, str], SessionTiming] = {}
        self._controller_intervals: list[tuple[str, float, float]] = []
        self._groups: dict[str, GroupProgress] = {}
        self._finished_count = 0
        self._lock = threading.Lock()
        self._poll_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._unreliable = False
        self._last_log = time.monotonic()

    @contextmanager
    def active_step(self):
        token = step_phase_recorder.set(self.record_controller_interval)
        try:
            yield
        finally:
            step_phase_recorder.reset(token)

    def record_controller_interval(
        self, phase: str, started_at: float, ended_at: float
    ) -> None:
        if phase not in {"schedule_requests", "data_collect"}:
            raise ValueError(f"Unsupported controller rollout phase: {phase}")
        if ended_at <= started_at:
            return
        with self._lock:
            self._controller_intervals.append((phase, started_at, ended_at))

    def load_direct_sglang_sessions(self, sessions: list[dict[str, Any]]) -> None:
        """Load worker session bounds and SGLang spans for one direct-workflow step."""
        loaded: dict[tuple[str, str], SessionTiming] = {}
        for item in sessions:
            session_uuid = str(item["session_uuid"])
            session = SessionTiming(session_id=session_uuid)
            session.apply(
                {
                    "kind": "session_start",
                    "at": float(item["started_at"]),
                    "session_uuid": session_uuid,
                }
            )
            for rid, start, end in item["intervals"]:
                session.apply(
                    {
                        "kind": "phase_interval",
                        "at": end,
                        "phase": "server_generate",
                        "started_at": start,
                        "ended_at": end,
                        "source_id": rid,
                    }
                )
            if item["ended_at"] is not None:
                session.apply({"kind": "session_end", "at": float(item["ended_at"])})
            loaded[("direct", session_uuid)] = session
        with self._lock:
            self._sessions = loaded

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
                    if getattr(self.config, "log_mode", "long_tail") == "progress":
                        self.log_progress()
                    else:
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
                        session = self._sessions.get(key)
                        if session is None or (
                            event["kind"] == "session_start"
                            and session.ended_at is not None
                        ):
                            session = SessionTiming(session_id=key[1])
                            self._sessions[key] = session
                        was_started = session.started_at is not None
                        was_ended = session.ended_at is not None
                        session.apply(event)
                        if event["kind"] == "session_start" and not was_started:
                            if session.group_id is not None:
                                self._groups.setdefault(
                                    session.group_id, GroupProgress(session.group_size)
                                )
                        elif event["kind"] == "session_end" and not was_ended:
                            self._finished_count += 1
                            if session.group_id in self._groups:
                                group = self._groups[session.group_id]
                                group.finished += 1
                                if group.finished >= group.size:
                                    del self._groups[session.group_id]
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
                durations = session.phase_durations(
                    session.started_at,
                    now,
                    server_generate=self.use_sglang_server_timing,
                )
                generate = durations["generate"]
                reward = durations["reward"]
                pause = durations["pause_wait"]
                other = durations["other"]
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
                    f"| session={session.session_id} uuid={session.session_uuid or '?'} "
                    f"| elapsed={elapsed / 60:.1f}m "
                    f"| phase={phase} ({(now - phase_start) / 60:.1f}m) "
                    f"| pre_process={durations['pre_process'] / 60:.1f}m "
                    f"| generate={generate / 60:.1f}m "
                    f"| post_process={durations['post_process'] / 60:.1f}m "
                    f"| reward={reward / 60:.1f}m "
                    f"| pause={pause / 60:.1f}m "
                    f"| toolcall={durations['toolcall'] / 60:.1f}m "
                    f"| {'other_or_pending' if self.use_sglang_server_timing else 'other'}={other / 60:.1f}m "
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

    def log_progress(self) -> None:
        """Log AReaL-observed session progress without Arena phase inference."""
        now = time.time()
        with self._lock:
            running = [
                session
                for session in self._sessions.values()
                if session.started_at is not None and session.ended_at is None
            ]
            running.sort(key=lambda session: session.started_at or now)
            selected = (
                running
                if self.config.show_all_sessions
                else running[: self.config.top_n]
            )
            lines = []
            for session in selected:
                assert session.started_at is not None
                elapsed = max(0.0, now - session.started_at)
                generate = session.cumulative(
                    "generate", now, server_generate=self.use_sglang_server_timing
                )
                non_generate = max(0.0, elapsed - generate)
                last_generate_end = max(
                    (end for _, end in session.generate_intervals), default=None
                )
                if session.generate_started_at is not None:
                    phase = "generate"
                    phase_started_at = session.generate_started_at
                else:
                    phase = "non_generate"
                    phase_started_at = last_generate_end or session.started_at
                group = self._groups.get(session.group_id or "")
                group_progress = (
                    f"{group.finished}/{group.size}"
                    if group is not None
                    else f"?/{session.group_size}"
                )
                recent = (
                    datetime.fromtimestamp(last_generate_end)
                    .astimezone()
                    .strftime("%H:%M:%S")
                    if last_generate_end is not None
                    else "?"
                )
                lines.append(
                    f"group={session.group_id or '?'} finished={group_progress} "
                    f"| sample={session.sample_idx if session.sample_idx is not None else '?'} "
                    f"| session={session.session_id} uuid={session.session_uuid or '?'} "
                    f"| arena_task={session.arena_task_id or '?'} "
                    f"arena_attempt={session.attempt_id or '?'} "
                    f"| elapsed={elapsed / 60:.1f}m "
                    f"| phase={phase} ({max(0.0, now - phase_started_at) / 60:.1f}m) "
                    f"| generate={generate / 60:.1f}m "
                    f"| {'non_generate_or_pending' if self.use_sglang_server_timing else 'non_generate'}={non_generate / 60:.1f}m "
                    f"| last_generate_end={recent} | requests={session.request_count}"
                )
            finished = self._finished_count
            unreliable = self._unreliable
        header = (
            f"Rollout progress: running={len(running)} finished={finished} "
            f"showing={len(lines)}"
        )
        if unreliable:
            header += " trace_lost=true"
        if lines:
            header += "\n" + "\n".join(lines)
        self.logger.info("%s", header)

    def attribute_step(
        self, started_at: float, ended_at: float, parent_seconds: float
    ) -> dict[str, float]:
        """Divide each wall-clock slice equally among active session phases."""
        if ended_at <= started_at or self._unreliable:
            return {
                name: parent_seconds if name == "other" else 0.0 for name in self.PHASES
            }
        with self._lock:
            boundaries = [
                (at, key, phase, delta)
                for key, session in self._sessions.items()
                for at, phase, delta in session.boundaries(
                    server_generate=self.use_sglang_server_timing
                )
                if at <= ended_at
            ]
            for index, (phase, start, end) in enumerate(self._controller_intervals):
                if start >= ended_at or end <= started_at:
                    continue
                key = ("__rollout_controller__", str(index))
                boundaries.extend(
                    (
                        (start, key, "session", 1),
                        (start, key, phase, 1),
                        (end, key, phase, -1),
                        (end, key, "session", -1),
                    )
                )
        boundaries.sort(key=lambda item: item[0])
        state: dict[tuple[str, str], dict[str, int]] = {}
        counts = dict.fromkeys(self.PHASES, 0)
        attributed = dict.fromkeys(self.PHASES, 0.0)
        previous = started_at

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
            before = SessionTiming.category(
                values, prefer_server_generate=self.use_sglang_server_timing
            )
            values[phase] = values.get(phase, 0) + delta
            after = SessionTiming.category(
                values, prefer_server_generate=self.use_sglang_server_timing
            )
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
            self._controller_intervals = [
                interval
                for interval in self._controller_intervals
                if interval[2] > before
            ]
            for key, session in tuple(self._sessions.items()):
                if session.ended_at is not None and session.ended_at <= before:
                    del self._sessions[key]
