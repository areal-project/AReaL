# SPDX-License-Identifier: Apache-2.0

"""Live inputs for the Reference evaluator on a shared run directory."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from statistics import median
from threading import Lock
from uuid import uuid4

from .evaluator.workload import (
    Backlog,
    EpisodeView,
    InflightCall,
    OnlineObservation,
    Turn,
)


class EventWriter:
    """Each producer or DataProxy owns one append-only stream per run."""

    def __init__(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.lock = Lock()
        self.path = directory / f"events-{uuid4().hex}.jsonl"

    def write(self, kind, **fields):
        with self.lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"kind": kind, **fields}, allow_nan=False) + "\n")


class LiveObservation:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.offsets = {}
        self.arrivals = {}
        self.sessions = {}
        self.calls = defaultdict(dict)
        self.receipts = {}
        self.ready = {}
        self.origin = None

    def _read(self):
        for path in sorted(self.directory.glob("events-*.jsonl")):
            with path.open("rb") as stream:
                stream.seek(self.offsets.get(path, 0))
                while line := stream.readline():
                    if not line.endswith(b"\n"):
                        break
                    row = json.loads(line)
                    key = row["request_id"]
                    kind = row["kind"]
                    if kind == "offered":
                        self.arrivals.setdefault(key, row["at"])
                    elif kind == "session":
                        self.sessions[key] = (row["session_id"], row["at"])
                    elif kind == "call":
                        self.calls[row["session_id"]][key] = row
                    elif kind == "finished":
                        self.receipts[key] = row
                    elif kind == "ready":
                        self.ready[row["session_id"]] = row["at"]
                    self.offsets[path] = stream.tell()

    def snapshot(self, version, *, at=None):
        at = time.time() if at is None else at
        self._read()
        if not self.arrivals:
            return None
        if self.origin is None:
            self.origin = min(self.arrivals.values())
        episodes = []
        for key, offered_at in self.arrivals.items():
            if offered_at > at:
                continue
            session = self.sessions.get(key)
            calls = (
                {
                    rid: call
                    for rid, call in self.calls.get(session[0], {}).items()
                    if call["completed_at"] <= at
                }
                if session and session[1] <= at
                else {}
            )
            ready_at = self.ready.get(session[0]) if session else None
            row = self.receipts.get(key)
            completed_at = None
            turns = ()
            if ready_at is not None and ready_at <= at:
                # The controller can take a batch before clients finish cleanup.
                turns = _serial_turns(calls, session[1], ready_at)
                completed_at = ready_at
            elif row and row["finished"] <= at:
                turns = _completed_turns(row, calls)
                completed_at = row["finished"]
            first = min(calls.values(), key=lambda c: c["ready_at"], default=None)
            episodes.append(
                EpisodeView(
                    key,
                    "online",
                    max(0.0, offered_at - self.origin),
                    completed_at - self.origin if turns else None,
                    turns,
                    first["input_tokens"] if first else None,
                )
            )
        return OnlineObservation(
            at - self.origin, tuple(episodes), version, version, False, False
        )

    def episode_ids(self, session_ids):
        requests = {sid: key for key, (sid, _) in self.sessions.items()}
        return tuple(requests.get(sid, "") for sid in session_ids)

    def backlog(self, view, pattern, actor_events, *, consumed=None):
        origin = self.origin
        calls = {
            key: [
                dict(
                    c,
                    ready_at=c["ready_at"] - origin,
                    completed_at=c["completed_at"] - origin,
                )
                for c in self.calls[sid].values()
            ]
            for key, (sid, _) in self.sessions.items()
        }
        sessions = {key: at - origin for key, (_, at) in self.sessions.items()}
        finished = {key: row["finished"] - origin for key, row in self.receipts.items()}
        for key, (sid, _) in self.sessions.items():
            if sid in self.ready:
                finished[key] = min(
                    finished.get(key, float("inf")), self.ready[sid] - origin
                )
        events = [(kind, at - origin, count) for kind, at, count in actor_events]
        return estimate_backlog(
            view, pattern, sessions, calls, finished, events, consumed=consumed
        )


def _completed_turns(row, calls):
    if row["status"] != "completed":
        return ()
    expected = row["api_calls"]
    if not calls or len(calls) != expected:
        return ()
    return _serial_turns(calls, row["started"], row["finished"])


def _serial_turns(calls, started, finished):
    ordered = sorted(calls.values(), key=lambda event: event["ready_at"])
    turns = []
    for i, event in enumerate(ordered):
        start, end = event["ready_at"], event["completed_at"]
        if not started <= start <= end <= finished:
            return ()
        if event["input_tokens"] <= 0 or event["output_tokens"] <= 0:
            return ()
        gap = None
        if i + 1 < len(ordered):
            gap = ordered[i + 1]["ready_at"] - end
            if gap < 0:
                return ()  # The evaluator models serial turns.
        turns.append(Turn(event["input_tokens"], event["output_tokens"], gap))
    return tuple(turns)


def estimate_backlog(
    view, pattern, sessions, calls, finished, actor_events, *, consumed=None
):
    """All timestamps use the external trace clock. Call records are completion events."""
    now = view.now
    templates = [e.completed_turns for e in pattern.templates]
    if not templates or not pattern.arrivals:
        return None
    visible = {
        k: sorted(
            (c for c in events if c["completed_at"] <= now), key=lambda c: c["ready_at"]
        )
        for k, events in calls.items()
    }
    sample_calls = [c for e in pattern.templates for c in visible.get(e.episode_id, ())]
    first_delays = [
        c["ttft_seconds"] for c in sample_calls if c.get("ttft_seconds") is not None
    ]
    intervals = [
        c["tpot_seconds"] for c in sample_calls if (c.get("tpot_seconds") or 0) > 0
    ]
    ttft = median(first_delays) if first_delays else None
    tpot = median(intervals) if intervals else None
    queued, running, tools, shapes = [], [], [], {}
    receipts = sorted(
        (e.completed_at, e.episode_id, e.completed_turns)
        for e in view.episodes
        if e.completed_turns and e.completed_at is not None and e.completed_at <= now
    )
    taken = 0
    started = publish_start = None
    durations = {"take": [], "publish": []}
    for kind, at, count in actor_events:
        if at > now:
            break
        if kind == "take":
            taken += count
            started = at
        elif kind == "publish":
            if started is not None:
                durations["take"].append(at - started)
            publish_start = at
        elif kind == "done":
            if publish_start is not None:
                durations["publish"].append(at - publish_start)
            started = publish_start = None

    # Historical replay inputs may have counts only.
    pending = (
        receipts[taken:]
        if consumed is None
        else [(at, key, turns) for at, key, turns in receipts if key not in consumed]
    )
    shapes.update((k, ts) for _, k, ts in pending)
    actor_event = None
    if started is not None:
        phase = "publish" if publish_start is not None else "take"
        at = publish_start if publish_start is not None else started
        # Before the first phase duration has been observed, schedule its
        # completion at the snapshot time and expose the missing estimate.
        duration = median(durations[phase][-5:]) if durations[phase] else 0.0
        actor_event = (
            "published" if phase == "publish" else "actor",
            max(0.0, at + duration - now),
        )
    for e in view.episodes:
        k = e.episode_id
        if e.available_at > now or finished.get(k, float("inf")) <= now:
            continue
        index = int.from_bytes(sha256(k.encode()).digest()[:8], "little")
        if sessions.get(k, float("inf")) > now:
            queued.append((k, e.available_at))
            shapes[k] = templates[index % len(templates)]
            continue
        seen = visible.get(k, ())
        n = len(seen)
        suffixes = [ts[n:] for ts in templates if len(ts) > n]
        tail = (
            suffixes[index % len(suffixes)]
            if suffixes
            else (templates[index % len(templates)][-1],)
        )
        prefix = []
        for i, c in enumerate(seen):
            if i + 1 < n:
                gap = max(0.0, seen[i + 1]["ready_at"] - c["completed_at"])
            else:
                waits = [ts[n - 1].tool_seconds for ts in templates if len(ts) > n]
                gap = median(waits) if waits else 0.0
            prefix.append(Turn(c["input_tokens"], c["output_tokens"], gap))
        shapes[k] = tuple(prefix) + tail
        start = (
            seen[-1]["completed_at"] + prefix[-1].tool_seconds if seen else sessions[k]
        )
        if start > now:
            tools.append((k, n, start))
            continue
        first = start + ttft if ttft is not None else None
        generated = 0
        if first is not None and first <= now and tpot is not None:
            generated = min(tail[0].output_tokens - 1, 1 + int((now - first) / tpot))
        # Replica placement and resident KV are unobserved. Model a re-prefill
        # of the estimated output prefix instead of inventing a resident replica.
        running.append(
            InflightCall(
                k,
                n,
                start if n else e.available_at,
                first if generated else None,
                retained=generated,
            )
        )
    return Backlog(
        shapes,
        tuple(queued),
        tuple(running),
        tuple(tools),
        tuple(k for _, k, _ in pending),
        actor_event,
    )
