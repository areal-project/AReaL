# SPDX-License-Identifier: Apache-2.0

"""Live workload observations from shared run event streams."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from threading import Lock
from uuid import uuid4

from .evaluator.workload import (
    EpisodeView,
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
