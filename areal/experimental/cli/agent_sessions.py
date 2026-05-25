# SPDX-License-Identifier: Apache-2.0

"""Per-service local session registry under ``~/.areal/agent/sessions/<service>.json``.

Each agent service holds N active sessions. Design §9.2 lays out the fields.
The CLI never persists the actual RL session API key in plain text — only a
boolean flag indicating whether one was negotiated. The session_key itself
is what the agent gateway's ``POST /v1/responses`` uses for session affinity
(via the ``user`` request field).

Default-session promotion mirrors ``inf_models``: when the current default
is removed, the next active session is promoted; otherwise default becomes
``None``.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from areal.experimental.cli.agent_state import agent_dir


def agent_sessions_dir() -> Path:
    d = agent_dir() / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def agent_sessions_file(service: str) -> Path:
    return agent_sessions_dir() / f"{service}.json"


@dataclass
class SessionEntry:
    key: str
    active: bool = True
    rl_session_key_present: bool = False
    last_reward: float | None = None
    created_at: float = 0.0
    last_active_at: float = 0.0
    session_timeout: float = 1800.0
    metadata: dict = field(default_factory=dict)

    def is_expired(self, now: float | None = None) -> bool:
        if not self.active:
            return False
        now = now or time.time()
        return (now - self.last_active_at) > self.session_timeout

    def touch(self) -> None:
        self.last_active_at = time.time()


@dataclass
class SessionRegistry:
    service: str
    default_session: str | None = None
    sessions: list[SessionEntry] = field(default_factory=list)

    @classmethod
    def load(cls, service: str) -> SessionRegistry:
        p = agent_sessions_file(service)
        if not p.exists():
            return cls(service=service, default_session=None, sessions=[])
        with open(p) as f:
            raw = json.load(f)
        return cls(
            service=raw.get("service", service),
            default_session=raw.get("default_session"),
            sessions=[SessionEntry(**s) for s in raw.get("sessions", [])],
        )

    def save(self) -> None:
        p = agent_sessions_file(self.service)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        doc = {
            "service": self.service,
            "default_session": self.default_session,
            "sessions": [asdict(s) for s in self.sessions],
        }
        with open(tmp, "w") as f:
            json.dump(doc, f, indent=2)
        os.replace(tmp, p)

    def get(self, key: str) -> SessionEntry | None:
        for s in self.sessions:
            if s.key == key:
                return s
        return None

    def active_sessions(self) -> list[SessionEntry]:
        return [s for s in self.sessions if s.active]

    def add(self, entry: SessionEntry) -> None:
        if self.get(entry.key) is not None:
            raise ValueError(f"Session {entry.key!r} already exists.")
        if not entry.created_at:
            entry.created_at = time.time()
        if not entry.last_active_at:
            entry.last_active_at = entry.created_at
        self.sessions.append(entry)
        if self.default_session is None and entry.active:
            self.default_session = entry.key

    def mark_ended(self, key: str) -> SessionEntry | None:
        for s in self.sessions:
            if s.key == key:
                s.active = False
                if self.default_session == key:
                    next_active = next(
                        (a for a in self.sessions if a.active and a.key != key), None
                    )
                    self.default_session = next_active.key if next_active else None
                return s
        return None


def generate_session_key() -> str:
    return f"sess-{uuid.uuid4().hex[:12]}"


def resolve_session_key(service: str, explicit: str | None) -> str:
    """Apply: explicit > default > sole-active > error.

    Only ACTIVE sessions are eligible for the implicit fallbacks. An explicit
    key is returned even if the local registry has never seen it (so users can
    refer to sessions known to the agent gateway but not locally tracked).
    """
    reg = SessionRegistry.load(service)
    if explicit:
        return explicit
    if reg.default_session:
        entry = reg.get(reg.default_session)
        if entry and entry.active:
            return reg.default_session
    active = reg.active_sessions()
    if len(active) == 1:
        return active[0].key
    if not active:
        raise SystemExit(
            f"No active sessions on service {service!r}. "
            f"Create one with `areal agent new_session`."
        )
    raise SystemExit(
        f"Multiple active sessions on {service!r}; specify one. "
        f"Known: {', '.join(s.key for s in active)}."
    )
