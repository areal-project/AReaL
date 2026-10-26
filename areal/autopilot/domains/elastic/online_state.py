# SPDX-License-Identifier: Apache-2.0

"""Private mutable counters and replica indexes; public observations stay frozen."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING

from areal.elastic.policy.basic.forecast import RequestView

if TYPE_CHECKING:
    from .online_decode import _Plan
    from .online_simulation import _Request


_REQUEST_FIELDS = tuple(f.name for f in fields(RequestView))


@dataclass(slots=True)
class RequestState:
    request_id: str
    episode_id: str
    turn: int
    input_tokens: int
    ready_at: float
    prefilled: int = 0
    generated: int = 0
    first_token_at: float | None = None
    last_token_at: float | None = None
    maximum_tbt_seconds: float = 0.0
    completed_at: float | None = None
    deployment: int | None = None
    replica: int | None = None
    version_counts: tuple[tuple[int, int], ...] = ()

    @classmethod
    def copy(cls, view: RequestView) -> RequestState:
        return cls(**{name: getattr(view, name) for name in _REQUEST_FIELDS})

    def snapshot(self) -> RequestView:
        return RequestView(**{name: getattr(self, name) for name in _REQUEST_FIELDS})

    def add_tokens(self, count: int, version: int, at: float, gap: float) -> None:
        self.generated += count
        if self.first_token_at is None:
            self.first_token_at = at
        self.last_token_at = at
        self.maximum_tbt_seconds = max(self.maximum_tbt_seconds, gap)
        if self.version_counts and self.version_counts[-1][0] == version:
            old = self.version_counts[-1][1]
            self.version_counts = self.version_counts[:-1] + ((version, old + count),)
        else:
            versions = dict(self.version_counts)
            versions[version] = versions.get(version, 0) + count
            self.version_counts = tuple(sorted(versions.items()))


@dataclass(slots=True)
class _DeferredDecode:
    plan: _Plan
    rows: tuple[RequestState, ...]
    version: int
    steps: int
    first_end: float
    last_end: float
    maximum_gap: float


@dataclass
class ReplicaState:
    rows: dict[str, _Request] = field(default_factory=dict)
    reserved_tokens: int = 0
    kv_tokens: int = 0
    revision: int = 0
    decode_plan: _Plan | None = None
    pending: _DeferredDecode | None = None

    def defer_decode(
        self,
        plan: _Plan,
        steps: int,
        first_end: float,
        last_end: float,
        maximum_gap: float,
    ) -> None:
        """Summarize committed tokens; KV and event accounting stay immediate."""
        pending = self.pending
        if pending is not None and (
            pending.plan is not plan or pending.version != plan.version
        ):
            self.flush_decode()
            pending = None
        if pending is None:
            self.pending = _DeferredDecode(
                plan,
                tuple(self.rows[key].view for key, *_ in plan.batch.slices),
                plan.version,
                steps,
                first_end,
                last_end,
                maximum_gap,
            )
        else:
            pending.maximum_gap = max(
                pending.maximum_gap, maximum_gap, first_end - pending.last_end
            )
            pending.steps += steps
            pending.last_end = last_end

    def flush_decode(self) -> None:
        pending = self.pending
        if pending is None:
            return
        self.pending = None
        for view in pending.rows:
            gap = max(
                pending.maximum_gap,
                pending.first_end - view.last_token_at
                if view.last_token_at is not None
                else 0.0,
            )
            if view.first_token_at is None:
                view.first_token_at = pending.first_end
            view.add_tokens(pending.steps, pending.version, pending.last_end, gap)

    def add(self, request: _Request) -> None:
        self.flush_decode()
        self.rows[request.view.request_id] = request
        self.reserved_tokens += request.view.input_tokens + request.reservation - 1
        self.kv_tokens += request.kv_tokens
        self.revision += 1

    def remove(self, request: _Request) -> None:
        self.flush_decode()
        del self.rows[request.view.request_id]
        self.reserved_tokens -= request.view.input_tokens + request.reservation - 1
        self.kv_tokens -= request.kv_tokens
        self.revision += 1
