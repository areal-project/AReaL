# SPDX-License-Identifier: Apache-2.0

"""Backend execution leases, independent of DataProxy ownership.

All mutations run under one lock without network I/O. Records live for the run:
neither an idle session nor an uncertain execution is expired by a timer.
"""

from __future__ import annotations

import asyncio
import random
import uuid
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, field_validator

PoolState = Literal["PAUSED", "READY", "FAILED", "CLOSED"]
ReleaseOutcome = Literal["completed", "aborted", "failed", "not_sent"]


class RoutingError(Exception):
    def __init__(self, code: str, message: str = ""):
        self.code = code
        super().__init__(message or code)


class PoolIdentity(BaseModel):
    run_id: str = Field(min_length=1)
    pool_id: str = Field(min_length=1)


class BackendSpec(BaseModel):
    backend_id: str = Field(min_length=1)
    backend_addr: str = Field(min_length=1)
    capacity: int = Field(gt=0)

    @field_validator("backend_addr")
    @classmethod
    def canonical_address(cls, address: str) -> str:
        parsed = urlsplit(address)
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path not in ("", "/")
        ):
            raise ValueError(
                "backend_addr must be an HTTP(S) origin without credentials"
            )
        return address.rstrip("/")


class RegisterBackends(PoolIdentity):
    backends: list[BackendSpec] = Field(min_length=1)
    weight_version: int = Field(ge=0)


class Acquire(PoolIdentity):
    request_id: str = Field(min_length=1)
    session_id: str | None = Field(default=None, min_length=1)


class LeaseResponse(BaseModel):
    lease_id: str
    backend_id: str
    backend_addr: str
    weight_version: int


class Release(PoolIdentity):
    lease_id: str
    outcome: ReleaseOutcome


class CloseSession(PoolIdentity):
    session_id: str = Field(min_length=1)


class SetPoolState(BaseModel):
    run_id: str = Field(min_length=1)
    control_seq: int = Field(ge=0)
    state: PoolState
    weight_version: int = Field(ge=0)
    old_processes_stopped: bool = False


@dataclass
class Backend:
    spec: BackendSpec
    outstanding: int = 0
    healthy: bool = True


@dataclass
class Lease:
    request: Acquire
    response: LeaseResponse
    outcome: ReleaseOutcome | None = None


@dataclass
class Pool:
    registration: RegisterBackends
    backends: dict[str, Backend]
    state: PoolState = "PAUSED"
    control_seq: int = -1
    version: int = 0
    last_command: SetPoolState | None = None
    sessions: dict[str, str] = field(default_factory=dict)
    closed_sessions: set[str] = field(default_factory=set)
    session_outstanding: dict[str, int] = field(default_factory=dict)
    requests: dict[str, Lease] = field(default_factory=dict)
    leases: dict[str, Lease] = field(default_factory=dict)

    @property
    def outstanding(self) -> int:
        return sum(b.outstanding for b in self.backends.values())


class BackendRegistry:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._pools: dict[str, Pool] = {}

    def _pool(self, pool_id: str, run_id: str) -> Pool:
        pool = self._pools.get(pool_id)
        if pool is None or pool.registration.run_id != run_id:
            raise RoutingError("STALE_RUN")
        return pool

    async def register(self, request: RegisterBackends) -> None:
        async with self._lock:
            if request.pool_id in self._pools:
                if self._pools[request.pool_id].registration == request:
                    return
                raise RoutingError("REGISTRATION_CONFLICT")
            ids = [b.backend_id for b in request.backends]
            addresses = [b.backend_addr.rstrip("/") for b in request.backends]
            if len(set(ids)) != len(ids) or len(set(addresses)) != len(addresses):
                raise RoutingError("DUPLICATE_BACKEND")
            owned = {
                b.spec.backend_addr.rstrip("/")
                for pool in self._pools.values()
                if pool.state != "CLOSED"
                for b in pool.backends.values()
            }
            if owned.intersection(addresses):
                raise RoutingError("BACKEND_ALREADY_OWNED")
            self._pools[request.pool_id] = Pool(
                registration=request.model_copy(deep=True),
                backends={
                    b.backend_id: Backend(b.model_copy()) for b in request.backends
                },
                version=request.weight_version,
            )

    async def acquire(self, request: Acquire) -> LeaseResponse:
        async with self._lock:
            pool = self._pool(request.pool_id, request.run_id)
            if pool.state in ("FAILED", "CLOSED"):
                raise RoutingError(pool.state)
            existing = pool.requests.get(request.request_id)
            if existing is not None:
                if existing.request != request:
                    raise RoutingError("REQUEST_CONFLICT")
                if existing.outcome is not None:
                    raise RoutingError("REQUEST_FINISHED")
                return existing.response.model_copy()
            if request.session_id in pool.closed_sessions:
                raise RoutingError("SESSION_CLOSED")
            if pool.state != "READY":
                raise RoutingError(pool.state)
            bound = pool.sessions.get(request.session_id)
            if bound is not None:
                backend = pool.backends[bound]
                if not backend.healthy:
                    raise RoutingError("UNAVAILABLE")
            else:
                candidates = [b for b in pool.backends.values() if b.healthy]
                if not candidates:
                    raise RoutingError("UNAVAILABLE")
                minimum = min(b.outstanding / b.spec.capacity for b in candidates)
                backend = random.choice(
                    [
                        b
                        for b in candidates
                        if b.outstanding / b.spec.capacity == minimum
                    ]
                )
            response = LeaseResponse(
                lease_id=uuid.uuid4().hex,
                backend_id=backend.spec.backend_id,
                backend_addr=backend.spec.backend_addr,
                weight_version=pool.version,
            )
            lease = Lease(request.model_copy(), response)
            pool.requests[request.request_id] = lease
            pool.leases[response.lease_id] = lease
            backend.outstanding += 1
            if request.session_id is not None:
                pool.sessions[request.session_id] = response.backend_id
                pool.session_outstanding[request.session_id] = (
                    pool.session_outstanding.get(request.session_id, 0) + 1
                )
            return response.model_copy()

    async def release(self, request: Release) -> None:
        async with self._lock:
            pool = self._pool(request.pool_id, request.run_id)
            lease = pool.leases.get(request.lease_id)
            if lease is None:
                raise RoutingError("UNKNOWN_LEASE")
            if lease.outcome is not None:
                return
            lease.outcome = request.outcome
            pool.backends[lease.response.backend_id].outstanding -= 1
            if lease.request.session_id is not None:
                pool.session_outstanding[lease.request.session_id] -= 1
            self._remove_closed_binding(pool, lease.request.session_id)

    @staticmethod
    def _remove_closed_binding(pool: Pool, session_id: str | None) -> None:
        if session_id in pool.closed_sessions and not pool.session_outstanding.get(
            session_id, 0
        ):
            pool.sessions.pop(session_id, None)
            pool.session_outstanding.pop(session_id, None)

    async def close_session(self, request: CloseSession) -> None:
        async with self._lock:
            pool = self._pool(request.pool_id, request.run_id)
            pool.closed_sessions.add(request.session_id)
            self._remove_closed_binding(pool, request.session_id)

    async def set_state(self, pool_id: str, request: SetPoolState) -> None:
        async with self._lock:
            pool = self._pool(pool_id, request.run_id)
            if request == pool.last_command:
                return
            if request.control_seq <= pool.control_seq:
                raise RoutingError("STALE_CONTROL")
            retiring = request.state == "CLOSED" and request.old_processes_stopped
            if request.old_processes_stopped and not retiring:
                raise RoutingError("INVALID_RETIREMENT")
            if (
                pool.state in ("FAILED", "CLOSED")
                and request.state != pool.state
                and not retiring
            ):
                # Failure requires whole-run restart, never a late resume.
                raise RoutingError("POOL_TERMINAL")
            if request.weight_version < pool.version:
                raise RoutingError("STALE_VERSION")
            if request.weight_version != pool.version and (
                pool.state != "PAUSED" or pool.outstanding
            ):
                raise RoutingError("NOT_DRAINED")
            if request.state == "CLOSED" and pool.outstanding and not retiring:
                raise RoutingError("NOT_DRAINED")
            if retiring:
                # This is an explicit control-plane assertion AFTER process
                # teardown, never a lease TTL or a hot takeover mechanism.
                for lease in pool.leases.values():
                    if lease.outcome is None:
                        lease.outcome = "failed"
                for backend in pool.backends.values():
                    backend.outstanding = 0
                pool.closed_sessions.update(pool.sessions)
                pool.sessions.clear()
                pool.session_outstanding.clear()
            pool.state = request.state
            pool.version = request.weight_version
            pool.control_seq = request.control_seq
            pool.last_command = request.model_copy()

    async def snapshot(self, pool_id: str, run_id: str) -> dict:
        async with self._lock:
            pool = self._pool(pool_id, run_id)
            return {
                "state": pool.state,
                "control_seq": pool.control_seq,
                "weight_version": pool.version,
                "outstanding": pool.outstanding,
                "sessions": len(pool.sessions),
                "closed_sessions": len(pool.closed_sessions),
                "backends": {
                    key: {
                        "outstanding": b.outstanding,
                        "capacity": b.spec.capacity,
                        "healthy": b.healthy,
                    }
                    for key, b in pool.backends.items()
                },
            }

    async def health_targets(self) -> list[tuple[str, str, str, str, int]]:
        async with self._lock:
            return [
                (
                    pool_id,
                    pool.registration.run_id,
                    key,
                    b.spec.backend_addr,
                    pool.control_seq,
                )
                for pool_id, pool in self._pools.items()
                # SGLang's /health may generate a token. A deliberate training
                # pause cannot answer it and must not become a backend failure.
                if pool.state == "READY"
                for key, b in pool.backends.items()
            ]

    async def update_health(
        self,
        pool_id: str,
        run_id: str,
        backend_id: str,
        healthy: bool,
        *,
        control_seq: int | None = None,
    ) -> None:
        async with self._lock:
            pool = self._pool(pool_id, run_id)
            if control_seq is not None and (
                pool.state != "READY" or pool.control_seq != control_seq
            ):
                # A probe begun before pause can finish after resume. Neither
                # its failure nor its success describes the current generation.
                return
            pool.backends[backend_id].healthy = healthy
