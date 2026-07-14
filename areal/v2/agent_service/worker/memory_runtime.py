# SPDX-License-Identifier: Apache-2.0

"""Worker-local ownership for authorized Agent Memory sessions.

This default-off runtime is deliberately smaller than a Worker integration.  A
trusted host passes an already authenticated :class:`MemoryPrincipalV1`; HTTP
metadata, session keys, assignment scopes, and hop credentials are not identity
inputs here.  The runtime owns one :class:`AuthorizedMemoryAgentBroker` and
publishes only a descriptive reservation.  It does not expose the broker's
private session handle or enable any Worker route.

The reservation's audience and incarnation are non-secret replay domains, not
bearer credentials.  Runtime operations accept only the exact object they
issued for their current private state.  Reconstructing an equal dataclass is
therefore descriptive forgery, not local authority.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from areal.v2.agent_service.memory import (
    MemoryAgentCoordinatorClosedError,
    MemoryAgentSessionConflictError,
)
from areal.v2.agent_service.memory_authorization import (
    MemoryPrincipalV1,
    MemorySessionIncarnationV1,
    MemoryWorkerAudienceV1,
)
from areal.v2.agent_service.memory_broker import (
    AuthorizedMemoryAgentBroker,
    AuthorizedMemorySessionV1,
)


def _principal(value: object) -> MemoryPrincipalV1:
    if type(value) is not MemoryPrincipalV1:
        raise TypeError("principal must be a MemoryPrincipalV1")
    value.canonical_bytes()
    return MemoryPrincipalV1(issuer=value.issuer, subject=value.subject)


def _session_key(value: object) -> str:
    if type(value) is not str:
        raise TypeError("session_key must be a str")
    if not value.strip():
        raise ValueError("session_key must not be blank")
    try:
        value.encode("utf-8", "strict")
    except UnicodeEncodeError as error:
        raise ValueError("session_key must be valid UTF-8") from error
    return value


@dataclass(frozen=True, slots=True)
class MemoryWorkerSessionReservationV1:
    """Public description of one Worker-local session reservation.

    This record intentionally contains no principal, grant, secret, or private
    broker handle.  Its values may be logged or carried by a future protocol,
    but possession of them authorizes no Memory operation.
    """

    session_key: str
    session: MemorySessionIncarnationV1
    audience: MemoryWorkerAudienceV1

    def __post_init__(self) -> None:
        if type(self.session) is not MemorySessionIncarnationV1:
            raise TypeError("session must be a MemorySessionIncarnationV1")
        if type(self.audience) is not MemoryWorkerAudienceV1:
            raise TypeError("audience must be a MemoryWorkerAudienceV1")
        session = MemorySessionIncarnationV1(
            session_key=self.session.session_key,
            incarnation_id=self.session.incarnation_id,
        )
        audience = MemoryWorkerAudienceV1(self.audience.audience_id)
        if type(self.session_key) is not str or self.session_key != session.session_key:
            raise ValueError("session_key must match the session incarnation")
        object.__setattr__(self, "session_key", session.session_key)
        object.__setattr__(self, "session", session)
        object.__setattr__(self, "audience", audience)


@dataclass(slots=True, eq=False)
class _ReservationState:
    descriptor: MemoryWorkerSessionReservationV1
    broker_session: AuthorizedMemorySessionV1
    principal_issuer: str
    principal_subject: str
    session_key: str
    incarnation_id: str
    audience_id: str
    closing: bool = False
    close_task: asyncio.Task[None] | None = None


@dataclass(frozen=True, slots=True)
class _ReservationAttempt:
    principal_issuer: str
    principal_subject: str
    task: asyncio.Task[MemoryWorkerSessionReservationV1]


class AuthorizedMemoryWorkerRuntime:
    """Own one authorized broker and its Worker-local session reservations.

    Construction does not register routes or make Memory reachable from an
    ``AgentRequest``.  The caller is trusted host code and remains responsible
    for establishing ``MemoryPrincipalV1`` outside untrusted request data.
    Passing a broker transfers its ownership: the caller must not wrap, operate,
    or close that broker afterwards.  ``aclose`` owns and closes it.
    """

    def __init__(self, broker: AuthorizedMemoryAgentBroker) -> None:
        """Take exclusive lifecycle ownership of ``broker``."""

        if type(broker) is not AuthorizedMemoryAgentBroker:
            raise TypeError("broker must be an AuthorizedMemoryAgentBroker")
        audience = broker.audience
        audience.canonical_bytes()
        broker._claim_worker_runtime()
        self.__broker = broker
        self.__audience_id = audience.audience_id
        self.__state_lock = asyncio.Lock()
        self.__owner_loop: asyncio.AbstractEventLoop | None = None
        self.__sessions: dict[str, _ReservationState] = {}
        self.__pending: dict[str, _ReservationAttempt] = {}
        self.__closed = False
        self.__shutdown_task: asyncio.Task[None] | None = None

    @property
    def audience(self) -> MemoryWorkerAudienceV1:
        """Return a detached description of this Worker runtime incarnation."""

        return MemoryWorkerAudienceV1(self.__audience_id)

    def _running_loop(self) -> asyncio.AbstractEventLoop:
        loop = asyncio.get_running_loop()
        if self.__owner_loop is None:
            self.__owner_loop = loop
        elif self.__owner_loop is not loop:
            raise RuntimeError(
                "AuthorizedMemoryWorkerRuntime cannot be shared across event loops"
            )
        return loop

    def _ensure_open(self) -> None:
        if self.__closed:
            raise MemoryAgentCoordinatorClosedError("Memory Worker runtime is closed")

    @staticmethod
    def _state_is_intact(state: _ReservationState) -> bool:
        try:
            state.descriptor.session.canonical_bytes()
            state.descriptor.audience.canonical_bytes()
            state.broker_session.principal.canonical_bytes()
            state.broker_session.session.canonical_bytes()
            state.broker_session.audience.canonical_bytes()
        except (AttributeError, TypeError, ValueError):
            return False
        return (
            state.descriptor.session_key,
            state.descriptor.session.session_key,
            state.descriptor.session.incarnation_id,
            state.descriptor.audience.audience_id,
            state.broker_session.principal.issuer,
            state.broker_session.principal.subject,
            state.broker_session.session.session_key,
            state.broker_session.session.incarnation_id,
            state.broker_session.audience.audience_id,
        ) == (
            state.session_key,
            state.session_key,
            state.incarnation_id,
            state.audience_id,
            state.principal_issuer,
            state.principal_subject,
            state.session_key,
            state.incarnation_id,
            state.audience_id,
        )

    def _reservation_state(
        self,
        descriptor: MemoryWorkerSessionReservationV1,
        *,
        allow_closing: bool = False,
    ) -> _ReservationState:
        if type(descriptor) is not MemoryWorkerSessionReservationV1:
            raise TypeError("reservation must be a MemoryWorkerSessionReservationV1")
        state = self.__sessions.get(descriptor.session_key)
        if (
            state is None
            or state.descriptor is not descriptor
            or not self._state_is_intact(state)
        ):
            raise MemoryAgentSessionConflictError(
                "Memory reservation is not current for this Worker runtime"
            )
        if state.closing and not allow_closing:
            raise MemoryAgentSessionConflictError("Memory reservation is closing")
        return state

    async def _reserve(
        self,
        principal: MemoryPrincipalV1,
        session_key: str,
    ) -> MemoryWorkerSessionReservationV1:
        current_task = asyncio.current_task()
        try:
            broker_session = await self.__broker.open_session(principal, session_key)
            async with self.__state_lock:
                self._ensure_open()
                try:
                    broker_session.principal.canonical_bytes()
                    broker_session.session.canonical_bytes()
                    broker_session.audience.canonical_bytes()
                except (AttributeError, TypeError, ValueError) as error:
                    raise MemoryAgentSessionConflictError(
                        "broker returned a malformed Memory session"
                    ) from error
                if type(broker_session) is not AuthorizedMemorySessionV1 or (
                    broker_session.principal.issuer,
                    broker_session.principal.subject,
                    broker_session.session.session_key,
                    broker_session.audience.audience_id,
                ) != (
                    principal.issuer,
                    principal.subject,
                    session_key,
                    self.__audience_id,
                ):
                    raise MemoryAgentSessionConflictError(
                        "broker returned a session outside the reservation context"
                    )
                descriptor = MemoryWorkerSessionReservationV1(
                    session_key=session_key,
                    session=MemorySessionIncarnationV1(
                        session_key=session_key,
                        incarnation_id=broker_session.session.incarnation_id,
                    ),
                    audience=MemoryWorkerAudienceV1(self.__audience_id),
                )
                self.__sessions[session_key] = _ReservationState(
                    descriptor=descriptor,
                    broker_session=broker_session,
                    principal_issuer=broker_session.principal.issuer,
                    principal_subject=broker_session.principal.subject,
                    session_key=session_key,
                    incarnation_id=broker_session.session.incarnation_id,
                    audience_id=self.__audience_id,
                )
                return descriptor
        finally:
            async with self.__state_lock:
                pending = self.__pending.get(session_key)
                if pending is not None and pending.task is current_task:
                    self.__pending.pop(session_key, None)

    async def reserve_session(
        self,
        principal: MemoryPrincipalV1,
        session_key: str,
    ) -> MemoryWorkerSessionReservationV1:
        """Reserve ``session_key`` for one trusted principal, idempotently.

        The runtime owns and shields the per-key broker task.  Caller
        cancellation only abandons that wait; it cannot discard a completed
        broker binding or make the key available to another principal.
        """

        self._running_loop()
        principal = _principal(principal)
        session_key = _session_key(session_key)
        async with self.__state_lock:
            self._ensure_open()
            existing = self.__sessions.get(session_key)
            if existing is not None:
                if existing.closing:
                    raise MemoryAgentSessionConflictError(
                        "Memory reservation is closing"
                    )
                if not self._state_is_intact(existing):
                    raise MemoryAgentSessionConflictError(
                        "Memory reservation state disagrees with its broker"
                    )
                if (
                    existing.principal_issuer,
                    existing.principal_subject,
                ) != (principal.issuer, principal.subject):
                    raise MemoryAgentSessionConflictError(
                        "session key is already bound to another principal"
                    )
                return existing.descriptor

            pending = self.__pending.get(session_key)
            if pending is not None:
                if (pending.principal_issuer, pending.principal_subject) != (
                    principal.issuer,
                    principal.subject,
                ):
                    raise MemoryAgentSessionConflictError(
                        "session key is being bound to another principal"
                    )
                task = pending.task
            else:
                task = asyncio.create_task(
                    self._reserve(principal, session_key),
                    name=f"areal-memory-worker-reserve:{session_key}",
                )
                self.__pending[session_key] = _ReservationAttempt(
                    principal_issuer=principal.issuer,
                    principal_subject=principal.subject,
                    task=task,
                )
                task.add_done_callback(self._observe_reservation_task)
        return await asyncio.shield(task)

    @staticmethod
    def _observe_reservation_task(
        task: asyncio.Task[MemoryWorkerSessionReservationV1],
    ) -> None:
        if not task.cancelled():
            task.exception()

    @staticmethod
    def _observe_task(task: asyncio.Task[None]) -> None:
        if not task.cancelled():
            task.exception()

    async def _close_state(self, state: _ReservationState) -> None:
        try:
            await self.__broker.close_session(state.broker_session)
        except BaseException:
            async with self.__state_lock:
                if self.__sessions.get(state.session_key) is state:
                    state.close_task = None
            raise
        else:
            async with self.__state_lock:
                if self.__sessions.get(state.session_key) is state:
                    self.__sessions.pop(state.session_key, None)

    async def close_session(
        self,
        reservation: MemoryWorkerSessionReservationV1,
    ) -> None:
        """Close one exact reservation; equal or stale records are rejected."""

        self._running_loop()
        async with self.__state_lock:
            self._ensure_open()
            state = self._reservation_state(reservation, allow_closing=True)
            task = state.close_task
            if task is None:
                state.closing = True
                task = asyncio.create_task(
                    self._close_state(state),
                    name=f"areal-memory-worker-close:{state.session_key}",
                )
                state.close_task = task
                task.add_done_callback(self._observe_task)
        await asyncio.shield(task)

    async def _shutdown(
        self,
        pending_tasks: tuple[
            asyncio.Task[MemoryWorkerSessionReservationV1],
            ...,
        ],
    ) -> None:
        try:
            broker_close = asyncio.create_task(
                self.__broker.aclose(),
                name="areal-memory-worker-owned-broker-shutdown",
            )
            if pending_tasks:
                await asyncio.gather(*pending_tasks, return_exceptions=True)
            await broker_close
        finally:
            async with self.__state_lock:
                self.__sessions.clear()
                self.__pending.clear()

    async def aclose(self) -> None:
        """Reject new reservations and close the exclusively owned broker."""

        self._running_loop()
        async with self.__state_lock:
            task = self.__shutdown_task
            if task is None:
                self.__closed = True
                for state in self.__sessions.values():
                    state.closing = True
                task = asyncio.create_task(
                    self._shutdown(
                        tuple(pending.task for pending in self.__pending.values())
                    ),
                    name="areal-memory-worker-runtime-shutdown",
                )
                self.__shutdown_task = task
                task.add_done_callback(self._observe_task)
        await asyncio.shield(task)

    async def __aenter__(self) -> AuthorizedMemoryWorkerRuntime:
        self._running_loop()
        self._ensure_open()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()


__all__ = [
    "AuthorizedMemoryWorkerRuntime",
    "MemoryWorkerSessionReservationV1",
]
