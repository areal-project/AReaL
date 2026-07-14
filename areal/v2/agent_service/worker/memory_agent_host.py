# SPDX-License-Identifier: Apache-2.0

"""Exact Worker ownership across Agent and Memory session state.

This module is a default-off, process-local host adapter.  It deliberately
registers no HTTP route and resolves no principal from request data.  Trusted
host code transfers exclusive ownership of an Agent and an authorized Memory
broker here, then addresses local operations with the exact reservation object
issued by this host.

The outer host is necessary because legacy Agent plugins release state by the
reusable text ``session_key``.  Memory cleanup may finish before that hook.  If
the Memory runtime released the key directly, a successor B could open before
A's old key-only hook ran and the hook could delete B.  This host therefore
keeps the key in ``closing`` state until active response bodies drain, Memory
closes, the Agent hook returns, and a full-host retirement receipt is published
atomically.
"""

from __future__ import annotations

import asyncio
import threading
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum

from areal.v2.agent_service.memory import (
    MemoryAgentSessionConflictError,
    MemoryAgentSessionPinV1,
    MemoryAgentTurnConflictError,
)
from areal.v2.agent_service.memory_authorization import (
    MemoryPrincipalV1,
    MemorySessionIncarnationV1,
    MemoryWorkerAudienceV1,
)
from areal.v2.agent_service.memory_broker import AuthorizedMemoryAgentBroker
from areal.v2.agent_service.memory_session_lifecycle import (
    MemoryWorkerSessionCloseOutcomeV1,
    MemoryWorkerSessionIdentityV1,
)
from areal.v2.agent_service.streaming import AsyncCleanupOnce, CleanupAsyncIterator
from areal.v2.agent_service.types import (
    AgentRequest,
    AgentResponse,
    AgentRunnable,
    EventEmitter,
    StreamResponse,
)

from .memory_runtime import (
    AuthorizedMemoryWorkerRuntime,
    MemoryWorkerSessionReservationV1,
)

_AGENT_CLAIM_LOCK = threading.Lock()
_AGENT_CLAIMS: dict[int, AgentRunnable] = {}


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


def _close_hook(
    agent: AgentRunnable,
    name: str,
) -> Callable[..., Awaitable[None]] | None:
    hook = getattr(agent, name, None)
    if hook is None:
        return None
    if not callable(hook):
        raise TypeError(f"agent.{name} must be callable")
    return hook


def _claim_agent(agent: AgentRunnable) -> None:
    """Claim one Agent object identity across all exact hosts in this process."""

    with _AGENT_CLAIM_LOCK:
        existing = _AGENT_CLAIMS.get(id(agent))
        if existing is not None:
            raise MemoryAgentSessionConflictError(
                "Agent already belongs to an exact Memory Worker host"
            )
        # Retain the object itself so Python cannot reuse its id while claimed.
        _AGENT_CLAIMS[id(agent)] = agent


def _release_agent(agent: AgentRunnable) -> None:
    with _AGENT_CLAIM_LOCK:
        if _AGENT_CLAIMS.get(id(agent)) is agent:
            _AGENT_CLAIMS.pop(id(agent), None)


@dataclass(frozen=True, slots=True, eq=False)
class MemoryAgentWorkerSessionReservationV1:
    """Non-authoritative description whose object identity is local authority."""

    session: MemorySessionIncarnationV1
    audience: MemoryWorkerAudienceV1

    def __post_init__(self) -> None:
        if type(self.session) is not MemorySessionIncarnationV1:
            raise TypeError("session must be a MemorySessionIncarnationV1")
        if type(self.audience) is not MemoryWorkerAudienceV1:
            raise TypeError("audience must be a MemoryWorkerAudienceV1")
        self.session.canonical_bytes()
        self.audience.canonical_bytes()
        object.__setattr__(
            self,
            "session",
            MemorySessionIncarnationV1(
                session_key=self.session.session_key,
                incarnation_id=self.session.incarnation_id,
            ),
        )
        object.__setattr__(
            self,
            "audience",
            MemoryWorkerAudienceV1(self.audience.audience_id),
        )

    @property
    def session_key(self) -> str:
        return self.session.session_key

    @property
    def identity(self) -> MemoryWorkerSessionIdentityV1:
        return MemoryWorkerSessionIdentityV1(
            session=self.session,
            audience=self.audience,
        )


class MemoryAgentWorkerSessionCloseOutcomeV1(StrEnum):
    """Terminal result of a full Agent-and-Memory host close."""

    CLOSED = "closed"
    NOT_CURRENT = "not_current"


@dataclass(frozen=True, slots=True)
class MemoryAgentWorkerSessionCloseReceiptV1:
    """Detached full-host result, distinct from a Memory-only receipt."""

    identity: MemoryWorkerSessionIdentityV1
    outcome: MemoryAgentWorkerSessionCloseOutcomeV1

    def __post_init__(self) -> None:
        if type(self.identity) is not MemoryWorkerSessionIdentityV1:
            raise TypeError("identity must be a MemoryWorkerSessionIdentityV1")
        if type(self.outcome) is not MemoryAgentWorkerSessionCloseOutcomeV1:
            raise TypeError("outcome must be a MemoryAgentWorkerSessionCloseOutcomeV1")
        object.__setattr__(
            self,
            "identity",
            MemoryWorkerSessionIdentityV1(
                session=self.identity.session,
                audience=self.identity.audience,
            ),
        )


@dataclass(slots=True, eq=False)
class _HostSessionState:
    descriptor: MemoryAgentWorkerSessionReservationV1
    memory_reservation: MemoryWorkerSessionReservationV1
    principal_issuer: str
    principal_subject: str
    session_key: str
    incarnation_id: str
    audience_id: str
    active_turns: int = 0
    turns_drained: asyncio.Event = field(default_factory=asyncio.Event)
    active_execution_tasks: set[asyncio.Task[object]] = field(default_factory=set)
    active_streams: set[CleanupAsyncIterator] = field(default_factory=set)
    closing: bool = False
    close_task: asyncio.Task[MemoryAgentWorkerSessionCloseReceiptV1] | None = None
    agent_close_started: bool = False

    def __post_init__(self) -> None:
        self.turns_drained.set()


@dataclass(frozen=True, slots=True)
class _RetiredHostSession:
    descriptor: MemoryAgentWorkerSessionReservationV1
    session_key: str
    incarnation_id: str
    audience_id: str


@dataclass(frozen=True, slots=True)
class _HostReservationAttempt:
    principal_issuer: str
    principal_subject: str
    task: asyncio.Task[MemoryAgentWorkerSessionReservationV1]


class AuthorizedMemoryAgentWorkerHost:
    """Own one Agent and its complete authorized Memory session lifecycle.

    The caller must not use ``agent`` or ``broker`` after construction.  The
    Memory runtime is created privately, so a caller cannot bypass this host's
    key gate by retaining a runtime reference and reopening B between Memory
    cleanup and the legacy Agent hook.
    """

    def __init__(
        self,
        broker: AuthorizedMemoryAgentBroker,
        agent: AgentRunnable,
        *,
        max_retired_sessions: int = 4096,
    ) -> None:
        if type(broker) is not AuthorizedMemoryAgentBroker:
            raise TypeError("broker must be an AuthorizedMemoryAgentBroker")
        if not isinstance(agent, AgentRunnable):
            raise TypeError("agent must satisfy AgentRunnable")
        if type(max_retired_sessions) is not int:
            raise TypeError("max_retired_sessions must be an int")
        if max_retired_sessions <= 0:
            raise ValueError("max_retired_sessions must be positive")

        close_session = _close_hook(agent, "close_session")
        if close_session is None:
            raise TypeError(
                "exact Memory Agent Worker host requires agent.close_session"
            )
        close_all_sessions = _close_hook(agent, "close_all_sessions")
        _claim_agent(agent)
        try:
            runtime = AuthorizedMemoryWorkerRuntime(
                broker,
                max_retired_sessions=max_retired_sessions,
            )
        except BaseException:
            _release_agent(agent)
            raise
        self.__agent = agent
        self.__agent_close_session = close_session
        self.__agent_close_all_sessions = close_all_sessions
        self.__runtime = runtime
        self.__audience_id = self.__runtime.audience.audience_id
        self.__state_lock = asyncio.Lock()
        self.__owner_loop: asyncio.AbstractEventLoop | None = None
        self.__sessions: dict[str, _HostSessionState] = {}
        self.__pending: dict[str, _HostReservationAttempt] = {}
        self.__retired_by_descriptor: dict[int, _RetiredHostSession] = {}
        self.__retired_by_identity: OrderedDict[
            tuple[str, str, str],
            _RetiredHostSession,
        ] = OrderedDict()
        self.__max_retired_sessions = max_retired_sessions
        self.__quarantined_session_keys: set[str] = set()
        self.__closed = False
        self.__shutdown_task: asyncio.Task[None] | None = None

    @property
    def audience(self) -> MemoryWorkerAudienceV1:
        return MemoryWorkerAudienceV1(self.__audience_id)

    def _running_loop(self) -> asyncio.AbstractEventLoop:
        loop = asyncio.get_running_loop()
        if self.__owner_loop is None:
            self.__owner_loop = loop
        elif self.__owner_loop is not loop:
            raise RuntimeError(
                "Memory Agent Worker host used from a different event loop"
            )
        return loop

    def _ensure_open(self) -> None:
        if self.__closed:
            raise MemoryAgentSessionConflictError("Memory Agent Worker host is closed")

    @staticmethod
    def _identity_key(
        identity: object,
    ) -> tuple[MemoryWorkerSessionIdentityV1, tuple[str, str, str]]:
        if type(identity) is not MemoryWorkerSessionIdentityV1:
            raise TypeError("identity must be a MemoryWorkerSessionIdentityV1")
        detached = MemoryWorkerSessionIdentityV1(
            session=identity.session,
            audience=identity.audience,
        )
        return detached, (
            detached.audience.audience_id,
            detached.session.session_key,
            detached.session.incarnation_id,
        )

    @staticmethod
    def _state_identity_key(state: _HostSessionState) -> tuple[str, str, str]:
        return (state.audience_id, state.session_key, state.incarnation_id)

    @staticmethod
    def _state_is_intact(state: _HostSessionState) -> bool:
        try:
            state.descriptor.session.canonical_bytes()
            state.descriptor.audience.canonical_bytes()
            state.memory_reservation.session.canonical_bytes()
            state.memory_reservation.audience.canonical_bytes()
        except (AttributeError, TypeError, ValueError):
            return False
        return (
            state.descriptor.session_key == state.session_key
            and state.descriptor.session.incarnation_id == state.incarnation_id
            and state.descriptor.audience.audience_id == state.audience_id
            and state.memory_reservation.session_key == state.session_key
            and state.memory_reservation.session.incarnation_id == state.incarnation_id
            and state.memory_reservation.audience.audience_id == state.audience_id
        )

    @staticmethod
    def _receipt(
        *,
        session_key: str,
        incarnation_id: str,
        audience_id: str,
        outcome: MemoryAgentWorkerSessionCloseOutcomeV1,
    ) -> MemoryAgentWorkerSessionCloseReceiptV1:
        return MemoryAgentWorkerSessionCloseReceiptV1(
            identity=MemoryWorkerSessionIdentityV1(
                session=MemorySessionIncarnationV1(
                    session_key=session_key,
                    incarnation_id=incarnation_id,
                ),
                audience=MemoryWorkerAudienceV1(audience_id),
            ),
            outcome=outcome,
        )

    @classmethod
    def _retired_receipt(
        cls,
        retired: _RetiredHostSession,
    ) -> MemoryAgentWorkerSessionCloseReceiptV1:
        try:
            return cls._receipt(
                session_key=retired.session_key,
                incarnation_id=retired.incarnation_id,
                audience_id=retired.audience_id,
                outcome=MemoryAgentWorkerSessionCloseOutcomeV1.CLOSED,
            )
        except (TypeError, ValueError) as error:
            raise MemoryAgentSessionConflictError(
                "retired Memory Agent Worker identity is corrupted"
            ) from error

    def _reservation_state(
        self,
        descriptor: MemoryAgentWorkerSessionReservationV1,
        *,
        allow_closing: bool = False,
    ) -> _HostSessionState:
        if type(descriptor) is not MemoryAgentWorkerSessionReservationV1:
            raise TypeError(
                "reservation must be a MemoryAgentWorkerSessionReservationV1"
            )
        state = self.__sessions.get(descriptor.session_key)
        if (
            state is None
            or state.descriptor is not descriptor
            or not self._state_is_intact(state)
        ):
            raise MemoryAgentSessionConflictError(
                "Memory Agent Worker reservation is not current"
            )
        if state.closing and not allow_closing:
            raise MemoryAgentSessionConflictError(
                "Memory Agent Worker reservation is closing"
            )
        return state

    async def _reserve(
        self,
        principal: MemoryPrincipalV1,
        session_key: str,
    ) -> MemoryAgentWorkerSessionReservationV1:
        current_task = asyncio.current_task()
        try:
            memory_reservation = await self.__runtime.reserve_session(
                principal,
                session_key,
            )
            async with self.__state_lock:
                self._ensure_open()
                identity_key = (
                    memory_reservation.audience.audience_id,
                    memory_reservation.session_key,
                    memory_reservation.session.incarnation_id,
                )
                if identity_key in self.__retired_by_identity:
                    self.__quarantined_session_keys.add(session_key)
                    raise MemoryAgentSessionConflictError(
                        "Memory runtime reused a retained full-host identity"
                    )
                if session_key in self.__sessions:
                    raise MemoryAgentSessionConflictError(
                        "Memory Agent Worker reservation changed while opening"
                    )
                descriptor = MemoryAgentWorkerSessionReservationV1(
                    session=memory_reservation.session,
                    audience=memory_reservation.audience,
                )
                self.__sessions[session_key] = _HostSessionState(
                    descriptor=descriptor,
                    memory_reservation=memory_reservation,
                    principal_issuer=principal.issuer,
                    principal_subject=principal.subject,
                    session_key=session_key,
                    incarnation_id=memory_reservation.session.incarnation_id,
                    audience_id=memory_reservation.audience.audience_id,
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
    ) -> MemoryAgentWorkerSessionReservationV1:
        """Open one exact full-host session in a shielded owned task."""

        self._running_loop()
        principal = _principal(principal)
        session_key = _session_key(session_key)
        async with self.__state_lock:
            self._ensure_open()
            if session_key in self.__quarantined_session_keys:
                raise MemoryAgentSessionConflictError(
                    "session key is quarantined by the Memory Agent Worker host"
                )
            existing = self.__sessions.get(session_key)
            if existing is not None:
                if existing.closing:
                    raise MemoryAgentSessionConflictError(
                        "Memory Agent Worker reservation is closing"
                    )
                if not self._state_is_intact(existing):
                    raise MemoryAgentSessionConflictError(
                        "Memory Agent Worker reservation state is corrupted"
                    )
                if (
                    existing.principal_issuer,
                    existing.principal_subject,
                ) != (principal.issuer, principal.subject):
                    raise MemoryAgentSessionConflictError(
                        "session key belongs to another principal"
                    )
                return existing.descriptor

            pending = self.__pending.get(session_key)
            if pending is not None:
                if (pending.principal_issuer, pending.principal_subject) != (
                    principal.issuer,
                    principal.subject,
                ):
                    raise MemoryAgentSessionConflictError(
                        "session key is opening for another principal"
                    )
                task = pending.task
            else:
                task = asyncio.create_task(
                    self._reserve(principal, session_key),
                    name=f"areal-memory-agent-host-reserve:{session_key}",
                )
                self.__pending[session_key] = _HostReservationAttempt(
                    principal_issuer=principal.issuer,
                    principal_subject=principal.subject,
                    task=task,
                )
                task.add_done_callback(self._observe_reservation_task)
        return await asyncio.shield(task)

    async def _release_turn(
        self,
        state: _HostSessionState,
        execution_task: asyncio.Task[object],
    ) -> None:
        async with self.__state_lock:
            if state.active_turns <= 0:
                raise RuntimeError("Memory Agent Worker turn lease underflow")
            state.active_execution_tasks.discard(execution_task)
            state.active_turns -= 1
            if state.active_turns == 0:
                state.turns_drained.set()

    async def _admit_turn(
        self,
        reservation: MemoryAgentWorkerSessionReservationV1,
        request: AgentRequest,
    ) -> tuple[_HostSessionState, AsyncCleanupOnce]:
        execution_task = asyncio.current_task()
        if execution_task is None:  # pragma: no cover - asyncio always owns callers
            raise RuntimeError("Memory Agent Worker turn has no owning task")
        async with self.__state_lock:
            self._ensure_open()
            state = self._reservation_state(reservation)
            if type(request) is not AgentRequest:
                raise TypeError("request must be an AgentRequest")
            if request.session_key != state.session_key:
                raise MemoryAgentTurnConflictError(
                    "AgentRequest session does not match the full-host reservation"
                )
            state.active_turns += 1
            state.turns_drained.clear()
            state.active_execution_tasks.add(execution_task)
            release = AsyncCleanupOnce(
                lambda: self._release_turn(state, execution_task),
                task_name=f"areal-memory-agent-host-turn-release:{state.session_key}",
            )
            return state, release

    @staticmethod
    async def _finish_stream_handoff(
        source: object,
        release: AsyncCleanupOnce,
        primary_error: BaseException,
    ) -> None:
        async def finish() -> None:
            try:
                close = getattr(source, "aclose", None)
                if callable(close):
                    await close()
            finally:
                await release()

        task = asyncio.create_task(
            finish(),
            name="areal-memory-agent-host-failed-stream-handoff",
        )
        task.add_done_callback(AuthorizedMemoryAgentWorkerHost._observe_task)
        try:
            await asyncio.shield(task)
        except BaseException as cleanup_error:
            primary_error.add_note(
                f"full-host stream handoff cleanup also failed: {cleanup_error!r}"
            )

    async def run_agent(
        self,
        reservation: MemoryAgentWorkerSessionReservationV1,
        request: AgentRequest,
        *,
        assignment_pin: MemoryAgentSessionPinV1,
        emitter: EventEmitter,
    ) -> AgentResponse | StreamResponse:
        """Run one exact turn while retaining the outer gate through body EOF."""

        self._running_loop()
        state, release = await self._admit_turn(reservation, request)
        stream_transferred = False
        primary_error: BaseException | None = None
        try:
            lease = await self.__runtime.bind_turn(
                state.memory_reservation,
                request,
                assignment_pin=assignment_pin,
            )
            response = await lease.run_agent(self.__agent, emitter=emitter)
            if isinstance(response, StreamResponse):
                source = response.body
                body: CleanupAsyncIterator | None = None

                async def release_stream() -> None:
                    try:
                        await release()
                    finally:
                        if body is not None:
                            async with self.__state_lock:
                                state.active_streams.discard(body)

                try:
                    body = CleanupAsyncIterator(
                        source,
                        cleanup=release_stream,
                        cleanup_task_name=(
                            f"areal-memory-agent-host-stream:{state.session_key}"
                        ),
                    )
                    wrapped = StreamResponse(
                        status_code=response.status_code,
                        headers=dict(response.headers),
                        body=body,
                    )
                    # Registration is the only await between inner stream
                    # ownership and publication.  Shutdown closes the gate
                    # under this same lock and snapshots every registered
                    # body.  If it won first, fail the handoff and close the
                    # inner Memory-owned body instead of returning a stream
                    # that shutdown can no longer find.
                    async with self.__state_lock:
                        self._ensure_open()
                        if self.__sessions.get(state.session_key) is not state:
                            raise MemoryAgentSessionConflictError(
                                "full-host reservation changed during stream handoff"
                            )
                        state.active_execution_tasks.discard(asyncio.current_task())
                        state.active_streams.add(body)
                except BaseException as error:
                    await self._finish_stream_handoff(
                        body if body is not None else source,
                        release,
                        error,
                    )
                    raise
                stream_transferred = True
                return wrapped
            return response
        except BaseException as error:
            primary_error = error
            raise
        finally:
            if not stream_transferred:
                try:
                    await release()
                except BaseException as cleanup_error:
                    if primary_error is None:
                        raise
                    primary_error.add_note(
                        f"full-host turn release also failed: {cleanup_error!r}"
                    )

    def _start_close_task(
        self,
        state: _HostSessionState,
    ) -> asyncio.Task[MemoryAgentWorkerSessionCloseReceiptV1]:
        task = state.close_task
        if task is None:
            state.closing = True
            task = asyncio.create_task(
                self._close_state(state),
                name=f"areal-memory-agent-host-close:{state.session_key}",
            )
            state.close_task = task
            task.add_done_callback(self._observe_close_task)
        return task

    async def _close_state(
        self,
        state: _HostSessionState,
    ) -> MemoryAgentWorkerSessionCloseReceiptV1:
        retired = _RetiredHostSession(
            descriptor=state.descriptor,
            session_key=state.session_key,
            incarnation_id=state.incarnation_id,
            audience_id=state.audience_id,
        )
        receipt = self._retired_receipt(retired)
        try:
            await state.turns_drained.wait()
            memory_receipt = await self.__runtime.close_session(
                state.memory_reservation
            )
            if (
                memory_receipt.identity
                != MemoryWorkerSessionIdentityV1(
                    session=MemorySessionIncarnationV1(
                        session_key=state.session_key,
                        incarnation_id=state.incarnation_id,
                    ),
                    audience=MemoryWorkerAudienceV1(state.audience_id),
                )
                or memory_receipt.outcome
                is not MemoryWorkerSessionCloseOutcomeV1.CLOSED
            ):
                raise MemoryAgentSessionConflictError(
                    "Memory runtime returned a mismatched close receipt"
                )

            state.agent_close_started = True
            await self.__agent_close_session(state.session_key)

            async with self.__state_lock:
                if self.__sessions.get(state.session_key) is not state:
                    raise MemoryAgentSessionConflictError(
                        "full-host reservation changed while closing"
                    )
                if not self._state_is_intact(state):
                    raise MemoryAgentSessionConflictError(
                        "full-host reservation is corrupted while closing"
                    )
                identity_key = self._state_identity_key(state)
                existing_identity = self.__retired_by_identity.get(identity_key)
                existing_descriptor = self.__retired_by_descriptor.get(
                    id(state.descriptor)
                )
                if (
                    existing_identity is not None
                    and existing_identity.descriptor is not state.descriptor
                ) or (
                    existing_descriptor is not None
                    and existing_descriptor.descriptor is not state.descriptor
                ):
                    raise MemoryAgentSessionConflictError(
                        "full-host retirement identity was already used"
                    )
                self.__retired_by_identity[identity_key] = retired
                self.__retired_by_identity.move_to_end(identity_key)
                self.__retired_by_descriptor[id(state.descriptor)] = retired
                while len(self.__retired_by_identity) > self.__max_retired_sessions:
                    _, evicted = self.__retired_by_identity.popitem(last=False)
                    descriptor_id = id(evicted.descriptor)
                    if self.__retired_by_descriptor.get(descriptor_id) is evicted:
                        self.__retired_by_descriptor.pop(descriptor_id, None)
                self.__sessions.pop(state.session_key, None)
        except BaseException:
            async with self.__state_lock:
                if self.__sessions.get(state.session_key) is state:
                    if state.agent_close_started:
                        self.__quarantined_session_keys.add(state.session_key)
                    else:
                        state.close_task = None
            raise
        return receipt

    async def close_session(
        self,
        reservation: MemoryAgentWorkerSessionReservationV1,
    ) -> MemoryAgentWorkerSessionCloseReceiptV1:
        """Close an exact local handle and replay only full-host completion."""

        self._running_loop()
        async with self.__state_lock:
            self._ensure_open()
            if type(reservation) is not MemoryAgentWorkerSessionReservationV1:
                raise TypeError(
                    "reservation must be a MemoryAgentWorkerSessionReservationV1"
                )
            retired = self.__retired_by_descriptor.get(id(reservation))
            if retired is not None:
                if retired.descriptor is not reservation:
                    raise MemoryAgentSessionConflictError(
                        "full-host reservation is not current"
                    )
                retired_key = (
                    retired.audience_id,
                    retired.session_key,
                    retired.incarnation_id,
                )
                if self.__retired_by_identity.get(retired_key) is not retired:
                    raise MemoryAgentSessionConflictError(
                        "full-host retirement indexes disagree"
                    )
                self.__retired_by_identity.move_to_end(retired_key)
                return self._retired_receipt(retired)
            state = self._reservation_state(reservation, allow_closing=True)
            task = self._start_close_task(state)
        result = await asyncio.shield(task)
        return MemoryAgentWorkerSessionCloseReceiptV1(
            identity=result.identity,
            outcome=result.outcome,
        )

    async def close_session_if_current(
        self,
        identity: MemoryWorkerSessionIdentityV1,
    ) -> MemoryAgentWorkerSessionCloseReceiptV1:
        """Compare and close a descriptive identity for a future trusted hop."""

        self._running_loop()
        identity, identity_key = self._identity_key(identity)
        async with self.__state_lock:
            self._ensure_open()
            retired = self.__retired_by_identity.get(identity_key)
            if retired is not None:
                self.__retired_by_identity.move_to_end(identity_key)
                return self._retired_receipt(retired)
            state = self.__sessions.get(identity.session_key)
            if state is None:
                return MemoryAgentWorkerSessionCloseReceiptV1(
                    identity=identity,
                    outcome=MemoryAgentWorkerSessionCloseOutcomeV1.NOT_CURRENT,
                )
            if not self._state_is_intact(state):
                raise MemoryAgentSessionConflictError(
                    "full-host reservation state is corrupted"
                )
            if self._state_identity_key(state) != identity_key:
                return MemoryAgentWorkerSessionCloseReceiptV1(
                    identity=identity,
                    outcome=MemoryAgentWorkerSessionCloseOutcomeV1.NOT_CURRENT,
                )
            task = self._start_close_task(state)
        result = await asyncio.shield(task)
        return MemoryAgentWorkerSessionCloseReceiptV1(
            identity=result.identity,
            outcome=result.outcome,
        )

    @staticmethod
    def _observe_reservation_task(
        task: asyncio.Task[MemoryAgentWorkerSessionReservationV1],
    ) -> None:
        if not task.cancelled():
            task.exception()

    @staticmethod
    def _observe_close_task(
        task: asyncio.Task[MemoryAgentWorkerSessionCloseReceiptV1],
    ) -> None:
        if not task.cancelled():
            task.exception()

    @staticmethod
    def _observe_task(task: asyncio.Task[None]) -> None:
        if not task.cancelled():
            task.exception()

    async def _shutdown(
        self,
        pending_tasks: tuple[
            asyncio.Task[MemoryAgentWorkerSessionReservationV1],
            ...,
        ],
        active_execution_tasks: tuple[asyncio.Task[object], ...],
        active_streams: tuple[CleanupAsyncIterator, ...],
        close_tasks: tuple[
            asyncio.Task[MemoryAgentWorkerSessionCloseReceiptV1],
            ...,
        ],
    ) -> None:
        primary_error: BaseException | None = None

        def retain_error(error: BaseException, context: str) -> None:
            nonlocal primary_error
            if primary_error is None:
                primary_error = error
                return
            if error is primary_error:
                return
            primary_error.add_note(f"{context}: {error!r}")
            for note in getattr(error, "__notes__", ()):
                primary_error.add_note(f"{context} detail: {note}")

        try:
            for execution_task in active_execution_tasks:
                if not execution_task.done():
                    execution_task.cancel()
            if active_execution_tasks:
                results = await asyncio.gather(
                    *active_execution_tasks,
                    return_exceptions=True,
                )
                for result in results:
                    if isinstance(result, BaseException) and not isinstance(
                        result,
                        asyncio.CancelledError,
                    ):
                        retain_error(result, "Agent execution shutdown also failed")
            if active_streams:
                results = await asyncio.gather(
                    *(stream.aclose() for stream in active_streams),
                    return_exceptions=True,
                )
                for result in results:
                    if isinstance(result, BaseException):
                        retain_error(result, "response stream shutdown also failed")
            if pending_tasks:
                await asyncio.gather(*pending_tasks, return_exceptions=True)
            if close_tasks:
                results = await asyncio.gather(*close_tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, BaseException):
                        retain_error(result, "full-host session close also failed")
            try:
                await self.__runtime.aclose()
            except BaseException as error:
                retain_error(error, "Memory runtime shutdown also failed")
            close_all = self.__agent_close_all_sessions
            if close_all is not None:
                try:
                    await close_all()
                except BaseException as error:
                    retain_error(error, "Agent global shutdown also failed")
            if primary_error is not None:
                raise primary_error
            _release_agent(self.__agent)
        finally:
            async with self.__state_lock:
                self.__sessions.clear()
                self.__pending.clear()
                self.__retired_by_descriptor.clear()
                self.__retired_by_identity.clear()
                self.__quarantined_session_keys.clear()

    async def aclose(self) -> None:
        """Drain exact sessions, then close the owned runtime and Agent."""

        self._running_loop()
        async with self.__state_lock:
            task = self.__shutdown_task
            if task is None:
                self.__closed = True
                active_streams = tuple(
                    stream
                    for state in self.__sessions.values()
                    for stream in state.active_streams
                )
                active_execution_tasks = tuple(
                    execution_task
                    for state in self.__sessions.values()
                    for execution_task in state.active_execution_tasks
                )
                close_tasks = tuple(
                    self._start_close_task(state) for state in self.__sessions.values()
                )
                task = asyncio.create_task(
                    self._shutdown(
                        tuple(pending.task for pending in self.__pending.values()),
                        active_execution_tasks,
                        active_streams,
                        close_tasks,
                    ),
                    name="areal-memory-agent-host-shutdown",
                )
                self.__shutdown_task = task
                task.add_done_callback(self._observe_task)
        await asyncio.shield(task)

    async def __aenter__(self) -> AuthorizedMemoryAgentWorkerHost:
        self._running_loop()
        self._ensure_open()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()


__all__ = [
    "AuthorizedMemoryAgentWorkerHost",
    "MemoryAgentWorkerSessionCloseOutcomeV1",
    "MemoryAgentWorkerSessionCloseReceiptV1",
    "MemoryAgentWorkerSessionReservationV1",
]
