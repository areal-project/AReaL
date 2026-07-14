# SPDX-License-Identifier: Apache-2.0

"""Default-off host broker for authorized Agent Memory turns.

The grant contract in :mod:`areal.v2.agent_service.memory_authorization` is
deliberately synchronous and has no wire representation.  This module joins it
to :class:`~areal.v2.agent_service.memory.AsyncMemoryAgentCoordinator` without
trusting Agent request metadata or blocking the Worker event loop.

The broker owns the two replay domains that a caller must not choose: one fresh
Worker audience per broker and one fresh incarnation per opened session.  It
authorizes an exact assignment before the coordinator may pin it, and prepares
a fresh ``expose_memory`` authorization ticket for every public capability
call.  Tickets are single-use admission decisions, not bearer credentials or
leases.

This is a host-only, disabled-by-default seam.  It is not connected to Worker
HTTP ingress, and it is not a sandbox against arbitrary Python in the same
process.  One broker exclusively owns one coordinator and closes it during
``aclose``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol

from areal.v2.memory_service.release_control_types import MemoryReleaseAssignmentV1
from areal.v2.memory_service.types import MemoryScope

from .memory import (
    AsyncMemoryAgentCoordinator,
    MemoryAgentCoordinatorClosedError,
    MemoryAgentSessionConflictError,
    MemoryAgentSessionPinV1,
    MemoryAgentTurnConflictError,
    MemoryAgentTurnV1,
)
from .memory_authorization import (
    MemoryAssignmentGrantTargetV1,
    MemoryPrincipalV1,
    MemoryScopeActionV1,
    MemoryScopeGrantAuthorizer,
    MemoryScopeGrantRequestV1,
    MemorySessionIncarnationV1,
    MemoryWorkerAudienceV1,
)


class _ClosableMemoryCapability(Protocol):
    async def aclose(self) -> None: ...


def _string(value: object, field_name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be a str")
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")
    try:
        value.encode("utf-8", "strict")
    except UnicodeEncodeError as error:
        raise ValueError(f"{field_name} must be valid UTF-8") from error
    return value


def _clone_principal(value: object) -> MemoryPrincipalV1:
    if type(value) is not MemoryPrincipalV1:
        raise TypeError("principal must be a MemoryPrincipalV1")
    value.canonical_bytes()
    return MemoryPrincipalV1(issuer=value.issuer, subject=value.subject)


@dataclass(frozen=True, slots=True)
class AuthorizedMemorySessionV1:
    """Broker-issued handle for one principal and session incarnation.

    Equality is descriptive only.  The broker accepts only the exact object it
    issued for its current session state, so constructing an equal dataclass is
    not enough to cross the local capability boundary.
    """

    principal: MemoryPrincipalV1
    session: MemorySessionIncarnationV1
    audience: MemoryWorkerAudienceV1

    def __post_init__(self) -> None:
        if type(self.principal) is not MemoryPrincipalV1:
            raise TypeError("principal must be a MemoryPrincipalV1")
        if type(self.session) is not MemorySessionIncarnationV1:
            raise TypeError("session must be a MemorySessionIncarnationV1")
        if type(self.audience) is not MemoryWorkerAudienceV1:
            raise TypeError("audience must be a MemoryWorkerAudienceV1")
        self.principal.canonical_bytes()
        self.session.canonical_bytes()
        self.audience.canonical_bytes()


@dataclass(frozen=True, slots=True)
class AuthorizedMemoryTurnV1:
    """Broker-issued turn whose authority dimensions cannot be reselected."""

    session: AuthorizedMemorySessionV1
    turn: MemoryAgentTurnV1

    def __post_init__(self) -> None:
        if type(self.session) is not AuthorizedMemorySessionV1:
            raise TypeError("session must be an AuthorizedMemorySessionV1")
        if type(self.turn) is not MemoryAgentTurnV1:
            raise TypeError("turn must be a MemoryAgentTurnV1")
        if self.session.session.session_key != self.turn.session_key:
            raise ValueError("authorized session and Memory turn disagree")


@dataclass(frozen=True, slots=True)
class _AssignmentSnapshot:
    tenant_id: str
    namespace: str
    subject_id: str
    rollout_group_id: str
    rollout_group_incarnation_sha256: str
    assignment_id: str
    assignment_content_sha256: str

    @classmethod
    def from_pin(cls, pin: MemoryAgentSessionPinV1) -> _AssignmentSnapshot:
        if type(pin) is not MemoryAgentSessionPinV1:
            raise TypeError("pin must be a MemoryAgentSessionPinV1")
        if type(pin.scope) is not MemoryScope:
            raise TypeError("pin.scope must be a MemoryScope")
        # Reconstructing the pin validates every nested scalar and prevents the
        # resolver request from sharing a mutable alias with the coordinator.
        canonical = MemoryAgentSessionPinV1(
            scope=MemoryScope(
                tenant_id=pin.scope.tenant_id,
                namespace=pin.scope.namespace,
                subject_id=pin.scope.subject_id,
            ),
            rollout_group_id=pin.rollout_group_id,
            rollout_group_incarnation_sha256=(pin.rollout_group_incarnation_sha256),
            assignment_id=pin.assignment_id,
            assignment_content_sha256=pin.assignment_content_sha256,
        )
        return cls(
            tenant_id=canonical.scope.tenant_id,
            namespace=canonical.scope.namespace,
            subject_id=canonical.scope.subject_id,
            rollout_group_id=canonical.rollout_group_id,
            rollout_group_incarnation_sha256=(
                canonical.rollout_group_incarnation_sha256
            ),
            assignment_id=canonical.assignment_id,
            assignment_content_sha256=canonical.assignment_content_sha256,
        )

    def pin(self) -> MemoryAgentSessionPinV1:
        return MemoryAgentSessionPinV1(
            scope=MemoryScope(
                tenant_id=self.tenant_id,
                namespace=self.namespace,
                subject_id=self.subject_id,
            ),
            rollout_group_id=self.rollout_group_id,
            rollout_group_incarnation_sha256=(self.rollout_group_incarnation_sha256),
            assignment_id=self.assignment_id,
            assignment_content_sha256=self.assignment_content_sha256,
        )

    def target(self) -> MemoryAssignmentGrantTargetV1:
        # A fresh scope is intentional: a resolver may retain or even corrupt
        # its one-shot request, but it cannot poison later broker decisions.
        return MemoryAssignmentGrantTargetV1.from_session_pin(self.pin())


@dataclass(slots=True, eq=False)
class _SessionState:
    handle: AuthorizedMemorySessionV1
    principal_issuer: str
    principal_subject: str
    session_key: str
    incarnation_id: str
    audience_id: str
    epoch: int
    drained: asyncio.Event
    target: _AssignmentSnapshot | None = None
    pin_candidate: _AssignmentSnapshot | None = None
    pin_candidate_count: int = 0
    admitted: int = 0
    closing: bool = False
    close_task: asyncio.Task[None] | None = None
    turns: dict[str, _AuthorizedTurnState] | None = None
    capabilities: set[_ClosableMemoryCapability] | None = None

    def __post_init__(self) -> None:
        if self.turns is None:
            self.turns = {}
        if self.capabilities is None:
            self.capabilities = set()


@dataclass(frozen=True, slots=True)
class _AuthorizedTurnState:
    handle: AuthorizedMemoryTurnV1
    coordinator_turn: MemoryAgentTurnV1
    session_key: str
    turn_idempotency_key: str
    memory_trajectory_id: str

    def matches_public_handle(self) -> bool:
        try:
            turn = self.handle.turn
            session = self.handle.session.session
            return (
                type(turn) is MemoryAgentTurnV1
                and session.session_key == self.session_key
                and turn.session_key == self.session_key
                and turn.turn_idempotency_key == self.turn_idempotency_key
                and turn.memory_trajectory_id == self.memory_trajectory_id
            )
        except (AttributeError, TypeError, ValueError):
            return False


@dataclass(slots=True, eq=False)
class _ExposureAuthorizationTicket:
    broker: AuthorizedMemoryAgentBroker
    state: _SessionState
    turn: AuthorizedMemoryTurnV1
    target: _AssignmentSnapshot
    epoch: int
    consumed: bool = False


class AuthorizedMemoryAgentBroker:
    """Host-owned exact-grant facade around one Memory coordinator.

    A synchronous grant resolver is executed through the coordinator's bounded
    executor.  Authorization I/O never holds the broker lock.  After it
    returns, an epoch check and operation admission happen on the owner event
    loop without an intervening await; session close therefore cannot race an
    old authorization into a reopened incarnation.
    """

    def __init__(
        self,
        coordinator: AsyncMemoryAgentCoordinator,
        authorizer: MemoryScopeGrantAuthorizer,
    ) -> None:
        if not isinstance(coordinator, AsyncMemoryAgentCoordinator):
            raise TypeError("coordinator must be an AsyncMemoryAgentCoordinator")
        if type(authorizer) is not MemoryScopeGrantAuthorizer:
            raise TypeError("authorizer must be a MemoryScopeGrantAuthorizer")
        audience = MemoryWorkerAudienceV1.create()
        self.__coordinator = coordinator
        self.__authorizer = authorizer
        # Snapshot the selected host method.  The authorizer itself snapshots
        # and rechecks the resolver identity on every call.
        self.__authorize = authorizer.authorize
        self.__audience_id = audience.audience_id
        self.__state_lock = asyncio.Lock()
        self.__owner_loop: asyncio.AbstractEventLoop | None = None
        self.__sessions: dict[str, _SessionState] = {}
        self.__closed = False
        self.__shutdown_task: asyncio.Task[None] | None = None

    @property
    def audience(self) -> MemoryWorkerAudienceV1:
        """Return a detached description of this broker incarnation."""

        return MemoryWorkerAudienceV1(self.__audience_id)

    def _running_loop(self) -> asyncio.AbstractEventLoop:
        loop = asyncio.get_running_loop()
        if self.__owner_loop is None:
            self.__owner_loop = loop
        elif self.__owner_loop is not loop:
            raise RuntimeError(
                "AuthorizedMemoryAgentBroker cannot be shared across event loops"
            )
        return loop

    def _ensure_open(self) -> None:
        if self.__closed:
            raise MemoryAgentCoordinatorClosedError("Memory broker is closed")

    @staticmethod
    def _handle_matches_state(state: _SessionState) -> bool:
        try:
            state.handle.principal.canonical_bytes()
            state.handle.session.canonical_bytes()
            state.handle.audience.canonical_bytes()
        except (TypeError, ValueError):
            return False
        return (
            state.handle.principal.issuer,
            state.handle.principal.subject,
            state.handle.session.session_key,
            state.handle.session.incarnation_id,
            state.handle.audience.audience_id,
        ) == (
            state.principal_issuer,
            state.principal_subject,
            state.session_key,
            state.incarnation_id,
            state.audience_id,
        )

    def _session_state(
        self,
        handle: AuthorizedMemorySessionV1,
        *,
        allow_closing: bool = False,
    ) -> _SessionState:
        if type(handle) is not AuthorizedMemorySessionV1:
            raise TypeError("session must be an AuthorizedMemorySessionV1")
        try:
            session_key = handle.session.session_key
        except AttributeError as error:
            raise MemoryAgentSessionConflictError(
                "authorized Memory session handle is malformed"
            ) from error
        state = self.__sessions.get(session_key)
        if (
            state is None
            or state.handle is not handle
            or not self._handle_matches_state(state)
        ):
            raise MemoryAgentSessionConflictError(
                "authorized Memory session is not current for this broker"
            )
        if state.closing and not allow_closing:
            raise MemoryAgentSessionConflictError(
                "authorized Memory session is closing"
            )
        return state

    def _turn_state(
        self,
        handle: AuthorizedMemoryTurnV1,
    ) -> tuple[_SessionState, _AuthorizedTurnState]:
        if type(handle) is not AuthorizedMemoryTurnV1:
            raise TypeError("turn must be an AuthorizedMemoryTurnV1")
        state = self._session_state(handle.session)
        assert state.turns is not None
        current = state.turns.get(handle.turn.turn_idempotency_key)
        if (
            current is None
            or current.handle is not handle
            or not current.matches_public_handle()
        ):
            raise MemoryAgentTurnConflictError(
                "authorized Memory turn was not issued by this broker"
            )
        return state, current

    def _request(
        self,
        state: _SessionState,
        target: _AssignmentSnapshot,
        action: MemoryScopeActionV1,
    ) -> MemoryScopeGrantRequestV1:
        # Every object is freshly reconstructed from private scalar snapshots.
        # No request alias is reused across resolver calls or shared with the
        # coordinator's pin.
        return MemoryScopeGrantRequestV1(
            principal=MemoryPrincipalV1(
                issuer=state.principal_issuer,
                subject=state.principal_subject,
            ),
            session=MemorySessionIncarnationV1(
                session_key=state.session_key,
                incarnation_id=state.incarnation_id,
            ),
            audience=MemoryWorkerAudienceV1(state.audience_id),
            target=target.target(),
            action=action,
        )

    async def _authorize_request(
        self,
        request: MemoryScopeGrantRequestV1,
    ) -> None:
        # _call_sync is the coordinator's package-private, cancellation-safe,
        # bounded bridge.  Cancellation abandons the result but not the Python
        # callback; importantly, no broker admission follows a cancelled await.
        await self.__coordinator._call_sync(self.__authorize, request)

    async def open_session(
        self,
        principal: MemoryPrincipalV1,
        session_key: str,
    ) -> AuthorizedMemorySessionV1:
        """Open or get one host-minted incarnation for an exact principal."""

        self._running_loop()
        principal = _clone_principal(principal)
        session_key = _string(session_key, "session_key")
        async with self.__state_lock:
            self._ensure_open()
            existing = self.__sessions.get(session_key)
            if existing is not None:
                if existing.closing:
                    raise MemoryAgentSessionConflictError(
                        "authorized Memory session is closing"
                    )
                if not self._handle_matches_state(existing):
                    raise MemoryAgentSessionConflictError(
                        "authorized Memory session handle was corrupted"
                    )
                if (
                    existing.principal_issuer,
                    existing.principal_subject,
                ) != (principal.issuer, principal.subject):
                    raise MemoryAgentSessionConflictError(
                        "session key is already bound to another principal"
                    )
                return existing.handle

            incarnation = MemorySessionIncarnationV1.create(session_key)
            handle = AuthorizedMemorySessionV1(
                principal=MemoryPrincipalV1(
                    issuer=principal.issuer,
                    subject=principal.subject,
                ),
                session=MemorySessionIncarnationV1(
                    session_key=incarnation.session_key,
                    incarnation_id=incarnation.incarnation_id,
                ),
                audience=MemoryWorkerAudienceV1(self.__audience_id),
            )
            drained = asyncio.Event()
            drained.set()
            self.__sessions[session_key] = _SessionState(
                handle=handle,
                principal_issuer=principal.issuer,
                principal_subject=principal.subject,
                session_key=session_key,
                incarnation_id=incarnation.incarnation_id,
                audience_id=self.__audience_id,
                epoch=0,
                drained=drained,
            )
            return handle

    def _admit_locked(
        self,
        state: _SessionState,
        operation: object,
        *,
        task_name: str,
    ) -> asyncio.Task[object]:
        if not hasattr(operation, "__await__"):
            raise TypeError("admitted operation must be awaitable")
        state.admitted += 1
        state.drained.clear()
        task = asyncio.create_task(
            self._run_admitted(state, operation),  # type: ignore[arg-type]
            name=task_name,
        )
        task.add_done_callback(self._observe_task)
        return task

    async def _run_admitted(self, state: _SessionState, operation: object) -> object:
        try:
            return await operation  # type: ignore[misc]
        finally:
            async with self.__state_lock:
                state.admitted -= 1
                if state.admitted == 0:
                    state.drained.set()

    @staticmethod
    def _observe_task(task: asyncio.Task[object]) -> None:
        if not task.cancelled():
            task.exception()

    async def _pin_operation(
        self,
        state: _SessionState,
        target: _AssignmentSnapshot,
        expected_epoch: int,
    ) -> MemoryReleaseAssignmentV1:
        succeeded = False
        try:
            assignment = await self.__coordinator.pin_session(
                state.session_key,
                target.pin(),
            )
            succeeded = True
            return assignment
        finally:
            async with self.__state_lock:
                state.pin_candidate_count -= 1
                try:
                    if (
                        succeeded
                        and self.__sessions.get(state.session_key) is state
                        and state.epoch == expected_epoch
                    ):
                        if state.target is not None and state.target != target:
                            raise MemoryAgentSessionConflictError(
                                "coordinator pinned a different Memory assignment"
                            )
                        state.target = target
                finally:
                    # Integrity failures above must not strand an in-flight
                    # candidate after its last operation has drained.  The
                    # candidate is transient admission state, not a durable
                    # assignment binding.
                    if state.pin_candidate_count == 0:
                        state.pin_candidate = None

    async def pin_session(
        self,
        session: AuthorizedMemorySessionV1,
        pin: MemoryAgentSessionPinV1,
    ) -> MemoryReleaseAssignmentV1:
        """Authorize, then resolve and CAS one exact assignment pin.

        The resolver wait is pending, not admitted.  If the caller is
        cancelled or the session closes while it waits, the returned grant can
        no longer start a coordinator operation.
        """

        self._running_loop()
        target = _AssignmentSnapshot.from_pin(pin)
        async with self.__state_lock:
            self._ensure_open()
            state = self._session_state(session)
            bound = state.target or state.pin_candidate
            if bound is not None and bound != target:
                raise MemoryAgentSessionConflictError(
                    "session is already bound to another Memory assignment"
                )
            expected_epoch = state.epoch
            request = self._request(
                state,
                target,
                MemoryScopeActionV1.PIN_ASSIGNMENT,
            )

        await self._authorize_request(request)

        async with self.__state_lock:
            self._ensure_open()
            if (
                self.__sessions.get(state.session_key) is not state
                or state.epoch != expected_epoch
                or state.closing
                or not self._handle_matches_state(state)
            ):
                raise MemoryAgentSessionConflictError(
                    "session changed while its pin grant was resolving"
                )
            bound = state.target or state.pin_candidate
            if bound is not None and bound != target:
                raise MemoryAgentSessionConflictError(
                    "a concurrent caller selected another Memory assignment"
                )
            state.pin_candidate = target
            state.pin_candidate_count += 1
            task = self._admit_locked(
                state,
                self._pin_operation(state, target, expected_epoch),
                task_name=f"areal-authorized-memory-pin:{state.session_key}",
            )

        assignment = await asyncio.shield(task)
        async with self.__state_lock:
            if (
                self.__sessions.get(state.session_key) is not state
                or state.epoch != expected_epoch
                or state.closing
                or state.target != target
            ):
                raise MemoryAgentSessionConflictError(
                    "session closed while its Memory assignment was pinning"
                )
        return assignment  # type: ignore[return-value]

    async def _start_turn_operation(
        self,
        state: _SessionState,
        turn_idempotency_key: str,
        expected_epoch: int,
    ) -> AuthorizedMemoryTurnV1:
        turn = await self.__coordinator.start_turn(
            state.session_key,
            turn_idempotency_key,
        )
        async with self.__state_lock:
            if (
                self.__sessions.get(state.session_key) is not state
                or state.epoch != expected_epoch
                or state.closing
            ):
                raise MemoryAgentSessionConflictError(
                    "session closed while its Memory turn was starting"
                )
            assert state.turns is not None
            existing = state.turns.get(turn_idempotency_key)
            if existing is None:
                if type(turn) is not MemoryAgentTurnV1:
                    raise MemoryAgentTurnConflictError(
                        "coordinator returned a non-canonical Memory turn"
                    )
                coordinator_turn = MemoryAgentTurnV1(
                    session_key=turn.session_key,
                    turn_idempotency_key=turn.turn_idempotency_key,
                    memory_trajectory_id=turn.memory_trajectory_id,
                )
                public_turn = AuthorizedMemoryTurnV1(
                    state.handle,
                    MemoryAgentTurnV1(
                        session_key=turn.session_key,
                        turn_idempotency_key=turn.turn_idempotency_key,
                        memory_trajectory_id=turn.memory_trajectory_id,
                    ),
                )
                existing = _AuthorizedTurnState(
                    handle=public_turn,
                    coordinator_turn=coordinator_turn,
                    session_key=turn.session_key,
                    turn_idempotency_key=turn.turn_idempotency_key,
                    memory_trajectory_id=turn.memory_trajectory_id,
                )
                state.turns[turn_idempotency_key] = existing
            elif (
                existing.session_key,
                existing.turn_idempotency_key,
                existing.memory_trajectory_id,
            ) != (
                turn.session_key,
                turn.turn_idempotency_key,
                turn.memory_trajectory_id,
            ):
                raise MemoryAgentTurnConflictError(
                    "coordinator changed an existing Memory turn identity"
                )
            return existing.handle

    async def start_turn(
        self,
        session: AuthorizedMemorySessionV1,
        turn_idempotency_key: str,
    ) -> AuthorizedMemoryTurnV1:
        """Start one turn only after this incarnation has a successful pin."""

        self._running_loop()
        turn_idempotency_key = _string(
            turn_idempotency_key,
            "turn_idempotency_key",
        )
        async with self.__state_lock:
            self._ensure_open()
            state = self._session_state(session)
            if state.target is None:
                raise MemoryAgentSessionConflictError(
                    "session must be authorized and pinned before starting a turn"
                )
            assert state.turns is not None
            existing = state.turns.get(turn_idempotency_key)
            if existing is not None:
                if not existing.matches_public_handle():
                    raise MemoryAgentTurnConflictError(
                        "authorized Memory turn handle was corrupted"
                    )
                return existing.handle
            expected_epoch = state.epoch
            task = self._admit_locked(
                state,
                self._start_turn_operation(
                    state,
                    turn_idempotency_key,
                    expected_epoch,
                ),
                task_name=(
                    f"areal-authorized-memory-turn:{state.session_key}:"
                    f"{turn_idempotency_key}"
                ),
            )
        return await asyncio.shield(task)  # type: ignore[return-value]

    async def _authorize_exposure(
        self,
        turn: AuthorizedMemoryTurnV1,
    ) -> _ExposureAuthorizationTicket:
        """Prepare one ticket; capability admission consumes it exactly once."""

        self._running_loop()
        async with self.__state_lock:
            self._ensure_open()
            state, current = self._turn_state(turn)
            target = state.target
            if target is None:
                raise MemoryAgentSessionConflictError(
                    "authorized Memory session no longer has a pin"
                )
            expected_epoch = state.epoch
            request = self._request(
                state,
                target,
                MemoryScopeActionV1.EXPOSE_MEMORY,
            )

        await self._authorize_request(request)
        return _ExposureAuthorizationTicket(
            broker=self,
            state=state,
            turn=current.handle,
            target=target,
            epoch=expected_epoch,
        )

    def _consume_exposure_ticket(
        self,
        capability: _ClosableMemoryCapability,
        ticket: object,
    ) -> None:
        """Atomically turn a resolved grant into one capability admission.

        This method has no await and is called while the capability lock is
        held.  All broker state mutations happen on the same owner event loop,
        so close cannot interleave between this check and task registration.
        """

        self._running_loop()
        if type(ticket) is not _ExposureAuthorizationTicket:
            raise MemoryAgentTurnConflictError(
                "Memory exposure authorization ticket is malformed"
            )
        if ticket.broker is not self or ticket.consumed:
            raise MemoryAgentTurnConflictError(
                "Memory exposure authorization ticket is not active"
            )
        state = ticket.state
        try:
            current_state, current_turn = self._turn_state(ticket.turn)
        except (MemoryAgentSessionConflictError, MemoryAgentTurnConflictError):
            raise
        assert state.capabilities is not None
        if (
            self.__closed
            or current_state is not state
            or current_turn.handle is not ticket.turn
            or state.epoch != ticket.epoch
            or state.target != ticket.target
            or capability not in state.capabilities
        ):
            raise MemoryAgentTurnConflictError(
                "Memory exposure authorization became stale before admission"
            )
        ticket.consumed = True

    def _coordinator_for_turn(
        self,
        turn: AuthorizedMemoryTurnV1,
    ) -> tuple[AsyncMemoryAgentCoordinator, MemoryAgentTurnV1]:
        """Validate one issued turn before a capability is constructed."""

        self._running_loop()
        self._ensure_open()
        _, current = self._turn_state(turn)
        return self.__coordinator, current.coordinator_turn

    def _register_capability(
        self,
        turn: AuthorizedMemoryTurnV1,
        capability: _ClosableMemoryCapability,
    ) -> None:
        """Register synchronously so close observes every authorized reader."""

        self._running_loop()
        self._ensure_open()
        state, current = self._turn_state(turn)
        if current.handle is not turn:
            raise MemoryAgentTurnConflictError(
                "authorized Memory turn is no longer current"
            )
        assert state.capabilities is not None
        state.capabilities.add(capability)

    def _unregister_capability(
        self,
        capability: _ClosableMemoryCapability,
    ) -> None:
        """Forget a drained turn reader without trusting a public handle alias."""

        self._running_loop()
        for state in self.__sessions.values():
            assert state.capabilities is not None
            state.capabilities.discard(capability)

    async def _close_state(self, state: _SessionState) -> None:
        try:
            assert state.capabilities is not None
            capabilities = tuple(state.capabilities)
            if capabilities:
                await asyncio.gather(
                    *(capability.aclose() for capability in capabilities),
                    return_exceptions=False,
                )
            await state.drained.wait()
            await self.__coordinator.close_session(state.session_key)
        except BaseException:
            async with self.__state_lock:
                if self.__sessions.get(state.session_key) is state:
                    state.close_task = None
            raise
        else:
            async with self.__state_lock:
                if self.__sessions.get(state.session_key) is state:
                    self.__sessions.pop(state.session_key, None)

    async def close_session(self, session: AuthorizedMemorySessionV1) -> None:
        """Close one exact incarnation and permanently invalidate its turns."""

        self._running_loop()
        async with self.__state_lock:
            self._ensure_open()
            state = self._session_state(session, allow_closing=True)
            task = state.close_task
            if task is None:
                state.closing = True
                state.epoch += 1
                task = asyncio.create_task(
                    self._close_state(state),
                    name=f"areal-authorized-memory-close:{state.session_key}",
                )
                state.close_task = task
                task.add_done_callback(self._observe_task)
        await asyncio.shield(task)

    async def _shutdown(self, tasks: tuple[asyncio.Task[None], ...]) -> None:
        errors: list[BaseException] = []
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            errors.extend(
                result for result in results if isinstance(result, BaseException)
            )
        try:
            await self.__coordinator.aclose()
        except BaseException as error:
            errors.append(error)
        async with self.__state_lock:
            self.__sessions.clear()
        if errors:
            raise errors[0]

    async def aclose(self) -> None:
        """Close all capabilities/sessions, then drain the owned coordinator."""

        self._running_loop()
        async with self.__state_lock:
            task = self.__shutdown_task
            if task is None:
                self.__closed = True
                close_tasks: list[asyncio.Task[None]] = []
                for state in tuple(self.__sessions.values()):
                    close_task = state.close_task
                    if close_task is None:
                        state.closing = True
                        state.epoch += 1
                        close_task = asyncio.create_task(
                            self._close_state(state),
                            name=(f"areal-authorized-memory-close:{state.session_key}"),
                        )
                        state.close_task = close_task
                        close_task.add_done_callback(self._observe_task)
                    close_tasks.append(close_task)
                task = asyncio.create_task(
                    self._shutdown(tuple(close_tasks)),
                    name="areal-authorized-memory-broker-shutdown",
                )
                self.__shutdown_task = task
        await asyncio.shield(task)

    async def __aenter__(self) -> AuthorizedMemoryAgentBroker:
        self._running_loop()
        self._ensure_open()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()


__all__ = [
    "AuthorizedMemoryAgentBroker",
    "AuthorizedMemorySessionV1",
    "AuthorizedMemoryTurnV1",
]
