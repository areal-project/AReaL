# SPDX-License-Identifier: Apache-2.0

"""Cancellation-safe async bridge from Agent Service to Memory runtime.

The Memory runtime is intentionally synchronous: its query, render, and
consumer transitions share one linearizable ledger.  Agent implementations,
on the other hand, run on an asyncio event loop.  This module joins the two
without making synchronous Memory callbacks block that loop.

The coordinator is designed for the Worker/agent boundary, not as an exposure
API for the DataProxy.  A DataProxy may pin an assignment reference to a
session.  Only :meth:`AsyncMemoryAgentCoordinator.expose_memory`, which reaches
the registered runtime consumer through ``submit_delivery``, can return a
``MemoryExposureV1``.  A raw-passthrough path that bypasses that consumer must
remain unattributed; forwarding bytes is not evidence that Memory reached a
model.

This first adapter deliberately does not modify the production Worker or
DataProxy applications.  It supplies the small concurrency and identity core
that a later integration can instantiate with deployment-selected stores.
"""

from __future__ import annotations

import asyncio
import secrets
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from hashlib import sha256
from typing import TypeVar

from areal.v2.memory_service.errors import MemoryServiceError
from areal.v2.memory_service.release_control_store import MemoryReleaseControlStore
from areal.v2.memory_service.release_control_types import MemoryReleaseAssignmentV1
from areal.v2.memory_service.runtime_store import MemoryRuntimeStore
from areal.v2.memory_service.runtime_types import (
    MemoryExposureV1,
    MemoryQuerySpecV1,
)
from areal.v2.memory_service.types import MemoryScope

_T = TypeVar("_T")
_SHA256_HEX_LENGTH = 64


class MemoryAgentCoordinatorError(MemoryServiceError):
    """Base class for Agent-to-Memory coordination failures."""


class MemoryAgentSessionConflictError(MemoryAgentCoordinatorError):
    """Raised when a session is rebound to a different rollout incarnation."""


class MemoryAgentTurnConflictError(MemoryAgentCoordinatorError):
    """Raised when an immutable turn or operation identity is reused."""


class MemoryAgentCoordinatorClosedError(MemoryAgentCoordinatorError):
    """Raised after the coordinator has begun its graceful shutdown."""


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


def _digest(value: object, field_name: str) -> str:
    value = _string(value, field_name)
    if len(value) != _SHA256_HEX_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")
    return value


def _positive_integer(value: object, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an int")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return value


def _history(value: object) -> tuple[bytes, ...]:
    if type(value) is not tuple:
        raise TypeError("history must be a tuple")
    result = tuple(tuple.__iter__(value))
    if any(type(item) is not bytes for item in result):
        raise TypeError("history must contain bytes")
    return result


@dataclass(frozen=True, slots=True)
class MemoryAgentSessionPinV1:
    """The only Memory authority reference a DataProxy may pin.

    The value is not an exposure receipt and is not itself authority to use a
    release.  ``pin_session`` resolves it in the trusted control store, and the
    runtime resolves it again at every query, render, and consumer boundary.
    """

    scope: MemoryScope
    rollout_group_id: str
    rollout_group_incarnation_sha256: str
    assignment_id: str
    assignment_content_sha256: str

    def __post_init__(self) -> None:
        if type(self.scope) is not MemoryScope:
            raise TypeError("scope must be a MemoryScope")
        object.__setattr__(
            self,
            "rollout_group_id",
            _string(self.rollout_group_id, "rollout_group_id"),
        )
        object.__setattr__(
            self,
            "rollout_group_incarnation_sha256",
            _digest(
                self.rollout_group_incarnation_sha256,
                "rollout_group_incarnation_sha256",
            ),
        )
        content_hash = _digest(
            self.assignment_content_sha256,
            "assignment_content_sha256",
        )
        assignment_id = _string(self.assignment_id, "assignment_id")
        if assignment_id != f"masn_{content_hash[:24]}":
            raise ValueError("assignment_id disagrees with assignment_content_sha256")
        object.__setattr__(self, "assignment_id", assignment_id)
        object.__setattr__(self, "assignment_content_sha256", content_hash)


@dataclass(frozen=True, slots=True)
class MemoryAgentTurnV1:
    """Coordinator-issued handle for one Agent turn.

    ``memory_trajectory_id`` is generated independently.  ``session_key`` and
    ``turn_idempotency_key`` (which may be an Agent ``run_id``) are lookup
    inputs only and are never reused as the Memory trajectory identity.
    """

    session_key: str
    turn_idempotency_key: str
    memory_trajectory_id: str

    def __post_init__(self) -> None:
        for field_name in (
            "session_key",
            "turn_idempotency_key",
            "memory_trajectory_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _string(getattr(self, field_name), field_name),
            )
        if self.memory_trajectory_id in (
            self.session_key,
            self.turn_idempotency_key,
        ):
            raise ValueError("memory_trajectory_id must be independently generated")


@dataclass(slots=True)
class _PinnedSession:
    pin: MemoryAgentSessionPinV1
    assignment: MemoryReleaseAssignmentV1
    epoch: int
    closing: bool = False


@dataclass(slots=True)
class _TurnState:
    handle: MemoryAgentTurnV1
    next_query_sequence_no: int = 0


@dataclass(slots=True)
class _ExposureOperation:
    query: bytes
    history: tuple[bytes, ...]
    spec: MemoryQuerySpecV1
    call_id: str
    task: asyncio.Task[tuple[MemoryExposureV1, object]] | None = None


class AsyncMemoryAgentCoordinator:
    """Worker-side async owner of session binding and actual Memory exposure.

    Every synchronous store call runs in this instance's dedicated executor.
    ``max_pending_calls`` bounds running plus queued calls before submission,
    so an overloaded Memory backend cannot create an unbounded executor queue.

    The first successful ``pin_session`` is a compare-and-set for the full
    ``(scope, assignment, group, incarnation)`` reference.  A later caller may
    repeat the exact pin but cannot replace it.  Concurrent turns are safe:
    each logical turn and query operation is installed under an asyncio lock,
    then executes independently with its own generated Memory trajectory.

    Cancellation by an HTTP/client task does not cancel the cached exposure
    operation.  A retry with the same logical operation key awaits that same
    task and reuses its runtime idempotency key and consumer ``call_id``.  The
    runtime consumer remains responsible for durable idempotency when its side
    effect crosses a process boundary.
    """

    def __init__(
        self,
        release_control_store: MemoryReleaseControlStore,
        runtime_store: MemoryRuntimeStore,
        *,
        max_workers: int = 4,
        max_pending_calls: int = 16,
    ) -> None:
        max_workers = _positive_integer(max_workers, "max_workers")
        max_pending_calls = _positive_integer(
            max_pending_calls,
            "max_pending_calls",
        )
        if not callable(
            getattr(release_control_store, "resolve_active_assignment", None)
        ):
            raise TypeError(
                "release_control_store must define resolve_active_assignment"
            )
        for method_name in (
            "begin_query",
            "resolve_query",
            "prepare_delivery",
            "submit_delivery",
        ):
            if not callable(getattr(runtime_store, method_name, None)):
                raise TypeError(f"runtime_store must define {method_name}")

        self._release_control_store = release_control_store
        self._runtime_store = runtime_store
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="areal-memory-agent",
        )
        self._pending_gate = asyncio.Semaphore(max_pending_calls)
        self._submitted_drained = asyncio.Event()
        self._submitted_drained.set()
        self._active_sync_calls_drained = asyncio.Event()
        self._active_sync_calls_drained.set()
        self._state_lock = asyncio.Lock()
        self._owner_loop: asyncio.AbstractEventLoop | None = None
        self._sessions: dict[str, _PinnedSession] = {}
        self._session_epoch: dict[str, int] = {}
        self._session_close_tasks: dict[str, asyncio.Task[None]] = {}
        self._turns: dict[tuple[str, str], _TurnState] = {}
        self._operations: dict[
            tuple[str, str, str],
            _ExposureOperation,
        ] = {}
        self._trajectory_ids: set[str] = set()
        self._submitted: set[Future[object]] = set()
        self._active_sync_calls = 0
        self._background_tasks: set[asyncio.Task[object]] = set()
        self._closed = False
        self._shutdown_task: asyncio.Task[None] | None = None

    def _running_loop(self) -> asyncio.AbstractEventLoop:
        loop = asyncio.get_running_loop()
        if self._owner_loop is None:
            self._owner_loop = loop
        elif self._owner_loop is not loop:
            raise RuntimeError(
                "AsyncMemoryAgentCoordinator cannot be shared across event loops"
            )
        return loop

    def _ensure_open(self) -> None:
        if self._closed:
            raise MemoryAgentCoordinatorClosedError("coordinator is closed")

    async def _call_sync(
        self,
        function: Callable[..., _T],
        /,
        *args: object,
        _allow_during_close: bool = False,
        **kwargs: object,
    ) -> _T:
        """Submit one call only after obtaining a running+queued capacity slot."""

        loop = self._running_loop()
        if self._closed and not _allow_during_close:
            raise MemoryAgentCoordinatorClosedError("coordinator is closed")
        self._active_sync_calls += 1
        self._active_sync_calls_drained.clear()
        try:
            await self._pending_gate.acquire()
            if self._closed and not _allow_during_close:
                self._pending_gate.release()
                raise MemoryAgentCoordinatorClosedError("coordinator is closed")
            try:
                future = self._executor.submit(partial(function, *args, **kwargs))
            except BaseException:
                self._pending_gate.release()
                raise
            self._submitted.add(future)
            self._submitted_drained.clear()
            future.add_done_callback(
                lambda completed: loop.call_soon_threadsafe(
                    self._submission_done,
                    completed,
                )
            )
            # asyncio cancellation cannot terminate a running Python callback.
            # Shielding prevents wrap_future from cancelling a queued callback;
            # capacity is released only by the concurrent Future's done hook.
            return await asyncio.shield(asyncio.wrap_future(future, loop=loop))
        finally:
            self._active_sync_calls -= 1
            if self._active_sync_calls == 0:
                self._active_sync_calls_drained.set()

    def _submission_done(self, future: Future[object]) -> None:
        """Release queue capacity only after the executor callback has ended."""

        if future in self._submitted:
            self._submitted.remove(future)
            self._pending_gate.release()
        if not self._submitted:
            self._submitted_drained.set()

    def _track_background_task(self, task: asyncio.Task[object]) -> None:
        self._background_tasks.add(task)
        task.add_done_callback(self._background_task_done)

    def _background_task_done(self, task: asyncio.Task[object]) -> None:
        self._background_tasks.discard(task)
        if not task.cancelled():
            task.exception()

    @staticmethod
    def _matches_pin(
        assignment: MemoryReleaseAssignmentV1,
        pin: MemoryAgentSessionPinV1,
    ) -> bool:
        return (
            assignment.scope,
            assignment.rollout_group_id,
            assignment.rollout_group_incarnation_sha256,
            assignment.assignment_id,
            assignment.content_hash,
        ) == (
            pin.scope,
            pin.rollout_group_id,
            pin.rollout_group_incarnation_sha256,
            pin.assignment_id,
            pin.assignment_content_sha256,
        )

    async def pin_session(
        self,
        session_key: str,
        pin: MemoryAgentSessionPinV1,
    ) -> MemoryReleaseAssignmentV1:
        """Resolve and compare-and-set one session's immutable assignment pin."""

        self._running_loop()
        self._ensure_open()
        session_key = _string(session_key, "session_key")
        if type(pin) is not MemoryAgentSessionPinV1:
            raise TypeError("pin must be a MemoryAgentSessionPinV1")

        async with self._state_lock:
            self._ensure_open()
            expected_epoch = self._session_epoch.get(session_key, 0)

        # Resolve on every invocation.  A cached pin is historical identity,
        # not a lease: revocation and expiry must fail even for an exact retry.
        assignment = await self._call_sync(
            self._release_control_store.resolve_active_assignment,
            pin.scope,
            pin.rollout_group_id,
            pin.rollout_group_incarnation_sha256,
            pin.assignment_id,
            pin.assignment_content_sha256,
        )
        if type(assignment) is not MemoryReleaseAssignmentV1:
            raise MemoryAgentSessionConflictError(
                "active assignment resolver returned a non-canonical value"
            )
        try:
            assignment.canonical_bytes()
        except (TypeError, ValueError) as error:
            raise MemoryAgentSessionConflictError(
                "active assignment failed integrity validation"
            ) from error
        if not self._matches_pin(assignment, pin):
            raise MemoryAgentSessionConflictError(
                "active assignment does not match the requested session pin"
            )

        async with self._state_lock:
            self._ensure_open()
            if self._session_epoch.get(session_key, 0) != expected_epoch:
                raise MemoryAgentSessionConflictError(
                    "session was closed while its Memory assignment was resolving"
                )
            existing = self._sessions.get(session_key)
            if existing is None:
                self._sessions[session_key] = _PinnedSession(
                    pin,
                    assignment,
                    epoch=expected_epoch,
                )
                return assignment
            if existing.closing:
                raise MemoryAgentSessionConflictError("session is closing")
            if existing.pin == pin:
                return existing.assignment
            raise MemoryAgentSessionConflictError(
                "a concurrent caller pinned the session to another assignment"
            )

    def _new_trajectory_id(
        self,
        session_key: str,
        turn_idempotency_key: str,
    ) -> str:
        while True:
            value = f"mtraj_{secrets.token_hex(32)}"
            if (
                value not in self._trajectory_ids
                and value != session_key
                and value != turn_idempotency_key
            ):
                self._trajectory_ids.add(value)
                return value

    async def start_turn(
        self,
        session_key: str,
        turn_idempotency_key: str,
    ) -> MemoryAgentTurnV1:
        """Get or create one independently identified Memory trajectory."""

        self._running_loop()
        self._ensure_open()
        session_key = _string(session_key, "session_key")
        turn_idempotency_key = _string(
            turn_idempotency_key,
            "turn_idempotency_key",
        )
        address = (session_key, turn_idempotency_key)
        async with self._state_lock:
            self._ensure_open()
            if session_key not in self._sessions:
                raise MemoryAgentSessionConflictError(
                    "session must be pinned before starting a Memory turn"
                )
            if self._sessions[session_key].closing:
                raise MemoryAgentSessionConflictError("session is closing")
            existing = self._turns.get(address)
            if existing is not None:
                return existing.handle
            handle = MemoryAgentTurnV1(
                session_key=session_key,
                turn_idempotency_key=turn_idempotency_key,
                memory_trajectory_id=self._new_trajectory_id(
                    session_key,
                    turn_idempotency_key,
                ),
            )
            self._turns[address] = _TurnState(handle=handle)
            return handle

    @staticmethod
    def _new_operation_id(prefix: str) -> str:
        return f"{prefix}_{secrets.token_hex(32)}"

    @staticmethod
    def _spec(
        *,
        assignment: MemoryReleaseAssignmentV1,
        turn: MemoryAgentTurnV1,
        query: bytes,
        query_sequence_no: int,
        idempotency_key: str,
    ) -> MemoryQuerySpecV1:
        return MemoryQuerySpecV1(
            scope=assignment.scope,
            assignment_id=assignment.assignment_id,
            assignment_content_sha256=assignment.content_hash,
            release_id=assignment.release_id,
            trajectory_id=turn.memory_trajectory_id,
            rollout_group_id=assignment.rollout_group_id,
            rollout_group_incarnation_sha256=(
                assignment.rollout_group_incarnation_sha256
            ),
            query_sequence_no=query_sequence_no,
            query_sha256=sha256(query).hexdigest(),
            task_policy_id=assignment.task_policy_id,
            task_policy_version_sha256=assignment.task_policy_version_sha256,
            task_policy_config_sha256=assignment.task_policy_config_sha256,
            retrieval_policy_id=assignment.retrieval_policy_id,
            retrieval_policy_version_sha256=(
                assignment.retrieval_policy_version_sha256
            ),
            retrieval_policy_config_sha256=(assignment.retrieval_policy_config_sha256),
            max_returned_items=assignment.max_returned_items,
            max_context_utf8_bytes=assignment.max_context_utf8_bytes,
            idempotency_key=idempotency_key,
        )

    async def _execute_exposure(
        self,
        assignment: MemoryReleaseAssignmentV1,
        operation: _ExposureOperation,
    ) -> tuple[MemoryExposureV1, object]:
        """Run the only path that may return an actual exposure record."""

        attempt = await self._call_sync(
            self._runtime_store.begin_query,
            operation.spec,
            _allow_during_close=True,
        )
        result = await self._call_sync(
            self._runtime_store.resolve_query,
            operation.spec.scope,
            attempt.attempt_id,
            query=operation.query,
            _allow_during_close=True,
        )
        delivery = await self._call_sync(
            self._runtime_store.prepare_delivery,
            operation.spec.scope,
            result.query_result_id,
            renderer_id=assignment.renderer_id,
            renderer_version_sha256=assignment.renderer_version_sha256,
            _allow_during_close=True,
        )
        submitted = await self._call_sync(
            self._runtime_store.submit_delivery,
            operation.spec.scope,
            delivery.delivery_id,
            consumer_id=assignment.consumer_id,
            consumer_version_sha256=assignment.consumer_version_sha256,
            call_id=operation.call_id,
            query=operation.query,
            history=operation.history,
            _allow_during_close=True,
        )
        if type(submitted) is not tuple or len(submitted) != 2:
            raise MemoryAgentTurnConflictError(
                "runtime consumer did not return an exposure and output pair"
            )
        exposure, output = submitted
        if type(exposure) is not MemoryExposureV1:
            raise MemoryAgentTurnConflictError(
                "runtime consumer did not return a canonical Memory exposure"
            )
        try:
            exposure.canonical_bytes()
        except (AttributeError, TypeError, ValueError) as error:
            raise MemoryAgentTurnConflictError(
                "runtime exposure failed canonical integrity validation"
            ) from error
        if (
            exposure.scope,
            exposure.assignment_id,
            exposure.assignment_content_sha256,
            exposure.trajectory_id,
            exposure.rollout_group_id,
            exposure.rollout_group_incarnation_sha256,
            exposure.delivery_id,
        ) != (
            operation.spec.scope,
            operation.spec.assignment_id,
            operation.spec.assignment_content_sha256,
            operation.spec.trajectory_id,
            operation.spec.rollout_group_id,
            operation.spec.rollout_group_incarnation_sha256,
            delivery.delivery_id,
        ):
            raise MemoryAgentTurnConflictError(
                "runtime exposure does not match the coordinator operation"
            )
        return exposure, output

    async def expose_memory(
        self,
        turn: MemoryAgentTurnV1,
        operation_key: str,
        *,
        query: bytes,
        history: tuple[bytes, ...] = (),
    ) -> tuple[MemoryExposureV1, object]:
        """Retrieve, render, and submit Memory through the actual consumer.

        The same ``operation_key`` is an immutable, cancellation-safe retry.
        Reuse with different query/history bytes fails closed.  There is no
        API for a DataProxy or raw-passthrough route to manufacture an exposure
        from forwarded content; the returned record must come from
        ``runtime_store.submit_delivery``.
        """

        self._running_loop()
        self._ensure_open()
        if type(turn) is not MemoryAgentTurnV1:
            raise TypeError("turn must be a MemoryAgentTurnV1")
        operation_key = _string(operation_key, "operation_key")
        if type(query) is not bytes:
            raise TypeError("query must be bytes")
        history = _history(history)
        turn_address = (turn.session_key, turn.turn_idempotency_key)
        operation_address = (*turn_address, operation_key)

        async with self._state_lock:
            self._ensure_open()
            turn_state = self._turns.get(turn_address)
            if turn_state is None or turn_state.handle != turn:
                raise MemoryAgentTurnConflictError(
                    "turn was not issued by this coordinator"
                )
            session = self._sessions.get(turn.session_key)
            if session is None:
                raise MemoryAgentSessionConflictError(
                    "turn session is no longer pinned"
                )
            if session.closing:
                raise MemoryAgentSessionConflictError("turn session is closing")
            operation = self._operations.get(operation_address)
            if operation is None:
                operation = _ExposureOperation(
                    query=query,
                    history=history,
                    spec=self._spec(
                        assignment=session.assignment,
                        turn=turn,
                        query=query,
                        query_sequence_no=turn_state.next_query_sequence_no,
                        idempotency_key=self._new_operation_id("mquery"),
                    ),
                    call_id=self._new_operation_id("mcall"),
                )
                turn_state.next_query_sequence_no += 1
                task = asyncio.create_task(
                    self._execute_exposure(session.assignment, operation),
                    name=(
                        "areal-memory-exposure:"
                        f"{turn.memory_trajectory_id}:{operation_key}"
                    ),
                )
                operation.task = task
                self._track_background_task(task)
                self._operations[operation_address] = operation
            elif operation.query != query or operation.history != history:
                raise MemoryAgentTurnConflictError(
                    "operation key is already bound to different query/history bytes"
                )
            task = operation.task
            if task is None:  # pragma: no cover - construction is atomic above
                raise MemoryAgentTurnConflictError("exposure operation is incomplete")

        # Caller cancellation must not cancel the consumer operation.  A retry
        # finds and awaits this exact task, preserving the generated call_id.
        return await asyncio.shield(task)

    async def _close_session(
        self,
        session_key: str,
        session: _PinnedSession,
    ) -> None:
        """Drain one session before deleting its immutable binding."""

        async with self._state_lock:
            tasks = tuple(
                operation.task
                for address, operation in self._operations.items()
                if address[0] == session_key and operation.task is not None
            )
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        async with self._state_lock:
            if self._sessions.get(session_key) is session:
                self._sessions.pop(session_key, None)
                for address in tuple(self._turns):
                    if address[0] == session_key:
                        self._turns.pop(address, None)
                for address in tuple(self._operations):
                    if address[0] == session_key:
                        self._operations.pop(address, None)
            self._session_close_tasks.pop(session_key, None)

    async def close_session(self, session_key: str) -> None:
        """Drain and forget a session so a new incarnation may bind its key.

        Closing increments a tombstone epoch before waiting.  Consequently, a
        pin that began before close cannot race in afterwards, and every old
        turn handle remains invalid even if the same textual session/run keys
        are later reused.
        """

        self._running_loop()
        self._ensure_open()
        session_key = _string(session_key, "session_key")
        async with self._state_lock:
            self._ensure_open()
            existing_task = self._session_close_tasks.get(session_key)
            if existing_task is not None:
                task = existing_task
            else:
                self._session_epoch[session_key] = (
                    self._session_epoch.get(session_key, 0) + 1
                )
                session = self._sessions.get(session_key)
                if session is None:
                    return
                session.closing = True
                task = asyncio.create_task(
                    self._close_session(session_key, session),
                    name=f"areal-memory-close-session:{session_key}",
                )
                self._session_close_tasks[session_key] = task
                self._track_background_task(task)
        await asyncio.shield(task)

    async def _shutdown(self, tasks: tuple[asyncio.Task[object], ...]) -> None:
        """Drain all admitted work without blocking the owner event loop."""

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        # Include pin_session calls (which are not background tasks), callers
        # cancelled while waiting for executor capacity, and their still-running
        # Python callbacks before shutting down the dedicated executor.
        while self._active_sync_calls:
            await self._active_sync_calls_drained.wait()
        while self._submitted:
            await self._submitted_drained.wait()
        await asyncio.to_thread(
            self._executor.shutdown,
            wait=True,
            cancel_futures=False,
        )

    async def aclose(self) -> None:
        """Stop admission, drain shielded operations, then close the executor."""

        self._running_loop()
        async with self._state_lock:
            task = self._shutdown_task
            if task is None:
                self._closed = True
                task = asyncio.create_task(
                    self._shutdown(tuple(self._background_tasks)),
                    name="areal-memory-agent-shutdown",
                )
                self._shutdown_task = task
        # Concurrent and cancelled close callers converge on the same drain.
        await asyncio.shield(task)

    async def __aenter__(self) -> AsyncMemoryAgentCoordinator:
        self._running_loop()
        self._ensure_open()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        del exc_info
        await self.aclose()
