# SPDX-License-Identifier: Apache-2.0

"""Worker-owned, turn-scoped access to the Agent Memory coordinator.

This module defines an in-process capability seam only.  It intentionally does
not parse the DataProxy wire envelope or enable Memory in the Worker HTTP app.
The explicit ``from_authorized_turn`` path consumes broker-issued turns and
fresh grants; the legacy bare-coordinator path remains for compatibility.  HTTP
integration still needs a trusted principal source in addition to the
authenticated DataProxy-to-Worker hop.  A pin is not authorization.

The object narrows accidental API use by agent implementations.  It is not a
sandbox against malicious Python loaded into the same process: such code can
introspect or monkey-patch process state.  Deployments that treat plugins as
adversarial must put the Memory broker and its credentials in another process.
"""

from __future__ import annotations

import asyncio

from areal.v2.agent_service.memory import (
    AsyncMemoryAgentCoordinator,
    MemoryAgentTurnConflictError,
    MemoryAgentTurnV1,
)
from areal.v2.agent_service.memory_broker import (
    AuthorizedMemoryAgentBroker,
    AuthorizedMemoryTurnV1,
)
from areal.v2.agent_service.types import AgentRequest, MemoryTurnResultV1
from areal.v2.memory_service.runtime_types import MemoryExposureV1


def _operation_inputs(
    operation_key: object,
    query: object,
    history: object,
) -> tuple[str, bytes, tuple[bytes, ...]]:
    if type(operation_key) is not str:
        raise TypeError("operation_key must be a str")
    if not operation_key.strip():
        raise ValueError("operation_key must not be blank")
    try:
        operation_key.encode("utf-8", "strict")
    except UnicodeEncodeError as error:
        raise ValueError("operation_key must be valid UTF-8") from error
    if type(query) is not bytes:
        raise TypeError("query must be bytes")
    if type(history) is not tuple:
        raise TypeError("history must be a tuple")
    history = tuple(tuple.__iter__(history))
    if any(type(item) is not bytes for item in history):
        raise TypeError("history must contain bytes")
    return operation_key, query, history


class WorkerMemoryTurnCapability:
    """Host-owned adapter exposing one coordinator-issued turn to an agent.

    New operations are rejected after :meth:`aclose`.  Operations admitted
    before close are drained by one cancellation-shielded cleanup task.  This
    preserves the coordinator's consumer-boundary semantics: cancellation may
    not erase or duplicate a side effect that has already reached a consumer.
    """

    def __init__(
        self,
        coordinator: AsyncMemoryAgentCoordinator,
        turn: MemoryAgentTurnV1,
    ) -> None:
        if not isinstance(coordinator, AsyncMemoryAgentCoordinator):
            raise TypeError("coordinator must be an AsyncMemoryAgentCoordinator")
        if type(turn) is not MemoryAgentTurnV1:
            raise TypeError("turn must be a MemoryAgentTurnV1")
        self.__coordinator = coordinator
        self.__turn = turn
        self.__state_lock = asyncio.Lock()
        self.__active: set[asyncio.Task[MemoryTurnResultV1]] = set()
        self.__closed = False
        self.__close_task: asyncio.Task[None] | None = None
        self.__authorization_broker: AuthorizedMemoryAgentBroker | None = None
        self.__authorized_turn: AuthorizedMemoryTurnV1 | None = None

    @classmethod
    def from_authorized_turn(
        cls,
        broker: AuthorizedMemoryAgentBroker,
        turn: AuthorizedMemoryTurnV1,
    ) -> WorkerMemoryTurnCapability:
        """Construct the explicit grant-checked path for a broker-issued turn."""

        if type(broker) is not AuthorizedMemoryAgentBroker:
            raise TypeError("broker must be an AuthorizedMemoryAgentBroker")
        if type(turn) is not AuthorizedMemoryTurnV1:
            raise TypeError("turn must be an AuthorizedMemoryTurnV1")
        coordinator, coordinator_turn = broker._coordinator_for_turn(turn)
        capability = cls(coordinator, coordinator_turn)
        capability.__authorization_broker = broker
        capability.__authorized_turn = turn
        # Construction and registration have no intervening await, so session
        # close cannot miss a capability that may later reach the coordinator.
        broker._register_capability(turn, capability)
        return capability

    async def _prepare_authorization(self) -> object:
        broker = self.__authorization_broker
        turn = self.__authorized_turn
        if broker is None:
            if turn is not None:  # pragma: no cover - private invariant
                raise MemoryAgentTurnConflictError(
                    "Memory capability authorization state is incomplete"
                )
            return None
        if turn is None:  # pragma: no cover - private invariant
            raise MemoryAgentTurnConflictError(
                "Memory capability authorization state is incomplete"
            )
        return await broker._authorize_exposure(turn)

    def _consume_authorization(self, ticket: object) -> None:
        broker = self.__authorization_broker
        if broker is None:
            if ticket is not None:  # pragma: no cover - private invariant
                raise MemoryAgentTurnConflictError(
                    "unexpected Memory authorization ticket"
                )
            return
        broker._consume_exposure_ticket(self, ticket)

    async def _expose(
        self,
        operation_key: str,
        *,
        query: bytes,
        history: tuple[bytes, ...],
    ) -> MemoryTurnResultV1:
        submitted = await self.__coordinator.expose_memory(
            self.__turn,
            operation_key,
            query=query,
            history=history,
        )
        if type(submitted) is not tuple or len(submitted) != 2:
            raise MemoryAgentTurnConflictError(
                "coordinator did not return an exposure and output pair"
            )
        exposure, output = submitted
        if type(exposure) is not MemoryExposureV1:
            raise MemoryAgentTurnConflictError(
                "coordinator did not return a canonical Memory exposure"
            )
        try:
            exposure.canonical_bytes()
        except (TypeError, ValueError) as error:
            raise MemoryAgentTurnConflictError(
                "coordinator exposure failed canonical integrity validation"
            ) from error
        if exposure.trajectory_id != self.__turn.memory_trajectory_id:
            raise MemoryAgentTurnConflictError(
                "coordinator exposure does not match the bound Memory turn"
            )
        return MemoryTurnResultV1(
            output=output,
            exposure_id=exposure.exposure_id,
            exposure_content_sha256=exposure.content_hash,
        )

    def _operation_done(self, task: asyncio.Task[MemoryTurnResultV1]) -> None:
        self.__active.discard(task)
        # A cancelled HTTP/agent caller leaves the shielded operation running.
        # Retrieve its eventual exception here so it cannot become an
        # unobserved-task warning when close happens after it has finished.
        if not task.cancelled():
            task.exception()

    async def expose_memory(
        self,
        operation_key: str,
        *,
        query: bytes,
        history: tuple[bytes, ...] = (),
    ) -> MemoryTurnResultV1:
        """Submit one immutable operation through the trusted consumer path."""

        operation_key, query, history = _operation_inputs(
            operation_key,
            query,
            history,
        )
        async with self.__state_lock:
            if self.__closed:
                raise MemoryAgentTurnConflictError("Memory turn capability is closed")
        # Authorization is deliberately outside the capability lock and before
        # operation registration.  A caller cancelled here cannot later start
        # a coordinator side effect when a blocking resolver finally returns.
        ticket = await self._prepare_authorization()

        async with self.__state_lock:
            if self.__closed:
                raise MemoryAgentTurnConflictError("Memory turn capability is closed")
            # This synchronous consume rechecks broker epoch/turn/target while
            # the capability lock is held.  Task creation follows without an
            # await, making close versus admission a single event-loop step.
            self._consume_authorization(ticket)
            task = asyncio.create_task(
                self._expose(
                    operation_key,
                    query=query,
                    history=history,
                ),
                name=(
                    f"areal-worker-memory-operation:{self.__turn.memory_trajectory_id}"
                ),
            )
            self.__active.add(task)
            task.add_done_callback(self._operation_done)

        # The coordinator also shields its consumer operation.  Shielding this
        # adapter task keeps it visible to aclose so the Worker can drain every
        # operation admitted during the turn.
        return await asyncio.shield(task)

    async def _drain(self, tasks: tuple[asyncio.Task[MemoryTurnResultV1], ...]) -> None:
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def aclose(self) -> None:
        """Revoke new operations and drain every operation already admitted."""

        async with self.__state_lock:
            task = self.__close_task
            if task is None:
                self.__closed = True
                task = asyncio.create_task(
                    self._drain(tuple(self.__active)),
                    name=(
                        "areal-worker-memory-capability-close:"
                        f"{self.__turn.memory_trajectory_id}"
                    ),
                )
                self.__close_task = task
                broker = self.__authorization_broker
                if broker is not None:
                    task.add_done_callback(
                        lambda _: broker._unregister_capability(self)
                    )
        await asyncio.shield(task)


def bind_memory_turn_capability(
    request: AgentRequest,
    coordinator: AsyncMemoryAgentCoordinator,
    turn: MemoryAgentTurnV1,
) -> WorkerMemoryTurnCapability:
    """Bind one exact request/turn identity without changing request wire state."""

    if type(request) is not AgentRequest:
        raise TypeError("request must be an AgentRequest")
    _validate_request_turn(request, turn)
    capability = WorkerMemoryTurnCapability(coordinator, turn)
    _attach_capability(request, capability)
    return capability


def bind_authorized_memory_turn_capability(
    request: AgentRequest,
    broker: AuthorizedMemoryAgentBroker,
    turn: AuthorizedMemoryTurnV1,
) -> WorkerMemoryTurnCapability:
    """Bind a broker-issued turn whose every exposure is freshly authorized."""

    if type(request) is not AgentRequest:
        raise TypeError("request must be an AgentRequest")
    if type(turn) is not AuthorizedMemoryTurnV1:
        raise TypeError("turn must be an AuthorizedMemoryTurnV1")
    _validate_request_turn(request, turn.turn)
    capability = WorkerMemoryTurnCapability.from_authorized_turn(broker, turn)
    _attach_capability(request, capability)
    return capability


def _validate_request_turn(request: AgentRequest, turn: MemoryAgentTurnV1) -> None:
    if type(turn) is not MemoryAgentTurnV1:
        raise TypeError("turn must be a MemoryAgentTurnV1")
    if (
        type(request.session_key) is not str
        or type(request.run_id) is not str
        or request.session_key != turn.session_key
        or request.run_id != turn.turn_idempotency_key
    ):
        raise MemoryAgentTurnConflictError(
            "AgentRequest identity does not match the Memory turn"
        )
    if request.memory is not None:
        raise MemoryAgentTurnConflictError(
            "AgentRequest already has a Memory turn capability"
        )


def _attach_capability(
    request: AgentRequest,
    capability: WorkerMemoryTurnCapability,
) -> None:
    # The read-only property is backed by a deliberately non-dataclass runtime
    # attribute.  asdict(request), repr(request), and other dataclass-field
    # serializers keep exactly the pre-Memory request surface.
    request._areal_memory_turn_capability = capability  # type: ignore[attr-defined]


__all__ = [
    "WorkerMemoryTurnCapability",
    "bind_authorized_memory_turn_capability",
    "bind_memory_turn_capability",
]
