# SPDX-License-Identifier: Apache-2.0

"""Worker-owned, turn-scoped access to the Agent Memory coordinator.

This module defines an in-process capability seam only.  It intentionally does
not parse the DataProxy wire envelope or enable Memory in the Worker HTTP app.
That integration needs an independently authenticated DataProxy-to-Worker hop
and a server-side principal/session-to-scope authorization grant; a pin is not
authorization.

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
from areal.v2.agent_service.types import AgentRequest, MemoryTurnResultV1
from areal.v2.memory_service.runtime_types import MemoryExposureV1


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

        async with self.__state_lock:
            if self.__closed:
                raise MemoryAgentTurnConflictError("Memory turn capability is closed")
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
        await asyncio.shield(task)


def bind_memory_turn_capability(
    request: AgentRequest,
    coordinator: AsyncMemoryAgentCoordinator,
    turn: MemoryAgentTurnV1,
) -> WorkerMemoryTurnCapability:
    """Bind one exact request/turn identity without changing request wire state."""

    if type(request) is not AgentRequest:
        raise TypeError("request must be an AgentRequest")
    capability = WorkerMemoryTurnCapability(coordinator, turn)
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
    # The read-only property is backed by a deliberately non-dataclass runtime
    # attribute.  asdict(request), repr(request), and other dataclass-field
    # serializers keep exactly the pre-Memory request surface.
    request._areal_memory_turn_capability = capability  # type: ignore[attr-defined]
    return capability


__all__ = ["WorkerMemoryTurnCapability", "bind_memory_turn_capability"]
