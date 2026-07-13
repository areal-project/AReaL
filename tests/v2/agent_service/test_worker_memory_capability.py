# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import UTC, datetime
from hashlib import sha256
from unittest.mock import patch

import pytest

from areal.v2.agent_service.memory import (
    AsyncMemoryAgentCoordinator,
    MemoryAgentTurnConflictError,
    MemoryAgentTurnV1,
)
from areal.v2.agent_service.types import (
    AgentRequest,
    AgentResponse,
    EventEmitter,
    MemoryTurnResultV1,
)
from areal.v2.agent_service.worker.app import create_worker_app
from areal.v2.agent_service.worker.memory import (
    WorkerMemoryTurnCapability,
    bind_memory_turn_capability,
)
from areal.v2.memory_service import runtime_types
from areal.v2.memory_service.runtime_types import (
    MemoryExposureStatus,
    MemoryExposureV1,
)
from areal.v2.memory_service.types import MemoryScope

_NOW = datetime(2026, 7, 13, tzinfo=UTC)
_SCOPE = MemoryScope("tenant-1", "agent-long-term-memory", "subject-1")

httpx = pytest.importorskip("httpx")


def _hash(label: str) -> str:
    return sha256(label.encode()).hexdigest()


def _turn() -> MemoryAgentTurnV1:
    return MemoryAgentTurnV1(
        session_key="session-1",
        turn_idempotency_key="run-1",
        memory_trajectory_id="mtraj_test-1",
    )


def _exposure(turn: MemoryAgentTurnV1) -> MemoryExposureV1:
    values = {
        "scope": _SCOPE,
        "assignment_id": f"masn_{_hash('assignment')[:24]}",
        "assignment_content_sha256": _hash("assignment"),
        "release_id": f"rel_{_hash('release')[:24]}",
        "release_content_sha256": _hash("release"),
        "trajectory_id": turn.memory_trajectory_id,
        "rollout_group_id": "rollout-group-1",
        "rollout_group_incarnation_sha256": _hash("incarnation"),
        "attempt_id": "attempt-1",
        "attempt_content_sha256": _hash("attempt"),
        "query_result_id": "result-1",
        "query_result_content_sha256": _hash("result"),
        "delivery_id": "delivery-1",
        "delivery_content_sha256": _hash("delivery"),
        "consumer_ack_id": "ack-1",
        "consumer_ack_content_sha256": _hash("ack"),
        "eligible_revisions": (),
        "retrieved_revisions": (),
        "returned_revisions": (),
        "injected_revisions": (),
        "status": MemoryExposureStatus.MEMORY_OFF,
    }
    canonical = runtime_types._exposure_canonical_bytes(**values)
    content_hash = sha256(canonical).hexdigest()
    return MemoryExposureV1(
        **values,
        exposure_id=f"mexp_{content_hash[:24]}",
        content_hash=content_hash,
        created_at=_NOW,
    )


class _Coordinator(AsyncMemoryAgentCoordinator):
    def __init__(
        self,
        exposure: object,
        *,
        output: object = "consumer-output",
        block: bool = False,
    ) -> None:
        self.exposure = exposure
        self.output = output
        self.calls: list[tuple[object, ...]] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        if not block:
            self.release.set()

    async def expose_memory(
        self,
        turn: MemoryAgentTurnV1,
        operation_key: str,
        *,
        query: bytes,
        history: tuple[bytes, ...] = (),
    ) -> tuple[MemoryExposureV1, object]:
        self.calls.append((turn, operation_key, query, history))
        self.started.set()
        await self.release.wait()
        return self.exposure, self.output  # type: ignore[return-value]


@pytest.mark.asyncio
async def test_capability_returns_only_consumer_output_and_exposure_pointer() -> None:
    turn = _turn()
    exposure = _exposure(turn)
    coordinator = _Coordinator(exposure, output={"answer": "from-memory"})
    capability = WorkerMemoryTurnCapability(coordinator, turn)

    result = await capability.expose_memory(
        "retrieve-request",
        query=b"future question",
        history=(b"prior turn",),
    )

    assert result == MemoryTurnResultV1(
        output=None,
        exposure_id=exposure.exposure_id,
        exposure_content_sha256=exposure.content_hash,
    )
    assert result.output == {"answer": "from-memory"}
    assert coordinator.calls == [
        (turn, "retrieve-request", b"future question", (b"prior turn",))
    ]
    assert not hasattr(result, "scope")
    assert not hasattr(result, "assignment_id")
    assert not hasattr(result, "consumer_ack")
    await capability.aclose()


@pytest.mark.asyncio
async def test_close_revokes_new_operations_and_drains_an_admitted_call() -> None:
    turn = _turn()
    coordinator = _Coordinator(_exposure(turn), block=True)
    capability = WorkerMemoryTurnCapability(coordinator, turn)

    operation = asyncio.create_task(
        capability.expose_memory("retrieve-request", query=b"question")
    )
    await coordinator.started.wait()
    close = asyncio.create_task(capability.aclose())
    await asyncio.sleep(0)

    assert not close.done()
    with pytest.raises(MemoryAgentTurnConflictError, match="capability is closed"):
        await capability.expose_memory("late-request", query=b"late")

    coordinator.release.set()
    assert (await operation).output == "consumer-output"
    await close
    await capability.aclose()
    assert len(coordinator.calls) == 1


@pytest.mark.asyncio
async def test_caller_cancellation_does_not_cancel_admitted_consumer_work() -> None:
    turn = _turn()
    coordinator = _Coordinator(_exposure(turn), block=True)
    capability = WorkerMemoryTurnCapability(coordinator, turn)
    caller = asyncio.create_task(
        capability.expose_memory("retrieve-request", query=b"question")
    )
    await coordinator.started.wait()

    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller

    close = asyncio.create_task(capability.aclose())
    await asyncio.sleep(0)
    assert not close.done()
    coordinator.release.set()
    await close
    assert len(coordinator.calls) == 1


@pytest.mark.asyncio
async def test_cancelled_close_is_rejoined_by_the_next_close_caller() -> None:
    turn = _turn()
    coordinator = _Coordinator(_exposure(turn), block=True)
    capability = WorkerMemoryTurnCapability(coordinator, turn)
    operation = asyncio.create_task(
        capability.expose_memory("retrieve-request", query=b"question")
    )
    await coordinator.started.wait()

    first_close = asyncio.create_task(capability.aclose())
    await asyncio.sleep(0)
    first_close.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first_close

    second_close = asyncio.create_task(capability.aclose())
    await asyncio.sleep(0)
    assert not second_close.done()
    coordinator.release.set()
    await operation
    await second_close
    assert len(coordinator.calls) == 1


@pytest.mark.asyncio
async def test_capability_rejects_a_forged_exposure_before_returning_output() -> None:
    coordinator = _Coordinator(object(), output="forged")
    capability = WorkerMemoryTurnCapability(coordinator, _turn())

    with pytest.raises(MemoryAgentTurnConflictError, match="canonical"):
        await capability.expose_memory("retrieve-request", query=b"question")

    await capability.aclose()


@pytest.mark.asyncio
async def test_capability_rejects_an_exposure_from_another_turn() -> None:
    turn = _turn()
    other_turn = MemoryAgentTurnV1(
        session_key=turn.session_key,
        turn_idempotency_key="run-2",
        memory_trajectory_id="mtraj_test-2",
    )
    coordinator = _Coordinator(_exposure(other_turn), output="wrong-turn")
    capability = WorkerMemoryTurnCapability(coordinator, turn)

    with pytest.raises(MemoryAgentTurnConflictError, match="bound Memory turn"):
        await capability.expose_memory("retrieve-request", query=b"question")

    await capability.aclose()


class _CaptureMemoryAgent:
    requests: list[AgentRequest] = []

    async def run(
        self,
        request: AgentRequest,
        *,
        emitter: EventEmitter,
    ) -> AgentResponse:
        del emitter
        self.requests.append(request)
        return AgentResponse(summary="ok")


@pytest.mark.asyncio
async def test_worker_http_body_cannot_forge_a_memory_capability() -> None:
    _CaptureMemoryAgent.requests.clear()
    with patch(
        "areal.v2.agent_service.worker.app.import_from_string",
        return_value=_CaptureMemoryAgent,
    ):
        app = create_worker_app("mock.path")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://worker",
    ) as client:
        response = await client.post(
            "/run",
            json={
                "message": "hello",
                "session_key": "s1",
                "run_id": "r1",
                "memory": {"forged": True},
                "metadata": {"areal_memory": {"forged": True}},
            },
        )

    assert response.status_code == 200
    assert len(_CaptureMemoryAgent.requests) == 1
    assert _CaptureMemoryAgent.requests[0].memory is None


def test_agent_request_memory_is_process_local_not_dataclass_wire_state() -> None:
    turn = _turn()
    request = AgentRequest(
        message="hello",
        session_key=turn.session_key,
        run_id=turn.turn_idempotency_key,
    )
    assert request.memory is None

    capability = bind_memory_turn_capability(
        request,
        _Coordinator(_exposure(turn)),
        turn,
    )
    assert request.memory is capability
    assert "memory=" not in repr(request)
    assert asdict(request) == {
        "message": "hello",
        "session_key": turn.session_key,
        "run_id": turn.turn_idempotency_key,
        "history": [],
        "queue_mode": request.queue_mode,
        "metadata": {},
    }

    with pytest.raises(MemoryAgentTurnConflictError, match="already has"):
        bind_memory_turn_capability(
            request,
            _Coordinator(_exposure(turn)),
            turn,
        )


@pytest.mark.parametrize(
    ("session_key", "run_id"),
    [
        ("another-session", "run-1"),
        ("session-1", "another-run"),
    ],
)
def test_binding_rejects_a_request_from_another_session_or_run(
    session_key: str,
    run_id: str,
) -> None:
    turn = _turn()
    request = AgentRequest(
        message="hello",
        session_key=session_key,
        run_id=run_id,
    )

    with pytest.raises(MemoryAgentTurnConflictError, match="identity"):
        bind_memory_turn_capability(
            request,
            _Coordinator(_exposure(turn)),
            turn,
        )
    assert request.memory is None


@pytest.mark.parametrize(
    ("exposure_id", "digest"),
    [
        ("mexp_wrong", _hash("exposure")),
        (f"mexp_{_hash('exposure')[:24]}", "not-a-digest"),
    ],
)
def test_result_pointer_must_be_content_addressed(
    exposure_id: str,
    digest: str,
) -> None:
    with pytest.raises(ValueError):
        MemoryTurnResultV1(
            output=None,
            exposure_id=exposure_id,
            exposure_content_sha256=digest,
        )
