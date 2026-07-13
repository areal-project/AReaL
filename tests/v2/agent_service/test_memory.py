# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import wraps
from hashlib import sha256
from threading import Event, Lock, current_thread, get_ident
from types import SimpleNamespace

import pytest

from areal.v2.agent_service.memory import (
    AsyncMemoryAgentCoordinator,
    MemoryAgentCoordinatorClosedError,
    MemoryAgentSessionConflictError,
    MemoryAgentSessionPinV1,
    MemoryAgentTurnConflictError,
    MemoryAgentTurnV1,
)
from areal.v2.memory_service.errors import MemoryReleaseAssignmentConflictError
from areal.v2.memory_service.release_control_types import (
    MemoryReleaseAssignmentConsumerKind,
    MemoryReleaseAssignmentV1,
)
from areal.v2.memory_service.runtime_types import MemoryExposureV1
from areal.v2.memory_service.types import MemoryScope

_NOW = datetime(2026, 7, 13, tzinfo=UTC)
_SCOPE = MemoryScope("tenant-1", "agent-long-term-memory", "subject-1")


def _hash(label: str) -> str:
    return sha256(label.encode()).hexdigest()


def _assignment(
    *,
    suffix: str = "a",
    incarnation: str | None = None,
) -> MemoryReleaseAssignmentV1:
    release_hash = _hash("release")
    attestation_hash = _hash("attestation")
    return MemoryReleaseAssignmentV1.create(
        scope=_SCOPE,
        rollout_group_id="rollout-group-1",
        rollout_group_incarnation_sha256=(
            incarnation or _hash("rollout-incarnation-1")
        ),
        attestation_id=f"mrat_{attestation_hash[:24]}",
        attestation_content_sha256=attestation_hash,
        release_id=f"rel_{release_hash[:24]}",
        release_content_sha256=release_hash,
        release_graph_sha256=_hash("release-graph"),
        assignment_policy_id="assignment-policy",
        assignment_policy_version_sha256=_hash("assignment-policy-v1"),
        assignment_policy_config_sha256=_hash("assignment-policy-config"),
        task_policy_id="agent-task-policy",
        task_policy_version_sha256=_hash("task-policy-v1"),
        task_policy_config_sha256=_hash(f"task-policy-config-{suffix}"),
        retrieval_policy_id="release-order-retriever",
        retrieval_policy_version_sha256=_hash("retriever-v1"),
        retrieval_policy_config_sha256=_hash("retriever-config"),
        renderer_id="agent-memory-renderer",
        renderer_version_sha256=_hash("renderer-v1"),
        renderer_config_sha256=_hash("renderer-config"),
        consumer_kind=MemoryReleaseAssignmentConsumerKind.MODEL_CALL,
        consumer_id="agent-model-boundary",
        consumer_version_sha256=_hash("consumer-v1"),
        consumer_config_sha256=_hash("consumer-config"),
        max_returned_items=4,
        max_context_utf8_bytes=4096,
        evaluated_at=_NOW,
        assigned_at=_NOW,
        assignment_valid_until=_NOW + timedelta(days=1),
        idempotency_key=f"assignment-{suffix}",
    )


def _pin(assignment: MemoryReleaseAssignmentV1) -> MemoryAgentSessionPinV1:
    return MemoryAgentSessionPinV1(
        scope=assignment.scope,
        rollout_group_id=assignment.rollout_group_id,
        rollout_group_incarnation_sha256=(assignment.rollout_group_incarnation_sha256),
        assignment_id=assignment.assignment_id,
        assignment_content_sha256=assignment.content_hash,
    )


class _ControlStore:
    def __init__(
        self,
        *assignments: MemoryReleaseAssignmentV1,
        block: bool = False,
    ) -> None:
        self.assignments = {item.assignment_id: item for item in assignments}
        self.block = block
        self.started = Event()
        self.release = Event()
        self.calls = 0
        self.thread_names: list[str] = []
        self._lock = Lock()
        self.active_calls = 0
        self.max_active_calls = 0
        self.active = True

    def resolve_active_assignment(
        self,
        scope,
        rollout_group_id,
        rollout_group_incarnation_sha256,
        assignment_id,
        assignment_content_sha256,
    ):
        with self._lock:
            self.calls += 1
            self.active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)
        self.thread_names.append(current_thread().name)
        self.started.set()
        try:
            if self.block:
                assert self.release.wait(timeout=5)
            if not self.active:
                raise MemoryReleaseAssignmentConflictError("assignment revoked")
            assignment = self.assignments[assignment_id]
            assert (
                assignment.scope,
                assignment.rollout_group_id,
                assignment.rollout_group_incarnation_sha256,
                assignment.content_hash,
            ) == (
                scope,
                rollout_group_id,
                rollout_group_incarnation_sha256,
                assignment_content_sha256,
            )
            return assignment
        finally:
            with self._lock:
                self.active_calls -= 1


@dataclass
class _RuntimeCall:
    stage: str
    thread_id: int
    value: object


class _RuntimeStore:
    def __init__(self) -> None:
        self.calls: list[_RuntimeCall] = []
        self.specs = []
        self.spec_by_attempt_id = {}
        self.spec_by_result_id = {}
        self.spec_by_delivery_id = {}
        self.call_ids: list[str] = []
        self.consumer_side_effects = 0
        self.submit_started = Event()
        self.submit_release = Event()
        self.block_submit = False
        self.invalid_consumer_result = False

    def begin_query(self, spec):
        self.calls.append(_RuntimeCall("begin", get_ident(), spec))
        self.specs.append(spec)
        attempt_id = f"attempt-{spec.idempotency_key}"
        self.spec_by_attempt_id[attempt_id] = spec
        return SimpleNamespace(attempt_id=attempt_id)

    def resolve_query(self, scope, attempt_id, *, query):
        del scope, query
        self.calls.append(_RuntimeCall("resolve", get_ident(), attempt_id))
        result_id = f"result-{attempt_id}"
        self.spec_by_result_id[result_id] = self.spec_by_attempt_id[attempt_id]
        return SimpleNamespace(query_result_id=result_id)

    def prepare_delivery(
        self,
        scope,
        query_result_id,
        *,
        renderer_id,
        renderer_version_sha256,
    ):
        del scope, renderer_id, renderer_version_sha256
        self.calls.append(_RuntimeCall("render", get_ident(), query_result_id))
        delivery_id = f"delivery-{query_result_id}"
        self.spec_by_delivery_id[delivery_id] = self.spec_by_result_id[query_result_id]
        return SimpleNamespace(delivery_id=delivery_id)

    def submit_delivery(
        self,
        scope,
        delivery_id,
        *,
        consumer_id,
        consumer_version_sha256,
        call_id,
        query,
        history,
    ):
        del consumer_id, consumer_version_sha256, query, history
        self.calls.append(_RuntimeCall("consumer", get_ident(), delivery_id))
        self.call_ids.append(call_id)
        self.consumer_side_effects += 1
        self.submit_started.set()
        if self.block_submit:
            assert self.submit_release.wait(timeout=5)
        if self.invalid_consumer_result:
            return "raw-forwarded-without-consumer-receipt", object()

        spec = self.spec_by_delivery_id[delivery_id]
        exposure = object.__new__(MemoryExposureV1)
        for field_name, value in (
            ("scope", scope),
            ("assignment_id", spec.assignment_id),
            ("assignment_content_sha256", spec.assignment_content_sha256),
            ("trajectory_id", spec.trajectory_id),
            ("rollout_group_id", spec.rollout_group_id),
            (
                "rollout_group_incarnation_sha256",
                spec.rollout_group_incarnation_sha256,
            ),
            ("delivery_id", delivery_id),
        ):
            object.__setattr__(exposure, field_name, value)
        return exposure, f"output:{call_id}"


async def _wait_until(event: Event) -> None:
    for _ in range(500):
        if event.is_set():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("thread event was not set")


def _async_test(function):
    """Keep these asyncio tests runnable in the minimal Memory test environment."""

    @wraps(function)
    def run():
        return asyncio.run(function())

    return run


@_async_test
async def test_pin_is_nonblocking_first_write_cas_and_incarnation_immutable() -> None:
    assignment = _assignment()
    replacement = _assignment(
        suffix="replacement",
        incarnation=_hash("rollout-incarnation-2"),
    )
    control = _ControlStore(assignment, replacement, block=True)
    runtime = _RuntimeStore()
    coordinator = AsyncMemoryAgentCoordinator(control, runtime, max_workers=1)
    try:
        pin_task = asyncio.create_task(
            coordinator.pin_session("session-1", _pin(assignment))
        )
        await _wait_until(control.started)

        # The synchronous resolver is blocked, but the Agent event loop is not.
        heartbeat = False
        await asyncio.sleep(0)
        heartbeat = True
        assert heartbeat
        assert not pin_task.done()
        assert control.thread_names[0].startswith("areal-memory-agent")

        control.release.set()
        assert await pin_task is assignment
        assert (
            await coordinator.pin_session("session-1", _pin(assignment)) is assignment
        )
        with pytest.raises(MemoryAgentSessionConflictError, match="another assignment"):
            await coordinator.pin_session("session-1", _pin(replacement))
        # Every retry first resolves live authority; the cached pin is not
        # treated as a lease.  CAS then rejects the valid-but-different pin.
        assert control.calls == 3
        control.active = False
        with pytest.raises(MemoryReleaseAssignmentConflictError, match="revoked"):
            await coordinator.pin_session("session-1", _pin(assignment))
        assert control.calls == 4
    finally:
        control.release.set()
        await coordinator.aclose()


@_async_test
async def test_executor_bounds_submitted_calls_before_its_queue() -> None:
    assignment = _assignment()
    control = _ControlStore(assignment, block=True)
    coordinator = AsyncMemoryAgentCoordinator(
        control,
        _RuntimeStore(),
        max_workers=1,
        max_pending_calls=1,
    )
    try:
        first = asyncio.create_task(
            coordinator.pin_session("session-1", _pin(assignment))
        )
        await _wait_until(control.started)
        second = asyncio.create_task(
            coordinator.pin_session("session-2", _pin(assignment))
        )
        await asyncio.sleep(0.05)
        assert control.calls == 1
        assert control.max_active_calls == 1
        control.release.set()
        await asyncio.gather(first, second)
        assert control.calls == 2
        assert control.max_active_calls == 1
    finally:
        control.release.set()
        await coordinator.aclose()


@_async_test
async def test_cancelled_pin_holds_capacity_until_executor_callback_finishes() -> None:
    assignment = _assignment()
    control = _ControlStore(assignment, block=True)
    coordinator = AsyncMemoryAgentCoordinator(
        control,
        _RuntimeStore(),
        max_workers=1,
        max_pending_calls=1,
    )
    try:
        first = asyncio.create_task(
            coordinator.pin_session("session-1", _pin(assignment))
        )
        await _wait_until(control.started)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first

        second = asyncio.create_task(
            coordinator.pin_session("session-2", _pin(assignment))
        )
        await asyncio.sleep(0.05)
        # Cancellation did not release the semaphore while the Python callback
        # was still blocked in the dedicated executor.
        assert control.calls == 1

        shutdown = asyncio.create_task(coordinator.aclose())
        await asyncio.sleep(0)
        heartbeat = True
        assert heartbeat
        assert not shutdown.done()
        shutdown.cancel()
        with pytest.raises(asyncio.CancelledError):
            await shutdown
        second_shutdown = asyncio.create_task(coordinator.aclose())
        await asyncio.sleep(0)
        assert not second_shutdown.done()
        control.release.set()
        await second_shutdown
        with pytest.raises(MemoryAgentCoordinatorClosedError):
            await second
    finally:
        control.release.set()
        await coordinator.aclose()


@_async_test
async def test_close_session_allows_new_incarnation_and_invalidates_old_turn() -> None:
    assignment = _assignment()
    replacement = _assignment(
        suffix="replacement",
        incarnation=_hash("rollout-incarnation-2"),
    )
    runtime = _RuntimeStore()
    coordinator = AsyncMemoryAgentCoordinator(
        _ControlStore(assignment, replacement),
        runtime,
    )
    try:
        await coordinator.pin_session("session-1", _pin(assignment))
        old_turn = await coordinator.start_turn("session-1", "run-1")
        await coordinator.close_session("session-1")

        await coordinator.pin_session("session-1", _pin(replacement))
        new_turn = await coordinator.start_turn("session-1", "run-1")
        assert new_turn.memory_trajectory_id != old_turn.memory_trajectory_id
        with pytest.raises(MemoryAgentTurnConflictError, match="not issued"):
            await coordinator.expose_memory(old_turn, "query-1", query=b"old")
        exposure, _ = await coordinator.expose_memory(
            new_turn,
            "query-1",
            query=b"new",
        )
        assert exposure.assignment_id == replacement.assignment_id
    finally:
        await coordinator.aclose()


@_async_test
async def test_same_session_concurrent_turns_get_independent_trajectories() -> None:
    assignment = _assignment()
    runtime = _RuntimeStore()
    coordinator = AsyncMemoryAgentCoordinator(_ControlStore(assignment), runtime)
    main_thread_id = get_ident()
    try:
        await coordinator.pin_session("session-1", _pin(assignment))
        turn_a, turn_b = await asyncio.gather(
            coordinator.start_turn("session-1", "run-a"),
            coordinator.start_turn("session-1", "run-b"),
        )
        assert turn_a.memory_trajectory_id != turn_b.memory_trajectory_id
        assert turn_a.memory_trajectory_id not in ("session-1", "run-a")
        assert turn_b.memory_trajectory_id not in ("session-1", "run-b")

        (exposure_a, _), (exposure_b, _) = await asyncio.gather(
            coordinator.expose_memory(turn_a, "query-1", query=b"alpha"),
            coordinator.expose_memory(turn_b, "query-1", query=b"beta"),
        )
        assert exposure_a.trajectory_id == turn_a.memory_trajectory_id
        assert exposure_b.trajectory_id == turn_b.memory_trajectory_id
        assert {spec.query_sequence_no for spec in runtime.specs} == {0}
        assert all(call.thread_id != main_thread_id for call in runtime.calls)
    finally:
        await coordinator.aclose()


@_async_test
async def test_cancelled_caller_retry_reuses_task_and_consumer_call_id() -> None:
    assignment = _assignment()
    runtime = _RuntimeStore()
    runtime.block_submit = True
    coordinator = AsyncMemoryAgentCoordinator(_ControlStore(assignment), runtime)
    try:
        await coordinator.pin_session("session-1", _pin(assignment))
        turn = await coordinator.start_turn("session-1", "run-1")
        first = asyncio.create_task(
            coordinator.expose_memory(
                turn,
                "model-call-1",
                query=b"what is my timezone?",
                history=(b"hello",),
            )
        )
        await _wait_until(runtime.submit_started)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first

        retry = asyncio.create_task(
            coordinator.expose_memory(
                turn,
                "model-call-1",
                query=b"what is my timezone?",
                history=(b"hello",),
            )
        )
        await asyncio.sleep(0)
        assert runtime.consumer_side_effects == 1
        runtime.submit_release.set()
        exposure, output = await retry

        assert exposure.trajectory_id == turn.memory_trajectory_id
        assert output == f"output:{runtime.call_ids[0]}"
        assert runtime.consumer_side_effects == 1
        assert len(runtime.call_ids) == 1
        assert len(runtime.specs) == 1
    finally:
        runtime.submit_release.set()
        await coordinator.aclose()


@_async_test
async def test_operation_retry_conflict_and_forged_turn_fail_before_runtime() -> None:
    assignment = _assignment()
    runtime = _RuntimeStore()
    coordinator = AsyncMemoryAgentCoordinator(_ControlStore(assignment), runtime)
    try:
        await coordinator.pin_session("session-1", _pin(assignment))
        turn = await coordinator.start_turn("session-1", "run-1")
        await coordinator.expose_memory(turn, "query-1", query=b"same")

        with pytest.raises(MemoryAgentTurnConflictError, match="different query"):
            await coordinator.expose_memory(turn, "query-1", query=b"changed")
        forged = MemoryAgentTurnV1(
            session_key="session-1",
            turn_idempotency_key="run-1",
            memory_trajectory_id="mtraj_forged",
        )
        with pytest.raises(MemoryAgentTurnConflictError, match="not issued"):
            await coordinator.expose_memory(forged, "query-2", query=b"same")
        assert runtime.consumer_side_effects == 1
    finally:
        await coordinator.aclose()


@_async_test
async def test_raw_passthrough_without_runtime_exposure_is_rejected() -> None:
    assignment = _assignment()
    runtime = _RuntimeStore()
    runtime.invalid_consumer_result = True
    coordinator = AsyncMemoryAgentCoordinator(_ControlStore(assignment), runtime)
    try:
        await coordinator.pin_session("session-1", _pin(assignment))
        turn = await coordinator.start_turn("session-1", "raw-run")
        with pytest.raises(
            MemoryAgentTurnConflictError,
            match="canonical Memory exposure",
        ):
            await coordinator.expose_memory(
                turn,
                "raw-passthrough",
                query=b"forwarded bytes are not an acknowledgement",
            )
    finally:
        await coordinator.aclose()
