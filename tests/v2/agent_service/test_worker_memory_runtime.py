# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from areal.v2.agent_service.memory import (
    AsyncMemoryAgentCoordinator,
    MemoryAgentCoordinatorClosedError,
    MemoryAgentSessionConflictError,
)
from areal.v2.agent_service.memory_authorization import (
    MemoryPrincipalV1,
    MemoryScopeGrantAuthorizer,
    MemorySessionIncarnationV1,
    MemoryWorkerAudienceV1,
)
from areal.v2.agent_service.memory_broker import (
    AuthorizedMemoryAgentBroker,
    AuthorizedMemorySessionV1,
)
from areal.v2.agent_service.worker.memory_runtime import (
    AuthorizedMemoryWorkerRuntime,
    MemoryWorkerSessionReservationV1,
)


class _UnusedReleaseStore:
    def resolve_active_assignment(self, *args: object) -> object:
        raise AssertionError("reservation tests never resolve assignments")


class _UnusedRuntimeStore:
    def begin_query(self, *args: object) -> object:
        raise AssertionError("reservation tests never query Memory")

    resolve_query = begin_query
    prepare_delivery = begin_query
    submit_delivery = begin_query


class _Coordinator(AsyncMemoryAgentCoordinator):
    def __init__(self) -> None:
        super().__init__(
            _UnusedReleaseStore(),  # type: ignore[arg-type]
            _UnusedRuntimeStore(),  # type: ignore[arg-type]
            max_workers=1,
            max_pending_calls=1,
        )
        self.close_calls: list[str] = []
        self.close_started = asyncio.Event()
        self.close_release = asyncio.Event()
        self.close_release.set()
        self.close_failures = 0

    async def close_session(self, session_key: str) -> None:
        self.close_calls.append(session_key)
        self.close_started.set()
        await self.close_release.wait()
        if self.close_failures:
            self.close_failures -= 1
            raise RuntimeError("injected close failure")


def _principal(suffix: str) -> MemoryPrincipalV1:
    return MemoryPrincipalV1(
        issuer="https://identity.example",
        subject=f"principal-{suffix}",
    )


def _broker() -> tuple[
    AuthorizedMemoryAgentBroker,
    _Coordinator,
]:
    coordinator = _Coordinator()
    broker = AuthorizedMemoryAgentBroker(
        coordinator,
        MemoryScopeGrantAuthorizer(None),
    )
    return broker, coordinator


def _runtime() -> tuple[
    AuthorizedMemoryWorkerRuntime,
    AuthorizedMemoryAgentBroker,
    _Coordinator,
]:
    broker, coordinator = _broker()
    return AuthorizedMemoryWorkerRuntime(broker), broker, coordinator


@pytest.mark.asyncio
async def test_one_broker_cannot_be_owned_by_two_worker_runtimes() -> None:
    runtime, broker, _ = _runtime()
    with pytest.raises(MemoryAgentSessionConflictError, match="already belongs"):
        AuthorizedMemoryWorkerRuntime(broker)
    await runtime.aclose()


def _claim_runtime_concurrently(
    barrier: threading.Barrier,
    broker: AuthorizedMemoryAgentBroker,
) -> AuthorizedMemoryWorkerRuntime | BaseException:
    try:
        barrier.wait(timeout=5)
        return AuthorizedMemoryWorkerRuntime(broker)
    except BaseException as error:
        return error


@pytest.mark.asyncio
async def test_concurrent_host_threads_have_exactly_one_broker_owner() -> None:
    contenders = 24
    for _ in range(4):
        broker, _ = _broker()
        barrier = threading.Barrier(contenders + 1)
        with ThreadPoolExecutor(max_workers=contenders) as executor:
            futures = [
                executor.submit(_claim_runtime_concurrently, barrier, broker)
                for _ in range(contenders)
            ]
            barrier.wait(timeout=5)
            results = [future.result(timeout=5) for future in futures]

        winners = [
            result
            for result in results
            if type(result) is AuthorizedMemoryWorkerRuntime
        ]
        conflicts = [
            result
            for result in results
            if isinstance(result, MemoryAgentSessionConflictError)
        ]
        assert len(winners) == 1
        assert len(conflicts) == contenders - 1
        await winners[0].aclose()


@pytest.mark.asyncio
async def test_closed_broker_cannot_be_transferred_to_a_runtime() -> None:
    coordinator = _Coordinator()
    broker = AuthorizedMemoryAgentBroker(
        coordinator,
        MemoryScopeGrantAuthorizer(None),
    )
    await broker.aclose()
    with pytest.raises(MemoryAgentCoordinatorClosedError, match="broker is closed"):
        AuthorizedMemoryWorkerRuntime(broker)


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("principal", "outside"),
        ("session", "outside"),
        ("audience", "outside"),
        ("malformed", "malformed"),
    ],
)
@pytest.mark.asyncio
async def test_broker_identity_mismatch_is_not_published(
    fault: str,
    message: str,
) -> None:
    runtime, broker, _ = _runtime()
    original_open = broker.open_session

    async def mismatched_open(
        principal: MemoryPrincipalV1,
        session_key: str,
    ) -> object:
        issued = await original_open(principal, session_key)
        if fault == "principal":
            return AuthorizedMemorySessionV1(
                principal=_principal("wrong"),
                session=issued.session,
                audience=issued.audience,
            )
        if fault == "session":
            return AuthorizedMemorySessionV1(
                principal=issued.principal,
                session=MemorySessionIncarnationV1(
                    session_key="wrong-session",
                    incarnation_id=issued.session.incarnation_id,
                ),
                audience=issued.audience,
            )
        if fault == "audience":
            return AuthorizedMemorySessionV1(
                principal=issued.principal,
                session=issued.session,
                audience=MemoryWorkerAudienceV1.create(),
            )
        return object()

    broker.open_session = mismatched_open  # type: ignore[method-assign]
    with pytest.raises(MemoryAgentSessionConflictError, match=message):
        await runtime.reserve_session(_principal("one"), "session-1")
    sessions = getattr(runtime, "_AuthorizedMemoryWorkerRuntime__sessions")
    pending = getattr(runtime, "_AuthorizedMemoryWorkerRuntime__pending")
    assert sessions == {}
    assert pending == {}
    await runtime.aclose()


@pytest.mark.asyncio
async def test_concurrent_principals_have_one_owner_and_same_owner_is_idempotent() -> (
    None
):
    runtime, _, _ = _runtime()
    principals = (_principal("one"), _principal("two"))
    results = await asyncio.gather(
        *(
            runtime.reserve_session(principal, "shared-session")
            for principal in principals
        ),
        return_exceptions=True,
    )

    winners = [
        (index, result)
        for index, result in enumerate(results)
        if type(result) is MemoryWorkerSessionReservationV1
    ]
    failures = [
        result
        for result in results
        if isinstance(result, MemoryAgentSessionConflictError)
    ]
    assert len(winners) == 1
    assert len(failures) == 1

    winner_index, reservation = winners[0]
    again = await runtime.reserve_session(
        principals[winner_index],
        "shared-session",
    )
    assert again is reservation
    assert reservation.session_key == reservation.session.session_key
    assert reservation.audience == runtime.audience
    assert set(reservation.__dataclass_fields__) == {  # type: ignore[attr-defined]
        "session_key",
        "session",
        "audience",
    }

    same_principal = _principal("same")
    same_one, same_two = await asyncio.gather(
        runtime.reserve_session(same_principal, "idempotent-session"),
        runtime.reserve_session(same_principal, "idempotent-session"),
    )
    assert same_one is same_two
    await runtime.aclose()


@pytest.mark.asyncio
async def test_close_reopen_rejects_stale_and_value_equal_descriptors() -> None:
    runtime, _, _ = _runtime()
    principal = _principal("one")
    original = await runtime.reserve_session(principal, "reused-session")
    forged = MemoryWorkerSessionReservationV1(
        session_key=original.session_key,
        session=original.session,
        audience=original.audience,
    )
    assert forged == original
    assert forged is not original
    with pytest.raises(MemoryAgentSessionConflictError, match="not current"):
        await runtime.close_session(forged)

    await runtime.close_session(original)
    replacement = await runtime.reserve_session(principal, "reused-session")
    assert replacement.session.incarnation_id != original.session.incarnation_id
    with pytest.raises(MemoryAgentSessionConflictError, match="not current"):
        await runtime.close_session(original)
    await runtime.aclose()


@pytest.mark.asyncio
async def test_new_runtime_rejects_an_old_runtime_audience_and_descriptor() -> None:
    first, _, _ = _runtime()
    old = await first.reserve_session(_principal("one"), "session-1")
    await first.aclose()

    second, _, _ = _runtime()
    current = await second.reserve_session(_principal("one"), "session-1")
    assert current.audience != old.audience
    with pytest.raises(MemoryAgentSessionConflictError, match="not current"):
        await second.close_session(old)
    await second.aclose()


async def _wait_for_broker_binding(
    broker: AuthorizedMemoryAgentBroker,
    session_key: str,
) -> None:
    sessions = getattr(broker, "_AuthorizedMemoryAgentBroker__sessions")
    while session_key not in sessions:
        await asyncio.sleep(0)


async def _wait_for_pending_reservation(
    runtime: AuthorizedMemoryWorkerRuntime,
    session_key: str,
) -> None:
    pending = getattr(runtime, "_AuthorizedMemoryWorkerRuntime__pending")
    while session_key not in pending:
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_cancel_after_broker_binding_cannot_rebind_another_principal() -> None:
    runtime, broker, _ = _runtime()
    runtime_lock = getattr(runtime, "_AuthorizedMemoryWorkerRuntime__state_lock")
    runtime_sessions = getattr(runtime, "_AuthorizedMemoryWorkerRuntime__sessions")
    broker_lock = getattr(broker, "_AuthorizedMemoryAgentBroker__state_lock")
    await broker_lock.acquire()
    pending = asyncio.create_task(
        runtime.reserve_session(_principal("one"), "cancelled-session")
    )
    try:
        await asyncio.wait_for(
            _wait_for_pending_reservation(runtime, "cancelled-session"),
            timeout=1,
        )
        await runtime_lock.acquire()
        broker_lock.release()
        await asyncio.wait_for(
            _wait_for_broker_binding(broker, "cancelled-session"),
            timeout=1,
        )
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert runtime_sessions == {}
    finally:
        if broker_lock.locked():
            broker_lock.release()
        if runtime_lock.locked():
            runtime_lock.release()

    with pytest.raises(MemoryAgentSessionConflictError, match="another principal"):
        await runtime.reserve_session(_principal("two"), "cancelled-session")
    recovered = await runtime.reserve_session(
        _principal("one"),
        "cancelled-session",
    )
    assert recovered.session_key == "cancelled-session"
    await runtime.aclose()


@pytest.mark.asyncio
async def test_failed_or_cancelled_close_does_not_release_ownership_early() -> None:
    runtime, _, coordinator = _runtime()
    original = await runtime.reserve_session(_principal("one"), "session-1")

    coordinator.close_failures = 1
    with pytest.raises(RuntimeError, match="injected close failure"):
        await runtime.close_session(original)
    with pytest.raises(MemoryAgentSessionConflictError, match="closing"):
        await runtime.reserve_session(_principal("two"), "session-1")

    coordinator.close_release.clear()
    coordinator.close_started.clear()
    closing = asyncio.create_task(runtime.close_session(original))
    await asyncio.wait_for(coordinator.close_started.wait(), timeout=1)
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing
    with pytest.raises(MemoryAgentSessionConflictError, match="closing"):
        await runtime.reserve_session(_principal("two"), "session-1")

    coordinator.close_release.set()
    await runtime.close_session(original)
    replacement = await runtime.reserve_session(_principal("two"), "session-1")
    assert replacement.session.incarnation_id != original.session.incarnation_id
    assert coordinator.close_calls == ["session-1", "session-1"]
    await runtime.aclose()


@pytest.mark.asyncio
async def test_cancelled_shutdown_is_rejoined_and_drains_owned_broker() -> None:
    runtime, _, coordinator = _runtime()
    await runtime.reserve_session(_principal("one"), "session-1")
    coordinator.close_release.clear()

    first = asyncio.create_task(runtime.aclose())
    await asyncio.wait_for(coordinator.close_started.wait(), timeout=1)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    with pytest.raises(MemoryAgentCoordinatorClosedError, match="runtime is closed"):
        await runtime.reserve_session(_principal("one"), "session-1")

    coordinator.close_release.set()
    await runtime.aclose()
    sessions = getattr(runtime, "_AuthorizedMemoryWorkerRuntime__sessions")
    pending = getattr(runtime, "_AuthorizedMemoryWorkerRuntime__pending")
    assert sessions == {}
    assert pending == {}


@pytest.mark.asyncio
async def test_shutdown_drains_pending_reservation_without_late_publication() -> None:
    runtime, broker, _ = _runtime()
    broker_lock = getattr(broker, "_AuthorizedMemoryAgentBroker__state_lock")
    pending_states = getattr(runtime, "_AuthorizedMemoryWorkerRuntime__pending")
    sessions = getattr(runtime, "_AuthorizedMemoryWorkerRuntime__sessions")
    await broker_lock.acquire()
    caller = asyncio.create_task(
        runtime.reserve_session(_principal("one"), "pending-session")
    )
    try:
        await asyncio.wait_for(
            _wait_for_pending_reservation(runtime, "pending-session"),
            timeout=1,
        )
        owned_task = pending_states["pending-session"].task
        shutdown = asyncio.create_task(runtime.aclose())
        while not getattr(runtime, "_AuthorizedMemoryWorkerRuntime__closed"):
            await asyncio.sleep(0)
    finally:
        broker_lock.release()

    with pytest.raises(MemoryAgentCoordinatorClosedError):
        await caller
    await shutdown
    assert owned_task.done()
    assert sessions == {}
    assert pending_states == {}
