# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from hashlib import sha256

import pytest

from areal.v2.agent_service.memory import (
    AsyncMemoryAgentCoordinator,
    MemoryAgentCoordinatorClosedError,
    MemoryAgentSessionConflictError,
    MemoryAgentSessionPinV1,
    MemoryAgentTurnConflictError,
    MemoryAgentTurnV1,
)
from areal.v2.agent_service.memory_authorization import (
    MemoryPrincipalV1,
    MemoryScopeActionV1,
    MemoryScopeAuthorizationDeniedError,
    MemoryScopeGrantAuthorizer,
    MemoryScopeGrantRequestV1,
    MemoryScopeGrantV1,
    MemorySessionIncarnationV1,
    MemoryWorkerAudienceV1,
)
from areal.v2.agent_service.memory_broker import (
    AuthorizedMemoryAgentBroker,
    AuthorizedMemorySessionV1,
)
from areal.v2.agent_service.memory_session_lifecycle import (
    MemoryWorkerSessionCloseOutcomeV1,
    MemoryWorkerSessionCloseReceiptV1,
    MemoryWorkerSessionIdentityV1,
)
from areal.v2.agent_service.types import (
    AgentRequest,
    AgentResponse,
    EventEmitter,
    MemoryTurnResultV1,
    StreamResponse,
)
from areal.v2.agent_service.worker.memory_runtime import (
    AuthorizedMemoryWorkerRuntime,
    MemoryWorkerSessionReservationV1,
    MemoryWorkerTurnLease,
)
from areal.v2.memory_service import runtime_types
from areal.v2.memory_service.runtime_types import (
    MemoryExposureStatus,
    MemoryExposureV1,
)
from areal.v2.memory_service.types import MemoryScope

_NOW = datetime(2026, 7, 14, 10, 0, tzinfo=UTC)
_RESOLVER_VERSION = sha256(b"worker-runtime-resolver-v1").hexdigest()
_RESOLVER_CONFIG = sha256(b"worker-runtime-resolver-config-v1").hexdigest()


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


def _identity(
    reservation: MemoryWorkerSessionReservationV1,
    *,
    session_key: str | None = None,
    incarnation_id: str | None = None,
    audience_id: str | None = None,
) -> MemoryWorkerSessionIdentityV1:
    return MemoryWorkerSessionIdentityV1(
        session=MemorySessionIncarnationV1(
            session_key=(
                reservation.session_key if session_key is None else session_key
            ),
            incarnation_id=(
                reservation.session.incarnation_id
                if incarnation_id is None
                else incarnation_id
            ),
        ),
        audience=MemoryWorkerAudienceV1(
            reservation.audience.audience_id if audience_id is None else audience_id
        ),
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


def _runtime_with_retirement_capacity(
    capacity: int,
) -> tuple[
    AuthorizedMemoryWorkerRuntime,
    AuthorizedMemoryAgentBroker,
    _Coordinator,
]:
    broker, coordinator = _broker()
    return (
        AuthorizedMemoryWorkerRuntime(
            broker,
            max_retired_sessions=capacity,
        ),
        broker,
        coordinator,
    )


def _hash(label: str) -> str:
    return sha256(label.encode()).hexdigest()


def _pin(suffix: str = "1") -> MemoryAgentSessionPinV1:
    assignment_hash = _hash(f"assignment-{suffix}")
    return MemoryAgentSessionPinV1(
        scope=MemoryScope(
            tenant_id=f"tenant-{suffix}",
            namespace="agent-long-term-memory",
            subject_id=f"subject-{suffix}",
        ),
        rollout_group_id=f"rollout-group-{suffix}",
        rollout_group_incarnation_sha256=_hash(f"incarnation-{suffix}"),
        assignment_id=f"masn_{assignment_hash[:24]}",
        assignment_content_sha256=assignment_hash,
    )


def _exposure(
    turn: MemoryAgentTurnV1,
    pin: MemoryAgentSessionPinV1,
) -> MemoryExposureV1:
    values = {
        "scope": pin.scope,
        "assignment_id": pin.assignment_id,
        "assignment_content_sha256": pin.assignment_content_sha256,
        "release_id": f"rel_{_hash('release')[:24]}",
        "release_content_sha256": _hash("release"),
        "trajectory_id": turn.memory_trajectory_id,
        "rollout_group_id": pin.rollout_group_id,
        "rollout_group_incarnation_sha256": (pin.rollout_group_incarnation_sha256),
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


class _BindingResolver:
    resolver_id = "test-worker-runtime-grant-policy"
    resolver_version_sha256 = _RESOLVER_VERSION
    resolver_config_sha256 = _RESOLVER_CONFIG

    def __init__(self) -> None:
        self.active_actions = {
            MemoryScopeActionV1.PIN_ASSIGNMENT,
            MemoryScopeActionV1.EXPOSE_MEMORY,
        }
        self.calls: list[MemoryScopeGrantRequestV1] = []
        self.block_action: MemoryScopeActionV1 | None = None
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()

    def resolve_active_grant(
        self,
        request: MemoryScopeGrantRequestV1,
    ) -> MemoryScopeGrantV1:
        self.calls.append(request)
        if request.action is self.block_action:
            self.started.set()
            self.release.wait(timeout=5)
        if request.action not in self.active_actions:
            raise MemoryScopeAuthorizationDeniedError(
                "active Memory scope grant is unavailable"
            )
        return MemoryScopeGrantV1.create(
            request=request,
            resolver_id=self.resolver_id,
            resolver_version_sha256=self.resolver_version_sha256,
            resolver_config_sha256=self.resolver_config_sha256,
            valid_from=_NOW - timedelta(minutes=1),
            valid_until=_NOW + timedelta(minutes=1),
            evaluated_at=_NOW - timedelta(minutes=2),
            granted_at=_NOW - timedelta(minutes=2),
            idempotency_key=f"grant-{len(self.calls)}",
        )


class _BindingCoordinator(AsyncMemoryAgentCoordinator):
    def __init__(self, pin: MemoryAgentSessionPinV1 | None = None) -> None:
        super().__init__(
            _UnusedReleaseStore(),  # type: ignore[arg-type]
            _UnusedRuntimeStore(),  # type: ignore[arg-type]
            max_workers=1,
            max_pending_calls=1,
        )
        self.pin = pin or _pin()
        self.pin_calls: list[tuple[str, MemoryAgentSessionPinV1]] = []
        self.start_calls: list[tuple[str, str]] = []
        self.expose_calls: list[tuple[object, ...]] = []
        self.close_calls: list[str] = []
        self.pin_started = asyncio.Event()
        self.pin_release = asyncio.Event()
        self.pin_release.set()
        self.start_started = asyncio.Event()
        self.start_release = asyncio.Event()
        self.start_release.set()
        self.expose_started = asyncio.Event()
        self.expose_release = asyncio.Event()
        self.expose_release.set()
        self.close_started = asyncio.Event()
        self.close_release = asyncio.Event()
        self.close_release.set()
        self.pin_failures = 0
        self.start_failures = 0

    async def pin_session(  # type: ignore[override]
        self,
        session_key: str,
        pin: MemoryAgentSessionPinV1,
    ) -> object:
        self.pin_calls.append((session_key, pin))
        self.pin_started.set()
        await self.pin_release.wait()
        if self.pin_failures:
            self.pin_failures -= 1
            raise RuntimeError("injected pin failure")
        self.pin = pin
        return {"assignment": pin.assignment_id}

    async def start_turn(
        self,
        session_key: str,
        turn_idempotency_key: str,
    ) -> MemoryAgentTurnV1:
        self.start_calls.append((session_key, turn_idempotency_key))
        self.start_started.set()
        await self.start_release.wait()
        if self.start_failures:
            self.start_failures -= 1
            raise RuntimeError("injected start failure")
        trajectory_hash = _hash(f"{session_key}:{turn_idempotency_key}")
        return MemoryAgentTurnV1(
            session_key=session_key,
            turn_idempotency_key=turn_idempotency_key,
            memory_trajectory_id=f"mtraj_{trajectory_hash}",
        )

    async def expose_memory(
        self,
        turn: MemoryAgentTurnV1,
        operation_key: str,
        *,
        query: bytes,
        history: tuple[bytes, ...] = (),
    ) -> tuple[MemoryExposureV1, object]:
        self.expose_calls.append((turn, operation_key, query, history))
        self.expose_started.set()
        await self.expose_release.wait()
        return _exposure(turn, self.pin), {"answer": "from-memory"}

    async def close_session(self, session_key: str) -> None:
        self.close_calls.append(session_key)
        self.close_started.set()
        await self.close_release.wait()


def _binding_runtime() -> tuple[
    AuthorizedMemoryWorkerRuntime,
    AuthorizedMemoryAgentBroker,
    _BindingCoordinator,
    _BindingResolver,
]:
    coordinator = _BindingCoordinator()
    resolver = _BindingResolver()
    broker = AuthorizedMemoryAgentBroker(
        coordinator,
        MemoryScopeGrantAuthorizer(resolver, clock=lambda: _NOW),
    )
    return AuthorizedMemoryWorkerRuntime(broker), broker, coordinator, resolver


def test_session_lifecycle_values_are_strict_detached_descriptions() -> None:
    source_session = MemorySessionIncarnationV1(
        session_key="session-1",
        incarnation_id=f"msinc_{'1' * 64}",
    )
    source_audience = MemoryWorkerAudienceV1(f"maud_{'2' * 64}")
    identity = MemoryWorkerSessionIdentityV1(
        session=source_session,
        audience=source_audience,
    )
    receipt = MemoryWorkerSessionCloseReceiptV1(
        identity=identity,
        outcome=MemoryWorkerSessionCloseOutcomeV1.CLOSED,
    )

    assert identity.session_key == "session-1"
    assert identity.session is not source_session
    assert identity.audience is not source_audience
    assert receipt.identity == identity
    assert receipt.identity is not identity
    assert set(receipt.__dataclass_fields__) == {  # type: ignore[attr-defined]
        "identity",
        "outcome",
    }

    with pytest.raises(TypeError, match="identity"):
        MemoryWorkerSessionCloseReceiptV1(  # type: ignore[arg-type]
            identity=object(),
            outcome=MemoryWorkerSessionCloseOutcomeV1.CLOSED,
        )
    with pytest.raises(TypeError, match="outcome"):
        MemoryWorkerSessionCloseReceiptV1(  # type: ignore[arg-type]
            identity=identity,
            outcome="closed",
        )


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
    runtime, _, coordinator = _runtime()
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

    closed = await runtime.close_session(original)
    assert closed == MemoryWorkerSessionCloseReceiptV1(
        identity=original.identity,
        outcome=MemoryWorkerSessionCloseOutcomeV1.CLOSED,
    )
    replacement = await runtime.reserve_session(principal, "reused-session")
    assert replacement.session.incarnation_id != original.session.incarnation_id

    # A response-loss retry is served from A's retirement record.  It neither
    # re-runs cleanup nor resolves the reusable key to the current B.
    replayed = await runtime.close_session(original)
    assert replayed == closed
    assert replayed is not closed
    assert replayed.identity is not closed.identity
    assert await runtime.reserve_session(principal, "reused-session") is replacement
    assert coordinator.close_calls == ["reused-session"]

    # Equal data remains descriptive rather than acquiring local handle power,
    # even after the genuine descriptor has become replayable.
    with pytest.raises(MemoryAgentSessionConflictError, match="not current"):
        await runtime.close_session(forged)
    await runtime.aclose()


@pytest.mark.asyncio
async def test_retired_receipt_uses_private_snapshot_not_public_descriptor_alias() -> (
    None
):
    runtime, _, coordinator = _runtime()
    reservation = await runtime.reserve_session(_principal("one"), "session-1")
    identity = reservation.identity
    closed = await runtime.close_session(reservation)
    object.__setattr__(
        reservation.session,
        "incarnation_id",
        f"msinc_{'4' * 64}",
    )

    scalar_replay = await runtime.close_session_if_current(identity)
    object_replay = await runtime.close_session(reservation)

    assert scalar_replay == object_replay == closed
    assert scalar_replay.identity == identity
    assert coordinator.close_calls == ["session-1"]
    await runtime.aclose()


@pytest.mark.asyncio
async def test_conditional_close_replays_a_without_touching_same_key_b() -> None:
    runtime, _, coordinator, _ = _binding_runtime()
    principal = _principal("one")
    original = await runtime.reserve_session(principal, "reused-session")
    original_identity = original.identity
    coordinator.close_release.clear()

    # Model a lost HTTP response: the caller disappears after the Worker has
    # admitted A's destructive operation, while the shielded owned task keeps
    # running to its retirement linearization point.
    lost_waiter = asyncio.create_task(
        runtime.close_session_if_current(original_identity)
    )
    await asyncio.wait_for(coordinator.close_started.wait(), timeout=1)
    lost_waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await lost_waiter
    coordinator.close_release.set()
    await asyncio.wait_for(
        _wait_for_runtime_session_absent(runtime, "reused-session"),
        timeout=1,
    )

    replacement = await runtime.reserve_session(principal, "reused-session")
    assert replacement.identity != original_identity

    delayed_retry = await runtime.close_session_if_current(original_identity)

    assert delayed_retry.outcome is MemoryWorkerSessionCloseOutcomeV1.CLOSED
    assert delayed_retry.identity == original_identity
    assert await runtime.reserve_session(principal, "reused-session") is replacement
    assert coordinator.close_calls == ["reused-session"]
    request = AgentRequest("B remains usable", "reused-session", "run-b")
    lease = await runtime.bind_turn(replacement, request, assignment_pin=_pin("b"))
    assert request.memory is not None
    await lease.aclose()
    await runtime.aclose()


@pytest.mark.asyncio
async def test_object_and_value_close_share_one_owned_destructive_task() -> None:
    runtime, _, coordinator = _runtime()
    reservation = await runtime.reserve_session(_principal("one"), "session-1")
    coordinator.close_release.clear()
    second_admission = asyncio.Event()
    start_close_task = runtime._start_close_task
    admissions = 0

    def observed_start_close_task(state: object):
        nonlocal admissions
        admissions += 1
        if admissions == 2:
            second_admission.set()
        return start_close_task(state)  # type: ignore[arg-type]

    runtime._start_close_task = observed_start_close_task  # type: ignore[method-assign]

    object_close = asyncio.create_task(runtime.close_session(reservation))
    await asyncio.wait_for(coordinator.close_started.wait(), timeout=1)
    value_close = asyncio.create_task(
        runtime.close_session_if_current(reservation.identity)
    )
    await asyncio.wait_for(second_admission.wait(), timeout=1)

    assert coordinator.close_calls == ["session-1"]
    coordinator.close_release.set()
    object_receipt, value_receipt = await asyncio.gather(object_close, value_close)

    assert object_receipt == value_receipt
    assert object_receipt.outcome is MemoryWorkerSessionCloseOutcomeV1.CLOSED
    assert coordinator.close_calls == ["session-1"]
    await runtime.aclose()


@pytest.mark.parametrize(
    "identity_mutation",
    (
        {"session_key": "another-session"},
        {"incarnation_id": f"msinc_{'1' * 64}"},
        {"audience_id": f"maud_{'2' * 64}"},
    ),
)
@pytest.mark.asyncio
async def test_conditional_close_mismatch_is_a_side_effect_free_receipt(
    identity_mutation: dict[str, str],
) -> None:
    runtime, _, coordinator = _runtime()
    principal = _principal("one")
    current = await runtime.reserve_session(principal, "session-1")
    requested = _identity(current, **identity_mutation)

    receipt = await runtime.close_session_if_current(requested)

    assert receipt == MemoryWorkerSessionCloseReceiptV1(
        identity=requested,
        outcome=MemoryWorkerSessionCloseOutcomeV1.NOT_CURRENT,
    )
    assert receipt.identity == requested
    assert receipt.identity != current.identity
    assert coordinator.close_calls == []
    assert await runtime.reserve_session(principal, "session-1") is current
    await runtime.aclose()


@pytest.mark.asyncio
async def test_retirement_cache_eviction_remains_safe_for_a_same_key_successor() -> (
    None
):
    runtime, _, coordinator = _runtime_with_retirement_capacity(1)
    principal = _principal("one")
    original = await runtime.reserve_session(principal, "session-1")
    await runtime.close_session(original)
    evicting = await runtime.reserve_session(principal, "session-2")
    await runtime.close_session(evicting)

    replacement = await runtime.reserve_session(principal, "session-1")
    evicted_retry = await runtime.close_session_if_current(original.identity)

    assert evicted_retry == MemoryWorkerSessionCloseReceiptV1(
        identity=original.identity,
        outcome=MemoryWorkerSessionCloseOutcomeV1.NOT_CURRENT,
    )
    with pytest.raises(MemoryAgentSessionConflictError, match="not current"):
        await runtime.close_session(original)
    assert await runtime.reserve_session(principal, "session-1") is replacement
    assert coordinator.close_calls == ["session-1", "session-2"]
    retired_identities = getattr(
        runtime,
        "_AuthorizedMemoryWorkerRuntime__retired_by_identity",
    )
    retired_descriptors = getattr(
        runtime,
        "_AuthorizedMemoryWorkerRuntime__retired_by_descriptor",
    )
    assert len(retired_identities) == len(retired_descriptors) == 1
    await runtime.aclose()
    assert retired_identities == retired_descriptors == {}


@pytest.mark.asyncio
async def test_retained_incarnation_collision_fails_before_successor_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, broker, coordinator = _runtime_with_retirement_capacity(1)
    principal = _principal("one")
    original = await runtime.reserve_session(principal, "session-1")
    await runtime.close_session(original)

    def colliding_incarnation(
        cls: type[MemorySessionIncarnationV1],
        session_key: str,
    ) -> MemorySessionIncarnationV1:
        del cls
        return MemorySessionIncarnationV1(
            session_key=session_key,
            incarnation_id=original.session.incarnation_id,
        )

    monkeypatch.setattr(
        MemorySessionIncarnationV1,
        "create",
        classmethod(colliding_incarnation),
    )
    with pytest.raises(MemoryAgentSessionConflictError, match="reused"):
        await runtime.reserve_session(principal, "session-1")

    sessions = getattr(runtime, "_AuthorizedMemoryWorkerRuntime__sessions")
    assert sessions == {}
    assert coordinator.close_calls == ["session-1"]
    retired = await runtime.close_session_if_current(original.identity)
    assert retired.outcome is MemoryWorkerSessionCloseOutcomeV1.CLOSED

    # Evict A's ordinary replay record.  Collision knowledge is a stronger
    # invariant than receipt caching and must keep the key quarantined.
    other = await runtime.reserve_session(principal, "session-2")
    await runtime.close_session(other)
    evicted = await runtime.close_session_if_current(original.identity)
    assert evicted.outcome is MemoryWorkerSessionCloseOutcomeV1.NOT_CURRENT
    with pytest.raises(MemoryAgentSessionConflictError, match="quarantined"):
        await runtime.reserve_session(principal, "session-1")
    assert coordinator.close_calls == ["session-1", "session-2"]

    await runtime.aclose()
    # The unpublishable private broker binding is cleaned only at shutdown; no
    # Agent turn or runtime descriptor could observe it in between.
    assert coordinator.close_calls == ["session-1", "session-2", "session-1"]
    assert getattr(broker, "_AuthorizedMemoryAgentBroker__sessions") == {}
    assert (
        getattr(
            runtime,
            "_AuthorizedMemoryWorkerRuntime__quarantined_session_keys",
        )
        == set()
    )


@pytest.mark.asyncio
async def test_conditional_close_fails_closed_on_corrupted_private_state() -> None:
    runtime, _, coordinator = _runtime()
    current = await runtime.reserve_session(_principal("one"), "session-1")
    requested = current.identity
    object.__setattr__(
        current.session,
        "incarnation_id",
        f"msinc_{'3' * 64}",
    )

    with pytest.raises(MemoryAgentSessionConflictError, match="disagrees"):
        await runtime.close_session_if_current(requested)

    assert coordinator.close_calls == []
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
    conditional = await second.close_session_if_current(old.identity)
    assert conditional == MemoryWorkerSessionCloseReceiptV1(
        identity=old.identity,
        outcome=MemoryWorkerSessionCloseOutcomeV1.NOT_CURRENT,
    )
    assert await second.reserve_session(_principal("one"), "session-1") is current
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


async def _wait_for_runtime_session_absent(
    runtime: AuthorizedMemoryWorkerRuntime,
    session_key: str,
) -> None:
    sessions = getattr(runtime, "_AuthorizedMemoryWorkerRuntime__sessions")
    while session_key in sessions:
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
    closed = await runtime.close_session(original)
    assert closed.outcome is MemoryWorkerSessionCloseOutcomeV1.CLOSED
    replacement = await runtime.reserve_session(_principal("two"), "session-1")
    assert replacement.session.incarnation_id != original.session.incarnation_id
    assert await runtime.close_session(original) == closed
    assert await runtime.reserve_session(_principal("two"), "session-1") is replacement
    assert coordinator.close_calls == ["session-1", "session-1"]
    await runtime.aclose()


@pytest.mark.asyncio
async def test_shutdown_rejoins_admitted_close_until_receipt_publication() -> None:
    runtime, broker, coordinator = _runtime()
    reservation = await runtime.reserve_session(_principal("one"), "session-1")
    coordinator.close_release.clear()
    broker_cleanup_returned = asyncio.Event()
    allow_runtime_publication = asyncio.Event()
    close_broker_session = broker.close_session

    async def gated_broker_close(session: object) -> None:
        await close_broker_session(session)  # type: ignore[arg-type]
        broker_cleanup_returned.set()
        await allow_runtime_publication.wait()

    broker.close_session = gated_broker_close  # type: ignore[method-assign]
    closing = asyncio.create_task(runtime.close_session(reservation))
    await asyncio.wait_for(coordinator.close_started.wait(), timeout=1)
    shutdown = asyncio.create_task(runtime.aclose())

    async def wait_for_shutdown_admission() -> None:
        while (
            getattr(
                runtime,
                "_AuthorizedMemoryWorkerRuntime__shutdown_task",
            )
            is None
        ):
            await asyncio.sleep(0)

    await asyncio.wait_for(wait_for_shutdown_admission(), timeout=1)
    coordinator.close_release.set()
    await asyncio.wait_for(broker_cleanup_returned.wait(), timeout=1)

    async def wait_for_broker_shutdown() -> None:
        while True:
            task = getattr(broker, "_AuthorizedMemoryAgentBroker__shutdown_task")
            if task is not None and task.done():
                await task
                return
            await asyncio.sleep(0)

    await asyncio.wait_for(wait_for_broker_shutdown(), timeout=1)

    # Broker cleanup has succeeded, but runtime close is deliberately paused
    # before publishing A's retirement.  Shutdown must not clear A underneath
    # that already-admitted task.
    assert not shutdown.done()
    sessions = getattr(runtime, "_AuthorizedMemoryWorkerRuntime__sessions")
    assert sessions["session-1"].closing

    allow_runtime_publication.set()
    receipt = await closing
    await shutdown

    assert receipt.outcome is MemoryWorkerSessionCloseOutcomeV1.CLOSED
    assert coordinator.close_calls == ["session-1"]
    assert sessions == {}
    assert (
        getattr(
            runtime,
            "_AuthorizedMemoryWorkerRuntime__retired_by_identity",
        )
        == {}
    )


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


@pytest.mark.asyncio
async def test_bind_turn_uses_only_explicit_pin_and_broker_owned_identity() -> None:
    runtime, _, coordinator, resolver = _binding_runtime()
    principal = _principal("one")
    reservation = await runtime.reserve_session(principal, "session-1")
    pin = _pin()
    request = AgentRequest(
        message="hello",
        session_key="session-1",
        run_id="run-1",
        metadata={"areal_memory": _pin("forged-metadata")},
    )

    lease = await runtime.bind_turn(
        reservation,
        request,
        assignment_pin=pin,
    )
    assert type(lease) is MemoryWorkerTurnLease
    assert not hasattr(lease, "broker_session")
    assert not hasattr(lease, "authorized_turn")
    assert not hasattr(lease, "__dict__")
    assert request.memory is not None
    result = await request.memory.expose_memory(
        "lookup",
        query=b"future question",
        history=(b"prior turn",),
    )

    assert type(result) is MemoryTurnResultV1
    assert result.output == {"answer": "from-memory"}
    assert [call.action for call in resolver.calls] == [
        MemoryScopeActionV1.PIN_ASSIGNMENT,
        MemoryScopeActionV1.EXPOSE_MEMORY,
    ]
    for grant_request in resolver.calls:
        assert grant_request.principal == principal
        assert grant_request.session == reservation.session
        assert grant_request.audience == reservation.audience == runtime.audience
        assert grant_request.target.scope == pin.scope
        assert grant_request.target.rollout_group_id == pin.rollout_group_id
        assert (
            grant_request.target.rollout_group_incarnation_sha256
            == pin.rollout_group_incarnation_sha256
        )
        assert grant_request.target.assignment_id == pin.assignment_id
        assert (
            grant_request.target.assignment_content_sha256
            == pin.assignment_content_sha256
        )
    assert coordinator.pin_calls == [("session-1", pin)]
    assert coordinator.start_calls == [("session-1", "run-1")]
    assert coordinator.expose_calls[0][1:] == (
        "lookup",
        b"future question",
        (b"prior turn",),
    )
    await lease.aclose()
    await runtime.aclose()


class _DiscardEmitter:
    async def emit_delta(self, text: str) -> None:
        del text

    async def emit_tool_call(self, name: str, args: str) -> None:
        del name, args

    async def emit_tool_result(self, name: str, result: str) -> None:
        del name, result


async def _bound_response_lease(
    run_id: str,
) -> tuple[
    AuthorizedMemoryWorkerRuntime,
    AuthorizedMemoryAgentBroker,
    _BindingCoordinator,
    AgentRequest,
    MemoryWorkerTurnLease,
]:
    runtime, broker, coordinator, _ = _binding_runtime()
    reservation = await runtime.reserve_session(_principal("one"), "session-1")
    request = AgentRequest("hello", "session-1", run_id)
    lease = await runtime.bind_turn(reservation, request, assignment_pin=_pin())
    return runtime, broker, coordinator, request, lease


@pytest.mark.parametrize("mutation", ["session_key", "run_id", "capability"])
@pytest.mark.asyncio
async def test_request_authority_mutation_before_run_rejects_agent(
    mutation: str,
) -> None:
    runtime, _, _, request, lease = await _bound_response_lease(
        f"mutated-before-run-{mutation}"
    )
    capability = request.memory
    assert capability is not None
    calls = 0

    class NeverRunAgent:
        async def run(
            self,
            agent_request: AgentRequest,
            *,
            emitter: EventEmitter,
        ) -> AgentResponse:
            nonlocal calls
            del agent_request, emitter
            calls += 1
            return AgentResponse(summary="unreachable")

    if mutation == "session_key":
        request.session_key = "changed-session"
    elif mutation == "run_id":
        request.run_id = "changed-run"
    else:
        request._areal_memory_turn_capability = object()  # type: ignore[attr-defined]

    with pytest.raises(MemoryAgentTurnConflictError, match="changed after"):
        await lease.run_agent(NeverRunAgent(), emitter=_DiscardEmitter())
    assert calls == 0
    with pytest.raises(MemoryAgentTurnConflictError, match="closed"):
        await capability.expose_memory("after-mutation", query=b"question")
    await runtime.aclose()


@pytest.mark.asyncio
async def test_agent_cannot_return_after_mutating_bound_request_identity() -> None:
    runtime, _, _, request, lease = await _bound_response_lease("mutated-during-run")
    capability = request.memory
    assert capability is not None

    class MutatingAgent:
        async def run(
            self,
            agent_request: AgentRequest,
            *,
            emitter: EventEmitter,
        ) -> AgentResponse:
            del emitter
            agent_request.run_id = "mutated-by-agent"
            return AgentResponse(summary="must not escape")

    with pytest.raises(MemoryAgentTurnConflictError, match="identity changed"):
        await lease.run_agent(MutatingAgent(), emitter=_DiscardEmitter())
    with pytest.raises(MemoryAgentTurnConflictError, match="closed"):
        await capability.expose_memory("after-agent-mutation", query=b"question")
    await runtime.aclose()


@pytest.mark.asyncio
async def test_lease_runs_exact_request_and_closes_structured_response() -> None:
    runtime, broker, _, request, lease = await _bound_response_lease("structured")
    capability = request.memory
    assert capability is not None

    class StructuredAgent:
        seen_request: AgentRequest | None = None
        calls = 0

        async def run(
            self,
            agent_request: AgentRequest,
            *,
            emitter: EventEmitter,
        ) -> AgentResponse:
            del emitter
            self.calls += 1
            self.seen_request = agent_request
            assert agent_request.memory is capability
            result = await agent_request.memory.expose_memory(
                "structured-read",
                query=b"question",
            )
            return AgentResponse(summary=result.output["answer"])

    agent = StructuredAgent()
    response = await lease.run_agent(agent, emitter=_DiscardEmitter())

    assert response == AgentResponse(summary="from-memory")
    assert agent.calls == 1
    assert agent.seen_request is request
    with pytest.raises(MemoryAgentTurnConflictError, match="closed"):
        await capability.expose_memory("after-response", query=b"question")
    with pytest.raises(MemoryAgentTurnConflictError, match="already run"):
        await lease.run_agent(agent, emitter=_DiscardEmitter())
    states = getattr(broker, "_AuthorizedMemoryAgentBroker__sessions")
    assert capability not in states["session-1"].capabilities
    await runtime.aclose()


@pytest.mark.parametrize(
    ("mode", "error_type", "message"),
    [
        ("agent-error", RuntimeError, "injected Agent failure"),
        ("invalid-return", TypeError, "must return AgentResponse or StreamResponse"),
    ],
)
@pytest.mark.asyncio
async def test_agent_failure_or_invalid_return_closes_lease(
    mode: str,
    error_type: type[BaseException],
    message: str,
) -> None:
    runtime, _, _, request, lease = await _bound_response_lease(mode)
    capability = request.memory
    assert capability is not None

    class BrokenAgent:
        async def run(
            self,
            agent_request: AgentRequest,
            *,
            emitter: EventEmitter,
        ) -> object:
            del agent_request, emitter
            if mode == "agent-error":
                raise RuntimeError("injected Agent failure")
            return object()

    with pytest.raises(error_type, match=message):
        await lease.run_agent(  # type: ignore[arg-type]
            BrokenAgent(),
            emitter=_DiscardEmitter(),
        )
    with pytest.raises(MemoryAgentTurnConflictError, match="closed"):
        await capability.expose_memory("after-failure", query=b"question")
    await runtime.aclose()


@pytest.mark.asyncio
async def test_agent_cancellation_closes_lease() -> None:
    runtime, _, _, request, lease = await _bound_response_lease("cancel-agent")
    capability = request.memory
    assert capability is not None
    started = asyncio.Event()

    class BlockingAgent:
        async def run(
            self,
            agent_request: AgentRequest,
            *,
            emitter: EventEmitter,
        ) -> AgentResponse:
            del agent_request, emitter
            started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    running = asyncio.create_task(
        lease.run_agent(BlockingAgent(), emitter=_DiscardEmitter())
    )
    await asyncio.wait_for(started.wait(), timeout=1)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    with pytest.raises(MemoryAgentTurnConflictError, match="closed"):
        await capability.expose_memory("after-cancel", query=b"question")
    await runtime.aclose()


@pytest.mark.asyncio
async def test_close_during_structured_agent_run_cannot_return_success() -> None:
    runtime, _, _, _, lease = await _bound_response_lease("close-during-run")
    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingStructuredAgent:
        async def run(
            self,
            request: AgentRequest,
            *,
            emitter: EventEmitter,
        ) -> AgentResponse:
            del request, emitter
            started.set()
            await release.wait()
            return AgentResponse(summary="must not escape after close")

    running = asyncio.create_task(
        lease.run_agent(BlockingStructuredAgent(), emitter=_DiscardEmitter())
    )
    await asyncio.wait_for(started.wait(), timeout=1)
    await lease.aclose()
    release.set()
    with pytest.raises(MemoryAgentTurnConflictError, match="closed while"):
        await running
    await runtime.aclose()


@pytest.mark.parametrize(
    ("failure_phase", "message", "source_close_fails"),
    [
        ("iterator", "injected iterator construction failure", False),
        ("headers", "injected headers clone failure", False),
        ("headers-close-error", "injected headers clone failure", True),
    ],
)
@pytest.mark.asyncio
async def test_failed_stream_handoff_closes_source_before_lease(
    failure_phase: str,
    message: str,
    source_close_fails: bool,
) -> None:
    runtime, _, _, request, lease = await _bound_response_lease(
        f"stream-handoff-{failure_phase}"
    )
    capability = request.memory
    assert capability is not None
    events: list[str] = []
    capability_close = capability.aclose

    async def observed_capability_close() -> None:
        events.append("lease-close")
        await capability_close()

    capability.aclose = observed_capability_close  # type: ignore[method-assign]

    class CloseAwareBody:
        def __aiter__(self):
            if failure_phase == "iterator":
                raise RuntimeError("injected iterator construction failure")
            return self

        async def __anext__(self) -> bytes:
            raise AssertionError("failed handoff body must never be read")

        async def aclose(self) -> None:
            events.append("source-close-start")
            result = await capability.expose_memory(
                "failed-handoff-finalizer",
                query=b"finalize",
            )
            assert result.output == {"answer": "from-memory"}
            events.append("source-close-done")
            if source_close_fails:
                raise LookupError("injected source close failure")

    class BrokenHeaders:
        def keys(self):
            raise RuntimeError("injected headers clone failure")

        def __getitem__(self, key: object) -> object:  # pragma: no cover
            raise AssertionError(key)

    source = CloseAwareBody()

    class BrokenStreamAgent:
        async def run(
            self,
            agent_request: AgentRequest,
            *,
            emitter: EventEmitter,
        ) -> StreamResponse:
            del agent_request, emitter
            headers = BrokenHeaders() if failure_phase.startswith("headers") else {}
            return StreamResponse(  # type: ignore[arg-type]
                status_code=200,
                headers=headers,
                body=source,
            )

    with pytest.raises(RuntimeError, match=message) as raised:
        await lease.run_agent(BrokenStreamAgent(), emitter=_DiscardEmitter())

    assert events == ["source-close-start", "source-close-done", "lease-close"]
    notes = getattr(raised.value, "__notes__", ())
    assert any("injected source close failure" in note for note in notes) is (
        source_close_fails
    )
    with pytest.raises(MemoryAgentTurnConflictError, match="closed"):
        await capability.expose_memory("after-handoff-failure", query=b"question")
    await runtime.aclose()


@pytest.mark.asyncio
async def test_stream_keeps_memory_until_eof_and_rejects_a_second_run() -> None:
    runtime, _, coordinator, request, lease = await _bound_response_lease("stream-eof")
    capability = request.memory
    assert capability is not None

    class LazyAgent:
        calls = 0

        async def run(
            self,
            agent_request: AgentRequest,
            *,
            emitter: EventEmitter,
        ) -> StreamResponse:
            del emitter
            self.calls += 1

            async def body():
                first = await agent_request.memory.expose_memory(
                    "stream-first",
                    query=b"first",
                )
                yield first.output["answer"].encode()
                second = await agent_request.memory.expose_memory(
                    "stream-second",
                    query=b"second",
                )
                yield second.output["answer"].encode()

            return StreamResponse(
                status_code=201,
                headers={"content-type": "application/octet-stream"},
                body=body(),
            )

    agent = LazyAgent()
    response = await lease.run_agent(agent, emitter=_DiscardEmitter())
    assert response.status_code == 201
    assert response.headers == {"content-type": "application/octet-stream"}
    assert coordinator.expose_calls == []

    with pytest.raises(MemoryAgentTurnConflictError, match="already run"):
        await lease.run_agent(agent, emitter=_DiscardEmitter())
    assert agent.calls == 1
    assert [chunk async for chunk in response.body] == [
        b"from-memory",
        b"from-memory",
    ]
    assert len(coordinator.expose_calls) == 2
    with pytest.raises(MemoryAgentTurnConflictError, match="closed"):
        await capability.expose_memory("after-eof", query=b"question")
    await runtime.aclose()


@pytest.mark.asyncio
async def test_stream_error_closes_lease() -> None:
    runtime, _, _, request, lease = await _bound_response_lease("stream-error")
    capability = request.memory
    assert capability is not None

    class FailingStreamAgent:
        async def run(
            self,
            agent_request: AgentRequest,
            *,
            emitter: EventEmitter,
        ) -> StreamResponse:
            del agent_request, emitter

            async def body():
                yield b"first"
                raise RuntimeError("injected stream failure")

            return StreamResponse(status_code=200, headers={}, body=body())

    response = await lease.run_agent(FailingStreamAgent(), emitter=_DiscardEmitter())
    assert await anext(response.body) == b"first"
    with pytest.raises(RuntimeError, match="injected stream failure"):
        await anext(response.body)
    with pytest.raises(MemoryAgentTurnConflictError, match="closed"):
        await capability.expose_memory("after-stream-error", query=b"question")
    await runtime.aclose()


@pytest.mark.asyncio
async def test_stream_read_cancellation_closes_lease() -> None:
    runtime, _, _, request, lease = await _bound_response_lease("stream-cancel")
    capability = request.memory
    assert capability is not None
    blocked = asyncio.Event()

    class BlockingStreamAgent:
        async def run(
            self,
            agent_request: AgentRequest,
            *,
            emitter: EventEmitter,
        ) -> StreamResponse:
            del agent_request, emitter

            async def body():
                yield b"first"
                blocked.set()
                await asyncio.Event().wait()
                yield b"unreachable"

            return StreamResponse(status_code=200, headers={}, body=body())

    response = await lease.run_agent(BlockingStreamAgent(), emitter=_DiscardEmitter())
    assert await anext(response.body) == b"first"
    reading = asyncio.create_task(anext(response.body))
    await asyncio.wait_for(blocked.wait(), timeout=1)
    reading.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reading
    with pytest.raises(MemoryAgentTurnConflictError, match="closed"):
        await capability.expose_memory("after-stream-cancel", query=b"question")
    await runtime.aclose()


@pytest.mark.asyncio
async def test_explicit_close_cancels_active_read_before_lease_close() -> None:
    runtime, _, _, request, lease = await _bound_response_lease("active-stream-close")
    capability = request.memory
    assert capability is not None
    read_started = asyncio.Event()
    events: list[str] = []
    capability_close = capability.aclose

    async def observed_capability_close() -> None:
        events.append("lease-close")
        await capability_close()

    capability.aclose = observed_capability_close  # type: ignore[method-assign]

    class ActiveStreamAgent:
        async def run(
            self,
            agent_request: AgentRequest,
            *,
            emitter: EventEmitter,
        ) -> StreamResponse:
            del emitter

            async def body():
                try:
                    read_started.set()
                    await asyncio.Event().wait()
                    yield b"unreachable"
                finally:
                    events.append("source-finally-start")
                    await agent_request.memory.expose_memory(
                        "active-read-finalizer",
                        query=b"finalize",
                    )
                    events.append("source-finally-done")

            return StreamResponse(status_code=200, headers={}, body=body())

    response = await lease.run_agent(ActiveStreamAgent(), emitter=_DiscardEmitter())
    reading = asyncio.create_task(anext(response.body))
    await asyncio.wait_for(read_started.wait(), timeout=1)
    await asyncio.wait_for(response.body.aclose(), timeout=1)
    with pytest.raises(asyncio.CancelledError):
        await reading

    assert events == [
        "source-finally-start",
        "source-finally-done",
        "lease-close",
    ]
    await runtime.aclose()


@pytest.mark.asyncio
async def test_explicit_stream_close_runs_source_finally_before_lease_close() -> None:
    runtime, _, _, request, lease = await _bound_response_lease("stream-close")
    capability = request.memory
    assert capability is not None
    events: list[str] = []
    capability_close = capability.aclose

    async def observed_capability_close() -> None:
        events.append("lease-close")
        await capability_close()

    capability.aclose = observed_capability_close  # type: ignore[method-assign]

    class FinalizingStreamAgent:
        async def run(
            self,
            agent_request: AgentRequest,
            *,
            emitter: EventEmitter,
        ) -> StreamResponse:
            del emitter

            async def body():
                try:
                    yield b"first"
                    await asyncio.Event().wait()
                finally:
                    events.append("source-finally-start")
                    await agent_request.memory.expose_memory(
                        "stream-finalizer",
                        query=b"finalize",
                    )
                    events.append("source-finally-done")

            return StreamResponse(status_code=200, headers={}, body=body())

    response = await lease.run_agent(FinalizingStreamAgent(), emitter=_DiscardEmitter())
    assert await anext(response.body) == b"first"
    await response.body.aclose()

    assert events == [
        "source-finally-start",
        "source-finally-done",
        "lease-close",
    ]
    with pytest.raises(MemoryAgentTurnConflictError, match="closed"):
        await capability.expose_memory("after-explicit-close", query=b"question")
    await runtime.aclose()


@pytest.mark.asyncio
async def test_closed_lease_rejects_agent_before_invocation() -> None:
    runtime, _, _, _, lease = await _bound_response_lease("closed-before-run")
    calls = 0

    class NeverRunAgent:
        async def run(
            self,
            request: AgentRequest,
            *,
            emitter: EventEmitter,
        ) -> AgentResponse:
            nonlocal calls
            del request, emitter
            calls += 1
            return AgentResponse(summary="unreachable")

    await lease.aclose()
    with pytest.raises(
        MemoryAgentTurnConflictError, match="already run or been closed"
    ):
        await lease.run_agent(NeverRunAgent(), emitter=_DiscardEmitter())
    assert calls == 0
    await runtime.aclose()


def _pin_with_one_changed_field(field_name: str) -> MemoryAgentSessionPinV1:
    candidate = _pin()
    if field_name in {"tenant_id", "namespace", "subject_id"}:
        object.__setattr__(candidate.scope, field_name, f"changed-{field_name}")
    elif field_name == "rollout_group_id":
        object.__setattr__(candidate, field_name, "changed-rollout-group")
    elif field_name == "rollout_group_incarnation_sha256":
        object.__setattr__(candidate, field_name, _hash("changed-incarnation"))
    elif field_name == "assignment_id":
        object.__setattr__(
            candidate,
            field_name,
            f"masn_{_hash('changed-assignment')[:24]}",
        )
    elif field_name == "assignment_content_sha256":
        object.__setattr__(candidate, field_name, _hash("changed-assignment"))
    else:  # pragma: no cover - test helper invariant
        raise AssertionError(field_name)
    return candidate


@pytest.mark.parametrize(
    "field_name",
    [
        "tenant_id",
        "namespace",
        "subject_id",
        "rollout_group_id",
        "rollout_group_incarnation_sha256",
        "assignment_id",
        "assignment_content_sha256",
    ],
)
@pytest.mark.asyncio
async def test_bound_assignment_rejects_every_changed_pin_field_before_resolution(
    field_name: str,
) -> None:
    runtime, _, coordinator, resolver = _binding_runtime()
    reservation = await runtime.reserve_session(_principal("one"), "session-1")
    original_request = AgentRequest("hello", "session-1", "run-original")
    original_lease = await runtime.bind_turn(
        reservation,
        original_request,
        assignment_pin=_pin(),
    )
    await original_lease.aclose()
    resolver_calls = len(resolver.calls)
    pin_calls = len(coordinator.pin_calls)
    start_calls = len(coordinator.start_calls)

    with pytest.raises((MemoryAgentSessionConflictError, ValueError)):
        await runtime.bind_turn(
            reservation,
            AgentRequest("retry", "session-1", f"run-{field_name}"),
            assignment_pin=_pin_with_one_changed_field(field_name),
        )

    assert len(resolver.calls) == resolver_calls
    assert len(coordinator.pin_calls) == pin_calls
    assert len(coordinator.start_calls) == start_calls
    await runtime.aclose()


@pytest.mark.asyncio
async def test_bound_assignment_rejects_complete_alternative_before_resolution() -> (
    None
):
    runtime, _, coordinator, resolver = _binding_runtime()
    reservation = await runtime.reserve_session(_principal("one"), "session-1")
    original_pin = _pin()
    original_lease = await runtime.bind_turn(
        reservation,
        AgentRequest("original", "session-1", "run-original"),
        assignment_pin=original_pin,
    )
    await original_lease.aclose()
    resolver_calls = len(resolver.calls)
    pin_calls = len(coordinator.pin_calls)
    alternative_assignment = _pin("alternative")
    alternative_pin = MemoryAgentSessionPinV1(
        scope=original_pin.scope,
        rollout_group_id=original_pin.rollout_group_id,
        rollout_group_incarnation_sha256=(
            original_pin.rollout_group_incarnation_sha256
        ),
        assignment_id=alternative_assignment.assignment_id,
        assignment_content_sha256=(alternative_assignment.assignment_content_sha256),
    )

    with pytest.raises(
        MemoryAgentSessionConflictError,
        match="another Memory assignment",
    ):
        await runtime.bind_turn(
            reservation,
            AgentRequest("alternative", "session-1", "run-alternative"),
            assignment_pin=alternative_pin,
        )

    assert len(resolver.calls) == resolver_calls
    assert len(coordinator.pin_calls) == pin_calls
    assert coordinator.start_calls == [("session-1", "run-original")]
    await runtime.aclose()


@pytest.mark.asyncio
async def test_bind_rejects_equal_stale_and_cross_runtime_reservations_first() -> None:
    runtime, _, coordinator, resolver = _binding_runtime()
    other_runtime, _, other_coordinator, other_resolver = _binding_runtime()
    principal = _principal("one")
    reservation = await runtime.reserve_session(principal, "session-1")
    forged = MemoryWorkerSessionReservationV1(
        session_key=reservation.session_key,
        session=reservation.session,
        audience=reservation.audience,
    )
    assert forged == reservation and forged is not reservation

    with pytest.raises(MemoryAgentSessionConflictError, match="not current"):
        await runtime.bind_turn(
            forged,
            object(),  # type: ignore[arg-type]
            assignment_pin=object(),  # type: ignore[arg-type]
        )

    await runtime.close_session(reservation)
    replacement = await runtime.reserve_session(principal, "session-1")
    assert replacement.session != reservation.session
    with pytest.raises(MemoryAgentSessionConflictError, match="not current"):
        await runtime.bind_turn(
            reservation,
            AgentRequest("hello", "session-1", "stale-run"),
            assignment_pin=_pin(),
        )

    cross_runtime = await other_runtime.reserve_session(principal, "session-1")
    with pytest.raises(MemoryAgentSessionConflictError, match="not current"):
        await runtime.bind_turn(
            cross_runtime,
            AgentRequest("hello", "session-1", "cross-run"),
            assignment_pin=_pin(),
        )

    assert resolver.calls == []
    assert coordinator.pin_calls == []
    assert coordinator.start_calls == []
    assert other_resolver.calls == []
    assert other_coordinator.pin_calls == []
    assert other_coordinator.start_calls == []
    await runtime.aclose()
    await other_runtime.aclose()


@pytest.mark.asyncio
async def test_request_and_pin_validation_precede_resolver_and_coordinator() -> None:
    runtime, _, coordinator, resolver = _binding_runtime()
    reservation = await runtime.reserve_session(_principal("one"), "session-1")
    invalid_requests: list[object] = [
        object(),
        AgentRequest("hello", "another-session", "run-session"),
        AgentRequest("hello", "session-1", ""),
        AgentRequest("hello", "session-1", "   "),
        AgentRequest("hello", "session-1", "invalid-\ud800"),
        AgentRequest("hello", "session-1", 7),  # type: ignore[arg-type]
    ]
    already_bound = AgentRequest("hello", "session-1", "run-bound")
    already_bound._areal_memory_turn_capability = object()  # type: ignore[attr-defined]
    invalid_requests.append(already_bound)

    for invalid in invalid_requests:
        with pytest.raises(
            (TypeError, ValueError, MemoryAgentTurnConflictError),
        ):
            await runtime.bind_turn(
                reservation,
                invalid,  # type: ignore[arg-type]
                assignment_pin=_pin(),
            )

    with pytest.raises(TypeError, match="assignment_pin"):
        await runtime.bind_turn(
            reservation,
            AgentRequest("hello", "session-1", "run-bad-pin-type"),
            assignment_pin=object(),  # type: ignore[arg-type]
        )
    malformed_pin = _pin()
    object.__setattr__(malformed_pin, "assignment_content_sha256", "not-a-hash")
    with pytest.raises(ValueError):
        await runtime.bind_turn(
            reservation,
            AgentRequest("hello", "session-1", "run-bad-pin"),
            assignment_pin=malformed_pin,
        )

    assert resolver.calls == []
    assert coordinator.pin_calls == []
    assert coordinator.start_calls == []
    await runtime.aclose()


@pytest.mark.asyncio
async def test_duplicate_run_reuses_trajectory_with_independent_capabilities() -> None:
    runtime, broker, coordinator, _ = _binding_runtime()
    reservation = await runtime.reserve_session(_principal("one"), "session-1")
    first_request = AgentRequest("first", "session-1", "same-run")
    second_request = AgentRequest("second", "session-1", "same-run")
    first_lease = await runtime.bind_turn(
        reservation,
        first_request,
        assignment_pin=_pin(),
    )
    second_lease = await runtime.bind_turn(
        reservation,
        second_request,
        assignment_pin=_pin(),
    )
    assert first_request.memory is not None
    assert second_request.memory is not None
    assert first_request.memory is not second_request.memory

    await first_request.memory.expose_memory("first-read", query=b"one")
    await first_lease.aclose()
    with pytest.raises(MemoryAgentTurnConflictError, match="closed"):
        await first_request.memory.expose_memory("closed-read", query=b"closed")
    await second_request.memory.expose_memory("second-read", query=b"two")

    first_turn = coordinator.expose_calls[0][0]
    second_turn = coordinator.expose_calls[1][0]
    assert type(first_turn) is MemoryAgentTurnV1
    assert type(second_turn) is MemoryAgentTurnV1
    assert first_turn.memory_trajectory_id == second_turn.memory_trajectory_id
    assert coordinator.start_calls == [("session-1", "same-run")]
    states = getattr(broker, "_AuthorizedMemoryAgentBroker__sessions")
    assert second_request.memory in states["session-1"].capabilities
    await second_lease.aclose()
    await runtime.aclose()


@pytest.mark.asyncio
async def test_turn_lease_close_is_idempotent_and_cancellation_safe() -> None:
    runtime, _, coordinator, _ = _binding_runtime()
    reservation = await runtime.reserve_session(_principal("one"), "session-1")
    request = AgentRequest("hello", "session-1", "run-1")
    lease = await runtime.bind_turn(reservation, request, assignment_pin=_pin())
    assert request.memory is not None
    capability = request.memory
    coordinator.expose_release.clear()
    exposing = asyncio.create_task(capability.expose_memory("read", query=b"question"))
    await asyncio.wait_for(coordinator.expose_started.wait(), timeout=1)

    capability_close = capability.aclose
    close_started = asyncio.Event()

    async def observed_capability_close() -> None:
        close_started.set()
        await capability_close()

    capability.aclose = observed_capability_close  # type: ignore[method-assign]
    close_waiter = asyncio.create_task(lease.aclose())
    await asyncio.wait_for(close_started.wait(), timeout=1)
    close_waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await close_waiter

    coordinator.expose_release.set()
    result = await exposing
    assert result.output == {"answer": "from-memory"}
    await lease.aclose()
    await lease.aclose()
    with pytest.raises(MemoryAgentTurnConflictError, match="closed"):
        await capability.expose_memory("after-close", query=b"question")
    await runtime.aclose()


@pytest.mark.asyncio
async def test_denied_pin_never_binds_and_valid_retry_recovers() -> None:
    runtime, _, coordinator, resolver = _binding_runtime()
    reservation = await runtime.reserve_session(_principal("one"), "session-1")
    request = AgentRequest("hello", "session-1", "run-1")
    resolver.active_actions.remove(MemoryScopeActionV1.PIN_ASSIGNMENT)

    with pytest.raises(MemoryScopeAuthorizationDeniedError):
        await runtime.bind_turn(reservation, request, assignment_pin=_pin())
    assert request.memory is None
    assert coordinator.pin_calls == []
    assert coordinator.start_calls == []

    resolver.active_actions.add(MemoryScopeActionV1.PIN_ASSIGNMENT)
    lease = await runtime.bind_turn(reservation, request, assignment_pin=_pin())
    assert request.memory is not None
    await lease.aclose()
    await runtime.aclose()


@pytest.mark.asyncio
async def test_cancelled_pin_leaves_no_capability_and_retry_recovers() -> None:
    runtime, broker, coordinator, _ = _binding_runtime()
    reservation = await runtime.reserve_session(_principal("one"), "session-1")
    request = AgentRequest("hello", "session-1", "run-1")
    coordinator.pin_release.clear()
    binding = asyncio.create_task(
        runtime.bind_turn(reservation, request, assignment_pin=_pin())
    )
    try:
        await asyncio.wait_for(coordinator.pin_started.wait(), timeout=1)
        binding.cancel()
        with pytest.raises(asyncio.CancelledError):
            await binding
        assert request.memory is None
    finally:
        coordinator.pin_release.set()

    lease = await runtime.bind_turn(reservation, request, assignment_pin=_pin())
    states = getattr(broker, "_AuthorizedMemoryAgentBroker__sessions")
    assert states["session-1"].admitted == 0
    assert request.memory in states["session-1"].capabilities
    await lease.aclose()
    await runtime.aclose()


@pytest.mark.parametrize("failure_mode", ["cancel", "failure"])
@pytest.mark.asyncio
async def test_cancelled_or_failed_start_leaves_no_capability_and_retry_recovers(
    failure_mode: str,
) -> None:
    runtime, broker, coordinator, _ = _binding_runtime()
    reservation = await runtime.reserve_session(_principal("one"), "session-1")
    failed_request = AgentRequest("hello", "session-1", "same-run")
    if failure_mode == "cancel":
        coordinator.start_release.clear()
    else:
        coordinator.start_failures = 1

    binding = asyncio.create_task(
        runtime.bind_turn(reservation, failed_request, assignment_pin=_pin())
    )
    if failure_mode == "cancel":
        try:
            await asyncio.wait_for(coordinator.start_started.wait(), timeout=1)
            binding.cancel()
            with pytest.raises(asyncio.CancelledError):
                await binding
        finally:
            coordinator.start_release.set()
    else:
        with pytest.raises(RuntimeError, match="injected start failure"):
            await binding
    assert failed_request.memory is None

    retry = AgentRequest("retry", "session-1", "same-run")
    lease = await runtime.bind_turn(reservation, retry, assignment_pin=_pin())
    states = getattr(broker, "_AuthorizedMemoryAgentBroker__sessions")
    assert states["session-1"].admitted == 0
    assert retry.memory in states["session-1"].capabilities
    await lease.aclose()
    await runtime.aclose()


@pytest.mark.parametrize("blocked_phase", ["pin", "start"])
@pytest.mark.asyncio
async def test_close_race_cannot_publish_a_capability_and_reopen_recovers(
    blocked_phase: str,
) -> None:
    runtime, broker, coordinator, _ = _binding_runtime()
    principal = _principal("one")
    reservation = await runtime.reserve_session(principal, "session-1")
    request = AgentRequest("hello", "session-1", "run-before-close")
    if blocked_phase == "pin":
        coordinator.pin_release.clear()
        phase_started = coordinator.pin_started
        phase_release = coordinator.pin_release
    else:
        coordinator.start_release.clear()
        phase_started = coordinator.start_started
        phase_release = coordinator.start_release

    binding = asyncio.create_task(
        runtime.bind_turn(reservation, request, assignment_pin=_pin())
    )
    await asyncio.wait_for(phase_started.wait(), timeout=1)

    close_admitted = asyncio.Event()
    broker_close_state = broker._close_state

    async def observed_close_state(state: object) -> None:
        close_admitted.set()
        await broker_close_state(state)  # type: ignore[arg-type]

    broker._close_state = observed_close_state  # type: ignore[method-assign]
    closing = asyncio.create_task(runtime.close_session(reservation))
    await asyncio.wait_for(close_admitted.wait(), timeout=1)
    broker_states = getattr(broker, "_AuthorizedMemoryAgentBroker__sessions")
    assert broker_states["session-1"].closing

    phase_release.set()
    with pytest.raises(MemoryAgentSessionConflictError):
        await binding
    await closing
    assert request.memory is None
    assert "session-1" not in broker_states

    replacement = await runtime.reserve_session(principal, "session-1")
    retry = AgentRequest("retry", "session-1", "run-after-close")
    lease = await runtime.bind_turn(replacement, retry, assignment_pin=_pin())
    assert retry.memory is not None
    await lease.aclose()
    await runtime.aclose()


@pytest.mark.asyncio
async def test_runtime_close_handoff_cannot_publish_an_existing_turn() -> None:
    runtime, broker, _, _ = _binding_runtime()
    reservation = await runtime.reserve_session(_principal("one"), "session-1")
    first_request = AgentRequest("first", "session-1", "same-run")
    first_lease = await runtime.bind_turn(
        reservation,
        first_request,
        assignment_pin=_pin(),
    )
    await first_lease.aclose()

    pin_entered = asyncio.Event()
    pin_release = asyncio.Event()

    async def delayed_existing_pin(session: object, pin: object) -> object:
        pin_entered.set()
        await pin_release.wait()
        return object()

    broker.pin_session = delayed_existing_pin  # type: ignore[method-assign]
    second_request = AgentRequest("second", "session-1", "same-run")
    binding = asyncio.create_task(
        runtime.bind_turn(reservation, second_request, assignment_pin=_pin())
    )
    await asyncio.wait_for(pin_entered.wait(), timeout=1)

    async def release_then_close() -> None:
        # Queue the binding first, then synchronously mark the runtime state as
        # closing before its broker close task gets an event-loop turn.
        pin_release.set()
        await runtime.close_session(reservation)

    closing = asyncio.create_task(release_then_close())
    with pytest.raises(MemoryAgentSessionConflictError, match="closing"):
        await binding
    await closing

    assert second_request.memory is None
    runtime_states = getattr(runtime, "_AuthorizedMemoryWorkerRuntime__sessions")
    assert "session-1" not in runtime_states
    await runtime.aclose()


@pytest.mark.asyncio
async def test_lease_construction_failure_closes_registered_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from areal.v2.agent_service.worker import memory_runtime as runtime_module

    runtime, broker, _, _ = _binding_runtime()
    reservation = await runtime.reserve_session(_principal("one"), "session-1")
    request = AgentRequest("hello", "session-1", "run-1")
    unregistered = asyncio.Event()
    unregister = broker._unregister_capability

    def observed_unregister(capability: object) -> None:
        unregister(capability)  # type: ignore[arg-type]
        unregistered.set()

    broker._unregister_capability = observed_unregister  # type: ignore[method-assign]

    def fail_lease_construction(
        bound_request: object,
        capability: object,
    ) -> None:
        assert bound_request is request
        raise RuntimeError("injected lease construction failure")

    monkeypatch.setattr(
        runtime_module,
        "MemoryWorkerTurnLease",
        fail_lease_construction,
    )
    with pytest.raises(RuntimeError, match="injected lease construction failure"):
        await runtime.bind_turn(reservation, request, assignment_pin=_pin())
    await asyncio.wait_for(unregistered.wait(), timeout=1)

    assert request.memory is None
    states = getattr(broker, "_AuthorizedMemoryAgentBroker__sessions")
    assert states["session-1"].capabilities == set()
    await runtime.aclose()


@pytest.mark.asyncio
async def test_lease_construction_error_survives_cleanup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from areal.v2.agent_service.worker import memory_runtime as runtime_module

    runtime, broker, _, _ = _binding_runtime()
    reservation = await runtime.reserve_session(_principal("one"), "session-1")
    request = AgentRequest("hello", "session-1", "run-1")
    captured: dict[str, object] = {}

    def fail_lease_construction(
        bound_request: object,
        capability: object,
    ) -> None:
        assert bound_request is request
        captured["capability"] = capability
        captured["aclose"] = capability.aclose  # type: ignore[attr-defined]

        async def fail_cleanup() -> None:
            raise RuntimeError("injected cleanup failure")

        capability.aclose = fail_cleanup  # type: ignore[attr-defined]
        raise ValueError("injected construction failure")

    monkeypatch.setattr(
        runtime_module,
        "MemoryWorkerTurnLease",
        fail_lease_construction,
    )
    with pytest.raises(ValueError, match="injected construction failure") as raised:
        await runtime.bind_turn(reservation, request, assignment_pin=_pin())

    assert any(
        "injected cleanup failure" in note
        for note in getattr(raised.value, "__notes__", ())
    )
    assert request.memory is None
    capability = captured["capability"]
    capability.aclose = captured["aclose"]  # type: ignore[attr-defined]
    await capability.aclose()  # type: ignore[attr-defined]
    states = getattr(broker, "_AuthorizedMemoryAgentBroker__sessions")
    assert states["session-1"].capabilities == set()
    await runtime.aclose()
