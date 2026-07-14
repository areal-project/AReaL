# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import threading
from datetime import UTC, datetime, timedelta
from hashlib import sha256

import pytest

from areal.v2.agent_service.memory import (
    AsyncMemoryAgentCoordinator,
    MemoryAgentSessionConflictError,
    MemoryAgentSessionPinV1,
    MemoryAgentTurnConflictError,
    MemoryAgentTurnV1,
)
from areal.v2.agent_service.memory_authorization import (
    MemoryScopeActionV1,
    MemoryScopeAuthorizationConflictError,
    MemoryScopeAuthorizationDeniedError,
    MemoryScopeAuthorizationDisabledError,
    MemoryScopeGrantAuthorizer,
    MemoryScopeGrantRequestV1,
    MemoryScopeGrantV1,
)
from areal.v2.agent_service.memory_broker import (
    AuthorizedMemoryAgentBroker,
    AuthorizedMemorySessionV1,
)
from areal.v2.agent_service.types import AgentRequest
from areal.v2.agent_service.worker.memory import (
    WorkerMemoryTurnCapability,
    bind_authorized_memory_turn_capability,
)
from areal.v2.memory_service import runtime_types
from areal.v2.memory_service.runtime_types import (
    MemoryExposureStatus,
    MemoryExposureV1,
)
from areal.v2.memory_service.types import MemoryScope

_NOW = datetime(2026, 7, 13, 10, 0, tzinfo=UTC)
_RESOLVER_ID = "test-worker-memory-grant-policy"
_RESOLVER_VERSION = sha256(b"worker-grant-policy-v1").hexdigest()
_RESOLVER_CONFIG = sha256(b"worker-grant-config-v1").hexdigest()


def _hash(label: str) -> str:
    return sha256(label.encode()).hexdigest()


def _principal(*, suffix: str = "1"):
    from areal.v2.agent_service.memory_authorization import MemoryPrincipalV1

    return MemoryPrincipalV1(
        issuer="https://identity.example",
        subject=f"principal-{suffix}",
    )


def _pin(*, suffix: str = "1") -> MemoryAgentSessionPinV1:
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
    turn: MemoryAgentTurnV1, pin: MemoryAgentSessionPinV1
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


class _UnusedReleaseStore:
    def resolve_active_assignment(self, *args: object) -> object:
        raise AssertionError("broker tests override coordinator.pin_session")


class _UnusedRuntimeStore:
    def begin_query(self, *args: object) -> object:
        raise AssertionError("broker tests override coordinator.expose_memory")

    resolve_query = begin_query
    prepare_delivery = begin_query
    submit_delivery = begin_query


class _Coordinator(AsyncMemoryAgentCoordinator):
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
        self.expose_started = asyncio.Event()
        self.expose_release = asyncio.Event()
        self.expose_release.set()
        self.close_started = asyncio.Event()
        self.close_release = asyncio.Event()
        self.close_release.set()

    async def pin_session(  # type: ignore[override]
        self,
        session_key: str,
        pin: MemoryAgentSessionPinV1,
    ) -> object:
        self.pin_calls.append((session_key, pin))
        self.pin_started.set()
        await self.pin_release.wait()
        return {"assignment": pin.assignment_id}

    async def start_turn(
        self,
        session_key: str,
        turn_idempotency_key: str,
    ) -> MemoryAgentTurnV1:
        self.start_calls.append((session_key, turn_idempotency_key))
        return MemoryAgentTurnV1(
            session_key=session_key,
            turn_idempotency_key=turn_idempotency_key,
            memory_trajectory_id=(
                f"mtraj_{sha256(f'{session_key}:{turn_idempotency_key}'.encode()).hexdigest()}"
            ),
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


class _Resolver:
    resolver_id = _RESOLVER_ID
    resolver_version_sha256 = _RESOLVER_VERSION
    resolver_config_sha256 = _RESOLVER_CONFIG

    def __init__(self) -> None:
        self.active_actions = {
            MemoryScopeActionV1.PIN_ASSIGNMENT,
            MemoryScopeActionV1.EXPOSE_MEMORY,
        }
        self.calls: list[MemoryScopeGrantRequestV1] = []
        self.snapshots: list[bytes] = []
        self.block_action: MemoryScopeActionV1 | None = None
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.mutate_next = False
        self.retained: MemoryScopeGrantRequestV1 | None = None
        self._concurrency_lock = threading.Lock()
        self.concurrent = 0
        self.max_concurrent = 0

    def resolve_active_grant(
        self,
        request: MemoryScopeGrantRequestV1,
    ) -> MemoryScopeGrantV1:
        with self._concurrency_lock:
            self.concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            self.calls.append(request)
            self.snapshots.append(request.canonical_bytes())
            self.retained = request
            if request.action is self.block_action:
                self.started.set()
                self.release.wait(timeout=5)
            if request.action not in self.active_actions:
                raise MemoryScopeAuthorizationDeniedError(
                    "active Memory scope grant is unavailable"
                )
            if self.mutate_next:
                self.mutate_next = False
                object.__setattr__(request.target.scope, "tenant_id", "poisoned")
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
        finally:
            with self._concurrency_lock:
                self.concurrent -= 1


def _broker(
    *,
    resolver: _Resolver | None = None,
    coordinator: _Coordinator | None = None,
) -> tuple[AuthorizedMemoryAgentBroker, _Coordinator, _Resolver | None]:
    coordinator = coordinator or _Coordinator()
    authorizer = MemoryScopeGrantAuthorizer(resolver, clock=lambda: _NOW)
    return AuthorizedMemoryAgentBroker(coordinator, authorizer), coordinator, resolver


async def _authorized_capability(
    broker: AuthorizedMemoryAgentBroker,
    *,
    pin: MemoryAgentSessionPinV1 | None = None,
    session_key: str = "session-1",
    run_id: str = "run-1",
) -> tuple[AuthorizedMemorySessionV1, WorkerMemoryTurnCapability]:
    pin = pin or _pin()
    session = await broker.open_session(_principal(), session_key)
    await broker.pin_session(session, pin)
    turn = await broker.start_turn(session, run_id)
    request = AgentRequest(message="hello", session_key=session_key, run_id=run_id)
    capability = bind_authorized_memory_turn_capability(request, broker, turn)
    assert request.memory is capability
    return session, capability


@pytest.mark.asyncio
async def test_disabled_and_denied_pin_never_reach_coordinator() -> None:
    disabled, disabled_coordinator, _ = _broker()
    session = await disabled.open_session(_principal(), "disabled-session")
    with pytest.raises(MemoryScopeAuthorizationDisabledError):
        await disabled.pin_session(session, _pin())
    assert disabled_coordinator.pin_calls == []
    await disabled.aclose()

    resolver = _Resolver()
    resolver.active_actions.remove(MemoryScopeActionV1.PIN_ASSIGNMENT)
    denied, denied_coordinator, _ = _broker(resolver=resolver)
    session = await denied.open_session(_principal(), "denied-session")
    with pytest.raises(MemoryScopeAuthorizationDeniedError):
        await denied.pin_session(session, _pin())
    assert denied_coordinator.pin_calls == []
    await denied.aclose()


@pytest.mark.asyncio
async def test_pin_and_every_exposure_use_the_exact_broker_owned_context() -> None:
    resolver = _Resolver()
    broker, coordinator, _ = _broker(resolver=resolver)
    pin = _pin()
    session, capability = await _authorized_capability(broker, pin=pin)

    first = await capability.expose_memory("lookup", query=b"future question")
    second = await capability.expose_memory("lookup", query=b"future question")

    assert first == second
    assert [call.action for call in resolver.calls] == [
        MemoryScopeActionV1.PIN_ASSIGNMENT,
        MemoryScopeActionV1.EXPOSE_MEMORY,
        MemoryScopeActionV1.EXPOSE_MEMORY,
    ]
    for request in resolver.calls:
        assert request.principal == session.principal
        assert request.session == session.session
        assert request.audience == session.audience == broker.audience
        assert request.target.scope == pin.scope
        assert request.target.rollout_group_id == pin.rollout_group_id
        assert (
            request.target.rollout_group_incarnation_sha256
            == pin.rollout_group_incarnation_sha256
        )
        assert request.target.assignment_id == pin.assignment_id
        assert request.target.assignment_content_sha256 == pin.assignment_content_sha256
    assert len(coordinator.pin_calls) == 1
    assert len(coordinator.expose_calls) == 2
    await broker.aclose()


@pytest.mark.asyncio
async def test_exact_pin_retry_reauthorizes_before_coordinator_lookup() -> None:
    resolver = _Resolver()
    broker, coordinator, _ = _broker(resolver=resolver)
    session = await broker.open_session(_principal(), "session-1")

    await broker.pin_session(session, _pin())
    resolver.active_actions.remove(MemoryScopeActionV1.PIN_ASSIGNMENT)
    with pytest.raises(MemoryScopeAuthorizationDeniedError):
        await broker.pin_session(session, _pin())

    assert len(coordinator.pin_calls) == 1
    assert [call.action for call in resolver.calls].count(
        MemoryScopeActionV1.PIN_ASSIGNMENT
    ) == 2
    await broker.aclose()


@pytest.mark.asyncio
async def test_pin_integrity_conflict_always_clears_transient_candidate() -> None:
    """An invariant error cannot leak admission-only pin state.

    Normal broker admission prevents two different targets from reaching the
    coordinator concurrently.  This deliberately injects the defensive
    post-coordinator mismatch to prove that its exception path still performs
    the unconditional candidate cleanup promised by ``finally``.
    """

    resolver = _Resolver()
    broker, coordinator, _ = _broker(resolver=resolver)
    session = await broker.open_session(_principal(), "session-1")
    first_pin = _pin(suffix="1")
    replacement_pin = _pin(suffix="2")
    coordinator.pin_release.clear()

    pinning = asyncio.create_task(broker.pin_session(session, first_pin))
    try:
        await asyncio.wait_for(coordinator.pin_started.wait(), timeout=1)
        states = getattr(broker, "_AuthorizedMemoryAgentBroker__sessions")
        state = states["session-1"]
        candidate = state.pin_candidate
        assert candidate is not None
        assert state.pin_candidate_count == 1

        # Simulate detection that the coordinator/session already carries another
        # durable assignment.  The integrity error is expected; the transient
        # first-pin candidate must still be released.
        state.target = type(candidate).from_pin(replacement_pin)
        coordinator.pin_release.set()
        with pytest.raises(
            MemoryAgentSessionConflictError,
            match="coordinator pinned a different",
        ):
            await pinning

        assert state.pin_candidate_count == 0
        assert state.pin_candidate is None

        # Once the injected durable state is cleared by its hypothetical recovery
        # owner, a valid replacement can be admitted.  The old implementation left
        # ``first_pin`` behind here and rejected this public call.
        state.target = None
        await broker.pin_session(session, replacement_pin)
        assert len(coordinator.pin_calls) == 2
    finally:
        coordinator.pin_release.set()
        if not pinning.done():
            pinning.cancel()
        await asyncio.gather(pinning, return_exceptions=True)
        await broker.aclose()


@pytest.mark.asyncio
async def test_revoked_exposure_cannot_read_a_completed_coordinator_result() -> None:
    resolver = _Resolver()
    broker, coordinator, _ = _broker(resolver=resolver)
    _, capability = await _authorized_capability(broker)

    result = await capability.expose_memory("same-operation", query=b"question")
    assert result.output == {"answer": "from-memory"}
    resolver.active_actions.remove(MemoryScopeActionV1.EXPOSE_MEMORY)

    with pytest.raises(MemoryScopeAuthorizationDeniedError):
        await capability.expose_memory("same-operation", query=b"question")

    assert len(coordinator.expose_calls) == 1
    assert [call.action for call in resolver.calls].count(
        MemoryScopeActionV1.EXPOSE_MEMORY
    ) == 2
    await broker.aclose()


@pytest.mark.asyncio
async def test_revocation_does_not_cancel_an_already_admitted_consumer() -> None:
    resolver = _Resolver()
    coordinator = _Coordinator()
    coordinator.expose_release.clear()
    broker, coordinator, _ = _broker(resolver=resolver, coordinator=coordinator)
    _, capability = await _authorized_capability(broker)

    admitted = asyncio.create_task(
        capability.expose_memory("same-operation", query=b"question")
    )
    await coordinator.expose_started.wait()
    resolver.active_actions.remove(MemoryScopeActionV1.EXPOSE_MEMORY)

    with pytest.raises(MemoryScopeAuthorizationDeniedError):
        await capability.expose_memory("same-operation", query=b"question")
    assert not admitted.done()
    assert len(coordinator.expose_calls) == 1

    coordinator.expose_release.set()
    assert (await admitted).output == {"answer": "from-memory"}
    await broker.aclose()


@pytest.mark.asyncio
async def test_real_coordinator_preserves_cache_and_detached_turn_identity() -> None:
    from tests.v2.agent_service import test_memory as memory_fakes

    assignment = memory_fakes._assignment()
    control = memory_fakes._ControlStore(assignment)
    runtime = memory_fakes._RuntimeStore()
    coordinator = AsyncMemoryAgentCoordinator(control, runtime)
    resolver = _Resolver()
    broker = AuthorizedMemoryAgentBroker(
        coordinator,
        MemoryScopeGrantAuthorizer(resolver, clock=lambda: _NOW),
    )
    try:
        session = await broker.open_session(_principal(), "real-session")
        await broker.pin_session(session, memory_fakes._pin(assignment))
        turn = await broker.start_turn(session, "real-run")
        capability = WorkerMemoryTurnCapability.from_authorized_turn(broker, turn)

        first = await capability.expose_memory(
            "same-operation",
            query=b"future question",
        )
        second = await capability.expose_memory(
            "same-operation",
            query=b"future question",
        )

        assert first == second
        assert control.calls == 1
        assert runtime.consumer_side_effects == 1
        assert runtime.specs[0].trajectory_id == turn.turn.memory_trajectory_id
        assert [call.action for call in resolver.calls].count(
            MemoryScopeActionV1.EXPOSE_MEMORY
        ) == 2
    finally:
        await broker.aclose()


@pytest.mark.asyncio
async def test_cancelled_admitted_exposure_drains_but_revoked_retry_is_denied() -> None:
    resolver = _Resolver()
    coordinator = _Coordinator()
    coordinator.expose_release.clear()
    broker, coordinator, _ = _broker(resolver=resolver, coordinator=coordinator)
    _, capability = await _authorized_capability(broker)

    caller = asyncio.create_task(
        capability.expose_memory("same-operation", query=b"question")
    )
    await coordinator.expose_started.wait()
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller

    resolver.active_actions.remove(MemoryScopeActionV1.EXPOSE_MEMORY)
    with pytest.raises(MemoryScopeAuthorizationDeniedError):
        await capability.expose_memory("same-operation", query=b"question")
    closing = asyncio.create_task(capability.aclose())
    await asyncio.sleep(0)
    assert not closing.done()

    coordinator.expose_release.set()
    await closing
    assert len(coordinator.expose_calls) == 1
    resolver_calls = len(resolver.calls)
    with pytest.raises(MemoryAgentTurnConflictError, match="closed"):
        await capability.expose_memory("after-close", query=b"question")
    assert len(resolver.calls) == resolver_calls
    await broker.aclose()


@pytest.mark.asyncio
async def test_close_during_pending_pin_invalidates_old_incarnation() -> None:
    resolver = _Resolver()
    resolver.block_action = MemoryScopeActionV1.PIN_ASSIGNMENT
    resolver.release.clear()
    broker, coordinator, _ = _broker(resolver=resolver)
    old = await broker.open_session(_principal(), "reused-session")

    pending = asyncio.create_task(broker.pin_session(old, _pin()))
    while not resolver.started.is_set():
        await asyncio.sleep(0)
    await broker.close_session(old)
    replacement = await broker.open_session(_principal(), "reused-session")
    assert replacement.session.incarnation_id != old.session.incarnation_id

    resolver.release.set()
    with pytest.raises(MemoryAgentSessionConflictError):
        await pending
    assert coordinator.pin_calls == []

    await broker.pin_session(replacement, _pin())
    assert len(coordinator.pin_calls) == 1
    with pytest.raises(MemoryAgentSessionConflictError):
        await broker.start_turn(old, "old-run")
    await broker.aclose()


@pytest.mark.asyncio
async def test_corrupted_session_handle_cannot_promote_a_pending_pin_grant() -> None:
    resolver = _Resolver()
    resolver.block_action = MemoryScopeActionV1.PIN_ASSIGNMENT
    resolver.release.clear()
    broker, coordinator, _ = _broker(resolver=resolver)
    session = await broker.open_session(_principal(), "session-1")

    pending = asyncio.create_task(broker.pin_session(session, _pin()))
    while not resolver.started.is_set():
        await asyncio.sleep(0)
    object.__setattr__(session.principal, "subject", "corrupted")
    resolver.release.set()

    with pytest.raises(MemoryAgentSessionConflictError):
        await pending
    assert coordinator.pin_calls == []
    await broker.aclose()


@pytest.mark.asyncio
async def test_close_waits_for_a_pin_admitted_after_authorization() -> None:
    resolver = _Resolver()
    coordinator = _Coordinator()
    coordinator.pin_release.clear()
    broker, coordinator, _ = _broker(resolver=resolver, coordinator=coordinator)
    session = await broker.open_session(_principal(), "session-1")

    pinning = asyncio.create_task(broker.pin_session(session, _pin()))
    await coordinator.pin_started.wait()
    closing = asyncio.create_task(broker.close_session(session))
    await asyncio.sleep(0)
    assert not closing.done()
    assert coordinator.close_calls == []

    coordinator.pin_release.set()
    with pytest.raises(MemoryAgentSessionConflictError):
        await pinning
    await closing
    assert coordinator.close_calls == ["session-1"]
    await broker.aclose()


@pytest.mark.asyncio
async def test_close_while_exposure_authorization_is_pending_has_no_side_effect() -> (
    None
):
    resolver = _Resolver()
    broker, coordinator, _ = _broker(resolver=resolver)
    session, capability = await _authorized_capability(broker)
    resolver.block_action = MemoryScopeActionV1.EXPOSE_MEMORY
    resolver.release.clear()
    resolver.started.clear()

    pending = asyncio.create_task(capability.expose_memory("lookup", query=b"question"))
    while not resolver.started.is_set():
        await asyncio.sleep(0)
    await broker.close_session(session)
    resolver.release.set()

    with pytest.raises((MemoryAgentSessionConflictError, MemoryAgentTurnConflictError)):
        await pending
    assert coordinator.expose_calls == []
    await broker.aclose()


@pytest.mark.asyncio
async def test_cancellation_during_authorization_cannot_start_a_late_exposure() -> None:
    resolver = _Resolver()
    broker, coordinator, _ = _broker(resolver=resolver)
    _, capability = await _authorized_capability(broker)
    resolver.block_action = MemoryScopeActionV1.EXPOSE_MEMORY
    resolver.release.clear()
    resolver.started.clear()

    caller = asyncio.create_task(capability.expose_memory("lookup", query=b"question"))
    while not resolver.started.is_set():
        await asyncio.sleep(0)
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller

    resolver.release.set()
    while resolver.concurrent:
        await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert coordinator.expose_calls == []
    await broker.aclose()


@pytest.mark.asyncio
async def test_resolver_alias_mutation_does_not_poison_later_requests_or_pin() -> None:
    resolver = _Resolver()
    resolver.mutate_next = True
    broker, coordinator, _ = _broker(resolver=resolver)
    session = await broker.open_session(_principal(), "session-1")
    pin = _pin()

    with pytest.raises(
        MemoryScopeAuthorizationConflictError,
        match="mutated the authorization request",
    ):
        await broker.pin_session(session, pin)
    assert coordinator.pin_calls == []

    await broker.pin_session(session, pin)
    assert resolver.calls[-1].target.scope.tenant_id == pin.scope.tenant_id
    assert coordinator.pin_calls[0][1].scope == pin.scope

    retained = resolver.retained
    assert retained is not None
    object.__setattr__(retained.target.scope, "tenant_id", "late-poison")
    turn = await broker.start_turn(session, "run-1")
    capability = WorkerMemoryTurnCapability.from_authorized_turn(broker, turn)
    await capability.expose_memory("lookup", query=b"question")
    assert resolver.calls[-1].target.scope.tenant_id == pin.scope.tenant_id
    await broker.aclose()


@pytest.mark.asyncio
async def test_public_turn_alias_cannot_mutate_the_coordinator_trajectory() -> None:
    resolver = _Resolver()
    broker, coordinator, _ = _broker(resolver=resolver)
    session = await broker.open_session(_principal(), "session-1")
    await broker.pin_session(session, _pin())
    turn = await broker.start_turn(session, "run-1")
    original_trajectory = turn.turn.memory_trajectory_id

    object.__setattr__(
        turn.turn,
        "memory_trajectory_id",
        "mtraj_attacker-selected",
    )
    with pytest.raises(MemoryAgentTurnConflictError, match="not issued|corrupted"):
        WorkerMemoryTurnCapability.from_authorized_turn(broker, turn)
    assert coordinator.expose_calls == []

    # A retry cannot silently replace the poisoned public object; close is the
    # boundary that permits a fresh session/turn incarnation.
    with pytest.raises(MemoryAgentTurnConflictError, match="corrupted"):
        await broker.start_turn(session, "run-1")
    assert original_trajectory != turn.turn.memory_trajectory_id
    await broker.aclose()


@pytest.mark.asyncio
async def test_blocking_resolver_is_offloaded_and_bounded() -> None:
    resolver = _Resolver()
    resolver.block_action = MemoryScopeActionV1.PIN_ASSIGNMENT
    resolver.release.clear()
    broker, _, _ = _broker(resolver=resolver)
    first = await broker.open_session(_principal(suffix="1"), "session-1")
    second = await broker.open_session(_principal(suffix="2"), "session-2")

    one = asyncio.create_task(broker.pin_session(first, _pin(suffix="1")))
    while not resolver.started.is_set():
        await asyncio.sleep(0)
    two = asyncio.create_task(broker.pin_session(second, _pin(suffix="2")))

    heartbeat = 0
    for _ in range(20):
        await asyncio.sleep(0)
        heartbeat += 1
    assert heartbeat == 20
    assert resolver.max_concurrent == 1
    assert len(resolver.calls) == 1

    resolver.release.set()
    await asyncio.gather(one, two)
    assert resolver.max_concurrent == 1
    assert len(resolver.calls) == 2
    await broker.aclose()


@pytest.mark.asyncio
async def test_cancelled_close_is_rejoined_and_reopen_waits_for_cleanup() -> None:
    resolver = _Resolver()
    coordinator = _Coordinator()
    coordinator.close_release.clear()
    broker, coordinator, _ = _broker(resolver=resolver, coordinator=coordinator)
    session, _ = await _authorized_capability(broker)

    first = asyncio.create_task(broker.close_session(session))
    await coordinator.close_started.wait()
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    with pytest.raises(MemoryAgentSessionConflictError, match="closing"):
        await broker.open_session(_principal(), "session-1")

    second = asyncio.create_task(broker.close_session(session))
    await asyncio.sleep(0)
    assert not second.done()
    coordinator.close_release.set()
    await second

    replacement = await broker.open_session(_principal(), "session-1")
    assert replacement.session.incarnation_id != session.session.incarnation_id
    await broker.aclose()


@pytest.mark.asyncio
async def test_equal_or_cross_broker_handles_are_not_local_authority() -> None:
    # Runtime identity is checked by the async broker methods.  The public
    # record remains descriptive and cannot be converted into a wire bearer.
    assert not hasattr(AuthorizedMemorySessionV1, "from_wire")
    assert not hasattr(AuthorizedMemorySessionV1, "to_wire")

    first_resolver = _Resolver()
    first, _, _ = _broker(resolver=first_resolver)
    issued = await first.open_session(_principal(), "session-1")
    forged_equal = AuthorizedMemorySessionV1(
        principal=issued.principal,
        session=issued.session,
        audience=issued.audience,
    )
    with pytest.raises(MemoryAgentSessionConflictError, match="not current"):
        await first.pin_session(forged_equal, _pin())

    second_resolver = _Resolver()
    second, _, _ = _broker(resolver=second_resolver)
    with pytest.raises(MemoryAgentSessionConflictError, match="not current"):
        await second.pin_session(issued, _pin())
    assert first_resolver.calls == []
    assert second_resolver.calls == []
    await first.aclose()
    await second.aclose()
