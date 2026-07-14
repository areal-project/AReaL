# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from hashlib import sha256

import pytest

from areal.v2.agent_service.memory import (
    AsyncMemoryAgentCoordinator,
    MemoryAgentSessionConflictError,
    MemoryAgentSessionPinV1,
    MemoryAgentTurnV1,
)
from areal.v2.agent_service.memory_authorization import (
    MemoryPrincipalV1,
    MemoryScopeActionV1,
    MemoryScopeGrantAuthorizer,
    MemoryScopeGrantRequestV1,
    MemoryScopeGrantV1,
    MemorySessionIncarnationV1,
    MemoryWorkerAudienceV1,
)
from areal.v2.agent_service.memory_broker import AuthorizedMemoryAgentBroker
from areal.v2.agent_service.memory_session_lifecycle import (
    MemoryWorkerSessionIdentityV1,
)
from areal.v2.agent_service.types import (
    AgentRequest,
    AgentResponse,
    EventEmitter,
    StreamResponse,
)
from areal.v2.agent_service.worker.memory_agent_host import (
    AuthorizedMemoryAgentWorkerHost,
    MemoryAgentWorkerSessionCloseOutcomeV1,
    MemoryAgentWorkerSessionCloseReceiptV1,
    MemoryAgentWorkerSessionReservationV1,
)
from areal.v2.memory_service.types import MemoryScope

_NOW = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)
_RESOLVER_VERSION = sha256(b"memory-agent-host-resolver-v1").hexdigest()
_RESOLVER_CONFIG = sha256(b"memory-agent-host-resolver-config-v1").hexdigest()


class _UnusedReleaseStore:
    def resolve_active_assignment(self, *args: object) -> object:
        raise AssertionError("host tests never resolve release assignments")


class _UnusedRuntimeStore:
    def begin_query(self, *args: object) -> object:
        raise AssertionError("host tests never expose Memory")

    resolve_query = begin_query
    prepare_delivery = begin_query
    submit_delivery = begin_query


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


def _principal(suffix: str = "1") -> MemoryPrincipalV1:
    return MemoryPrincipalV1(
        issuer="https://identity.example",
        subject=f"principal-{suffix}",
    )


class _Resolver:
    resolver_id = "test-memory-agent-host-policy"
    resolver_version_sha256 = _RESOLVER_VERSION
    resolver_config_sha256 = _RESOLVER_CONFIG

    def resolve_active_grant(
        self,
        request: MemoryScopeGrantRequestV1,
    ) -> MemoryScopeGrantV1:
        assert request.action in {
            MemoryScopeActionV1.PIN_ASSIGNMENT,
            MemoryScopeActionV1.EXPOSE_MEMORY,
        }
        return MemoryScopeGrantV1.create(
            request=request,
            resolver_id=self.resolver_id,
            resolver_version_sha256=self.resolver_version_sha256,
            resolver_config_sha256=self.resolver_config_sha256,
            valid_from=_NOW - timedelta(minutes=1),
            valid_until=_NOW + timedelta(minutes=1),
            evaluated_at=_NOW - timedelta(minutes=2),
            granted_at=_NOW - timedelta(minutes=2),
            idempotency_key=f"grant:{request.session.incarnation_id}:{request.action}",
        )


class _Coordinator(AsyncMemoryAgentCoordinator):
    def __init__(self, order: list[str] | None = None) -> None:
        super().__init__(
            _UnusedReleaseStore(),  # type: ignore[arg-type]
            _UnusedRuntimeStore(),  # type: ignore[arg-type]
            max_workers=1,
            max_pending_calls=1,
        )
        self.pin_calls: list[tuple[str, MemoryAgentSessionPinV1]] = []
        self.start_calls: list[tuple[str, str]] = []
        self.close_calls: list[str] = []
        self.close_started = asyncio.Event()
        self.close_release = asyncio.Event()
        self.close_release.set()
        self.close_failures = 0
        self.order = [] if order is None else order

    async def pin_session(  # type: ignore[override]
        self,
        session_key: str,
        pin: MemoryAgentSessionPinV1,
    ) -> object:
        self.pin_calls.append((session_key, pin))
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
            memory_trajectory_id=f"mtraj_{_hash(f'{session_key}:{turn_idempotency_key}')}",
        )

    async def close_session(self, session_key: str) -> None:
        self.close_calls.append(session_key)
        self.order.append("memory_session_close")
        self.close_started.set()
        await self.close_release.wait()
        if self.close_failures:
            self.close_failures -= 1
            raise RuntimeError("injected Memory close failure")


class _Emitter:
    async def emit_delta(self, text: str) -> None:
        del text

    async def emit_tool_call(self, name: str, args: str) -> None:
        del name, args

    async def emit_tool_result(self, name: str, result: str) -> None:
        del name, result


class _Agent:
    def __init__(self, order: list[str] | None = None) -> None:
        self.sessions: dict[str, str] = {}
        self.run_calls: list[tuple[str, str]] = []
        self.run_started = asyncio.Event()
        self.run_release = asyncio.Event()
        self.run_release.set()
        self.run_cancelled = asyncio.Event()
        self.run_cancel_failure = False
        self.close_calls: list[str] = []
        self.close_started = asyncio.Event()
        self.close_release = asyncio.Event()
        self.close_release.set()
        self.close_failures = 0
        self.close_all_calls = 0
        self.order = [] if order is None else order

    async def run(
        self,
        request: AgentRequest,
        *,
        emitter: EventEmitter,
    ) -> AgentResponse | StreamResponse:
        del emitter
        self.run_calls.append((request.session_key, request.run_id))
        self.sessions[request.session_key] = request.run_id
        self.run_started.set()
        try:
            await self.run_release.wait()
        except asyncio.CancelledError:
            self.run_cancelled.set()
            if self.run_cancel_failure:
                raise RuntimeError("injected Agent cancellation cleanup failure")
            raise
        if request.metadata.get("stream"):

            async def body():
                try:
                    yield b"first"
                    await request.metadata["stream_release"].wait()
                    yield b"second"
                finally:
                    self.order.append("agent_source_finally")

            return StreamResponse(
                status_code=200,
                headers={"content-type": "text/event-stream"},
                body=body(),
            )
        return AgentResponse(summary=request.run_id)

    async def close_session(self, session_key: str) -> None:
        self.close_calls.append(session_key)
        self.order.append("agent_session_close")
        self.close_started.set()
        await self.close_release.wait()
        if self.close_failures:
            self.close_failures -= 1
            raise RuntimeError("injected Agent close failure")
        self.sessions.pop(session_key, None)

    async def close_all_sessions(self) -> None:
        self.close_all_calls += 1
        self.order.append("agent_close_all")
        self.sessions.clear()


class _RunOnlyAgent:
    async def run(
        self,
        request: AgentRequest,
        *,
        emitter: EventEmitter,
    ) -> AgentResponse:
        del request, emitter
        return AgentResponse(summary="ok")


def _host(
    *,
    max_retired_sessions: int = 4096,
) -> tuple[AuthorizedMemoryAgentWorkerHost, _Agent, _Coordinator]:
    order: list[str] = []
    coordinator = _Coordinator(order)
    broker = AuthorizedMemoryAgentBroker(
        coordinator,
        MemoryScopeGrantAuthorizer(_Resolver(), clock=lambda: _NOW),
    )
    agent = _Agent(order)
    return (
        AuthorizedMemoryAgentWorkerHost(
            broker,
            agent,
            max_retired_sessions=max_retired_sessions,
        ),
        agent,
        coordinator,
    )


def _request(
    session_key: str,
    run_id: str,
    *,
    stream_release: asyncio.Event | None = None,
) -> AgentRequest:
    metadata: dict[str, object] = {}
    if stream_release is not None:
        metadata = {"stream": True, "stream_release": stream_release}
    return AgentRequest(
        message="test",
        session_key=session_key,
        run_id=run_id,
        metadata=metadata,
    )


async def _run(
    host: AuthorizedMemoryAgentWorkerHost,
    reservation: MemoryAgentWorkerSessionReservationV1,
    run_id: str,
    *,
    stream_release: asyncio.Event | None = None,
) -> AgentResponse | StreamResponse:
    return await host.run_agent(
        reservation,
        _request(
            reservation.session_key,
            run_id,
            stream_release=stream_release,
        ),
        assignment_pin=_pin(),
        emitter=_Emitter(),
    )


def test_full_host_lifecycle_values_are_strict_and_detached() -> None:
    source_session = MemorySessionIncarnationV1(
        session_key="session-1",
        incarnation_id=f"msinc_{'1' * 64}",
    )
    source_audience = MemoryWorkerAudienceV1(f"maud_{'2' * 64}")
    reservation = MemoryAgentWorkerSessionReservationV1(
        session=source_session,
        audience=source_audience,
    )
    receipt = MemoryAgentWorkerSessionCloseReceiptV1(
        identity=reservation.identity,
        outcome=MemoryAgentWorkerSessionCloseOutcomeV1.CLOSED,
    )

    assert reservation.session is not source_session
    assert reservation.audience is not source_audience
    assert reservation.session_key == "session-1"
    assert receipt.identity == reservation.identity
    assert receipt.identity is not reservation.identity
    assert receipt.outcome is MemoryAgentWorkerSessionCloseOutcomeV1.CLOSED
    assert set(receipt.__dataclass_fields__) == {  # type: ignore[attr-defined]
        "identity",
        "outcome",
    }

    with pytest.raises(TypeError, match="outcome"):
        MemoryAgentWorkerSessionCloseReceiptV1(  # type: ignore[arg-type]
            identity=reservation.identity,
            outcome="closed",
        )


@pytest.mark.asyncio
async def test_exact_host_rejects_agent_without_per_session_cleanup() -> None:
    coordinator = _Coordinator()
    broker = AuthorizedMemoryAgentBroker(
        coordinator,
        MemoryScopeGrantAuthorizer(_Resolver(), clock=lambda: _NOW),
    )

    with pytest.raises(TypeError, match="requires agent.close_session"):
        AuthorizedMemoryAgentWorkerHost(broker, _RunOnlyAgent())
    await broker.aclose()


@pytest.mark.asyncio
async def test_exact_host_claim_is_exclusive_until_successful_shutdown() -> None:
    agent = _Agent()
    first_coordinator = _Coordinator()
    first_broker = AuthorizedMemoryAgentBroker(
        first_coordinator,
        MemoryScopeGrantAuthorizer(_Resolver(), clock=lambda: _NOW),
    )
    second_coordinator = _Coordinator()
    second_broker = AuthorizedMemoryAgentBroker(
        second_coordinator,
        MemoryScopeGrantAuthorizer(_Resolver(), clock=lambda: _NOW),
    )
    first = AuthorizedMemoryAgentWorkerHost(first_broker, agent)

    with pytest.raises(MemoryAgentSessionConflictError, match="already belongs"):
        AuthorizedMemoryAgentWorkerHost(second_broker, agent)

    await first.aclose()
    second = AuthorizedMemoryAgentWorkerHost(second_broker, agent)
    await second.aclose()


@pytest.mark.asyncio
async def test_memory_close_cannot_release_key_before_agent_hook() -> None:
    host, agent, coordinator = _host()
    original = await host.reserve_session(_principal(), "same-key")
    response = await _run(host, original, "run-a")
    assert isinstance(response, AgentResponse)
    assert agent.sessions == {"same-key": "run-a"}

    agent.close_release.clear()
    closing = asyncio.create_task(host.close_session(original))
    await asyncio.wait_for(coordinator.close_started.wait(), timeout=1)
    await asyncio.wait_for(agent.close_started.wait(), timeout=1)

    with pytest.raises(MemoryAgentSessionConflictError, match="closing"):
        await host.reserve_session(_principal(), "same-key")
    assert not closing.done()

    agent.close_release.set()
    closed = await asyncio.wait_for(closing, timeout=1)
    assert closed == MemoryAgentWorkerSessionCloseReceiptV1(
        identity=original.identity,
        outcome=MemoryAgentWorkerSessionCloseOutcomeV1.CLOSED,
    )

    replacement = await host.reserve_session(_principal(), "same-key")
    assert replacement.identity != original.identity
    await _run(host, replacement, "run-b")
    replayed = await host.close_session(original)

    assert replayed == closed
    assert agent.close_calls == ["same-key"]
    assert coordinator.close_calls == ["same-key"]
    assert agent.sessions == {"same-key": "run-b"}

    await host.close_session(replacement)
    await host.aclose()


@pytest.mark.asyncio
async def test_stream_body_holds_outer_turn_until_explicit_close() -> None:
    host, agent, coordinator = _host()
    reservation = await host.reserve_session(_principal(), "session-1")
    stream_release = asyncio.Event()
    response = await _run(
        host,
        reservation,
        "stream-a",
        stream_release=stream_release,
    )
    assert isinstance(response, StreamResponse)
    assert await anext(response.body) == b"first"

    closing = asyncio.create_task(host.close_session(reservation))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert not coordinator.close_started.is_set()
    assert agent.close_calls == []

    await response.body.aclose()
    closed = await asyncio.wait_for(closing, timeout=1)
    assert closed.outcome is MemoryAgentWorkerSessionCloseOutcomeV1.CLOSED
    assert coordinator.close_calls == ["session-1"]
    assert agent.close_calls == ["session-1"]
    assert agent.order[:3] == [
        "agent_source_finally",
        "memory_session_close",
        "agent_session_close",
    ]
    await host.aclose()


@pytest.mark.asyncio
async def test_wrong_identity_is_side_effect_free_and_does_not_reveal_current() -> None:
    host, agent, coordinator = _host()
    current = await host.reserve_session(_principal(), "session-1")
    wrong = MemoryWorkerSessionIdentityV1(
        session=MemorySessionIncarnationV1(
            session_key=current.session_key,
            incarnation_id=f"msinc_{'f' * 64}",
        ),
        audience=current.audience,
    )

    receipt = await host.close_session_if_current(wrong)

    assert receipt == MemoryAgentWorkerSessionCloseReceiptV1(
        identity=wrong,
        outcome=MemoryAgentWorkerSessionCloseOutcomeV1.NOT_CURRENT,
    )
    assert receipt.identity != current.identity
    assert agent.close_calls == []
    assert coordinator.close_calls == []
    await _run(host, current, "still-current")
    await host.close_session(current)
    await host.aclose()


@pytest.mark.asyncio
async def test_agent_hook_failure_quarantines_without_repeating_unknown_effects() -> (
    None
):
    host, agent, coordinator = _host()
    reservation = await host.reserve_session(_principal(), "session-1")
    await _run(host, reservation, "run-a")
    agent.close_failures = 1

    with pytest.raises(RuntimeError, match="injected Agent close failure"):
        await host.close_session(reservation)

    with pytest.raises(RuntimeError, match="injected Agent close failure"):
        await host.close_session(reservation)
    with pytest.raises(MemoryAgentSessionConflictError, match="quarantined"):
        await host.reserve_session(_principal(), "session-1")

    assert coordinator.close_calls == ["session-1"]
    assert agent.close_calls == ["session-1"]
    assert agent.sessions == {"session-1": "run-a"}
    with pytest.raises(RuntimeError, match="injected Agent close failure"):
        await host.aclose()
    assert agent.close_all_calls == 1
    assert agent.sessions == {}

    # Even though close_all happened to clean this fake Agent, the host cannot
    # infer that arbitrary plugin cleanup succeeded after a primary close error.
    # Retaining the process claim prevents uncertain state from being reattached.
    other_coordinator = _Coordinator()
    other_broker = AuthorizedMemoryAgentBroker(
        other_coordinator,
        MemoryScopeGrantAuthorizer(_Resolver(), clock=lambda: _NOW),
    )
    with pytest.raises(MemoryAgentSessionConflictError, match="already belongs"):
        AuthorizedMemoryAgentWorkerHost(other_broker, agent)
    await other_broker.aclose()


@pytest.mark.asyncio
async def test_cancelled_close_waiter_cannot_cancel_owned_full_close() -> None:
    host, agent, coordinator = _host()
    reservation = await host.reserve_session(_principal(), "session-1")
    agent.close_release.clear()

    first = asyncio.create_task(host.close_session(reservation))
    await asyncio.wait_for(agent.close_started.wait(), timeout=1)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first

    agent.close_release.set()
    replayed = await asyncio.wait_for(host.close_session(reservation), timeout=1)
    assert replayed.outcome is MemoryAgentWorkerSessionCloseOutcomeV1.CLOSED
    assert coordinator.close_calls == ["session-1"]
    assert agent.close_calls == ["session-1"]
    await host.aclose()


@pytest.mark.asyncio
async def test_concurrent_object_and_identity_close_share_one_owned_task() -> None:
    host, agent, coordinator = _host()
    reservation = await host.reserve_session(_principal(), "session-1")
    agent.close_release.clear()

    object_close = asyncio.create_task(host.close_session(reservation))
    await asyncio.wait_for(agent.close_started.wait(), timeout=1)
    identity_close = asyncio.create_task(
        host.close_session_if_current(reservation.identity)
    )
    await asyncio.sleep(0)
    assert not object_close.done()
    assert not identity_close.done()

    agent.close_release.set()
    first, second = await asyncio.gather(object_close, identity_close)
    assert first == second
    assert first.outcome is MemoryAgentWorkerSessionCloseOutcomeV1.CLOSED
    assert coordinator.close_calls == ["session-1"]
    assert agent.close_calls == ["session-1"]
    await host.aclose()


@pytest.mark.asyncio
async def test_memory_close_failure_retries_before_agent_hook() -> None:
    host, agent, coordinator = _host()
    reservation = await host.reserve_session(_principal(), "session-1")
    coordinator.close_failures = 1

    with pytest.raises(RuntimeError, match="injected Memory close failure"):
        await host.close_session(reservation)
    assert agent.close_calls == []
    with pytest.raises(MemoryAgentSessionConflictError, match="closing"):
        await _run(host, reservation, "late-turn")

    closed = await host.close_session(reservation)
    assert closed.outcome is MemoryAgentWorkerSessionCloseOutcomeV1.CLOSED
    assert coordinator.close_calls == ["session-1", "session-1"]
    assert agent.close_calls == ["session-1"]
    await host.aclose()


@pytest.mark.asyncio
async def test_outer_retirement_lru_eviction_never_resolves_old_a_to_b() -> None:
    host, agent, coordinator = _host(max_retired_sessions=1)
    original = await host.reserve_session(_principal(), "same-key")
    await _run(host, original, "run-a")
    await host.close_session(original)
    evicting = await host.reserve_session(_principal(), "other-key")
    await host.close_session(evicting)

    replacement = await host.reserve_session(_principal(), "same-key")
    await _run(host, replacement, "run-b")
    stale = await host.close_session_if_current(original.identity)

    assert stale == MemoryAgentWorkerSessionCloseReceiptV1(
        identity=original.identity,
        outcome=MemoryAgentWorkerSessionCloseOutcomeV1.NOT_CURRENT,
    )
    with pytest.raises(MemoryAgentSessionConflictError, match="not current"):
        await host.close_session(original)
    assert agent.sessions == {"same-key": "run-b"}
    assert coordinator.close_calls == ["same-key", "other-key"]
    assert agent.close_calls == ["same-key", "other-key"]

    await host.close_session(replacement)
    await host.aclose()


@pytest.mark.asyncio
async def test_host_identity_collision_quarantines_hidden_runtime_successor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host, agent, coordinator = _host(max_retired_sessions=2)
    original = await host.reserve_session(_principal(), "same-key")
    await host.close_session(original)
    second = await host.reserve_session(_principal(), "second-key")
    await host.close_session(second)

    # Move A only in the outer full-host LRU.  The private Memory LRU still
    # considers A oldest, so closing a third session evicts Memory A while the
    # outer host deliberately retains it.
    await host.close_session(original)
    third = await host.reserve_session(_principal(), "third-key")
    await host.close_session(third)

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
    with pytest.raises(MemoryAgentSessionConflictError, match="retained full-host"):
        await host.reserve_session(_principal(), "same-key")
    with pytest.raises(MemoryAgentSessionConflictError, match="quarantined"):
        await host.reserve_session(_principal(), "same-key")

    assert coordinator.close_calls == ["same-key", "second-key", "third-key"]
    assert agent.close_calls == ["same-key", "second-key", "third-key"]
    await host.aclose()
    # The collided Memory reservation was never published to the Agent host;
    # only owned runtime shutdown can drain it.
    assert coordinator.close_calls == [
        "same-key",
        "second-key",
        "third-key",
        "same-key",
    ]


@pytest.mark.asyncio
async def test_cancelled_reserve_waiter_rejoins_one_owned_publication() -> None:
    host, _, _ = _host()
    runtime = getattr(host, "_AuthorizedMemoryAgentWorkerHost__runtime")
    original_reserve = runtime.reserve_session
    reserve_started = asyncio.Event()
    reserve_release = asyncio.Event()
    reserve_calls = 0

    async def gated_reserve(principal: object, session_key: str):
        nonlocal reserve_calls
        reserve_calls += 1
        reserve_started.set()
        await reserve_release.wait()
        return await original_reserve(principal, session_key)

    runtime.reserve_session = gated_reserve  # type: ignore[method-assign]
    abandoned = asyncio.create_task(host.reserve_session(_principal(), "session-1"))
    await asyncio.wait_for(reserve_started.wait(), timeout=1)
    abandoned.cancel()
    with pytest.raises(asyncio.CancelledError):
        await abandoned

    rejoined = asyncio.create_task(host.reserve_session(_principal(), "session-1"))
    reserve_release.set()
    reservation = await asyncio.wait_for(rejoined, timeout=1)
    assert reservation.session_key == "session-1"
    assert reserve_calls == 1
    assert await host.reserve_session(_principal(), "session-1") is reservation
    await host.aclose()


@pytest.mark.asyncio
async def test_shutdown_closes_abandoned_stream_before_session_hook() -> None:
    host, agent, coordinator = _host()
    reservation = await host.reserve_session(_principal(), "session-1")
    response = await _run(
        host,
        reservation,
        "stream-a",
        stream_release=asyncio.Event(),
    )
    assert isinstance(response, StreamResponse)
    assert await anext(response.body) == b"first"

    await asyncio.wait_for(host.aclose(), timeout=1)

    assert coordinator.close_calls == ["session-1"]
    assert agent.close_calls == ["session-1"]
    assert agent.close_all_calls == 1
    assert agent.order == [
        "agent_source_finally",
        "memory_session_close",
        "agent_session_close",
        "agent_close_all",
    ]
    with pytest.raises(StopAsyncIteration):
        await anext(response.body)


@pytest.mark.asyncio
async def test_shutdown_cancels_inflight_agent_execution_then_drains() -> None:
    host, agent, coordinator = _host()
    reservation = await host.reserve_session(_principal(), "session-1")
    agent.run_release.clear()
    running = asyncio.create_task(_run(host, reservation, "blocked-run"))
    await asyncio.wait_for(agent.run_started.wait(), timeout=1)

    await asyncio.wait_for(host.aclose(), timeout=1)

    with pytest.raises(asyncio.CancelledError):
        await running
    assert agent.run_cancelled.is_set()
    assert coordinator.close_calls == ["session-1"]
    assert agent.close_calls == ["session-1"]
    assert agent.close_all_calls == 1


@pytest.mark.asyncio
async def test_shutdown_retains_agent_claim_after_execution_cleanup_failure() -> None:
    host, agent, coordinator = _host()
    reservation = await host.reserve_session(_principal(), "session-1")
    agent.run_release.clear()
    agent.run_cancel_failure = True
    running = asyncio.create_task(_run(host, reservation, "blocked-run"))
    await asyncio.wait_for(agent.run_started.wait(), timeout=1)

    with pytest.raises(
        RuntimeError,
        match="injected Agent cancellation cleanup failure",
    ):
        await asyncio.wait_for(host.aclose(), timeout=1)
    with pytest.raises(
        RuntimeError,
        match="injected Agent cancellation cleanup failure",
    ):
        await running

    assert agent.run_cancelled.is_set()
    assert coordinator.close_calls == ["session-1"]
    assert agent.close_calls == ["session-1"]
    assert agent.close_all_calls == 1

    other_coordinator = _Coordinator()
    other_broker = AuthorizedMemoryAgentBroker(
        other_coordinator,
        MemoryScopeGrantAuthorizer(_Resolver(), clock=lambda: _NOW),
    )
    with pytest.raises(MemoryAgentSessionConflictError, match="already belongs"):
        AuthorizedMemoryAgentWorkerHost(other_broker, agent)
    await other_broker.aclose()


@pytest.mark.asyncio
async def test_shutdown_rejoins_inflight_agent_hook_without_repeating_it() -> None:
    host, agent, coordinator = _host()
    reservation = await host.reserve_session(_principal(), "session-1")
    agent.close_release.clear()
    closing = asyncio.create_task(host.close_session(reservation))
    await asyncio.wait_for(agent.close_started.wait(), timeout=1)

    shutdown = asyncio.create_task(host.aclose())
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert not closing.done()
    assert not shutdown.done()
    assert coordinator.close_calls == ["session-1"]
    assert agent.close_calls == ["session-1"]

    agent.close_release.set()
    receipt = await asyncio.wait_for(closing, timeout=1)
    await asyncio.wait_for(shutdown, timeout=1)
    assert receipt.outcome is MemoryAgentWorkerSessionCloseOutcomeV1.CLOSED
    assert coordinator.close_calls == ["session-1"]
    assert agent.close_calls == ["session-1"]
    assert agent.close_all_calls == 1
