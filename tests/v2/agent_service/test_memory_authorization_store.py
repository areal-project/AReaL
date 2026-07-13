# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta
from hashlib import sha256

import pytest

from tests.v2.agent_service.test_authorized_memory_broker import _Coordinator
from tests.v2.agent_service.test_memory_authorization import (
    _NOW,
    _authority_variants,
    _hash,
    _pin,
    _request,
    _scope,
    _target,
)

from areal.v2.agent_service.memory_authorization import (
    MemoryAssignmentGrantTargetV1,
    MemoryScopeActionV1,
    MemoryScopeAuthorizationDeniedError,
    MemoryScopeGrantAuthorizer,
    MemoryScopeGrantRequestV1,
)
from areal.v2.agent_service.memory_authorization_store import (
    InMemoryMemoryScopeGrantStore,
    MemoryScopeGrantConflictError,
    MemoryScopeGrantNotFoundError,
    MemoryScopeGrantRevocationReasonV1,
)
from areal.v2.agent_service.memory_broker import AuthorizedMemoryAgentBroker
from areal.v2.agent_service.worker.memory import WorkerMemoryTurnCapability


class _Clock:
    def __init__(self, value: datetime = _NOW) -> None:
        self.value = value
        self.block = False
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self._lock = threading.Lock()

    def __call__(self) -> datetime:
        with self._lock:
            value = self.value
            block = self.block
        if block:
            self.started.set()
            assert self.release.wait(timeout=5)
        return value

    def set(self, value: datetime) -> None:
        with self._lock:
            self.value = value


def _store(
    clock: _Clock | None = None,
) -> tuple[InMemoryMemoryScopeGrantStore, _Clock]:
    clock = clock or _Clock()
    return (
        InMemoryMemoryScopeGrantStore(
            resolver_id="test-memory-scope-grant-store",
            resolver_version_sha256=_hash("grant-store-version-v1"),
            resolver_config_sha256=_hash("grant-store-config-v1"),
            revoker_id="test-memory-scope-grant-revoker",
            revoker_version_sha256=_hash("grant-revoker-version-v1"),
            revoker_config_sha256=_hash("grant-revoker-config-v1"),
            clock=clock,
        ),
        clock,
    )


def _create(
    store: InMemoryMemoryScopeGrantStore,
    request: MemoryScopeGrantRequestV1 | None = None,
    *,
    valid_from: datetime | None = None,
    valid_until: datetime | None = None,
    idempotency_key: str = "grant-create-1",
):
    return store.create_grant(
        request or _request(),
        valid_from=valid_from or (_NOW - timedelta(minutes=1)),
        valid_until=valid_until or (_NOW + timedelta(minutes=1)),
        idempotency_key=idempotency_key,
    )


def _revoke(
    store: InMemoryMemoryScopeGrantStore,
    grant,
    *,
    reason: MemoryScopeGrantRevocationReasonV1 = (
        MemoryScopeGrantRevocationReasonV1.OPERATOR
    ),
    reason_detail_sha256: str | None = None,
    idempotency_key: str = "grant-revoke-1",
):
    return store.revoke_grant(
        grant.request.target.scope,
        grant.grant_id,
        grant_content_sha256=grant.content_hash,
        reason=reason,
        reason_detail_sha256=reason_detail_sha256,
        idempotency_key=idempotency_key,
    )


def test_real_authorizer_resolves_only_the_exact_request() -> None:
    store, clock = _store()
    request = _request()
    created = _create(store, request)
    authorizer = MemoryScopeGrantAuthorizer(store, clock=clock)

    resolved = authorizer.authorize(request)
    assert resolved == created
    assert resolved is not created
    assert resolved.request is not created.request
    assert resolved.request.target.scope is not created.request.target.scope

    for variant in _authority_variants(request):
        with pytest.raises(MemoryScopeAuthorizationDeniedError):
            authorizer.authorize(variant)


def test_create_idempotency_is_scoped_exact_and_lifetime_single_use() -> None:
    store, _ = _store()
    request = _request()
    created = _create(store, request)

    retry = _create(store, request)
    assert retry == created
    assert retry is not created
    with pytest.raises(MemoryScopeGrantConflictError, match="different request"):
        _create(
            store,
            request,
            valid_until=_NOW + timedelta(minutes=2),
        )
    with pytest.raises(MemoryScopeGrantConflictError, match="different request"):
        _create(store, replace(request, action=MemoryScopeActionV1.PIN_ASSIGNMENT))
    with pytest.raises(MemoryScopeGrantConflictError, match="lifetime history"):
        _create(store, request, idempotency_key="another-create")

    _revoke(store, created)
    assert _create(store, request) == created
    with pytest.raises(MemoryScopeGrantConflictError, match="lifetime history"):
        _create(
            store,
            request,
            idempotency_key="regrant-after-revoke",
        )
    with pytest.raises(MemoryScopeAuthorizationDeniedError):
        store.resolve_active_grant(request)


def test_idempotency_is_scoped_but_exact_request_history_is_not_renewable() -> None:
    store, clock = _store()
    first_request = _request()
    second_request = _request(target=_target(scope=_scope(suffix="2"), suffix="2"))
    first = _create(store, first_request)
    second = _create(store, second_request)
    assert first.idempotency_key == second.idempotency_key
    assert first.content_hash != second.content_hash

    clock.set(first.valid_until)
    with pytest.raises(MemoryScopeAuthorizationDeniedError):
        store.resolve_active_grant(first_request)
    with pytest.raises(MemoryScopeGrantConflictError, match="lifetime history"):
        _create(
            store,
            first_request,
            idempotency_key="regrant-after-expiry",
        )


def test_revocation_idempotency_is_scoped() -> None:
    store, _ = _store()
    first = _create(store)
    second = _create(
        store,
        _request(target=_target(scope=_scope(suffix="2"), suffix="2")),
    )

    first_revocation = _revoke(store, first)
    second_revocation = _revoke(store, second)
    assert first_revocation.idempotency_key == second_revocation.idempotency_key
    assert first_revocation.content_hash != second_revocation.content_hash


def test_revocation_is_exact_idempotent_and_audit_only() -> None:
    store, _ = _store()
    request = _request()
    grant = _create(store, request)
    revocation = _revoke(store, grant)

    assert revocation.grant_id == grant.grant_id
    assert revocation.grant_content_sha256 == grant.content_hash
    assert revocation.scope == request.target.scope
    assert (
        revocation.request_content_sha256
        == sha256(request.canonical_bytes()).hexdigest()
    )
    retry = _revoke(store, grant)
    assert retry == revocation
    assert retry is not revocation
    with pytest.raises(MemoryScopeGrantConflictError, match="already"):
        _revoke(store, grant, idempotency_key="another-revocation")
    with pytest.raises(MemoryScopeAuthorizationDeniedError):
        store.resolve_active_grant(request)

    audited_grant = store.get_grant_for_audit(
        request.target.scope,
        grant.grant_id,
        grant_content_sha256=grant.content_hash,
    )
    audited_revocation = store.get_grant_revocation_for_audit(
        request.target.scope,
        grant.grant_id,
        grant_content_sha256=grant.content_hash,
    )
    assert audited_grant == grant
    assert audited_revocation == revocation
    assert audited_grant is not grant
    assert audited_revocation is not revocation

    unknown_hash = _hash("unknown-grant")
    with pytest.raises(MemoryScopeGrantNotFoundError):
        store.get_grant_for_audit(
            request.target.scope,
            f"msgr_{unknown_hash[:24]}",
            grant_content_sha256=unknown_hash,
        )


def test_control_addresses_require_the_exact_scope_id_and_full_hash() -> None:
    store, _ = _store()
    grant = _create(store)
    wrong_scope = _scope(suffix="2")

    with pytest.raises(MemoryScopeGrantNotFoundError):
        store.get_grant_for_audit(
            wrong_scope,
            grant.grant_id,
            grant_content_sha256=grant.content_hash,
        )
    with pytest.raises(MemoryScopeGrantNotFoundError):
        store.revoke_grant(
            wrong_scope,
            grant.grant_id,
            grant_content_sha256=grant.content_hash,
            reason=MemoryScopeGrantRevocationReasonV1.OPERATOR,
            idempotency_key="wrong-scope-revocation",
        )
    with pytest.raises(MemoryScopeGrantConflictError, match="disagrees"):
        store.get_grant_for_audit(
            grant.request.target.scope,
            grant.grant_id,
            grant_content_sha256=_hash("different-full-hash"),
        )
    assert store.resolve_active_grant(_request()) == grant


def test_other_revocation_reason_requires_a_nonsecret_detail_digest() -> None:
    store, _ = _store()
    grant = _create(store)
    with pytest.raises(ValueError, match="requires"):
        _revoke(
            store,
            grant,
            reason=MemoryScopeGrantRevocationReasonV1.OTHER,
        )
    detail = _hash("external incident record")
    revocation = _revoke(
        store,
        grant,
        reason=MemoryScopeGrantRevocationReasonV1.OTHER,
        reason_detail_sha256=detail,
    )
    assert revocation.reason_detail_sha256 == detail
    assert not hasattr(revocation, "reason_detail")


def test_half_open_validity_and_clock_rollback_fail_closed() -> None:
    clock = _Clock(_NOW)
    store, _ = _store(clock)
    request = _request()
    grant = _create(
        store,
        request,
        valid_from=_NOW,
        valid_until=_NOW + timedelta(seconds=3),
    )

    assert store.resolve_active_grant(request) == grant
    clock.set(grant.valid_until - timedelta(microseconds=1))
    assert store.resolve_active_grant(request) == grant
    clock.set(grant.valid_until)
    with pytest.raises(MemoryScopeAuthorizationDeniedError):
        store.resolve_active_grant(request)

    clock.set(grant.valid_from)
    with pytest.raises(MemoryScopeGrantConflictError, match="backwards"):
        store.resolve_active_grant(request)


def test_inactive_window_is_not_published_and_can_be_corrected() -> None:
    store, _ = _store()
    request = _request()
    with pytest.raises(MemoryScopeGrantConflictError, match="creation time"):
        _create(
            store,
            request,
            valid_from=_NOW - timedelta(minutes=2),
            valid_until=_NOW,
        )
    with pytest.raises(MemoryScopeGrantConflictError, match="creation time"):
        _create(
            store,
            request,
            valid_from=_NOW + timedelta(minutes=1),
            valid_until=_NOW + timedelta(minutes=2),
        )
    corrected = _create(store, request)
    assert store.resolve_active_grant(request) == corrected


def test_input_and_every_return_value_are_detached_from_private_state() -> None:
    store, _ = _store()
    original = _request()
    grant = _create(store, original)
    expected_tenant = grant.request.target.scope.tenant_id

    object.__setattr__(original.target.scope, "tenant_id", "input-poison")
    object.__setattr__(grant.request.target.scope, "tenant_id", "create-poison")
    first = store.resolve_active_grant(_request())
    assert first.request.target.scope.tenant_id == expected_tenant

    object.__setattr__(first.request.target.scope, "tenant_id", "resolve-poison")
    audit = store.get_grant_for_audit(
        _request().target.scope,
        first.grant_id,
        grant_content_sha256=first.content_hash,
    )
    assert audit.request.target.scope.tenant_id == expected_tenant
    object.__setattr__(audit.request.target.scope, "tenant_id", "audit-poison")
    assert store.resolve_active_grant(_request()).request.target.scope.tenant_id == (
        expected_tenant
    )

    revocation = _revoke(store, store.resolve_active_grant(_request()))
    object.__setattr__(
        revocation, "reason", MemoryScopeGrantRevocationReasonV1.SECURITY
    )
    audited = store.get_grant_revocation_for_audit(
        _request().target.scope,
        first.grant_id,
        grant_content_sha256=first.content_hash,
    )
    assert audited.reason is MemoryScopeGrantRevocationReasonV1.OPERATOR


def test_concurrent_create_calls_publish_one_canonical_history() -> None:
    store, _ = _store()
    request = _request()
    with ThreadPoolExecutor(max_workers=8) as executor:
        grants = tuple(
            executor.map(
                lambda _: _create(store, request),
                range(32),
            )
        )
    assert all(grant == grants[0] for grant in grants)
    assert len({grant.content_hash for grant in grants}) == 1

    racing, _ = _store()
    barrier = threading.Barrier(8)

    def create_with_key(index: int):
        barrier.wait(timeout=5)
        try:
            return _create(
                racing,
                request,
                idempotency_key=f"racing-key-{index}",
            )
        except MemoryScopeGrantConflictError as error:
            return error

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = tuple(executor.map(create_with_key, range(8)))
    assert sum(not isinstance(item, Exception) for item in results) == 1
    assert sum(isinstance(item, MemoryScopeGrantConflictError) for item in results) == 7


def test_resolve_and_revoke_have_one_lock_linearization_order() -> None:
    store, clock = _store()
    request = _request()
    grant = _create(store, request)

    clock.block = True
    clock.started.clear()
    clock.release.clear()
    with ThreadPoolExecutor(max_workers=2) as executor:
        resolving = executor.submit(store.resolve_active_grant, request)
        assert clock.started.wait(timeout=5)
        revoking = executor.submit(
            store.revoke_grant,
            grant.request.target.scope,
            grant.grant_id,
            grant_content_sha256=grant.content_hash,
            reason=MemoryScopeGrantRevocationReasonV1.OPERATOR,
            idempotency_key="linearized-revoke",
        )
        assert not revoking.done()
        clock.block = False
        clock.release.set()
        assert resolving.result(timeout=5) == grant
        revoking.result(timeout=5)

    with pytest.raises(MemoryScopeAuthorizationDeniedError):
        store.resolve_active_grant(request)


class _InjectedPublicationFailure(BaseException):
    pass


class _FailingDict(dict):
    def __init__(self) -> None:
        super().__init__()
        self.fail = True

    def __setitem__(self, key, value) -> None:
        dict.__setitem__(self, key, value)
        if self.fail:
            raise _InjectedPublicationFailure


def test_atomic_publish_rolls_back_grant_and_revocation_indexes() -> None:
    store, _ = _store()
    request = _request()
    failing_grants = _FailingDict()
    store._InMemoryMemoryScopeGrantStore__grant_hash_by_idempotency = failing_grants
    with pytest.raises(_InjectedPublicationFailure):
        _create(store, request)
    assert store._InMemoryMemoryScopeGrantStore__grant_by_hash == {}
    assert store._InMemoryMemoryScopeGrantStore__grant_hash_by_request == {}
    assert failing_grants == {}
    assert store._InMemoryMemoryScopeGrantStore__grant_hash_by_display_id == {}

    failing_grants.fail = False
    grant = _create(store, request)
    failing_revocations = _FailingDict()
    store._InMemoryMemoryScopeGrantStore__revocation_hash_by_idempotency = (
        failing_revocations
    )
    with pytest.raises(_InjectedPublicationFailure):
        _revoke(store, grant)
    assert store._InMemoryMemoryScopeGrantStore__revocation_by_hash == {}
    assert store._InMemoryMemoryScopeGrantStore__revocation_hash_by_grant_hash == {}
    assert failing_revocations == {}
    assert store._InMemoryMemoryScopeGrantStore__revocation_hash_by_display_id == {}

    failing_revocations.fail = False
    revocation = _revoke(store, grant)
    assert revocation.grant_content_sha256 == grant.content_hash


def test_single_sided_address_index_damage_fails_as_conflict() -> None:
    store, _ = _store()
    grant = _create(store)
    store._InMemoryMemoryScopeGrantStore__grant_hash_by_display_id = {}

    with pytest.raises(MemoryScopeGrantConflictError, match="indexes"):
        store.get_grant_for_audit(
            grant.request.target.scope,
            grant.grant_id,
            grant_content_sha256=grant.content_hash,
        )


def test_full_hash_records_are_never_overwritten_by_partial_indexes() -> None:
    source, _ = _store()
    source_grant = _create(source)
    source_revocation = _revoke(source, source_grant)

    target, _ = _store()
    target._InMemoryMemoryScopeGrantStore__grant_by_hash[source_grant.content_hash] = (
        source_grant
    )
    with pytest.raises(MemoryScopeGrantConflictError, match="full hash"):
        _create(target)
    assert (
        target._InMemoryMemoryScopeGrantStore__grant_by_hash[source_grant.content_hash]
        is source_grant
    )

    target, _ = _store()
    target_grant = _create(target)
    target._InMemoryMemoryScopeGrantStore__revocation_by_hash[
        source_revocation.content_hash
    ] = source_revocation
    with pytest.raises(MemoryScopeGrantConflictError, match="full hash"):
        _revoke(target, target_grant)
    assert target._InMemoryMemoryScopeGrantStore__revocation_hash_by_grant_hash == {}


def test_clock_reentry_and_invalid_clock_fail_closed_without_history() -> None:
    holder = {}

    def reentrant_clock():
        return holder["store"].resolve_active_grant(_request())

    store = InMemoryMemoryScopeGrantStore(
        resolver_id="resolver",
        resolver_version_sha256=_hash("resolver-version"),
        resolver_config_sha256=_hash("resolver-config"),
        revoker_id="revoker",
        revoker_version_sha256=_hash("revoker-version"),
        revoker_config_sha256=_hash("revoker-config"),
        clock=reentrant_clock,
    )
    holder["store"] = store
    with pytest.raises(MemoryScopeGrantConflictError, match="clock"):
        _create(store)

    invalid = InMemoryMemoryScopeGrantStore(
        resolver_id="resolver",
        resolver_version_sha256=_hash("resolver-version"),
        resolver_config_sha256=_hash("resolver-config"),
        revoker_id="revoker",
        revoker_version_sha256=_hash("revoker-version"),
        revoker_config_sha256=_hash("revoker-config"),
        clock=lambda: "not-a-datetime",  # type: ignore[return-value]
    )
    with pytest.raises(MemoryScopeGrantConflictError, match="invalid"):
        _create(invalid)


@pytest.mark.asyncio
async def test_real_store_revocation_blocks_broker_completed_result_retry() -> None:
    store, clock = _store()
    pin = _pin()
    coordinator = _Coordinator(pin)
    broker = AuthorizedMemoryAgentBroker(
        coordinator,
        MemoryScopeGrantAuthorizer(store, clock=clock),
    )
    session = await broker.open_session(_request().principal, "session-1")
    target = MemoryAssignmentGrantTargetV1.from_session_pin(pin)
    pin_request = MemoryScopeGrantRequestV1(
        principal=session.principal,
        session=session.session,
        audience=session.audience,
        target=target,
        action=MemoryScopeActionV1.PIN_ASSIGNMENT,
    )
    expose_request = replace(pin_request, action=MemoryScopeActionV1.EXPOSE_MEMORY)
    _create(store, pin_request, idempotency_key="broker-pin-grant")
    expose_grant = _create(
        store,
        expose_request,
        idempotency_key="broker-expose-grant",
    )
    try:
        await broker.pin_session(session, pin)
        turn = await broker.start_turn(session, "run-1")
        capability = WorkerMemoryTurnCapability.from_authorized_turn(broker, turn)
        await capability.expose_memory("same-operation", query=b"question")

        _revoke(store, expose_grant, idempotency_key="broker-expose-revoke")
        with pytest.raises(MemoryScopeAuthorizationDeniedError):
            await capability.expose_memory("same-operation", query=b"question")
        assert len(coordinator.expose_calls) == 1
    finally:
        await broker.aclose()
