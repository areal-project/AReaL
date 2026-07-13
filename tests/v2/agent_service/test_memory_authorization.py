# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from unittest.mock import patch

import pytest

from areal.v2.agent_service.memory import MemoryAgentSessionPinV1
from areal.v2.agent_service.memory_authorization import (
    MemoryAssignmentGrantTargetV1,
    MemoryPrincipalV1,
    MemoryScopeActionV1,
    MemoryScopeAuthorizationConflictError,
    MemoryScopeAuthorizationDeniedError,
    MemoryScopeAuthorizationDisabledError,
    MemoryScopeGrantAuthorizer,
    MemoryScopeGrantRequestV1,
    MemoryScopeGrantV1,
    MemorySessionIncarnationV1,
    MemoryWorkerAudienceV1,
)
from areal.v2.agent_service.memory_transport import (
    MemoryAgentMetadataWireV1,
    MemoryAssignmentPinWireV1,
    MemoryPinWireFormatError,
)
from areal.v2.memory_service.types import MemoryScope

_NOW = datetime(2026, 7, 13, 8, 0, tzinfo=UTC)
_RESOLVER_ID = "test-memory-scope-policy"
_RESOLVER_VERSION = sha256(b"resolver-version-v1").hexdigest()
_RESOLVER_CONFIG = sha256(b"resolver-config-v1").hexdigest()


class _StringSubclass(str):
    pass


class _ScopeSubclass(MemoryScope):
    pass


def _hash(label: str) -> str:
    return sha256(label.encode()).hexdigest()


def _scope(*, suffix: str = "1") -> MemoryScope:
    return MemoryScope(
        tenant_id=f"tenant-{suffix}",
        namespace="agent-long-term-memory",
        subject_id=f"subject-{suffix}",
    )


def _principal(*, suffix: str = "1") -> MemoryPrincipalV1:
    return MemoryPrincipalV1(
        issuer="https://identity.example",
        subject=f"principal-{suffix}",
    )


def _audience(character: str = "a") -> MemoryWorkerAudienceV1:
    return MemoryWorkerAudienceV1(f"maud_{character * 64}")


def _session(
    *,
    session_key: str = "session-1",
    character: str = "b",
) -> MemorySessionIncarnationV1:
    return MemorySessionIncarnationV1(
        session_key=session_key,
        incarnation_id=f"msinc_{character * 64}",
    )


def _target(
    *,
    scope: MemoryScope | None = None,
    suffix: str = "1",
) -> MemoryAssignmentGrantTargetV1:
    assignment_hash = _hash(f"assignment-{suffix}")
    return MemoryAssignmentGrantTargetV1(
        scope=scope or _scope(),
        rollout_group_id=f"rollout-group-{suffix}",
        rollout_group_incarnation_sha256=_hash(f"group-incarnation-{suffix}"),
        assignment_id=f"masn_{assignment_hash[:24]}",
        assignment_content_sha256=assignment_hash,
    )


def _pin(
    target: MemoryAssignmentGrantTargetV1 | None = None,
) -> MemoryAgentSessionPinV1:
    target = target or _target()
    return MemoryAgentSessionPinV1(
        scope=target.scope,
        rollout_group_id=target.rollout_group_id,
        rollout_group_incarnation_sha256=(target.rollout_group_incarnation_sha256),
        assignment_id=target.assignment_id,
        assignment_content_sha256=target.assignment_content_sha256,
    )


def _request(
    *,
    principal: MemoryPrincipalV1 | None = None,
    session: MemorySessionIncarnationV1 | None = None,
    audience: MemoryWorkerAudienceV1 | None = None,
    target: MemoryAssignmentGrantTargetV1 | None = None,
    action: MemoryScopeActionV1 = MemoryScopeActionV1.EXPOSE_MEMORY,
) -> MemoryScopeGrantRequestV1:
    return MemoryScopeGrantRequestV1(
        principal=principal or _principal(),
        session=session or _session(),
        audience=audience or _audience(),
        target=target or _target(),
        action=action,
    )


def _grant(
    request: MemoryScopeGrantRequestV1 | None = None,
    *,
    valid_from: datetime | None = None,
    valid_until: datetime | None = None,
    resolver_id: str = _RESOLVER_ID,
    resolver_version_sha256: str = _RESOLVER_VERSION,
    resolver_config_sha256: str = _RESOLVER_CONFIG,
    evaluated_at: datetime | None = None,
    granted_at: datetime | None = None,
) -> MemoryScopeGrantV1:
    return MemoryScopeGrantV1.create(
        request=request or _request(),
        resolver_id=resolver_id,
        resolver_version_sha256=resolver_version_sha256,
        resolver_config_sha256=resolver_config_sha256,
        valid_from=valid_from or (_NOW - timedelta(minutes=1)),
        valid_until=valid_until or (_NOW + timedelta(minutes=1)),
        evaluated_at=evaluated_at or (_NOW - timedelta(minutes=2)),
        granted_at=granted_at or (_NOW - timedelta(minutes=2)),
        idempotency_key="grant-request-1",
    )


class _Resolver:
    resolver_id = _RESOLVER_ID
    resolver_version_sha256 = _RESOLVER_VERSION
    resolver_config_sha256 = _RESOLVER_CONFIG

    def __init__(
        self,
        grant: object,
        *,
        exact_request: bool = False,
    ) -> None:
        self.grant = grant
        self.exact_request = exact_request
        self.active = True
        self.calls: list[MemoryScopeGrantRequestV1] = []

    def resolve_active_grant(
        self,
        request: MemoryScopeGrantRequestV1,
    ) -> MemoryScopeGrantV1:
        self.calls.append(request)
        if not self.active or (
            self.exact_request
            and isinstance(self.grant, MemoryScopeGrantV1)
            and request != self.grant.request
        ):
            raise MemoryScopeAuthorizationDeniedError(
                "active Memory scope grant is unavailable"
            )
        return self.grant  # type: ignore[return-value]


class _MutatingResolver(_Resolver):
    def resolve_active_grant(
        self,
        request: MemoryScopeGrantRequestV1,
    ) -> MemoryScopeGrantV1:
        self.calls.append(request)
        self.resolver_config_sha256 = _hash("mutated-config")
        return self.grant  # type: ignore[return-value]


class _RequestMutatingResolver(_Resolver):
    def resolve_active_grant(
        self,
        request: MemoryScopeGrantRequestV1,
    ) -> MemoryScopeGrantV1:
        self.calls.append(request)
        object.__setattr__(
            request,
            "action",
            MemoryScopeActionV1.PIN_ASSIGNMENT,
        )
        return _grant(request)


def _authority_variants(
    request: MemoryScopeGrantRequestV1,
) -> tuple[MemoryScopeGrantRequestV1, ...]:
    other_assignment_hash = _hash("other-assignment")
    target = request.target
    return (
        replace(request, principal=replace(request.principal, issuer="other-issuer")),
        replace(request, principal=replace(request.principal, subject="other-subject")),
        replace(request, session=replace(request.session, session_key="session-2")),
        replace(
            request,
            session=replace(
                request.session,
                incarnation_id=f"msinc_{'c' * 64}",
            ),
        ),
        replace(request, audience=_audience("d")),
        replace(
            request,
            target=replace(target, scope=replace(target.scope, tenant_id="tenant-2")),
        ),
        replace(
            request,
            target=replace(
                target, scope=replace(target.scope, namespace="other-memory")
            ),
        ),
        replace(
            request,
            target=replace(target, scope=replace(target.scope, subject_id="subject-2")),
        ),
        replace(request, target=replace(target, rollout_group_id="other-group")),
        replace(
            request,
            target=replace(
                target,
                rollout_group_incarnation_sha256=_hash("other-incarnation"),
            ),
        ),
        replace(
            request,
            target=replace(
                target,
                assignment_id=f"masn_{other_assignment_hash[:24]}",
                assignment_content_sha256=other_assignment_hash,
            ),
        ),
        replace(request, action=MemoryScopeActionV1.PIN_ASSIGNMENT),
    )


def test_server_minted_audience_and_session_incarnations_are_independent() -> None:
    with patch(
        "areal.v2.agent_service.memory_authorization.secrets.token_hex",
        side_effect=("a" * 64, "b" * 64, "c" * 64),
    ):
        audience = MemoryWorkerAudienceV1.create()
        first = MemorySessionIncarnationV1.create("shared-session-key")
        replacement = MemorySessionIncarnationV1.create("shared-session-key")

    assert audience.audience_id == f"maud_{'a' * 64}"
    assert first.incarnation_id == f"msinc_{'b' * 64}"
    assert replacement.incarnation_id == f"msinc_{'c' * 64}"
    assert first.session_key == replacement.session_key
    assert first != replacement
    assert audience.audience_id not in (first.session_key, first.incarnation_id)


def test_authorization_contract_is_exported_from_agent_service() -> None:
    from areal.v2 import agent_service

    assert agent_service.MemoryPrincipalV1 is MemoryPrincipalV1
    assert agent_service.MemoryScopeGrantAuthorizer is MemoryScopeGrantAuthorizer
    assert agent_service.MemoryScopeGrantV1 is MemoryScopeGrantV1


def test_assignment_target_snapshots_the_complete_session_pin() -> None:
    pin = _pin()
    target = MemoryAssignmentGrantTargetV1.from_session_pin(pin)

    assert target == _target()
    assert target.scope is pin.scope
    assert target.rollout_group_id == pin.rollout_group_id
    assert (
        target.rollout_group_incarnation_sha256 == pin.rollout_group_incarnation_sha256
    )
    assert target.assignment_id == pin.assignment_id
    assert target.assignment_content_sha256 == pin.assignment_content_sha256


def test_every_authority_dimension_changes_the_canonical_request() -> None:
    request = _request()
    variants = _authority_variants(request)

    canonical_values = {request.canonical_bytes()}
    canonical_values.update(variant.canonical_bytes() for variant in variants)

    assert len(canonical_values) == len(variants) + 1


def test_grant_has_canonical_audit_identity_but_is_not_a_bearer() -> None:
    request = _request()
    grant = _grant(request)
    duplicate = _grant(request)

    assert grant == duplicate
    assert grant.content_hash == sha256(grant.canonical_bytes()).hexdigest()
    assert grant.grant_id == f"msgr_{grant.content_hash[:24]}"
    grant.verify_integrity()

    authorizer = MemoryScopeGrantAuthorizer(clock=lambda: _NOW)
    for bearer_like_value in (
        grant.grant_id,
        grant.content_hash,
        request.session.session_key,
    ):
        with pytest.raises(TypeError, match="request must be"):
            authorizer.authorize(bearer_like_value)  # type: ignore[arg-type]


def test_authorization_is_default_disabled_without_a_resolver() -> None:
    authorizer = MemoryScopeGrantAuthorizer(clock=lambda: _NOW)

    with pytest.raises(
        MemoryScopeAuthorizationDisabledError,
        match="not configured",
    ):
        authorizer.authorize(_request())


def test_exact_active_grant_is_resolved_on_every_call_and_not_cached() -> None:
    request = _request()
    grant = _grant(request)
    resolver = _Resolver(grant, exact_request=True)
    authorizer = MemoryScopeGrantAuthorizer(resolver, clock=lambda: _NOW)

    assert authorizer.authorize(request) is grant
    resolver.active = False
    with pytest.raises(MemoryScopeAuthorizationDeniedError):
        authorizer.authorize(request)

    assert resolver.calls == [request, request]


@pytest.mark.parametrize(
    ("granted_action", "requested_action"),
    (
        (
            MemoryScopeActionV1.PIN_ASSIGNMENT,
            MemoryScopeActionV1.EXPOSE_MEMORY,
        ),
        (
            MemoryScopeActionV1.EXPOSE_MEMORY,
            MemoryScopeActionV1.PIN_ASSIGNMENT,
        ),
    ),
)
def test_pin_and_exposure_actions_never_imply_each_other(
    granted_action: MemoryScopeActionV1,
    requested_action: MemoryScopeActionV1,
) -> None:
    granted_request = _request(action=granted_action)
    resolver = _Resolver(_grant(granted_request), exact_request=True)
    authorizer = MemoryScopeGrantAuthorizer(resolver, clock=lambda: _NOW)

    assert authorizer.authorize(granted_request).request.action is granted_action
    with pytest.raises(MemoryScopeAuthorizationDeniedError):
        authorizer.authorize(replace(granted_request, action=requested_action))


@pytest.mark.parametrize("variant_index", range(12))
def test_exact_resolver_denies_every_authority_substitution(
    variant_index: int,
) -> None:
    request = _request()
    resolver = _Resolver(_grant(request), exact_request=True)
    authorizer = MemoryScopeGrantAuthorizer(resolver, clock=lambda: _NOW)
    variant = _authority_variants(request)[variant_index]

    with pytest.raises(MemoryScopeAuthorizationDeniedError):
        authorizer.authorize(variant)


@pytest.mark.parametrize("variant_index", range(12))
def test_host_rejects_a_resolver_that_substitutes_any_authority_dimension(
    variant_index: int,
) -> None:
    request = _request()
    variant = _authority_variants(request)[variant_index]
    resolver = _Resolver(_grant(request))
    authorizer = MemoryScopeGrantAuthorizer(resolver, clock=lambda: _NOW)

    with pytest.raises(
        MemoryScopeAuthorizationConflictError,
        match="different grant",
    ):
        authorizer.authorize(variant)


@pytest.mark.parametrize(
    ("valid_from", "valid_until"),
    (
        (_NOW + timedelta(seconds=1), _NOW + timedelta(minutes=1)),
        (_NOW - timedelta(minutes=1), _NOW),
        (_NOW - timedelta(minutes=2), _NOW - timedelta(minutes=1)),
    ),
)
def test_authorizer_denies_future_expired_and_boundary_expired_grants(
    valid_from: datetime,
    valid_until: datetime,
) -> None:
    request = _request()
    resolver = _Resolver(
        _grant(
            request,
            valid_from=valid_from,
            valid_until=valid_until,
        )
    )

    with pytest.raises(MemoryScopeAuthorizationDeniedError):
        MemoryScopeGrantAuthorizer(resolver, clock=lambda: _NOW).authorize(request)


def test_authorizer_rejects_a_future_grant_timestamp() -> None:
    request = _request()
    future = _NOW + timedelta(seconds=1)
    resolver = _Resolver(
        _grant(
            request,
            valid_from=_NOW - timedelta(minutes=1),
            valid_until=_NOW + timedelta(minutes=1),
            evaluated_at=future,
            granted_at=future,
        )
    )

    with pytest.raises(
        MemoryScopeAuthorizationConflictError,
        match="future grant timestamp",
    ):
        MemoryScopeGrantAuthorizer(resolver, clock=lambda: _NOW).authorize(request)

    with pytest.raises(ValueError, match="granted_at must precede"):
        _grant(
            request,
            valid_until=_NOW,
            granted_at=_NOW,
        )


def test_authorizer_rejects_tampered_or_noncanonical_resolver_results() -> None:
    request = _request()
    tampered = _grant(request)
    object.__setattr__(tampered, "content_hash", "0" * 64)

    for returned in (tampered, True, None, object()):
        resolver = _Resolver(returned)
        authorizer = MemoryScopeGrantAuthorizer(resolver, clock=lambda: _NOW)
        with pytest.raises(MemoryScopeAuthorizationConflictError):
            authorizer.authorize(request)


def test_authorizer_binds_and_rechecks_the_selected_resolver_identity() -> None:
    request = _request()
    resolver = _Resolver(_grant(request))
    authorizer = MemoryScopeGrantAuthorizer(resolver, clock=lambda: _NOW)
    resolver.resolver_id = "replacement-policy"

    with pytest.raises(
        MemoryScopeAuthorizationConflictError,
        match="identity changed",
    ):
        authorizer.authorize(request)
    assert resolver.calls == []

    mutating = _MutatingResolver(_grant(request))
    with pytest.raises(
        MemoryScopeAuthorizationConflictError,
        match="identity changed",
    ):
        MemoryScopeGrantAuthorizer(mutating, clock=lambda: _NOW).authorize(request)
    assert mutating.calls == [request]


def test_resolver_cannot_mutate_the_request_and_compare_it_to_itself() -> None:
    request = _request(action=MemoryScopeActionV1.EXPOSE_MEMORY)
    resolver = _RequestMutatingResolver(_grant(request))

    with pytest.raises(
        MemoryScopeAuthorizationConflictError,
        match="mutated the authorization request",
    ):
        MemoryScopeGrantAuthorizer(resolver, clock=lambda: _NOW).authorize(request)

    assert request.action is MemoryScopeActionV1.PIN_ASSIGNMENT
    assert resolver.calls == [request]


def test_authorizer_rejects_a_grant_claiming_another_resolver() -> None:
    request = _request()
    resolver = _Resolver(_grant(request, resolver_id="another-policy"))

    with pytest.raises(
        MemoryScopeAuthorizationConflictError,
        match="different grant",
    ):
        MemoryScopeGrantAuthorizer(resolver, clock=lambda: _NOW).authorize(request)


def test_invalid_clock_and_resolver_contracts_fail_closed() -> None:
    request = _request()
    resolver = _Resolver(_grant(request))

    with pytest.raises(TypeError, match="resolve_active_grant"):
        MemoryScopeGrantAuthorizer(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="resolver_id"):
        MemoryScopeGrantAuthorizer(
            type("MissingIdentity", (), {"resolve_active_grant": lambda *_: None})()
        )
    with pytest.raises(
        MemoryScopeAuthorizationConflictError,
        match="clock",
    ):
        MemoryScopeGrantAuthorizer(
            resolver,
            clock=lambda: "not-a-datetime",  # type: ignore[arg-type,return-value]
        ).authorize(request)


def test_strict_types_reject_claims_and_wildcard_actions() -> None:
    request = _request()

    with pytest.raises(TypeError, match="principal"):
        replace(request, principal="victim")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="session"):
        replace(request, session=request.session.session_key)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="audience"):
        replace(request, audience=request.audience.audience_id)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="target"):
        replace(request, target=_pin())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="action"):
        replace(request, action="*")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="maud_"):
        MemoryWorkerAudienceV1("worker-1")
    with pytest.raises(ValueError, match="msinc_"):
        MemorySessionIncarnationV1("session-1", "session-1")
    with pytest.raises(TypeError, match="issuer"):
        MemoryPrincipalV1(_StringSubclass("issuer"), "principal")
    with pytest.raises(TypeError, match="scope"):
        _target(scope=_ScopeSubclass("tenant", "memory", "subject"))

    tampered_scope_request = _request()
    object.__setattr__(tampered_scope_request.target.scope, "tenant_id", 123)
    with pytest.raises(TypeError, match="scope.tenant_id"):
        tampered_scope_request.canonical_bytes()


@pytest.mark.parametrize(
    "forged_field",
    (
        "principal",
        "session_incarnation_id",
        "worker_audience_id",
        "grant_id",
        "action",
    ),
)
def test_pin_wire_rejects_every_authorization_claim(forged_field: str) -> None:
    envelope = MemoryAgentMetadataWireV1(
        MemoryAssignmentPinWireV1.from_runtime_pin(_pin())
    ).to_wire()
    envelope[forged_field] = "attacker-controlled"

    with pytest.raises(MemoryPinWireFormatError, match="unknown"):
        MemoryAgentMetadataWireV1.from_wire(envelope)
