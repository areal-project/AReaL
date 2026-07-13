# SPDX-License-Identifier: Apache-2.0

"""Linearizable in-memory control store for exact Agent Memory grants.

The store is intentionally stricter than a renewable credential service.  One
exact :class:`MemoryScopeGrantRequestV1` may produce at most one grant during
the lifetime of a store.  Revocation is an irreversible tombstone; expiry or
revocation therefore cannot be bypassed with another idempotency key.  A host
that wants to authorize a new lifetime must mint a new session incarnation or
Worker audience.  Explicit generations and supersession can be added later
without hiding an ABA transition inside this V1 contract.

Authority is resolved only from an exact request.  Grant IDs, hashes, grant
records, and revocation records are audit pointers and are never bearer
credentials.  All authority-changing operations and active resolution
linearize under one lock.  Records cross the store boundary as detached object
graphs so caller mutation cannot poison the private indexes.

Control addresses and idempotency keys are scoped by ``MemoryScope``.  A new
grant must be active at its commit point; scheduled activation is deferred
until an explicit generation and supersession contract exists.

This reference backend is process-local and non-durable.  Restarting loses all
grants and therefore fails closed.  It does not expose HTTP administration,
authenticate operators, or sandbox arbitrary Python in the same process.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from threading import RLock, get_ident
from typing import Protocol

from areal.v2.memory_service._atomic import _atomic_publish
from areal.v2.memory_service.types import MemoryScope

from .memory_authorization import (
    MemoryAssignmentGrantTargetV1,
    MemoryPrincipalV1,
    MemoryScopeAuthorizationConflictError,
    MemoryScopeAuthorizationDeniedError,
    MemoryScopeAuthorizationError,
    MemoryScopeGrantRequestV1,
    MemoryScopeGrantResolver,
    MemoryScopeGrantV1,
    MemorySessionIncarnationV1,
    MemoryWorkerAudienceV1,
    _aware_datetime,
    _digest,
    _record_bytes,
    _request,
    _string,
)

_ScopeKey = tuple[str, str, str]


class MemoryScopeGrantNotFoundError(MemoryScopeAuthorizationError):
    """Raised when an audit or revocation address has no exact record."""


class MemoryScopeGrantConflictError(MemoryScopeAuthorizationConflictError):
    """Raised when control-plane identity or immutable state conflicts."""


class MemoryScopeGrantRevocationReasonV1(StrEnum):
    """Bounded revocation reasons; sensitive free text stays off-ledger."""

    POLICY = "policy"
    SECURITY = "security"
    OPERATOR = "operator"
    SUPERSEDED = "superseded"
    OTHER = "other"


def _optional_digest(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _digest(value, field_name)


def _scope(value: object) -> MemoryScope:
    if type(value) is not MemoryScope:
        raise TypeError("scope must be a MemoryScope")
    # Reconstruct to validate nested scalars after possible object.__setattr__.
    return MemoryScope(
        tenant_id=value.tenant_id,
        namespace=value.namespace,
        subject_id=value.subject_id,
    )


def _scope_value(value: object) -> dict[str, str]:
    scope = _scope(value)
    return {
        "namespace": scope.namespace,
        "subject_id": scope.subject_id,
        "tenant_id": scope.tenant_id,
    }


def _scope_key(value: object) -> _ScopeKey:
    scope = _scope(value)
    return (scope.tenant_id, scope.namespace, scope.subject_id)


def _grant_reference(grant_id: object, content_hash: object) -> tuple[str, str]:
    grant_id = _string(grant_id, "grant_id")
    content_hash = _digest(content_hash, "grant_content_sha256")
    if grant_id != f"msgr_{content_hash[:24]}":
        raise ValueError("grant_id disagrees with grant_content_sha256")
    return grant_id, content_hash


def _control_grant_reference(
    grant_id: object,
    content_hash: object,
) -> tuple[str, str]:
    try:
        return _grant_reference(grant_id, content_hash)
    except (TypeError, ValueError) as error:
        raise MemoryScopeGrantConflictError(
            "grant ID disagrees with its full hash"
        ) from error


def _reason(value: object) -> MemoryScopeGrantRevocationReasonV1:
    if type(value) is not MemoryScopeGrantRevocationReasonV1:
        raise TypeError("reason must be a MemoryScopeGrantRevocationReasonV1")
    return value


@dataclass(frozen=True, slots=True)
class MemoryScopeGrantRevocationV1:
    """Content-addressed irreversible tombstone for one exact grant."""

    revocation_id: str
    scope: MemoryScope
    grant_id: str
    grant_content_sha256: str
    request_content_sha256: str
    resolver_id: str
    resolver_version_sha256: str
    resolver_config_sha256: str
    revoker_id: str
    revoker_version_sha256: str
    revoker_config_sha256: str
    reason: MemoryScopeGrantRevocationReasonV1
    reason_detail_sha256: str | None
    revoked_at: datetime
    idempotency_key: str
    content_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", _scope(self.scope))
        grant_id, grant_hash = _grant_reference(
            self.grant_id,
            self.grant_content_sha256,
        )
        object.__setattr__(self, "grant_id", grant_id)
        object.__setattr__(self, "grant_content_sha256", grant_hash)
        object.__setattr__(
            self,
            "request_content_sha256",
            _digest(self.request_content_sha256, "request_content_sha256"),
        )
        for field_name in ("resolver_id", "revoker_id", "idempotency_key"):
            object.__setattr__(
                self,
                field_name,
                _string(getattr(self, field_name), field_name),
            )
        for field_name in (
            "resolver_version_sha256",
            "resolver_config_sha256",
            "revoker_version_sha256",
            "revoker_config_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _digest(getattr(self, field_name), field_name),
            )
        object.__setattr__(self, "reason", _reason(self.reason))
        detail = _optional_digest(
            self.reason_detail_sha256,
            "reason_detail_sha256",
        )
        if self.reason is MemoryScopeGrantRevocationReasonV1.OTHER and detail is None:
            raise ValueError("OTHER revocation reason requires reason_detail_sha256")
        object.__setattr__(self, "reason_detail_sha256", detail)
        object.__setattr__(
            self,
            "revoked_at",
            _aware_datetime(self.revoked_at, "revoked_at"),
        )
        self.verify_integrity()

    def _canonical_value(self) -> dict[str, object]:
        grant_id, grant_hash = _grant_reference(
            self.grant_id,
            self.grant_content_sha256,
        )
        reason = _reason(self.reason)
        detail = _optional_digest(
            self.reason_detail_sha256,
            "reason_detail_sha256",
        )
        if reason is MemoryScopeGrantRevocationReasonV1.OTHER and detail is None:
            raise ValueError("OTHER revocation reason requires reason_detail_sha256")
        return {
            "grant_content_sha256": grant_hash,
            "grant_id": grant_id,
            "idempotency_key": _string(
                self.idempotency_key,
                "idempotency_key",
            ),
            "reason": reason.value,
            "reason_detail_sha256": detail,
            "request_content_sha256": _digest(
                self.request_content_sha256,
                "request_content_sha256",
            ),
            "resolved_by": {
                "config_sha256": _digest(
                    self.resolver_config_sha256,
                    "resolver_config_sha256",
                ),
                "id": _string(self.resolver_id, "resolver_id"),
                "version_sha256": _digest(
                    self.resolver_version_sha256,
                    "resolver_version_sha256",
                ),
            },
            "revoked_at": _aware_datetime(
                self.revoked_at,
                "revoked_at",
            ).isoformat(),
            "revoked_by": {
                "config_sha256": _digest(
                    self.revoker_config_sha256,
                    "revoker_config_sha256",
                ),
                "id": _string(self.revoker_id, "revoker_id"),
                "version_sha256": _digest(
                    self.revoker_version_sha256,
                    "revoker_version_sha256",
                ),
            },
            "scope": _scope_value(self.scope),
        }

    def canonical_bytes(self) -> bytes:
        return _record_bytes(
            "memory_scope_grant_revocation",
            self._canonical_value(),
        )

    def verify_integrity(self) -> None:
        expected_hash = sha256(self.canonical_bytes()).hexdigest()
        content_hash = _digest(self.content_hash, "content_hash")
        if content_hash != expected_hash:
            raise ValueError("content_hash disagrees with canonical revocation bytes")
        revocation_id = _string(self.revocation_id, "revocation_id")
        if revocation_id != f"msgrv_{expected_hash[:24]}":
            raise ValueError("revocation_id disagrees with canonical revocation bytes")

    @classmethod
    def create(
        cls,
        *,
        scope: MemoryScope,
        grant_id: str,
        grant_content_sha256: str,
        request_content_sha256: str,
        resolver_id: str,
        resolver_version_sha256: str,
        resolver_config_sha256: str,
        revoker_id: str,
        revoker_version_sha256: str,
        revoker_config_sha256: str,
        reason: MemoryScopeGrantRevocationReasonV1,
        reason_detail_sha256: str | None,
        revoked_at: datetime,
        idempotency_key: str,
    ) -> MemoryScopeGrantRevocationV1:
        values = {
            "scope": _scope(scope),
            "grant_id": _string(grant_id, "grant_id"),
            "grant_content_sha256": _digest(
                grant_content_sha256,
                "grant_content_sha256",
            ),
            "request_content_sha256": _digest(
                request_content_sha256,
                "request_content_sha256",
            ),
            "resolver_id": _string(resolver_id, "resolver_id"),
            "resolver_version_sha256": _digest(
                resolver_version_sha256,
                "resolver_version_sha256",
            ),
            "resolver_config_sha256": _digest(
                resolver_config_sha256,
                "resolver_config_sha256",
            ),
            "revoker_id": _string(revoker_id, "revoker_id"),
            "revoker_version_sha256": _digest(
                revoker_version_sha256,
                "revoker_version_sha256",
            ),
            "revoker_config_sha256": _digest(
                revoker_config_sha256,
                "revoker_config_sha256",
            ),
            "reason": _reason(reason),
            "reason_detail_sha256": _optional_digest(
                reason_detail_sha256,
                "reason_detail_sha256",
            ),
            "revoked_at": _aware_datetime(revoked_at, "revoked_at"),
            "idempotency_key": _string(idempotency_key, "idempotency_key"),
        }
        if (
            values["reason"] is MemoryScopeGrantRevocationReasonV1.OTHER
            and values["reason_detail_sha256"] is None
        ):
            raise ValueError("OTHER revocation reason requires reason_detail_sha256")
        _grant_reference(values["grant_id"], values["grant_content_sha256"])
        canonical = _record_bytes(
            "memory_scope_grant_revocation",
            {
                "grant_content_sha256": values["grant_content_sha256"],
                "grant_id": values["grant_id"],
                "idempotency_key": values["idempotency_key"],
                "reason": values["reason"].value,  # type: ignore[union-attr]
                "reason_detail_sha256": values["reason_detail_sha256"],
                "request_content_sha256": values["request_content_sha256"],
                "resolved_by": {
                    "config_sha256": values["resolver_config_sha256"],
                    "id": values["resolver_id"],
                    "version_sha256": values["resolver_version_sha256"],
                },
                "revoked_at": values["revoked_at"].isoformat(),  # type: ignore[union-attr]
                "revoked_by": {
                    "config_sha256": values["revoker_config_sha256"],
                    "id": values["revoker_id"],
                    "version_sha256": values["revoker_version_sha256"],
                },
                "scope": _scope_value(values["scope"]),
            },
        )
        content_hash = sha256(canonical).hexdigest()
        return cls(
            revocation_id=f"msgrv_{content_hash[:24]}",
            content_hash=content_hash,
            **values,  # type: ignore[arg-type]
        )


def _clone_request(value: object) -> MemoryScopeGrantRequestV1:
    request = _request(value)
    before = request.canonical_bytes()
    target = request.target
    clone = MemoryScopeGrantRequestV1(
        principal=MemoryPrincipalV1(
            issuer=request.principal.issuer,
            subject=request.principal.subject,
        ),
        session=MemorySessionIncarnationV1(
            session_key=request.session.session_key,
            incarnation_id=request.session.incarnation_id,
        ),
        audience=MemoryWorkerAudienceV1(request.audience.audience_id),
        target=MemoryAssignmentGrantTargetV1(
            scope=_scope(target.scope),
            rollout_group_id=target.rollout_group_id,
            rollout_group_incarnation_sha256=(target.rollout_group_incarnation_sha256),
            assignment_id=target.assignment_id,
            assignment_content_sha256=target.assignment_content_sha256,
        ),
        action=request.action,
    )
    try:
        after = request.canonical_bytes()
    except (TypeError, ValueError) as error:
        raise MemoryScopeGrantConflictError(
            "Memory grant request changed while it was being copied"
        ) from error
    if before != after or clone.canonical_bytes() != before:
        raise MemoryScopeGrantConflictError(
            "Memory grant request changed while it was being copied"
        )
    return clone


def _clone_grant(value: object) -> MemoryScopeGrantV1:
    if type(value) is not MemoryScopeGrantV1:
        raise MemoryScopeGrantConflictError("stored grant is not canonical")
    try:
        before = value.canonical_bytes()
        value.verify_integrity()
        clone = MemoryScopeGrantV1(
            grant_id=value.grant_id,
            request=_clone_request(value.request),
            resolver_id=value.resolver_id,
            resolver_version_sha256=value.resolver_version_sha256,
            resolver_config_sha256=value.resolver_config_sha256,
            valid_from=value.valid_from,
            valid_until=value.valid_until,
            evaluated_at=value.evaluated_at,
            granted_at=value.granted_at,
            idempotency_key=value.idempotency_key,
            content_hash=value.content_hash,
        )
        after = value.canonical_bytes()
    except (TypeError, ValueError) as error:
        raise MemoryScopeGrantConflictError("stored grant is corrupted") from error
    if (
        before != after
        or clone.canonical_bytes() != before
        or clone.content_hash != value.content_hash
    ):
        raise MemoryScopeGrantConflictError(
            "stored grant changed while it was being copied"
        )
    return clone


def _clone_revocation(value: object) -> MemoryScopeGrantRevocationV1:
    if type(value) is not MemoryScopeGrantRevocationV1:
        raise MemoryScopeGrantConflictError("stored revocation is not canonical")
    try:
        before = value.canonical_bytes()
        value.verify_integrity()
        clone = MemoryScopeGrantRevocationV1(
            revocation_id=value.revocation_id,
            scope=_scope(value.scope),
            grant_id=value.grant_id,
            grant_content_sha256=value.grant_content_sha256,
            request_content_sha256=value.request_content_sha256,
            resolver_id=value.resolver_id,
            resolver_version_sha256=value.resolver_version_sha256,
            resolver_config_sha256=value.resolver_config_sha256,
            revoker_id=value.revoker_id,
            revoker_version_sha256=value.revoker_version_sha256,
            revoker_config_sha256=value.revoker_config_sha256,
            reason=value.reason,
            reason_detail_sha256=value.reason_detail_sha256,
            revoked_at=value.revoked_at,
            idempotency_key=value.idempotency_key,
            content_hash=value.content_hash,
        )
        after = value.canonical_bytes()
    except (TypeError, ValueError) as error:
        raise MemoryScopeGrantConflictError("stored revocation is corrupted") from error
    if before != after or clone.canonical_bytes() != before:
        raise MemoryScopeGrantConflictError(
            "stored revocation changed while it was being copied"
        )
    return clone


@dataclass(frozen=True, slots=True)
class _GrantCreationSpec:
    request_bytes: bytes
    valid_from: datetime
    valid_until: datetime
    idempotency_key: str


@dataclass(frozen=True, slots=True)
class _RevocationSpec:
    grant_id: str
    grant_content_sha256: str
    reason: MemoryScopeGrantRevocationReasonV1
    reason_detail_sha256: str | None
    idempotency_key: str


class MemoryScopeGrantStore(MemoryScopeGrantResolver, Protocol):
    """Trusted control and resolver contract; audit getters grant nothing."""

    revoker_id: str
    revoker_version_sha256: str
    revoker_config_sha256: str

    def create_grant(
        self,
        request: MemoryScopeGrantRequestV1,
        *,
        valid_from: datetime,
        valid_until: datetime,
        idempotency_key: str,
    ) -> MemoryScopeGrantV1: ...

    def revoke_grant(
        self,
        scope: MemoryScope,
        grant_id: str,
        *,
        grant_content_sha256: str,
        reason: MemoryScopeGrantRevocationReasonV1,
        reason_detail_sha256: str | None = None,
        idempotency_key: str,
    ) -> MemoryScopeGrantRevocationV1: ...

    def get_grant_for_audit(
        self,
        scope: MemoryScope,
        grant_id: str,
        *,
        grant_content_sha256: str,
    ) -> MemoryScopeGrantV1: ...

    def get_grant_revocation_for_audit(
        self,
        scope: MemoryScope,
        grant_id: str,
        *,
        grant_content_sha256: str,
    ) -> MemoryScopeGrantRevocationV1: ...


def _utc_now() -> datetime:
    return datetime.now(UTC)


class InMemoryMemoryScopeGrantStore:
    """Single-lock reference backend with irreversible exact-request history."""

    def __init__(
        self,
        *,
        resolver_id: str,
        resolver_version_sha256: str,
        resolver_config_sha256: str,
        revoker_id: str,
        revoker_version_sha256: str,
        revoker_config_sha256: str,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        """Build the reference store around trusted component identities.

        ``clock`` must be pure and non-blocking. It must not call this store,
        directly or through another thread, because it is sampled while the
        store lock is held at authority linearization points.
        """

        if not callable(clock):
            raise TypeError("clock must be callable")
        self.__resolver_id = _string(resolver_id, "resolver_id")
        self.__resolver_version_sha256 = _digest(
            resolver_version_sha256,
            "resolver_version_sha256",
        )
        self.__resolver_config_sha256 = _digest(
            resolver_config_sha256,
            "resolver_config_sha256",
        )
        self.__revoker_id = _string(revoker_id, "revoker_id")
        self.__revoker_version_sha256 = _digest(
            revoker_version_sha256,
            "revoker_version_sha256",
        )
        self.__revoker_config_sha256 = _digest(
            revoker_config_sha256,
            "revoker_config_sha256",
        )
        self.__clock = clock
        self.__lock = RLock()
        self.__clock_owner_thread_id: int | None = None
        self.__last_observed_at: datetime | None = None
        self.__grant_by_hash: dict[str, MemoryScopeGrantV1] = {}
        self.__grant_hash_by_request: dict[bytes, str] = {}
        self.__grant_hash_by_idempotency: dict[tuple[_ScopeKey, str], str] = {}
        self.__grant_hash_by_display_id: dict[tuple[_ScopeKey, str], str] = {}
        self.__revocation_by_hash: dict[str, MemoryScopeGrantRevocationV1] = {}
        self.__revocation_hash_by_grant_hash: dict[str, str] = {}
        self.__revocation_hash_by_idempotency: dict[tuple[_ScopeKey, str], str] = {}
        self.__revocation_hash_by_display_id: dict[tuple[_ScopeKey, str], str] = {}

    @property
    def resolver_id(self) -> str:
        return self.__resolver_id

    @property
    def resolver_version_sha256(self) -> str:
        return self.__resolver_version_sha256

    @property
    def resolver_config_sha256(self) -> str:
        return self.__resolver_config_sha256

    @property
    def revoker_id(self) -> str:
        return self.__revoker_id

    @property
    def revoker_version_sha256(self) -> str:
        return self.__revoker_version_sha256

    @property
    def revoker_config_sha256(self) -> str:
        return self.__revoker_config_sha256

    def _reject_clock_reentry(self) -> None:
        if self.__clock_owner_thread_id == get_ident():
            raise MemoryScopeGrantConflictError(
                "grant-store clock must not call the grant store"
            )

    def _sample_clock_locked(self) -> datetime:
        if self.__clock_owner_thread_id is not None:
            raise MemoryScopeGrantConflictError("grant-store clock is reentrant")
        self.__clock_owner_thread_id = get_ident()
        try:
            try:
                value = self.__clock()
            except MemoryScopeAuthorizationError:
                raise
            except Exception as error:
                raise MemoryScopeGrantConflictError(
                    "grant-store clock failed"
                ) from error
            try:
                now = _aware_datetime(value, "grant-store clock")
            except (TypeError, ValueError) as error:
                raise MemoryScopeGrantConflictError(
                    "grant-store clock returned an invalid value"
                ) from error
        finally:
            self.__clock_owner_thread_id = None
        if self.__last_observed_at is not None and now < self.__last_observed_at:
            raise MemoryScopeGrantConflictError("grant-store clock moved backwards")
        self.__last_observed_at = now
        return now

    def _validated_grant_locked(self, grant_hash: str) -> MemoryScopeGrantV1:
        grant = self.__grant_by_hash.get(grant_hash)
        if grant is None:
            raise MemoryScopeGrantConflictError("grant indexes disagree")
        cloned = _clone_grant(grant)
        request_bytes = cloned.request.canonical_bytes()
        idempotency_address = (
            _scope_key(cloned.request.target.scope),
            cloned.idempotency_key,
        )
        display_address = (
            _scope_key(cloned.request.target.scope),
            cloned.grant_id,
        )
        if (
            cloned.content_hash != grant_hash
            or (
                cloned.resolver_id,
                cloned.resolver_version_sha256,
                cloned.resolver_config_sha256,
            )
            != (
                self.__resolver_id,
                self.__resolver_version_sha256,
                self.__resolver_config_sha256,
            )
            or self.__grant_hash_by_display_id.get(display_address) != grant_hash
            or self.__grant_hash_by_request.get(request_bytes) != grant_hash
            or self.__grant_hash_by_idempotency.get(idempotency_address) != grant_hash
        ):
            raise MemoryScopeGrantConflictError("grant indexes disagree")
        return cloned

    def _grant_locked(
        self,
        scope_key: _ScopeKey,
        grant_id: str,
        grant_hash: str,
    ) -> MemoryScopeGrantV1:
        indexed_hash = self.__grant_hash_by_display_id.get((scope_key, grant_id))
        grant = self.__grant_by_hash.get(grant_hash)
        if indexed_hash is None and grant is None:
            raise MemoryScopeGrantNotFoundError("Memory scope grant was not found")
        if indexed_hash is None and grant is not None:
            unaddressed = _clone_grant(grant)
            if _scope_key(unaddressed.request.target.scope) != scope_key:
                raise MemoryScopeGrantNotFoundError("Memory scope grant was not found")
            raise MemoryScopeGrantConflictError("grant address indexes disagree")
        if indexed_hash is None or grant is None or indexed_hash != grant_hash:
            raise MemoryScopeGrantConflictError(
                "grant address indexes disagree or the display ID collides"
            )
        cloned = self._validated_grant_locked(grant_hash)
        if cloned.grant_id != grant_id:
            raise MemoryScopeGrantConflictError("grant indexes disagree")
        if _scope_key(cloned.request.target.scope) != scope_key:
            raise MemoryScopeGrantNotFoundError("Memory scope grant was not found")
        return cloned

    def _validated_revocation_locked(
        self,
        revocation_hash: str,
    ) -> MemoryScopeGrantRevocationV1:
        revocation = self.__revocation_by_hash.get(revocation_hash)
        if revocation is None:
            raise MemoryScopeGrantConflictError("revocation indexes disagree")
        cloned = _clone_revocation(revocation)
        idempotency_address = (
            _scope_key(cloned.scope),
            cloned.idempotency_key,
        )
        display_address = (_scope_key(cloned.scope), cloned.revocation_id)
        if (
            cloned.content_hash != revocation_hash
            or self.__revocation_hash_by_display_id.get(display_address)
            != revocation_hash
            or self.__revocation_hash_by_idempotency.get(idempotency_address)
            != revocation_hash
        ):
            raise MemoryScopeGrantConflictError("revocation indexes disagree")
        return cloned

    def _revocation_locked(
        self,
        grant: MemoryScopeGrantV1,
    ) -> MemoryScopeGrantRevocationV1 | None:
        grant_hash = grant.content_hash
        revocation_hash = self.__revocation_hash_by_grant_hash.get(grant_hash)
        if revocation_hash is None:
            return None
        cloned = self._validated_revocation_locked(revocation_hash)
        if (
            cloned.grant_content_sha256 != grant_hash
            or cloned.grant_id != grant.grant_id
            or cloned.request_content_sha256
            != sha256(grant.request.canonical_bytes()).hexdigest()
            or _scope_key(cloned.scope) != _scope_key(grant.request.target.scope)
            or (
                cloned.resolver_id,
                cloned.resolver_version_sha256,
                cloned.resolver_config_sha256,
            )
            != (
                self.__resolver_id,
                self.__resolver_version_sha256,
                self.__resolver_config_sha256,
            )
            or (
                cloned.revoker_id,
                cloned.revoker_version_sha256,
                cloned.revoker_config_sha256,
            )
            != (
                self.__revoker_id,
                self.__revoker_version_sha256,
                self.__revoker_config_sha256,
            )
        ):
            raise MemoryScopeGrantConflictError("revocation indexes disagree")
        return cloned

    def create_grant(
        self,
        request: MemoryScopeGrantRequestV1,
        *,
        valid_from: datetime,
        valid_until: datetime,
        idempotency_key: str,
    ) -> MemoryScopeGrantV1:
        self._reject_clock_reentry()
        request = _clone_request(request)
        request_bytes = request.canonical_bytes()
        valid_from = _aware_datetime(valid_from, "valid_from")
        valid_until = _aware_datetime(valid_until, "valid_until")
        if valid_from >= valid_until:
            raise ValueError("grant validity window must be non-empty")
        idempotency_key = _string(idempotency_key, "idempotency_key")
        scope_key = _scope_key(request.target.scope)
        idempotency_address = (scope_key, idempotency_key)
        spec = _GrantCreationSpec(
            request_bytes=request_bytes,
            valid_from=valid_from,
            valid_until=valid_until,
            idempotency_key=idempotency_key,
        )

        with self.__lock:
            existing_hash = self.__grant_hash_by_idempotency.get(idempotency_address)
            if existing_hash is not None:
                existing = self._validated_grant_locked(existing_hash)
                existing_spec = _GrantCreationSpec(
                    request_bytes=existing.request.canonical_bytes(),
                    valid_from=existing.valid_from,
                    valid_until=existing.valid_until,
                    idempotency_key=existing.idempotency_key,
                )
                if existing_spec != spec:
                    raise MemoryScopeGrantConflictError(
                        "grant idempotency key has a different request"
                    )
                return _clone_grant(existing)

            if request_bytes in self.__grant_hash_by_request:
                raise MemoryScopeGrantConflictError(
                    "exact Memory grant request already has lifetime history"
                )

            granted_at = self._sample_clock_locked()
            if not valid_from <= granted_at < valid_until:
                raise MemoryScopeGrantConflictError(
                    "grant validity window must contain its creation time"
                )
            grant = MemoryScopeGrantV1.create(
                request=request,
                resolver_id=self.__resolver_id,
                resolver_version_sha256=self.__resolver_version_sha256,
                resolver_config_sha256=self.__resolver_config_sha256,
                valid_from=valid_from,
                valid_until=valid_until,
                evaluated_at=granted_at,
                granted_at=granted_at,
                idempotency_key=idempotency_key,
            )
            internal = _clone_grant(grant)
            if grant.content_hash in self.__grant_by_hash:
                raise MemoryScopeGrantConflictError(
                    "grant full hash already has immutable history"
                )
            display_address = (scope_key, grant.grant_id)
            display_collision = self.__grant_hash_by_display_id.get(display_address)
            if display_collision is not None:
                raise MemoryScopeGrantConflictError(
                    "grant display ID already has immutable history"
                )
            _atomic_publish(
                mapping_writes=(
                    (self.__grant_by_hash, grant.content_hash, internal),
                    (
                        self.__grant_hash_by_request,
                        request_bytes,
                        grant.content_hash,
                    ),
                    (
                        self.__grant_hash_by_idempotency,
                        idempotency_address,
                        grant.content_hash,
                    ),
                    (
                        self.__grant_hash_by_display_id,
                        display_address,
                        grant.content_hash,
                    ),
                )
            )
            return _clone_grant(internal)

    def revoke_grant(
        self,
        scope: MemoryScope,
        grant_id: str,
        *,
        grant_content_sha256: str,
        reason: MemoryScopeGrantRevocationReasonV1,
        reason_detail_sha256: str | None = None,
        idempotency_key: str,
    ) -> MemoryScopeGrantRevocationV1:
        self._reject_clock_reentry()
        scope_key = _scope_key(scope)
        grant_id, grant_hash = _control_grant_reference(
            grant_id,
            grant_content_sha256,
        )
        reason = _reason(reason)
        detail = _optional_digest(reason_detail_sha256, "reason_detail_sha256")
        if reason is MemoryScopeGrantRevocationReasonV1.OTHER and detail is None:
            raise ValueError("OTHER revocation reason requires reason_detail_sha256")
        idempotency_key = _string(idempotency_key, "idempotency_key")
        idempotency_address = (scope_key, idempotency_key)
        spec = _RevocationSpec(
            grant_id=grant_id,
            grant_content_sha256=grant_hash,
            reason=reason,
            reason_detail_sha256=detail,
            idempotency_key=idempotency_key,
        )

        with self.__lock:
            existing_hash = self.__revocation_hash_by_idempotency.get(
                idempotency_address
            )
            if existing_hash is not None:
                existing = self._validated_revocation_locked(existing_hash)
                existing_grant = self._grant_locked(
                    scope_key,
                    existing.grant_id,
                    existing.grant_content_sha256,
                )
                linked = self._revocation_locked(existing_grant)
                if linked is None or linked.content_hash != existing_hash:
                    raise MemoryScopeGrantConflictError("revocation indexes disagree")
                existing_spec = _RevocationSpec(
                    grant_id=existing.grant_id,
                    grant_content_sha256=existing.grant_content_sha256,
                    reason=existing.reason,
                    reason_detail_sha256=existing.reason_detail_sha256,
                    idempotency_key=existing.idempotency_key,
                )
                if existing_spec != spec:
                    raise MemoryScopeGrantConflictError(
                        "revocation idempotency key has a different request"
                    )
                return _clone_revocation(linked)

            grant = self._grant_locked(scope_key, grant_id, grant_hash)
            if self._revocation_locked(grant) is not None:
                raise MemoryScopeGrantConflictError(
                    "Memory scope grant is already irreversibly revoked"
                )
            revoked_at = self._sample_clock_locked()
            revocation = MemoryScopeGrantRevocationV1.create(
                scope=grant.request.target.scope,
                grant_id=grant.grant_id,
                grant_content_sha256=grant.content_hash,
                request_content_sha256=sha256(
                    grant.request.canonical_bytes()
                ).hexdigest(),
                resolver_id=self.__resolver_id,
                resolver_version_sha256=self.__resolver_version_sha256,
                resolver_config_sha256=self.__resolver_config_sha256,
                revoker_id=self.__revoker_id,
                revoker_version_sha256=self.__revoker_version_sha256,
                revoker_config_sha256=self.__revoker_config_sha256,
                reason=reason,
                reason_detail_sha256=detail,
                revoked_at=revoked_at,
                idempotency_key=idempotency_key,
            )
            internal = _clone_revocation(revocation)
            if revocation.content_hash in self.__revocation_by_hash:
                raise MemoryScopeGrantConflictError(
                    "revocation full hash already has immutable history"
                )
            display_address = (scope_key, revocation.revocation_id)
            display_collision = self.__revocation_hash_by_display_id.get(
                display_address
            )
            if display_collision is not None:
                raise MemoryScopeGrantConflictError(
                    "revocation display ID already has immutable history"
                )
            _atomic_publish(
                mapping_writes=(
                    (
                        self.__revocation_by_hash,
                        revocation.content_hash,
                        internal,
                    ),
                    (
                        self.__revocation_hash_by_grant_hash,
                        grant_hash,
                        revocation.content_hash,
                    ),
                    (
                        self.__revocation_hash_by_idempotency,
                        idempotency_address,
                        revocation.content_hash,
                    ),
                    (
                        self.__revocation_hash_by_display_id,
                        display_address,
                        revocation.content_hash,
                    ),
                )
            )
            return _clone_revocation(internal)

    def get_grant_for_audit(
        self,
        scope: MemoryScope,
        grant_id: str,
        *,
        grant_content_sha256: str,
    ) -> MemoryScopeGrantV1:
        """Return detached history; the receipt grants no authority."""

        self._reject_clock_reentry()
        scope_key = _scope_key(scope)
        grant_id, grant_hash = _control_grant_reference(
            grant_id,
            grant_content_sha256,
        )
        with self.__lock:
            return self._grant_locked(scope_key, grant_id, grant_hash)

    def get_grant_revocation_for_audit(
        self,
        scope: MemoryScope,
        grant_id: str,
        *,
        grant_content_sha256: str,
    ) -> MemoryScopeGrantRevocationV1:
        """Return a detached tombstone; absence never means active authority."""

        self._reject_clock_reentry()
        scope_key = _scope_key(scope)
        grant_id, grant_hash = _control_grant_reference(
            grant_id,
            grant_content_sha256,
        )
        with self.__lock:
            grant = self._grant_locked(scope_key, grant_id, grant_hash)
            revocation = self._revocation_locked(grant)
            if revocation is None:
                raise MemoryScopeGrantNotFoundError(
                    "Memory scope grant revocation was not found"
                )
            return revocation

    def resolve_active_grant(
        self,
        request: MemoryScopeGrantRequestV1,
    ) -> MemoryScopeGrantV1:
        """Resolve exact authority at one lock/clock linearization point."""

        self._reject_clock_reentry()
        request = _clone_request(request)
        request_bytes = request.canonical_bytes()
        with self.__lock:
            grant_hash = self.__grant_hash_by_request.get(request_bytes)
            if grant_hash is None:
                raise MemoryScopeAuthorizationDeniedError(
                    "active Memory scope grant is unavailable"
                )
            grant = self._validated_grant_locked(grant_hash)
            if grant.request.canonical_bytes() != request_bytes:
                raise MemoryScopeGrantConflictError("grant indexes disagree")
            if self._revocation_locked(grant) is not None:
                raise MemoryScopeAuthorizationDeniedError(
                    "active Memory scope grant is unavailable"
                )
            now = self._sample_clock_locked()
            if grant.granted_at > now:
                raise MemoryScopeGrantConflictError(
                    "stored grant has a future grant timestamp"
                )
            if not grant.valid_from <= now < grant.valid_until:
                raise MemoryScopeAuthorizationDeniedError(
                    "active Memory scope grant is unavailable"
                )
            return _clone_grant(grant)


__all__ = [
    "InMemoryMemoryScopeGrantStore",
    "MemoryScopeGrantConflictError",
    "MemoryScopeGrantNotFoundError",
    "MemoryScopeGrantRevocationReasonV1",
    "MemoryScopeGrantRevocationV1",
    "MemoryScopeGrantStore",
]
