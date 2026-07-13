# SPDX-License-Identifier: Apache-2.0

"""Server-owned authorization records for Agent Memory access.

Hop authentication answers *which service called this Worker*.  An assignment
pin answers *which immutable Memory execution snapshot was requested*.  Neither
answers whether the authenticated end principal may use that snapshot.  This
module defines that separate, default-deny decision.

A request binds every authority-bearing dimension exactly:

* a server-authenticated principal;
* a server-minted session incarnation, not a caller session key alone;
* a server-minted Worker audience, independent of the Worker hop secret;
* the full assignment pin, including scope, rollout incarnation, and hash; and
* one action, either pinning the assignment or exposing Memory.

Grant records are audit values, not bearer credentials.  They have no wire
format and must never be accepted from Agent request JSON.  A trusted resolver
must return an active exact grant on every authorization call; the authorizer
does not cache a successful result as a lease.  Current Agent HTTP ingress does
not establish a principal or either server-minted incarnation, so merely adding
this module enables no Memory access.

These contracts narrow trusted host integration mistakes.  They are not a
sandbox against arbitrary Python running in the same process.
"""

from __future__ import annotations

import json
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from typing import TYPE_CHECKING, Protocol

from areal.v2.memory_service.errors import MemoryServiceError
from areal.v2.memory_service.types import MemoryScope

if TYPE_CHECKING:
    from .memory import MemoryAgentSessionPinV1

_SCHEMA_VERSION = 1
_DIGEST_LENGTH = 64


class MemoryScopeAuthorizationError(MemoryServiceError):
    """Base class for principal/session-to-Memory authorization failures."""


class MemoryScopeAuthorizationDeniedError(MemoryScopeAuthorizationError):
    """Raised when no active exact grant authorizes one requested action."""


class MemoryScopeAuthorizationDisabledError(MemoryScopeAuthorizationError):
    """Raised when the host has not configured a trusted grant resolver."""


class MemoryScopeAuthorizationConflictError(MemoryScopeAuthorizationError):
    """Raised when a trusted resolver returns a malformed or different grant."""


def _canonical_json_bytes(value: dict[str, object]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def _record_bytes(record_kind: str, value: dict[str, object]) -> bytes:
    return _canonical_json_bytes(
        {
            "record_kind": record_kind,
            "schema_version": _SCHEMA_VERSION,
            **value,
        }
    )


def _string(value: object, field_name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be a str")
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")
    try:
        value.encode("utf-8", "strict")
    except UnicodeEncodeError as error:
        raise ValueError(f"{field_name} must be valid UTF-8") from error
    return value


def _digest(value: object, field_name: str) -> str:
    value = _string(value, field_name)
    if len(value) != _DIGEST_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")
    return value


def _random_identity(value: object, field_name: str, prefix: str) -> str:
    value = _string(value, field_name)
    suffix = value.removeprefix(prefix)
    if (
        not value.startswith(prefix)
        or len(suffix) != _DIGEST_LENGTH
        or any(character not in "0123456789abcdef" for character in suffix)
    ):
        raise ValueError(
            f"{field_name} must be {prefix} followed by 64 lowercase hex characters"
        )
    return value


def _aware_datetime(value: object, field_name: str) -> datetime:
    if type(value) is not datetime:
        raise TypeError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    try:
        normalized = value.astimezone(UTC)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{field_name} must be normalizable to UTC") from error
    return datetime(
        normalized.year,
        normalized.month,
        normalized.day,
        normalized.hour,
        normalized.minute,
        normalized.second,
        normalized.microsecond,
        tzinfo=UTC,
    )


def _datetime_value(value: object, field_name: str) -> str:
    return _aware_datetime(value, field_name).isoformat()


def _scope(value: object) -> MemoryScope:
    if type(value) is not MemoryScope:
        raise TypeError("scope must be a MemoryScope")
    return value


def _scope_value(value: object) -> dict[str, str]:
    scope = _scope(value)
    return {
        "namespace": _string(scope.namespace, "scope.namespace"),
        "subject_id": _string(scope.subject_id, "scope.subject_id"),
        "tenant_id": _string(scope.tenant_id, "scope.tenant_id"),
    }


@dataclass(frozen=True, slots=True)
class MemoryPrincipalV1:
    """Stable identity established by a trusted server authenticator.

    ``issuer`` namespaces ``subject`` so equal subject strings from different
    identity systems cannot collide.  Neither value may be copied from an
    unverified request body or inferred from a Memory scope.
    """

    issuer: str
    subject: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "issuer", _string(self.issuer, "issuer"))
        object.__setattr__(self, "subject", _string(self.subject, "subject"))

    def _canonical_value(self) -> dict[str, str]:
        return {
            "issuer": _string(self.issuer, "issuer"),
            "subject": _string(self.subject, "subject"),
        }

    def canonical_bytes(self) -> bytes:
        return _record_bytes("memory_principal", self._canonical_value())


def _principal(value: object) -> MemoryPrincipalV1:
    if type(value) is not MemoryPrincipalV1:
        raise TypeError("principal must be a MemoryPrincipalV1")
    value.canonical_bytes()
    return value


@dataclass(frozen=True, slots=True)
class MemoryWorkerAudienceV1:
    """Non-secret replay domain for one Worker/pair process incarnation.

    The host mints a fresh 256-bit value at Worker startup.  It must not be a
    pair index, network address, hop key, or hash of that key.  Knowledge of the
    value grants nothing; the resolver still decides every request.
    """

    audience_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "audience_id",
            _random_identity(self.audience_id, "audience_id", "maud_"),
        )

    @classmethod
    def create(cls) -> MemoryWorkerAudienceV1:
        return cls(f"maud_{secrets.token_hex(32)}")

    def _canonical_value(self) -> dict[str, str]:
        return {
            "audience_id": _random_identity(
                self.audience_id,
                "audience_id",
                "maud_",
            )
        }

    def canonical_bytes(self) -> bytes:
        return _record_bytes("memory_worker_audience", self._canonical_value())


def _audience(value: object) -> MemoryWorkerAudienceV1:
    if type(value) is not MemoryWorkerAudienceV1:
        raise TypeError("audience must be a MemoryWorkerAudienceV1")
    value.canonical_bytes()
    return value


@dataclass(frozen=True, slots=True)
class MemorySessionIncarnationV1:
    """One server-minted lifetime of a caller-visible session key.

    Closing and reopening the same textual key must mint a different
    ``incarnation_id``.  The random ID is an anti-replay identity, not a
    credential, and must never be accepted from untrusted session metadata.
    """

    session_key: str
    incarnation_id: str

    def __post_init__(self) -> None:
        session_key = _string(self.session_key, "session_key")
        incarnation_id = _random_identity(
            self.incarnation_id,
            "incarnation_id",
            "msinc_",
        )
        if incarnation_id == session_key:
            raise ValueError("incarnation_id must be independent of session_key")
        object.__setattr__(self, "session_key", session_key)
        object.__setattr__(self, "incarnation_id", incarnation_id)

    @classmethod
    def create(cls, session_key: str) -> MemorySessionIncarnationV1:
        return cls(
            session_key=session_key,
            incarnation_id=f"msinc_{secrets.token_hex(32)}",
        )

    def _canonical_value(self) -> dict[str, str]:
        session_key = _string(self.session_key, "session_key")
        incarnation_id = _random_identity(
            self.incarnation_id,
            "incarnation_id",
            "msinc_",
        )
        if incarnation_id == session_key:
            raise ValueError("incarnation_id must be independent of session_key")
        return {
            "incarnation_id": incarnation_id,
            "session_key": session_key,
        }

    def canonical_bytes(self) -> bytes:
        return _record_bytes("memory_session_incarnation", self._canonical_value())


def _session(value: object) -> MemorySessionIncarnationV1:
    if type(value) is not MemorySessionIncarnationV1:
        raise TypeError("session must be a MemorySessionIncarnationV1")
    value.canonical_bytes()
    return value


@dataclass(frozen=True, slots=True)
class MemoryAssignmentGrantTargetV1:
    """The full immutable assignment pin narrowed by one grant.

    Scope-only authorization is insufficient: changing an assignment inside
    the same scope can change the release, retriever, renderer, or consumer and
    would invalidate both access control and causal experiment attribution.
    """

    scope: MemoryScope
    rollout_group_id: str
    rollout_group_incarnation_sha256: str
    assignment_id: str
    assignment_content_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", _scope(self.scope))
        object.__setattr__(
            self,
            "rollout_group_id",
            _string(self.rollout_group_id, "rollout_group_id"),
        )
        object.__setattr__(
            self,
            "rollout_group_incarnation_sha256",
            _digest(
                self.rollout_group_incarnation_sha256,
                "rollout_group_incarnation_sha256",
            ),
        )
        assignment_hash = _digest(
            self.assignment_content_sha256,
            "assignment_content_sha256",
        )
        assignment_id = _string(self.assignment_id, "assignment_id")
        if assignment_id != f"masn_{assignment_hash[:24]}":
            raise ValueError("assignment_id disagrees with assignment_content_sha256")
        object.__setattr__(self, "assignment_id", assignment_id)
        object.__setattr__(
            self,
            "assignment_content_sha256",
            assignment_hash,
        )

    @classmethod
    def from_session_pin(
        cls,
        pin: MemoryAgentSessionPinV1,
    ) -> MemoryAssignmentGrantTargetV1:
        # Local import keeps the later coordinator integration free to import
        # this authorization module without creating an import cycle.
        from .memory import MemoryAgentSessionPinV1

        if type(pin) is not MemoryAgentSessionPinV1:
            raise TypeError("pin must be a MemoryAgentSessionPinV1")
        return cls(
            scope=pin.scope,
            rollout_group_id=pin.rollout_group_id,
            rollout_group_incarnation_sha256=(pin.rollout_group_incarnation_sha256),
            assignment_id=pin.assignment_id,
            assignment_content_sha256=pin.assignment_content_sha256,
        )

    def _canonical_value(self) -> dict[str, object]:
        assignment_hash = _digest(
            self.assignment_content_sha256,
            "assignment_content_sha256",
        )
        assignment_id = _string(self.assignment_id, "assignment_id")
        if assignment_id != f"masn_{assignment_hash[:24]}":
            raise ValueError("assignment_id disagrees with assignment_content_sha256")
        return {
            "assignment_content_sha256": assignment_hash,
            "assignment_id": assignment_id,
            "rollout_group_id": _string(
                self.rollout_group_id,
                "rollout_group_id",
            ),
            "rollout_group_incarnation_sha256": _digest(
                self.rollout_group_incarnation_sha256,
                "rollout_group_incarnation_sha256",
            ),
            "scope": _scope_value(self.scope),
        }

    def canonical_bytes(self) -> bytes:
        return _record_bytes(
            "memory_assignment_grant_target",
            self._canonical_value(),
        )


def _target(value: object) -> MemoryAssignmentGrantTargetV1:
    if type(value) is not MemoryAssignmentGrantTargetV1:
        raise TypeError("target must be a MemoryAssignmentGrantTargetV1")
    value.canonical_bytes()
    return value


class MemoryScopeActionV1(StrEnum):
    """Single-purpose actions; pinning never implies actual exposure."""

    PIN_ASSIGNMENT = "pin_assignment"
    EXPOSE_MEMORY = "expose_memory"


def _action(value: object) -> MemoryScopeActionV1:
    if type(value) is not MemoryScopeActionV1:
        raise TypeError("action must be a MemoryScopeActionV1")
    return value


@dataclass(frozen=True, slots=True)
class MemoryScopeGrantRequestV1:
    """Exact server-side authorization input; never a caller wire DTO."""

    principal: MemoryPrincipalV1
    session: MemorySessionIncarnationV1
    audience: MemoryWorkerAudienceV1
    target: MemoryAssignmentGrantTargetV1
    action: MemoryScopeActionV1

    def __post_init__(self) -> None:
        object.__setattr__(self, "principal", _principal(self.principal))
        object.__setattr__(self, "session", _session(self.session))
        object.__setattr__(self, "audience", _audience(self.audience))
        object.__setattr__(self, "target", _target(self.target))
        object.__setattr__(self, "action", _action(self.action))

    def _canonical_value(self) -> dict[str, object]:
        return {
            "action": _action(self.action).value,
            "audience": _audience(self.audience)._canonical_value(),
            "principal": _principal(self.principal)._canonical_value(),
            "session": _session(self.session)._canonical_value(),
            "target": _target(self.target)._canonical_value(),
        }

    def canonical_bytes(self) -> bytes:
        return _record_bytes("memory_scope_grant_request", self._canonical_value())


def _request(value: object) -> MemoryScopeGrantRequestV1:
    if type(value) is not MemoryScopeGrantRequestV1:
        raise TypeError("request must be a MemoryScopeGrantRequestV1")
    value.canonical_bytes()
    return value


@dataclass(frozen=True, slots=True)
class MemoryScopeGrantV1:
    """Canonical audit receipt returned by a trusted active-grant resolver.

    ``grant_id`` and ``content_hash`` are integrity pointers only.  A caller
    cannot present either value to obtain access.
    """

    grant_id: str
    request: MemoryScopeGrantRequestV1
    resolver_id: str
    resolver_version_sha256: str
    resolver_config_sha256: str
    valid_from: datetime
    valid_until: datetime
    evaluated_at: datetime
    granted_at: datetime
    idempotency_key: str
    content_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "request", _request(self.request))
        for field_name in ("resolver_id", "idempotency_key"):
            object.__setattr__(
                self,
                field_name,
                _string(getattr(self, field_name), field_name),
            )
        for field_name in (
            "resolver_version_sha256",
            "resolver_config_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _digest(getattr(self, field_name), field_name),
            )
        for field_name in (
            "valid_from",
            "valid_until",
            "evaluated_at",
            "granted_at",
        ):
            object.__setattr__(
                self,
                field_name,
                _aware_datetime(getattr(self, field_name), field_name),
            )
        if self.valid_from >= self.valid_until:
            raise ValueError("grant validity window must be non-empty")
        if self.evaluated_at > self.granted_at:
            raise ValueError("evaluated_at must not follow granted_at")
        if self.granted_at >= self.valid_until:
            raise ValueError("granted_at must precede valid_until")
        self.verify_integrity()

    def _canonical_value(self) -> dict[str, object]:
        valid_from = _aware_datetime(self.valid_from, "valid_from")
        valid_until = _aware_datetime(self.valid_until, "valid_until")
        evaluated_at = _aware_datetime(self.evaluated_at, "evaluated_at")
        granted_at = _aware_datetime(self.granted_at, "granted_at")
        if valid_from >= valid_until:
            raise ValueError("grant validity window must be non-empty")
        if evaluated_at > granted_at:
            raise ValueError("evaluated_at must not follow granted_at")
        if granted_at >= valid_until:
            raise ValueError("granted_at must precede valid_until")
        return {
            "resolver_config_sha256": _digest(
                self.resolver_config_sha256,
                "resolver_config_sha256",
            ),
            "resolver_id": _string(self.resolver_id, "resolver_id"),
            "resolver_version_sha256": _digest(
                self.resolver_version_sha256,
                "resolver_version_sha256",
            ),
            "evaluated_at": evaluated_at.isoformat(),
            "granted_at": granted_at.isoformat(),
            "idempotency_key": _string(
                self.idempotency_key,
                "idempotency_key",
            ),
            "request": _request(self.request)._canonical_value(),
            "valid_from": valid_from.isoformat(),
            "valid_until": valid_until.isoformat(),
        }

    def canonical_bytes(self) -> bytes:
        return _record_bytes("memory_scope_grant", self._canonical_value())

    def verify_integrity(self) -> None:
        canonical = self.canonical_bytes()
        expected_hash = sha256(canonical).hexdigest()
        content_hash = _digest(self.content_hash, "content_hash")
        if content_hash != expected_hash:
            raise ValueError("content_hash disagrees with canonical grant bytes")
        grant_id = _string(self.grant_id, "grant_id")
        if grant_id != f"msgr_{expected_hash[:24]}":
            raise ValueError("grant_id disagrees with canonical grant bytes")

    @classmethod
    def create(
        cls,
        *,
        request: MemoryScopeGrantRequestV1,
        resolver_id: str,
        resolver_version_sha256: str,
        resolver_config_sha256: str,
        valid_from: datetime,
        valid_until: datetime,
        evaluated_at: datetime,
        granted_at: datetime,
        idempotency_key: str,
    ) -> MemoryScopeGrantV1:
        request = _request(request)
        values = {
            "request": request,
            "resolver_id": _string(resolver_id, "resolver_id"),
            "resolver_version_sha256": _digest(
                resolver_version_sha256,
                "resolver_version_sha256",
            ),
            "resolver_config_sha256": _digest(
                resolver_config_sha256,
                "resolver_config_sha256",
            ),
            "valid_from": _aware_datetime(valid_from, "valid_from"),
            "valid_until": _aware_datetime(valid_until, "valid_until"),
            "evaluated_at": _aware_datetime(evaluated_at, "evaluated_at"),
            "granted_at": _aware_datetime(granted_at, "granted_at"),
            "idempotency_key": _string(idempotency_key, "idempotency_key"),
        }
        canonical_value = {
            "resolver_config_sha256": values["resolver_config_sha256"],
            "resolver_id": values["resolver_id"],
            "resolver_version_sha256": values["resolver_version_sha256"],
            "evaluated_at": _datetime_value(values["evaluated_at"], "evaluated_at"),
            "granted_at": _datetime_value(values["granted_at"], "granted_at"),
            "idempotency_key": values["idempotency_key"],
            "request": request._canonical_value(),
            "valid_from": _datetime_value(values["valid_from"], "valid_from"),
            "valid_until": _datetime_value(values["valid_until"], "valid_until"),
        }
        content_hash = sha256(
            _record_bytes("memory_scope_grant", canonical_value)
        ).hexdigest()
        return cls(
            grant_id=f"msgr_{content_hash[:24]}",
            content_hash=content_hash,
            **values,  # type: ignore[arg-type]
        )


class MemoryScopeGrantResolver(Protocol):
    """Trusted backend that resolves one exact request at its linearization point.

    Missing, expired, or revoked grants raise
    :class:`MemoryScopeAuthorizationDeniedError`.  Implementations must not
    treat a grant ID or hash supplied by a caller as authority.
    """

    resolver_id: str
    resolver_version_sha256: str
    resolver_config_sha256: str

    def resolve_active_grant(
        self,
        request: MemoryScopeGrantRequestV1,
    ) -> MemoryScopeGrantV1: ...


def _utc_now() -> datetime:
    return datetime.now(UTC)


class MemoryScopeGrantAuthorizer:
    """Default-deny exact-grant verifier with no positive-result cache."""

    def __init__(
        self,
        resolver: MemoryScopeGrantResolver | None = None,
        *,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        if not callable(clock):
            raise TypeError("clock must be callable")
        resolve = None
        resolver_identity: tuple[str, str, str] | None = None
        if resolver is not None:
            resolve = getattr(resolver, "resolve_active_grant", None)
            if not callable(resolve):
                raise TypeError("resolver must define resolve_active_grant")
            resolver_identity = self._read_resolver_identity(resolver)
        # Snapshot the trusted method so later attribute replacement cannot
        # silently swap the authorization component selected by the host.
        self.__resolve: (
            Callable[[MemoryScopeGrantRequestV1], MemoryScopeGrantV1] | None
        ) = resolve
        self.__resolver = resolver
        self.__resolver_identity = resolver_identity
        self.__clock = clock

    @staticmethod
    def _read_resolver_identity(resolver: object) -> tuple[str, str, str]:
        return (
            _string(getattr(resolver, "resolver_id", None), "resolver_id"),
            _digest(
                getattr(resolver, "resolver_version_sha256", None),
                "resolver_version_sha256",
            ),
            _digest(
                getattr(resolver, "resolver_config_sha256", None),
                "resolver_config_sha256",
            ),
        )

    def _verify_resolver_identity(self) -> None:
        if self.__resolver is None or self.__resolver_identity is None:
            return
        try:
            current = self._read_resolver_identity(self.__resolver)
        except (TypeError, ValueError) as error:
            raise MemoryScopeAuthorizationConflictError(
                "grant resolver identity became invalid"
            ) from error
        if current != self.__resolver_identity:
            raise MemoryScopeAuthorizationConflictError(
                "grant resolver identity changed after configuration"
            )

    @staticmethod
    def _verify_request_snapshot(
        request: MemoryScopeGrantRequestV1,
        expected: bytes,
    ) -> None:
        try:
            current = request.canonical_bytes()
        except (TypeError, ValueError) as error:
            raise MemoryScopeAuthorizationConflictError(
                "grant resolver corrupted the authorization request"
            ) from error
        if current != expected:
            raise MemoryScopeAuthorizationConflictError(
                "grant resolver mutated the authorization request"
            )

    def authorize(
        self,
        request: MemoryScopeGrantRequestV1,
    ) -> MemoryScopeGrantV1:
        """Resolve and verify one action without converting it into a lease."""

        request = _request(request)
        request_snapshot = request.canonical_bytes()
        if self.__resolve is None:
            raise MemoryScopeAuthorizationDisabledError(
                "Memory scope authorization is not configured"
            )

        self._verify_resolver_identity()
        try:
            grant = self.__resolve(request)
        finally:
            self._verify_resolver_identity()
            self._verify_request_snapshot(request, request_snapshot)
        if type(grant) is not MemoryScopeGrantV1:
            raise MemoryScopeAuthorizationConflictError(
                "grant resolver returned a non-canonical value"
            )
        try:
            grant.verify_integrity()
            if grant.request.canonical_bytes() != request_snapshot:
                raise ValueError("resolved grant does not match the exact request")
            if (
                grant.resolver_id,
                grant.resolver_version_sha256,
                grant.resolver_config_sha256,
            ) != self.__resolver_identity:
                raise ValueError("resolved grant does not match the trusted resolver")
        except (TypeError, ValueError) as error:
            raise MemoryScopeAuthorizationConflictError(
                "grant resolver returned a malformed or different grant"
            ) from error

        try:
            now = _aware_datetime(self.__clock(), "authorization clock")
        except (TypeError, ValueError) as error:
            raise MemoryScopeAuthorizationConflictError(
                "authorization clock returned an invalid value"
            ) from error
        if grant.granted_at > now:
            raise MemoryScopeAuthorizationConflictError(
                "active grant has a future grant timestamp"
            )
        if not grant.valid_from <= now < grant.valid_until:
            raise MemoryScopeAuthorizationDeniedError(
                "active Memory scope grant is unavailable"
            )
        return grant


__all__ = [
    "MemoryAssignmentGrantTargetV1",
    "MemoryPrincipalV1",
    "MemoryScopeActionV1",
    "MemoryScopeAuthorizationConflictError",
    "MemoryScopeAuthorizationDeniedError",
    "MemoryScopeAuthorizationDisabledError",
    "MemoryScopeAuthorizationError",
    "MemoryScopeGrantAuthorizer",
    "MemoryScopeGrantRequestV1",
    "MemoryScopeGrantResolver",
    "MemoryScopeGrantV1",
    "MemorySessionIncarnationV1",
    "MemoryWorkerAudienceV1",
]
