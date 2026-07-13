# SPDX-License-Identifier: Apache-2.0

"""Strict Agent Service wire transport for immutable Memory assignment pins.

This module transports assignment *references*, never exposure or authorization
claims.  A trusted rollout control-plane caller may ask a DataProxy to bind one
pin to a session; the DataProxy then forwards a closed-schema reserved envelope.
This transport layer does not yet install a Worker coordinator: a future
Worker-owned integration must parse the envelope and call
``AsyncMemoryAgentCoordinator.pin_session`` so the trusted control store can
resolve the assignment again.  Until then, the raw envelope remains ordinary
Worker request metadata.  Parsing alone does not retrieve, render, inject,
acknowledge, or authorize access to Memory.

The active-assignment resolver proves integrity, expiry, and revocation state;
it cannot prove that the caller may read the referenced tenant or subject.
Deployments must separately bind an authenticated principal and session to the
requested :class:`MemoryScope` at ingress.  The current per-turn
``memory_control_authorized`` boolean is only a trusted internal-hop assertion;
it is not a credential and cannot establish principal-to-scope authorization.
DataProxy accepts a true assertion only with the authenticated internal hop
header.  The controller generates a dedicated credential for this hop instead
of reusing the externally configured Agent admin key; standalone deployments
that omit it keep ordinary turns enabled but fail closed for Memory pins.
The Gateway also disables Memory control while the external admin credential
is either source-visible project default, so compatibility defaults cannot
authorize cross-scope assignment selection. The same public values are rejected
as internal hop credentials.
Deployments should still isolate this hop and may replace the credential with
mTLS.

The process-local value contract for that separate decision lives in
``memory_authorization``.  Current routes do not create its trusted principal,
session incarnation, or Worker audience and do not invoke its default-disabled
resolver.  Its grant records have no wire representation and must not be added
to this assignment-pin envelope.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

from areal.v2.agent_service.memory import MemoryAgentSessionPinV1
from areal.v2.memory_service.types import MemoryScope

MEMORY_ASSIGNMENT_PIN_FIELD = "memory_assignment_pin"
MEMORY_CONTROL_AUTHORIZED_FIELD = "memory_control_authorized"
AREAL_MEMORY_METADATA_KEY = "areal_memory"
AREAL_INFERENCE_METADATA_KEY = "areal_inference"
CHAT_REQUEST_METADATA_KEY = "chat_request"

_SCHEMA_VERSION = 1
_PIN_KEYS = frozenset(
    {
        "schema_version",
        "scope",
        "rollout_group_id",
        "rollout_group_incarnation_sha256",
        "assignment_id",
        "assignment_content_sha256",
    }
)
_SCOPE_KEYS = frozenset({"tenant_id", "namespace", "subject_id"})
_METADATA_KEYS = frozenset({"schema_version", "assignment_pin"})
_PIN_OMITTED = object()


class MemoryPinTransportError(ValueError):
    """Base class for malformed or conflicting Memory pin transport."""


class MemoryPinWireFormatError(MemoryPinTransportError):
    """Raised when a Memory pin or reserved envelope is not closed-schema."""


class MemorySessionPinConflictError(MemoryPinTransportError):
    """Raised when a session already has a different immutable pin."""


class ReservedMemoryMetadataError(MemoryPinTransportError):
    """Raised when caller metadata attempts to set a reserved Agent field."""


def _exact_dict(value: object, field_name: str) -> dict[str, object]:
    if type(value) is not dict:
        raise MemoryPinWireFormatError(f"{field_name} must be an object")
    if any(type(key) is not str for key in value):
        raise MemoryPinWireFormatError(f"{field_name} keys must be strings")
    return value


def _closed_keys(
    value: dict[str, object],
    expected: frozenset[str],
    field_name: str,
) -> None:
    keys = set(value)
    if keys != expected:
        missing = sorted(expected - keys)
        unknown = sorted(keys - expected)
        details: list[str] = []
        if missing:
            details.append(f"missing={missing!r}")
        if unknown:
            details.append(f"unknown={unknown!r}")
        raise MemoryPinWireFormatError(
            f"{field_name} must use the exact V1 schema ({', '.join(details)})"
        )


def _schema_version(value: object, field_name: str) -> None:
    if type(value) is not int or value != _SCHEMA_VERSION:
        raise MemoryPinWireFormatError(f"{field_name} must be integer 1")


def _wire_string(value: object, field_name: str) -> str:
    if type(value) is not str:
        raise MemoryPinWireFormatError(f"{field_name} must be a string")
    return value


@dataclass(frozen=True, slots=True)
class MemoryAssignmentPinWireV1:
    """Closed-schema JSON representation of one immutable assignment pin."""

    pin: MemoryAgentSessionPinV1

    def __post_init__(self) -> None:
        if type(self.pin) is not MemoryAgentSessionPinV1:
            raise TypeError("pin must be a MemoryAgentSessionPinV1")

    @classmethod
    def from_wire(cls, value: object) -> MemoryAssignmentPinWireV1:
        wire = _exact_dict(value, MEMORY_ASSIGNMENT_PIN_FIELD)
        _closed_keys(wire, _PIN_KEYS, MEMORY_ASSIGNMENT_PIN_FIELD)
        _schema_version(
            wire["schema_version"],
            f"{MEMORY_ASSIGNMENT_PIN_FIELD}.schema_version",
        )
        scope_wire = _exact_dict(
            wire["scope"],
            f"{MEMORY_ASSIGNMENT_PIN_FIELD}.scope",
        )
        _closed_keys(
            scope_wire,
            _SCOPE_KEYS,
            f"{MEMORY_ASSIGNMENT_PIN_FIELD}.scope",
        )
        try:
            pin = MemoryAgentSessionPinV1(
                scope=MemoryScope(
                    tenant_id=_wire_string(
                        scope_wire["tenant_id"],
                        f"{MEMORY_ASSIGNMENT_PIN_FIELD}.scope.tenant_id",
                    ),
                    namespace=_wire_string(
                        scope_wire["namespace"],
                        f"{MEMORY_ASSIGNMENT_PIN_FIELD}.scope.namespace",
                    ),
                    subject_id=_wire_string(
                        scope_wire["subject_id"],
                        f"{MEMORY_ASSIGNMENT_PIN_FIELD}.scope.subject_id",
                    ),
                ),
                rollout_group_id=_wire_string(
                    wire["rollout_group_id"],
                    f"{MEMORY_ASSIGNMENT_PIN_FIELD}.rollout_group_id",
                ),
                rollout_group_incarnation_sha256=_wire_string(
                    wire["rollout_group_incarnation_sha256"],
                    f"{MEMORY_ASSIGNMENT_PIN_FIELD}.rollout_group_incarnation_sha256",
                ),
                assignment_id=_wire_string(
                    wire["assignment_id"],
                    f"{MEMORY_ASSIGNMENT_PIN_FIELD}.assignment_id",
                ),
                assignment_content_sha256=_wire_string(
                    wire["assignment_content_sha256"],
                    f"{MEMORY_ASSIGNMENT_PIN_FIELD}.assignment_content_sha256",
                ),
            )
        except (TypeError, ValueError) as error:
            raise MemoryPinWireFormatError(
                f"{MEMORY_ASSIGNMENT_PIN_FIELD} contains invalid V1 values"
            ) from error
        return cls(pin)

    @classmethod
    def from_runtime_pin(
        cls,
        pin: MemoryAgentSessionPinV1,
    ) -> MemoryAssignmentPinWireV1:
        return cls(pin)

    def to_runtime_pin(self) -> MemoryAgentSessionPinV1:
        return self.pin

    def to_wire(self) -> dict[str, object]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "scope": {
                "tenant_id": self.pin.scope.tenant_id,
                "namespace": self.pin.scope.namespace,
                "subject_id": self.pin.scope.subject_id,
            },
            "rollout_group_id": self.pin.rollout_group_id,
            "rollout_group_incarnation_sha256": (
                self.pin.rollout_group_incarnation_sha256
            ),
            "assignment_id": self.pin.assignment_id,
            "assignment_content_sha256": self.pin.assignment_content_sha256,
        }


@dataclass(frozen=True, slots=True)
class MemoryAgentMetadataWireV1:
    """Reserved Worker metadata envelope; deliberately has no exposure field."""

    assignment_pin: MemoryAssignmentPinWireV1

    def __post_init__(self) -> None:
        if type(self.assignment_pin) is not MemoryAssignmentPinWireV1:
            raise TypeError("assignment_pin must be a MemoryAssignmentPinWireV1")

    @classmethod
    def from_wire(cls, value: object) -> MemoryAgentMetadataWireV1:
        wire = _exact_dict(value, AREAL_MEMORY_METADATA_KEY)
        _closed_keys(wire, _METADATA_KEYS, AREAL_MEMORY_METADATA_KEY)
        _schema_version(
            wire["schema_version"],
            f"{AREAL_MEMORY_METADATA_KEY}.schema_version",
        )
        return cls(MemoryAssignmentPinWireV1.from_wire(wire["assignment_pin"]))

    def to_wire(self) -> dict[str, object]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "assignment_pin": self.assignment_pin.to_wire(),
        }


class MemorySessionPinCache:
    """Thread-safe first-writer CAS for DataProxy session pin reuse."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._pin_by_session: dict[str, MemoryAssignmentPinWireV1] = {}

    @staticmethod
    def _session_key(value: object) -> str:
        if type(value) is not str:
            raise TypeError("session_key must be a str")
        if not value.strip():
            raise ValueError("session_key must not be blank")
        return value

    def resolve(
        self,
        session_key: str,
        submitted: MemoryAssignmentPinWireV1 | object = _PIN_OMITTED,
    ) -> MemoryAssignmentPinWireV1 | None:
        """Bind a submitted first pin or reuse the existing pin when omitted.

        DataProxy code parses a present JSON field before calling this method,
        so ``None`` is never treated as an omitted pin.
        """

        session_key = self._session_key(session_key)
        submitted_present = submitted is not _PIN_OMITTED
        if submitted_present and type(submitted) is not MemoryAssignmentPinWireV1:
            raise TypeError("a submitted pin must be a MemoryAssignmentPinWireV1")
        with self._lock:
            existing = self._pin_by_session.get(session_key)
            if not submitted_present:
                return existing
            if existing is None:
                self._pin_by_session[session_key] = submitted
                return submitted
            if existing == submitted:
                return existing
            raise MemorySessionPinConflictError(
                "session is already bound to a different Memory assignment pin"
            )

    def clear(self, session_key: str) -> None:
        session_key = self._session_key(session_key)
        with self._lock:
            self._pin_by_session.pop(session_key, None)

    def clear_all(self) -> None:
        with self._lock:
            self._pin_by_session.clear()


def copy_user_metadata(value: object) -> dict[str, object]:
    """Copy ordinary metadata while rejecting Agent transport namespaces."""

    metadata = _exact_dict(value, "metadata")
    reserved = tuple(
        key
        for key in (
            AREAL_MEMORY_METADATA_KEY,
            AREAL_INFERENCE_METADATA_KEY,
            CHAT_REQUEST_METADATA_KEY,
            MEMORY_CONTROL_AUTHORIZED_FIELD,
            MEMORY_ASSIGNMENT_PIN_FIELD,
        )
        if key in metadata
    )
    if reserved:
        raise ReservedMemoryMetadataError(
            f"metadata.{reserved[0]} is reserved for Memory transport"
        )
    return dict(metadata)


def inject_memory_assignment_pin(
    metadata: object,
    assignment_pin: MemoryAssignmentPinWireV1 | None,
) -> dict[str, object]:
    """Inject only a trusted pin envelope; never an exposure assertion."""

    result = copy_user_metadata(metadata)
    if assignment_pin is not None:
        if type(assignment_pin) is not MemoryAssignmentPinWireV1:
            raise TypeError("assignment_pin must be a MemoryAssignmentPinWireV1")
        result[AREAL_MEMORY_METADATA_KEY] = MemoryAgentMetadataWireV1(
            assignment_pin
        ).to_wire()
    return result


def parse_memory_assignment_pin_metadata(
    metadata: object,
) -> MemoryAgentSessionPinV1 | None:
    """Strict Worker/Agent helper for an optional reserved pin envelope.

    This helper is for the Worker-owned integration boundary.  The returned
    value is suitable for an explicit call to
    ``AsyncMemoryAgentCoordinator.pin_session(session_key, pin)``.  It performs
    no coordinator call, grants no caller authorization, and creates no
    exposure record.
    """

    wire = _exact_dict(metadata, "metadata")
    if AREAL_MEMORY_METADATA_KEY not in wire:
        return None
    envelope = MemoryAgentMetadataWireV1.from_wire(wire[AREAL_MEMORY_METADATA_KEY])
    return envelope.assignment_pin.to_runtime_pin()
