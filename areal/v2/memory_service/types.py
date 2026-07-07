# SPDX-License-Identifier: Apache-2.0

"""Immutable evidence value objects for the Memory Service."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

_MAX_SEQUENCE_NO = 2**63 - 1


def _validate_string(
    value: object, field_name: str, *, allow_blank: bool = False
) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    snapshot = str.__str__(value)
    if not allow_blank and not str.strip(snapshot):
        raise ValueError(f"{field_name} must not be blank")
    try:
        str.encode(snapshot, "utf-8", "strict")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{field_name} must be valid UTF-8") from exc
    return snapshot


def _validate_aware_datetime(value: object, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime")
    if value.tzinfo is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    try:
        offset = value.utcoffset()
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{field_name} must be normalizable to UTC") from exc
    if offset is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    try:
        normalized = datetime.astimezone(value, UTC)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{field_name} must be normalizable to UTC") from exc
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


class EvidenceKind(StrEnum):
    """The source or role represented by an evidence event."""

    USER_MESSAGE = "user_message"
    AGENT_MESSAGE = "agent_message"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ENVIRONMENT = "environment"
    FEEDBACK = "feedback"
    OUTCOME = "outcome"


@dataclass(frozen=True, slots=True)
class MemoryScope:
    """The tenant, namespace, and subject that own a memory."""

    tenant_id: str
    namespace: str
    subject_id: str

    def __post_init__(self) -> None:
        tenant_id = _validate_string(self.tenant_id, "tenant_id")
        namespace = _validate_string(self.namespace, "namespace")
        subject_id = _validate_string(self.subject_id, "subject_id")
        object.__setattr__(self, "tenant_id", tenant_id)
        object.__setattr__(self, "namespace", namespace)
        object.__setattr__(self, "subject_id", subject_id)


@dataclass(frozen=True, slots=True)
class EvidenceEvent:
    """An immutable observation submitted to the Memory Service."""

    scope: MemoryScope
    session_id: str
    run_id: str
    sequence_no: int
    kind: EvidenceKind
    payload: str
    observed_at: datetime
    idempotency_key: str

    def __post_init__(self) -> None:
        if type(self.scope) is not MemoryScope:
            raise TypeError("scope must be a MemoryScope")
        session_id = _validate_string(self.session_id, "session_id")
        run_id = _validate_string(self.run_id, "run_id")
        idempotency_key = _validate_string(self.idempotency_key, "idempotency_key")
        if type(self.sequence_no) is not int:
            raise TypeError("sequence_no must be an integer")
        if self.sequence_no < 0:
            raise ValueError("sequence_no must be non-negative")
        if self.sequence_no > _MAX_SEQUENCE_NO:
            raise ValueError("sequence_no must fit in a signed 64-bit integer")
        if not isinstance(self.kind, EvidenceKind):
            raise TypeError("kind must be an EvidenceKind")
        payload = _validate_string(self.payload, "payload", allow_blank=True)
        observed_at = _validate_aware_datetime(self.observed_at, "observed_at")
        object.__setattr__(self, "session_id", session_id)
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "idempotency_key", idempotency_key)
        object.__setattr__(self, "observed_at", observed_at)

    def canonical_bytes(self) -> bytes:
        """Serialize the event as deterministic, compact UTF-8 JSON."""

        value = {
            "scope": {
                "tenant_id": self.scope.tenant_id,
                "namespace": self.scope.namespace,
                "subject_id": self.scope.subject_id,
            },
            "session_id": self.session_id,
            "run_id": self.run_id,
            "sequence_no": self.sequence_no,
            "kind": self.kind.value,
            "payload": self.payload,
            "observed_at": self.observed_at.astimezone(UTC).isoformat(),
            "idempotency_key": self.idempotency_key,
        }
        return json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    """A persisted evidence event and its storage metadata."""

    evidence_id: str
    event: EvidenceEvent
    content_hash: str
    created_at: datetime

    def __post_init__(self) -> None:
        if type(self.event) is not EvidenceEvent:
            raise TypeError("event must be an EvidenceEvent")
        evidence_id = _validate_string(self.evidence_id, "evidence_id")
        content_hash = _validate_string(self.content_hash, "content_hash")
        created_at = _validate_aware_datetime(self.created_at, "created_at")
        object.__setattr__(self, "evidence_id", evidence_id)
        object.__setattr__(self, "content_hash", content_hash)
        object.__setattr__(self, "created_at", created_at)
