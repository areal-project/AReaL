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
) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not allow_blank and not value.strip():
        raise ValueError(f"{field_name} must not be blank")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{field_name} must be valid UTF-8") from exc


def _validate_aware_datetime(value: object, field_name: str) -> None:
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
        value.astimezone(UTC)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{field_name} must be normalizable to UTC") from exc


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
        _validate_string(self.tenant_id, "tenant_id")
        _validate_string(self.namespace, "namespace")
        _validate_string(self.subject_id, "subject_id")


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
        if not isinstance(self.scope, MemoryScope):
            raise TypeError("scope must be a MemoryScope")
        _validate_string(self.session_id, "session_id")
        _validate_string(self.run_id, "run_id")
        _validate_string(self.idempotency_key, "idempotency_key")
        if isinstance(self.sequence_no, bool) or not isinstance(self.sequence_no, int):
            raise TypeError("sequence_no must be an integer")
        if self.sequence_no < 0:
            raise ValueError("sequence_no must be non-negative")
        if self.sequence_no > _MAX_SEQUENCE_NO:
            raise ValueError("sequence_no must fit in a signed 64-bit integer")
        if not isinstance(self.kind, EvidenceKind):
            raise TypeError("kind must be an EvidenceKind")
        _validate_string(self.payload, "payload", allow_blank=True)
        _validate_aware_datetime(self.observed_at, "observed_at")

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
        if not isinstance(self.event, EvidenceEvent):
            raise TypeError("event must be an EvidenceEvent")
        _validate_string(self.evidence_id, "evidence_id")
        _validate_string(self.content_hash, "content_hash")
        _validate_aware_datetime(self.created_at, "created_at")
