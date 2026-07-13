# SPDX-License-Identifier: Apache-2.0

"""Immutable runtime query, source-read, delivery, and exposure values.

The records in this module distinguish memory selected by a retriever from
memory acknowledged at a consumer or model-call boundary.  Hashes are
integrity commitments, not signatures or proof against malicious in-process
code.  A trusted runtime component must own consumer acknowledgement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256

from areal.v2.memory_service.types import MemoryScope

_SCHEMA_VERSION = 1
_MAX_INTEGER = 2**63 - 1
_SHA256_LENGTH = 64
_NONCE_HEX_LENGTH = 64


def _canonical_json_bytes(value: dict[str, object]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def _string(value: object, field_name: str, *, allow_blank: bool = False) -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be a str")
    if not allow_blank and not value.strip():
        raise ValueError(f"{field_name} must not be blank")
    try:
        value.encode("utf-8", "strict")
    except UnicodeEncodeError as error:
        raise ValueError(f"{field_name} must be valid UTF-8") from error
    return value


def _sha256(value: object, field_name: str) -> str:
    value = _string(value, field_name)
    if len(value) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")
    return value


def _nonce(value: object, field_name: str) -> str:
    value = _string(value, field_name)
    if len(value) != _NONCE_HEX_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{field_name} must be 256 bits of lowercase hex")
    return value


def _integer(
    value: object,
    field_name: str,
    *,
    minimum: int = 0,
) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an int")
    if not minimum <= value <= _MAX_INTEGER:
        raise ValueError(f"{field_name} must be between {minimum} and {_MAX_INTEGER}")
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


def _scope_value(value: object) -> dict[str, str]:
    if type(value) is not MemoryScope:
        raise TypeError("scope must be a MemoryScope")
    return {
        "namespace": value.namespace,
        "subject_id": value.subject_id,
        "tenant_id": value.tenant_id,
    }


def _exact_tuple(value: object, field_name: str) -> tuple[object, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{field_name} must be a tuple")
    return tuple(tuple.__iter__(value))


def _record_id(
    value: object,
    field_name: str,
    *,
    prefix: str,
    content_hash: str,
) -> str:
    value = _string(value, field_name)
    expected = f"{prefix}{content_hash[:24]}"
    if value != expected:
        raise ValueError(f"{field_name} disagrees with canonical record bytes")
    return value


def _is_unique_subsequence(
    subsequence: tuple[object, ...],
    sequence: tuple[object, ...],
) -> bool:
    if len(set(subsequence)) != len(subsequence):
        return False
    cursor = iter(sequence)
    return all(any(candidate == item for candidate in cursor) for item in subsequence)


def _is_unique_subset(
    subset: tuple[object, ...],
    superset: tuple[object, ...],
) -> bool:
    """Return whether every distinct subset member occurs in the superset.

    Tuple order is deliberately ignored here.  A retriever owns ranking order,
    while the release tuple preserves the independent eligibility order.
    """

    superset_members = set(superset)
    return len(set(subset)) == len(subset) and all(
        item in superset_members for item in subset
    )


@dataclass(frozen=True, slots=True)
class MemoryQuerySpecV1:
    """One explicit release-pinned runtime query assignment."""

    scope: MemoryScope
    release_id: str
    trajectory_id: str
    rollout_group_id: str
    query_sequence_no: int
    query_sha256: str
    task_policy_id: str
    task_policy_version_sha256: str
    retrieval_policy_id: str
    retrieval_policy_version_sha256: str
    max_returned_items: int
    max_context_utf8_bytes: int
    idempotency_key: str

    def __post_init__(self) -> None:
        _scope_value(self.scope)
        for field_name in (
            "release_id",
            "trajectory_id",
            "rollout_group_id",
            "task_policy_id",
            "retrieval_policy_id",
            "idempotency_key",
        ):
            object.__setattr__(
                self,
                field_name,
                _string(getattr(self, field_name), field_name),
            )
        _integer(self.query_sequence_no, "query_sequence_no")
        _integer(self.max_returned_items, "max_returned_items")
        _integer(self.max_context_utf8_bytes, "max_context_utf8_bytes")
        for field_name in (
            "query_sha256",
            "task_policy_version_sha256",
            "retrieval_policy_version_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _sha256(getattr(self, field_name), field_name),
            )

    def _canonical_value(self) -> dict[str, object]:
        return {
            "idempotency_key": _string(self.idempotency_key, "idempotency_key"),
            "max_context_utf8_bytes": _integer(
                self.max_context_utf8_bytes,
                "max_context_utf8_bytes",
            ),
            "max_returned_items": _integer(
                self.max_returned_items,
                "max_returned_items",
            ),
            "query_sequence_no": _integer(
                self.query_sequence_no,
                "query_sequence_no",
            ),
            "query_sha256": _sha256(self.query_sha256, "query_sha256"),
            "release_id": _string(self.release_id, "release_id"),
            "retrieval_policy_id": _string(
                self.retrieval_policy_id,
                "retrieval_policy_id",
            ),
            "retrieval_policy_version_sha256": _sha256(
                self.retrieval_policy_version_sha256,
                "retrieval_policy_version_sha256",
            ),
            "rollout_group_id": _string(
                self.rollout_group_id,
                "rollout_group_id",
            ),
            "scope": _scope_value(self.scope),
            "task_policy_id": _string(self.task_policy_id, "task_policy_id"),
            "task_policy_version_sha256": _sha256(
                self.task_policy_version_sha256,
                "task_policy_version_sha256",
            ),
            "trajectory_id": _string(self.trajectory_id, "trajectory_id"),
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(
            {
                "record_kind": "memory_query_spec",
                "schema_version": _SCHEMA_VERSION,
                **self._canonical_value(),
            }
        )


@dataclass(frozen=True, slots=True)
class MemoryRevisionRefV1:
    """A public revision ID paired with its full integrity commitment."""

    revision_id: str
    revision_content_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "revision_id",
            _string(self.revision_id, "revision_id"),
        )
        object.__setattr__(
            self,
            "revision_content_sha256",
            _sha256(self.revision_content_sha256, "revision_content_sha256"),
        )

    def _canonical_value(self) -> dict[str, object]:
        return {
            "revision_content_sha256": _sha256(
                self.revision_content_sha256,
                "revision_content_sha256",
            ),
            "revision_id": _string(self.revision_id, "revision_id"),
        }


def _revision_refs(
    value: object,
    field_name: str,
    *,
    unique: bool = True,
) -> tuple[MemoryRevisionRefV1, ...]:
    values = _exact_tuple(value, field_name)
    if any(type(item) is not MemoryRevisionRefV1 for item in values):
        raise TypeError(f"{field_name} must contain MemoryRevisionRefV1 values")
    result = tuple(values)
    if unique and len({item.revision_id for item in result}) != len(result):
        raise ValueError(f"{field_name} must not contain duplicate revision IDs")
    return result


@dataclass(frozen=True, slots=True)
class MemoryEvidenceRefV1:
    """An evidence ID paired with its full integrity commitment."""

    evidence_id: str
    evidence_content_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evidence_id",
            _string(self.evidence_id, "evidence_id"),
        )
        object.__setattr__(
            self,
            "evidence_content_sha256",
            _sha256(self.evidence_content_sha256, "evidence_content_sha256"),
        )

    def _canonical_value(self) -> dict[str, object]:
        return {
            "evidence_content_sha256": _sha256(
                self.evidence_content_sha256,
                "evidence_content_sha256",
            ),
            "evidence_id": _string(self.evidence_id, "evidence_id"),
        }


class MemorySourceObjectKind(StrEnum):
    """Kinds of immutable source objects returned to the runtime."""

    RELEASE = "release"
    REVISION = "revision"
    CANDIDATE = "candidate"
    EVIDENCE = "evidence"


class MemorySourceReadOperation(StrEnum):
    """Source-store calls whose returned objects can affect query resolution."""

    GET_RELEASE = "get_release"
    GET_RELEASE_REVISIONS = "get_release_revisions"
    GET_REVISION = "get_revision"
    GET_CANDIDATE = "get_candidate"
    GET_CANDIDATE_EVIDENCE = "get_candidate_evidence"


_SOURCE_PREFIX_BY_KIND = {
    MemorySourceObjectKind.RELEASE: "rel_",
    MemorySourceObjectKind.REVISION: "rev_",
    MemorySourceObjectKind.CANDIDATE: "cand_",
    MemorySourceObjectKind.EVIDENCE: "evd_",
}

_SOURCE_KIND_BY_OPERATION = {
    MemorySourceReadOperation.GET_RELEASE: MemorySourceObjectKind.RELEASE,
    MemorySourceReadOperation.GET_RELEASE_REVISIONS: MemorySourceObjectKind.REVISION,
    MemorySourceReadOperation.GET_REVISION: MemorySourceObjectKind.REVISION,
    MemorySourceReadOperation.GET_CANDIDATE: MemorySourceObjectKind.CANDIDATE,
    MemorySourceReadOperation.GET_CANDIDATE_EVIDENCE: MemorySourceObjectKind.EVIDENCE,
}


@dataclass(frozen=True, slots=True)
class MemorySourceObjectRefV1:
    """A source object address paired with its full content commitment."""

    kind: MemorySourceObjectKind
    object_id: str
    object_content_sha256: str

    def __post_init__(self) -> None:
        if type(self.kind) is not MemorySourceObjectKind:
            raise TypeError("kind must be a MemorySourceObjectKind")
        object_id = _string(self.object_id, "object_id")
        content_hash = _sha256(
            self.object_content_sha256,
            "object_content_sha256",
        )
        if object_id != f"{_SOURCE_PREFIX_BY_KIND[self.kind]}{content_hash[:24]}":
            raise ValueError("object_id disagrees with kind and full content hash")
        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(self, "object_content_sha256", content_hash)

    def _canonical_value(self) -> dict[str, object]:
        if type(self.kind) is not MemorySourceObjectKind:
            raise TypeError("kind must be a MemorySourceObjectKind")
        object_id = _string(self.object_id, "object_id")
        content_hash = _sha256(
            self.object_content_sha256,
            "object_content_sha256",
        )
        if object_id != f"{_SOURCE_PREFIX_BY_KIND[self.kind]}{content_hash[:24]}":
            raise ValueError("object_id disagrees with kind and full content hash")
        return {
            "kind": self.kind.value,
            "object_content_sha256": content_hash,
            "object_id": object_id,
        }


def _source_object_refs(
    value: object,
    field_name: str,
) -> tuple[MemorySourceObjectRefV1, ...]:
    values = _exact_tuple(value, field_name)
    if any(type(item) is not MemorySourceObjectRefV1 for item in values):
        raise TypeError(f"{field_name} must contain MemorySourceObjectRefV1 values")
    result = tuple(values)
    if len({(item.kind, item.object_id) for item in result}) != len(result):
        raise ValueError(f"{field_name} must not contain duplicate source objects")
    return result


@dataclass(frozen=True, slots=True)
class MemorySourceReadEventV1:
    """One ordered source-store call and the immutable objects it returned."""

    sequence_no: int
    operation: MemorySourceReadOperation
    returned_objects: tuple[MemorySourceObjectRefV1, ...]

    def __post_init__(self) -> None:
        _integer(self.sequence_no, "sequence_no")
        if type(self.operation) is not MemorySourceReadOperation:
            raise TypeError("operation must be a MemorySourceReadOperation")
        returned = _source_object_refs(self.returned_objects, "returned_objects")
        expected_kind = _SOURCE_KIND_BY_OPERATION[self.operation]
        if any(item.kind is not expected_kind for item in returned):
            raise ValueError("returned object kind disagrees with read operation")
        if (
            self.operation
            in {
                MemorySourceReadOperation.GET_RELEASE,
                MemorySourceReadOperation.GET_REVISION,
                MemorySourceReadOperation.GET_CANDIDATE,
            }
            and len(returned) != 1
        ):
            raise ValueError("single-object read operations must return one object")

    def _canonical_value(self) -> dict[str, object]:
        returned = _source_object_refs(self.returned_objects, "returned_objects")
        if type(self.operation) is not MemorySourceReadOperation:
            raise TypeError("operation must be a MemorySourceReadOperation")
        expected_kind = _SOURCE_KIND_BY_OPERATION[self.operation]
        if any(item.kind is not expected_kind for item in returned):
            raise ValueError("returned object kind disagrees with read operation")
        if (
            self.operation
            in {
                MemorySourceReadOperation.GET_RELEASE,
                MemorySourceReadOperation.GET_REVISION,
                MemorySourceReadOperation.GET_CANDIDATE,
            }
            and len(returned) != 1
        ):
            raise ValueError("single-object read operations must return one object")
        return {
            "operation": self.operation.value,
            "returned_objects": [item._canonical_value() for item in returned],
            "sequence_no": _integer(self.sequence_no, "sequence_no"),
        }


def _source_read_events(value: object) -> tuple[MemorySourceReadEventV1, ...]:
    values = _exact_tuple(value, "read_events")
    if not values:
        raise ValueError("read_events must not be empty")
    if any(type(item) is not MemorySourceReadEventV1 for item in values):
        raise TypeError("read_events must contain MemorySourceReadEventV1 values")
    result = tuple(values)
    if tuple(item.sequence_no for item in result) != tuple(range(len(result))):
        raise ValueError("read_events must have contiguous canonical sequence numbers")
    return result


def _source_read_receipt_canonical_bytes(
    *,
    scope: object,
    attempt_id: object,
    attempt_content_sha256: object,
    read_events: object,
) -> bytes:
    events = _source_read_events(read_events)
    return _canonical_json_bytes(
        {
            "attempt_content_sha256": _sha256(
                attempt_content_sha256,
                "attempt_content_sha256",
            ),
            "attempt_id": _string(attempt_id, "attempt_id"),
            "read_events": [item._canonical_value() for item in events],
            "record_kind": "memory_source_read_receipt",
            "schema_version": _SCHEMA_VERSION,
            "scope": _scope_value(scope),
        }
    )


@dataclass(frozen=True, slots=True)
class MemorySourceReadReceiptV1:
    """Runtime-owned proof of source objects read and validated for one attempt.

    This receipt proves which immutable source objects the honest runtime read
    while resolving a query.  It does not prove that a model used those values
    or that they improved task utility.
    """

    scope: MemoryScope
    attempt_id: str
    attempt_content_sha256: str
    read_events: tuple[MemorySourceReadEventV1, ...]
    source_read_receipt_id: str
    content_hash: str
    created_at: datetime

    def __post_init__(self) -> None:
        canonical = _source_read_receipt_canonical_bytes(
            scope=self.scope,
            attempt_id=self.attempt_id,
            attempt_content_sha256=self.attempt_content_sha256,
            read_events=self.read_events,
        )
        expected_hash = sha256(canonical).hexdigest()
        object.__setattr__(
            self,
            "source_read_receipt_id",
            _record_id(
                self.source_read_receipt_id,
                "source_read_receipt_id",
                prefix="msrr_",
                content_hash=expected_hash,
            ),
        )
        if _sha256(self.content_hash, "content_hash") != expected_hash:
            raise ValueError("content_hash disagrees with canonical source-read bytes")
        object.__setattr__(
            self,
            "created_at",
            _aware_datetime(self.created_at, "created_at"),
        )

    @classmethod
    def create(
        cls,
        *,
        attempt: MemoryQueryAttemptV1,
        read_events: tuple[MemorySourceReadEventV1, ...],
        created_at: datetime | None = None,
    ) -> MemorySourceReadReceiptV1:
        if type(attempt) is not MemoryQueryAttemptV1:
            raise TypeError("attempt must be a MemoryQueryAttemptV1")
        values = {
            "scope": attempt.spec.scope,
            "attempt_id": attempt.attempt_id,
            "attempt_content_sha256": attempt.content_hash,
            "read_events": read_events,
        }
        canonical = _source_read_receipt_canonical_bytes(**values)
        content_hash = sha256(canonical).hexdigest()
        return cls(
            **values,
            source_read_receipt_id=f"msrr_{content_hash[:24]}",
            content_hash=content_hash,
            created_at=datetime.now(UTC) if created_at is None else created_at,
        )

    def canonical_bytes(self) -> bytes:
        canonical = _source_read_receipt_canonical_bytes(
            scope=self.scope,
            attempt_id=self.attempt_id,
            attempt_content_sha256=self.attempt_content_sha256,
            read_events=self.read_events,
        )
        expected_hash = sha256(canonical).hexdigest()
        if _sha256(self.content_hash, "content_hash") != expected_hash:
            raise ValueError("content_hash disagrees with canonical source-read bytes")
        _record_id(
            self.source_read_receipt_id,
            "source_read_receipt_id",
            prefix="msrr_",
            content_hash=expected_hash,
        )
        return canonical


def _evidence_refs(value: object) -> tuple[MemoryEvidenceRefV1, ...]:
    values = _exact_tuple(value, "evidence")
    if not values:
        raise ValueError("evidence must not be empty")
    if any(type(item) is not MemoryEvidenceRefV1 for item in values):
        raise TypeError("evidence must contain MemoryEvidenceRefV1 values")
    result = tuple(values)
    if len({item.evidence_id for item in result}) != len(result):
        raise ValueError("evidence must not contain duplicate evidence IDs")
    return result


@dataclass(frozen=True, slots=True)
class MemoryQueryItemV1:
    """One returned memory with exact revision, candidate, and evidence links."""

    release_position: int
    revision: MemoryRevisionRefV1
    memory_id: str
    generation: int
    candidate_id: str
    candidate_content_sha256: str
    evidence: tuple[MemoryEvidenceRefV1, ...]
    content: str

    def __post_init__(self) -> None:
        _integer(self.release_position, "release_position")
        if type(self.revision) is not MemoryRevisionRefV1:
            raise TypeError("revision must be a MemoryRevisionRefV1")
        object.__setattr__(self, "memory_id", _string(self.memory_id, "memory_id"))
        _integer(self.generation, "generation")
        object.__setattr__(
            self,
            "candidate_id",
            _string(self.candidate_id, "candidate_id"),
        )
        object.__setattr__(
            self,
            "candidate_content_sha256",
            _sha256(self.candidate_content_sha256, "candidate_content_sha256"),
        )
        _evidence_refs(self.evidence)
        object.__setattr__(self, "content", _string(self.content, "content"))

    def _canonical_value(self) -> dict[str, object]:
        return {
            "candidate_content_sha256": _sha256(
                self.candidate_content_sha256,
                "candidate_content_sha256",
            ),
            "candidate_id": _string(self.candidate_id, "candidate_id"),
            "content": _string(self.content, "content"),
            "evidence": [
                item._canonical_value() for item in _evidence_refs(self.evidence)
            ],
            "generation": _integer(self.generation, "generation"),
            "memory_id": _string(self.memory_id, "memory_id"),
            "release_position": _integer(
                self.release_position,
                "release_position",
            ),
            "revision": self.revision._canonical_value(),
        }


def _query_items(value: object) -> tuple[MemoryQueryItemV1, ...]:
    values = _exact_tuple(value, "returned_items")
    if any(type(item) is not MemoryQueryItemV1 for item in values):
        raise TypeError("returned_items must contain MemoryQueryItemV1 values")
    result = tuple(values)
    refs = tuple(item.revision for item in result)
    if len({item.revision_id for item in refs}) != len(refs):
        raise ValueError("returned_items must not contain duplicate revision IDs")
    if len({item.candidate_id for item in result}) != len(result):
        raise ValueError("returned_items must not contain duplicate candidate IDs")
    if len({item.memory_id for item in result}) != len(result):
        raise ValueError("returned_items must not contain duplicate memory IDs")
    return result


def _attempt_canonical_bytes(
    *,
    spec: object,
    release_content_sha256: object,
    release_revisions: object,
    attempt_nonce: object,
) -> bytes:
    if type(spec) is not MemoryQuerySpecV1:
        raise TypeError("spec must be a MemoryQuerySpecV1")
    revisions = _revision_refs(release_revisions, "release_revisions")
    return _canonical_json_bytes(
        {
            "attempt_nonce": _nonce(attempt_nonce, "attempt_nonce"),
            "record_kind": "memory_query_attempt",
            "release_content_sha256": _sha256(
                release_content_sha256,
                "release_content_sha256",
            ),
            "release_revisions": [item._canonical_value() for item in revisions],
            "schema_version": _SCHEMA_VERSION,
            "spec": spec._canonical_value(),
        }
    )


@dataclass(frozen=True, slots=True)
class MemoryQueryAttemptV1:
    """A store-validated, release-pinned query attempt with anti-replay nonce."""

    spec: MemoryQuerySpecV1
    release_content_sha256: str
    release_revisions: tuple[MemoryRevisionRefV1, ...]
    attempt_nonce: str
    attempt_id: str
    content_hash: str
    created_at: datetime

    def __post_init__(self) -> None:
        canonical = _attempt_canonical_bytes(
            spec=self.spec,
            release_content_sha256=self.release_content_sha256,
            release_revisions=self.release_revisions,
            attempt_nonce=self.attempt_nonce,
        )
        expected_hash = sha256(canonical).hexdigest()
        object.__setattr__(
            self,
            "attempt_id",
            _record_id(
                self.attempt_id,
                "attempt_id",
                prefix="mqat_",
                content_hash=expected_hash,
            ),
        )
        if _sha256(self.content_hash, "content_hash") != expected_hash:
            raise ValueError("content_hash disagrees with canonical attempt bytes")
        object.__setattr__(
            self,
            "created_at",
            _aware_datetime(self.created_at, "created_at"),
        )

    @classmethod
    def create(
        cls,
        *,
        spec: MemoryQuerySpecV1,
        release_content_sha256: str,
        release_revisions: tuple[MemoryRevisionRefV1, ...],
        attempt_nonce: str,
        created_at: datetime | None = None,
    ) -> MemoryQueryAttemptV1:
        canonical = _attempt_canonical_bytes(
            spec=spec,
            release_content_sha256=release_content_sha256,
            release_revisions=release_revisions,
            attempt_nonce=attempt_nonce,
        )
        content_hash = sha256(canonical).hexdigest()
        return cls(
            spec=spec,
            release_content_sha256=release_content_sha256,
            release_revisions=release_revisions,
            attempt_nonce=attempt_nonce,
            attempt_id=f"mqat_{content_hash[:24]}",
            content_hash=content_hash,
            created_at=datetime.now(UTC) if created_at is None else created_at,
        )

    def canonical_bytes(self) -> bytes:
        canonical = _attempt_canonical_bytes(
            spec=self.spec,
            release_content_sha256=self.release_content_sha256,
            release_revisions=self.release_revisions,
            attempt_nonce=self.attempt_nonce,
        )
        expected_hash = sha256(canonical).hexdigest()
        if _sha256(self.content_hash, "content_hash") != expected_hash:
            raise ValueError("content_hash disagrees with canonical attempt bytes")
        _record_id(
            self.attempt_id,
            "attempt_id",
            prefix="mqat_",
            content_hash=expected_hash,
        )
        return canonical


def _query_result_canonical_bytes(
    *,
    scope: object,
    release_id: object,
    release_content_sha256: object,
    trajectory_id: object,
    rollout_group_id: object,
    attempt_id: object,
    attempt_content_sha256: object,
    source_read_receipt_id: object,
    source_read_receipt_content_sha256: object,
    eligible_revisions: object,
    retrieved_revisions: object,
    returned_items: object,
) -> bytes:
    eligible = _revision_refs(eligible_revisions, "eligible_revisions")
    retrieved = _revision_refs(retrieved_revisions, "retrieved_revisions")
    returned = _query_items(returned_items)
    returned_refs = tuple(item.revision for item in returned)
    if not _is_unique_subset(retrieved, eligible):
        raise ValueError("retrieved_revisions must be a unique eligible subset")
    if not _is_unique_subsequence(returned_refs, retrieved):
        raise ValueError("returned_items must be an ordered retrieved subsequence")
    if any(
        item.release_position >= len(eligible)
        or eligible[item.release_position] != item.revision
        for item in returned
    ):
        raise ValueError("returned item positions must match the pinned release")
    source_receipt_hash = _sha256(
        source_read_receipt_content_sha256,
        "source_read_receipt_content_sha256",
    )
    source_receipt_id = _record_id(
        source_read_receipt_id,
        "source_read_receipt_id",
        prefix="msrr_",
        content_hash=source_receipt_hash,
    )
    return _canonical_json_bytes(
        {
            "attempt_content_sha256": _sha256(
                attempt_content_sha256,
                "attempt_content_sha256",
            ),
            "attempt_id": _string(attempt_id, "attempt_id"),
            "eligible_revisions": [item._canonical_value() for item in eligible],
            "record_kind": "memory_query_result",
            "release_content_sha256": _sha256(
                release_content_sha256,
                "release_content_sha256",
            ),
            "release_id": _string(release_id, "release_id"),
            "retrieved_revisions": [item._canonical_value() for item in retrieved],
            "returned_items": [item._canonical_value() for item in returned],
            "rollout_group_id": _string(rollout_group_id, "rollout_group_id"),
            "schema_version": _SCHEMA_VERSION,
            "scope": _scope_value(scope),
            "source_read_receipt_content_sha256": source_receipt_hash,
            "source_read_receipt_id": source_receipt_id,
            "trajectory_id": _string(trajectory_id, "trajectory_id"),
        }
    )


@dataclass(frozen=True, slots=True)
class MemoryQueryResultV1:
    """Ordered retrieval provenance and store-authentic returned content."""

    scope: MemoryScope
    release_id: str
    release_content_sha256: str
    trajectory_id: str
    rollout_group_id: str
    attempt_id: str
    attempt_content_sha256: str
    source_read_receipt_id: str
    source_read_receipt_content_sha256: str
    eligible_revisions: tuple[MemoryRevisionRefV1, ...]
    retrieved_revisions: tuple[MemoryRevisionRefV1, ...]
    returned_items: tuple[MemoryQueryItemV1, ...]
    query_result_id: str
    content_hash: str
    created_at: datetime

    def __post_init__(self) -> None:
        canonical = _query_result_canonical_bytes(
            scope=self.scope,
            release_id=self.release_id,
            release_content_sha256=self.release_content_sha256,
            trajectory_id=self.trajectory_id,
            rollout_group_id=self.rollout_group_id,
            attempt_id=self.attempt_id,
            attempt_content_sha256=self.attempt_content_sha256,
            source_read_receipt_id=self.source_read_receipt_id,
            source_read_receipt_content_sha256=(
                self.source_read_receipt_content_sha256
            ),
            eligible_revisions=self.eligible_revisions,
            retrieved_revisions=self.retrieved_revisions,
            returned_items=self.returned_items,
        )
        expected_hash = sha256(canonical).hexdigest()
        object.__setattr__(
            self,
            "query_result_id",
            _record_id(
                self.query_result_id,
                "query_result_id",
                prefix="mqres_",
                content_hash=expected_hash,
            ),
        )
        if _sha256(self.content_hash, "content_hash") != expected_hash:
            raise ValueError("content_hash disagrees with canonical query result bytes")
        object.__setattr__(
            self,
            "created_at",
            _aware_datetime(self.created_at, "created_at"),
        )

    @classmethod
    def create(
        cls,
        *,
        attempt: MemoryQueryAttemptV1,
        source_read_receipt: MemorySourceReadReceiptV1,
        retrieved_revisions: tuple[MemoryRevisionRefV1, ...],
        returned_items: tuple[MemoryQueryItemV1, ...],
        created_at: datetime | None = None,
    ) -> MemoryQueryResultV1:
        if type(attempt) is not MemoryQueryAttemptV1:
            raise TypeError("attempt must be a MemoryQueryAttemptV1")
        if type(source_read_receipt) is not MemorySourceReadReceiptV1:
            raise TypeError("source_read_receipt must be a MemorySourceReadReceiptV1")
        source_read_receipt.canonical_bytes()
        if (
            source_read_receipt.scope != attempt.spec.scope
            or source_read_receipt.attempt_id != attempt.attempt_id
            or source_read_receipt.attempt_content_sha256 != attempt.content_hash
        ):
            raise ValueError("source-read receipt is not bound to the exact attempt")
        values = {
            "scope": attempt.spec.scope,
            "release_id": attempt.spec.release_id,
            "release_content_sha256": attempt.release_content_sha256,
            "trajectory_id": attempt.spec.trajectory_id,
            "rollout_group_id": attempt.spec.rollout_group_id,
            "attempt_id": attempt.attempt_id,
            "attempt_content_sha256": attempt.content_hash,
            "source_read_receipt_id": (source_read_receipt.source_read_receipt_id),
            "source_read_receipt_content_sha256": source_read_receipt.content_hash,
            "eligible_revisions": attempt.release_revisions,
            "retrieved_revisions": retrieved_revisions,
            "returned_items": returned_items,
        }
        canonical = _query_result_canonical_bytes(**values)
        content_hash = sha256(canonical).hexdigest()
        return cls(
            **values,
            query_result_id=f"mqres_{content_hash[:24]}",
            content_hash=content_hash,
            created_at=datetime.now(UTC) if created_at is None else created_at,
        )

    @property
    def returned_revisions(self) -> tuple[MemoryRevisionRefV1, ...]:
        return tuple(item.revision for item in self.returned_items)

    def canonical_bytes(self) -> bytes:
        canonical = _query_result_canonical_bytes(
            scope=self.scope,
            release_id=self.release_id,
            release_content_sha256=self.release_content_sha256,
            trajectory_id=self.trajectory_id,
            rollout_group_id=self.rollout_group_id,
            attempt_id=self.attempt_id,
            attempt_content_sha256=self.attempt_content_sha256,
            source_read_receipt_id=self.source_read_receipt_id,
            source_read_receipt_content_sha256=(
                self.source_read_receipt_content_sha256
            ),
            eligible_revisions=self.eligible_revisions,
            retrieved_revisions=self.retrieved_revisions,
            returned_items=self.returned_items,
        )
        expected_hash = sha256(canonical).hexdigest()
        if _sha256(self.content_hash, "content_hash") != expected_hash:
            raise ValueError("content_hash disagrees with canonical query result bytes")
        _record_id(
            self.query_result_id,
            "query_result_id",
            prefix="mqres_",
            content_hash=expected_hash,
        )
        return canonical


@dataclass(frozen=True, slots=True)
class MemoryRenderedRevisionRangeV1:
    """Renderer-reported range request; the runtime computes its hash itself."""

    revision_id: str
    rendered_start: int
    rendered_end: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "revision_id",
            _string(self.revision_id, "revision_id"),
        )
        start = _integer(self.rendered_start, "rendered_start")
        end = _integer(self.rendered_end, "rendered_end")
        if end <= start:
            raise ValueError("rendered range must be non-empty")


@dataclass(frozen=True, slots=True)
class MemoryRenderedRevisionSpanV1:
    """One returned revision's exact half-open range in rendered context."""

    revision: MemoryRevisionRefV1
    rendered_start: int
    rendered_end: int
    rendered_fragment_sha256: str

    def __post_init__(self) -> None:
        if type(self.revision) is not MemoryRevisionRefV1:
            raise TypeError("revision must be a MemoryRevisionRefV1")
        start = _integer(self.rendered_start, "rendered_start")
        end = _integer(self.rendered_end, "rendered_end")
        if end <= start:
            raise ValueError("rendered span must be non-empty")
        object.__setattr__(
            self,
            "rendered_fragment_sha256",
            _sha256(self.rendered_fragment_sha256, "rendered_fragment_sha256"),
        )

    def _canonical_value(self) -> dict[str, object]:
        return {
            "rendered_end": _integer(self.rendered_end, "rendered_end"),
            "rendered_fragment_sha256": _sha256(
                self.rendered_fragment_sha256,
                "rendered_fragment_sha256",
            ),
            "rendered_start": _integer(self.rendered_start, "rendered_start"),
            "revision": self.revision._canonical_value(),
        }


def _rendered_spans(
    value: object,
    *,
    context_utf8_bytes: int,
) -> tuple[MemoryRenderedRevisionSpanV1, ...]:
    values = _exact_tuple(value, "rendered_spans")
    if any(type(item) is not MemoryRenderedRevisionSpanV1 for item in values):
        raise TypeError(
            "rendered_spans must contain MemoryRenderedRevisionSpanV1 values"
        )
    result = tuple(values)
    refs = tuple(item.revision for item in result)
    if len({item.revision_id for item in refs}) != len(refs):
        raise ValueError("rendered_spans must not contain duplicate revision IDs")
    last_end = 0
    for item in result:
        if item.rendered_start < last_end or item.rendered_end > context_utf8_bytes:
            raise ValueError("rendered_spans must be ordered, disjoint, and in range")
        last_end = item.rendered_end
    return result


def _delivery_canonical_bytes(
    *,
    scope: object,
    release_id: object,
    release_content_sha256: object,
    trajectory_id: object,
    query_result_id: object,
    query_result_content_sha256: object,
    renderer_id: object,
    renderer_version_sha256: object,
    rendered_context_sha256: object,
    rendered_context_utf8_bytes: object,
    rendered_spans: object,
    delivery_nonce: object,
) -> bytes:
    byte_count = _integer(
        rendered_context_utf8_bytes,
        "rendered_context_utf8_bytes",
    )
    spans = _rendered_spans(rendered_spans, context_utf8_bytes=byte_count)
    return _canonical_json_bytes(
        {
            "delivery_nonce": _nonce(delivery_nonce, "delivery_nonce"),
            "query_result_content_sha256": _sha256(
                query_result_content_sha256,
                "query_result_content_sha256",
            ),
            "query_result_id": _string(query_result_id, "query_result_id"),
            "record_kind": "memory_delivery",
            "release_content_sha256": _sha256(
                release_content_sha256,
                "release_content_sha256",
            ),
            "release_id": _string(release_id, "release_id"),
            "rendered_context_sha256": _sha256(
                rendered_context_sha256,
                "rendered_context_sha256",
            ),
            "rendered_context_utf8_bytes": byte_count,
            "rendered_spans": [item._canonical_value() for item in spans],
            "renderer_id": _string(renderer_id, "renderer_id"),
            "renderer_version_sha256": _sha256(
                renderer_version_sha256,
                "renderer_version_sha256",
            ),
            "schema_version": _SCHEMA_VERSION,
            "scope": _scope_value(scope),
            "trajectory_id": _string(trajectory_id, "trajectory_id"),
        }
    )


@dataclass(frozen=True, slots=True)
class MemoryDeliveryV1:
    """A rendered query result awaiting one consumer-boundary acknowledgement."""

    scope: MemoryScope
    release_id: str
    release_content_sha256: str
    trajectory_id: str
    query_result_id: str
    query_result_content_sha256: str
    renderer_id: str
    renderer_version_sha256: str
    rendered_context_sha256: str
    rendered_context_utf8_bytes: int
    rendered_spans: tuple[MemoryRenderedRevisionSpanV1, ...]
    delivery_nonce: str
    delivery_id: str
    content_hash: str
    created_at: datetime

    def __post_init__(self) -> None:
        canonical = _delivery_canonical_bytes(
            scope=self.scope,
            release_id=self.release_id,
            release_content_sha256=self.release_content_sha256,
            trajectory_id=self.trajectory_id,
            query_result_id=self.query_result_id,
            query_result_content_sha256=self.query_result_content_sha256,
            renderer_id=self.renderer_id,
            renderer_version_sha256=self.renderer_version_sha256,
            rendered_context_sha256=self.rendered_context_sha256,
            rendered_context_utf8_bytes=self.rendered_context_utf8_bytes,
            rendered_spans=self.rendered_spans,
            delivery_nonce=self.delivery_nonce,
        )
        expected_hash = sha256(canonical).hexdigest()
        object.__setattr__(
            self,
            "delivery_id",
            _record_id(
                self.delivery_id,
                "delivery_id",
                prefix="mdel_",
                content_hash=expected_hash,
            ),
        )
        if _sha256(self.content_hash, "content_hash") != expected_hash:
            raise ValueError("content_hash disagrees with canonical delivery bytes")
        object.__setattr__(
            self,
            "created_at",
            _aware_datetime(self.created_at, "created_at"),
        )

    @classmethod
    def create(
        cls,
        *,
        query_result: MemoryQueryResultV1,
        renderer_id: str,
        renderer_version_sha256: str,
        rendered_context_sha256: str,
        rendered_context_utf8_bytes: int,
        rendered_spans: tuple[MemoryRenderedRevisionSpanV1, ...],
        delivery_nonce: str,
        created_at: datetime | None = None,
    ) -> MemoryDeliveryV1:
        if type(query_result) is not MemoryQueryResultV1:
            raise TypeError("query_result must be a MemoryQueryResultV1")
        values = {
            "scope": query_result.scope,
            "release_id": query_result.release_id,
            "release_content_sha256": query_result.release_content_sha256,
            "trajectory_id": query_result.trajectory_id,
            "query_result_id": query_result.query_result_id,
            "query_result_content_sha256": query_result.content_hash,
            "renderer_id": renderer_id,
            "renderer_version_sha256": renderer_version_sha256,
            "rendered_context_sha256": rendered_context_sha256,
            "rendered_context_utf8_bytes": rendered_context_utf8_bytes,
            "rendered_spans": rendered_spans,
            "delivery_nonce": delivery_nonce,
        }
        canonical = _delivery_canonical_bytes(**values)
        content_hash = sha256(canonical).hexdigest()
        return cls(
            **values,
            delivery_id=f"mdel_{content_hash[:24]}",
            content_hash=content_hash,
            created_at=datetime.now(UTC) if created_at is None else created_at,
        )

    def canonical_bytes(self) -> bytes:
        canonical = _delivery_canonical_bytes(
            scope=self.scope,
            release_id=self.release_id,
            release_content_sha256=self.release_content_sha256,
            trajectory_id=self.trajectory_id,
            query_result_id=self.query_result_id,
            query_result_content_sha256=self.query_result_content_sha256,
            renderer_id=self.renderer_id,
            renderer_version_sha256=self.renderer_version_sha256,
            rendered_context_sha256=self.rendered_context_sha256,
            rendered_context_utf8_bytes=self.rendered_context_utf8_bytes,
            rendered_spans=self.rendered_spans,
            delivery_nonce=self.delivery_nonce,
        )
        expected_hash = sha256(canonical).hexdigest()
        if _sha256(self.content_hash, "content_hash") != expected_hash:
            raise ValueError("content_hash disagrees with canonical delivery bytes")
        _record_id(
            self.delivery_id,
            "delivery_id",
            prefix="mdel_",
            content_hash=expected_hash,
        )
        return canonical


class MemoryConsumerKind(StrEnum):
    CONTEXT = "context"
    MODEL_CALL = "model_call"


def _ack_canonical_bytes(
    *,
    scope: object,
    trajectory_id: object,
    delivery_id: object,
    delivery_content_sha256: object,
    delivery_nonce_sha256: object,
    consumer_kind: object,
    consumer_id: object,
    consumer_version_sha256: object,
    call_id: object,
    submitted_prompt_sha256: object,
    submitted_prompt_context_start: object,
    submitted_prompt_context_end: object,
    submitted_prompt_context_sha256: object,
    submitted_prompt_context_utf8_bytes: object,
    observed_query_sha256: object,
    observed_history_sha256: object,
    observed_history_length: object,
    submitted_input_token_ids_sha256: object,
    submitted_input_token_count: object,
) -> bytes:
    if type(consumer_kind) is not MemoryConsumerKind:
        raise TypeError("consumer_kind must be a MemoryConsumerKind")
    start = _integer(
        submitted_prompt_context_start,
        "submitted_prompt_context_start",
    )
    end = _integer(submitted_prompt_context_end, "submitted_prompt_context_end")
    byte_count = _integer(
        submitted_prompt_context_utf8_bytes,
        "submitted_prompt_context_utf8_bytes",
    )
    if end < start or end - start != byte_count:
        raise ValueError("submitted prompt context offsets disagree with byte count")
    token_hash = submitted_input_token_ids_sha256
    token_count = submitted_input_token_count
    if consumer_kind is MemoryConsumerKind.CONTEXT:
        if token_hash is not None or token_count is not None:
            raise ValueError("context consumers must not report model token fields")
    else:
        token_hash = _sha256(token_hash, "submitted_input_token_ids_sha256")
        token_count = _integer(token_count, "submitted_input_token_count", minimum=1)
    return _canonical_json_bytes(
        {
            "call_id": _string(call_id, "call_id"),
            "consumer_id": _string(consumer_id, "consumer_id"),
            "consumer_kind": consumer_kind.value,
            "consumer_version_sha256": _sha256(
                consumer_version_sha256,
                "consumer_version_sha256",
            ),
            "delivery_content_sha256": _sha256(
                delivery_content_sha256,
                "delivery_content_sha256",
            ),
            "delivery_id": _string(delivery_id, "delivery_id"),
            "delivery_nonce_sha256": _sha256(
                delivery_nonce_sha256,
                "delivery_nonce_sha256",
            ),
            "record_kind": "memory_consumer_ack",
            "schema_version": _SCHEMA_VERSION,
            "scope": _scope_value(scope),
            "observed_history_length": _integer(
                observed_history_length,
                "observed_history_length",
            ),
            "observed_history_sha256": _sha256(
                observed_history_sha256,
                "observed_history_sha256",
            ),
            "observed_query_sha256": _sha256(
                observed_query_sha256,
                "observed_query_sha256",
            ),
            "submitted_input_token_count": token_count,
            "submitted_input_token_ids_sha256": token_hash,
            "submitted_prompt_context_end": end,
            "submitted_prompt_context_sha256": _sha256(
                submitted_prompt_context_sha256,
                "submitted_prompt_context_sha256",
            ),
            "submitted_prompt_context_start": start,
            "submitted_prompt_context_utf8_bytes": byte_count,
            "submitted_prompt_sha256": _sha256(
                submitted_prompt_sha256,
                "submitted_prompt_sha256",
            ),
            "trajectory_id": _string(trajectory_id, "trajectory_id"),
        }
    )


@dataclass(frozen=True, slots=True)
class MemoryConsumerAckV1:
    """A receipt computed where exact context enters a consumer/model call."""

    scope: MemoryScope
    trajectory_id: str
    delivery_id: str
    delivery_content_sha256: str
    delivery_nonce_sha256: str
    consumer_kind: MemoryConsumerKind
    consumer_id: str
    consumer_version_sha256: str
    call_id: str
    submitted_prompt_sha256: str
    submitted_prompt_context_start: int
    submitted_prompt_context_end: int
    submitted_prompt_context_sha256: str
    submitted_prompt_context_utf8_bytes: int
    observed_query_sha256: str
    observed_history_sha256: str
    observed_history_length: int
    submitted_input_token_ids_sha256: str | None
    submitted_input_token_count: int | None
    consumer_ack_id: str
    content_hash: str
    created_at: datetime

    def __post_init__(self) -> None:
        canonical = _ack_canonical_bytes(
            scope=self.scope,
            trajectory_id=self.trajectory_id,
            delivery_id=self.delivery_id,
            delivery_content_sha256=self.delivery_content_sha256,
            delivery_nonce_sha256=self.delivery_nonce_sha256,
            consumer_kind=self.consumer_kind,
            consumer_id=self.consumer_id,
            consumer_version_sha256=self.consumer_version_sha256,
            call_id=self.call_id,
            submitted_prompt_sha256=self.submitted_prompt_sha256,
            submitted_prompt_context_start=self.submitted_prompt_context_start,
            submitted_prompt_context_end=self.submitted_prompt_context_end,
            submitted_prompt_context_sha256=self.submitted_prompt_context_sha256,
            submitted_prompt_context_utf8_bytes=(
                self.submitted_prompt_context_utf8_bytes
            ),
            observed_query_sha256=self.observed_query_sha256,
            observed_history_sha256=self.observed_history_sha256,
            observed_history_length=self.observed_history_length,
            submitted_input_token_ids_sha256=(self.submitted_input_token_ids_sha256),
            submitted_input_token_count=self.submitted_input_token_count,
        )
        expected_hash = sha256(canonical).hexdigest()
        object.__setattr__(
            self,
            "consumer_ack_id",
            _record_id(
                self.consumer_ack_id,
                "consumer_ack_id",
                prefix="mack_",
                content_hash=expected_hash,
            ),
        )
        if _sha256(self.content_hash, "content_hash") != expected_hash:
            raise ValueError("content_hash disagrees with canonical consumer ack bytes")
        object.__setattr__(
            self,
            "created_at",
            _aware_datetime(self.created_at, "created_at"),
        )

    @classmethod
    def create(
        cls,
        *,
        delivery: MemoryDeliveryV1,
        consumer_kind: MemoryConsumerKind,
        consumer_id: str,
        consumer_version_sha256: str,
        call_id: str,
        submitted_prompt_sha256: str,
        submitted_prompt_context_start: int,
        submitted_prompt_context_end: int,
        submitted_prompt_context_sha256: str,
        submitted_prompt_context_utf8_bytes: int,
        observed_query_sha256: str,
        observed_history_sha256: str,
        observed_history_length: int,
        submitted_input_token_ids_sha256: str | None,
        submitted_input_token_count: int | None,
        created_at: datetime | None = None,
    ) -> MemoryConsumerAckV1:
        if type(delivery) is not MemoryDeliveryV1:
            raise TypeError("delivery must be a MemoryDeliveryV1")
        values = {
            "scope": delivery.scope,
            "trajectory_id": delivery.trajectory_id,
            "delivery_id": delivery.delivery_id,
            "delivery_content_sha256": delivery.content_hash,
            "delivery_nonce_sha256": sha256(
                bytes.fromhex(delivery.delivery_nonce)
            ).hexdigest(),
            "consumer_kind": consumer_kind,
            "consumer_id": consumer_id,
            "consumer_version_sha256": consumer_version_sha256,
            "call_id": call_id,
            "submitted_prompt_sha256": submitted_prompt_sha256,
            "submitted_prompt_context_start": submitted_prompt_context_start,
            "submitted_prompt_context_end": submitted_prompt_context_end,
            "submitted_prompt_context_sha256": submitted_prompt_context_sha256,
            "submitted_prompt_context_utf8_bytes": (
                submitted_prompt_context_utf8_bytes
            ),
            "observed_query_sha256": observed_query_sha256,
            "observed_history_sha256": observed_history_sha256,
            "observed_history_length": observed_history_length,
            "submitted_input_token_ids_sha256": submitted_input_token_ids_sha256,
            "submitted_input_token_count": submitted_input_token_count,
        }
        canonical = _ack_canonical_bytes(**values)
        content_hash = sha256(canonical).hexdigest()
        return cls(
            **values,
            consumer_ack_id=f"mack_{content_hash[:24]}",
            content_hash=content_hash,
            created_at=datetime.now(UTC) if created_at is None else created_at,
        )

    def canonical_bytes(self) -> bytes:
        canonical = _ack_canonical_bytes(
            scope=self.scope,
            trajectory_id=self.trajectory_id,
            delivery_id=self.delivery_id,
            delivery_content_sha256=self.delivery_content_sha256,
            delivery_nonce_sha256=self.delivery_nonce_sha256,
            consumer_kind=self.consumer_kind,
            consumer_id=self.consumer_id,
            consumer_version_sha256=self.consumer_version_sha256,
            call_id=self.call_id,
            submitted_prompt_sha256=self.submitted_prompt_sha256,
            submitted_prompt_context_start=self.submitted_prompt_context_start,
            submitted_prompt_context_end=self.submitted_prompt_context_end,
            submitted_prompt_context_sha256=self.submitted_prompt_context_sha256,
            submitted_prompt_context_utf8_bytes=(
                self.submitted_prompt_context_utf8_bytes
            ),
            observed_query_sha256=self.observed_query_sha256,
            observed_history_sha256=self.observed_history_sha256,
            observed_history_length=self.observed_history_length,
            submitted_input_token_ids_sha256=(self.submitted_input_token_ids_sha256),
            submitted_input_token_count=self.submitted_input_token_count,
        )
        expected_hash = sha256(canonical).hexdigest()
        if _sha256(self.content_hash, "content_hash") != expected_hash:
            raise ValueError("content_hash disagrees with canonical consumer ack bytes")
        _record_id(
            self.consumer_ack_id,
            "consumer_ack_id",
            prefix="mack_",
            content_hash=expected_hash,
        )
        return canonical


class MemoryExposureStatus(StrEnum):
    MEMORY_OFF = "memory_off"
    NO_MEMORY_RETURNED = "no_memory_returned"
    DELIVERED = "delivered"


def _exposure_status(
    eligible: tuple[MemoryRevisionRefV1, ...],
    returned: tuple[MemoryRevisionRefV1, ...],
    injected: tuple[MemoryRevisionRefV1, ...],
) -> MemoryExposureStatus:
    if not eligible:
        if returned or injected:
            raise ValueError("an empty release cannot return or inject memory")
        return MemoryExposureStatus.MEMORY_OFF
    if not returned:
        if injected:
            raise ValueError("an empty result cannot inject memory")
        return MemoryExposureStatus.NO_MEMORY_RETURNED
    if injected == returned:
        return MemoryExposureStatus.DELIVERED
    raise ValueError("V1 exposure requires all returned revisions to be injected")


def _exposure_canonical_bytes(
    *,
    scope: object,
    release_id: object,
    release_content_sha256: object,
    trajectory_id: object,
    rollout_group_id: object,
    attempt_id: object,
    attempt_content_sha256: object,
    query_result_id: object,
    query_result_content_sha256: object,
    delivery_id: object,
    delivery_content_sha256: object,
    consumer_ack_id: object,
    consumer_ack_content_sha256: object,
    eligible_revisions: object,
    retrieved_revisions: object,
    returned_revisions: object,
    injected_revisions: object,
    status: object,
) -> bytes:
    eligible = _revision_refs(eligible_revisions, "eligible_revisions")
    retrieved = _revision_refs(retrieved_revisions, "retrieved_revisions")
    returned = _revision_refs(returned_revisions, "returned_revisions")
    injected = _revision_refs(injected_revisions, "injected_revisions")
    if not _is_unique_subset(retrieved, eligible):
        raise ValueError("retrieved_revisions must be a unique eligible subset")
    if not _is_unique_subsequence(returned, retrieved):
        raise ValueError("returned_revisions must be an ordered retrieved subsequence")
    if not _is_unique_subsequence(injected, returned):
        raise ValueError("injected_revisions must be an ordered returned subsequence")
    expected_status = _exposure_status(eligible, returned, injected)
    if type(status) is not MemoryExposureStatus or status is not expected_status:
        raise ValueError("status disagrees with returned and injected revisions")
    return _canonical_json_bytes(
        {
            "attempt_content_sha256": _sha256(
                attempt_content_sha256,
                "attempt_content_sha256",
            ),
            "attempt_id": _string(attempt_id, "attempt_id"),
            "consumer_ack_content_sha256": _sha256(
                consumer_ack_content_sha256,
                "consumer_ack_content_sha256",
            ),
            "consumer_ack_id": _string(consumer_ack_id, "consumer_ack_id"),
            "delivery_content_sha256": _sha256(
                delivery_content_sha256,
                "delivery_content_sha256",
            ),
            "delivery_id": _string(delivery_id, "delivery_id"),
            "eligible_revisions": [item._canonical_value() for item in eligible],
            "injected_revisions": [item._canonical_value() for item in injected],
            "query_result_content_sha256": _sha256(
                query_result_content_sha256,
                "query_result_content_sha256",
            ),
            "query_result_id": _string(query_result_id, "query_result_id"),
            "record_kind": "memory_exposure",
            "release_content_sha256": _sha256(
                release_content_sha256,
                "release_content_sha256",
            ),
            "release_id": _string(release_id, "release_id"),
            "retrieved_revisions": [item._canonical_value() for item in retrieved],
            "returned_revisions": [item._canonical_value() for item in returned],
            "rollout_group_id": _string(rollout_group_id, "rollout_group_id"),
            "schema_version": _SCHEMA_VERSION,
            "scope": _scope_value(scope),
            "status": status.value,
            "trajectory_id": _string(trajectory_id, "trajectory_id"),
        }
    )


@dataclass(frozen=True, slots=True)
class MemoryExposureV1:
    """The store-joined record of what exact memory reached a consumer."""

    scope: MemoryScope
    release_id: str
    release_content_sha256: str
    trajectory_id: str
    rollout_group_id: str
    attempt_id: str
    attempt_content_sha256: str
    query_result_id: str
    query_result_content_sha256: str
    delivery_id: str
    delivery_content_sha256: str
    consumer_ack_id: str
    consumer_ack_content_sha256: str
    eligible_revisions: tuple[MemoryRevisionRefV1, ...]
    retrieved_revisions: tuple[MemoryRevisionRefV1, ...]
    returned_revisions: tuple[MemoryRevisionRefV1, ...]
    injected_revisions: tuple[MemoryRevisionRefV1, ...]
    status: MemoryExposureStatus
    exposure_id: str
    content_hash: str
    created_at: datetime

    def __post_init__(self) -> None:
        canonical = _exposure_canonical_bytes(
            scope=self.scope,
            release_id=self.release_id,
            release_content_sha256=self.release_content_sha256,
            trajectory_id=self.trajectory_id,
            rollout_group_id=self.rollout_group_id,
            attempt_id=self.attempt_id,
            attempt_content_sha256=self.attempt_content_sha256,
            query_result_id=self.query_result_id,
            query_result_content_sha256=self.query_result_content_sha256,
            delivery_id=self.delivery_id,
            delivery_content_sha256=self.delivery_content_sha256,
            consumer_ack_id=self.consumer_ack_id,
            consumer_ack_content_sha256=self.consumer_ack_content_sha256,
            eligible_revisions=self.eligible_revisions,
            retrieved_revisions=self.retrieved_revisions,
            returned_revisions=self.returned_revisions,
            injected_revisions=self.injected_revisions,
            status=self.status,
        )
        expected_hash = sha256(canonical).hexdigest()
        object.__setattr__(
            self,
            "exposure_id",
            _record_id(
                self.exposure_id,
                "exposure_id",
                prefix="mexp_",
                content_hash=expected_hash,
            ),
        )
        if _sha256(self.content_hash, "content_hash") != expected_hash:
            raise ValueError("content_hash disagrees with canonical exposure bytes")
        object.__setattr__(
            self,
            "created_at",
            _aware_datetime(self.created_at, "created_at"),
        )

    @classmethod
    def create(
        cls,
        *,
        attempt: MemoryQueryAttemptV1,
        query_result: MemoryQueryResultV1,
        delivery: MemoryDeliveryV1,
        consumer_ack: MemoryConsumerAckV1,
        created_at: datetime | None = None,
    ) -> MemoryExposureV1:
        if (
            type(attempt) is not MemoryQueryAttemptV1
            or type(query_result) is not MemoryQueryResultV1
            or type(delivery) is not MemoryDeliveryV1
            or type(consumer_ack) is not MemoryConsumerAckV1
        ):
            raise TypeError("exposure sources must use exact Memory runtime types")
        returned = query_result.returned_revisions
        injected_revisions = tuple(item.revision for item in delivery.rendered_spans)
        if (
            query_result.scope != attempt.spec.scope
            or query_result.release_id != attempt.spec.release_id
            or query_result.release_content_sha256 != attempt.release_content_sha256
            or query_result.trajectory_id != attempt.spec.trajectory_id
            or query_result.rollout_group_id != attempt.spec.rollout_group_id
            or query_result.attempt_id != attempt.attempt_id
            or query_result.attempt_content_sha256 != attempt.content_hash
            or delivery.scope != query_result.scope
            or delivery.release_id != query_result.release_id
            or delivery.release_content_sha256 != query_result.release_content_sha256
            or delivery.trajectory_id != query_result.trajectory_id
            or delivery.query_result_id != query_result.query_result_id
            or delivery.query_result_content_sha256 != query_result.content_hash
            or consumer_ack.scope != delivery.scope
            or consumer_ack.trajectory_id != delivery.trajectory_id
            or consumer_ack.delivery_id != delivery.delivery_id
            or consumer_ack.delivery_content_sha256 != delivery.content_hash
            or consumer_ack.delivery_nonce_sha256
            != sha256(bytes.fromhex(delivery.delivery_nonce)).hexdigest()
            or consumer_ack.submitted_prompt_context_sha256
            != delivery.rendered_context_sha256
            or consumer_ack.submitted_prompt_context_utf8_bytes
            != delivery.rendered_context_utf8_bytes
            or consumer_ack.observed_query_sha256 != attempt.spec.query_sha256
            or injected_revisions != returned
        ):
            raise ValueError("exposure sources do not form one acknowledged chain")
        values = {
            "scope": attempt.spec.scope,
            "release_id": attempt.spec.release_id,
            "release_content_sha256": attempt.release_content_sha256,
            "trajectory_id": attempt.spec.trajectory_id,
            "rollout_group_id": attempt.spec.rollout_group_id,
            "attempt_id": attempt.attempt_id,
            "attempt_content_sha256": attempt.content_hash,
            "query_result_id": query_result.query_result_id,
            "query_result_content_sha256": query_result.content_hash,
            "delivery_id": delivery.delivery_id,
            "delivery_content_sha256": delivery.content_hash,
            "consumer_ack_id": consumer_ack.consumer_ack_id,
            "consumer_ack_content_sha256": consumer_ack.content_hash,
            "eligible_revisions": query_result.eligible_revisions,
            "retrieved_revisions": query_result.retrieved_revisions,
            "returned_revisions": returned,
            "injected_revisions": injected_revisions,
            "status": _exposure_status(
                query_result.eligible_revisions,
                returned,
                injected_revisions,
            ),
        }
        canonical = _exposure_canonical_bytes(**values)
        content_hash = sha256(canonical).hexdigest()
        return cls(
            **values,
            exposure_id=f"mexp_{content_hash[:24]}",
            content_hash=content_hash,
            created_at=datetime.now(UTC) if created_at is None else created_at,
        )

    def canonical_bytes(self) -> bytes:
        canonical = _exposure_canonical_bytes(
            scope=self.scope,
            release_id=self.release_id,
            release_content_sha256=self.release_content_sha256,
            trajectory_id=self.trajectory_id,
            rollout_group_id=self.rollout_group_id,
            attempt_id=self.attempt_id,
            attempt_content_sha256=self.attempt_content_sha256,
            query_result_id=self.query_result_id,
            query_result_content_sha256=self.query_result_content_sha256,
            delivery_id=self.delivery_id,
            delivery_content_sha256=self.delivery_content_sha256,
            consumer_ack_id=self.consumer_ack_id,
            consumer_ack_content_sha256=self.consumer_ack_content_sha256,
            eligible_revisions=self.eligible_revisions,
            retrieved_revisions=self.retrieved_revisions,
            returned_revisions=self.returned_revisions,
            injected_revisions=self.injected_revisions,
            status=self.status,
        )
        expected_hash = sha256(canonical).hexdigest()
        if _sha256(self.content_hash, "content_hash") != expected_hash:
            raise ValueError("content_hash disagrees with canonical exposure bytes")
        _record_id(
            self.exposure_id,
            "exposure_id",
            prefix="mexp_",
            content_hash=expected_hash,
        )
        return canonical
