# SPDX-License-Identifier: Apache-2.0

"""Immutable candidate and revision values for the Memory Service."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from areal.v2.memory_service.types import (
    MemoryScope,
    _validate_aware_datetime,
    _validate_string,
)

_SCHEMA_VERSION = 1
_MAX_GENERATION = 2**63 - 1


def _canonical_json_bytes(value: dict[str, object]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _snapshot_evidence_ids(value: object) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise TypeError("evidence_ids must be a tuple")
    snapshot = tuple(
        _validate_string(item, "evidence_ids") for item in tuple.__iter__(value)
    )
    if not snapshot:
        raise ValueError("evidence_ids must not be empty")
    if len(set(snapshot)) != len(snapshot):
        raise ValueError("evidence_ids must not contain duplicates")
    return snapshot


class RevisionOperation(StrEnum):
    ADD = "add"
    REFINE = "refine"
    SUPERSEDE = "supersede"
    CONTRADICT = "contradict"


@dataclass(frozen=True, slots=True)
class CandidateProposal:
    scope: MemoryScope
    content: str
    evidence_ids: tuple[str, ...]
    idempotency_key: str

    def __post_init__(self) -> None:
        if type(self.scope) is not MemoryScope:
            raise TypeError("scope must be a MemoryScope")
        content = _validate_string(self.content, "content")
        evidence_ids = _snapshot_evidence_ids(self.evidence_ids)
        idempotency_key = _validate_string(self.idempotency_key, "idempotency_key")
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "evidence_ids", evidence_ids)
        object.__setattr__(self, "idempotency_key", idempotency_key)

    def canonical_bytes(self) -> bytes:
        value = {
            "schema_version": _SCHEMA_VERSION,
            "scope": {
                "tenant_id": self.scope.tenant_id,
                "namespace": self.scope.namespace,
                "subject_id": self.scope.subject_id,
            },
            "content": self.content,
            "evidence_ids": list(self.evidence_ids),
            "idempotency_key": self.idempotency_key,
        }
        return _canonical_json_bytes(value)


@dataclass(frozen=True, slots=True)
class MemoryCandidate:
    candidate_id: str
    proposal: CandidateProposal
    content_hash: str
    created_at: datetime

    def __post_init__(self) -> None:
        if type(self.proposal) is not CandidateProposal:
            raise TypeError("proposal must be a CandidateProposal")
        candidate_id = _validate_string(self.candidate_id, "candidate_id")
        content_hash = _validate_string(self.content_hash, "content_hash")
        created_at = _validate_aware_datetime(self.created_at, "created_at")
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "content_hash", content_hash)
        object.__setattr__(self, "created_at", created_at)


@dataclass(frozen=True, slots=True)
class RevisionProposal:
    scope: MemoryScope
    candidate_id: str
    operation: RevisionOperation
    parent_revision_id: str | None
    idempotency_key: str

    def __post_init__(self) -> None:
        if type(self.scope) is not MemoryScope:
            raise TypeError("scope must be a MemoryScope")
        candidate_id = _validate_string(self.candidate_id, "candidate_id")
        if type(self.operation) is not RevisionOperation:
            raise TypeError("operation must be a RevisionOperation")
        parent_revision_id = self.parent_revision_id
        if parent_revision_id is not None:
            parent_revision_id = _validate_string(
                parent_revision_id, "parent_revision_id"
            )
        if self.operation is RevisionOperation.ADD and parent_revision_id is not None:
            raise ValueError("parent_revision_id must be absent for ADD")
        if self.operation is not RevisionOperation.ADD and parent_revision_id is None:
            raise ValueError("parent_revision_id is required for non-ADD operations")
        idempotency_key = _validate_string(self.idempotency_key, "idempotency_key")
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "parent_revision_id", parent_revision_id)
        object.__setattr__(self, "idempotency_key", idempotency_key)

    def canonical_bytes(self) -> bytes:
        value = {
            "schema_version": _SCHEMA_VERSION,
            "scope": {
                "tenant_id": self.scope.tenant_id,
                "namespace": self.scope.namespace,
                "subject_id": self.scope.subject_id,
            },
            "candidate_id": self.candidate_id,
            "operation": self.operation.value,
            "parent_revision_id": self.parent_revision_id,
            "idempotency_key": self.idempotency_key,
        }
        return _canonical_json_bytes(value)


@dataclass(frozen=True, slots=True)
class MemoryRevision:
    revision_id: str
    memory_id: str
    generation: int
    proposal: RevisionProposal
    content_hash: str
    created_at: datetime

    def __post_init__(self) -> None:
        revision_id = _validate_string(self.revision_id, "revision_id")
        memory_id = _validate_string(self.memory_id, "memory_id")
        if type(self.generation) is not int:
            raise TypeError("generation must be an integer")
        if self.generation < 0 or self.generation > _MAX_GENERATION:
            raise ValueError("generation must fit the non-negative signed-64 range")
        if type(self.proposal) is not RevisionProposal:
            raise TypeError("proposal must be a RevisionProposal")
        content_hash = _validate_string(self.content_hash, "content_hash")
        created_at = _validate_aware_datetime(self.created_at, "created_at")
        object.__setattr__(self, "revision_id", revision_id)
        object.__setattr__(self, "memory_id", memory_id)
        object.__setattr__(self, "content_hash", content_hash)
        object.__setattr__(self, "created_at", created_at)
