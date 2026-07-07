# SPDX-License-Identifier: Apache-2.0

"""Evidence store contracts and an in-memory implementation."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from threading import RLock
from typing import Protocol

from areal.v2.memory_service.errors import (
    EvidenceConflictError,
    EvidenceNotFoundError,
)
from areal.v2.memory_service.types import EvidenceEvent, EvidenceRecord, MemoryScope


class EvidenceStore(Protocol):
    """Storage contract for immutable evidence records."""

    def append(self, event: EvidenceEvent) -> EvidenceRecord:
        """Persist an event or return its existing idempotent record."""

        ...

    def get(self, scope: MemoryScope, evidence_id: str) -> EvidenceRecord:
        """Return evidence from a scope or raise ``EvidenceNotFoundError``."""

        ...

    def list(
        self,
        scope: MemoryScope,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> tuple[EvidenceRecord, ...]:
        """Return deterministically ordered evidence matching the filters."""

        ...


class InMemoryEvidenceStore:
    """Lock-protected process-local storage for immutable evidence records."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._by_evidence_id: dict[tuple[MemoryScope, str], EvidenceRecord] = {}
        self._by_idempotency_key: dict[tuple[MemoryScope, str], EvidenceRecord] = {}
        self._by_scope: dict[MemoryScope, list[EvidenceRecord]] = {}

    def append(self, event: EvidenceEvent) -> EvidenceRecord:
        """Persist an event, enforcing scoped idempotency and collision safety."""

        if type(event) is not EvidenceEvent:
            raise TypeError("event must be an EvidenceEvent")
        canonical_bytes = event.canonical_bytes()
        content_hash = hashlib.sha256(canonical_bytes).hexdigest()
        evidence_id = f"evd_{content_hash[:24]}"
        evidence_index = (event.scope, evidence_id)
        idempotency_index = (event.scope, event.idempotency_key)

        with self._lock:
            existing_record = self._by_idempotency_key.get(idempotency_index)
            if existing_record is not None:
                if existing_record.event.canonical_bytes() == canonical_bytes:
                    return existing_record
                raise EvidenceConflictError(
                    "scoped idempotency key already refers to different evidence"
                )

            existing_record = self._by_evidence_id.get(evidence_index)
            if existing_record is not None:
                if existing_record.event.canonical_bytes() == canonical_bytes:
                    return existing_record
                raise EvidenceConflictError(
                    f"evidence ID collision for {evidence_id!r}"
                )

            record = EvidenceRecord(
                evidence_id=evidence_id,
                event=event,
                content_hash=content_hash,
                created_at=datetime.now(UTC),
            )
            self._by_evidence_id[evidence_index] = record
            self._by_idempotency_key[idempotency_index] = record
            self._by_scope.setdefault(event.scope, []).append(record)
            return record

    def get(self, scope: MemoryScope, evidence_id: str) -> EvidenceRecord:
        """Return evidence only when it belongs to the requested scope."""

        if type(scope) is not MemoryScope:
            raise TypeError("scope must be a MemoryScope")
        with self._lock:
            record = self._by_evidence_id.get((scope, evidence_id))
            if record is None:
                raise EvidenceNotFoundError(f"evidence {evidence_id!r} was not found")
            return record

    def list(
        self,
        scope: MemoryScope,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> tuple[EvidenceRecord, ...]:
        """Return a stable snapshot of records belonging to the requested scope."""

        if type(scope) is not MemoryScope:
            raise TypeError("scope must be a MemoryScope")
        with self._lock:
            matches = (
                record
                for record in self._by_scope.get(scope, ())
                if (session_id is None or record.event.session_id == session_id)
                and (run_id is None or record.event.run_id == run_id)
            )
            return tuple(sorted(matches, key=_record_sort_key))


def _record_sort_key(record: EvidenceRecord) -> tuple[str, str, int, datetime, str]:
    event = record.event
    return (
        event.session_id,
        event.run_id,
        event.sequence_no,
        event.observed_at,
        record.evidence_id,
    )
