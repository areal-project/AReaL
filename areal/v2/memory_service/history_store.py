# SPDX-License-Identifier: Apache-2.0

"""History storage contracts and an in-memory reference implementation."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from threading import RLock
from typing import Protocol

from areal.v2.memory_service.errors import (
    CandidateConflictError,
    CandidateNotFoundError,
)
from areal.v2.memory_service.history_types import (
    CandidateProposal,
    MemoryCandidate,
)
from areal.v2.memory_service.store import EvidenceStore
from areal.v2.memory_service.types import (
    EvidenceRecord,
    MemoryScope,
    _validate_string,
)


class MemoryHistoryStore(Protocol):
    """Storage contract for immutable memory candidates."""

    def append_candidate(self, proposal: CandidateProposal) -> MemoryCandidate:
        """Persist one evidence-grounded candidate."""

        ...

    def get_candidate(self, scope: MemoryScope, candidate_id: str) -> MemoryCandidate:
        """Return one scoped candidate."""

        ...

    def get_candidate_evidence(
        self, scope: MemoryScope, candidate_id: str
    ) -> tuple[EvidenceRecord, ...]:
        """Resolve candidate evidence in proposal order."""

        ...

    def list_candidates(self, scope: MemoryScope) -> tuple[MemoryCandidate, ...]:
        """Return candidates in stable identifier order."""

        ...


class InMemoryMemoryHistoryStore:
    """Lock-protected process-local storage for immutable memory candidates."""

    def __init__(self, evidence_store: EvidenceStore) -> None:
        self._evidence_store = evidence_store
        self._lock = RLock()
        self._candidate_by_id: dict[tuple[MemoryScope, str], MemoryCandidate] = {}
        self._candidate_by_idempotency: dict[
            tuple[MemoryScope, str], MemoryCandidate
        ] = {}
        self._candidates_by_scope: dict[MemoryScope, list[MemoryCandidate]] = {}

    def append_candidate(self, proposal: CandidateProposal) -> MemoryCandidate:
        """Persist a candidate after validating all referenced evidence."""

        if type(proposal) is not CandidateProposal:
            raise TypeError("proposal must be a CandidateProposal")
        canonical_bytes = proposal.canonical_bytes()
        content_hash = hashlib.sha256(canonical_bytes).hexdigest()
        candidate_id = f"cand_{content_hash[:24]}"
        candidate_index = (proposal.scope, candidate_id)
        idempotency_index = (proposal.scope, proposal.idempotency_key)

        with self._lock:
            existing = self._candidate_by_idempotency.get(idempotency_index)
            if existing is not None:
                if existing.proposal.canonical_bytes() == canonical_bytes:
                    return existing
                raise CandidateConflictError(
                    "scoped candidate idempotency key already refers to different content"
                )

        # Evidence is append-only, so this outside-lock preflight cannot be invalidated.
        for evidence_id in proposal.evidence_ids:
            self._evidence_store.get(proposal.scope, evidence_id)

        with self._lock:
            # Mandatory recheck closes the validation/insertion race.
            existing = self._candidate_by_idempotency.get(idempotency_index)
            if existing is not None:
                if existing.proposal.canonical_bytes() == canonical_bytes:
                    return existing
                raise CandidateConflictError(
                    "scoped candidate idempotency key already refers to different content"
                )
            existing = self._candidate_by_id.get(candidate_index)
            if existing is not None:
                if existing.proposal.canonical_bytes() == canonical_bytes:
                    return existing
                raise CandidateConflictError(
                    f"candidate ID collision for {candidate_id!r}"
                )
            candidate = MemoryCandidate(
                candidate_id=candidate_id,
                proposal=proposal,
                content_hash=content_hash,
                created_at=datetime.now(UTC),
            )
            self._candidate_by_id[candidate_index] = candidate
            self._candidate_by_idempotency[idempotency_index] = candidate
            self._candidates_by_scope.setdefault(proposal.scope, []).append(candidate)
            return candidate

    def get_candidate(self, scope: MemoryScope, candidate_id: str) -> MemoryCandidate:
        """Return a candidate only when it belongs to the requested scope."""

        if type(scope) is not MemoryScope:
            raise TypeError("scope must be a MemoryScope")
        candidate_id = _validate_string(candidate_id, "candidate_id", allow_blank=True)
        with self._lock:
            candidate = self._candidate_by_id.get((scope, candidate_id))
            if candidate is None:
                raise CandidateNotFoundError(
                    f"candidate {candidate_id!r} was not found"
                )
            return candidate

    def get_candidate_evidence(
        self, scope: MemoryScope, candidate_id: str
    ) -> tuple[EvidenceRecord, ...]:
        """Resolve a candidate's evidence in its proposal order."""

        candidate = self.get_candidate(scope, candidate_id)
        return tuple(
            self._evidence_store.get(scope, evidence_id)
            for evidence_id in candidate.proposal.evidence_ids
        )

    def list_candidates(self, scope: MemoryScope) -> tuple[MemoryCandidate, ...]:
        """Return a stable snapshot of candidates in the requested scope."""

        if type(scope) is not MemoryScope:
            raise TypeError("scope must be a MemoryScope")
        with self._lock:
            return tuple(
                sorted(
                    self._candidates_by_scope.get(scope, ()),
                    key=lambda item: item.candidate_id,
                )
            )
