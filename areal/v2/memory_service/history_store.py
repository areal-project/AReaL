# SPDX-License-Identifier: Apache-2.0

"""History storage contracts and an in-memory reference implementation."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from threading import RLock
from typing import Protocol

from areal.v2.memory_service._atomic import _atomic_publish
from areal.v2.memory_service.errors import (
    CandidateConflictError,
    CandidateNotFoundError,
    RevisionConflictError,
    RevisionNotFoundError,
)
from areal.v2.memory_service.history_types import (
    CandidateProposal,
    MemoryCandidate,
    MemoryRevision,
    RevisionOperation,
    RevisionProposal,
)
from areal.v2.memory_service.store import EvidenceStore
from areal.v2.memory_service.types import (
    EvidenceRecord,
    MemoryScope,
    _validate_string,
)


class MemoryHistoryStore(Protocol):
    """Storage contract for immutable memory candidates and revisions."""

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

    def append_revision(self, proposal: RevisionProposal) -> MemoryRevision:
        """Persist one immutable candidate transition."""

        ...

    def get_revision(self, scope: MemoryScope, revision_id: str) -> MemoryRevision:
        """Return one scoped revision."""

        ...

    def list_revisions(
        self, scope: MemoryScope, *, memory_id: str | None = None
    ) -> tuple[MemoryRevision, ...]:
        """Return parent-before-child immutable revisions."""

        ...


class InMemoryMemoryHistoryStore:
    """Lock-protected process-local immutable memory history."""

    def __init__(self, evidence_store: EvidenceStore) -> None:
        self._evidence_store = evidence_store
        self._lock = RLock()
        self._candidate_by_id: dict[tuple[MemoryScope, str], MemoryCandidate] = {}
        self._candidate_by_idempotency: dict[
            tuple[MemoryScope, str], MemoryCandidate
        ] = {}
        self._candidates_by_scope: dict[MemoryScope, list[MemoryCandidate]] = {}
        self._revision_by_id: dict[tuple[MemoryScope, str], MemoryRevision] = {}
        self._revision_by_idempotency: dict[
            tuple[MemoryScope, str], MemoryRevision
        ] = {}
        self._revision_by_candidate: dict[tuple[MemoryScope, str], MemoryRevision] = {}
        self._revisions_by_scope: dict[MemoryScope, list[MemoryRevision]] = {}

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
            _atomic_publish(
                mapping_writes=(
                    (self._candidate_by_id, candidate_index, candidate),
                    (
                        self._candidate_by_idempotency,
                        idempotency_index,
                        candidate,
                    ),
                ),
                sequence_appends=(
                    (self._candidates_by_scope, proposal.scope, candidate),
                ),
            )
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

    def append_revision(self, proposal: RevisionProposal) -> MemoryRevision:
        """Persist one immutable candidate transition."""

        if type(proposal) is not RevisionProposal:
            raise TypeError("proposal must be a RevisionProposal")
        canonical_bytes = proposal.canonical_bytes()
        content_hash = hashlib.sha256(canonical_bytes).hexdigest()
        revision_id = f"rev_{content_hash[:24]}"
        revision_index = (proposal.scope, revision_id)
        idempotency_index = (proposal.scope, proposal.idempotency_key)
        candidate_index = (proposal.scope, proposal.candidate_id)

        with self._lock:
            existing = self._revision_by_idempotency.get(idempotency_index)
            if existing is not None:
                if existing.proposal.canonical_bytes() == canonical_bytes:
                    return existing
                raise RevisionConflictError(
                    "scoped revision idempotency key already refers to different content"
                )

            existing = self._revision_by_id.get(revision_index)
            if existing is not None:
                if existing.proposal.canonical_bytes() == canonical_bytes:
                    return existing
                raise RevisionConflictError(
                    f"revision ID collision for {revision_id!r}"
                )

            candidate = self._candidate_by_id.get(candidate_index)
            if candidate is None:
                raise CandidateNotFoundError(
                    f"candidate {proposal.candidate_id!r} was not found"
                )
            if candidate_index in self._revision_by_candidate:
                raise RevisionConflictError(
                    f"candidate {proposal.candidate_id!r} already backs a revision"
                )

            if proposal.operation is RevisionOperation.ADD:
                memory_id = f"mem_{content_hash[:24]}"
                generation = 0
            else:
                assert proposal.parent_revision_id is not None
                parent = self._revision_by_id.get(
                    (proposal.scope, proposal.parent_revision_id)
                )
                if parent is None:
                    raise RevisionNotFoundError(
                        f"revision {proposal.parent_revision_id!r} was not found"
                    )
                if parent.generation == 2**63 - 1:
                    raise RevisionConflictError(
                        "revision generation exceeds the signed-64 range"
                    )
                memory_id = parent.memory_id
                generation = parent.generation + 1

            revision = MemoryRevision(
                revision_id=revision_id,
                memory_id=memory_id,
                generation=generation,
                proposal=proposal,
                content_hash=content_hash,
                created_at=datetime.now(UTC),
            )
            _atomic_publish(
                mapping_writes=(
                    (self._revision_by_id, revision_index, revision),
                    (
                        self._revision_by_idempotency,
                        idempotency_index,
                        revision,
                    ),
                    (self._revision_by_candidate, candidate_index, revision),
                ),
                sequence_appends=(
                    (self._revisions_by_scope, proposal.scope, revision),
                ),
            )
            return revision

    def get_revision(self, scope: MemoryScope, revision_id: str) -> MemoryRevision:
        """Return a revision only when it belongs to the requested scope."""

        if type(scope) is not MemoryScope:
            raise TypeError("scope must be a MemoryScope")
        revision_id = _validate_string(revision_id, "revision_id", allow_blank=True)
        with self._lock:
            revision = self._revision_by_id.get((scope, revision_id))
            if revision is None:
                raise RevisionNotFoundError(f"revision {revision_id!r} was not found")
            return revision

    def list_revisions(
        self, scope: MemoryScope, *, memory_id: str | None = None
    ) -> tuple[MemoryRevision, ...]:
        """Return a stable, parent-before-child revision snapshot."""

        if type(scope) is not MemoryScope:
            raise TypeError("scope must be a MemoryScope")
        if memory_id is not None:
            memory_id = _validate_string(memory_id, "memory_id", allow_blank=True)
        with self._lock:
            matches = (
                revision
                for revision in self._revisions_by_scope.get(scope, ())
                if memory_id is None or revision.memory_id == memory_id
            )
            return tuple(
                sorted(
                    matches,
                    key=lambda item: (
                        item.memory_id,
                        item.generation,
                        item.revision_id,
                    ),
                )
            )
