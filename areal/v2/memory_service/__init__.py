# SPDX-License-Identifier: Apache-2.0

"""Public contracts for immutable Memory Service evidence and history."""

from areal.v2.memory_service.errors import (
    CandidateConflictError,
    CandidateNotFoundError,
    EvidenceConflictError,
    EvidenceNotFoundError,
    MemoryServiceError,
    RevisionConflictError,
    RevisionNotFoundError,
)
from areal.v2.memory_service.history_store import (
    InMemoryMemoryHistoryStore,
    MemoryHistoryStore,
)
from areal.v2.memory_service.history_types import (
    CandidateProposal,
    MemoryCandidate,
    MemoryRevision,
    RevisionOperation,
    RevisionProposal,
)
from areal.v2.memory_service.store import EvidenceStore, InMemoryEvidenceStore
from areal.v2.memory_service.types import (
    EvidenceEvent,
    EvidenceKind,
    EvidenceRecord,
    MemoryScope,
)

__all__ = [
    "CandidateConflictError",
    "CandidateNotFoundError",
    "CandidateProposal",
    "EvidenceConflictError",
    "EvidenceEvent",
    "EvidenceKind",
    "EvidenceNotFoundError",
    "EvidenceRecord",
    "EvidenceStore",
    "InMemoryEvidenceStore",
    "InMemoryMemoryHistoryStore",
    "MemoryCandidate",
    "MemoryHistoryStore",
    "MemoryRevision",
    "MemoryScope",
    "MemoryServiceError",
    "RevisionConflictError",
    "RevisionNotFoundError",
    "RevisionOperation",
    "RevisionProposal",
]
