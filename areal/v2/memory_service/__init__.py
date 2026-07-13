# SPDX-License-Identifier: Apache-2.0

"""Public contracts for immutable Memory Service evidence, history, and releases."""

from areal.v2.memory_service.errors import (
    CandidateConflictError,
    CandidateNotFoundError,
    EvidenceConflictError,
    EvidenceNotFoundError,
    MemoryReleaseAssignmentConflictError,
    MemoryReleaseAssignmentNotFoundError,
    MemoryReleaseAttestationConflictError,
    MemoryReleaseAttestationNotFoundError,
    MemoryReleaseRevocationConflictError,
    MemoryReleaseRevocationNotFoundError,
    MemoryServiceError,
    ReleaseConflictError,
    ReleaseNotFoundError,
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
from areal.v2.memory_service.release_control_types import (
    MemoryReleaseAssignmentConsumerKind,
    MemoryReleaseAssignmentV1,
    MemoryReleaseAttestationRevocationV1,
    MemoryReleaseAttestationV1,
    MemoryReleaseRevocationReason,
)
from areal.v2.memory_service.release_store import (
    InMemoryMemoryReleaseStore,
    MemoryReleaseStore,
)
from areal.v2.memory_service.release_types import MemoryRelease, ReleaseManifest
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
    "InMemoryMemoryReleaseStore",
    "MemoryCandidate",
    "MemoryHistoryStore",
    "MemoryRelease",
    "MemoryReleaseAssignmentConsumerKind",
    "MemoryReleaseAssignmentConflictError",
    "MemoryReleaseAssignmentNotFoundError",
    "MemoryReleaseAssignmentV1",
    "MemoryReleaseAttestationConflictError",
    "MemoryReleaseAttestationNotFoundError",
    "MemoryReleaseAttestationRevocationV1",
    "MemoryReleaseAttestationV1",
    "MemoryReleaseRevocationConflictError",
    "MemoryReleaseRevocationNotFoundError",
    "MemoryReleaseRevocationReason",
    "MemoryReleaseStore",
    "MemoryRevision",
    "MemoryScope",
    "MemoryServiceError",
    "ReleaseConflictError",
    "ReleaseManifest",
    "ReleaseNotFoundError",
    "RevisionConflictError",
    "RevisionNotFoundError",
    "RevisionOperation",
    "RevisionProposal",
]
