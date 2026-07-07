# SPDX-License-Identifier: Apache-2.0

"""Public contracts for immutable Memory Service evidence."""

from areal.v2.memory_service.errors import (
    EvidenceConflictError,
    EvidenceNotFoundError,
    MemoryServiceError,
)
from areal.v2.memory_service.store import EvidenceStore, InMemoryEvidenceStore
from areal.v2.memory_service.types import (
    EvidenceEvent,
    EvidenceKind,
    EvidenceRecord,
    MemoryScope,
)

__all__ = [
    "EvidenceConflictError",
    "EvidenceEvent",
    "EvidenceKind",
    "EvidenceNotFoundError",
    "EvidenceRecord",
    "EvidenceStore",
    "InMemoryEvidenceStore",
    "MemoryScope",
    "MemoryServiceError",
]
