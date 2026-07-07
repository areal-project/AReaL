# SPDX-License-Identifier: Apache-2.0

"""Errors raised by the Memory Service."""

from __future__ import annotations


class MemoryServiceError(Exception):
    """Base class for Memory Service failures."""


class EvidenceNotFoundError(MemoryServiceError):
    """Raised when requested evidence is unavailable in the requested scope."""


class EvidenceConflictError(MemoryServiceError):
    """Raised when evidence conflicts with an existing immutable record."""


class CandidateNotFoundError(MemoryServiceError):
    """Raised when a candidate is unavailable in the requested scope."""


class CandidateConflictError(MemoryServiceError):
    """Raised when a candidate write conflicts with immutable history."""


class RevisionNotFoundError(MemoryServiceError):
    """Raised when a revision is unavailable in the requested scope."""


class RevisionConflictError(MemoryServiceError):
    """Raised when a revision write conflicts with immutable history."""
