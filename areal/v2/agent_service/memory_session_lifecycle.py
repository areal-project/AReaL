# SPDX-License-Identifier: Apache-2.0

"""Descriptive identities and results for exact Worker session retirement.

The caller-visible session key is a reusable label, not a lifetime identity.
An exact Worker session is the tuple of its broker-minted session incarnation
and Worker audience.  These values are non-secret replay domains: possessing
them grants no authority, and a future HTTP adapter must authenticate its hop
before using them in a conditional close.

This module deliberately defines values only.  It does not register a route,
resolve a principal, or turn a reconstructed identity into a local capability.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .memory_authorization import (
    MemorySessionIncarnationV1,
    MemoryWorkerAudienceV1,
)


def _session(value: object) -> MemorySessionIncarnationV1:
    if type(value) is not MemorySessionIncarnationV1:
        raise TypeError("session must be a MemorySessionIncarnationV1")
    value.canonical_bytes()
    return MemorySessionIncarnationV1(
        session_key=value.session_key,
        incarnation_id=value.incarnation_id,
    )


def _audience(value: object) -> MemoryWorkerAudienceV1:
    if type(value) is not MemoryWorkerAudienceV1:
        raise TypeError("audience must be a MemoryWorkerAudienceV1")
    value.canonical_bytes()
    return MemoryWorkerAudienceV1(value.audience_id)


@dataclass(frozen=True, slots=True)
class MemoryWorkerSessionIdentityV1:
    """Detached description of one Worker-local session lifetime.

    Equality is descriptive only.  This record is not a bearer credential and
    does not authorize pinning, running, exposing, or closing Memory by itself.
    """

    session: MemorySessionIncarnationV1
    audience: MemoryWorkerAudienceV1

    def __post_init__(self) -> None:
        object.__setattr__(self, "session", _session(self.session))
        object.__setattr__(self, "audience", _audience(self.audience))

    @property
    def session_key(self) -> str:
        return self.session.session_key


class MemoryWorkerSessionCloseOutcomeV1(StrEnum):
    """Terminal result of a conditional exact-session close."""

    CLOSED = "closed"
    NOT_CURRENT = "not_current"


@dataclass(frozen=True, slots=True)
class MemoryWorkerSessionCloseReceiptV1:
    """Detached result bound to every dimension of the requested identity.

    ``CLOSED`` includes an idempotent replay of a previously completed close.
    ``NOT_CURRENT`` means the identity was neither current nor known retired;
    it never implies that another incarnation sharing the key was modified.
    """

    identity: MemoryWorkerSessionIdentityV1
    outcome: MemoryWorkerSessionCloseOutcomeV1

    def __post_init__(self) -> None:
        if type(self.identity) is not MemoryWorkerSessionIdentityV1:
            raise TypeError("identity must be a MemoryWorkerSessionIdentityV1")
        if type(self.outcome) is not MemoryWorkerSessionCloseOutcomeV1:
            raise TypeError("outcome must be a MemoryWorkerSessionCloseOutcomeV1")
        object.__setattr__(
            self,
            "identity",
            MemoryWorkerSessionIdentityV1(
                session=self.identity.session,
                audience=self.identity.audience,
            ),
        )


__all__ = [
    "MemoryWorkerSessionCloseOutcomeV1",
    "MemoryWorkerSessionCloseReceiptV1",
    "MemoryWorkerSessionIdentityV1",
]
