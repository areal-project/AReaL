# SPDX-License-Identifier: Apache-2.0

"""Immutable release values for the Memory Service."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime

from areal.v2.memory_service.types import (
    MemoryScope,
    _validate_aware_datetime,
    _validate_string,
)

_SCHEMA_VERSION = 1


def _canonical_json_bytes(value: dict[str, object]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _snapshot_revision_ids(value: object) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise TypeError("revision_ids must be a tuple")
    snapshot = tuple(
        _validate_string(item, "revision_ids") for item in tuple.__iter__(value)
    )
    if len(set(snapshot)) != len(snapshot):
        raise ValueError("revision_ids must not contain duplicates")
    return snapshot


@dataclass(frozen=True, slots=True)
class ReleaseManifest:
    scope: MemoryScope
    revision_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.scope) is not MemoryScope:
            raise TypeError("scope must be a MemoryScope")
        revision_ids = _snapshot_revision_ids(self.revision_ids)
        object.__setattr__(self, "revision_ids", revision_ids)

    def canonical_bytes(self) -> bytes:
        value = {
            "schema_version": _SCHEMA_VERSION,
            "scope": {
                "tenant_id": self.scope.tenant_id,
                "namespace": self.scope.namespace,
                "subject_id": self.scope.subject_id,
            },
            "revision_ids": list(self.revision_ids),
        }
        return _canonical_json_bytes(value)


@dataclass(frozen=True, slots=True)
class MemoryRelease:
    release_id: str
    manifest: ReleaseManifest
    content_hash: str
    created_at: datetime

    def __post_init__(self) -> None:
        if type(self.manifest) is not ReleaseManifest:
            raise TypeError("manifest must be a ReleaseManifest")
        release_id = _validate_string(self.release_id, "release_id")
        content_hash = _validate_string(self.content_hash, "content_hash")
        created_at = _validate_aware_datetime(self.created_at, "created_at")
        object.__setattr__(self, "release_id", release_id)
        object.__setattr__(self, "content_hash", content_hash)
        object.__setattr__(self, "created_at", created_at)
