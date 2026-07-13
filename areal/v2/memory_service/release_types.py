# SPDX-License-Identifier: Apache-2.0

"""Immutable release values for the Memory Service."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256

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


def _release_commitment_bytes(
    manifest_sha256: str,
    release_graph_sha256: str,
) -> bytes:
    """Bind the caller's selection and the store-derived source graph."""

    return _canonical_json_bytes(
        {
            "manifest_sha256": manifest_sha256,
            "record_kind": "memory_release_commitment",
            "release_graph_sha256": release_graph_sha256,
            "schema_version": _SCHEMA_VERSION,
        }
    )


def _snapshot_revision_ids(value: object) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise TypeError("revision_ids must be a tuple")
    snapshot = tuple(
        _validate_string(item, "revision_ids") for item in tuple.__iter__(value)
    )
    if len(set(snapshot)) != len(snapshot):
        raise ValueError("revision_ids must not contain duplicates")
    return snapshot


def _validate_sha256(value: object, field_name: str) -> str:
    digest = _validate_string(value, field_name)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")
    return digest


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
    release_graph_sha256: str
    created_at: datetime

    def __post_init__(self) -> None:
        if type(self.manifest) is not ReleaseManifest:
            raise TypeError("manifest must be a ReleaseManifest")
        release_id = _validate_string(self.release_id, "release_id")
        content_hash = _validate_sha256(self.content_hash, "content_hash")
        release_graph_sha256 = _validate_sha256(
            self.release_graph_sha256,
            "release_graph_sha256",
        )
        created_at = _validate_aware_datetime(self.created_at, "created_at")
        object.__setattr__(self, "release_id", release_id)
        object.__setattr__(self, "content_hash", content_hash)
        object.__setattr__(self, "release_graph_sha256", release_graph_sha256)
        object.__setattr__(self, "created_at", created_at)
        expected_content_hash = sha256(self.commitment_bytes()).hexdigest()
        if content_hash != expected_content_hash:
            raise ValueError("content_hash does not match the release commitment")
        if release_id != f"rel_{content_hash[:24]}":
            raise ValueError("release_id does not match content_hash")

    def commitment_bytes(self) -> bytes:
        """Return the canonical release-identity commitment.

        The manifest identifies the requested selection.  The graph digest is
        independently derived by the release store from the immutable history
        it resolved; it is not supplied by the manifest caller.
        """

        manifest_sha256 = sha256(self.manifest.canonical_bytes()).hexdigest()
        return _release_commitment_bytes(
            manifest_sha256,
            self.release_graph_sha256,
        )
