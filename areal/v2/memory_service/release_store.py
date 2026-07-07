# SPDX-License-Identifier: Apache-2.0

"""Release storage contracts and an in-memory reference implementation."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from threading import RLock
from typing import Protocol

from areal.v2.memory_service.errors import (
    ReleaseConflictError,
    ReleaseNotFoundError,
)
from areal.v2.memory_service.history_store import MemoryHistoryStore
from areal.v2.memory_service.history_types import MemoryRevision
from areal.v2.memory_service.release_types import MemoryRelease, ReleaseManifest
from areal.v2.memory_service.types import MemoryScope, _validate_string


class MemoryReleaseStore(Protocol):
    """Storage contract for immutable release manifests."""

    def append_release(
        self, manifest: ReleaseManifest, *, idempotency_key: str
    ) -> MemoryRelease:
        """Validate and persist one immutable release manifest."""

        ...

    def get_release(self, scope: MemoryScope, release_id: str) -> MemoryRelease:
        """Return one release from the requested scope."""

        ...

    def get_release_revisions(
        self, scope: MemoryScope, release_id: str
    ) -> tuple[MemoryRevision, ...]:
        """Resolve release members in manifest order."""

        ...

    def list_releases(self, scope: MemoryScope) -> tuple[MemoryRelease, ...]:
        """Return releases in stable identifier order."""

        ...


class InMemoryMemoryReleaseStore:
    """Lock-protected process-local immutable release storage."""

    def __init__(self, history_store: MemoryHistoryStore) -> None:
        self._history_store = history_store
        self._lock = RLock()
        self._release_by_id: dict[tuple[MemoryScope, str], MemoryRelease] = {}
        self._release_by_idempotency: dict[tuple[MemoryScope, str], MemoryRelease] = {}
        self._releases_by_scope: dict[MemoryScope, list[MemoryRelease]] = {}

    def append_release(
        self, manifest: ReleaseManifest, *, idempotency_key: str
    ) -> MemoryRelease:
        """Persist a release after validating all referenced revisions."""

        if type(manifest) is not ReleaseManifest:
            raise TypeError("manifest must be a ReleaseManifest")
        idempotency_key = _validate_string(idempotency_key, "idempotency_key")
        canonical_bytes = manifest.canonical_bytes()
        content_hash = sha256(canonical_bytes).hexdigest()
        release_id = f"rel_{content_hash[:24]}"
        release_index = (manifest.scope, release_id)
        idempotency_index = (manifest.scope, idempotency_key)

        with self._lock:
            existing = self._release_by_idempotency.get(idempotency_index)
            if existing is not None:
                if existing.manifest.canonical_bytes() == canonical_bytes:
                    return existing
                raise ReleaseConflictError(
                    "scoped release idempotency key already refers to different content"
                )

        revisions = tuple(
            self._history_store.get_revision(manifest.scope, revision_id)
            for revision_id in manifest.revision_ids
        )
        memory_ids: set[str] = set()
        for revision in revisions:
            if revision.memory_id in memory_ids:
                raise ReleaseConflictError(
                    f"release contains more than one revision for memory_id "
                    f"{revision.memory_id!r}"
                )
            memory_ids.add(revision.memory_id)

        with self._lock:
            existing = self._release_by_idempotency.get(idempotency_index)
            if existing is not None:
                if existing.manifest.canonical_bytes() == canonical_bytes:
                    return existing
                raise ReleaseConflictError(
                    "scoped release idempotency key already refers to different content"
                )

            existing = self._release_by_id.get(release_index)
            if existing is not None:
                if existing.manifest.canonical_bytes() != canonical_bytes:
                    raise ReleaseConflictError(
                        f"release ID collision for {release_id!r}"
                    )
                self._release_by_idempotency[idempotency_index] = existing
                return existing

            release = MemoryRelease(
                release_id=release_id,
                manifest=manifest,
                content_hash=content_hash,
                created_at=datetime.now(UTC),
            )
            self._release_by_id[release_index] = release
            self._release_by_idempotency[idempotency_index] = release
            self._releases_by_scope.setdefault(manifest.scope, []).append(release)
            return release

    def get_release(self, scope: MemoryScope, release_id: str) -> MemoryRelease:
        """Return a release only when it belongs to the requested scope."""

        if type(scope) is not MemoryScope:
            raise TypeError("scope must be a MemoryScope")
        release_id = _validate_string(release_id, "release_id", allow_blank=True)
        with self._lock:
            release = self._release_by_id.get((scope, release_id))
            if release is None:
                raise ReleaseNotFoundError(f"release {release_id!r} was not found")
            return release

    def get_release_revisions(
        self, scope: MemoryScope, release_id: str
    ) -> tuple[MemoryRevision, ...]:
        """Resolve a release's revisions in manifest order."""

        release = self.get_release(scope, release_id)
        return tuple(
            self._history_store.get_revision(scope, revision_id)
            for revision_id in release.manifest.revision_ids
        )

    def list_releases(self, scope: MemoryScope) -> tuple[MemoryRelease, ...]:
        """Return a stable release snapshot for the requested scope."""

        if type(scope) is not MemoryScope:
            raise TypeError("scope must be a MemoryScope")
        with self._lock:
            return tuple(
                sorted(
                    self._releases_by_scope.get(scope, ()),
                    key=lambda item: item.release_id,
                )
            )
