# SPDX-License-Identifier: Apache-2.0

"""Release storage contracts and an in-memory reference implementation."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256
from threading import RLock
from typing import Protocol

from areal.v2.memory_service._atomic import _atomic_publish
from areal.v2.memory_service.errors import (
    MemoryServiceError,
    ReleaseConflictError,
    ReleaseNotFoundError,
)
from areal.v2.memory_service.history_store import MemoryHistoryStore
from areal.v2.memory_service.history_types import (
    CandidateProposal,
    MemoryCandidate,
    MemoryRevision,
    RevisionOperation,
    RevisionProposal,
)
from areal.v2.memory_service.release_types import (
    MemoryRelease,
    ReleaseManifest,
    _release_commitment_bytes,
)
from areal.v2.memory_service.types import (
    EvidenceEvent,
    EvidenceRecord,
    MemoryScope,
    _validate_string,
)

_RELEASE_GRAPH_SCHEMA_VERSION = 1


def _canonical_json_bytes(value: dict[str, object]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _commitment_error(message: str) -> ReleaseConflictError:
    return ReleaseConflictError(f"release graph integrity failure: {message}")


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

    def _validated_candidate_node(
        self,
        scope: MemoryScope,
        candidate_id: str,
    ) -> dict[str, object]:
        candidate = self._history_store.get_candidate(scope, candidate_id)
        if type(candidate) is not MemoryCandidate:
            raise _commitment_error("history returned a non-MemoryCandidate value")
        proposal = candidate.proposal
        if (
            type(proposal) is not CandidateProposal
            or type(proposal.scope) is not MemoryScope
            or proposal.scope != scope
            or type(proposal.evidence_ids) is not tuple
            or any(type(item) is not str for item in proposal.evidence_ids)
            or type(proposal.content) is not str
            or type(proposal.idempotency_key) is not str
        ):
            raise _commitment_error("candidate does not belong to the requested scope")
        if (
            type(candidate.candidate_id) is not str
            or type(candidate.content_hash) is not str
        ):
            raise _commitment_error("candidate returned malformed commitment fields")
        if candidate.candidate_id != candidate_id:
            raise _commitment_error("candidate address does not match the request")
        candidate_hash = sha256(proposal.canonical_bytes()).hexdigest()
        if (
            candidate.content_hash != candidate_hash
            or candidate.candidate_id != f"cand_{candidate_hash[:24]}"
        ):
            raise _commitment_error("candidate canonical commitment is invalid")

        evidence = self._history_store.get_candidate_evidence(scope, candidate_id)
        if type(evidence) is not tuple:
            raise _commitment_error("candidate evidence must be an exact tuple")
        records = tuple(tuple.__iter__(evidence))
        if tuple(record.evidence_id for record in records) != proposal.evidence_ids:
            raise _commitment_error(
                "candidate evidence addresses do not match proposal"
            )

        evidence_nodes: list[dict[str, str]] = []
        for expected_evidence_id, record in zip(
            proposal.evidence_ids,
            records,
            strict=True,
        ):
            if (
                type(record) is not EvidenceRecord
                or type(record.event) is not EvidenceEvent
            ):
                raise _commitment_error("history returned a non-EvidenceRecord value")
            if (
                type(record.evidence_id) is not str
                or type(record.content_hash) is not str
            ):
                raise _commitment_error("evidence returned malformed commitment fields")
            if (
                type(record.event.scope) is not MemoryScope
                or record.event.scope != scope
            ):
                raise _commitment_error("evidence does not belong to requested scope")
            if record.evidence_id != expected_evidence_id:
                raise _commitment_error("evidence address does not match proposal")
            evidence_hash = sha256(record.event.canonical_bytes()).hexdigest()
            if (
                record.content_hash != evidence_hash
                or record.evidence_id != f"evd_{evidence_hash[:24]}"
            ):
                raise _commitment_error("evidence canonical commitment is invalid")
            evidence_nodes.append(
                {
                    "evidence_id": record.evidence_id,
                    "evidence_sha256": evidence_hash,
                }
            )

        repeated_candidate_hash = sha256(proposal.canonical_bytes()).hexdigest()
        if (
            repeated_candidate_hash != candidate_hash
            or candidate.content_hash != candidate_hash
            or candidate.candidate_id != candidate_id
            or candidate.candidate_id != f"cand_{candidate_hash[:24]}"
        ):
            raise _commitment_error(
                "candidate changed while its evidence was being validated"
            )
        return {
            "candidate_id": candidate.candidate_id,
            "candidate_sha256": candidate_hash,
            "evidence": evidence_nodes,
        }

    def _validated_revision_node(
        self,
        scope: MemoryScope,
        requested_revision_id: str,
    ) -> tuple[MemoryRevision, dict[str, object]]:
        revision = self._history_store.get_revision(scope, requested_revision_id)
        if type(revision) is not MemoryRevision:
            raise _commitment_error("history returned a non-MemoryRevision value")
        proposal = revision.proposal
        if (
            type(proposal) is not RevisionProposal
            or type(proposal.scope) is not MemoryScope
            or proposal.scope != scope
            or type(proposal.candidate_id) is not str
            or type(proposal.operation) is not RevisionOperation
            or (
                proposal.parent_revision_id is not None
                and type(proposal.parent_revision_id) is not str
            )
            or type(proposal.idempotency_key) is not str
        ):
            raise _commitment_error("revision does not belong to requested scope")
        if (
            type(revision.revision_id) is not str
            or type(revision.content_hash) is not str
            or type(revision.memory_id) is not str
            or type(revision.generation) is not int
        ):
            raise _commitment_error("revision returned malformed commitment fields")
        if revision.revision_id != requested_revision_id:
            raise _commitment_error("revision address does not match the request")
        revision_hash = sha256(proposal.canonical_bytes()).hexdigest()
        if (
            revision.content_hash != revision_hash
            or revision.revision_id != f"rev_{revision_hash[:24]}"
        ):
            raise _commitment_error("revision canonical commitment is invalid")

        candidate_node = self._validated_candidate_node(
            scope,
            proposal.candidate_id,
        )
        repeated_revision_hash = sha256(proposal.canonical_bytes()).hexdigest()
        if (
            repeated_revision_hash != revision_hash
            or revision.content_hash != revision_hash
            or revision.revision_id != requested_revision_id
            or revision.revision_id != f"rev_{revision_hash[:24]}"
        ):
            raise _commitment_error(
                "revision changed while its source graph was being validated"
            )
        return revision, {
            "candidate": candidate_node,
            "generation": revision.generation,
            "memory_id": revision.memory_id,
            "revision_id": revision.revision_id,
            "revision_sha256": revision_hash,
        }

    def _derive_release_graph(
        self,
        manifest: ReleaseManifest,
    ) -> tuple[bytes, tuple[MemoryRevision, ...]]:
        """Resolve and validate every selected path without recursion."""

        selected_nodes: list[dict[str, object]] = []
        selected_revisions: list[MemoryRevision] = []
        resolved_nodes: dict[str, dict[str, object]] = {}

        try:
            for selected_revision_id in manifest.revision_ids:
                ancestry: list[dict[str, object]] = []
                visiting: set[str] = set()
                current_revision_id = selected_revision_id
                child: MemoryRevision | None = None
                selected: MemoryRevision | None = None

                while True:
                    if current_revision_id in visiting:
                        raise _commitment_error("revision ancestry contains a cycle")
                    visiting.add(current_revision_id)
                    revision, node = self._validated_revision_node(
                        manifest.scope,
                        current_revision_id,
                    )
                    previous_node = resolved_nodes.get(revision.revision_id)
                    if previous_node is not None and previous_node != node:
                        raise _commitment_error(
                            "one revision address resolved to different graph nodes"
                        )
                    resolved_nodes[revision.revision_id] = node
                    ancestry.append(node)
                    if selected is None:
                        selected = revision

                    if child is not None and (
                        child.memory_id != revision.memory_id
                        or child.generation != revision.generation + 1
                    ):
                        raise _commitment_error(
                            "revision lineage derived fields are invalid"
                        )

                    if revision.proposal.operation is RevisionOperation.ADD:
                        if (
                            revision.proposal.parent_revision_id is not None
                            or revision.generation != 0
                            or revision.memory_id != f"mem_{revision.content_hash[:24]}"
                        ):
                            raise _commitment_error(
                                "ADD revision derived fields are invalid"
                            )
                        break

                    parent_revision_id = revision.proposal.parent_revision_id
                    if type(parent_revision_id) is not str:
                        raise _commitment_error(
                            "non-ADD revision has no exact parent address"
                        )
                    child = revision
                    current_revision_id = parent_revision_id

                assert selected is not None
                selected_revisions.append(selected)
                selected_nodes.append(
                    {
                        "ancestry": ancestry,
                        "selected_revision_id": selected_revision_id,
                    }
                )
            selected_memory_ids: set[str] = set()
            for revision in selected_revisions:
                if revision.memory_id in selected_memory_ids:
                    raise ReleaseConflictError(
                        "release contains more than one revision for memory_id "
                        f"{revision.memory_id!r}"
                    )
                selected_memory_ids.add(revision.memory_id)
        except MemoryServiceError:
            raise
        except Exception as error:
            raise _commitment_error(
                "history lookup did not satisfy its contract"
            ) from error

        return (
            _canonical_json_bytes(
                {
                    "ancestry_order": "selected_to_add_root",
                    "record_kind": "memory_release_graph",
                    "schema_version": _RELEASE_GRAPH_SCHEMA_VERSION,
                    "selected_revisions": selected_nodes,
                }
            ),
            tuple(selected_revisions),
        )

    def append_release(
        self, manifest: ReleaseManifest, *, idempotency_key: str
    ) -> MemoryRelease:
        """Persist a release after validating all referenced revisions."""

        if type(manifest) is not ReleaseManifest:
            raise TypeError("manifest must be a ReleaseManifest")
        idempotency_key = _validate_string(idempotency_key, "idempotency_key")
        canonical_bytes = manifest.canonical_bytes()
        manifest_sha256 = sha256(canonical_bytes).hexdigest()
        idempotency_index = (manifest.scope, idempotency_key)

        with self._lock:
            existing = self._release_by_idempotency.get(idempotency_index)
            if existing is not None:
                if existing.manifest.canonical_bytes() != canonical_bytes:
                    raise ReleaseConflictError(
                        "scoped release idempotency key already refers to different "
                        "content"
                    )

        graph_bytes, _ = self._derive_release_graph(manifest)
        repeated_graph_bytes, _ = self._derive_release_graph(manifest)
        if repeated_graph_bytes != graph_bytes:
            raise _commitment_error("history changed during final graph recheck")
        release_graph_sha256 = sha256(graph_bytes).hexdigest()
        commitment_bytes = _release_commitment_bytes(
            manifest_sha256,
            release_graph_sha256,
        )
        content_hash = sha256(commitment_bytes).hexdigest()
        release_id = f"rel_{content_hash[:24]}"
        release_index = (manifest.scope, release_id)

        with self._lock:
            existing = self._release_by_idempotency.get(idempotency_index)
            if existing is not None:
                if (
                    existing.manifest.canonical_bytes() == canonical_bytes
                    and existing.release_graph_sha256 == release_graph_sha256
                    and existing.content_hash == content_hash
                ):
                    return existing
                raise ReleaseConflictError(
                    "scoped release idempotency key already refers to different content"
                )

            existing = self._release_by_id.get(release_index)
            if existing is not None:
                if (
                    existing.manifest.canonical_bytes() != canonical_bytes
                    or existing.release_graph_sha256 != release_graph_sha256
                    or existing.content_hash != content_hash
                ):
                    raise ReleaseConflictError(
                        f"release ID collision for {release_id!r}"
                    )
                _atomic_publish(
                    mapping_writes=(
                        (self._release_by_idempotency, idempotency_index, existing),
                    ),
                )
                return existing

            release = MemoryRelease(
                release_id=release_id,
                manifest=manifest,
                content_hash=content_hash,
                release_graph_sha256=release_graph_sha256,
                created_at=datetime.now(UTC),
            )
            _atomic_publish(
                mapping_writes=(
                    (self._release_by_id, release_index, release),
                    (self._release_by_idempotency, idempotency_index, release),
                ),
                sequence_appends=((self._releases_by_scope, manifest.scope, release),),
            )
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
        graph_bytes, _ = self._derive_release_graph(release.manifest)
        repeated_graph_bytes, revisions = self._derive_release_graph(release.manifest)
        if repeated_graph_bytes != graph_bytes:
            raise _commitment_error("history changed during final graph recheck")
        release_graph_sha256 = sha256(graph_bytes).hexdigest()
        manifest_sha256 = sha256(release.manifest.canonical_bytes()).hexdigest()
        content_hash = sha256(
            _release_commitment_bytes(manifest_sha256, release_graph_sha256)
        ).hexdigest()
        if (
            release.release_graph_sha256 != release_graph_sha256
            or release.content_hash != content_hash
            or release.release_id != f"rel_{content_hash[:24]}"
        ):
            raise _commitment_error(
                "resolved graph does not match the committed release identity"
            )
        return revisions

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
