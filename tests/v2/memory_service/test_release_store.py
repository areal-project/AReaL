# SPDX-License-Identifier: Apache-2.0

"""Tests for immutable Memory Service release manifests."""

from __future__ import annotations

import inspect
import re
from datetime import UTC, datetime
from hashlib import sha256

import pytest

from areal.v2.memory_service import release_store as release_store_module
from areal.v2.memory_service.errors import (
    MemoryServiceError,
    ReleaseConflictError,
    ReleaseNotFoundError,
    RevisionNotFoundError,
)
from areal.v2.memory_service.history_store import InMemoryMemoryHistoryStore
from areal.v2.memory_service.history_types import (
    CandidateProposal,
    MemoryRevision,
    RevisionOperation,
    RevisionProposal,
)
from areal.v2.memory_service.release_store import (
    InMemoryMemoryReleaseStore,
    MemoryReleaseStore,
)
from areal.v2.memory_service.release_types import ReleaseManifest
from areal.v2.memory_service.store import InMemoryEvidenceStore
from areal.v2.memory_service.types import (
    EvidenceEvent,
    EvidenceKind,
    EvidenceRecord,
    MemoryScope,
)


class MutableHashStr(str):
    """A string subclass that exposes unsafe hashing before snapshotting."""

    hash_calls: int

    def __new__(cls, value: str) -> MutableHashStr:
        instance = str.__new__(cls, value)
        instance.hash_calls = 0
        return instance

    def __hash__(self) -> int:
        self.hash_calls += 1
        return str.__hash__(self) + self.hash_calls


class MemoryScopeSubclass(MemoryScope):
    pass


class ReleaseManifestSubclass(ReleaseManifest):
    pass


class NoLookupHistory:
    """History stub that fails if an empty release performs member lookup."""

    def __init__(self) -> None:
        self.lookup_count = 0

    def get_revision(self, scope: MemoryScope, revision_id: str) -> MemoryRevision:
        self.lookup_count += 1
        raise AssertionError("release unexpectedly looked up a revision")


class StubHash:
    """Return one test-controlled hexadecimal digest."""

    def __init__(self, digest: str) -> None:
        self._digest = digest

    def hexdigest(self) -> str:
        return self._digest


UTC_INSTANT = datetime(2026, 7, 7, 4, 5, 6, tzinfo=UTC)
LONE_SURROGATE = "\ud800"


def make_scope(**overrides: str) -> MemoryScope:
    values = {
        "tenant_id": "tenant-1",
        "namespace": "assistant-memory",
        "subject_id": "user-1",
    }
    values.update(overrides)
    return MemoryScope(**values)


def make_event(
    *,
    scope: MemoryScope,
    sequence_no: int,
    payload: str,
    idempotency_key: str,
) -> EvidenceEvent:
    return EvidenceEvent(
        scope=scope,
        session_id="session-1",
        run_id="run-1",
        sequence_no=sequence_no,
        kind=EvidenceKind.USER_MESSAGE,
        payload=payload,
        observed_at=UTC_INSTANT,
        idempotency_key=idempotency_key,
    )


def append_candidate(
    history: InMemoryMemoryHistoryStore,
    *,
    scope: MemoryScope,
    evidence_ids: tuple[str, ...],
    content: str,
    idempotency_key: str,
):
    return history.append_candidate(
        CandidateProposal(
            scope=scope,
            content=content,
            evidence_ids=evidence_ids,
            idempotency_key=idempotency_key,
        )
    )


def append_revision(
    history: InMemoryMemoryHistoryStore,
    *,
    scope: MemoryScope,
    candidate_id: str,
    operation: RevisionOperation,
    parent_revision_id: str | None,
    idempotency_key: str,
) -> MemoryRevision:
    return history.append_revision(
        RevisionProposal(
            scope=scope,
            candidate_id=candidate_id,
            operation=operation,
            parent_revision_id=parent_revision_id,
            idempotency_key=idempotency_key,
        )
    )


def seeded_history(
    *,
    scope: MemoryScope | None = None,
) -> tuple[
    InMemoryMemoryHistoryStore,
    tuple[EvidenceRecord, EvidenceRecord],
    tuple[MemoryRevision, MemoryRevision, MemoryRevision],
]:
    scope = scope or make_scope()
    evidence_store = InMemoryEvidenceStore()
    first_evidence = evidence_store.append(
        make_event(
            scope=scope,
            sequence_no=0,
            payload="first observation",
            idempotency_key="evidence-1",
        )
    )
    second_evidence = evidence_store.append(
        make_event(
            scope=scope,
            sequence_no=1,
            payload="second observation",
            idempotency_key="evidence-2",
        )
    )
    history = InMemoryMemoryHistoryStore(evidence_store)
    root_candidate = append_candidate(
        history,
        scope=scope,
        evidence_ids=(second_evidence.evidence_id, first_evidence.evidence_id),
        content="root memory",
        idempotency_key="candidate-root",
    )
    root = append_revision(
        history,
        scope=scope,
        candidate_id=root_candidate.candidate_id,
        operation=RevisionOperation.ADD,
        parent_revision_id=None,
        idempotency_key="revision-root",
    )
    child_candidate = append_candidate(
        history,
        scope=scope,
        evidence_ids=(first_evidence.evidence_id,),
        content="refined memory",
        idempotency_key="candidate-child",
    )
    child = append_revision(
        history,
        scope=scope,
        candidate_id=child_candidate.candidate_id,
        operation=RevisionOperation.REFINE,
        parent_revision_id=root.revision_id,
        idempotency_key="revision-child",
    )
    other_candidate = append_candidate(
        history,
        scope=scope,
        evidence_ids=(second_evidence.evidence_id,),
        content="independent memory",
        idempotency_key="candidate-other",
    )
    other = append_revision(
        history,
        scope=scope,
        candidate_id=other_candidate.candidate_id,
        operation=RevisionOperation.ADD,
        parent_revision_id=None,
        idempotency_key="revision-other",
    )
    return history, (first_evidence, second_evidence), (root, child, other)


def seeded_sibling_revisions() -> tuple[
    InMemoryMemoryHistoryStore,
    MemoryRevision,
    MemoryRevision,
]:
    history, _, (root, _, _) = seeded_history()
    scope = root.proposal.scope
    root_evidence_id = history.get_candidate(
        scope, root.proposal.candidate_id
    ).proposal.evidence_ids[0]
    siblings = []
    for side, operation in (
        ("left", RevisionOperation.REFINE),
        ("right", RevisionOperation.CONTRADICT),
    ):
        candidate = append_candidate(
            history,
            scope=scope,
            evidence_ids=(root_evidence_id,),
            content=f"{side} sibling",
            idempotency_key=f"candidate-{side}",
        )
        siblings.append(
            append_revision(
                history,
                scope=scope,
                candidate_id=candidate.candidate_id,
                operation=operation,
                parent_revision_id=root.revision_id,
                idempotency_key=f"revision-{side}",
            )
        )
    left, right = siblings
    assert left.memory_id == right.memory_id == root.memory_id
    return history, left, right


def assert_release_indexes_empty(store: InMemoryMemoryReleaseStore) -> None:
    assert store._release_by_id == {}
    assert store._release_by_idempotency == {}
    assert store._releases_by_scope == {}


def test_release_contract_exposes_only_immutable_manifest_api() -> None:
    expected = {
        "append_release",
        "get_release",
        "get_release_revisions",
        "list_releases",
    }
    protocol_methods = {
        name
        for name, value in inspect.getmembers(MemoryReleaseStore)
        if not name.startswith("_") and callable(value)
    }
    implementation_methods = {
        name
        for name, value in inspect.getmembers(InMemoryMemoryReleaseStore)
        if not name.startswith("_") and callable(value)
    }

    assert protocol_methods == expected
    assert implementation_methods == expected


def test_release_errors_share_memory_service_base() -> None:
    assert issubclass(ReleaseNotFoundError, MemoryServiceError)
    assert issubclass(ReleaseConflictError, MemoryServiceError)


def test_release_resolves_revisions_in_manifest_order_with_evidence_provenance() -> (
    None
):
    history, (first_evidence, second_evidence), (root, child, other) = seeded_history()
    store = InMemoryMemoryReleaseStore(history)
    manifest = ReleaseManifest(
        scope=make_scope(),
        revision_ids=(other.revision_id, child.revision_id),
    )

    release = store.append_release(manifest, idempotency_key="release-1")

    assert release.manifest is manifest
    assert store.get_release_revisions(manifest.scope, release.release_id) == (
        other,
        child,
    )
    assert history.get_candidate_evidence(
        manifest.scope, root.proposal.candidate_id
    ) == (second_evidence, first_evidence)
    assert history.get_candidate_evidence(
        manifest.scope, child.proposal.candidate_id
    ) == (first_evidence,)


def test_empty_manifest_is_a_stable_memory_off_release_without_lookup() -> None:
    history = NoLookupHistory()
    store = InMemoryMemoryReleaseStore(history)  # type: ignore[arg-type]
    manifest = ReleaseManifest(make_scope(), ())
    expected_hash = sha256(manifest.canonical_bytes()).hexdigest()

    release = store.append_release(manifest, idempotency_key="memory-off")

    assert history.lookup_count == 0
    assert release.release_id == f"rel_{expected_hash[:24]}"
    assert release.content_hash == expected_hash
    assert store.get_release_revisions(manifest.scope, release.release_id) == ()
    assert history.lookup_count == 0
    assert store.list_releases(manifest.scope) == (release,)


def test_append_release_requires_exact_manifest_before_any_write_or_lookup() -> None:
    history = NoLookupHistory()
    store = InMemoryMemoryReleaseStore(history)  # type: ignore[arg-type]
    subclass = ReleaseManifestSubclass(make_scope(), ())

    with pytest.raises(TypeError, match="manifest must be a ReleaseManifest"):
        store.append_release(subclass, idempotency_key="release-1")
    with pytest.raises(TypeError, match="manifest must be a ReleaseManifest"):
        store.append_release(object(), idempotency_key="release-2")  # type: ignore[arg-type]

    assert history.lookup_count == 0
    assert_release_indexes_empty(store)


@pytest.mark.parametrize("idempotency_key", [None, 7, b"key"])
def test_append_release_rejects_non_string_idempotency_key(
    idempotency_key: object,
) -> None:
    history = NoLookupHistory()
    store = InMemoryMemoryReleaseStore(history)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="idempotency_key"):
        store.append_release(
            ReleaseManifest(make_scope(), ()),
            idempotency_key=idempotency_key,  # type: ignore[arg-type]
        )

    assert history.lookup_count == 0
    assert_release_indexes_empty(store)


@pytest.mark.parametrize("idempotency_key", ["", " \t\n", LONE_SURROGATE])
def test_append_release_rejects_blank_or_non_utf8_idempotency_key(
    idempotency_key: str,
) -> None:
    history = NoLookupHistory()
    store = InMemoryMemoryReleaseStore(history)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="idempotency_key"):
        store.append_release(
            ReleaseManifest(make_scope(), ()),
            idempotency_key=idempotency_key,
        )

    assert history.lookup_count == 0
    assert_release_indexes_empty(store)


def test_append_release_snapshots_idempotency_key_before_indexing() -> None:
    history = NoLookupHistory()
    store = InMemoryMemoryReleaseStore(history)  # type: ignore[arg-type]
    key = MutableHashStr("release-1")

    release = store.append_release(
        ReleaseManifest(make_scope(), ()), idempotency_key=key
    )

    assert key.hash_calls == 0
    stored_scope, stored_key = next(iter(store._release_by_idempotency))
    assert stored_scope == make_scope()
    assert type(stored_key) is str
    assert stored_key == "release-1"
    assert store._release_by_idempotency[(make_scope(), "release-1")] is release


def test_missing_revision_leaves_all_release_indexes_empty() -> None:
    history = InMemoryMemoryHistoryStore(InMemoryEvidenceStore())
    store = InMemoryMemoryReleaseStore(history)
    manifest = ReleaseManifest(make_scope(), ("rev_missing",))

    with pytest.raises(RevisionNotFoundError, match="rev_missing"):
        store.append_release(manifest, idempotency_key="release-1")

    assert_release_indexes_empty(store)


def test_foreign_revision_is_indistinguishable_from_missing() -> None:
    foreign_scope = make_scope(subject_id="user-2")
    foreign_history, _, (foreign_revision, _, _) = seeded_history(scope=foreign_scope)
    foreign_store = InMemoryMemoryReleaseStore(foreign_history)
    missing_store = InMemoryMemoryReleaseStore(
        InMemoryMemoryHistoryStore(InMemoryEvidenceStore())
    )
    manifest = ReleaseManifest(make_scope(), (foreign_revision.revision_id,))

    with pytest.raises(RevisionNotFoundError) as foreign_error:
        foreign_store.append_release(manifest, idempotency_key="release-1")
    with pytest.raises(RevisionNotFoundError) as missing_error:
        missing_store.append_release(manifest, idempotency_key="release-1")

    assert type(foreign_error.value) is RevisionNotFoundError
    assert str(foreign_error.value) == str(missing_error.value)
    assert_release_indexes_empty(foreign_store)


def test_sibling_revisions_conflict_in_one_manifest_but_release_separately() -> None:
    history, left, right = seeded_sibling_revisions()
    scope = left.proposal.scope
    store = InMemoryMemoryReleaseStore(history)

    with pytest.raises(ReleaseConflictError, match="memory_id"):
        store.append_release(
            ReleaseManifest(scope, (left.revision_id, right.revision_id)),
            idempotency_key="release-both",
        )
    assert_release_indexes_empty(store)

    left_release = store.append_release(
        ReleaseManifest(scope, (left.revision_id,)), idempotency_key="release-left"
    )
    right_release = store.append_release(
        ReleaseManifest(scope, (right.revision_id,)), idempotency_key="release-right"
    )
    assert set(store.list_releases(scope)) == {left_release, right_release}


def test_missing_member_precedes_duplicate_memory_conflict() -> None:
    history, left, right = seeded_sibling_revisions()
    store = InMemoryMemoryReleaseStore(history)
    manifest = ReleaseManifest(
        left.proposal.scope,
        (left.revision_id, right.revision_id, "rev_missing"),
    )

    with pytest.raises(RevisionNotFoundError, match="rev_missing"):
        store.append_release(manifest, idempotency_key="release-invalid")

    assert_release_indexes_empty(store)


def test_same_key_retry_returns_original_release() -> None:
    history, _, (root, _, _) = seeded_history()
    store = InMemoryMemoryReleaseStore(history)
    manifest = ReleaseManifest(make_scope(), (root.revision_id,))

    first = store.append_release(manifest, idempotency_key="release-1")
    retry = store.append_release(
        ReleaseManifest(manifest.scope, manifest.revision_ids),
        idempotency_key="release-1",
    )

    assert retry is first
    assert store.list_releases(manifest.scope) == (first,)
    assert len(store._release_by_id) == 1
    assert len(store._release_by_idempotency) == 1


def test_new_key_aliases_existing_release_without_duplicate_list_entry() -> None:
    history, _, (root, _, _) = seeded_history()
    store = InMemoryMemoryReleaseStore(history)
    manifest = ReleaseManifest(make_scope(), (root.revision_id,))

    first = store.append_release(manifest, idempotency_key="release-1")
    alias = store.append_release(manifest, idempotency_key="release-2")

    assert alias is first
    assert store.list_releases(manifest.scope) == (first,)
    assert len(store._release_by_id) == 1
    assert len(store._release_by_idempotency) == 2


def test_empty_control_aliases_across_keys_without_history_lookup() -> None:
    history = NoLookupHistory()
    store = InMemoryMemoryReleaseStore(history)  # type: ignore[arg-type]
    manifest = ReleaseManifest(make_scope(), ())

    first = store.append_release(manifest, idempotency_key="memory-off-1")
    alias = store.append_release(manifest, idempotency_key="memory-off-2")

    assert alias is first
    assert history.lookup_count == 0
    assert store.list_releases(manifest.scope) == (first,)
    assert len(store._release_by_idempotency) == 2


def test_changed_same_key_conflicts_before_missing_member_lookup() -> None:
    history = NoLookupHistory()
    store = InMemoryMemoryReleaseStore(history)  # type: ignore[arg-type]
    scope = make_scope()
    first = store.append_release(
        ReleaseManifest(scope, ()), idempotency_key="release-1"
    )

    with pytest.raises(ReleaseConflictError, match="idempotency"):
        store.append_release(
            ReleaseManifest(scope, ("rev_missing",)),
            idempotency_key="release-1",
        )

    assert history.lookup_count == 0
    assert store.list_releases(scope) == (first,)
    assert len(store._release_by_id) == 1
    assert len(store._release_by_idempotency) == 1


def test_release_hash_and_identifier_depend_only_on_manifest() -> None:
    history, _, (root, _, _) = seeded_history()
    manifest = ReleaseManifest(make_scope(), (root.revision_id,))
    expected_hash = sha256(manifest.canonical_bytes()).hexdigest()
    first_store = InMemoryMemoryReleaseStore(history)
    second_store = InMemoryMemoryReleaseStore(history)

    first = first_store.append_release(manifest, idempotency_key="first-key")
    second = second_store.append_release(manifest, idempotency_key="other-key")

    assert first.content_hash == second.content_hash == expected_hash
    assert first.release_id == second.release_id == f"rel_{expected_hash[:24]}"
    assert re.fullmatch(r"rel_[0-9a-f]{24}", first.release_id)


def test_append_release_uses_module_level_sha256(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = ReleaseManifest(make_scope(), ())
    digest = "a" * 64
    observed_bytes: list[bytes] = []

    def fake_sha256(canonical_bytes: bytes) -> StubHash:
        observed_bytes.append(canonical_bytes)
        return StubHash(digest)

    monkeypatch.setattr(release_store_module, "sha256", fake_sha256)
    store = InMemoryMemoryReleaseStore(NoLookupHistory())  # type: ignore[arg-type]

    release = store.append_release(manifest, idempotency_key="release-1")

    assert observed_bytes == [manifest.canonical_bytes()]
    assert release.content_hash == digest
    assert release.release_id == f"rel_{digest[:24]}"


def test_get_release_snapshots_identifier_subclass_before_lookup() -> None:
    history = NoLookupHistory()
    store = InMemoryMemoryReleaseStore(history)  # type: ignore[arg-type]
    release = store.append_release(
        ReleaseManifest(make_scope(), ()), idempotency_key="release-1"
    )
    release_id = MutableHashStr(release.release_id)

    assert store.get_release(make_scope(), release_id) is release
    assert release_id.hash_calls == 0


@pytest.mark.parametrize("release_id", ["", " \t\n"])
def test_blank_release_identifier_is_reported_as_not_found(release_id: str) -> None:
    store = InMemoryMemoryReleaseStore(NoLookupHistory())  # type: ignore[arg-type]

    with pytest.raises(ReleaseNotFoundError):
        store.get_release(make_scope(), release_id)


def test_release_queries_require_exact_scope() -> None:
    store = InMemoryMemoryReleaseStore(NoLookupHistory())  # type: ignore[arg-type]
    scope = MemoryScopeSubclass("tenant-1", "assistant-memory", "user-1")

    with pytest.raises(TypeError, match="scope must be a MemoryScope"):
        store.get_release(scope, "rel_missing")
    with pytest.raises(TypeError, match="scope must be a MemoryScope"):
        store.get_release_revisions(scope, "rel_missing")
    with pytest.raises(TypeError, match="scope must be a MemoryScope"):
        store.list_releases(scope)


def test_foreign_release_is_indistinguishable_from_missing() -> None:
    history = NoLookupHistory()
    store = InMemoryMemoryReleaseStore(history)  # type: ignore[arg-type]
    foreign_scope = make_scope(subject_id="user-2")
    release = store.append_release(
        ReleaseManifest(foreign_scope, ()), idempotency_key="release-1"
    )

    with pytest.raises(ReleaseNotFoundError) as foreign_error:
        store.get_release(make_scope(), release.release_id)
    with pytest.raises(ReleaseNotFoundError) as missing_error:
        InMemoryMemoryReleaseStore(history).get_release(
            make_scope(), release.release_id
        )

    assert type(foreign_error.value) is ReleaseNotFoundError
    assert str(foreign_error.value) == str(missing_error.value)


def test_list_releases_is_sorted_and_previous_tuple_stays_stable() -> None:
    history, _, (root, child, other) = seeded_history()
    store = InMemoryMemoryReleaseStore(history)
    scope = make_scope()
    first = store.append_release(
        ReleaseManifest(scope, (root.revision_id,)), idempotency_key="release-root"
    )
    second = store.append_release(
        ReleaseManifest(scope, (other.revision_id,)), idempotency_key="release-other"
    )
    snapshot = store.list_releases(scope)

    third = store.append_release(
        ReleaseManifest(scope, (child.revision_id,)), idempotency_key="release-child"
    )

    assert snapshot == tuple(sorted((first, second), key=lambda item: item.release_id))
    assert third not in snapshot
    assert store.list_releases(scope) == tuple(
        sorted((first, second, third), key=lambda item: item.release_id)
    )
