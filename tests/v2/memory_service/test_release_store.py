# SPDX-License-Identifier: Apache-2.0

"""Tests for immutable Memory Service release manifests."""

from __future__ import annotations

import faulthandler
import inspect
import json
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256
from threading import Barrier, Event, RLock
from time import monotonic, sleep
from typing import TypeVar

import pytest

import areal.v2.memory_service.release_store as release_store_module
import areal.v2.memory_service.release_types as release_types_module
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
from areal.v2.memory_service.release_types import (
    MemoryRelease,
    ReleaseManifest,
    _release_commitment_bytes,
)
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


class YieldingMissingDict(dict[object, object]):
    """Yield after a missing lookup to expose an unprotected check/write race."""

    def get(self, key: object, default: object = None) -> object:
        value = super().get(key, default)
        if value is default:
            sleep(0.02)
        return value


class StubHash:
    """Return one test-controlled hexadecimal digest."""

    def __init__(self, digest: str) -> None:
        self._digest = digest

    def hexdigest(self) -> str:
        return self._digest


UTC_INSTANT = datetime(2026, 7, 7, 4, 5, 6, tzinfo=UTC)
LONE_SURROGATE = "\ud800"
RACE_SIZE = 16
RACE_TIMEOUT_SECONDS = 10.0
HARD_RACE_TIMEOUT_SECONDS = 15.0
T = TypeVar("T")


def run_race(
    items: tuple[T, ...], operation: Callable[[T], object]
) -> tuple[object, ...]:
    """Release callers together and apply one deadline to result collection."""

    barrier = Barrier(len(items), timeout=RACE_TIMEOUT_SECONDS)

    def worker(item: T) -> object:
        barrier.wait()
        try:
            return operation(item)
        except MemoryServiceError as error:
            return error

    faulthandler.dump_traceback_later(HARD_RACE_TIMEOUT_SECONDS, exit=True)
    try:
        with ThreadPoolExecutor(max_workers=len(items)) as executor:
            futures = tuple(executor.submit(worker, item) for item in items)
            deadline = monotonic() + RACE_TIMEOUT_SECONDS
            return tuple(
                future.result(timeout=max(0.0, deadline - monotonic()))
                for future in futures
            )
    finally:
        faulthandler.cancel_dump_traceback_later()


def install_digest_map(
    monkeypatch: pytest.MonkeyPatch,
    digest_by_bytes: dict[bytes, str],
) -> None:
    """Install deterministic release digests for pre-seeded manifests."""

    def fake_sha256(canonical_bytes: bytes) -> StubHash:
        return StubHash(
            digest_by_bytes.get(canonical_bytes, sha256(canonical_bytes).hexdigest())
        )

    monkeypatch.setattr(release_store_module, "sha256", fake_sha256)
    monkeypatch.setattr(release_types_module, "sha256", fake_sha256)


def release_commitment_for(
    store: InMemoryMemoryReleaseStore,
    manifest: ReleaseManifest,
) -> bytes:
    graph_bytes, _ = store._derive_release_graph(manifest)
    return _release_commitment_bytes(
        sha256(manifest.canonical_bytes()).hexdigest(),
        sha256(graph_bytes).hexdigest(),
    )


def test_run_race_waits_for_workers_before_cancelling_watchdog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[object, ...]] = []

    def record_arm(timeout: float, *, exit: bool) -> None:
        events.append(("arm", timeout, exit))

    def record_cancel() -> None:
        events.append(("cancel",))

    monkeypatch.setattr(faulthandler, "dump_traceback_later", record_arm)
    monkeypatch.setattr(faulthandler, "cancel_dump_traceback_later", record_cancel)

    assert run_race(tuple(range(RACE_SIZE)), lambda item: item) == tuple(
        range(RACE_SIZE)
    )
    assert events == [
        ("arm", HARD_RACE_TIMEOUT_SECONDS, True),
        ("cancel",),
    ]

    real_executor = ThreadPoolExecutor
    slow_workers_ready = Event()
    ready_count = 0
    ready_lock = RLock()
    shutdown_started = Event()
    shutdown_finished = Event()
    finished: set[int] = set()
    finished_lock = RLock()

    class ObservedExecutor(real_executor):
        def shutdown(
            self,
            wait: bool = True,
            *,
            cancel_futures: bool = False,
        ) -> None:
            assert wait is True
            shutdown_started.set()
            events.append(("shutdown:start", wait, cancel_futures))
            super().shutdown(wait=True, cancel_futures=cancel_futures)
            shutdown_finished.set()
            events.append(("shutdown:finish",))

    def record_cancel_after_shutdown() -> None:
        with finished_lock:
            assert finished == set(range(1, RACE_SIZE))
        assert shutdown_finished.is_set()
        events.append(("cancel",))

    def fail_or_finish(item: int) -> object:
        if item == 0:
            assert slow_workers_ready.wait(timeout=RACE_TIMEOUT_SECONDS)
            raise RuntimeError("worker failed")
        nonlocal ready_count
        with ready_lock:
            ready_count += 1
            if ready_count == RACE_SIZE - 1:
                slow_workers_ready.set()
        assert shutdown_started.wait(timeout=RACE_TIMEOUT_SECONDS)
        with finished_lock:
            finished.add(item)
        return item

    events.clear()
    monkeypatch.setattr(
        faulthandler,
        "cancel_dump_traceback_later",
        record_cancel_after_shutdown,
    )
    monkeypatch.setitem(
        run_race.__globals__,
        "ThreadPoolExecutor",
        ObservedExecutor,
    )
    with pytest.raises(RuntimeError, match="worker failed"):
        run_race(tuple(range(RACE_SIZE)), fail_or_finish)
    assert slow_workers_ready.is_set()
    assert shutdown_started.is_set()
    assert shutdown_finished.is_set()
    assert finished == set(range(1, RACE_SIZE))
    assert events == [
        ("arm", HARD_RACE_TIMEOUT_SECONDS, True),
        ("shutdown:start", True, False),
        ("shutdown:finish",),
        ("cancel",),
    ]


def test_run_race_watchdog_wraps_executor_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[object, ...]] = []

    def record_arm(timeout: float, *, exit: bool) -> None:
        events.append(("arm", timeout, exit))

    def record_cancel() -> None:
        events.append(("cancel",))

    class FailingFuture:
        def result(self, timeout: float) -> object:
            events.append(("result",))
            raise RuntimeError("worker failed")

    class StubExecutor:
        def __init__(self, *, max_workers: int) -> None:
            events.append(("executor:init", max_workers))

        def __enter__(self) -> StubExecutor:
            events.append(("executor:enter",))
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            traceback: object,
        ) -> None:
            events.append(("executor:exit", exc_type))

        def submit(
            self,
            operation: Callable[[int], object],
            item: int,
        ) -> FailingFuture:
            events.append(("submit", item))
            return FailingFuture()

    monkeypatch.setattr(faulthandler, "dump_traceback_later", record_arm)
    monkeypatch.setattr(faulthandler, "cancel_dump_traceback_later", record_cancel)
    monkeypatch.setitem(run_race.__globals__, "ThreadPoolExecutor", StubExecutor)

    with pytest.raises(RuntimeError, match="worker failed"):
        run_race((0,), lambda item: item)
    assert events == [
        ("arm", HARD_RACE_TIMEOUT_SECONDS, True),
        ("executor:init", 1),
        ("executor:enter",),
        ("submit", 0),
        ("result",),
        ("executor:exit", RuntimeError),
        ("cancel",),
    ]


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


def seed_revisions(
    count: int = RACE_SIZE,
    *,
    siblings: bool = False,
    scope: MemoryScope | None = None,
) -> tuple[InMemoryMemoryHistoryStore, tuple[MemoryRevision, ...]]:
    """Seed independent roots or same-memory siblings for release races."""

    scope = scope or make_scope()
    evidence_store = InMemoryEvidenceStore()
    evidence = evidence_store.append(
        make_event(
            scope=scope,
            sequence_no=0,
            payload="race evidence",
            idempotency_key="race-evidence",
        )
    )
    history = InMemoryMemoryHistoryStore(evidence_store)

    def make_revision(index: int, parent: MemoryRevision | None) -> MemoryRevision:
        candidate = append_candidate(
            history,
            scope=scope,
            evidence_ids=(evidence.evidence_id,),
            content=f"race memory {index}",
            idempotency_key=f"race-candidate-{index}",
        )
        return append_revision(
            history,
            scope=scope,
            candidate_id=candidate.candidate_id,
            operation=(
                RevisionOperation.ADD if parent is None else RevisionOperation.REFINE
            ),
            parent_revision_id=None if parent is None else parent.revision_id,
            idempotency_key=f"race-revision-{index}",
        )

    if not siblings:
        return history, tuple(make_revision(index, None) for index in range(count))

    parent = make_revision(-1, None)
    return history, tuple(make_revision(index, parent) for index in range(count))


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
    expected_graph_bytes = (
        b'{"ancestry_order":"selected_to_add_root",'
        b'"record_kind":"memory_release_graph","schema_version":1,'
        b'"selected_revisions":[]}'
    )
    expected_graph_hash = sha256(expected_graph_bytes).hexdigest()
    expected_commitment = _release_commitment_bytes(
        sha256(manifest.canonical_bytes()).hexdigest(),
        expected_graph_hash,
    )
    expected_hash = sha256(expected_commitment).hexdigest()

    release = store.append_release(manifest, idempotency_key="memory-off")

    assert history.lookup_count == 0
    assert release.release_id == f"rel_{expected_hash[:24]}"
    assert release.content_hash == expected_hash
    assert release.release_graph_sha256 == expected_graph_hash
    assert release.commitment_bytes() == expected_commitment
    assert release.release_graph_sha256 == (
        "f1581b6dd2fc67d76188af8902552c835ec99debd31ec9f1f5f26b8f1e0a81a9"
    )
    assert release.content_hash == (
        "db06eccca5cebc8730e6cf2562fc9f2487a69add6a4ec4486cb96bab7ef1b0a4"
    )
    assert release.release_id == "rel_db06eccca5cebc8730e6cf25"
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


def test_release_hash_and_identifier_commit_to_manifest_and_derived_graph() -> None:
    history, _, (root, _, _) = seeded_history()
    manifest = ReleaseManifest(make_scope(), (root.revision_id,))
    first_store = InMemoryMemoryReleaseStore(history)
    second_store = InMemoryMemoryReleaseStore(history)

    first = first_store.append_release(manifest, idempotency_key="first-key")
    second = second_store.append_release(manifest, idempotency_key="other-key")

    expected_hash = sha256(first.commitment_bytes()).hexdigest()
    assert first.release_graph_sha256 == second.release_graph_sha256
    assert first.content_hash == second.content_hash == expected_hash
    assert first.release_id == second.release_id == f"rel_{expected_hash[:24]}"
    assert re.fullmatch(r"rel_[0-9a-f]{24}", first.release_id)


def test_append_release_uses_module_level_sha256(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = ReleaseManifest(make_scope(), ())
    observed_bytes: list[bytes] = []
    real_sha256 = release_store_module.sha256

    def fake_sha256(canonical_bytes: bytes) -> StubHash:
        observed_bytes.append(canonical_bytes)
        return StubHash(real_sha256(canonical_bytes).hexdigest())

    monkeypatch.setattr(release_store_module, "sha256", fake_sha256)
    store = InMemoryMemoryReleaseStore(NoLookupHistory())  # type: ignore[arg-type]

    release = store.append_release(manifest, idempotency_key="release-1")

    assert observed_bytes == [
        manifest.canonical_bytes(),
        (
            b'{"ancestry_order":"selected_to_add_root",'
            b'"record_kind":"memory_release_graph","schema_version":1,'
            b'"selected_revisions":[]}'
        ),
        release.commitment_bytes(),
    ]
    assert release.content_hash == sha256(release.commitment_bytes()).hexdigest()


def test_release_validation_and_commit_obey_exact_lock_epochs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history, _, (root, _, _) = seeded_history()
    records: list[tuple[str, int]] = []

    class EpochLock:
        def __init__(self) -> None:
            self._lock = RLock()
            self.held = False
            self.epoch = 0
            self.enter_count = 0

        def __enter__(self) -> EpochLock:
            self._lock.acquire()
            assert not self.held
            self.held = True
            self.epoch += 1
            self.enter_count += 1
            records.append(("lock:enter", self.epoch))
            return self

        def __exit__(
            self,
            exc_type: object,
            exc_value: object,
            traceback: object,
        ) -> None:
            records.append(("lock:exit", self.epoch))
            self.held = False
            self._lock.release()

        def record_locked(self, event: str) -> None:
            assert self.held, f"{event} accessed outside release lock"
            records.append((event, self.epoch))

        def record_unlocked(self, event: str) -> None:
            assert not self.held, f"{event} ran inside release lock"
            records.append((event, self.epoch))

    epoch_lock = EpochLock()

    class RecordingList(list[object]):
        def append(self, item: object) -> None:
            epoch_lock.record_locked("scope:append")
            super().append(item)

    class RecordingDict(dict[object, object]):
        def __init__(self, name: str) -> None:
            super().__init__()
            self._name = name

        def get(self, key: object, default: object = None) -> object:
            epoch_lock.record_locked(f"{self._name}:get")
            return super().get(key, default)

        def __setitem__(self, key: object, value: object) -> None:
            epoch_lock.record_locked(f"{self._name}:set")
            super().__setitem__(key, value)

        def setdefault(self, key: object, default: object = None) -> object:
            epoch_lock.record_locked(f"{self._name}:setdefault")
            if not dict.__contains__(self, key):
                default = RecordingList(default or ())
            return super().setdefault(key, default)

    class ObservedHistory:
        def get_revision(
            self,
            scope: MemoryScope,
            revision_id: str,
        ) -> MemoryRevision:
            epoch_lock.record_unlocked("history:get")
            return history.get_revision(scope, revision_id)

        def get_candidate(self, scope: MemoryScope, candidate_id: str):
            epoch_lock.record_unlocked("history:get_candidate")
            return history.get_candidate(scope, candidate_id)

        def get_candidate_evidence(self, scope: MemoryScope, candidate_id: str):
            epoch_lock.record_unlocked("history:get_candidate_evidence")
            return history.get_candidate_evidence(scope, candidate_id)

    original_sha256 = release_store_module.sha256

    def observed_sha256(canonical_bytes: bytes) -> object:
        epoch_lock.record_unlocked("sha256")
        return original_sha256(canonical_bytes)

    monkeypatch.setattr(release_store_module, "sha256", observed_sha256)
    store = InMemoryMemoryReleaseStore(ObservedHistory())  # type: ignore[arg-type]
    store._lock = epoch_lock  # type: ignore[assignment]
    store._release_by_id = RecordingDict("release")  # type: ignore[assignment]
    store._release_by_idempotency = RecordingDict(  # type: ignore[assignment]
        "idempotency"
    )
    store._releases_by_scope = RecordingDict("scope")  # type: ignore[assignment]
    manifest = ReleaseManifest(make_scope(), (root.revision_id,))

    release = store.append_release(manifest, idempotency_key="release-1")

    assert epoch_lock.enter_count == 2
    assert records[0] == ("sha256", 0)
    assert records.count(("lock:enter", 1)) == 1
    assert records.count(("lock:enter", 2)) == 1
    assert ("release:set", 2) in records
    assert ("idempotency:set", 2) in records
    assert ("scope:set", 2) in records
    assert all(
        event
        not in {
            "history:get",
            "history:get_candidate",
            "history:get_candidate_evidence",
            "sha256",
        }
        or epoch in {0, 1}
        for event, epoch in records
    )

    records.clear()
    epoch_lock.epoch = 0
    epoch_lock.enter_count = 0
    alias = store.append_release(manifest, idempotency_key="release-alias")

    assert alias is release
    assert epoch_lock.enter_count == 2
    assert records[0] == ("sha256", 0)
    assert records.count(("lock:enter", 1)) == 1
    assert records.count(("lock:enter", 2)) == 1
    assert ("idempotency:set", 2) in records
    assert ("release:set", 2) not in records


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


def test_missing_member_error_is_not_rewritten_by_same_key_winner() -> None:
    history, revisions = seed_revisions(1)
    scope = make_scope()
    missing_lookup_started = Event()
    allow_missing_lookup = Event()

    class BlockingMissingHistory:
        def get_revision(
            self,
            requested_scope: MemoryScope,
            revision_id: str,
        ) -> MemoryRevision:
            if revision_id == "rev_missing":
                missing_lookup_started.set()
                assert allow_missing_lookup.wait(timeout=RACE_TIMEOUT_SECONDS)
            return history.get_revision(requested_scope, revision_id)

        def get_candidate(self, requested_scope: MemoryScope, candidate_id: str):
            return history.get_candidate(requested_scope, candidate_id)

        def get_candidate_evidence(
            self,
            requested_scope: MemoryScope,
            candidate_id: str,
        ):
            return history.get_candidate_evidence(requested_scope, candidate_id)

    store = InMemoryMemoryReleaseStore(BlockingMissingHistory())  # type: ignore[arg-type]
    invalid_manifest = ReleaseManifest(scope, ("rev_missing",))
    valid_manifest = ReleaseManifest(scope, (revisions[0].revision_id,))
    winner: MemoryRelease | None = None
    invalid_error: RevisionNotFoundError | None = None

    faulthandler.dump_traceback_later(HARD_RACE_TIMEOUT_SECONDS, exit=True)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            invalid_future = executor.submit(
                store.append_release,
                invalid_manifest,
                idempotency_key="shared-release-key",
            )
            try:
                assert missing_lookup_started.wait(timeout=RACE_TIMEOUT_SECONDS)
                valid_future = executor.submit(
                    store.append_release,
                    valid_manifest,
                    idempotency_key="shared-release-key",
                )
                winner = valid_future.result(timeout=RACE_TIMEOUT_SECONDS)
            finally:
                allow_missing_lookup.set()

            with pytest.raises(RevisionNotFoundError, match="rev_missing") as raised:
                invalid_future.result(timeout=RACE_TIMEOUT_SECONDS)
            invalid_error = raised.value
    finally:
        faulthandler.cancel_dump_traceback_later()

    assert type(invalid_error) is RevisionNotFoundError
    assert winner is not None
    assert winner.manifest is valid_manifest
    assert store._release_by_id == {(scope, winner.release_id): winner}
    assert store._release_by_idempotency == {(scope, "shared-release-key"): winner}
    assert store._releases_by_scope == {scope: [winner]}


def test_concurrent_same_key_and_manifest_converge_to_one_release() -> None:
    history, revisions = seed_revisions(1)
    store = InMemoryMemoryReleaseStore(history)
    store._release_by_id = YieldingMissingDict()  # type: ignore[assignment]
    store._release_by_idempotency = YieldingMissingDict()  # type: ignore[assignment]
    manifest = ReleaseManifest(make_scope(), (revisions[0].revision_id,))

    outcomes = run_race(
        (manifest,) * RACE_SIZE,
        lambda item: store.append_release(item, idempotency_key="shared-key"),
    )

    release = outcomes[0]
    assert isinstance(release, MemoryRelease)
    assert all(item is release for item in outcomes)
    assert store._release_by_id == {(manifest.scope, release.release_id): release}
    assert store._release_by_idempotency == {(manifest.scope, "shared-key"): release}
    assert store._releases_by_scope == {manifest.scope: [release]}


def test_concurrent_keys_for_same_manifest_all_alias_one_release() -> None:
    history, revisions = seed_revisions(1)
    store = InMemoryMemoryReleaseStore(history)
    store._release_by_id = YieldingMissingDict()  # type: ignore[assignment]
    store._release_by_idempotency = YieldingMissingDict()  # type: ignore[assignment]
    manifest = ReleaseManifest(make_scope(), (revisions[0].revision_id,))
    keys = tuple(f"release-key-{index}" for index in range(RACE_SIZE))

    outcomes = run_race(
        keys,
        lambda key: store.append_release(manifest, idempotency_key=key),
    )

    release = outcomes[0]
    assert isinstance(release, MemoryRelease)
    assert all(item is release for item in outcomes)
    assert store._release_by_id == {(manifest.scope, release.release_id): release}
    assert store._release_by_idempotency == {
        (manifest.scope, key): release for key in keys
    }
    assert store._releases_by_scope == {manifest.scope: [release]}


def test_concurrent_manifests_for_shared_key_leave_losers_recoverable() -> None:
    history, revisions = seed_revisions()
    store = InMemoryMemoryReleaseStore(history)
    store._release_by_id = YieldingMissingDict()  # type: ignore[assignment]
    store._release_by_idempotency = YieldingMissingDict()  # type: ignore[assignment]
    manifests = tuple(
        ReleaseManifest(make_scope(), (revision.revision_id,)) for revision in revisions
    )

    outcomes = run_race(
        manifests,
        lambda manifest: store.append_release(
            manifest,
            idempotency_key="shared-release-key",
        ),
    )

    winners = tuple(item for item in outcomes if isinstance(item, MemoryRelease))
    conflicts = tuple(
        item for item in outcomes if isinstance(item, ReleaseConflictError)
    )
    assert len(winners) == 1
    assert len(conflicts) == RACE_SIZE - 1
    winner = winners[0]
    assert store._release_by_id == {(winner.manifest.scope, winner.release_id): winner}
    assert store._release_by_idempotency == {
        (winner.manifest.scope, "shared-release-key"): winner
    }
    assert store._releases_by_scope == {winner.manifest.scope: [winner]}

    loser_manifests = tuple(
        manifest for manifest in manifests if manifest is not winner.manifest
    )
    recovered = tuple(
        store.append_release(
            manifest,
            idempotency_key=f"recovered-release-{index}",
        )
        for index, manifest in enumerate(loser_manifests)
    )
    releases = store.list_releases(make_scope())
    assert len(recovered) == RACE_SIZE - 1
    assert len(releases) == RACE_SIZE
    assert len(store._release_by_id) == RACE_SIZE
    assert len(store._release_by_idempotency) == RACE_SIZE
    assert len(store._releases_by_scope[make_scope()]) == RACE_SIZE
    assert set(store._release_by_id.values()) == set(releases)
    assert set(store._release_by_idempotency.values()) == set(releases)
    assert set(store._releases_by_scope[make_scope()]) == set(releases)


def test_same_memory_siblings_survive_in_separate_concurrent_releases() -> None:
    history, siblings = seed_revisions(siblings=True)
    assert len({revision.memory_id for revision in siblings}) == 1
    store = InMemoryMemoryReleaseStore(history)
    store._release_by_id = YieldingMissingDict()  # type: ignore[assignment]
    store._release_by_idempotency = YieldingMissingDict()  # type: ignore[assignment]
    requests = tuple(
        (
            ReleaseManifest(make_scope(), (revision.revision_id,)),
            f"sibling-release-{index}",
        )
        for index, revision in enumerate(siblings)
    )

    outcomes = run_race(
        requests,
        lambda item: store.append_release(item[0], idempotency_key=item[1]),
    )

    releases = tuple(item for item in outcomes if isinstance(item, MemoryRelease))
    assert len(releases) == RACE_SIZE
    assert len({release.release_id for release in releases}) == RACE_SIZE
    assert {release.manifest.revision_ids[0] for release in releases} == {
        revision.revision_id for revision in siblings
    }
    assert set(store.list_releases(make_scope())) == set(releases)
    assert len(store._release_by_id) == RACE_SIZE
    assert len(store._release_by_idempotency) == RACE_SIZE
    assert len(store._releases_by_scope[make_scope()]) == RACE_SIZE


@pytest.mark.parametrize("same_full_hash", [True, False], ids=["full", "prefix"])
def test_concurrent_release_collision_is_atomic_and_all_losers_recover(
    monkeypatch: pytest.MonkeyPatch,
    same_full_hash: bool,
) -> None:
    history, revisions = seed_revisions()
    scope = make_scope()
    store = InMemoryMemoryReleaseStore(history)
    store._release_by_id = YieldingMissingDict()  # type: ignore[assignment]
    store._release_by_idempotency = YieldingMissingDict()  # type: ignore[assignment]
    manifests = tuple(
        ReleaseManifest(scope, (revision.revision_id,)) for revision in revisions
    )
    keys = tuple(f"collision-release-{index}" for index in range(RACE_SIZE))
    prefix = "a" * 24
    digest_by_bytes = {
        release_commitment_for(store, manifest): (
            prefix + ("b" * 40 if same_full_hash else f"{index:040x}")
        )
        for index, manifest in enumerate(manifests)
    }
    assert {digest[:24] for digest in digest_by_bytes.values()} == {prefix}
    assert len(set(digest_by_bytes.values())) == (1 if same_full_hash else RACE_SIZE)
    install_digest_map(monkeypatch, digest_by_bytes)

    outcomes = run_race(
        tuple(zip(manifests, keys, strict=True)),
        lambda item: store.append_release(item[0], idempotency_key=item[1]),
    )

    winners = tuple(item for item in outcomes if isinstance(item, MemoryRelease))
    conflicts = tuple(
        item for item in outcomes if isinstance(item, ReleaseConflictError)
    )
    assert len(winners) == 1
    assert len(conflicts) == RACE_SIZE - 1
    winner = winners[0]
    winner_index = next(
        index
        for index, outcome in enumerate(outcomes)
        if isinstance(outcome, MemoryRelease)
    )
    winner_key = keys[winner_index]
    loser_indexes = tuple(index for index in range(RACE_SIZE) if index != winner_index)
    assert store._release_by_id == {(scope, winner.release_id): winner}
    assert store._release_by_idempotency == {(scope, winner_key): winner}
    assert store._releases_by_scope == {scope: [winner]}
    assert all(
        (scope, keys[index]) not in store._release_by_idempotency
        for index in loser_indexes
    )

    monkeypatch.undo()
    recovered = tuple(
        store.append_release(
            manifests[index],
            idempotency_key=keys[index],
        )
        for index in loser_indexes
    )

    releases = store.list_releases(scope)
    assert len(recovered) == RACE_SIZE - 1
    assert len(releases) == RACE_SIZE
    assert len(store._release_by_id) == RACE_SIZE
    assert len(store._release_by_idempotency) == RACE_SIZE
    assert len(store._releases_by_scope[scope]) == RACE_SIZE
    assert set(store._release_by_id.values()) == set(releases)
    assert set(store._release_by_idempotency.values()) == set(releases)
    assert set(store._releases_by_scope[scope]) == set(releases)
    assert {release.manifest for release in releases} == set(manifests)


@pytest.mark.parametrize("same_full_hash", [True, False], ids=["full", "prefix"])
def test_forced_release_id_collision_is_isolated_across_scopes(
    monkeypatch: pytest.MonkeyPatch,
    same_full_hash: bool,
) -> None:
    first_scope = make_scope(subject_id="user-1")
    second_scope = make_scope(subject_id="user-2")
    manifests = (
        ReleaseManifest(first_scope, ()),
        ReleaseManifest(second_scope, ()),
    )
    prefix = "c" * 24
    digests = tuple(
        prefix + ("d" * 40 if same_full_hash else f"{index:040x}") for index in range(2)
    )
    history = NoLookupHistory()
    store = InMemoryMemoryReleaseStore(history)  # type: ignore[arg-type]
    install_digest_map(
        monkeypatch,
        {
            release_commitment_for(store, manifest): digest
            for manifest, digest in zip(manifests, digests, strict=True)
        },
    )
    store._release_by_id = YieldingMissingDict()  # type: ignore[assignment]
    store._release_by_idempotency = YieldingMissingDict()  # type: ignore[assignment]

    outcomes = run_race(
        manifests,
        lambda manifest: store.append_release(
            manifest,
            idempotency_key="shared-scoped-key",
        ),
    )
    aliases = run_race(
        manifests,
        lambda manifest: store.append_release(
            manifest,
            idempotency_key="shared-scoped-alias",
        ),
    )

    first, second = outcomes
    assert isinstance(first, MemoryRelease)
    assert isinstance(second, MemoryRelease)
    assert first is not second
    assert first.release_id == second.release_id == f"rel_{prefix}"
    assert (first.content_hash == second.content_hash) is same_full_hash
    assert aliases[0] is first
    assert aliases[1] is second
    assert store.get_release(first_scope, first.release_id) is first
    assert store.get_release(second_scope, second.release_id) is second
    assert store.list_releases(first_scope) == (first,)
    assert store.list_releases(second_scope) == (second,)
    assert store._release_by_id == {
        (first_scope, first.release_id): first,
        (second_scope, second.release_id): second,
    }
    assert store._release_by_idempotency == {
        (first_scope, "shared-scoped-key"): first,
        (second_scope, "shared-scoped-key"): second,
        (first_scope, "shared-scoped-alias"): first,
        (second_scope, "shared-scoped-alias"): second,
    }
    assert store._releases_by_scope == {
        first_scope: [first],
        second_scope: [second],
    }
    assert history.lookup_count == 0


def make_colliding_root_revision(
    monkeypatch: pytest.MonkeyPatch,
    *,
    root: MemoryRevision,
    alternate_candidate_id: str,
) -> MemoryRevision:
    """Build a valid alternate full hash behind the same 96-bit address."""

    proposal = RevisionProposal(
        scope=root.proposal.scope,
        candidate_id=alternate_candidate_id,
        operation=RevisionOperation.ADD,
        parent_revision_id=None,
        idempotency_key="revision-collision-alternate",
    )
    prefix = root.revision_id.removeprefix("rev_")
    content_hash = prefix + "f" * 40
    assert content_hash != root.content_hash
    install_digest_map(monkeypatch, {proposal.canonical_bytes(): content_hash})
    return MemoryRevision(
        revision_id=root.revision_id,
        memory_id=f"mem_{prefix}",
        generation=0,
        proposal=proposal,
        content_hash=content_hash,
        created_at=UTC_INSTANT,
    )


def test_same_manifest_over_different_valid_full_graphs_has_distinct_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history, _, (root, _, other) = seeded_history()
    alternate = make_colliding_root_revision(
        monkeypatch,
        root=root,
        alternate_candidate_id=other.proposal.candidate_id,
    )

    class AlternateHistory:
        def get_revision(self, scope: MemoryScope, revision_id: str):
            if revision_id == root.revision_id:
                return alternate
            return history.get_revision(scope, revision_id)

        def get_candidate(self, scope: MemoryScope, candidate_id: str):
            return history.get_candidate(scope, candidate_id)

        def get_candidate_evidence(self, scope: MemoryScope, candidate_id: str):
            return history.get_candidate_evidence(scope, candidate_id)

    manifest = ReleaseManifest(make_scope(), (root.revision_id,))
    original = InMemoryMemoryReleaseStore(history).append_release(
        manifest,
        idempotency_key="original",
    )
    collided = InMemoryMemoryReleaseStore(AlternateHistory()).append_release(
        manifest,
        idempotency_key="alternate",
    )

    assert original.manifest == collided.manifest == manifest
    assert root.revision_id == alternate.revision_id
    assert root.content_hash != alternate.content_hash
    assert original.release_graph_sha256 != collided.release_graph_sha256
    assert original.content_hash != collided.content_hash
    assert original.release_id != collided.release_id


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("wrong_type", "non-MemoryRevision"),
        ("wrong_scope", "requested scope"),
        ("wrong_address", "address"),
        ("wrong_hash", "canonical commitment"),
    ],
)
def test_append_release_rejects_malicious_revision_values(
    mutation: str,
    message: str,
) -> None:
    history, _, (root, _, _) = seeded_history()
    foreign_proposal = RevisionProposal(
        scope=make_scope(subject_id="attacker"),
        candidate_id=root.proposal.candidate_id,
        operation=RevisionOperation.ADD,
        parent_revision_id=None,
        idempotency_key="foreign",
    )

    class MaliciousHistory:
        def get_revision(self, scope: MemoryScope, revision_id: str):
            if mutation == "wrong_type":
                return object()
            if mutation == "wrong_scope":
                return replace(root, proposal=foreign_proposal)
            if mutation == "wrong_address":
                return replace(root, revision_id="rev_other")
            return replace(root, content_hash="0" * 64)

        def get_candidate(self, scope: MemoryScope, candidate_id: str):
            return history.get_candidate(scope, candidate_id)

        def get_candidate_evidence(self, scope: MemoryScope, candidate_id: str):
            return history.get_candidate_evidence(scope, candidate_id)

    store = InMemoryMemoryReleaseStore(MaliciousHistory())  # type: ignore[arg-type]
    with pytest.raises(ReleaseConflictError, match=message):
        store.append_release(
            ReleaseManifest(make_scope(), (root.revision_id,)),
            idempotency_key="malicious",
        )
    assert_release_indexes_empty(store)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("candidate_type", "non-MemoryCandidate"),
        ("candidate_address", "candidate address"),
        ("candidate_hash", "candidate canonical commitment"),
        ("evidence_order", "evidence addresses"),
        ("evidence_hash", "evidence canonical commitment"),
    ],
)
def test_append_release_rejects_candidate_and_evidence_mutation(
    mutation: str,
    message: str,
) -> None:
    history, _, (root, _, _) = seeded_history()
    candidate = history.get_candidate(make_scope(), root.proposal.candidate_id)
    evidence = history.get_candidate_evidence(make_scope(), candidate.candidate_id)

    class MaliciousHistory:
        def get_revision(self, scope: MemoryScope, revision_id: str):
            return history.get_revision(scope, revision_id)

        def get_candidate(self, scope: MemoryScope, candidate_id: str):
            if mutation == "candidate_type":
                return object()
            if mutation == "candidate_address":
                return replace(candidate, candidate_id="cand_other")
            if mutation == "candidate_hash":
                return replace(candidate, content_hash="0" * 64)
            return history.get_candidate(scope, candidate_id)

        def get_candidate_evidence(self, scope: MemoryScope, candidate_id: str):
            if mutation == "evidence_order":
                return tuple(reversed(evidence))
            if mutation == "evidence_hash":
                return (replace(evidence[0], content_hash="0" * 64), evidence[1])
            return history.get_candidate_evidence(scope, candidate_id)

    store = InMemoryMemoryReleaseStore(MaliciousHistory())  # type: ignore[arg-type]
    with pytest.raises(ReleaseConflictError, match=message):
        store.append_release(
            ReleaseManifest(make_scope(), (root.revision_id,)),
            idempotency_key="malicious-source",
        )
    assert_release_indexes_empty(store)


def test_release_graph_records_selected_to_add_root_and_excludes_other_branch() -> None:
    history, left, right = seeded_sibling_revisions()
    root_id = left.proposal.parent_revision_id
    assert root_id is not None
    store = InMemoryMemoryReleaseStore(history)
    left_manifest = ReleaseManifest(make_scope(), (left.revision_id,))
    right_manifest = ReleaseManifest(make_scope(), (right.revision_id,))

    left_graph_bytes, _ = store._derive_release_graph(left_manifest)
    right_graph_bytes, _ = store._derive_release_graph(right_manifest)
    left_graph = json.loads(left_graph_bytes)
    ancestry = left_graph["selected_revisions"][0]["ancestry"]

    assert left_graph["ancestry_order"] == "selected_to_add_root"
    assert [node["revision_id"] for node in ancestry] == [left.revision_id, root_id]
    assert right.revision_id not in {node["revision_id"] for node in ancestry}
    assert set(ancestry[0]) == {
        "candidate",
        "generation",
        "memory_id",
        "revision_id",
        "revision_sha256",
    }
    assert set(ancestry[0]["candidate"]) == {
        "candidate_id",
        "candidate_sha256",
        "evidence",
    }
    assert all(
        set(reference) == {"evidence_id", "evidence_sha256"}
        for node in ancestry
        for reference in node["candidate"]["evidence"]
    )
    assert b"root memory" not in left_graph_bytes
    assert b"left sibling" not in left_graph_bytes
    left_release = store.append_release(left_manifest, idempotency_key="left")
    right_release = store.append_release(right_manifest, idempotency_key="right")
    assert left_graph_bytes != right_graph_bytes
    assert left_release.release_graph_sha256 != right_release.release_graph_sha256


def test_deep_release_lineage_is_validated_iteratively() -> None:
    scope = make_scope()
    evidence_store = InMemoryEvidenceStore()
    evidence = evidence_store.append(
        make_event(
            scope=scope,
            sequence_no=0,
            payload="deep lineage",
            idempotency_key="deep-evidence",
        )
    )
    history = InMemoryMemoryHistoryStore(evidence_store)
    parent: MemoryRevision | None = None
    for generation in range(1050):
        candidate = append_candidate(
            history,
            scope=scope,
            evidence_ids=(evidence.evidence_id,),
            content=f"memory generation {generation}",
            idempotency_key=f"deep-candidate-{generation}",
        )
        parent = append_revision(
            history,
            scope=scope,
            candidate_id=candidate.candidate_id,
            operation=(
                RevisionOperation.ADD if parent is None else RevisionOperation.REFINE
            ),
            parent_revision_id=None if parent is None else parent.revision_id,
            idempotency_key=f"deep-revision-{generation}",
        )
    assert parent is not None
    store = InMemoryMemoryReleaseStore(history)
    release = store.append_release(
        ReleaseManifest(scope, (parent.revision_id,)),
        idempotency_key="deep-release",
    )
    assert store.get_release_revisions(scope, release.release_id) == (parent,)


def test_append_release_rejects_iterative_ancestry_cycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history, _, (_, child, _) = seeded_history()
    cyclic_proposal = RevisionProposal(
        scope=make_scope(),
        candidate_id=child.proposal.candidate_id,
        operation=RevisionOperation.REFINE,
        parent_revision_id=child.revision_id,
        idempotency_key="cyclic-revision",
    )
    prefix = child.revision_id.removeprefix("rev_")
    cyclic_hash = prefix + "e" * 40
    install_digest_map(monkeypatch, {cyclic_proposal.canonical_bytes(): cyclic_hash})
    cyclic = replace(child, proposal=cyclic_proposal, content_hash=cyclic_hash)

    class CyclicHistory:
        def get_revision(self, scope: MemoryScope, revision_id: str):
            return cyclic

        def get_candidate(self, scope: MemoryScope, candidate_id: str):
            return history.get_candidate(scope, candidate_id)

        def get_candidate_evidence(self, scope: MemoryScope, candidate_id: str):
            return history.get_candidate_evidence(scope, candidate_id)

    store = InMemoryMemoryReleaseStore(CyclicHistory())  # type: ignore[arg-type]
    with pytest.raises(ReleaseConflictError, match="cycle"):
        store.append_release(
            ReleaseManifest(make_scope(), (child.revision_id,)),
            idempotency_key="cycle",
        )
    assert_release_indexes_empty(store)


def test_final_graph_recheck_rejects_valid_toctou_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history, _, (root, _, other) = seeded_history()
    alternate = make_colliding_root_revision(
        monkeypatch,
        root=root,
        alternate_candidate_id=other.proposal.candidate_id,
    )

    class DriftingHistory:
        selected_reads = 0

        def get_revision(self, scope: MemoryScope, revision_id: str):
            self.selected_reads += 1
            return root if self.selected_reads == 1 else alternate

        def get_candidate(self, scope: MemoryScope, candidate_id: str):
            return history.get_candidate(scope, candidate_id)

        def get_candidate_evidence(self, scope: MemoryScope, candidate_id: str):
            return history.get_candidate_evidence(scope, candidate_id)

    store = InMemoryMemoryReleaseStore(DriftingHistory())  # type: ignore[arg-type]
    with pytest.raises(ReleaseConflictError, match="final graph recheck"):
        store.append_release(
            ReleaseManifest(make_scope(), (root.revision_id,)),
            idempotency_key="toctou",
        )
    assert_release_indexes_empty(store)


def test_get_release_revisions_rejects_valid_committed_graph_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history, _, (root, _, other) = seeded_history()
    alternate = make_colliding_root_revision(
        monkeypatch,
        root=root,
        alternate_candidate_id=other.proposal.candidate_id,
    )

    class MutableHistory:
        drift = False

        def get_revision(self, scope: MemoryScope, revision_id: str):
            if self.drift and revision_id == root.revision_id:
                return alternate
            return history.get_revision(scope, revision_id)

        def get_candidate(self, scope: MemoryScope, candidate_id: str):
            return history.get_candidate(scope, candidate_id)

        def get_candidate_evidence(self, scope: MemoryScope, candidate_id: str):
            return history.get_candidate_evidence(scope, candidate_id)

    mutable_history = MutableHistory()
    store = InMemoryMemoryReleaseStore(mutable_history)  # type: ignore[arg-type]
    release = store.append_release(
        ReleaseManifest(make_scope(), (root.revision_id,)),
        idempotency_key="stable",
    )
    mutable_history.drift = True

    with pytest.raises(ReleaseConflictError, match="committed release identity"):
        store.get_release_revisions(make_scope(), release.release_id)


class PublicationInterrupted(BaseException):
    pass


class SetThenInterruptDict(dict[object, object]):
    def __setitem__(self, key: object, value: object) -> None:
        super().__setitem__(key, value)
        raise PublicationInterrupted


def test_interrupted_second_release_index_rolls_back_every_write() -> None:
    store = InMemoryMemoryReleaseStore(NoLookupHistory())  # type: ignore[arg-type]
    store._release_by_idempotency = SetThenInterruptDict()  # type: ignore[assignment]

    with pytest.raises(PublicationInterrupted):
        store.append_release(
            ReleaseManifest(make_scope(), ()),
            idempotency_key="interrupt-second",
        )

    assert_release_indexes_empty(store)


def test_interrupted_third_release_index_rolls_back_every_write() -> None:
    store = InMemoryMemoryReleaseStore(NoLookupHistory())  # type: ignore[arg-type]
    store._releases_by_scope = SetThenInterruptDict()  # type: ignore[assignment]

    with pytest.raises(PublicationInterrupted):
        store.append_release(
            ReleaseManifest(make_scope(), ()),
            idempotency_key="interrupt-third",
        )

    assert_release_indexes_empty(store)
