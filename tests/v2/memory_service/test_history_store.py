# SPDX-License-Identifier: Apache-2.0

"""Tests for the evidence-grounded in-memory history store."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from hashlib import sha256
from threading import RLock

import pytest

from areal.v2.memory_service import history_store as history_store_module
from areal.v2.memory_service.errors import (
    CandidateConflictError,
    CandidateNotFoundError,
    EvidenceNotFoundError,
    MemoryServiceError,
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
from areal.v2.memory_service.store import InMemoryEvidenceStore
from areal.v2.memory_service.types import (
    EvidenceEvent,
    EvidenceKind,
    EvidenceRecord,
    MemoryScope,
)


class MutableHashStr(str):
    """String subclass that records unsafe hashing by store queries."""

    hash_calls: int
    hash_salt: int

    def __new__(cls, value: str) -> MutableHashStr:
        instance = str.__new__(cls, value)
        instance.hash_calls = 0
        instance.hash_salt = 0
        return instance

    def __hash__(self) -> int:
        self.hash_calls += 1
        return str.__hash__(self) + self.hash_salt


class AlwaysEqualStr(str):
    """String subclass that records unsafe equality by store queries."""

    equality_calls: int

    def __new__(cls, value: str) -> AlwaysEqualStr:
        instance = str.__new__(cls, value)
        instance.equality_calls = 0
        return instance

    def __eq__(self, other: object) -> bool:
        self.equality_calls += 1
        return True

    __hash__ = str.__hash__


class MemoryScopeSubclass(MemoryScope):
    pass


class CandidateProposalSubclass(CandidateProposal):
    def canonical_bytes(self) -> bytes:
        return b"overridden"


class RevisionProposalSubclass(RevisionProposal):
    def canonical_bytes(self) -> bytes:
        return b"overridden"


UTC_INSTANT = datetime(2026, 7, 7, 4, 5, 6, tzinfo=UTC)


def make_scope(**overrides: str) -> MemoryScope:
    values = {
        "tenant_id": "tenant-1",
        "namespace": "assistant-memory",
        "subject_id": "user-1",
    }
    values.update(overrides)
    return MemoryScope(**values)


def make_candidate_proposal(**overrides: object) -> CandidateProposal:
    values: dict[str, object] = {
        "scope": make_scope(),
        "content": "remember this",
        "evidence_ids": ("evd_missing",),
        "idempotency_key": "candidate-1",
    }
    values.update(overrides)
    return CandidateProposal(**values)  # type: ignore[arg-type]


def make_revision_proposal(**overrides: object) -> RevisionProposal:
    values: dict[str, object] = {
        "scope": make_scope(),
        "candidate_id": "cand_missing",
        "operation": RevisionOperation.ADD,
        "parent_revision_id": None,
        "idempotency_key": "revision-1",
    }
    values.update(overrides)
    return RevisionProposal(**values)  # type: ignore[arg-type]


def make_evidence_event(**overrides: object) -> EvidenceEvent:
    values: dict[str, object] = {
        "scope": make_scope(),
        "session_id": "session-1",
        "run_id": "run-1",
        "sequence_no": 0,
        "kind": EvidenceKind.USER_MESSAGE,
        "payload": "remember this",
        "observed_at": UTC_INSTANT,
        "idempotency_key": "evidence-1",
    }
    values.update(overrides)
    return EvidenceEvent(**values)  # type: ignore[arg-type]


def seeded_evidence_store() -> tuple[InMemoryEvidenceStore, EvidenceRecord]:
    evidence_store = InMemoryEvidenceStore()
    evidence = evidence_store.append(make_evidence_event())
    return evidence_store, evidence


def assert_candidate_indexes_empty(history: InMemoryMemoryHistoryStore) -> None:
    assert history._candidate_by_id == {}
    assert history._candidate_by_idempotency == {}
    assert history._candidates_by_scope == {}


def test_history_store_contract_exposes_only_candidate_and_revision_surface() -> None:
    public_members = {
        name for name in MemoryHistoryStore.__dict__ if not name.startswith("_")
    }

    assert public_members == {
        "append_candidate",
        "get_candidate",
        "get_candidate_evidence",
        "list_candidates",
        "append_revision",
        "get_revision",
        "list_revisions",
    }


def test_history_errors_share_memory_service_base() -> None:
    assert issubclass(CandidateNotFoundError, MemoryServiceError)
    assert issubclass(CandidateConflictError, MemoryServiceError)
    assert issubclass(RevisionNotFoundError, MemoryServiceError)
    assert issubclass(RevisionConflictError, MemoryServiceError)


def test_append_candidate_requires_existing_evidence_in_same_scope() -> None:
    evidence_store, evidence = seeded_evidence_store()
    history = InMemoryMemoryHistoryStore(evidence_store)
    proposal = make_candidate_proposal(evidence_ids=(evidence.evidence_id,))

    candidate = history.append_candidate(proposal)

    assert candidate.proposal is proposal
    assert candidate.proposal.evidence_ids == (evidence.evidence_id,)
    assert history.get_candidate(proposal.scope, candidate.candidate_id) is candidate
    assert history.get_candidate_evidence(proposal.scope, candidate.candidate_id) == (
        evidence,
    )


def test_append_candidate_rejects_proposal_subclass_before_any_write() -> None:
    proposal = CandidateProposalSubclass(
        make_scope(), "remember this", ("evd_missing",), "candidate-1"
    )
    history = InMemoryMemoryHistoryStore(InMemoryEvidenceStore())

    with pytest.raises(TypeError, match="CandidateProposal"):
        history.append_candidate(proposal)

    assert_candidate_indexes_empty(history)


def test_first_missing_evidence_leaves_every_candidate_index_empty() -> None:
    history = InMemoryMemoryHistoryStore(InMemoryEvidenceStore())
    proposal = make_candidate_proposal(evidence_ids=("evd_missing",))

    with pytest.raises(EvidenceNotFoundError):
        history.append_candidate(proposal)

    assert_candidate_indexes_empty(history)
    assert history.list_candidates(proposal.scope) == ()


def test_late_missing_evidence_leaves_every_candidate_index_empty() -> None:
    evidence_store, evidence = seeded_evidence_store()
    history = InMemoryMemoryHistoryStore(evidence_store)
    proposal = make_candidate_proposal(
        evidence_ids=(evidence.evidence_id, "evd_missing")
    )

    with pytest.raises(EvidenceNotFoundError):
        history.append_candidate(proposal)

    assert_candidate_indexes_empty(history)


def test_candidate_evidence_preserves_proposal_order() -> None:
    evidence_store, first = seeded_evidence_store()
    second = evidence_store.append(
        make_evidence_event(
            sequence_no=1,
            payload="second",
            idempotency_key="evidence-2",
        )
    )
    history = InMemoryMemoryHistoryStore(evidence_store)
    candidate = history.append_candidate(
        make_candidate_proposal(evidence_ids=(second.evidence_id, first.evidence_id))
    )

    assert history.get_candidate_evidence(
        candidate.proposal.scope, candidate.candidate_id
    ) == (second, first)


def test_foreign_scope_evidence_is_indistinguishable_from_missing() -> None:
    evidence_store = InMemoryEvidenceStore()
    foreign = evidence_store.append(
        make_evidence_event(scope=make_scope(subject_id="user-2"))
    )
    history = InMemoryMemoryHistoryStore(evidence_store)
    missing_history = InMemoryMemoryHistoryStore(InMemoryEvidenceStore())
    proposal = make_candidate_proposal(evidence_ids=(foreign.evidence_id,))

    with pytest.raises(EvidenceNotFoundError) as foreign_error:
        history.append_candidate(proposal)
    with pytest.raises(EvidenceNotFoundError) as missing_error:
        missing_history.append_candidate(proposal)

    assert type(foreign_error.value) is EvidenceNotFoundError
    assert str(foreign_error.value) == str(missing_error.value)
    assert_candidate_indexes_empty(history)


def test_identical_candidate_retry_returns_original_record() -> None:
    evidence_store, evidence = seeded_evidence_store()
    history = InMemoryMemoryHistoryStore(evidence_store)
    proposal = make_candidate_proposal(evidence_ids=(evidence.evidence_id,))

    first = history.append_candidate(proposal)
    retry = history.append_candidate(
        CandidateProposal(
            scope=proposal.scope,
            content=proposal.content,
            evidence_ids=proposal.evidence_ids,
            idempotency_key=proposal.idempotency_key,
        )
    )

    assert retry is first
    assert history.list_candidates(proposal.scope) == (first,)
    assert len(history._candidate_by_id) == 1
    assert len(history._candidate_by_idempotency) == 1


def test_changed_candidate_idempotency_key_reuse_conflicts_without_partial_write() -> (
    None
):
    evidence_store, evidence = seeded_evidence_store()
    history = InMemoryMemoryHistoryStore(evidence_store)
    first = history.append_candidate(
        make_candidate_proposal(evidence_ids=(evidence.evidence_id,))
    )

    with pytest.raises(CandidateConflictError, match="idempotency"):
        history.append_candidate(
            make_candidate_proposal(
                content="changed",
                evidence_ids=(evidence.evidence_id,),
            )
        )

    assert history.list_candidates(first.proposal.scope) == (first,)
    assert tuple(history._candidate_by_id.values()) == (first,)
    assert tuple(history._candidate_by_idempotency.values()) == (first,)


def test_committed_idempotency_conflict_precedes_missing_or_foreign_evidence() -> None:
    evidence_store, evidence = seeded_evidence_store()
    foreign = evidence_store.append(
        make_evidence_event(
            scope=make_scope(subject_id="user-2"),
            idempotency_key="foreign-evidence",
        )
    )
    history = InMemoryMemoryHistoryStore(evidence_store)
    first = history.append_candidate(
        make_candidate_proposal(evidence_ids=(evidence.evidence_id,))
    )

    for invalid_evidence_id in ("evd_missing", foreign.evidence_id):
        with pytest.raises(CandidateConflictError, match="idempotency"):
            history.append_candidate(
                make_candidate_proposal(
                    content="changed",
                    evidence_ids=(invalid_evidence_id,),
                )
            )

    assert history.list_candidates(first.proposal.scope) == (first,)
    assert tuple(history._candidate_by_id.values()) == (first,)
    assert tuple(history._candidate_by_idempotency.values()) == (first,)


def test_candidate_uses_full_sha256_truncated_identifier_and_utc_timestamp() -> None:
    evidence_store, evidence = seeded_evidence_store()
    history = InMemoryMemoryHistoryStore(evidence_store)
    proposal = make_candidate_proposal(evidence_ids=(evidence.evidence_id,))
    expected_hash = sha256(proposal.canonical_bytes()).hexdigest()

    candidate = history.append_candidate(proposal)

    assert candidate.content_hash == expected_hash
    assert re.fullmatch(r"[0-9a-f]{64}", candidate.content_hash)
    assert candidate.candidate_id == f"cand_{expected_hash[:24]}"
    assert re.fullmatch(r"cand_[0-9a-f]{24}", candidate.candidate_id)
    assert candidate.created_at.tzinfo is UTC


def test_candidate_lookup_and_lists_are_scope_isolated_and_deterministic() -> None:
    evidence_store, evidence = seeded_evidence_store()
    other_scope = make_scope(subject_id="user-2")
    other_evidence = evidence_store.append(
        make_evidence_event(
            scope=other_scope,
            payload="other",
            idempotency_key="other-evidence",
        )
    )
    history = InMemoryMemoryHistoryStore(evidence_store)
    right = history.append_candidate(
        make_candidate_proposal(
            content="right",
            evidence_ids=(evidence.evidence_id,),
            idempotency_key="right",
        )
    )
    left = history.append_candidate(
        make_candidate_proposal(
            content="left",
            evidence_ids=(evidence.evidence_id,),
            idempotency_key="left",
        )
    )
    other = history.append_candidate(
        make_candidate_proposal(
            scope=other_scope,
            content="other",
            evidence_ids=(other_evidence.evidence_id,),
            idempotency_key="other",
        )
    )
    expected = tuple(sorted((left, right), key=lambda item: item.candidate_id))

    assert (right, left) != expected
    assert history.list_candidates(make_scope()) == expected
    assert history.list_candidates(other_scope) == (other,)
    assert history.get_candidate(make_scope(), left.candidate_id) is left
    with pytest.raises(CandidateNotFoundError):
        history.get_candidate(other_scope, left.candidate_id)


def test_equal_candidate_content_under_new_attempt_is_not_semantically_deduplicated() -> (
    None
):
    evidence_store, evidence = seeded_evidence_store()
    history = InMemoryMemoryHistoryStore(evidence_store)
    first = history.append_candidate(
        make_candidate_proposal(
            evidence_ids=(evidence.evidence_id,), idempotency_key="attempt-a"
        )
    )
    second = history.append_candidate(
        make_candidate_proposal(
            evidence_ids=(evidence.evidence_id,), idempotency_key="attempt-b"
        )
    )

    assert first.candidate_id != second.candidate_id
    assert history.list_candidates(make_scope()) == tuple(
        sorted((first, second), key=lambda item: item.candidate_id)
    )


def test_candidate_list_snapshot_does_not_change_after_later_append() -> None:
    evidence_store, evidence = seeded_evidence_store()
    history = InMemoryMemoryHistoryStore(evidence_store)
    first = history.append_candidate(
        make_candidate_proposal(evidence_ids=(evidence.evidence_id,))
    )
    snapshot = history.list_candidates(make_scope())

    history.append_candidate(
        make_candidate_proposal(
            content="later",
            evidence_ids=(evidence.evidence_id,),
            idempotency_key="later",
        )
    )

    assert snapshot == (first,)


def test_candidate_queries_snapshot_ids_and_reject_scope_subclasses() -> None:
    evidence_store, evidence = seeded_evidence_store()
    history = InMemoryMemoryHistoryStore(evidence_store)
    candidate = history.append_candidate(
        make_candidate_proposal(evidence_ids=(evidence.evidence_id,))
    )
    query_id = MutableHashStr(candidate.candidate_id)
    query_id.hash_salt = 1_000_003

    assert history.get_candidate(make_scope(), query_id) is candidate
    assert history.get_candidate_evidence(make_scope(), query_id) == (evidence,)
    assert query_id.hash_calls == 0
    invalid_scope = MemoryScopeSubclass("tenant-1", "assistant-memory", "user-1")
    with pytest.raises(TypeError, match="scope"):
        history.get_candidate(invalid_scope, candidate.candidate_id)
    with pytest.raises(TypeError, match="scope"):
        history.list_candidates(invalid_scope)


def test_foreign_candidate_error_matches_genuinely_missing_error() -> None:
    evidence_store, evidence = seeded_evidence_store()
    history = InMemoryMemoryHistoryStore(evidence_store)
    candidate = history.append_candidate(
        make_candidate_proposal(evidence_ids=(evidence.evidence_id,))
    )
    empty_history = InMemoryMemoryHistoryStore(evidence_store)

    with pytest.raises(CandidateNotFoundError) as foreign:
        history.get_candidate(make_scope(subject_id="user-2"), candidate.candidate_id)
    with pytest.raises(CandidateNotFoundError) as missing:
        empty_history.get_candidate(make_scope(), candidate.candidate_id)

    assert str(foreign.value) == str(missing.value)


def append_candidate(
    history: InMemoryMemoryHistoryStore,
    evidence_store: InMemoryEvidenceStore,
    *,
    candidate_key: str,
    evidence_key: str,
) -> MemoryCandidate:
    evidence = evidence_store.append(
        make_evidence_event(
            sequence_no=len(evidence_store.list(make_scope())),
            payload=candidate_key,
            idempotency_key=evidence_key,
        )
    )
    return history.append_candidate(
        make_candidate_proposal(
            content=candidate_key,
            evidence_ids=(evidence.evidence_id,),
            idempotency_key=candidate_key,
        )
    )


def seeded_history_candidate() -> tuple[
    InMemoryMemoryHistoryStore, MemoryCandidate, EvidenceRecord
]:
    evidence_store, evidence = seeded_evidence_store()
    history = InMemoryMemoryHistoryStore(evidence_store)
    candidate = history.append_candidate(
        make_candidate_proposal(evidence_ids=(evidence.evidence_id,))
    )
    return history, candidate, evidence


def seeded_parent_and_candidate() -> tuple[
    InMemoryMemoryHistoryStore, MemoryRevision, MemoryCandidate
]:
    evidence_store, evidence = seeded_evidence_store()
    history = InMemoryMemoryHistoryStore(evidence_store)
    root_candidate = history.append_candidate(
        make_candidate_proposal(evidence_ids=(evidence.evidence_id,))
    )
    parent = history.append_revision(
        make_revision_proposal(candidate_id=root_candidate.candidate_id)
    )
    child_candidate = append_candidate(
        history,
        evidence_store,
        candidate_key="child-candidate",
        evidence_key="child-evidence",
    )
    return history, parent, child_candidate


def seeded_parent_and_two_candidates() -> tuple[
    InMemoryMemoryHistoryStore,
    MemoryRevision,
    MemoryCandidate,
    MemoryCandidate,
]:
    evidence_store, evidence = seeded_evidence_store()
    history = InMemoryMemoryHistoryStore(evidence_store)
    root_candidate = history.append_candidate(
        make_candidate_proposal(evidence_ids=(evidence.evidence_id,))
    )
    parent = history.append_revision(
        make_revision_proposal(candidate_id=root_candidate.candidate_id)
    )
    left = append_candidate(
        history,
        evidence_store,
        candidate_key="left-candidate",
        evidence_key="left-evidence",
    )
    right = append_candidate(
        history,
        evidence_store,
        candidate_key="right-candidate",
        evidence_key="right-evidence",
    )
    return history, parent, left, right


def test_add_creates_generation_zero_logical_memory() -> None:
    history, candidate, _ = seeded_history_candidate()
    revision = history.append_revision(
        make_revision_proposal(candidate_id=candidate.candidate_id)
    )

    assert revision.proposal.operation is RevisionOperation.ADD
    assert revision.proposal.parent_revision_id is None
    assert revision.generation == 0
    assert revision.memory_id == "mem_" + revision.content_hash[:24]


def test_append_revision_rejects_proposal_subclass_before_any_write() -> None:
    proposal = RevisionProposalSubclass(
        make_scope(), "cand_missing", RevisionOperation.ADD, None, "revision-1"
    )
    history = InMemoryMemoryHistoryStore(InMemoryEvidenceStore())

    with pytest.raises(TypeError, match="RevisionProposal"):
        history.append_revision(proposal)

    assert history._revision_by_id == {}
    assert history._revision_by_idempotency == {}
    assert history._revision_by_candidate == {}
    assert history._revisions_by_scope == {}


def test_revision_identity_is_derived_before_history_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history, candidate, _ = seeded_history_candidate()
    proposal = make_revision_proposal(candidate_id=candidate.candidate_id)
    expected_hash = sha256(proposal.canonical_bytes()).hexdigest()
    events: list[str] = []

    class ProbeLock:
        def __init__(self) -> None:
            self._lock = RLock()
            self.held = False

        def __enter__(self) -> ProbeLock:
            assert events == ["canonical", "sha256", "hexdigest", "revision-id"]
            self._lock.acquire()
            self.held = True
            return self

        def __exit__(
            self,
            exc_type: object,
            exc_value: object,
            traceback: object,
        ) -> None:
            self.held = False
            self._lock.release()

    probe_lock = ProbeLock()
    history._lock = probe_lock  # type: ignore[assignment]
    original_canonical_bytes = RevisionProposal.canonical_bytes
    original_sha256 = history_store_module.hashlib.sha256

    def observed_canonical_bytes(value: RevisionProposal) -> bytes:
        assert not probe_lock.held
        events.append("canonical")
        return original_canonical_bytes(value)

    class ObservedHexDigest(str):
        slice_count: int

        def __new__(cls, value: str) -> ObservedHexDigest:
            instance = str.__new__(cls, value)
            instance.slice_count = 0
            return instance

        def __getitem__(self, key: object) -> str:
            self.slice_count += 1
            if self.slice_count == 1:
                assert not probe_lock.held
                events.append("revision-id")
            return str.__getitem__(self, key)  # type: ignore[index]

    class ObservedHash:
        def __init__(self, canonical_bytes: bytes) -> None:
            self._hash = original_sha256(canonical_bytes)

        def hexdigest(self) -> ObservedHexDigest:
            assert not probe_lock.held
            events.append("hexdigest")
            return ObservedHexDigest(self._hash.hexdigest())

    def observed_sha256(canonical_bytes: bytes) -> ObservedHash:
        assert not probe_lock.held
        events.append("sha256")
        return ObservedHash(canonical_bytes)

    monkeypatch.setattr(RevisionProposal, "canonical_bytes", observed_canonical_bytes)
    monkeypatch.setattr(history_store_module.hashlib, "sha256", observed_sha256)

    revision = history.append_revision(proposal)

    assert revision.content_hash == expected_hash
    assert revision.revision_id == f"rev_{expected_hash[:24]}"
    assert events == ["canonical", "sha256", "hexdigest", "revision-id"]


@pytest.mark.parametrize(
    "operation",
    [
        RevisionOperation.REFINE,
        RevisionOperation.SUPERSEDE,
        RevisionOperation.CONTRADICT,
    ],
)
def test_child_transition_inherits_memory_and_preserves_parent(
    operation: RevisionOperation,
) -> None:
    history, parent, child_candidate = seeded_parent_and_candidate()
    parent_bytes = parent.proposal.canonical_bytes()

    child = history.append_revision(
        make_revision_proposal(
            candidate_id=child_candidate.candidate_id,
            operation=operation,
            parent_revision_id=parent.revision_id,
            idempotency_key=f"{operation.value}-1",
        )
    )

    assert child.memory_id == parent.memory_id
    assert child.generation == parent.generation + 1
    assert parent.proposal.canonical_bytes() == parent_bytes
    assert history.get_revision(parent.proposal.scope, parent.revision_id) is parent


def test_different_candidates_from_one_parent_survive_as_siblings() -> None:
    history, parent, left_candidate, right_candidate = (
        seeded_parent_and_two_candidates()
    )
    left = history.append_revision(
        make_revision_proposal(
            candidate_id=left_candidate.candidate_id,
            operation=RevisionOperation.REFINE,
            parent_revision_id=parent.revision_id,
            idempotency_key="left",
        )
    )
    right = history.append_revision(
        make_revision_proposal(
            candidate_id=right_candidate.candidate_id,
            operation=RevisionOperation.CONTRADICT,
            parent_revision_id=parent.revision_id,
            idempotency_key="right",
        )
    )

    assert left.memory_id == right.memory_id == parent.memory_id
    assert left.generation == right.generation == 1
    assert set(history.list_revisions(parent.proposal.scope)) == {parent, left, right}


def test_missing_parent_does_not_consume_candidate() -> None:
    history, candidate, _ = seeded_history_candidate()
    invalid = make_revision_proposal(
        candidate_id=candidate.candidate_id,
        operation=RevisionOperation.REFINE,
        parent_revision_id="rev_missing",
    )
    with pytest.raises(RevisionNotFoundError):
        history.append_revision(invalid)

    valid = history.append_revision(
        make_revision_proposal(candidate_id=candidate.candidate_id)
    )
    assert valid.generation == 0


def test_foreign_candidate_and_parent_are_hidden_as_missing() -> None:
    first_scope = make_scope(subject_id="user-1")
    second_scope = make_scope(subject_id="user-2")
    evidence_store = InMemoryEvidenceStore()
    first_evidence = evidence_store.append(make_evidence_event(scope=first_scope))
    second_evidence = evidence_store.append(
        make_evidence_event(
            scope=second_scope,
            idempotency_key="second-evidence",
        )
    )
    history = InMemoryMemoryHistoryStore(evidence_store)
    first_candidate = history.append_candidate(
        make_candidate_proposal(
            scope=first_scope,
            evidence_ids=(first_evidence.evidence_id,),
            idempotency_key="first-candidate",
        )
    )
    second_candidate = history.append_candidate(
        make_candidate_proposal(
            scope=second_scope,
            evidence_ids=(second_evidence.evidence_id,),
            idempotency_key="second-candidate",
        )
    )
    foreign_parent = history.append_revision(
        make_revision_proposal(
            scope=second_scope,
            candidate_id=second_candidate.candidate_id,
            idempotency_key="foreign-parent",
        )
    )

    with pytest.raises(CandidateNotFoundError):
        history.append_revision(
            make_revision_proposal(candidate_id=second_candidate.candidate_id)
        )
    with pytest.raises(RevisionNotFoundError):
        history.append_revision(
            make_revision_proposal(
                candidate_id=first_candidate.candidate_id,
                operation=RevisionOperation.REFINE,
                parent_revision_id=foreign_parent.revision_id,
            )
        )


def test_revision_idempotency_and_lists_are_scope_isolated() -> None:
    first_scope = make_scope(subject_id="user-1")
    second_scope = make_scope(subject_id="user-2")
    evidence_store = InMemoryEvidenceStore()
    history = InMemoryMemoryHistoryStore(evidence_store)

    def append_root(scope: MemoryScope) -> MemoryRevision:
        evidence = evidence_store.append(
            make_evidence_event(
                scope=scope,
                idempotency_key="shared-evidence",
            )
        )
        candidate = history.append_candidate(
            make_candidate_proposal(
                scope=scope,
                evidence_ids=(evidence.evidence_id,),
                idempotency_key="shared-candidate",
            )
        )
        return history.append_revision(
            make_revision_proposal(
                scope=scope,
                candidate_id=candidate.candidate_id,
                idempotency_key="shared-revision",
            )
        )

    first = append_root(first_scope)
    second = append_root(second_scope)

    assert first is not second
    assert history.list_revisions(first_scope) == (first,)
    assert history.list_revisions(second_scope) == (second,)


def test_revision_retry_and_conflicting_idempotency_are_atomic() -> None:
    evidence_store, evidence = seeded_evidence_store()
    history = InMemoryMemoryHistoryStore(evidence_store)
    parent_candidate = history.append_candidate(
        make_candidate_proposal(evidence_ids=(evidence.evidence_id,))
    )
    proposal = make_revision_proposal(candidate_id=parent_candidate.candidate_id)
    first = history.append_revision(proposal)
    retry = history.append_revision(
        RevisionProposal(
            scope=proposal.scope,
            candidate_id=proposal.candidate_id,
            operation=proposal.operation,
            parent_revision_id=proposal.parent_revision_id,
            idempotency_key=proposal.idempotency_key,
        )
    )
    assert retry is first

    other_candidate = append_candidate(
        history,
        evidence_store,
        candidate_key="other-candidate",
        evidence_key="other-evidence",
    )
    before = (
        dict(history._revision_by_id),
        dict(history._revision_by_idempotency),
        dict(history._revision_by_candidate),
        {scope: list(items) for scope, items in history._revisions_by_scope.items()},
    )
    with pytest.raises(RevisionConflictError, match="idempotency"):
        history.append_revision(
            make_revision_proposal(candidate_id=other_candidate.candidate_id)
        )
    assert (
        history._revision_by_id,
        history._revision_by_idempotency,
        history._revision_by_candidate,
        history._revisions_by_scope,
    ) == before

    recovered = history.append_revision(
        make_revision_proposal(
            candidate_id=other_candidate.candidate_id,
            idempotency_key="other-revision",
        )
    )
    assert recovered.proposal.candidate_id == other_candidate.candidate_id


def test_revision_error_precedence_and_id_collision_are_atomic() -> None:
    history, candidate, _ = seeded_history_candidate()
    first = history.append_revision(
        make_revision_proposal(candidate_id=candidate.candidate_id)
    )

    def revision_indexes() -> tuple[object, ...]:
        return (
            dict(history._revision_by_id),
            dict(history._revision_by_idempotency),
            dict(history._revision_by_candidate),
            {
                scope: list(revisions)
                for scope, revisions in history._revisions_by_scope.items()
            },
        )

    idempotency_attempt = make_revision_proposal(
        candidate_id="cand_missing",
        operation=RevisionOperation.REFINE,
        parent_revision_id="rev_missing",
    )
    idempotency_hash = sha256(idempotency_attempt.canonical_bytes()).hexdigest()
    idempotency_index = (
        idempotency_attempt.scope,
        f"rev_{idempotency_hash[:24]}",
    )
    history._revision_by_id[idempotency_index] = first
    before = revision_indexes()
    with pytest.raises(RevisionConflictError, match="idempotency"):
        history.append_revision(idempotency_attempt)
    assert revision_indexes() == before
    del history._revision_by_id[idempotency_index]

    collision_attempt = make_revision_proposal(
        candidate_id="cand_missing",
        operation=RevisionOperation.REFINE,
        parent_revision_id="rev_missing",
        idempotency_key="collision-attempt",
    )
    collision_hash = sha256(collision_attempt.canonical_bytes()).hexdigest()
    collision_index = (
        collision_attempt.scope,
        f"rev_{collision_hash[:24]}",
    )
    history._revision_by_id[collision_index] = first
    before = revision_indexes()
    with pytest.raises(RevisionConflictError, match="collision"):
        history.append_revision(collision_attempt)
    assert revision_indexes() == before
    del history._revision_by_id[collision_index]

    with pytest.raises(CandidateNotFoundError):
        history.append_revision(
            make_revision_proposal(
                candidate_id="cand_missing",
                operation=RevisionOperation.REFINE,
                parent_revision_id="rev_missing",
                idempotency_key="missing-relationships",
            )
        )
    with pytest.raises(RevisionConflictError, match="candidate"):
        history.append_revision(
            make_revision_proposal(
                candidate_id=candidate.candidate_id,
                operation=RevisionOperation.REFINE,
                parent_revision_id="rev_missing",
                idempotency_key="used-candidate",
            )
        )


def test_one_candidate_can_back_only_one_revision() -> None:
    history, candidate, _ = seeded_history_candidate()
    history.append_revision(make_revision_proposal(candidate_id=candidate.candidate_id))

    with pytest.raises(RevisionConflictError, match="candidate"):
        history.append_revision(
            make_revision_proposal(
                candidate_id=candidate.candidate_id,
                idempotency_key="second-use",
            )
        )


def test_revision_relationship_checks_and_writes_share_one_lock_epoch() -> None:
    history, candidate, _ = seeded_history_candidate()
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
            return self

        def __exit__(
            self,
            exc_type: object,
            exc_value: object,
            traceback: object,
        ) -> None:
            self.held = False
            self._lock.release()

        def current_epoch(self) -> int:
            assert self.held, "revision index accessed outside history lock"
            return self.epoch

    epoch_lock = EpochLock()

    def record(event: str) -> None:
        records.append((event, epoch_lock.current_epoch()))

    class RecordingList(list[object]):
        def append(self, item: object) -> None:
            record("scope:append")
            super().append(item)

    class RecordingDict(dict[object, object]):
        def __init__(self, values: dict[object, object], name: str) -> None:
            super().__init__(values)
            self._name = name

        def get(self, key: object, default: object = None) -> object:
            record(f"{self._name}:get")
            return super().get(key, default)

        def __contains__(self, key: object) -> bool:
            record(f"{self._name}:contains")
            return super().__contains__(key)

        def __setitem__(self, key: object, value: object) -> None:
            record(f"{self._name}:set")
            super().__setitem__(key, value)

        def setdefault(self, key: object, default: object = None) -> object:
            record(f"{self._name}:setdefault")
            if not dict.__contains__(self, key):
                default = RecordingList(default or ())
            return super().setdefault(key, default)

    history._lock = epoch_lock  # type: ignore[assignment]
    history._candidate_by_id = RecordingDict(  # type: ignore[assignment]
        history._candidate_by_id,
        "candidate",
    )
    history._revision_by_id = RecordingDict(  # type: ignore[assignment]
        history._revision_by_id,
        "revision",
    )
    history._revision_by_idempotency = RecordingDict(  # type: ignore[assignment]
        history._revision_by_idempotency,
        "idempotency",
    )
    history._revision_by_candidate = RecordingDict(  # type: ignore[assignment]
        history._revision_by_candidate,
        "candidate-revision",
    )
    history._revisions_by_scope = RecordingDict(  # type: ignore[assignment]
        history._revisions_by_scope,
        "scope",
    )
    revision = history.append_revision(
        make_revision_proposal(candidate_id=candidate.candidate_id)
    )

    assert epoch_lock.enter_count == 1
    assert {epoch for _, epoch in records} == {1}
    assert [event for event, _ in records] == [
        "idempotency:get",
        "revision:get",
        "candidate:get",
        "candidate-revision:contains",
        "revision:set",
        "idempotency:set",
        "candidate-revision:set",
        "scope:setdefault",
        "scope:append",
    ]
    assert tuple(history._revision_by_id.values()) == (revision,)
    assert tuple(history._revision_by_idempotency.values()) == (revision,)
    assert tuple(history._revision_by_candidate.values()) == (revision,)
    assert tuple(history._revisions_by_scope.values()) == ([revision],)


def test_revision_identity_order_filter_and_public_provenance() -> None:
    history, parent, child_candidate = seeded_parent_and_candidate()
    child = history.append_revision(
        make_revision_proposal(
            candidate_id=child_candidate.candidate_id,
            operation=RevisionOperation.REFINE,
            parent_revision_id=parent.revision_id,
            idempotency_key="child-revision",
        )
    )
    expected_hash = sha256(child.proposal.canonical_bytes()).hexdigest()

    assert child.content_hash == expected_hash
    assert re.fullmatch(r"[0-9a-f]{64}", child.content_hash)
    assert child.revision_id == f"rev_{expected_hash[:24]}"
    assert re.fullmatch(r"rev_[0-9a-f]{24}", child.revision_id)
    assert history.list_revisions(parent.proposal.scope) == (parent, child)
    assert history.list_revisions(
        parent.proposal.scope, memory_id=parent.memory_id
    ) == (parent, child)
    assert history.list_revisions(parent.proposal.scope, memory_id="mem_missing") == ()
    resolved = history.get_revision(child.proposal.scope, child.revision_id)
    candidate = history.get_candidate(
        resolved.proposal.scope, resolved.proposal.candidate_id
    )
    assert history.get_candidate_evidence(
        candidate.proposal.scope, candidate.candidate_id
    )
    with pytest.raises(RevisionNotFoundError):
        history.get_revision(make_scope(subject_id="user-2"), child.revision_id)


def test_candidate_and_revision_idempotency_domains_are_independent() -> None:
    evidence_store, evidence = seeded_evidence_store()
    history = InMemoryMemoryHistoryStore(evidence_store)
    candidate = history.append_candidate(
        make_candidate_proposal(
            evidence_ids=(evidence.evidence_id,),
            idempotency_key="shared-key",
        )
    )
    revision = history.append_revision(
        make_revision_proposal(
            candidate_id=candidate.candidate_id,
            idempotency_key="shared-key",
        )
    )
    assert revision.proposal.candidate_id == candidate.candidate_id


def test_revision_queries_snapshot_strings_and_hide_scope_occupancy() -> None:
    history, candidate, _ = seeded_history_candidate()
    revision = history.append_revision(
        make_revision_proposal(candidate_id=candidate.candidate_id)
    )
    query_id = MutableHashStr(revision.revision_id)
    query_id.hash_salt = 1_000_003
    false_filter = AlwaysEqualStr("mem_missing")

    assert history.get_revision(make_scope(), query_id) is revision
    assert query_id.hash_calls == 0
    assert history.list_revisions(make_scope(), memory_id=false_filter) == ()
    assert false_filter.equality_calls == 0
    invalid_scope = MemoryScopeSubclass("tenant-1", "assistant-memory", "user-1")
    with pytest.raises(TypeError, match="scope"):
        history.get_revision(invalid_scope, revision.revision_id)
    with pytest.raises(TypeError, match="scope"):
        history.list_revisions(invalid_scope)

    empty_history = InMemoryMemoryHistoryStore(InMemoryEvidenceStore())
    with pytest.raises(RevisionNotFoundError) as foreign:
        history.get_revision(make_scope(subject_id="user-2"), revision.revision_id)
    with pytest.raises(RevisionNotFoundError) as missing:
        empty_history.get_revision(make_scope(), revision.revision_id)
    assert str(foreign.value) == str(missing.value)


def test_revision_order_and_returned_snapshot_cover_roots_and_siblings() -> None:
    evidence_store, evidence = seeded_evidence_store()
    history = InMemoryMemoryHistoryStore(evidence_store)
    first_root_candidate = history.append_candidate(
        make_candidate_proposal(evidence_ids=(evidence.evidence_id,))
    )
    first_root = history.append_revision(
        make_revision_proposal(candidate_id=first_root_candidate.candidate_id)
    )
    left_candidate = append_candidate(
        history,
        evidence_store,
        candidate_key="left",
        evidence_key="left-evidence",
    )
    right_candidate = append_candidate(
        history,
        evidence_store,
        candidate_key="right",
        evidence_key="right-evidence",
    )
    right = history.append_revision(
        make_revision_proposal(
            candidate_id=right_candidate.candidate_id,
            operation=RevisionOperation.CONTRADICT,
            parent_revision_id=first_root.revision_id,
            idempotency_key="right-revision",
        )
    )
    left = history.append_revision(
        make_revision_proposal(
            candidate_id=left_candidate.candidate_id,
            operation=RevisionOperation.REFINE,
            parent_revision_id=first_root.revision_id,
            idempotency_key="left-revision",
        )
    )
    second_root_candidate = append_candidate(
        history,
        evidence_store,
        candidate_key="second-root",
        evidence_key="second-root-evidence",
    )
    second_root = history.append_revision(
        make_revision_proposal(
            candidate_id=second_root_candidate.candidate_id,
            idempotency_key="second-root-revision",
        )
    )
    expected = tuple(
        sorted(
            (first_root, left, right, second_root),
            key=lambda item: (item.memory_id, item.generation, item.revision_id),
        )
    )
    snapshot = history.list_revisions(make_scope())
    assert snapshot == expected

    later_candidate = append_candidate(
        history,
        evidence_store,
        candidate_key="later",
        evidence_key="later-evidence",
    )
    history.append_revision(
        make_revision_proposal(
            candidate_id=later_candidate.candidate_id,
            operation=RevisionOperation.REFINE,
            parent_revision_id=first_root.revision_id,
            idempotency_key="later-revision",
        )
    )
    assert snapshot == expected


def test_generation_overflow_leaves_all_revision_indexes_unchanged() -> None:
    evidence_store, evidence = seeded_evidence_store()
    history = InMemoryMemoryHistoryStore(evidence_store)
    parent_candidate = history.append_candidate(
        make_candidate_proposal(evidence_ids=(evidence.evidence_id,))
    )
    parent_proposal = make_revision_proposal(
        candidate_id=parent_candidate.candidate_id,
        idempotency_key="max-parent",
    )
    parent = MemoryRevision(
        revision_id="rev_max",
        memory_id="mem_max",
        generation=2**63 - 1,
        proposal=parent_proposal,
        content_hash="a" * 64,
        created_at=datetime.now(UTC),
    )
    history._revision_by_id[(make_scope(), parent.revision_id)] = parent
    history._revision_by_idempotency[(make_scope(), "max-parent")] = parent
    history._revision_by_candidate[
        (
            make_scope(),
            parent_candidate.candidate_id,
        )
    ] = parent
    history._revisions_by_scope[make_scope()] = [parent]
    child_candidate = append_candidate(
        history,
        evidence_store,
        candidate_key="overflow-child",
        evidence_key="overflow-evidence",
    )
    before = (
        dict(history._revision_by_id),
        dict(history._revision_by_idempotency),
        dict(history._revision_by_candidate),
        {scope: list(items) for scope, items in history._revisions_by_scope.items()},
    )

    with pytest.raises(RevisionConflictError, match="generation"):
        history.append_revision(
            make_revision_proposal(
                candidate_id=child_candidate.candidate_id,
                operation=RevisionOperation.REFINE,
                parent_revision_id=parent.revision_id,
                idempotency_key="overflow-child-revision",
            )
        )

    after = (
        history._revision_by_id,
        history._revision_by_idempotency,
        history._revision_by_candidate,
        history._revisions_by_scope,
    )
    assert after == before
