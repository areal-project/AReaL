# SPDX-License-Identifier: Apache-2.0

"""Tests for the evidence-grounded in-memory history store."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from hashlib import sha256

import pytest

from areal.v2.memory_service.errors import (
    CandidateConflictError,
    CandidateNotFoundError,
    EvidenceNotFoundError,
    MemoryServiceError,
)
from areal.v2.memory_service.history_store import (
    InMemoryMemoryHistoryStore,
    MemoryHistoryStore,
)
from areal.v2.memory_service.history_types import CandidateProposal
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


class MemoryScopeSubclass(MemoryScope):
    pass


class CandidateProposalSubclass(CandidateProposal):
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


def test_history_store_contract_exposes_only_candidate_surface() -> None:
    public_members = {
        name for name in MemoryHistoryStore.__dict__ if not name.startswith("_")
    }

    assert public_members == {
        "append_candidate",
        "get_candidate",
        "get_candidate_evidence",
        "list_candidates",
    }


def test_candidate_error_types_share_memory_service_base_error() -> None:
    assert issubclass(CandidateNotFoundError, MemoryServiceError)
    assert issubclass(CandidateConflictError, MemoryServiceError)


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
